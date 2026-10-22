import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHIDataset
from src.model_qat import ICBHIModelQAT
from brevitas.export.onnx.qonnx.manager import QONNXManager

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        spectrogram = batch["spectrogram"].to(device)
        demographics = batch["demographics"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(spectrogram, demographics)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * spectrogram.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            spectrogram = batch["spectrogram"].to(device)
            demographics = batch["demographics"].to(device)
            labels = batch["label"].to(device)

            outputs = model(spectrogram, demographics)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * spectrogram.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def run_qat_finetuning():
    DATA_DIR = "data/icbhi_dataset"
    DEMO_FILE = "data/icbhi_dataset/demographic_info.txt"
    SAVE_DIR = "checkpoints"
    FP32_CHECKPOINT = os.path.join(SAVE_DIR, "best_icbhi_model.pth")
    QAT_CHECKPOINT = os.path.join(SAVE_DIR, "best_icbhi_model_qat.pth")
    ONNX_EXPORT_PATH = os.path.join(SAVE_DIR, "finn_audio_backbone.onnx")
    
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT, NUM_CLASSES = 32, 15, 1e-4, 0.2, 4

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ICBHI Dataset...")
    full_dataset = ICBHIDataset(data_dir=DATA_DIR, demo_file=DEMO_FILE)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_labels = [sample["label"] for sample in full_dataset.samples]
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    total_samples = len(all_labels)
    class_weights = total_samples / (NUM_CLASSES * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nInitializing QAT Model (W4A4)...")
    model = ICBHIModelQAT(num_classes=NUM_CLASSES, weight_bit_width=4, act_bit_width=4)
    model = model.to(device)

    # Ensure Brevitas internal quantization buffers are explicitly moved to GPU
    for module in model.modules():
        if hasattr(module, 'to'):
            module.to(device)

    if os.path.exists(FP32_CHECKPOINT):
        checkpoint = torch.load(FP32_CHECKPOINT, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("--> Loaded FP32 floating-point checkpoint into QAT model!")
    else:
        print("--> Warning: FP32 checkpoint not found! Training QAT model from scratch.")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_acc = 0.0

    print("\n--- Starting QAT Fine-Tuning ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, QAT_CHECKPOINT)
            print(f"  --> Saved new best QAT checkpoint ({val_acc*100:.2f}% accuracy)")

    print("\n--- Exporting Fine-Tuned FINN ONNX Graph ---")
    model.load_state_dict(torch.load(QAT_CHECKPOINT)['model_state_dict'])
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 1, 129, 126)
    
    QONNXManager.export(model.backbone, args=dummy_input, export_path=ONNX_EXPORT_PATH)
    print(f"Successfully exported fine-tuned ONNX model to: {ONNX_EXPORT_PATH}")

if __name__ == "__main__":
    run_qat_finetuning()