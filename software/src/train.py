import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import ICBHIDataset
from src.model import ICBHIModel
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        spectrogram = batch["spectrogram"].to(device)  # shape [B, 1, F, T]
        demographics = batch["demographics"].to(device) # shape [B, 5]
        labels = batch["label"].to(device)             # shape [B]

        optimizer.zero_grad()

        # Forward pass (1-channel dataset training path)
        outputs = model(spectrogram, demographics)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * spectrogram.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            spectrogram = batch["spectrogram"].to(device)  # shape [B, 1, F, T]
            demographics = batch["demographics"].to(device) # shape [B, 5]
            labels = batch["label"].to(device)             # shape [B] 

            outputs = model(spectrogram, demographics)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * spectrogram.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_model():
    # Hyperparameters
    DATA_DIR = "data/icbhi_dataset"  # Path to ICBHI dataset directory
    DEMO_FILE = "data/icbhi_dataset/demographic_info.txt"
    SAVE_DIR = "checkpoints"
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.2
    NUM_CLASSES = 4

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset loading and train/val split
    print("Loading ICBHI Dataset...")
    full_dataset = ICBHIDataset(data_dir=DATA_DIR, demo_file=DEMO_FILE)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Ensures reproducible splits
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train samples: {train_size} | Validation samples: {val_size}")

    #Important - class weighting calculation, new
    #extract laels across dataset to compute inverse frquency
    all_labels = [sample["label"] for sample in full_dataset.samples]
    class_counts = np.bincount(all_labels, minlength = NUM_CLASSES)
    total_samples = len(all_labels)

    #calculate inverse weights
    class_weights = total_samples / (NUM_CLASSES * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"\nClass Distribution: {class_counts.tolist()}")
    print(f"Calculated Class Weights: {class_weights_tensor.tolist()}\n")


    # Model initialization, loss function, and optimizer
    model = ICBHIModel(num_classes=NUM_CLASSES).to(device)

    # Standard CrossEntropyLoss for 4-class classification
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0.0

    print("\n--- Starting Training ---")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Adjust learning rate based on val loss
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "best_icbhi_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path}")

if __name__ == "__main__":
    train_model()