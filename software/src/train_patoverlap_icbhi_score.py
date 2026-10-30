import os
import torch
import torch.nn as nn

#part of fix 1: patient overlap and data leakage
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

from src.dataset_circpad_logdb_specaug_demoscale import ICBHIDataset
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * spectrogram.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_icbhi(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

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

            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    val_loss = running_loss / total
    val_acc = correct / total

    # Sensitivity (Class 1,2,3 recall) & Specificity (Class 0 recall)
    sensitivity = (class_correct[1] + class_correct[2] + class_correct[3]) / max(1, (class_total[1] + class_total[2] + class_total[3]))
    specificity = class_correct[0] / max(1, class_total[0])
    icbhi_score = (sensitivity + specificity) / 2.0

    return val_loss, val_acc, sensitivity, specificity, icbhi_score

def train_model():
    # Hyperparameters
    DATA_DIR = "data/icbhi_dataset"  # Path to ICBHI dataset directory
    DEMO_FILE = "data/icbhi_dataset/demographic_info.txt"
    SAVE_DIR = "checkpoints"
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-4  # Reduced learning rate to prevent wild gradient spikes, lowered even further
    VAL_SPLIT = 0.2
    NUM_CLASSES = 4

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ICBHI Dataset...")
    base_dataset = ICBHIDataset(data_dir=DATA_DIR, demo_file=DEMO_FILE)

    # Patient-Aware Split using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(gss.split(base_dataset, groups=base_dataset.patient_ids))

    # Create two completely independent dataset instances
    train_full = ICBHIDataset(data_dir=DATA_DIR, demo_file=DEMO_FILE)
    train_full.is_train = True
    train_dataset = Subset(train_full, train_idx)

    val_full = ICBHIDataset(data_dir=DATA_DIR, demo_file=DEMO_FILE)
    val_full.is_train = False
    val_dataset = Subset(val_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    train_size = len(train_idx)
    val_size = len(val_idx)
    print(f"Train samples: {train_size} | Validation samples: {val_size}")

    #Important - class weighting calculation, new
    #extract laels across dataset to compute inverse frquency
    all_labels = [sample["label"] for sample in base_dataset.samples]
    class_counts = np.bincount(all_labels, minlength = NUM_CLASSES)
    total_samples = len(all_labels)

    # fix, calculate smoothed inverse weights
    raw_weights = total_samples / (NUM_CLASSES * class_counts.astype(np.float32))
    class_weights = np.power(raw_weights, 0.50) #changed back to square_root smoothing
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"\nClass Distribution: {class_counts.tolist()}")
    print(f"Calculated Class Weights: {class_weights_tensor.tolist()}\n")


    # Model initialization, loss function, and optimizer
    model = ICBHIModel(num_classes=NUM_CLASSES).to(device)

    # Standard CrossEntropyLoss for 4-class classification
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) #slightly reduced weight decay for smoother gradient trajectories
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_icbhi_score = 0.0

    print("\n--- Starting Training ---")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, se, sp, icbhi_score = validate_icbhi(model, val_loader, criterion, device)

        # Step scheduler on ICBHI Score (maximizing S)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Se: {se*100:.1f}% | Sp: {sp*100:.1f}% | Score: {icbhi_score*100:.2f}%"
        )

        # Save checkpoint on highest ICBHI Score
        if icbhi_score > best_icbhi_score:
            best_icbhi_score = icbhi_score
            save_path = os.path.join(SAVE_DIR, "best_icbhi_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'icbhi_score': icbhi_score,
                'sensitivity': se,
                'specificity': sp
            }, save_path)
            print(f"  --> Saved new best checkpoint (Score: {icbhi_score*100:.2f}%) to {save_path}")

if __name__ == "__main__":
    train_model()