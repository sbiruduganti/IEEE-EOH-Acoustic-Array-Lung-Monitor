import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from brevitas.export import export_qonnx

from src.dataset import ICBHIDataset
from src.model_brevitas import QuantResNet18LungModel, QuantDualStreamLungModel

def evaluate_8bit_sim(model, dataloader, criterion, device, is_dual_stream=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for wave, spec, demo, labels, _ in dataloader:
            wave, spec, demo, labels = wave.to(device), spec.to(device), demo.to(device), labels.to(device)
            outputs = model(wave, spec, demo) if is_dual_stream else model(spec, demo)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * spec.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute Official ICBHI Sensitivity (Pathology Recall) & Specificity (Healthy Recall)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    specificity = cm[0, 0] / (cm[0, :].sum() + 1e-8) # True Healthy / Total Healthy
    pathology_correct = cm[1, 1] + cm[2, 2] + cm[3, 3]
    pathology_total = cm[1:, :].sum()
    sensitivity = pathology_correct / (pathology_total + 1e-8)
    icbhi_score = 0.5 * (sensitivity + specificity)

    # Multi-class One-vs-Rest ROC-AUC
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    acc = (all_preds == all_labels).mean()
    return running_loss / len(all_labels), acc, icbhi_score, sensitivity, specificity, auc

def train_and_verify(model_type="dual_stream", epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ICBHIDataset(data_dir="data/icbhi", sample_rate=4000, max_len=4.0)

    # 1. Patient-wise split (no patient appears in both train and val)
    patient_ids = [c[1] for c in dataset.cycles]
    labels = [c[4] for c in dataset.cycles]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(np.zeros(len(labels)), labels, groups=patient_ids))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

    # 2. Class-weighted CrossEntropyLoss to handle imbalanced crackle/wheeze counts
    train_labels = np.array([labels[i] for i in train_idx])
    class_counts = np.bincount(train_labels, minlength=4) + 1e-6
    class_weights = torch.tensor(len(train_labels) / (4.0 * class_counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Initialize chosen 8-bit Brevitas model
    is_dual = (model_type == "dual_stream")
    model = QuantDualStreamLungModel(num_classes=4, bit_width=8).to(device) if is_dual else QuantResNet18LungModel(num_classes=4, bit_width=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_icbhi = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for wave, spec, demo, y, _ in train_loader:
            wave, spec, demo, y = wave.to(device), spec.to(device), demo.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(wave, spec, demo) if is_dual else model(spec, demo)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_loss, acc, icbhi, se, sp, auc = evaluate_8bit_sim(model, val_loader, criterion, device, is_dual_stream=is_dual)
        print(f"[{model_type} INT8-Sim] Epoch {epoch+1:02d} | Acc: {acc:.3f} | ICBHI: {icbhi:.3f} (Se: {se:.3f}, Sp: {sp:.3f}) | AUC: {auc:.3f}")

        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save({"model_state_dict": model.state_dict(), "icbhi_score": icbhi}, f"checkpoints/best_{model_type}_8bit.pth")

    # 4. Export acoustic branch to QONNX for Xilinx FINN
    model.cpu().eval()
    if not is_dual:
        # Export the single-channel 2D spectrogram acoustic backbone for FINN hardware synthesis
        dummy_spec = torch.randn(1, 1, 64, 63)
        export_qonnx(model.stem, input_t=dummy_spec, export_path="checkpoints/resnet18_stem_8bit_qonnx.onnx")