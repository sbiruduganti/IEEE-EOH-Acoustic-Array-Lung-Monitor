import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from brevitas.export import export_qonnx

warnings.filterwarnings("ignore", message=".*fast_hadamard_transform.*")

from src.dataset import ICBHIDataset
from src.model_brevitas import QuantResNet18LungModel, QuantDualStreamLungModel


def evaluate_8bit_sim(model, dataloader, criterion, device, is_dual_stream=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for wave_env, spec, demo, labels, _ in dataloader:
            wave_env, spec, demo, labels = (
                wave_env.to(device),
                spec.to(device),
                demo.to(device),
                labels.to(device),
            )
            outputs = model(wave_env, spec, demo) if is_dual_stream else model(spec, demo)
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

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    specificity = cm[0, 0] / (cm[0, :].sum() + 1e-8)
    pathology_correct = cm[1, 1] + cm[2, 2] + cm[3, 3]
    pathology_total = cm[1:, :].sum()
    sensitivity = pathology_correct / (pathology_total + 1e-8)
    icbhi_score = 0.5 * (sensitivity + specificity)

    # Also compute binary Healthy vs. Any-Pathology ICBHI score (screening benchmark)
    bin_true = (all_labels > 0).astype(int)
    bin_prob = 1.0 - all_probs[:, 0]
    bin_pred = (bin_prob >= 0.5).astype(int)
    bin_se = ((bin_pred == 1) & (bin_true == 1)).sum() / ((bin_true == 1).sum() + 1e-8)
    bin_sp = ((bin_pred == 0) & (bin_true == 0)).sum() / ((bin_true == 0).sum() + 1e-8)
    screen_icbhi = 0.5 * (bin_se + bin_sp)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    acc = (all_preds == all_labels).mean()
    return running_loss / len(all_labels), acc, icbhi_score, screen_icbhi, sensitivity, specificity, auc


def train_and_verify(model_type="dual_stream", epochs=25, data_dir="data/icbhi"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ICBHIDataset(data_dir=data_dir, sample_rate=4000, max_len=4.0)

    patient_ids = dataset.patient_ids
    labels = dataset.labels.numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(np.zeros(len(labels)), labels, groups=patient_ids))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)

    # Square-root inverse class weights prevent extreme Se/Sp oscillation!
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=4) + 1e-6
    raw_weights = len(train_labels) / (4.0 * class_counts)
    smooth_weights = np.sqrt(raw_weights)
    class_weights = torch.tensor(smooth_weights / smooth_weights.mean(), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # SpecAugment during training to prevent overfitting to patient microphone resonances
    freq_mask = T.FrequencyMasking(freq_mask_param=8)
    time_mask = T.TimeMasking(time_mask_param=8)

    is_dual = model_type == "dual_stream"
    model = (
        QuantDualStreamLungModel(in_channels=3, num_classes=4, bit_width=8).to(device)
        if is_dual
        else QuantResNet18LungModel(in_channels=3, num_classes=4, bit_width=8).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    best_icbhi = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for wave_env, spec, demo, y, _ in train_loader:
            wave_env, spec, demo, y = wave_env.to(device), spec.to(device), demo.to(device), y.to(device)
            # Apply SpecAugment to 2D spectrogram during training
            spec = time_mask(freq_mask(spec))

            optimizer.zero_grad()
            out = model(wave_env, spec, demo) if is_dual else model(spec, demo)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss, acc, icbhi, screen_icbhi, se, sp, auc = evaluate_8bit_sim(
            model, val_loader, criterion, device, is_dual_stream=is_dual
        )
        dt = time.time() - t0
        print(
            f"[{model_type} INT8] Ep {epoch+1:02d}/{epochs:02d} ({dt:.1f}s) | "
            f"4-Class Acc: {acc:.3f} | 4-Class ICBHI: {icbhi:.3f} | "
            f"Screen ICBHI: {screen_icbhi:.3f} (Se: {se:.3f}, Sp: {sp:.3f}) | AUC: {auc:.3f}",
            flush=True,
        )

        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save(
                {"model_state_dict": model.state_dict(), "icbhi_score": icbhi, "auc": auc},
                f"checkpoints/best_{model_type}_8bit.pth",
            )

    model.cpu().eval()
    if not is_dual:
        dummy_spec = torch.randn(1, 3, 64, 64)
        export_qonnx(model.stem, input_t=dummy_spec, export_path="checkpoints/resnet18_stem_8bit_qonnx.onnx")
        print("[Export] Saved QONNX graph to checkpoints/resnet18_stem_8bit_qonnx.onnx", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["dual_stream", "resnet18", "both"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--data_dir", type=str, default="data/icbhi")
    args = parser.parse_args()

    if args.model in ["dual_stream", "both"]:
        print("\n=== Training & Verifying Fast 8-Bit 1D+2D Dual-Stream Model ===", flush=True)
        train_and_verify(model_type="dual_stream", epochs=args.epochs, data_dir=args.data_dir)

    if args.model in ["resnet18", "both"]:
        print("\n=== Training & Verifying Fast 8-Bit 2D ResNet Model ===", flush=True)
        train_and_verify(model_type="resnet18", epochs=args.epochs, data_dir=args.data_dir)