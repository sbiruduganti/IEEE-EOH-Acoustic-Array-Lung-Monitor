"""
Drop-in replacement for `software/src/train_qat.py` ( plus `export_to_finn()` for `software/export_finn.py`)
Repository: https://github.com/sbiruduganti/IEEE-EOH-Acoustic-Array-Lung-Monitor

Usage (Local or Google Colab GPU):
    python software/src/train_qat.py --data_dir data/ICBHI_final_database --epochs 30 --batch_size 64
"""

import os
import sys
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Allow running either as `python software/src/train_qat.py` or from inside `software/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.dataset import ICBHIDataset
    from src.model_qat import ICBHIModelQAT, BinaryDiagnosisCascadeQAT
except ImportError:
    from dataset import ICBHIDataset
    from model_qat import ICBHIModelQAT, BinaryDiagnosisCascadeQAT

try:
    from brevitas.export.onnx.qonnx.manager import QONNXManager
    BREVITAS_EXPORT_AVAILABLE = True
except ImportError:
    BREVITAS_EXPORT_AVAILABLE = False


def apply_spec_augment(spec: torch.Tensor, max_f_mask: int = 12, max_t_mask: int = 16) -> torch.Tensor:
    """Fast GPU SpecAugment on `(B, 1, 128, 128)` spectrograms during training."""
    b, c, f_dim, t_dim = spec.shape
    out = spec.clone()
    for i in range(b):
        f_len = int(torch.randint(0, max_f_mask + 1, (1,)).item())
        f0 = int(torch.randint(0, max(1, f_dim - f_len), (1,)).item())
        out[i, :, f0 : f0 + f_len, :] = 0.0

        t_len = int(torch.randint(0, max_t_mask + 1, (1,)).item())
        t0 = int(torch.randint(0, max(1, t_dim - t_len), (1,)).item())
        out[i, :, :, t0 : t0 + t_len] = 0.0
    return out


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        spec = batch["spectrogram"].to(device, non_blocking=True)
        demo = batch["demographics"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        spec = apply_spec_augment(spec)

        optimizer.zero_grad()
        outputs = model(spec, demo)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        spec = batch["spectrogram"].to(device, non_blocking=True)
        demo = batch["demographics"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(spec, demo)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def export_to_finn(cascade: BinaryDiagnosisCascadeQAT, checkpoint_dir: str = "checkpoints") -> Dict[str, str]:
    """Exports all 3 binary specialist backbones to QONNX with power-of-2 shape `(1, 1, 128, 128)`."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    cascade.eval()
    cascade_cpu = cascade.to("cpu")
    dummy_input = torch.rand(1, 1, 128, 128)
    exported = {}

    for gate_name, gate_model in [
        ("gate1_healthy_vs_diseased", cascade_cpu.gate1),
        ("gate2_copd_vs_acute", cascade_cpu.gate2),
        ("gate3_pneumonia_vs_urti", cascade_cpu.gate3),
    ]:
        out_path = os.path.join(checkpoint_dir, f"finn_{gate_name}_backbone.onnx")
        try:
            if BREVITAS_EXPORT_AVAILABLE:
                QONNXManager.export(gate_model.backbone, input_t=dummy_input, export_path=out_path)
            else:
                torch.onnx.export(gate_model.backbone, dummy_input, out_path, opset_version=13)
            exported[gate_name] = out_path
            print(f"[FINN Export] Saved {out_path}")
        except Exception as e:
            exported[gate_name] = f"Skipped ONNX export ({e})"
    return exported


def run_qat_finetuning(
    data_dir: str = "data/ICBHI_final_database",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 8e-4,
    bit_width: int = 4,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[QAT Training] Using device: {device} | Bit-width: W{bit_width}A{bit_width}")

    cascade = BinaryDiagnosisCascadeQAT(weight_bit_width=bit_width, act_bit_width=bit_width).to(device)
    gate_specs = [
        ("gate1", cascade.gate1, "Gate 1: Healthy (0) vs. Diseased (1)"),
        ("gate2", cascade.gate2, "Gate 2: COPD (0) vs. Acute Infection (1)"),
        ("gate3", cascade.gate3, "Gate 3: Pneumonia (0) vs. URTI (1)"),
    ]

    for gate_mode, gate_model, title in gate_specs:
        ds = ICBHIDataset(data_dir=data_dir, gate_mode=gate_mode)
        if len(ds) == 0:
            print(f"[WARNING] 0 samples found for {title}. Check data_dir={data_dir}")
            continue

        # Stratified 80/20 split per binary class
        idx_c0 = [i for i, s in enumerate(ds.samples) if s["label"] == 0]
        idx_c1 = [i for i, s in enumerate(ds.samples) if s["label"] == 1]
        np.random.seed(42)
        np.random.shuffle(idx_c0)
        np.random.shuffle(idx_c1)

        c0_split = max(1, int(0.8 * len(idx_c0)))
        c1_split = max(1, int(0.8 * len(idx_c1)))
        train_idx = idx_c0[:c0_split] + idx_c1[:c1_split]
        val_idx = idx_c0[c0_split:] + idx_c1[c1_split:]

        train_sub = torch.utils.data.Subset(ds, train_idx)
        val_sub = torch.utils.data.Subset(ds, val_idx)

        # Exact 50%/50% class balancing via WeightedRandomSampler
        train_labels = np.array([ds.samples[i]["label"] for i in train_idx], dtype=np.int64)
        counts = np.bincount(train_labels, minlength=2)
        weights = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(
            torch.tensor(weights[train_labels], dtype=torch.double),
            num_samples=max(len(train_labels), 512),
            replacement=True,
        )

        train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, pin_memory=True)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(gate_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        ckpt_path = os.path.join(checkpoint_dir, f"best_{gate_mode}.pth")
        print(f"\n--- Training {title} | Train cycles: {len(train_idx)} | Val cycles: {len(val_idx)} ---")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(gate_model, train_loader, criterion, optimizer, device)
            va_loss, va_acc = validate(gate_model, val_loader, criterion, device)
            scheduler.step()

            if va_acc >= best_val_acc:
                best_val_acc = va_acc
                torch.save(gate_model.state_dict(), ckpt_path)

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"[{gate_mode.upper()}] Epoch {epoch:02d}/{epochs} | "
                    f"Train Acc: {tr_acc:.1f}% | Val Acc: {va_acc:.1f}% (Best: {best_val_acc:.1f}%)"
                )

        # Reload best weights for this gate before saving full cascade
        gate_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    cascade_ckpt = os.path.join(checkpoint_dir, "best_icbhi_model.pth")
    torch.save(cascade.state_dict(), cascade_ckpt)
    print(f"\n[SUCCESS] Saved full 3-gate cascade checkpoint to {cascade_ckpt}")
    export_to_finn(cascade, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ICBHI_final_database")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    run_qat_finetuning(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bit_width=args.bits,
    )