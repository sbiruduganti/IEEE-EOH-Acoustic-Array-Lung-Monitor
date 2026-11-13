"""
Drop-in replacement for `software/src/train_qat.py`
Repository: https://github.com/sbiruduganti/IEEE-EOH-Acoustic-Array-Lung-Monitor
Baseline Aligned with: `10_29_26_trained_circpad_other_fixes.ipynb` (25 epochs + ROC-AUC / Sensitivity / Specificity)

Key Features:
1. Defaults to `--epochs 25` and evaluates every epoch on:
   - Accuracy (`Acc %`)
   - Sensitivity (`Se %` / Recall / True Positive Rate)
   - Specificity (`Sp %` / True Negative Rate)
   - Official ICBHI Score (`SP % = 0.5 * (Se + Sp)`)
   - ROC-AUC (`auc`, computed via `sklearn.metrics.roc_auc_score` or exact trapezoidal fallback)
   - Optimal Youden's J threshold (`tau_opt = argmax(Se + Sp - 1)`)
2. Saves best checkpoints per gate based on ROC-AUC + ICBHI Score, generates a 3-panel ROC Curve
   figure (`checkpoints/roc_curves_3gates.png`), writes `checkpoints/validation_metrics_report.json`,
   and runs an end-to-end 4-Diagnosis Cascade evaluation (`Healthy`, `COPD`, `Pneumonia`, `URTI`).
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.dataset import ICBHIDataset, TARGET_DIAGNOSES
    from src.model_qat import ICBHIModelQAT, BinaryDiagnosisCascadeQAT
except ImportError:
    from dataset import ICBHIDataset, TARGET_DIAGNOSES
    from model_qat import ICBHIModelQAT, BinaryDiagnosisCascadeQAT

try:
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from brevitas.export.onnx.qonnx.manager import QONNXManager
    BREVITAS_EXPORT_AVAILABLE = True
except ImportError:
    BREVITAS_EXPORT_AVAILABLE = False


# ==============================================================================
# 1. BINARY & MULTI-CLASS ROC-AUC, SENSITIVITY & SPECIFICITY METRICS
# ==============================================================================

def compute_binary_roc_metrics(
    y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes Accuracy, Sensitivity (TPR), Specificity (TNR), ICBHI Score (0.5*(Se+Sp)),
    ROC-AUC, FPR/TPR curve arrays, and the optimal Youden's J threshold `tau_opt`.
    Works with `scikit-learn` or pure NumPy fallback.
    """
    y_pred = (y_prob_pos >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(tp + tn + fp + fn, 1)
    acc = 100.0 * (tp + tn) / total
    sensitivity = 100.0 * tp / max(tp + fn, 1)
    specificity = 100.0 * tn / max(tn + fp, 1)
    icbhi_score = 0.5 * (sensitivity + specificity)

    if len(np.unique(y_true)) > 1:
        if SKLEARN_AVAILABLE:
            auc_val = float(roc_auc_score(y_true, y_prob_pos))
            fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
        else:
            thresholds = np.linspace(1.0, 0.0, 201)
            tpr_list, fpr_list = [], []
            for th in thresholds:
                p = (y_prob_pos >= th).astype(np.int64)
                tpr_list.append(np.sum((y_true == 1) & (p == 1)) / max(np.sum(y_true == 1), 1))
                fpr_list.append(np.sum((y_true == 0) & (p == 1)) / max(np.sum(y_true == 0), 1))
            fpr = np.array(fpr_list)
            tpr = np.array(tpr_list)
            auc_val = float(np.trapz(tpr, fpr))

        # Optimal Youden's J threshold: maximizes (Sensitivity + Specificity - 1)
        youden_j = tpr - fpr
        best_idx = int(np.argmax(youden_j))
        tau_opt = float(np.clip(thresholds[best_idx], 0.05, 0.95))
        opt_se = 100.0 * float(tpr[best_idx])
        opt_sp = 100.0 * float(1.0 - fpr[best_idx])
    else:
        auc_val = 0.5
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        tau_opt, opt_se, opt_sp = 0.50, sensitivity, specificity

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "icbhi_score": icbhi_score,
        "roc_auc": auc_val,
        "tau_opt": tau_opt,
        "opt_sensitivity": opt_se,
        "opt_specificity": opt_sp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fpr": fpr,
        "tpr": tpr,
    }


def apply_spec_augment(spec: torch.Tensor, max_f_mask: int = 12, max_t_mask: int = 16) -> torch.Tensor:
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


# ==============================================================================
# 2. TRAINING & VALIDATION LOOPS (WITH ROC-AUC & SENSITIVITY/SPECIFICITY)
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

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
        probs_pos = F.softmax(outputs.detach(), dim=-1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs_pos.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_binary_roc_metrics(y_true, y_prob)
    metrics["loss"] = running_loss / max(len(y_true), 1)
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for batch in dataloader:
        spec = batch["spectrogram"].to(device, non_blocking=True)
        demo = batch["demographics"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(spec, demo)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        probs_pos = F.softmax(outputs, dim=-1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs_pos.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_binary_roc_metrics(y_true, y_prob)
    metrics["loss"] = running_loss / max(len(y_true), 1)
    return metrics


# ==============================================================================
# 3. END-TO-END 4-DIAGNOSIS CASCADE EVALUATION & ROC PLOTTING
# ==============================================================================

@torch.no_grad()
def evaluate_full_cascade(
    cascade: BinaryDiagnosisCascadeQAT,
    data_dir: str,
    val_pids: set,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates the combined 3-Gate Binary Cascade across all 4 specific diagnoses
    (`0: Healthy`, `1: COPD`, `2: Pneumonia`, `3: URTI`) on the held-out validation split,
    computing per-diagnosis Sensitivity, Specificity, OvR ROC-AUC, and Overall Accuracy.
    """
    cascade.eval()
    eval_ds = ICBHIDataset(data_dir=data_dir, gate_mode="cascade_eval")
    val_indices = [i for i, s in enumerate(eval_ds.samples) if s["pid"] in val_pids]
    if not val_indices:
        val_indices = list(range(len(eval_ds)))

    val_loader = DataLoader(
        torch.utils.data.Subset(eval_ds, val_indices), batch_size=batch_size, shuffle=False
    )

    all_y, all_probs = [], []
    for batch in val_loader:
        spec = batch["spectrogram"].to(device)
        demo = batch["demographics"].to(device)
        probs_4 = cascade(spec, demo)  # (B, 4) calibrated probabilities
        all_probs.append(probs_4.cpu().numpy())
        all_y.append(batch["label"].numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=-1)

    overall_acc = 100.0 * float(np.mean(y_true == y_pred))
    per_class_report = {}
    diag_names = ["Healthy", "COPD", "Pneumonia", "URTI"]

    print("\n=====================================================================================")
    print(f" END-TO-END 4-DIAGNOSIS BINARY CASCADE VALIDATION | Overall Accuracy: {overall_acc:.2f}%")
    print("=====================================================================================")
    print(f"{'Diagnosis':<14} | {'ROC-AUC':<9} | {'Sensitivity (Se)':<18} | {'Specificity (Sp)':<18} | {'ICBHI Score':<12}")
    print("-" * 85)

    for cid, cname in enumerate(diag_names):
        bin_true = (y_true == cid).astype(np.int64)
        bin_prob = y_prob[:, cid]
        m = compute_binary_roc_metrics(bin_true, bin_prob, threshold=0.25)
        # Also compute exact multi-class argmax Se/Sp for this class
        tp = int(np.sum((y_true == cid) & (y_pred == cid)))
        fn = int(np.sum((y_true == cid) & (y_pred != cid)))
        tn = int(np.sum((y_true != cid) & (y_pred != cid)))
        fp = int(np.sum((y_true != cid) & (y_pred == cid)))
        se = 100.0 * tp / max(tp + fn, 1)
        sp = 100.0 * tn / max(tn + fp, 1)
        sp_score = 0.5 * (se + sp)

        print(f"{cname:<14} | {m['roc_auc']:<9.4f} | {se:>15.2f}%  | {sp:>15.2f}%  | {sp_score:>10.2f}%")
        per_class_report[cname] = {
            "ovr_roc_auc": round(m["roc_auc"], 4),
            "sensitivity_pct": round(se, 2),
            "specificity_pct": round(sp, 2),
            "icbhi_score_pct": round(sp_score, 2),
        }

    return {"overall_accuracy_pct": round(overall_acc, 2), "per_diagnosis": per_class_report}


def save_roc_curve_plot(gate_roc_data: Dict[str, Dict[str, Any]], out_png: str) -> None:
    if not MATPLOTLIB_AVAILABLE or not gate_roc_data:
        return
    fig, axes = plt.subplots(1, len(gate_roc_data), figsize=(15, 4.5))
    if len(gate_roc_data) == 1:
        axes = [axes]
    for ax, (gname, m) in zip(axes, gate_roc_data.items()):
        ax.plot(m["fpr"], m["tpr"], lw=2.2, label=f"AUC = {m['roc_auc']:.3f}\nAcc = {m['accuracy']:.1f}%")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0)
        ax.set_title(gname, fontsize=11, fontweight="bold")
        ax.set_xlabel("1 - Specificity (False Positive Rate)")
        ax.set_ylabel("Sensitivity (True Positive Rate)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[ROC Plot] Saved 3-gate ROC curve plot to {out_png}")


def export_to_finn(cascade: BinaryDiagnosisCascadeQAT, checkpoint_dir: str = "checkpoints") -> Dict[str, str]:
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


# ==============================================================================
# 4. MAIN 25-EPOCH TRAINING & ROC-AUC VALIDATION DRIVER
# ==============================================================================

def run_qat_finetuning(
    data_dir: str = "data/ICBHI_final_database",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 8e-4,
    bit_width: int = 4,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[QAT Training] Device: {device} | Quantization: W{bit_width}A{bit_width} | Epochs: {epochs}")

    cascade = BinaryDiagnosisCascadeQAT(weight_bit_width=bit_width, act_bit_width=bit_width).to(device)
    gate_specs = [
        ("gate1", cascade.gate1, "Gate 1: Healthy (0) vs. Diseased (1)"),
        ("gate2", cascade.gate2, "Gate 2: COPD (0) vs. Acute Infection (1)"),
        ("gate3", cascade.gate3, "Gate 3: Pneumonia (0) vs. URTI (1)"),
    ]

    best_gate_metrics: Dict[str, Dict[str, Any]] = {}
    all_val_pids: set = set()

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

        for i in val_idx:
            all_val_pids.add(ds.samples[i]["pid"])

        train_sub = torch.utils.data.Subset(ds, train_idx)
        val_sub = torch.utils.data.Subset(ds, val_idx)

        # Balanced 50%/50% WeightedRandomSampler for training
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

        best_Composite = -1.0
        ckpt_path = os.path.join(checkpoint_dir, f"best_{gate_mode}.pth")
        print(f"\n=== {title} | Train: {len(train_idx)} | Val: {len(val_idx)} ===")

        for epoch in range(1, epochs + 1):
            tr_m = train_one_epoch(gate_model, train_loader, criterion, optimizer, device)
            va_m = validate(gate_model, val_loader, criterion, device)
            scheduler.step()

            # Select best checkpoint by combined ROC-AUC + ICBHI (Se+Sp)/2 score
            composite_score = 100.0 * va_m["roc_auc"] + va_m["icbhi_score"]
            if composite_score >= best_Composite:
                best_Composite = composite_score
                best_gate_metrics[title] = va_m
                torch.save(gate_model.state_dict(), ckpt_path)

            print(
                f"[{gate_mode.upper()} Ep {epoch:02d}/{epochs}] "
                f"Train Acc: {tr_m['accuracy']:5.1f}% (AUC {tr_m['roc_auc']:.3f}) | "
                f"Val Acc: {va_m['accuracy']:5.1f}% | AUC: {va_m['roc_auc']:.4f} | "
                f"Se: {va_m['sensitivity']:5.1f}% | Sp: {va_m['specificity']:5.1f}% | "
                f"ICBHI: {va_m['icbhi_score']:5.1f}% (tau*={va_m['tau_opt']:.2f})"
            )

        gate_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Save ROC curve plot across all 3 binary specialist gates
    save_roc_curve_plot(best_gate_metrics, os.path.join(checkpoint_dir, "roc_curves_3gates.png"))

    # Run End-to-End 4-Diagnosis Cascade Evaluation
    cascade_report = evaluate_full_cascade(cascade, data_dir, all_val_pids, batch_size, device)

    # Save JSON metrics summary
    summary_json = {
        "gates": {
            k: {
                "best_val_accuracy_pct": round(v["accuracy"], 2),
                "best_roc_auc": round(v["roc_auc"], 4),
                "sensitivity_pct": round(v["sensitivity"], 2),
                "specificity_pct": round(v["specificity"], 2),
                "icbhi_score_pct": round(v["icbhi_score"], 2),
                "youden_optimal_threshold": round(v["tau_opt"], 3),
                "opt_sensitivity_pct": round(v["opt_sensitivity"], 2),
                "opt_specificity_pct": round(v["opt_specificity"], 2),
            }
            for k, v in best_gate_metrics.items()
        },
        "end_to_end_4_diagnosis_cascade": cascade_report,
    }
    report_path = os.path.join(checkpoint_dir, "validation_metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[Metrics Report] Saved full ROC-AUC & Sensitivity/Specificity report to {report_path}")

    cascade_ckpt = os.path.join(checkpoint_dir, "best_icbhi_model.pth")
    torch.save(cascade.state_dict(), cascade_ckpt)
    export_to_finn(cascade, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ICBHI_final_database")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
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