import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataset import ICBHIDataset, DIAGNOSIS_NAMES
from model import HierarchicalICBHIModel


# ==============================================================================
# 1. HARDWARE SIM-TO-REAL ACOUSTIC AUGMENTATION (TRAINING ONLY)
# ==============================================================================

def augment_spectrogram(
    spec: torch.Tensor, max_f_mask: int = 10, max_t_mask: int = 14
) -> torch.Tensor:
    """
    Simulates analog front-end & stethoscope domain variation without erasing
    short (<20ms) crackles:
      1. Random gain offset (+/- 0.06 in normalized log2 domain)
      2. Random spectral tilt (simulates analog low/high-pass filter tolerances)
      3. Single gentle frequency & time SpecAugment mask
    """
    b, c, f_dim, t_dim = spec.shape
    out = spec.clone()

    # 1. Per-sample gain offset
    gain = torch.empty((b, 1, 1, 1), device=spec.device).uniform_(-0.06, 0.06)
    out = out + gain

    # 2. Mild linear frequency tilt across bins 0..127
    slope = torch.empty((b, 1, 1, 1), device=spec.device).uniform_(-0.08, 0.08)
    freq_ramp = torch.linspace(-0.5, 0.5, f_dim, device=spec.device).view(1, 1, f_dim, 1)
    out = (out + slope * freq_ramp).clamp(0.0, 1.0)

    # 3. Single moderate SpecAugment mask
    for i in range(b):
        f_len = int(torch.randint(0, max_f_mask + 1, (1,)).item())
        if f_len > 0:
            f0 = int(torch.randint(0, max(1, f_dim - f_len), (1,)).item())
            out[i, :, f0 : f0 + f_len, :] = 0.0

        t_len = int(torch.randint(0, max_t_mask + 1, (1,)).item())
        if t_len > 0:
            t0 = int(torch.randint(0, max(1, t_dim - t_len), (1,)).item())
            out[i, :, :, t0 : t0 + t_len] = 0.0

    return out


# ==============================================================================
# 2. HIERARCHICAL MULTI-TASK LOSS (JOINT 4-WAY TREE + CONDITIONAL GATE LOSSES)
# ==============================================================================

def compute_hierarchical_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Combines:
      1. Weighted 4-class NLL loss on the exact hierarchical tree log-probabilities
      2. Conditional binary cross-entropy on each active branch gate so Gate 1,
         Gate 2, and Gate 3 stay calibrated on their respective subsets.
    """
    tree_log_probs = outputs["tree_log_probs"]
    nll_loss = F.nll_loss(tree_log_probs, labels, weight=class_weights)

    # Gate 1: Healthy (0) vs. Diseased (1, 2, 3) - all samples
    y_g1 = (labels > 0).long()
    loss_g1 = F.cross_entropy(outputs["g1_logits"], y_g1, label_smoothing=label_smoothing)

    # Gate 2: COPD (1 -> 0) vs. Acute (2, 3 -> 1) - diseased samples only
    diseased_mask = labels > 0
    if diseased_mask.any():
        y_g2 = (labels[diseased_mask] >= 2).long()
        loss_g2 = F.cross_entropy(
            outputs["g2_logits"][diseased_mask], y_g2, label_smoothing=label_smoothing
        )
    else:
        loss_g2 = 0.0

    # Gate 3: Pneumonia (2 -> 0) vs. URTI (3 -> 1) - acute samples only
    acute_mask = labels >= 2
    if acute_mask.any():
        y_g3 = (labels[acute_mask] == 3).long()
        loss_g3 = F.cross_entropy(
            outputs["g3_logits"][acute_mask], y_g3, label_smoothing=label_smoothing
        )
    else:
        loss_g3 = 0.0

    return nll_loss + 0.5 * (loss_g1 + loss_g2 + loss_g3)


# ==============================================================================
# 3. INFERENCE, PRIOR ADJUSTMENT, PATIENT AGGREGATION & ROC THRESHOLDS
# ==============================================================================

@torch.no_grad()
def predict_subset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    zero_demographics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_log_probs, all_labels, all_pids = [], [], []

    for batch in dataloader:
        spec = batch["spectrogram"].to(device)
        demo = batch["demographics"].to(device)
        if zero_demographics:
            demo = torch.zeros_like(demo)

        out = model(spec, demo)
        all_log_probs.append(out["tree_log_probs"].cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_pids.extend(batch["pid"])

    return (
        np.concatenate(all_log_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_pids,
    )


def apply_prior_adjustment(
    log_probs: np.ndarray, log_priors: np.ndarray, tau: float
) -> np.ndarray:
    """Subtracts tau * log(prior) from tree log-probabilities and re-normalizes."""
    adj = log_probs - tau * log_priors[None, :]
    adj -= np.max(adj, axis=-1, keepdims=True)
    probs = np.exp(adj)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def aggregate_by_patient(
    log_probs: np.ndarray, labels: np.ndarray, pids: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Averages cycle log-probabilities per patient (mirroring multi-breath
    aggregation on the deployed system).
    """
    pid_to_lps = defaultdict(list)
    pid_to_label = {}
    for lp, y, pid in zip(log_probs, labels, pids):
        pid_to_lps[pid].append(lp)
        pid_to_label[pid] = int(y)

    unique_pids = sorted(pid_to_lps.keys())
    agg_lps = np.stack([np.mean(pid_to_lps[p], axis=0) for p in unique_pids], axis=0)
    agg_y = np.array([pid_to_label[p] for p in unique_pids], dtype=np.int64)
    return agg_lps, agg_y, unique_pids


def tune_prior_tau(
    val_log_probs: np.ndarray, val_labels: np.ndarray, log_priors: np.ndarray
) -> float:
    best_tau, best_macro = 0.0, -1.0
    for tau in np.linspace(0.0, 1.0, 21):
        probs = apply_prior_adjustment(val_log_probs, log_priors, float(tau))
        preds = np.argmax(probs, axis=-1)
        recalls = [
            np.mean(preds[val_labels == c] == c) if np.any(val_labels == c) else 0.0
            for c in range(4)
        ]
        macro_rec = float(np.mean(recalls))
        if macro_rec > best_macro:
            best_macro = macro_rec
            best_tau = float(tau)
    return best_tau


def tune_roc_thresholds(val_probs: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """
    Tunes per-class confidence thresholds on validation patients using Youden's J
    (with sensitivity-first 1.5x weighting for Pneumonia = class 2) to support
    an 'Uncertain / Re-sample' rejection output.
    """
    thresholds = np.full(4, 0.35, dtype=np.float32)
    for c in range(4):
        y_bin = (val_labels == c).astype(np.int64)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, ths = roc_curve(y_bin, val_probs[:, c])
        se_weight = 1.5 if c == 2 else 1.0  # Sensitivity-first for Pneumonia
        score = se_weight * tpr - fpr
        best_idx = int(np.argmax(score))
        thresholds[c] = float(np.clip(ths[best_idx], 0.15, 0.75))
    return thresholds


def compute_detailed_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    roc_thresholds: np.ndarray = None,
) -> Dict:
    y_pred = np.argmax(probs, axis=-1)
    overall_acc = 100.0 * float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class = {}
    recalls = []
    for c, cname in enumerate(DIAGNOSIS_NAMES):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        se = 100.0 * tp / max(tp + fn, 1)
        sp = 100.0 * tn / max(tn + fp, 1)
        recalls.append(se)

        y_bin = (y_true == c).astype(np.int64)
        auc = float(roc_auc_score(y_bin, probs[:, c])) if len(np.unique(y_bin)) > 1 else 0.5
        per_class[cname] = {
            "support": int(tp + fn),
            "recall_se_pct": round(se, 2),
            "specificity_pct": round(sp, 2),
            "ovr_roc_auc": round(auc, 4),
        }

    report = {
        "overall_accuracy_pct": round(overall_acc, 2),
        "macro_recall_pct": round(float(np.mean(recalls)), 2),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    if roc_thresholds is not None:
        top_probs = probs[np.arange(len(y_pred)), y_pred]
        confident_mask = top_probs >= roc_thresholds[y_pred]
        coverage = 100.0 * float(np.mean(confident_mask))
        conf_acc = (
            100.0 * float(np.mean(y_true[confident_mask] == y_pred[confident_mask]))
            if np.any(confident_mask)
            else 0.0
        )
        report["uncertain_rejection"] = {
            "confident_coverage_pct": round(coverage, 2),
            "uncertain_rate_pct": round(100.0 - coverage, 2),
            "confident_accuracy_pct": round(conf_acc, 2),
        }

    return report


def print_report_block(title: str, report: Dict) -> None:
    print("\n" + "=" * 88)
    print(
        f" {title} | Macro Recall: {report['macro_recall_pct']:.2f}% | "
        f"Overall Acc: {report['overall_accuracy_pct']:.2f}%"
    )
    print("=" * 88)
    print(f"{'Diagnosis':<12} | {'Support':<8} | {'Recall (Se)':<13} | {'Specificity':<13} | {'OvR ROC-AUC':<11}")
    print("-" * 88)
    for cname in DIAGNOSIS_NAMES:
        m = report["per_class"][cname]
        print(
            f"{cname:<12} | {m['support']:<8d} | {m['recall_se_pct']:>10.2f}%  | "
            f"{m['specificity_pct']:>10.2f}%  | {m['ovr_roc_auc']:>9.4f}"
        )
    print("\nConfusion Matrix (rows=True [Healthy, COPD, Pneumonia, URTI], cols=Pred):")
    for cname, row in zip(DIAGNOSIS_NAMES, report["confusion_matrix"]):
        print(f"  {cname:<10}: {row}")
    if "uncertain_rejection" in report:
        u = report["uncertain_rejection"]
        print(
            f"\nConfidence Gate -> Confident Coverage: {u['confident_coverage_pct']:.1f}% | "
            f"Flagged 'Uncertain': {u['uncertain_rate_pct']:.1f}% | "
            f"Accuracy on Confident Calls: {u['confident_accuracy_pct']:.2f}%"
        )


# ==============================================================================
# 4. MAIN TRAINING & LEAK-FREE EVALUATION DRIVER
# ==============================================================================

def train_fp32_baseline(
    data_dir: str = "data/icbhi_dataset",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1e-3,
    demo_dropout: float = 0.40,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FP32 Baseline] Device: {device} | Epochs: {epochs} | Demo Dropout: {demo_dropout}")

    dataset = ICBHIDataset(data_dir=data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid cycles loaded from {data_dir}.")

    # Audit patient and cycle counts per diagnosis
    print("\n--- Dataset Patient & Cycle Audit ---")
    for c, cname in enumerate(DIAGNOSIS_NAMES):
        c_mask = dataset.labels == c
        n_cycles = int(np.sum(c_mask))
        n_patients = len(np.unique(dataset.patient_ids[c_mask]))
        print(f"  {cname:<10}: {n_patients:>3d} patients | {n_cycles:>4d} cycles")

    # Global StratifiedGroupKFold (5 folds):
    #   Fold 0 -> Locked Final Test Set (never touched during training/tuning)
    #   Fold 1 -> Validation Set (checkpoint & threshold selection)
    #   Folds 2, 3, 4 -> Training Set
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(dataset)), dataset.labels, groups=dataset.patient_ids))
    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.concatenate([folds[2][1], folds[3][1], folds[4][1]])

    # Verify zero patient overlap across splits
    train_pids = set(dataset.patient_ids[train_idx])
    val_pids = set(dataset.patient_ids[val_idx])
    test_pids = set(dataset.patient_ids[test_idx])
    assert len(train_pids & val_pids) == 0 and len(train_pids & test_pids) == 0 and len(val_pids & test_pids) == 0
    print(
        f"\n[Patient-Disjoint Split Verified] "
        f"Train: {len(train_pids)} pts ({len(train_idx)} cycles) | "
        f"Val: {len(val_pids)} pts ({len(val_idx)} cycles) | "
        f"Locked Test: {len(test_pids)} pts ({len(test_idx)} cycles)"
    )

    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    test_sub = Subset(dataset, test_idx)

    # Balanced class sampling on training split + smoothed inverse class weights
    train_labels = dataset.labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=4)
    priors = class_counts / np.sum(class_counts)
    log_priors = np.log(np.maximum(priors, 1e-6))

    sample_weights = (1.0 / np.maximum(class_counts, 1))[train_labels]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False)

    smooth_weights = torch.tensor(
        np.power(np.sum(class_counts) / (4.0 * np.maximum(class_counts, 1)), 0.35),
        dtype=torch.float32,
        device=device,
    )

    model = HierarchicalICBHIModel(demo_dim=6, demo_dropout_p=demo_dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Shared-Backbone Hierarchical FP32 CNN | Total Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_macro = -1.0
    ckpt_path = os.path.join(checkpoint_dir, "best_fp32_hierarchical_model.pth")

    print("\n--- Starting Leak-Free FP32 Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            spec = augment_spectrogram(batch["spectrogram"].to(device))
            demo = batch["demographics"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(spec, demo)
            loss = compute_hierarchical_loss(out, labels, smooth_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_idx)

        val_lps, val_y, _ = predict_subset(model, val_loader, device)
        val_probs = np.exp(val_lps)
        val_rep = compute_detailed_metrics(val_probs, val_y)

        if val_rep["macro_recall_pct"] >= best_val_macro:
            best_val_macro = val_rep["macro_recall_pct"]
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"[Ep {epoch:02d}/{epochs:02d}] Loss: {train_loss:.4f} | "
            f"Val Macro Recall: {val_rep['macro_recall_pct']:5.2f}% | "
            f"Val Acc: {val_rep['overall_accuracy_pct']:5.2f}% | "
            f"Recalls [H/C/P/U]: "
            f"{val_rep['per_class']['Healthy']['recall_se_pct']:.0f}/"
            f"{val_rep['per_class']['COPD']['recall_se_pct']:.0f}/"
            f"{val_rep['per_class']['Pneumonia']['recall_se_pct']:.0f}/"
            f"{val_rep['per_class']['URTI']['recall_se_pct']:.0f}%"
        )

    # Load best validation checkpoint and calibrate prior adjustment + ROC thresholds on Val
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    val_lps, val_y, _ = predict_subset(model, val_loader, device)
    best_tau = tune_prior_tau(val_lps, val_y, log_priors)
    val_cal_probs = apply_prior_adjustment(val_lps, log_priors, best_tau)
    roc_thresholds = tune_roc_thresholds(val_cal_probs, val_y)
    print(f"\n[Calibration on Val] Optimal Prior Tau: {best_tau:.2f} | ROC Thresholds: {np.round(roc_thresholds, 3).tolist()}")

    # Final evaluation strictly on the LOCKED TEST SET (unseen patients)
    test_lps, test_y, test_pids_list = predict_subset(model, test_loader, device, zero_demographics=False)
    test_cal_probs = apply_prior_adjustment(test_lps, log_priors, best_tau)
    cycle_report = compute_detailed_metrics(test_cal_probs, test_y, roc_thresholds=roc_thresholds)
    print_report_block("LOCKED TEST SET (PER-CYCLE, PRIOR-CALIBRATED)", cycle_report)

    # Patient-level multi-cycle log-probability aggregation on Locked Test Set
    pat_lps, pat_y, _ = aggregate_by_patient(test_lps, test_y, test_pids_list)
    pat_cal_probs = apply_prior_adjustment(pat_lps, log_priors, best_tau)
    patient_report = compute_detailed_metrics(pat_cal_probs, pat_y, roc_thresholds=roc_thresholds)
    print_report_block("LOCKED TEST SET (PATIENT-LEVEL LOG-PROB AGGREGATED)", patient_report)

    # Zero-Demographics Ablation Check on Locked Test Set
    nodemo_lps, _, _ = predict_subset(model, test_loader, device, zero_demographics=True)
    nodemo_cal_probs = apply_prior_adjustment(nodemo_lps, log_priors, best_tau)
    nodemo_report = compute_detailed_metrics(nodemo_cal_probs, test_y)
    print_report_block("ABLATION CHECK: LOCKED TEST SET WITH ZEROED DEMOGRAPHICS", nodemo_report)

    full_report = {
        "best_val_macro_recall_pct": best_val_macro,
        "calibrated_prior_tau": best_tau,
        "roc_thresholds": [round(float(x), 4) for x in roc_thresholds],
        "locked_test_per_cycle": cycle_report,
        "locked_test_per_patient": patient_report,
        "locked_test_zero_demographics_ablation": nodemo_report,
    }
    report_path = os.path.join(checkpoint_dir, "fp32_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Done] Saved FP32 weights to {ckpt_path} and full report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/icbhi_dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--demo_dropout", type=float, default=0.40)
    args = parser.parse_args()

    train_fp32_baseline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        demo_dropout=args.demo_dropout,
    )