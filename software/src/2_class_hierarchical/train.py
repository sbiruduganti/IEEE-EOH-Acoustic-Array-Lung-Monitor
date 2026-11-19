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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score

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

    gain = torch.empty((b, 1, 1, 1), device=spec.device).uniform_(-0.06, 0.06)
    out = out + gain

    slope = torch.empty((b, 1, 1, 1), device=spec.device).uniform_(-0.08, 0.08)
    freq_ramp = torch.linspace(-0.5, 0.5, f_dim, device=spec.device).view(1, 1, f_dim, 1)
    out = (out + slope * freq_ramp).clamp(0.0, 1.0)

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
# 2. BALANCED HIERARCHICAL MULTI-TASK LOSS
# ==============================================================================

def compute_hierarchical_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Because WeightedRandomSampler already balances the 4 classes (25% each) in every
    training batch:
      - 4-way tree NLL uses unweighted classes (no double-reweighting).
      - Gate 1 (Healthy 25% vs. Diseased 75%) uses [3.0, 1.0] weights so Healthy
        and Diseased receive equal gradient weight.
      - Gate 2 (COPD 25% vs. Acute 50%) uses [2.0, 1.0] weights so COPD and Acute
        receive equal gradient weight.
      - Gate 3 (Pneumonia 25% vs. URTI 25%) is already 50/50 balanced.
    """
    tree_log_probs = outputs["tree_log_probs"]
    nll_loss = F.nll_loss(tree_log_probs, labels)

    # Gate 1: Healthy (0) vs. Diseased (1, 2, 3)
    y_g1 = (labels > 0).long()
    g1_weights = torch.tensor([3.0, 1.0], device=labels.device)
    loss_g1 = F.cross_entropy(
        outputs["g1_logits"], y_g1, weight=g1_weights, label_smoothing=label_smoothing
    )

    # Gate 2: COPD (1 -> 0) vs. Acute (2, 3 -> 1)
    diseased_mask = labels > 0
    if diseased_mask.any():
        y_g2 = (labels[diseased_mask] >= 2).long()
        g2_weights = torch.tensor([2.0, 1.0], device=labels.device)
        loss_g2 = F.cross_entropy(
            outputs["g2_logits"][diseased_mask],
            y_g2,
            weight=g2_weights,
            label_smoothing=label_smoothing,
        )
    else:
        loss_g2 = 0.0

    # Gate 3: Pneumonia (2 -> 0) vs. URTI (3 -> 1)
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
# 3. PATIENT-LEVEL STRATIFIED SPLITTING
# ==============================================================================

def build_patient_stratified_folds(
    dataset: ICBHIDataset, n_splits: int = 5, seed: int = 42
) -> List[np.ndarray]:
    """
    Stratifies strictly at the PATIENT level (unique pid -> diagnosis) so every
    fold receives an equal share of Healthy, COPD, Pneumonia, and URTI patients.
    """
    unique_pids, first_indices = np.unique(dataset.patient_ids, return_index=True)
    unique_labels = dataset.labels[first_indices]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_cycle_indices: List[np.ndarray] = []

    print(f"\n--- Patient-Level {n_splits}-Fold Stratification Audit ---")
    for fold_idx, (_, pat_val_idx) in enumerate(skf.split(unique_pids, unique_labels)):
        fold_pids = unique_pids[pat_val_idx]
        cycle_idx = np.where(np.isin(dataset.patient_ids, fold_pids))[0]
        fold_cycle_indices.append(cycle_idx)

        pat_counts = np.bincount(unique_labels[pat_val_idx], minlength=4)
        cyc_counts = np.bincount(dataset.labels[cycle_idx], minlength=4)
        print(
            f"  Fold {fold_idx}: {len(fold_pids):>2d} pts [H={pat_counts[0]}, C={pat_counts[1]}, "
            f"P={pat_counts[2]}, U={pat_counts[3]}] | "
            f"{len(cycle_idx):>4d} cycles [H={cyc_counts[0]}, C={cyc_counts[1]}, "
            f"P={cyc_counts[2]}, U={cyc_counts[3]}]"
        )

    return fold_cycle_indices


# ==============================================================================
# 4. INFERENCE, REGULARIZED OFFSET CALIBRATION & PATIENT AGGREGATION
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


def apply_class_offsets(log_probs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Adds calibrated per-class log-probability offsets and converts to probabilities."""
    adj = log_probs + offsets[None, :]
    adj -= np.max(adj, axis=-1, keepdims=True)
    probs = np.exp(adj)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def tune_class_offsets(
    val_log_probs: np.ndarray, val_labels: np.ndarray, l2_reg: float = 0.08
) -> np.ndarray:
    """
    Regularized coordinate search over per-class log-probability offsets on validation
    patients to maximize Macro-Average Recall without overfitting small folds.
    """
    offsets = np.zeros(4, dtype=np.float32)
    grid = np.linspace(-1.2, 1.2, 25)

    def _score(off: np.ndarray) -> float:
        probs = apply_class_offsets(val_log_probs, off)
        preds = np.argmax(probs, axis=-1)
        recalls = [
            np.mean(preds[val_labels == c] == c)
            for c in range(4)
            if np.any(val_labels == c)
        ]
        return float(np.mean(recalls)) - l2_reg * float(np.mean(off ** 2))

    for _ in range(2):
        for c in range(4):
            best_val = offsets[c]
            best_sc = _score(offsets)
            for cand in grid:
                trial = offsets.copy()
                trial[c] = cand
                sc = _score(trial)
                if sc > best_sc + 1e-5:
                    best_sc = sc
                    best_val = cand
            offsets[c] = best_val

    return offsets - np.mean(offsets)


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


def compute_detailed_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    conf_floor: float = 0.45,
    margin_floor: float = 0.15,
    pneumonia_conf_floor: float = 0.38,
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

    # Margin + confidence-floor Uncertain / Re-sample rejection
    sorted_probs = np.sort(probs, axis=-1)
    top1_prob = sorted_probs[:, -1]
    top2_prob = sorted_probs[:, -2]
    margin = top1_prob - top2_prob

    eff_floor = np.where(y_pred == 2, pneumonia_conf_floor, conf_floor)
    confident_mask = (top1_prob >= eff_floor) & (margin >= margin_floor)
    coverage = 100.0 * float(np.mean(confident_mask))
    conf_acc = (
        100.0 * float(np.mean(y_true[confident_mask] == y_pred[confident_mask]))
        if np.any(confident_mask)
        else 0.0
    )

    return {
        "overall_accuracy_pct": round(overall_acc, 2),
        "macro_recall_pct": round(float(np.mean(recalls)), 2),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "uncertain_rejection": {
            "confident_coverage_pct": round(coverage, 2),
            "uncertain_rate_pct": round(100.0 - coverage, 2),
            "confident_accuracy_pct": round(conf_acc, 2),
        },
    }


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
    u = report["uncertain_rejection"]
    print(
        f"\nConfidence Gate -> Confident Coverage: {u['confident_coverage_pct']:.1f}% | "
        f"Flagged 'Uncertain': {u['uncertain_rate_pct']:.1f}% | "
        f"Accuracy on Confident Calls: {u['confident_accuracy_pct']:.2f}%"
    )


# ==============================================================================
# 5. MAIN 5-FOLD LEAK-FREE TRAINING & OUT-OF-FOLD EVALUATION DRIVER
# ==============================================================================

def train_fp32_baseline(
    data_dir: str = "data/icbhi_dataset",
    checkpoint_dir: str = "checkpoints_fp32",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    demo_dropout: float = 0.50,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[FP32 Baseline] Device: {device} | Folds: {n_folds}/5 | "
        f"Epochs/Fold: {epochs} | Demo Dropout: {demo_dropout}"
    )

    dataset = ICBHIDataset(data_dir=data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid cycles loaded from {data_dir}.")

    fold_cycle_indices = build_patient_stratified_folds(dataset, n_splits=5, seed=seed)

    oof_cal_probs_list, oof_raw_lps_list, oof_nodemo_probs_list = [], [], []
    oof_labels_list, oof_pids_list = [], []
    fold_summaries = []

    best_overall_val_macro = -1.0
    best_global_ckpt = os.path.join(checkpoint_dir, "best_fp32_hierarchical_model.pth")

    for fold_k in range(n_folds):
        test_idx = fold_cycle_indices[fold_k]
        val_fold = (fold_k + 1) % 5
        val_idx = fold_cycle_indices[val_fold]
        train_idx = np.concatenate(
            [fold_cycle_indices[f] for f in range(5) if f not in (fold_k, val_fold)]
        )

        # Verify strict patient-level disjointness
        train_pids = set(dataset.patient_ids[train_idx])
        val_pids = set(dataset.patient_ids[val_idx])
        test_pids = set(dataset.patient_ids[test_idx])
        assert (
            len(train_pids & val_pids) == 0
            and len(train_pids & test_pids) == 0
            and len(val_pids & test_pids) == 0
        )

        print(
            f"\n==================== FOLD {fold_k + 1}/{n_folds} "
            f"(Test=Fold {fold_k}, Val=Fold {val_fold}) ===================="
        )

        train_labels = dataset.labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=4)
        sample_weights = (1.0 / np.maximum(class_counts, 1))[train_labels]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        torch.manual_seed(seed + fold_k)
        model = HierarchicalICBHIModel(demo_dim=6, demo_dropout_p=demo_dropout).to(device)
        if fold_k == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[Model] Shared-Backbone Hierarchical FP32 CNN | Parameters: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_score = -1.0
        fold_ckpt = os.path.join(checkpoint_dir, f"best_fp32_fold{fold_k}.pth")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                spec = augment_spectrogram(batch["spectrogram"].to(device))
                demo = batch["demographics"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                out = model(spec, demo)
                loss = compute_hierarchical_loss(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * labels.size(0)

            scheduler.step()
            train_loss = running_loss / len(train_idx)

            val_lps, val_y, _ = predict_subset(model, val_loader, device)
            val_rep = compute_detailed_metrics(np.exp(val_lps), val_y)
            # Primary selection on Macro Recall with tiny tie-breaker on Overall Acc
            val_score = val_rep["macro_recall_pct"] + 0.01 * val_rep["overall_accuracy_pct"]

            if val_score >= best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), fold_ckpt)
                if val_rep["macro_recall_pct"] >= best_overall_val_macro:
                    best_overall_val_macro = val_rep["macro_recall_pct"]
                    torch.save(model.state_dict(), best_global_ckpt)

            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"  [Ep {epoch:02d}/{epochs:02d}] Loss: {train_loss:.4f} | "
                    f"Val Macro Recall: {val_rep['macro_recall_pct']:5.2f}% | "
                    f"Val Acc: {val_rep['overall_accuracy_pct']:5.2f}% | "
                    f"Recalls [H/C/P/U]: "
                    f"{val_rep['per_class']['Healthy']['recall_se_pct']:.0f}/"
                    f"{val_rep['per_class']['COPD']['recall_se_pct']:.0f}/"
                    f"{val_rep['per_class']['Pneumonia']['recall_se_pct']:.0f}/"
                    f"{val_rep['per_class']['URTI']['recall_se_pct']:.0f}%"
                )

        # Load best validation checkpoint for this fold & calibrate class offsets on Val
        model.load_state_dict(torch.load(fold_ckpt, map_location=device, weights_only=True))
        val_lps, val_y, _ = predict_subset(model, val_loader, device)
        offsets = tune_class_offsets(val_lps, val_y)

        # Predict strictly out-of-sample on Test Fold k
        test_lps, test_y, test_pids_list = predict_subset(
            model, test_loader, device, zero_demographics=False
        )
        test_cal_probs = apply_class_offsets(test_lps, offsets)
        fold_rep = compute_detailed_metrics(test_cal_probs, test_y)

        nodemo_lps, _, _ = predict_subset(
            model, test_loader, device, zero_demographics=True
        )
        nodemo_cal_probs = apply_class_offsets(nodemo_lps, offsets)

        print(
            f"  --> Fold {fold_k} Test Macro Recall: {fold_rep['macro_recall_pct']:.2f}% | "
            f"Acc: {fold_rep['overall_accuracy_pct']:.2f}% | "
            f"Calibrated Offsets: {np.round(offsets, 3).tolist()}"
        )

        oof_cal_probs_list.append(test_cal_probs)
        oof_raw_lps_list.append(test_lps + offsets[None, :])
        oof_nodemo_probs_list.append(nodemo_cal_probs)
        oof_labels_list.append(test_y)
        oof_pids_list.extend(test_pids_list)
        fold_summaries.append(
            {
                "fold": fold_k,
                "offsets": [round(float(x), 4) for x in offsets],
                "test_macro_recall_pct": fold_rep["macro_recall_pct"],
                "test_overall_acc_pct": fold_rep["overall_accuracy_pct"],
            }
        )

    # Aggregate Out-of-Fold predictions across all evaluated folds
    all_oof_probs = np.concatenate(oof_cal_probs_list, axis=0)
    all_oof_lps = np.concatenate(oof_raw_lps_list, axis=0)
    all_oof_nodemo_probs = np.concatenate(oof_nodemo_probs_list, axis=0)
    all_oof_y = np.concatenate(oof_labels_list, axis=0)

    cycle_report = compute_detailed_metrics(all_oof_probs, all_oof_y)
    print_report_block(
        f"OUT-OF-FOLD TEST SET ({n_folds} FOLDS, PER-CYCLE)", cycle_report
    )

    pat_lps, pat_y, _ = aggregate_by_patient(all_oof_lps, all_oof_y, oof_pids_list)
    pat_lps -= np.max(pat_lps, axis=-1, keepdims=True)
    pat_probs = np.exp(pat_lps) / np.sum(np.exp(pat_lps), axis=-1, keepdims=True)
    patient_report = compute_detailed_metrics(pat_probs, pat_y)
    print_report_block(
        f"OUT-OF-FOLD TEST SET ({n_folds} FOLDS, PATIENT-LEVEL AGGREGATED)", patient_report
    )

    nodemo_report = compute_detailed_metrics(all_oof_nodemo_probs, all_oof_y)
    print_report_block(
        f"ABLATION CHECK: OUT-OF-FOLD TEST WITH ZEROED DEMOGRAPHICS", nodemo_report
    )

    full_report = {
        "n_folds_evaluated": n_folds,
        "fold_summaries": fold_summaries,
        "oof_per_cycle": cycle_report,
        "oof_per_patient": patient_report,
        "oof_zero_demographics_ablation": nodemo_report,
    }
    report_path = os.path.join(checkpoint_dir, "fp32_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Done] Saved best weights to {best_global_ckpt} and report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/icbhi_dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_fp32")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--demo_dropout", type=float, default=0.50)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    train_fp32_baseline(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        demo_dropout=args.demo_dropout,
        n_folds=args.n_folds,
    )