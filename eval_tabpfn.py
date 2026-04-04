#!/usr/bin/env python3
"""
TabPFN evaluation for flight classification tasks.

Evaluates TabPFN (a tabular foundation model) as a downstream classifier on
frozen encoder representations, comparing against logistic regression baselines.

Two modes:
  Option A (default): Use frozen embeddings from a pretrained encoder (BERT/LSTM/MLP)
  Option B (--raw_features): Use handcrafted summary statistics from raw flight data
"""

import argparse
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# Reuse infrastructure from eval_linear_probe
from eval_linear_probe import (
    AIRCRAFT_TYPES, AIRCRAFT_TYPE_TO_IDX,
    extract_flight_id, extract_aircraft_type,
    load_event_labels, get_split_files,
    compute_normalization_params, load_flight_data,
    load_model, extract_all_representations,
    FlightDataset, flight_collate_fn,
)

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="TabPFN evaluation on flight data")

    # Mode selection
    parser.add_argument("--raw_features", action="store_true",
                        help="Option B: use raw summary-stat features instead of encoder embeddings")

    # Encoder args (Option A)
    parser.add_argument("--model_type", type=str, default="bert",
                        choices=["bert", "lstm", "mlp", "simclr"],
                        help="Pretrained encoder type (Option A only)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to encoder checkpoint (Option A only)")

    # Data args
    parser.add_argument("--data_dir", type=str,
                        default="/oscar/data/sbach/shared/ngafid",
                        help="Root data directory")
    parser.add_argument("--events_file", type=str,
                        default="/oscar/data/sbach/bats/projects/ngafid/events.csv",
                        help="Path to events.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./tabpfn_results",
                        help="Output directory for results")

    # TabPFN args
    parser.add_argument("--n_estimators", type=int, default=4,
                        help="Number of TabPFN estimators (ensemble size)")
    parser.add_argument("--skip_per_event", action="store_true",
                        help="Skip per-event-type classification (saves API quota)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples for TabPFN (subsample if larger)")
    parser.add_argument("--pca_dim", type=int, default=100,
                        help="PCA dimensions for dimensionality reduction (0 to disable)")

    # General args
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true")

    return parser.parse_args()


# ---- Option B: Raw summary statistics ----

def compute_summary_stats(flight_data: np.ndarray) -> np.ndarray:
    """Compute per-channel summary statistics for a single flight.

    Args:
        flight_data: (seq_len, feat_dim) normalized flight array

    Returns:
        (feat_dim * 6,) vector of [mean, std, min, max, first, last] per channel
    """
    mean = np.mean(flight_data, axis=0)
    std = np.std(flight_data, axis=0)
    mn = np.min(flight_data, axis=0)
    mx = np.max(flight_data, axis=0)
    first = flight_data[0]
    last = flight_data[-1]
    return np.concatenate([mean, std, mn, mx, first, last])


def extract_raw_features(
    data_dir: str, split: str,
    event_flight_ids: Set[int],
    event_type_flight_ids: Dict[str, Set[int]],
    event_types: List[str],
    normalization_params: Dict[str, np.ndarray],
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Extract handcrafted summary-stat features for all flights in a split."""
    files = get_split_files(data_dir, split)
    if max_files is not None:
        files = files[:max_files]

    features_list = []
    binary_labels = []
    multilabel_rows = []
    aircraft_labels = []
    flight_ids = []

    for file_path in tqdm(files, desc=f"Extracting {split} raw features"):
        fid = extract_flight_id(file_path)
        aircraft_type = extract_aircraft_type(file_path)
        if fid is None or aircraft_type is None:
            continue

        try:
            flight_data = load_flight_data(file_path, "bert", normalization_params)
            stats = compute_summary_stats(flight_data)

            binary_label = 1 if fid in event_flight_ids else 0
            multi_label = [1 if fid in event_type_flight_ids[et] else 0 for et in event_types]
            aircraft_idx = AIRCRAFT_TYPE_TO_IDX[aircraft_type]

            features_list.append(stats)
            binary_labels.append(binary_label)
            multilabel_rows.append(multi_label)
            aircraft_labels.append(aircraft_idx)
            flight_ids.append(fid)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    X = np.stack(features_list)
    y_binary = np.array(binary_labels)
    y_multilabel = np.array(multilabel_rows)
    y_aircraft = np.array(aircraft_labels)
    return X, y_binary, y_multilabel, y_aircraft, flight_ids


# ---- Evaluation helpers ----

def evaluate_binary(clf, X_train, y_train, X_test, y_test, clf_name: str) -> Dict:
    """Train and evaluate a classifier on binary anomaly classification."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "classifier": clf_name,
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def evaluate_multiclass(clf, X_train, y_train, X_test, y_test, clf_name: str) -> Dict:
    """Train and evaluate a classifier on aircraft type classification."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    try:
        macro_roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
    except ValueError:
        macro_roc_auc = 0.0

    per_class = {}
    for i, aircraft in enumerate(AIRCRAFT_TYPES):
        y_true_bin = (y_test == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        per_class[aircraft] = {
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_bin, y_prob[:, i])) if y_true_bin.sum() > 0 else 0.0,
            "support": int(y_true_bin.sum()),
        }

    cm = confusion_matrix(y_test, y_pred)

    return {
        "classifier": clf_name,
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    mode = "Option B (raw features)" if args.raw_features else "Option A (frozen embeddings)"
    print("=" * 60)
    print(f"TabPFN Evaluation — {mode}")
    if not args.raw_features:
        print(f"Encoder: {args.model_type.upper()}, Checkpoint: {args.checkpoint}")
    print(f"PCA dim: {args.pca_dim if args.pca_dim > 0 else 'disabled'}")
    print("=" * 60)

    # Validate args
    if not args.raw_features and args.checkpoint is None:
        raise ValueError("--checkpoint required for Option A (encoder embeddings)")

    # Load event labels
    event_flight_ids, event_type_flight_ids, event_types = load_event_labels(args.events_file)

    # Compute normalization parameters (cached to disk)
    model_type_for_norm = args.model_type if not args.raw_features else "bert"
    norm_cache = os.path.join(args.data_dir, f"norm_params_{model_type_for_norm}.npz")
    if os.path.exists(norm_cache):
        print(f"Loading cached normalization parameters from {norm_cache}")
        cached = np.load(norm_cache)
        norm_params = {'mean': cached['mean'], 'std': cached['std']}
    else:
        print("Computing normalization parameters...")
        norm_params = compute_normalization_params(args.data_dir, model_type_for_norm)
        np.savez(norm_cache, mean=norm_params['mean'], std=norm_params['std'])
        print(f"Saved normalization parameters to {norm_cache}")

    # ---- Extract features ----
    if args.raw_features:
        # Option B: raw summary statistics
        print("\nExtracting raw summary-stat features...")
        X_train, y_train, y_train_multi, y_train_aircraft, train_ids = extract_raw_features(
            args.data_dir, "train", event_flight_ids, event_type_flight_ids,
            event_types, norm_params, args.max_files,
        )
        X_test, y_test, y_test_multi, y_test_aircraft, test_ids = extract_raw_features(
            args.data_dir, "test", event_flight_ids, event_type_flight_ids,
            event_types, norm_params, args.max_files,
        )
    else:
        # Option A: frozen encoder embeddings
        print(f"\nLoading {args.model_type} model...")
        model = load_model(args.model_type, args.checkpoint, device)
        if not args.no_compile and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')

        print("Extracting train representations...")
        X_train, y_train, y_train_multi, y_train_aircraft, train_ids = extract_all_representations(
            model, args.model_type, args.data_dir, "train",
            event_flight_ids, event_type_flight_ids, event_types,
            norm_params, device, args.max_files,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print("Extracting test representations...")
        X_test, y_test, y_test_multi, y_test_aircraft, test_ids = extract_all_representations(
            model, args.model_type, args.data_dir, "test",
            event_flight_ids, event_type_flight_ids, event_types,
            norm_params, device, args.max_files,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTrain: {X_train.shape[0]} flights, repr dim: {X_train.shape[1]}")
    print(f"  Anomaly positive rate: {y_train.mean():.1%}")
    for i, at in enumerate(AIRCRAFT_TYPES):
        print(f"  {at}: {(y_train_aircraft == i).sum()}")
    print(f"Test: {X_test.shape[0]} flights, repr dim: {X_test.shape[1]}")
    print(f"  Anomaly positive rate: {y_test.mean():.1%}")
    for i, at in enumerate(AIRCRAFT_TYPES):
        print(f"  {at}: {(y_test_aircraft == i).sum()}")

    # ---- Preprocessing: scale + optional PCA ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if args.pca_dim > 0 and X_train_scaled.shape[1] > args.pca_dim:
        print(f"\nApplying PCA: {X_train_scaled.shape[1]} -> {args.pca_dim} dims")
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained:.1%}")

    # Subsample training data if requested
    if args.max_train_samples and X_train_scaled.shape[0] > args.max_train_samples:
        print(f"\nSubsampling train: {X_train_scaled.shape[0]} -> {args.max_train_samples}")
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X_train_scaled.shape[0], args.max_train_samples, replace=False)
        X_train_sub = X_train_scaled[idx]
        y_train_sub = y_train[idx]
        y_train_aircraft_sub = y_train_aircraft[idx]
    else:
        X_train_sub = X_train_scaled
        y_train_sub = y_train
        y_train_aircraft_sub = y_train_aircraft

    # ---- Import TabPFN (local, open-source v1) ----
    print("\nLoading TabPFN (local)...")
    from tabpfn import TabPFNClassifier

    # ---- Binary Anomaly Classification ----
    print("\n" + "=" * 60)
    print("BINARY ANOMALY CLASSIFICATION")
    print("=" * 60)

    tabpfn_clf = TabPFNClassifier(device=str(device))
    logreg_clf = LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=args.seed,
        class_weight='balanced', C=1.0,
    )

    print("\nTabPFN...")
    tabpfn_binary = evaluate_binary(
        tabpfn_clf, X_train_sub, y_train_sub,
        X_test_scaled, y_test, "TabPFN"
    )
    print("\nLogistic Regression (baseline)...")
    logreg_binary = evaluate_binary(
        logreg_clf, X_train_sub, y_train_sub,
        X_test_scaled, y_test, "LogisticRegression"
    )

    print(f"\n{'Metric':<12} {'TabPFN':>10} {'LogReg':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10}")
    for metric in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
        print(f"{metric:<12} {tabpfn_binary[metric]:>10.4f} {logreg_binary[metric]:>10.4f}")

    # ---- Aircraft Type Classification ----
    print("\n" + "=" * 60)
    print("AIRCRAFT TYPE CLASSIFICATION (3-way)")
    print("=" * 60)

    tabpfn_clf2 = TabPFNClassifier(device=str(device))
    logreg_clf2 = LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=args.seed,
        class_weight='balanced', C=1.0,
    )

    print("\nTabPFN...")
    tabpfn_aircraft = evaluate_multiclass(
        tabpfn_clf2, X_train_sub, y_train_aircraft_sub,
        X_test_scaled, y_test_aircraft, "TabPFN"
    )
    print("\nLogistic Regression (baseline)...")
    logreg_aircraft = evaluate_multiclass(
        logreg_clf2, X_train_sub, y_train_aircraft_sub,
        X_test_scaled, y_test_aircraft, "LogisticRegression"
    )

    print(f"\n{'Metric':<16} {'TabPFN':>10} {'LogReg':>10}")
    print(f"{'-'*16} {'-'*10} {'-'*10}")
    for metric in ["accuracy", "macro_f1", "macro_roc_auc"]:
        print(f"{metric:<16} {tabpfn_aircraft[metric]:>10.4f} {logreg_aircraft[metric]:>10.4f}")

    print(f"\nPer-class accuracy (TabPFN):")
    for aircraft in AIRCRAFT_TYPES:
        m = tabpfn_aircraft['per_class'][aircraft]
        print(f"  {aircraft:<15} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  n={m['support']}")

    # ---- Save intermediate results (binary + aircraft) ----
    os.makedirs(args.output_dir, exist_ok=True)
    mode_tag = "raw" if args.raw_features else args.model_type
    results = {
        "mode": mode,
        "model_type": args.model_type if not args.raw_features else "raw_summary_stats",
        "checkpoint": args.checkpoint,
        "pca_dim": args.pca_dim if pca is not None else None,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()) if pca is not None else None,
        "train_flights": int(X_train.shape[0]),
        "train_flights_used": int(X_train_sub.shape[0]),
        "test_flights": int(X_test.shape[0]),
        "representation_dim_original": int(X_train.shape[1]),
        "representation_dim_used": int(X_train_scaled.shape[1]),
        "anomaly_classification": {
            "tabpfn": tabpfn_binary,
            "logistic_regression": logreg_binary,
        },
        "aircraft_classification": {
            "tabpfn": tabpfn_aircraft,
            "logistic_regression": logreg_aircraft,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output_file = os.path.join(args.output_dir, f"tabpfn_{mode_tag}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nIntermediate results saved to: {output_file}")

    # ---- Per-event-type classification with TabPFN ----
    if args.skip_per_event:
        print("\nSkipping per-event-type classification (--skip_per_event set).")
        print(f"\nFinal results saved to: {output_file}")
        return

    print("\n" + "=" * 60)
    print("PER-EVENT-TYPE ANOMALY CLASSIFICATION")
    print("=" * 60)

    min_positive = 5
    tabpfn_per_event = {}
    logreg_per_event = {}

    for i, et in enumerate(event_types):
        y_tr = y_train_multi[:, i]
        y_te = y_test_multi[:, i]

        # Subsample indices for training
        if args.max_train_samples and len(y_tr) > args.max_train_samples:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(y_tr), args.max_train_samples, replace=False)
            y_tr_sub = y_tr[idx]
        else:
            idx = None
            y_tr_sub = y_tr

        if y_tr_sub.sum() < min_positive or y_te.sum() < min_positive:
            tabpfn_per_event[et] = {"skipped": True, "reason": f"too few positives (train={int(y_tr_sub.sum())}, test={int(y_te.sum())})"}
            logreg_per_event[et] = {"skipped": True}
            continue

        X_tr_ev = X_train_scaled[idx] if idx is not None else X_train_scaled

        # TabPFN
        tpfn = TabPFNClassifier(device=str(device))
        tpfn.fit(X_tr_ev, y_tr_sub)
        tp_pred = tpfn.predict(X_test_scaled)
        tp_prob = tpfn.predict_proba(X_test_scaled)[:, 1]
        tabpfn_per_event[et] = {
            "skipped": False,
            "roc_auc": float(roc_auc_score(y_te, tp_prob)),
            "f1": float(f1_score(y_te, tp_pred, zero_division=0)),
            "train_positives": int(y_tr_sub.sum()),
            "test_positives": int(y_te.sum()),
        }

        # LogReg
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=args.seed, class_weight='balanced', C=1.0)
        lr.fit(X_tr_ev, y_tr_sub)
        lr_pred = lr.predict(X_test_scaled)
        lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
        logreg_per_event[et] = {
            "skipped": False,
            "roc_auc": float(roc_auc_score(y_te, lr_prob)),
            "f1": float(f1_score(y_te, lr_pred, zero_division=0)),
        }

    # Print per-event results
    print(f"\n{'Event Type':<35} {'TabPFN AUC':>10} {'LogReg AUC':>10} {'Train+':>7} {'Test+':>7}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*7} {'-'*7}")
    for et in event_types:
        tr = tabpfn_per_event[et]
        lr = logreg_per_event[et]
        if tr.get("skipped"):
            print(f"{et:<35} {'SKIP':>10} {'SKIP':>10}")
        else:
            print(f"{et:<35} {tr['roc_auc']:>10.4f} {lr['roc_auc']:>10.4f} {tr['train_positives']:>7} {tr['test_positives']:>7}")

    tabpfn_aucs = [v["roc_auc"] for v in tabpfn_per_event.values() if not v.get("skipped")]
    logreg_aucs = [v["roc_auc"] for v in logreg_per_event.values() if not v.get("skipped")]
    print(f"\nMacro AUC — TabPFN: {np.mean(tabpfn_aucs):.4f}, LogReg: {np.mean(logreg_aucs):.4f}")

    # ---- Update saved results with per-event results ----
    results["per_event_classification"] = {
        "tabpfn": tabpfn_per_event,
        "logistic_regression": logreg_per_event,
    }
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to: {output_file}")


if __name__ == "__main__":
    main()
