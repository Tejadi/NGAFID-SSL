#!/usr/bin/env python3
"""
Anomaly detection evaluation for PatchTST masked regressor.

PatchTST operates on fixed-length windows, so this script slides a window
across each full flight and aggregates per-timestep reconstruction error.
Higher error = more anomalous.

Metrics: ROC-AUC, PR-AUC, Top-k% Recall, Precision/Recall/F1 at threshold.

Usage:
    python eval_anomaly_detection_patchtst.py \
        --checkpoint patchtst_results/.../best_model.pt \
        --data_dir ./NGAFID-LOCI-GATS-Data/preprocessed_data/test \
        --events_file ./NGAFID-LOCI-GATS-Data/preprocessed_data/test/events.csv \
        --train_data_dir ./NGAFID-LOCI-GATS-Data/preprocessed_data/train
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from models.patchtst_masked_regressor import PatchTSTMaskedRegressor
from train_full_flights import compute_normalization_parameters


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint['args']

    model = PatchTSTMaskedRegressor(
        feat_dim=saved_args['feat_dim'],
        seq_len=saved_args['seq_len'],
        patch_len=saved_args['patch_len'],
        stride=saved_args['stride'],
        d_model=saved_args['d_model'],
        n_heads=saved_args['n_heads'],
        d_ff=saved_args['d_ff'],
        encoder_layers=saved_args['encoder_layers'],
        decoder_layers=saved_args['decoder_layers'],
        dropout=0.0,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from step {checkpoint['global_step']} "
          f"(eval_loss: {checkpoint.get('eval_loss', 'N/A')})")
    print(f"  seq_len={saved_args['seq_len']}, patch_len={saved_args['patch_len']}, "
          f"d_model={saved_args['d_model']}, encoder_layers={saved_args['encoder_layers']}")
    return model, saved_args


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_flight_files(data_dir: str, max_files: Optional[int] = None) -> List[Path]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    files = [f for f in files if not any(name in f.name.lower()
             for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]
    if max_files:
        files = files[:max_files]
    print(f"  Found {len(files)} flight files")
    return files


def extract_flight_id(file_path: Path) -> Optional[int]:
    name = file_path.stem
    parts = name.split('_')
    for i, part in enumerate(parts):
        if part == 'flight' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])
    return None


def load_and_normalize_flight(
    file_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Load a single flight CSV, forward-fill NaNs, and normalize."""
    df = pd.read_csv(file_path, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    arr = df[numeric_cols].ffill().bfill().to_numpy(dtype=np.float32)
    original_length = len(arr)

    if mean.shape[0] == arr.shape[1]:
        arr = (arr - mean) / std

    return arr.astype(np.float32), original_length


def load_events(events_file: str) -> pd.DataFrame:
    print(f"Loading events from: {events_file}")
    events_df = pd.read_csv(events_file)
    events_df.columns = events_df.columns.str.strip('"')
    events_df['flight_id'] = events_df['flight_id'].astype(str).str.strip('"').astype(int)
    events_df['start_line'] = events_df['start_line'].astype(int)
    events_df['end_line'] = events_df['end_line'].astype(int)
    events_df['severity'] = pd.to_numeric(events_df['severity'].astype(str).str.strip(), errors='coerce')
    print(f"  Loaded {len(events_df)} events across {events_df['flight_id'].nunique()} flights")
    return events_df


def create_ground_truth_labels(original_length: int, flight_events: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(original_length, dtype=np.int32)
    for _, event in flight_events.iterrows():
        start = max(0, int(event['start_line']))
        end = min(original_length, int(event['end_line']) + 1)
        labels[start:end] = 1
    return labels


# ---------------------------------------------------------------------------
# Windowed reconstruction error
# ---------------------------------------------------------------------------

def compute_reconstruction_error_windowed(
    model: torch.nn.Module,
    flight_data: np.ndarray,
    original_length: int,
    seq_len: int,
    device: torch.device,
    mask_ratio: float = 0.15,
    num_samples: int = 5,
    window_stride: Optional[int] = None,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Slide a window of `seq_len` across `flight_data` and compute per-timestep
    reconstruction error. Overlapping windows are averaged.

    Args:
        flight_data: Normalized flight array (T, feat_dim)
        original_length: Actual flight length (pre-padding)
        seq_len: Window size the model was trained on
        window_stride: Step size between windows. Defaults to seq_len // 2
                       for 50% overlap, giving smoother per-timestep scores.
        mask_ratio: Fraction of features to randomly mask per inference.
        num_samples: Number of random masks to average for robustness.
    """
    T, feat_dim = flight_data.shape
    if window_stride is None:
        window_stride = seq_len // 2

    # Accumulate error and count for averaging overlapping windows
    error_sum = np.zeros(T, dtype=np.float64)
    error_count = np.zeros(T, dtype=np.float64)

    # Generate all window start positions
    starts = list(range(0, max(1, T - seq_len + 1), window_stride))
    # Always include a window at the very end if not already covered
    if T >= seq_len and (T - seq_len) not in starts:
        starts.append(T - seq_len)

    x_windows = []
    window_slices = []
    for s in starts:
        e = s + seq_len
        if e > T:
            # Pad short tail with last row
            window = flight_data[s:].copy()
            pad_len = seq_len - len(window)
            window = np.vstack([window, np.repeat(window[-1:], pad_len, axis=0)])
        else:
            window = flight_data[s:e].copy()
        x_windows.append(window)
        window_slices.append((s, min(s + seq_len, T)))

    x_tensor = torch.from_numpy(
        np.stack(x_windows, axis=0)
    ).to(device, dtype=torch.float32, non_blocking=True)  # (W, seq_len, feat_dim)

    W = x_tensor.shape[0]
    accumulated = torch.zeros(W, seq_len, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(num_samples):
            mask = (torch.rand(W, seq_len, feat_dim, device=device) > mask_ratio).float()
            x_masked = x_tensor * mask

            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    recon = model(x_masked)
                    err = (recon - x_tensor) ** 2
            else:
                recon = model(x_masked)
                err = (recon - x_tensor) ** 2

            accumulated += err.mean(dim=-1)  # (W, seq_len)

    avg_errors = (accumulated / num_samples).cpu().numpy()  # (W, seq_len)

    # Scatter back into full timeline
    for i, (s, e) in enumerate(window_slices):
        valid_len = e - s
        error_sum[s:e] += avg_errors[i, :valid_len]
        error_count[s:e] += 1.0

    # Where count > 0, average; otherwise leave at 0 (shouldn't happen)
    mask_nonzero = error_count > 0
    error_sum[mask_nonzero] /= error_count[mask_nonzero]

    return error_sum[:original_length].astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_topk_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percents: List[float] = [1.0, 5.0, 10.0],
) -> Dict[str, float]:
    """
    Top-k% recall: fraction of anomaly timesteps captured in the top-k%
    highest-scored timesteps.
    """
    results = {}
    n_total = len(scores)
    n_anomalies = int(labels.sum())

    if n_anomalies == 0:
        for k in k_percents:
            results[f'top_k{int(k)}_recall'] = float('nan')
        return results

    sorted_indices = np.argsort(scores)[::-1]
    for k in k_percents:
        n_top = max(1, int(np.ceil(n_total * k / 100.0)))
        n_captured = int(labels[sorted_indices[:n_top]].sum())
        results[f'top_k{int(k)}_recall'] = float(n_captured / n_anomalies)

    return results


def evaluate_anomaly_detection(
    reconstruction_errors: List[np.ndarray],
    ground_truth_labels: List[np.ndarray],
    threshold_percentile: float = 95,
    topk_percents: List[float] = [1.0, 5.0, 10.0],
) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Top-k recall, and threshold-based metrics."""
    all_errors = np.concatenate(reconstruction_errors)
    all_labels = np.concatenate(ground_truth_labels)

    results = {}

    if len(np.unique(all_labels)) > 1:
        results['roc_auc'] = roc_auc_score(all_labels, all_errors)
        results['pr_auc'] = average_precision_score(all_labels, all_errors)
        results['avg_precision'] = results['pr_auc']
    else:
        print("Warning: Only one class in labels, cannot compute AUC")
        results['roc_auc'] = float('nan')
        results['pr_auc'] = float('nan')
        results['avg_precision'] = float('nan')

    results.update(compute_topk_recall(all_errors, all_labels, k_percents=topk_percents))

    threshold = np.percentile(all_errors, threshold_percentile)
    predictions = (all_errors > threshold).astype(int)
    results['threshold'] = float(threshold)
    results['threshold_percentile'] = threshold_percentile

    if len(np.unique(all_labels)) > 1:
        results['precision'] = precision_score(all_labels, predictions, zero_division=0)
        results['recall'] = recall_score(all_labels, predictions, zero_division=0)
        results['f1'] = f1_score(all_labels, predictions, zero_division=0)
    else:
        results['precision'] = float('nan')
        results['recall'] = float('nan')
        results['f1'] = float('nan')

    results['total_timesteps'] = len(all_labels)
    results['anomaly_timesteps'] = int(all_labels.sum())
    results['anomaly_ratio'] = float(all_labels.mean())
    results['predicted_anomalies'] = int(predictions.sum())

    return results


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Anomaly detection evaluation for PatchTST (windowed reconstruction)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PatchTST checkpoint (.pt)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test flight CSV files')
    parser.add_argument('--events_file', type=str, required=True,
                        help='Path to events.csv with anomaly labels')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory to compute normalization stats from (defaults to ../train)')
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Fraction of features to randomly mask per inference')
    parser.add_argument('--num_mask_samples', type=int, default=5,
                        help='Number of random masks to average per window')
    parser.add_argument('--window_stride', type=int, default=None,
                        help='Stride between windows (default: seq_len // 2 for 50%% overlap)')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                        help='Percentile threshold for binary anomaly prediction')
    parser.add_argument('--topk_percents', type=float, nargs='+', default=[1.0, 5.0, 10.0],
                        help='Top-k%% values for recall computation')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of test files to evaluate')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--output_dir', type=str, default='./anomaly_detection_results',
                        help='Directory to save results JSON')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (used in output filename)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalization
    if args.train_data_dir is not None:
        train_dir = args.train_data_dir
    else:
        train_dir = str(Path(args.data_dir).parent / 'train')
        if not Path(train_dir).exists():
            raise ValueError(
                f"Could not find training data at {train_dir}. "
                "Please provide --train_data_dir"
            )
    print(f"Computing normalization parameters from {train_dir}...")
    norm_params = compute_normalization_parameters(train_dir, max_files=500)
    mean = norm_params['mean']
    std  = norm_params['std']

    # Model
    print(f"Loading PatchTST model from {args.checkpoint}...")
    model, saved_args = load_model(args.checkpoint, device)
    seq_len = saved_args['seq_len']
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    use_amp = not args.no_amp and device.type == 'cuda'
    if use_amp:
        print("  Using automatic mixed precision (bfloat16)")

    # Warmup
    if device.type == 'cuda':
        print("  Running warmup pass...")
        dummy = torch.randn(1, seq_len, saved_args['feat_dim'], device=device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        torch.cuda.synchronize()
        del dummy

    # Events
    events_df = load_events(args.events_file)
    events_by_flight = events_df.groupby('flight_id')

    # Flight files
    print(f"Loading test flights from {args.data_dir}...")
    flight_files = get_flight_files(args.data_dir, args.max_files)

    window_stride = args.window_stride or (seq_len // 2)
    print(f"\nEvaluating: seq_len={seq_len}, window_stride={window_stride}, "
          f"mask_ratio={args.mask_ratio}, num_samples={args.num_mask_samples}")

    all_errors = []
    all_labels = []
    flights_with_events = 0
    flights_processed = 0

    def _load(fp):
        try:
            data, orig_len = load_and_normalize_flight(fp, mean, std)
            return data, orig_len, None
        except Exception as e:
            return None, None, str(e)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futures = {pool.submit(_load, fp): fp for fp in flight_files}

        for future in tqdm(futures, desc="Processing flights", total=len(flight_files)):
            fp = futures[future]
            flight_id = extract_flight_id(fp)
            if flight_id is None:
                continue

            flight_data, original_length, err = future.result()
            if err is not None:
                print(f"  Warning: {fp.name}: {err}")
                continue

            recon_error = compute_reconstruction_error_windowed(
                model, flight_data, original_length,
                seq_len=seq_len,
                device=device,
                mask_ratio=args.mask_ratio,
                num_samples=args.num_mask_samples,
                window_stride=window_stride,
                use_amp=use_amp,
            )

            if flight_id in events_by_flight.groups:
                flight_events = events_by_flight.get_group(flight_id)
                flights_with_events += 1
            else:
                flight_events = pd.DataFrame()

            labels = create_ground_truth_labels(original_length, flight_events)

            all_errors.append(recon_error)
            all_labels.append(labels)
            flights_processed += 1

    print(f"\n  Processed {flights_processed} flights ({flights_with_events} with labeled events)")

    # Metrics
    print("\nComputing metrics...")
    metrics = evaluate_anomaly_detection(
        all_errors, all_labels,
        threshold_percentile=args.threshold_percentile,
        topk_percents=args.topk_percents,
    )

    # Print
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULTS  (PatchTST)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"seq_len={seq_len}, window_stride={window_stride}, "
          f"mask_ratio={args.mask_ratio}, num_samples={args.num_mask_samples}")
    print("-" * 60)
    print(f"ROC-AUC:           {metrics['roc_auc']:.6f}")
    print(f"PR-AUC (Avg Prec): {metrics['pr_auc']:.6f}")
    print("-" * 60)
    for k in args.topk_percents:
        key = f'top_k{int(k)}_recall'
        print(f"Top-{int(k):2d}% Recall:    {metrics.get(key, float('nan')):.6f}")
    print("-" * 60)
    print(f"Precision:         {metrics['precision']:.6f}")
    print(f"Recall:            {metrics['recall']:.6f}")
    print(f"F1 Score:          {metrics['f1']:.6f}")
    print("-" * 60)
    print(f"Total timesteps:   {metrics['total_timesteps']:,}")
    print(f"Anomaly timesteps: {metrics['anomaly_timesteps']:,} ({metrics['anomaly_ratio']*100:.4f}%)")
    print(f"Threshold ({args.threshold_percentile}%ile): {metrics['threshold']:.6f}")
    print("=" * 60)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"patchtst_anomaly_{timestamp}"
    output = {
        'model_type': 'patchtst',
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'events_file': args.events_file,
        'timestamp': timestamp,
        'run_name': run_name,
        'config': {
            'seq_len': seq_len,
            'window_stride': window_stride,
            'mask_ratio': args.mask_ratio,
            'num_mask_samples': args.num_mask_samples,
            'threshold_percentile': args.threshold_percentile,
            'num_test_flights': flights_processed,
            'flights_with_events': flights_with_events,
            'num_parameters': num_params,
        },
        'model_config': saved_args,
        'metrics': metrics,
    }
    out_path = os.path.join(args.output_dir, f"{run_name}.json")
    with open(out_path, 'w') as f:
        json.dump(convert_to_serializable(output), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Per-event breakdown
    print("\n" + "=" * 60)
    print("RESULTS BY EVENT TYPE")
    print("=" * 60)
    per_event = {}
    for event_type in events_df['name'].unique():
        subset = events_df[events_df['name'] == event_type]
        event_fids = set(subset['flight_id'].unique())

        type_errors, type_labels = [], []
        for i, fp in enumerate(flight_files[:flights_processed]):
            fid = extract_flight_id(fp)
            if fid in event_fids:
                fe = subset[subset['flight_id'] == fid]
                orig_len = len(all_errors[i])
                lbl = create_ground_truth_labels(orig_len, fe)
                type_errors.append(all_errors[i])
                type_labels.append(lbl)

        if type_errors and sum(np.concatenate(type_labels)) > 0:
            r = evaluate_anomaly_detection(
                type_errors, type_labels,
                threshold_percentile=args.threshold_percentile,
                topk_percents=args.topk_percents,
            )
            per_event[event_type] = r
            print(f"{event_type:40s} ROC-AUC: {r['roc_auc']:.4f}, "
                  f"PR-AUC: {r['pr_auc']:.4f}, "
                  f"Top1%R: {r.get('top_k1_recall', float('nan')):.4f}, "
                  f"Events: {r['anomaly_timesteps']:,}")

    out_events = os.path.join(args.output_dir, f"{run_name}_by_event.json")
    with open(out_events, 'w') as f:
        json.dump(convert_to_serializable(per_event), f, indent=2)
    print(f"\nPer-event results saved to: {out_events}")


if __name__ == '__main__':
    main()
