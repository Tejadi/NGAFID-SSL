#!/usr/bin/env python3
"""
Unified evaluation script for anomaly detection benchmark.
Supports BERT, LSTM, and MLP models with identical evaluation protocol.

This is a zero-shot transfer task - models pretrained on masked regression
are evaluated on anomaly detection using reconstruction error.
Higher reconstruction error indicates anomalous behavior.

Usage:
    python eval_anomaly_detection.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_anomaly_detection.py --model_type lstm --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_anomaly_detection.py --model_type mlp --checkpoint path/to/model.pt --data_dir path/to/test_data
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from models.bert_masked_regressor import BertMaskedRegressor
from models.lstm_baseline import LSTMBaseline
from models.mlp_baseline import MLPBaseline


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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


def compute_normalization_parameters(data_dir: str, max_files: int = 100, aircraft_types: list = None):
    """
    Compute global normalization parameters from training data.

    Args:
        data_dir: Directory containing CSV flight data files (train split)
        max_files: Maximum number of files to use for computing statistics
        aircraft_types: List of aircraft type prefixes to filter by

    Returns:
        Dictionary containing 'mean' and 'std' arrays
    """
    print("Computing normalization parameters from training data...")

    data_path = Path(data_dir)
    train_files = list(data_path.glob("*.csv"))
    train_files = [f for f in train_files if not any(name in f.name.lower()
                  for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]

    # Filter by aircraft type if specified
    if aircraft_types is not None:
        before = len(train_files)
        train_files = [f for f in train_files if any(f.name.startswith(at) for at in aircraft_types)]
        print(f"  Aircraft filter {aircraft_types}: {before} -> {len(train_files)} files")

    train_files = train_files[:max_files]

    if not train_files:
        raise ValueError(f"No training CSV files found in {data_dir}")

    print(f"  Using {len(train_files)} files to compute normalization parameters")

    all_data = []
    for csv_file in tqdm(train_files, desc="Loading training data"):
        try:
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]
            df_clean = df_numeric.ffill().bfill()
            flight_data = df_clean.to_numpy(dtype=np.float32)

            if flight_data.shape[0] > 0 and flight_data.shape[1] > 0:
                all_data.append(flight_data)
        except Exception as e:
            print(f"  Warning: Error processing {csv_file.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data found for computing normalization parameters")

    concatenated_data = np.vstack(all_data)
    print(f"  Total data shape: {concatenated_data.shape}")

    data_mean = np.mean(concatenated_data, axis=0)
    data_std = np.std(concatenated_data, axis=0)
    data_std[data_std == 0] = 1.0

    print(f"  Computed normalization parameters for {len(data_mean)} features")

    return {
        'mean': data_mean,
        'std': data_std
    }


def load_bert_model(checkpoint_path, feat_dim, device):
    """Load BERT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {}

    # Get model parameters with defaults
    hidden_size = config.get('hidden_size', 1024)
    encoder_layers = config.get('encoder_layers', 8)
    decoder_layers = config.get('decoder_layers', 6)
    num_heads = config.get('num_heads', 16)
    max_seq_len = config.get('seq_len', config.get('max_seq_len', 10000))
    dropout = config.get('dropout', 0.1)

    # Create model
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
    )

    # Handle torch.compile() prefix in state dict keys
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    model_config = {
        'hidden_size': hidden_size,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'num_heads': num_heads,
        'max_seq_len': max_seq_len,
    }

    return model, model_config


def load_lstm_model(checkpoint_path, feat_dim, device):
    """Load LSTM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {}

    # Get model parameters with defaults
    hidden_size = config.get('hidden_size', 256)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.1)
    bidirectional = config.get('bidirectional', True)

    # Create model
    model = LSTMBaseline(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    model_config = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'bidirectional': bidirectional,
    }

    return model, model_config


def load_mlp_model(checkpoint_path, feat_dim, device):
    """Load MLP model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {}

    # Get model parameters with defaults
    hidden_sizes = config.get('hidden_sizes', [256, 512, 256])
    dropout = config.get('dropout', 0.1)

    # Create model
    model = MLPBaseline(
        feat_dim=feat_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    model_config = {
        'hidden_sizes': hidden_sizes,
        'dropout': dropout,
    }

    return model, model_config


def load_model(model_type, checkpoint_path, feat_dim, device):
    """Load model based on type."""
    loaders = {
        'bert': load_bert_model,
        'lstm': load_lstm_model,
        'mlp': load_mlp_model,
    }
    return loaders[model_type](checkpoint_path, feat_dim, device)


def load_events(events_file: str) -> pd.DataFrame:
    """Load and parse events file."""
    print(f"Loading events from: {events_file}")

    events_df = pd.read_csv(events_file)

    # Clean up column names (remove quotes if present)
    events_df.columns = events_df.columns.str.strip('"')

    # Convert to appropriate types
    events_df['flight_id'] = events_df['flight_id'].astype(str).str.strip('"').astype(int)
    events_df['start_line'] = events_df['start_line'].astype(int)
    events_df['end_line'] = events_df['end_line'].astype(int)
    events_df['severity'] = pd.to_numeric(events_df['severity'].astype(str).str.strip(), errors='coerce')

    print(f"  Loaded {len(events_df)} events")
    print(f"  Event types: {events_df['name'].nunique()}")
    print(f"  Flights with events: {events_df['flight_id'].nunique()}")

    return events_df


def get_flight_files(data_dir: str, max_files: Optional[int] = None) -> List[Path]:
    """Get list of flight files from a directory."""
    data_path = Path(data_dir)

    flight_files = sorted(data_path.glob("*.csv"))
    # Filter out metadata files
    flight_files = [f for f in flight_files if not any(name in f.name.lower()
                   for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]

    if max_files:
        flight_files = flight_files[:max_files]

    print(f"  Found {len(flight_files)} flight files")
    return flight_files


def extract_flight_id(file_path: Path) -> Optional[int]:
    """Extract flight ID from filename."""
    name = file_path.stem
    parts = name.split('_')

    for i, part in enumerate(parts):
        if part == 'flight' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass

    # Try to find any number in the filename
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])

    return None


def load_and_normalize_flight(
    file_path: Path,
    normalization_params: Dict[str, np.ndarray],
    seq_len: int = 10000
) -> Tuple[np.ndarray, int]:
    """Load and normalize a flight file."""
    df = pd.read_csv(file_path, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    # Handle missing values
    df_clean = df_numeric.ffill().bfill()

    # Convert to numpy
    flight_data = df_clean.to_numpy(dtype=np.float32)

    original_length = len(flight_data)

    # Apply normalization
    if normalization_params is not None:
        mean = normalization_params['mean']
        std = normalization_params['std']

        if mean.shape[0] == flight_data.shape[1]:
            flight_data = (flight_data - mean) / std

    # Pad to seq_len if needed
    if len(flight_data) < seq_len:
        pad_length = seq_len - len(flight_data)
        last_row = flight_data[-1:]
        padding = np.repeat(last_row, pad_length, axis=0)
        flight_data = np.vstack([flight_data, padding])
    else:
        flight_data = flight_data[:seq_len]

    return flight_data.astype(np.float32), original_length


def compute_reconstruction_error(
    model,
    flight_data: np.ndarray,
    original_length: int,
    device: torch.device,
    mask_ratio: float = 0.15,
    num_samples: int = 5
) -> np.ndarray:
    """
    Compute per-timestep reconstruction error.

    Uses multiple random masks and averages the reconstruction error
    for more robust anomaly scores.
    """
    seq_len, feat_dim = flight_data.shape

    # Convert to tensor
    x_original = torch.tensor(flight_data, dtype=torch.float32).unsqueeze(0).to(device)

    all_errors = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Create random mask (1 = keep, 0 = mask)
            mask = (torch.rand(1, seq_len, feat_dim) > mask_ratio).float().to(device)

            # Create masked input
            x_masked = x_original * mask

            # Get reconstruction
            reconstruction = model(x_masked)

            # Compute per-position error (only on masked positions)
            error = (reconstruction - x_original) ** 2

            # Average across features for per-timestep error
            timestep_error = error.mean(dim=-1).squeeze(0).cpu().numpy()

            all_errors.append(timestep_error)

    # Average across samples
    avg_error = np.mean(all_errors, axis=0)

    # Only return error for original (non-padded) timesteps
    return avg_error[:original_length]


def compute_reconstruction_error_batched(
    model,
    flight_data_batch: List[np.ndarray],
    original_lengths: List[int],
    device: torch.device,
    mask_ratio: float = 0.15,
    num_samples: int = 5,
    use_amp: bool = True
) -> List[np.ndarray]:
    """
    Compute per-timestep reconstruction error for a batch of flights.

    Uses multiple random masks and averages the reconstruction error
    for more robust anomaly scores.

    Args:
        model: The model to use for reconstruction
        flight_data_batch: List of flight data arrays (already padded to same seq_len)
        original_lengths: List of original lengths for each flight
        device: torch device
        mask_ratio: Ratio of features to mask
        num_samples: Number of mask samples to average over
        use_amp: Whether to use automatic mixed precision

    Returns:
        List of reconstruction error arrays, one per flight (trimmed to original length)
    """
    batch_size = len(flight_data_batch)
    seq_len, feat_dim = flight_data_batch[0].shape

    # Stack into batch tensor and move to GPU
    x_original = torch.from_numpy(
        np.stack(flight_data_batch, axis=0)
    ).to(device, dtype=torch.float32, non_blocking=True)  # (batch_size, seq_len, feat_dim)

    # Accumulate errors on GPU to avoid CPU-GPU sync per sample
    accumulated_errors = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(num_samples):
            # Create random mask directly on GPU (1 = keep, 0 = mask)
            mask = (torch.rand(batch_size, seq_len, feat_dim, device=device) > mask_ratio).float()

            # Create masked input
            x_masked = x_original * mask

            # Get reconstruction with optional AMP
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    reconstruction = model(x_masked)
                    # Compute error in mixed precision
                    error = (reconstruction - x_original) ** 2
            else:
                reconstruction = model(x_masked)
                error = (reconstruction - x_original) ** 2

            # Average across features for per-timestep error and accumulate
            accumulated_errors += error.mean(dim=-1)

    # Average across samples (single CPU transfer at the end)
    avg_errors = (accumulated_errors / num_samples).cpu().numpy()

    # Trim each flight to its original length
    results = []
    for i, orig_len in enumerate(original_lengths):
        results.append(avg_errors[i, :orig_len])

    return results


def create_ground_truth_labels(
    original_length: int,
    flight_events: pd.DataFrame
) -> np.ndarray:
    """Create binary labels for each timestep (1 = anomaly, 0 = normal)."""
    labels = np.zeros(original_length, dtype=np.int32)

    for _, event in flight_events.iterrows():
        start = max(0, int(event['start_line']))
        end = min(original_length, int(event['end_line']) + 1)
        labels[start:end] = 1

    return labels


def compute_topk_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percents: List[float] = [1.0, 5.0, 10.0]
) -> Dict[str, float]:
    """
    Compute Top-k% recall for anomaly detection.

    For each k, take the top k% of timesteps by anomaly score and compute
    what fraction of all true anomaly timesteps are captured.

    Args:
        scores: Anomaly scores (higher = more anomalous), shape (N,)
        labels: Binary ground truth labels (1 = anomaly), shape (N,)
        k_percents: List of k values (as percentages of total timesteps)

    Returns:
        Dictionary mapping 'top_k{k}_recall' -> recall value
    """
    results = {}
    n_total = len(scores)
    n_anomalies = int(labels.sum())

    if n_anomalies == 0:
        for k in k_percents:
            results[f'top_k{int(k)}_recall'] = float('nan')
        return results

    # Sort indices by score descending (highest anomaly score first)
    sorted_indices = np.argsort(scores)[::-1]

    for k in k_percents:
        n_top = max(1, int(np.ceil(n_total * k / 100.0)))
        top_indices = sorted_indices[:n_top]
        n_captured = int(labels[top_indices].sum())
        recall_at_k = n_captured / n_anomalies
        results[f'top_k{int(k)}_recall'] = float(recall_at_k)

    return results


def evaluate_anomaly_detection(
    reconstruction_errors: List[np.ndarray],
    ground_truth_labels: List[np.ndarray],
    threshold_percentile: float = 95,
    topk_percents: List[float] = [1.0, 5.0, 10.0]
) -> Dict[str, float]:
    """Compute anomaly detection metrics including PR-AUC and Top-k recall."""

    # Concatenate all errors and labels
    all_errors = np.concatenate(reconstruction_errors)
    all_labels = np.concatenate(ground_truth_labels)

    # Compute metrics
    results = {}

    # ROC-AUC and PR-AUC (threshold-free ranking metrics)
    if len(np.unique(all_labels)) > 1:
        results['roc_auc'] = roc_auc_score(all_labels, all_errors)
        # PR-AUC: area under the precision-recall curve
        # average_precision_score computes the interpolated AUPRC
        results['pr_auc'] = average_precision_score(all_labels, all_errors)
        # Keep avg_precision as an alias for backwards compatibility
        results['avg_precision'] = results['pr_auc']
    else:
        print("Warning: Only one class in labels, cannot compute AUC")
        results['roc_auc'] = float('nan')
        results['pr_auc'] = float('nan')
        results['avg_precision'] = float('nan')

    # Top-k recall: fraction of anomalies captured in top-k% of ranked timesteps
    topk_results = compute_topk_recall(all_errors, all_labels, k_percents=topk_percents)
    results.update(topk_results)

    # Threshold-based metrics
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

    # Class distribution
    results['total_timesteps'] = len(all_labels)
    results['anomaly_timesteps'] = int(all_labels.sum())
    results['anomaly_ratio'] = float(all_labels.mean())
    results['predicted_anomalies'] = int(predictions.sum())

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models on anomaly detection benchmark (zero-shot transfer)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'lstm', 'mlp'],
                        help='Type of model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test flight CSV files')
    parser.add_argument('--events_file', type=str, required=True,
                        help='Path to events.csv with anomaly labels')

    # Normalization parameters
    parser.add_argument('--norm_params', type=str, default=None,
                        help='Path to normalization parameters (.npy file)')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory containing training data for computing normalization')

    # Aircraft filtering for normalization
    parser.add_argument('--aircraft_type', type=str, nargs='+', default=None,
                        choices=["Cessna_172S", "PA-28-181", "PA-44-180"],
                        help='Filter normalization data to specific aircraft type(s)')
    parser.add_argument('--aircraft_class', type=str, default=None,
                        choices=["single_engine", "multi_engine"],
                        help='Filter normalization data by aircraft class')

    # Anomaly detection parameters
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Masking ratio for reconstruction (lower = more context)')
    parser.add_argument('--num_mask_samples', type=int, default=5,
                        help='Number of masking samples per flight for robust estimation')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                        help='Percentile threshold for anomaly detection')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to evaluate')
    parser.add_argument('--topk_percents', type=float, nargs='+', default=[1.0, 5.0, 10.0],
                        help='Top-k%% values for recall computation (e.g. 1.0 5.0 10.0)')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation (higher = faster but more GPU memory)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision (AMP)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile() optimization')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./anomaly_detection_results',
                        help='Directory to save results')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this evaluation run')

    args = parser.parse_args()

    # Resolve aircraft_class to aircraft_type list
    AIRCRAFT_CLASS_MAP = {
        "single_engine": ["Cessna_172S", "PA-28-181"],
        "multi_engine": ["PA-44-180"],
    }
    if args.aircraft_class is not None:
        if args.aircraft_type is not None:
            print("Warning: --aircraft_class overrides --aircraft_type")
        args.aircraft_type = AIRCRAFT_CLASS_MAP[args.aircraft_class]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or compute normalization parameters
    if args.norm_params is not None:
        print(f"Loading normalization parameters from {args.norm_params}...")
        normalization_params = np.load(args.norm_params, allow_pickle=True).item()
    else:
        # Compute from training data
        if args.train_data_dir is not None:
            train_dir = args.train_data_dir
        else:
            # Default to sibling "train" folder
            data_path = Path(args.data_dir)
            train_dir = data_path.parent / "train"
            if not train_dir.exists():
                raise ValueError(f"Could not find training data at {train_dir}. "
                               "Please provide --norm_params or --train_data_dir")
        print(f"Computing normalization parameters from {train_dir}...")
        normalization_params = compute_normalization_parameters(str(train_dir), aircraft_types=args.aircraft_type)

    feat_dim = len(normalization_params['mean'])

    # Load model
    print(f"Loading {args.model_type.upper()} model from {args.checkpoint}...")
    model, model_config = load_model(args.model_type, args.checkpoint, feat_dim, device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Enable cudnn benchmark for consistent input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Compile model for faster inference (PyTorch 2.0+)
    if not args.no_compile and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    use_amp = not args.no_amp and torch.cuda.is_available()
    if use_amp:
        print(f"  Using automatic mixed precision (bfloat16)")

    # Warmup pass to initialize CUDA kernels and trigger compilation
    if torch.cuda.is_available():
        print("  Running warmup pass...")
        warmup_seq_len = model_config.get('max_seq_len', 10000)
        dummy_input = torch.randn(1, warmup_seq_len, feat_dim, device=device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
        torch.cuda.synchronize()
        del dummy_input

    # Load events
    events_df = load_events(args.events_file)
    events_by_flight = events_df.groupby('flight_id')

    # Get flight files
    print(f"Loading test data from {args.data_dir}...")
    flight_files = get_flight_files(args.data_dir, args.max_files)

    # Process flights
    all_reconstruction_errors = []
    all_ground_truth = []
    flights_with_events = 0
    flights_processed = 0

    seq_len = model_config.get('max_seq_len', 10000)

    print(f"\nEvaluating with mask_ratio={args.mask_ratio}, num_samples={args.num_mask_samples}, batch_size={args.batch_size}...")

    # Collect flight metadata for batched processing
    flight_metadata = []  # List of (file_path, flight_id)
    for file_path in flight_files:
        flight_id = extract_flight_id(file_path)
        if flight_id is not None:
            flight_metadata.append((file_path, flight_id))

    # Helper function for parallel data loading
    def load_flight_data(args_tuple):
        file_path, flight_id, norm_params, seq_length = args_tuple
        try:
            flight_data, original_length = load_and_normalize_flight(
                file_path, norm_params, seq_length
            )
            return (flight_id, flight_data, original_length, None)
        except Exception as e:
            return (flight_id, None, None, str(e))

    # Process in batches with parallel data loading
    num_batches = (len(flight_metadata) + args.batch_size - 1) // args.batch_size
    num_workers = min(8, os.cpu_count() or 4)  # Limit parallel workers

    for batch_idx in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(flight_metadata))
        batch_metadata = flight_metadata[batch_start:batch_end]

        # Load all flights in this batch in parallel
        batch_data = []
        batch_lengths = []
        batch_flight_ids = []

        load_args = [(fp, fid, normalization_params, seq_len) for fp, fid in batch_metadata]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_flight_data, load_args))

        for flight_id, flight_data, original_length, error in results:
            if error is not None:
                continue
            batch_data.append(flight_data)
            batch_lengths.append(original_length)
            batch_flight_ids.append(flight_id)

        if not batch_data:
            continue

        # Compute reconstruction errors for the batch
        batch_errors = compute_reconstruction_error_batched(
            model, batch_data, batch_lengths, device,
            mask_ratio=args.mask_ratio,
            num_samples=args.num_mask_samples,
            use_amp=use_amp
        )

        # Process results and get ground truth
        for recon_error, original_length, flight_id in zip(batch_errors, batch_lengths, batch_flight_ids):
            # Get ground truth labels
            if flight_id in events_by_flight.groups:
                flight_events = events_by_flight.get_group(flight_id)
                flights_with_events += 1
            else:
                flight_events = pd.DataFrame()

            labels = create_ground_truth_labels(original_length, flight_events)

            all_reconstruction_errors.append(recon_error)
            all_ground_truth.append(labels)
            flights_processed += 1

    print(f"\n  Processed {flights_processed} flights")
    print(f"  Flights with labeled events: {flights_with_events}")

    # Evaluate
    print("\nComputing metrics...")
    metrics = evaluate_anomaly_detection(
        all_reconstruction_errors,
        all_ground_truth,
        threshold_percentile=args.threshold_percentile,
        topk_percents=args.topk_percents
    )

    # Print results
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Num mask samples: {args.num_mask_samples}")
    print("-" * 60)
    print(f"ROC-AUC:           {metrics['roc_auc']:.6f}")
    print(f"PR-AUC (Avg Prec): {metrics['pr_auc']:.6f}")
    print("-" * 60)
    for k in args.topk_percents:
        key = f'top_k{int(k)}_recall'
        val = metrics.get(key, float('nan'))
        print(f"Top-{int(k)}% Recall:     {val:.6f}  (top {int(k)}% highest-scored timesteps capture this fraction of anomalies)")
    print("-" * 60)
    print(f"Precision:         {metrics['precision']:.6f}")
    print(f"Recall:            {metrics['recall']:.6f}")
    print(f"F1 Score:          {metrics['f1']:.6f}")
    print("-" * 60)
    print(f"Total timesteps:   {metrics['total_timesteps']:,}")
    print(f"Anomaly timesteps: {metrics['anomaly_timesteps']:,} ({metrics['anomaly_ratio']*100:.2f}%)")
    print(f"Threshold ({args.threshold_percentile}%ile): {metrics['threshold']:.6f}")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_type}_{timestamp}"

    # Determine norm params source for logging
    if args.norm_params is not None:
        norm_source = args.norm_params
    elif args.train_data_dir is not None:
        norm_source = f"computed from {args.train_data_dir}"
    else:
        norm_source = f"computed from {Path(args.data_dir).parent / 'train'}"

    results = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'events_file': args.events_file,
        'norm_params_source': norm_source,
        'timestamp': timestamp,
        'run_name': run_name,
        'config': {
            'mask_ratio': args.mask_ratio,
            'num_mask_samples': args.num_mask_samples,
            'threshold_percentile': args.threshold_percentile,
            'batch_size': args.batch_size,
            'feat_dim': feat_dim,
            'num_test_flights': flights_processed,
            'flights_with_events': flights_with_events,
            'num_parameters': num_params,
        },
        'model_config': model_config,
        'metrics': metrics,
    }

    output_file = os.path.join(args.output_dir, f"{run_name}.json")
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also evaluate per event type
    print("\n" + "=" * 60)
    print("RESULTS BY EVENT TYPE")
    print("=" * 60)

    event_types = events_df['name'].unique()
    per_event_results = {}

    for event_type in event_types:
        event_subset = events_df[events_df['name'] == event_type]
        event_flight_ids = set(event_subset['flight_id'].unique())

        # Filter to flights we processed that have this event type
        type_errors = []
        type_labels = []

        for i, file_path in enumerate(flight_files[:flights_processed]):
            flight_id = extract_flight_id(file_path)
            if flight_id in event_flight_ids:
                # Create labels just for this event type
                flight_events = event_subset[event_subset['flight_id'] == flight_id]
                original_length = len(all_reconstruction_errors[i])
                labels = create_ground_truth_labels(original_length, flight_events)

                type_errors.append(all_reconstruction_errors[i])
                type_labels.append(labels)

        if type_errors and sum(np.concatenate(type_labels)) > 0:
            type_results = evaluate_anomaly_detection(
                type_errors, type_labels,
                threshold_percentile=args.threshold_percentile,
                topk_percents=args.topk_percents
            )
            per_event_results[event_type] = type_results
            print(f"{event_type:40s} ROC-AUC: {type_results['roc_auc']:.4f}, "
                  f"PR-AUC: {type_results['pr_auc']:.4f}, "
                  f"Top1%R: {type_results.get('top_k1_recall', float('nan')):.4f}, "
                  f"Events: {type_results['anomaly_timesteps']:,}")

    # Save per-event results
    output_file_events = os.path.join(args.output_dir, f"{run_name}_by_event.json")
    with open(output_file_events, 'w') as f:
        json.dump(convert_to_serializable(per_event_results), f, indent=2)
    print(f"\nPer-event results saved to: {output_file_events}")


if __name__ == "__main__":
    main()
