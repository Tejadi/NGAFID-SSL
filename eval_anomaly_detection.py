#!/usr/bin/env python3
"""
Anomaly Detection Evaluation Script

Evaluates a pretrained BERT Masked Regressor on anomaly detection using
reconstruction error. Higher reconstruction error indicates anomalous behavior.

This is a zero-shot approach - no additional training required.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

try:
    from models.bert_masked_regressor import BertMaskedRegressor
    from train_full_flights import compute_normalization_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection using BERT reconstruction error"
    )

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/oscar/data/sbach/shared/ngafid",
                        help="Directory containing flight data")
    parser.add_argument("--events_file", type=str, default=None,
                        help="Path to events.csv (default: data_dir/events.csv)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to evaluate on")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to evaluate")

    # Evaluation arguments
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                        help="Masking ratio for reconstruction (lower = more context)")
    parser.add_argument("--num_mask_samples", type=int, default=5,
                        help="Number of masking samples per flight for robust estimation")
    parser.add_argument("--threshold_percentile", type=float, default=95,
                        help="Percentile threshold for anomaly detection")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./anomaly_detection_results",
                        help="Output directory for results")

    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for full flights)")

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[BertMaskedRegressor, dict]:
    """Load pretrained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'args' in checkpoint:
        config = checkpoint['args']
    else:
        # Try to infer from checkpoint keys
        config = {
            'hidden_size': 1024,
            'encoder_layers': 8,
            'decoder_layers': 6,
            'num_heads': 16,
            'seq_len': 10000,
            'dropout': 0.1,
        }
        print("Warning: No config found in checkpoint, using defaults")

    # Get feat_dim
    feat_dim = checkpoint.get('feat_dim', 44)

    # Create model
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=config.get('hidden_size', 1024),
        encoder_layers=config.get('encoder_layers', 8),
        decoder_layers=config.get('decoder_layers', 6),
        num_heads=config.get('num_heads', 16),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('seq_len', 10000),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {config.get('hidden_size', 1024)}d, "
          f"{config.get('encoder_layers', 8)} enc, {config.get('decoder_layers', 6)} dec")
    print(f"Feature dim: {feat_dim}")

    return model, config


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

    print(f"Loaded {len(events_df)} events")
    print(f"Event types: {events_df['name'].nunique()}")
    print(f"Flights with events: {events_df['flight_id'].nunique()}")

    return events_df


def get_flight_files(data_dir: str, split: str, max_files: Optional[int] = None) -> List[Path]:
    """Get list of flight files for a given split."""
    split_dir = Path(data_dir) / "preprocessed_data" / split

    if not split_dir.exists():
        # Try alternative naming
        if split == "val":
            split_dir = Path(data_dir) / "preprocessed_data" / "validation"

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    flight_files = sorted(split_dir.glob("*.csv"))

    if max_files:
        flight_files = flight_files[:max_files]

    print(f"Found {len(flight_files)} flight files in {split} split")
    return flight_files


def extract_flight_id(file_path: Path) -> Optional[int]:
    """Extract flight ID from filename."""
    # Expected format: AircraftType_flight_ID.csv
    name = file_path.stem
    parts = name.split('_')

    for i, part in enumerate(parts):
        if part == 'flight' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass

    # Try to find any number in the filename
    import re
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])

    return None


def load_and_normalize_flight(
    file_path: Path,
    normalization_params: Dict[str, np.ndarray],
    seq_len: int = 10000
) -> np.ndarray:
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
    model: BertMaskedRegressor,
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


def evaluate_anomaly_detection(
    reconstruction_errors: List[np.ndarray],
    ground_truth_labels: List[np.ndarray],
    threshold_percentile: float = 95
) -> Dict[str, float]:
    """Compute anomaly detection metrics."""

    # Concatenate all errors and labels
    all_errors = np.concatenate(reconstruction_errors)
    all_labels = np.concatenate(ground_truth_labels)

    # Compute metrics
    results = {}

    # ROC-AUC (threshold-free)
    if len(np.unique(all_labels)) > 1:
        results['roc_auc'] = roc_auc_score(all_labels, all_errors)
        results['avg_precision'] = average_precision_score(all_labels, all_errors)
    else:
        print("Warning: Only one class in labels, cannot compute AUC")
        results['roc_auc'] = float('nan')
        results['avg_precision'] = float('nan')

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
    args = parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, device)
    seq_len = config.get('seq_len', 10000)

    # Load events
    events_file = args.events_file or os.path.join(args.data_dir, "events.csv")
    events_df = load_events(events_file)

    # Create flight_id to events mapping
    events_by_flight = events_df.groupby('flight_id')

    # Compute normalization parameters
    print("Computing normalization parameters...")
    try:
        normalization_params = compute_normalization_parameters(
            args.data_dir, max_files=100
        )
    except Exception as e:
        print(f"Warning: Could not compute normalization: {e}")
        normalization_params = None

    # Get flight files
    flight_files = get_flight_files(args.data_dir, args.split, args.max_files)

    # Process flights
    all_reconstruction_errors = []
    all_ground_truth = []
    flights_with_events = 0
    flights_processed = 0

    print(f"\nProcessing {len(flight_files)} flights...")

    for file_path in tqdm(flight_files, desc="Evaluating flights"):
        flight_id = extract_flight_id(file_path)

        if flight_id is None:
            continue

        # Load and normalize flight
        try:
            flight_data, original_length = load_and_normalize_flight(
                file_path, normalization_params, seq_len
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # Compute reconstruction error
        recon_error = compute_reconstruction_error(
            model, flight_data, original_length, device,
            mask_ratio=args.mask_ratio,
            num_samples=args.num_mask_samples
        )

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

    print(f"\nProcessed {flights_processed} flights")
    print(f"Flights with labeled events: {flights_with_events}")

    # Evaluate
    print("\nComputing metrics...")
    results = evaluate_anomaly_detection(
        all_reconstruction_errors,
        all_ground_truth,
        threshold_percentile=args.threshold_percentile
    )

    # Print results
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 60)
    print(f"ROC-AUC:           {results['roc_auc']:.4f}")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"F1 Score:          {results['f1']:.4f}")
    print("-" * 60)
    print(f"Total timesteps:   {results['total_timesteps']:,}")
    print(f"Anomaly timesteps: {results['anomaly_timesteps']:,} ({results['anomaly_ratio']*100:.2f}%)")
    print(f"Threshold ({args.threshold_percentile}%ile): {results['threshold']:.4f}")
    print("=" * 60)

    # Save results
    results['checkpoint'] = args.checkpoint
    results['split'] = args.split
    results['mask_ratio'] = args.mask_ratio
    results['num_mask_samples'] = args.num_mask_samples
    results['flights_processed'] = flights_processed
    results['flights_with_events'] = flights_with_events

    output_file = os.path.join(args.output_dir, f"anomaly_results_{args.split}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
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
                type_errors, type_labels, args.threshold_percentile
            )
            per_event_results[event_type] = type_results
            print(f"{event_type:40s} AUC: {type_results['roc_auc']:.4f}, "
                  f"AP: {type_results['avg_precision']:.4f}, "
                  f"Events: {type_results['anomaly_timesteps']:,}")

    # Save per-event results
    output_file_events = os.path.join(args.output_dir, f"anomaly_results_by_event_{args.split}.json")
    with open(output_file_events, 'w') as f:
        json.dump(per_event_results, f, indent=2)
    print(f"\nPer-event results saved to: {output_file_events}")


if __name__ == "__main__":
    main()
