#!/usr/bin/env python3
"""
Unified evaluation script for masked regression benchmark.
Supports BERT, LSTM, and MLP models with identical evaluation protocol.

Includes optional physics-based evaluation using MPC trajectory optimization.
For physics evaluation, uses contiguous middle masking instead of geometric.

Usage:
    python eval_masked_regression.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_masked_regression.py --model_type lstm --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_masked_regression.py --model_type mlp --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_masked_regression.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data --physics_eval
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ngafid_datasets.transformation_dataset import mask_transform
from models.bert_masked_regressor import BertMaskedRegressor
from models.lstm_baseline import LSTMBaseline
from models.mlp_baseline import MLPBaseline
from models.patchtst_masked_regressor import PatchTSTMaskedRegressor

# Physics evaluation imports
from physics_loss import (
    physics_trajectory_optimization,
    AircraftDynamics,
    AIRCRAFT_PRESETS,
)
from physics_loss.feature_mapping import PhysicsFeatureMap, PRESET_FEATURE_MAPS


def load_flight_data(flight_dir, aircraft_types=None):
    """Load flight data from CSV files in a directory.

    Args:
        flight_dir: Directory containing CSV flight files
        aircraft_types: Optional list of aircraft type prefixes to filter by
                       (e.g., ["PA-44-180"] for multi-engine only)

    Returns:
        Tuple of (flights_array, flight_ids)
    """
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")

    # Filter out metadata files
    csv_files = [f for f in csv_files if not any(name in f.name.lower()
                 for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])]

    # Filter by aircraft type if specified
    if aircraft_types is not None:
        before = len(csv_files)
        csv_files = [f for f in csv_files if any(f.name.startswith(at) for at in aircraft_types)]
        print(f"  Aircraft filter {aircraft_types}: {before} -> {len(csv_files)} files")

    flights = []
    flight_ids = []
    for path in tqdm(csv_files, desc='Loading flight data'):
        filename = path.name
        try:
            flight_id = int(filename.split('flight_')[1].split('.csv')[0])
        except (IndexError, ValueError):
            flight_id = len(flights)
        flight_ids.append(flight_id)

        flight = pd.read_csv(path)
        flight_array = flight.values
        flights.append(flight_array)

    if not flights:
        raise ValueError(f"No flights found in {flight_dir} with aircraft filter {aircraft_types}")

    flights_array = np.stack(flights, axis=0)
    return flights_array, flight_ids


def compute_normalization_parameters(data_dir: str, max_files: int = 100, aircraft_types: list = None):
    """
    Compute global normalization parameters from training data.

    Args:
        data_dir: Directory containing CSV flight data files (train split)
        max_files: Maximum number of files to use for computing statistics
        aircraft_types: List of aircraft type prefixes to filter by (e.g., ["Cessna_172S", "PA-28-181"])

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
        'std': data_std,
    }


def load_bert_model(checkpoint_path, feat_dim, device):
    """Load BERT masked regressor model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Try to get config from checkpoint, otherwise use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_size = config.get('hidden_size', 1024)
        encoder_layers = config.get('encoder_layers', 8)
        decoder_layers = config.get('decoder_layers', 6)
        num_heads = config.get('num_heads', 16)
        max_seq_len = config.get('seq_len', 10000)
    else:
        # Defaults for full flight BERT
        hidden_size = 1024
        encoder_layers = 8
        decoder_layers = 6
        num_heads = 16
        max_seq_len = 10000

    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
    )

    # Handle torch.compile() prefix in state dict keys
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, {
        'hidden_size': hidden_size,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'num_heads': num_heads,
        'max_seq_len': max_seq_len,
    }


def load_lstm_model(checkpoint_path, feat_dim, device):
    """Load LSTM baseline model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_size = config.get('hidden_size', 256)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        bidirectional = config.get('bidirectional', True)
    else:
        hidden_size = 256
        num_layers = 2
        dropout = 0.1
        bidirectional = True

    model = LSTMBaseline(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'bidirectional': bidirectional,
    }


def load_mlp_model(checkpoint_path, feat_dim, device):
    """Load MLP baseline model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_sizes = config.get('hidden_sizes', [256, 512, 256])
        dropout = config.get('dropout', 0.1)
    else:
        hidden_sizes = [256, 512, 256]
        dropout = 0.1

    model = MLPBaseline(
        feat_dim=feat_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, {
        'hidden_sizes': hidden_sizes,
        'dropout': dropout,
    }


def load_patchtst_model(checkpoint_path, feat_dim, device):
    """Load PatchTST masked regressor model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
    a = checkpoint['args']

    model = PatchTSTMaskedRegressor(
        feat_dim=a['feat_dim'],
        seq_len=a['seq_len'],
        patch_len=a['patch_len'],
        stride=a['stride'],
        d_model=a['d_model'],
        n_heads=a['n_heads'],
        d_ff=a['d_ff'],
        encoder_layers=a['encoder_layers'],
        decoder_layers=a['decoder_layers'],
        dropout=0.0,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, {k: a[k] for k in ['d_model', 'encoder_layers', 'decoder_layers', 'n_heads', 'seq_len']}


def load_model(model_type, checkpoint_path, feat_dim, device):
    """Load model based on type."""
    loaders = {
        'bert': load_bert_model,
        'lstm': load_lstm_model,
        'mlp': load_mlp_model,
        'patchtst': load_patchtst_model,
    }

    if model_type not in loaders:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(loaders.keys())}")

    return loaders[model_type](checkpoint_path, feat_dim, device)


def contiguous_middle_mask(sequence: np.ndarray, mask_ratio: float = 0.3, random_seed: int = None):
    """
    Create a contiguous mask in the middle of the sequence.

    Unlike geometric masking which creates scattered masked regions,
    this creates a single contiguous block suitable for physics evaluation.

    Args:
        sequence: Input array of shape (seq_len, feat_dim)
        mask_ratio: Fraction of sequence to mask (default: 0.3)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (masked_sequence, mask) where mask is 1 for visible, 0 for masked
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    seq_len, feat_dim = sequence.shape
    mask_length = int(seq_len * mask_ratio)

    # Ensure we have context on both sides (at least 10% each)
    min_start = max(1, int(seq_len * 0.1))
    max_start = seq_len - mask_length - max(1, int(seq_len * 0.1))

    if max_start <= min_start:
        # Sequence too short, just mask the middle
        mask_start = seq_len // 4
        mask_length = seq_len // 2
    else:
        mask_start = np.random.randint(min_start, max_start)

    # Create mask (1 = visible, 0 = masked)
    mask = np.ones((seq_len, feat_dim), dtype=np.float32)
    mask[mask_start:mask_start + mask_length, :] = 0.0

    # Create masked sequence
    masked_sequence = sequence * mask

    return masked_sequence, mask


def compute_physics_loss_for_flight(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    feature_map: PhysicsFeatureMap,
    dynamics: AircraftDynamics,
    lambda_control: float = 0.01,
    min_segment_length: int = 10,
    compute_gt_baseline: bool = True,
) -> dict:
    """
    Compute physics loss for a single flight.

    Args:
        original: Original flight data in original scale (seq_len, feat_dim)
        reconstructed: Reconstructed flight data in original scale (seq_len, feat_dim)
        mask: Mask array where 0 = masked positions (seq_len, feat_dim)
        feature_map: PhysicsFeatureMap for column mapping
        dynamics: AircraftDynamics model
        lambda_control: Control regularization weight
        min_segment_length: Minimum segment length to evaluate
        compute_gt_baseline: Whether to compute ground truth baseline

    Returns:
        Dict with physics loss metrics or None if no valid segment
    """
    # Convert mask to boolean (True = keep, False = masked)
    mask_bool = mask > 0.5

    result = physics_trajectory_optimization(
        original_sequence=original,
        reconstructed_sequence=reconstructed,
        mask=mask_bool,
        feature_map=feature_map,
        dynamics=dynamics,
        lambda_control=lambda_control,
        min_segment_length=min_segment_length,
        compute_gt_baseline=compute_gt_baseline,
    )

    return result


def evaluate_model(model, test_data, flight_ids, normalization_params,
                   masking_ratio=0.6, mean_mask_length=3, batch_size=32,
                   device="cuda", physics_eval=False, feature_map=None,
                   dynamics=None, lambda_control=0.01, physics_mask_ratio=0.3,
                   max_physics_samples=100, compute_gt_baseline=True,
                   use_amp=True):
    """
    Evaluate model on masked regression task.

    Uses the same masking protocol for fair comparison across models.
    Optionally includes physics-based evaluation with contiguous masking.

    Args:
        model: Trained model
        test_data: Test data array (num_flights, seq_len, feat_dim)
        flight_ids: List of flight IDs
        normalization_params: Dict with 'mean' and 'std'
        masking_ratio: Masking ratio for geometric masking
        mean_mask_length: Mean mask length for geometric masking
        batch_size: Batch size for evaluation
        device: Device to use
        physics_eval: Whether to compute physics loss
        feature_map: PhysicsFeatureMap for physics evaluation
        dynamics: AircraftDynamics for physics evaluation
        lambda_control: Control regularization weight for physics
        physics_mask_ratio: Mask ratio for contiguous physics masking
    """
    model.eval()

    mean = normalization_params['mean']
    std = normalization_params['std']

    # Normalize test data
    test_data_normalized = (test_data - mean) / std

    test_dataset = TensorDataset(
        torch.from_numpy(test_data_normalized).float(),
        torch.tensor(flight_ids, dtype=torch.long)
    )
    # Use pin_memory for faster CPU->GPU transfer
    pin_memory = device.type == 'cuda'
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=4,
        persistent_workers=True,
    )

    total_mse = 0.0
    total_mae = 0.0
    total_masked_mse = 0.0
    total_masked_mae = 0.0
    total_samples = 0
    total_masked_positions = 0

    # Physics loss accumulators
    physics_losses = []
    gt_physics_losses = []
    per_state_mses = []

    # Keep track of batch index for physics eval
    batch_start_idx = 0

    with torch.no_grad():
        for data, batch_ids in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device, non_blocking=True)
            batch_size_actual = data.shape[0]

            # Keep original on CPU for masking (avoid GPU->CPU->GPU roundtrip)
            original_data_cpu = data.cpu().numpy()
            masked_batch = []
            batch_masks = []

            # Apply masking with deterministic seed per flight
            for sequence, flight_id in zip(original_data_cpu, batch_ids):
                _, masked_sequence, mask = mask_transform(
                    sequence,
                    masking_ratio=masking_ratio,
                    mean_mask_length=mean_mask_length,
                    mode='separate',
                    distribution='geometric',
                    random_seed=int(flight_id)
                )
                masked_batch.append(masked_sequence.numpy())
                batch_masks.append(mask.numpy())

            masked_data = torch.from_numpy(np.stack(masked_batch)).to(device, dtype=torch.float32, non_blocking=True)
            masks = np.stack(batch_masks)  # (batch, seq_len, feat_dim)

            # Forward pass with optional AMP
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    reconstructed = model(masked_data)
            else:
                reconstructed = model(masked_data)

            original_np = original_data_cpu  # Reuse CPU copy
            recon_np = reconstructed.float().cpu().numpy()

            # Overall metrics (all positions)
            total_mse += np.sum((original_np - recon_np) ** 2)
            total_mae += np.sum(np.abs(original_np - recon_np))
            total_samples += original_np.size

            # Masked-only metrics (only masked positions)
            mask_bool = masks > 0  # True where masked
            masked_original = original_np[mask_bool]
            masked_recon = recon_np[mask_bool]

            total_masked_mse += np.sum((masked_original - masked_recon) ** 2)
            total_masked_mae += np.sum(np.abs(masked_original - masked_recon))
            total_masked_positions += masked_original.size

            # Physics evaluation with contiguous masking (separate pass, limited to max_physics_samples)
            if physics_eval and feature_map is not None and dynamics is not None:
                for i in range(batch_size_actual):
                    # Skip if we've already evaluated enough samples for physics
                    if len(physics_losses) >= max_physics_samples:
                        break
                    flight_id = int(batch_ids[i])
                    sequence_norm = original_data[i]
                    sequence_original = test_data[batch_start_idx + i]

                    # Create contiguous mask for physics evaluation
                    masked_seq_norm, phys_mask = contiguous_middle_mask(
                        sequence_norm,
                        mask_ratio=physics_mask_ratio,
                        random_seed=flight_id
                    )

                    # Run model with contiguous mask
                    masked_input = torch.from_numpy(masked_seq_norm).unsqueeze(0).to(device, dtype=torch.float32, non_blocking=True)
                    if use_amp and device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            phys_recon = model(masked_input)
                    else:
                        phys_recon = model(masked_input)
                    phys_recon_np = phys_recon.float().squeeze(0).cpu().numpy()

                    # Denormalize reconstruction
                    phys_recon_original = phys_recon_np * std + mean

                    # Compute physics loss
                    phys_result = compute_physics_loss_for_flight(
                        original=sequence_original,
                        reconstructed=phys_recon_original,
                        mask=phys_mask,
                        feature_map=feature_map,
                        dynamics=dynamics,
                        lambda_control=lambda_control,
                        min_segment_length=50,
                        compute_gt_baseline=compute_gt_baseline,
                    )

                    if phys_result is not None:
                        physics_losses.append(phys_result['physics_loss'])
                        gt_physics_losses.append(phys_result['gt_physics_loss'])
                        per_state_mses.append(phys_result['per_state_mse'])

            batch_start_idx += batch_size_actual

    metrics = {
        'overall_mse': float(total_mse / total_samples),
        'overall_mae': float(total_mae / total_samples),
        'overall_rmse': float(np.sqrt(total_mse / total_samples)),
        'masked_mse': float(total_masked_mse / total_masked_positions),
        'masked_mae': float(total_masked_mae / total_masked_positions),
        'masked_rmse': float(np.sqrt(total_masked_mse / total_masked_positions)),
        'num_samples': int(total_samples),
        'num_masked_positions': int(total_masked_positions),
    }

    # Add physics metrics if computed
    if physics_eval and physics_losses:
        state_names = feature_map.state_short_names() if feature_map else ["roll", "pitch", "altitude", "heading", "airspeed", "fuel"]
        per_state_mse_mean = np.mean(per_state_mses, axis=0)
        per_state_rmse_mean = np.sqrt(per_state_mse_mean)

        # State weights for normalized contributions (from aircraft_dynamics.py)
        # These normalize so each state contributes roughly equally regardless of units
        state_weights = np.array([
            1.0 / 60.0 ** 2,    # roll: typical range ~120 deg
            1.0 / 30.0 ** 2,    # pitch: typical range ~60 deg
            1.0 / 5000.0 ** 2,  # altitude: typical range ~10000 ft
            1.0 / 180.0 ** 2,   # heading: typical range ~360 deg
            1.0 / 100.0 ** 2,   # airspeed: typical range ~160 kts
            1.0 / 50.0 ** 2,    # fuel: typical range ~60 gal
        ])[:len(per_state_mse_mean)]
        per_state_weighted = per_state_mse_mean * state_weights

        metrics['physics'] = {
            'physics_loss_mean': float(np.mean(physics_losses)),
            'physics_loss_std': float(np.std(physics_losses)),
            'physics_loss_median': float(np.median(physics_losses)),
            'gt_physics_loss_mean': float(np.mean(gt_physics_losses)),
            'gt_physics_loss_std': float(np.std(gt_physics_losses)),
            'physics_loss_ratio': float(np.mean(physics_losses) / np.mean(gt_physics_losses)) if np.mean(gt_physics_losses) > 0 else float('inf'),
            'num_physics_samples': len(physics_losses),
            'physics_mask_ratio': physics_mask_ratio,
            'per_state_mse': {name: float(per_state_mse_mean[i]) for i, name in enumerate(state_names[:len(per_state_mse_mean)])},
            'per_state_rmse': {name: float(per_state_rmse_mean[i]) for i, name in enumerate(state_names[:len(per_state_rmse_mean)])},
            'per_state_weighted': {name: float(per_state_weighted[i]) for i, name in enumerate(state_names[:len(per_state_weighted)])},
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate masked regression models (BERT, LSTM, MLP) on test data'
    )

    # Required arguments
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'lstm', 'mlp', 'patchtst'],
                        help='Type of model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test flight CSV files')

    # Normalization parameters (optional - will compute from train data if not provided)
    parser.add_argument('--norm_params', type=str, default=None,
                        help='Path to normalization parameters (.npy file). If not provided, will compute from train data.')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory containing training data for computing normalization. Defaults to sibling "train" folder of data_dir.')

    # Aircraft filtering for normalization (to prevent data leakage)
    parser.add_argument('--aircraft_type', type=str, nargs='+', default=None,
                        choices=["Cessna_172S", "PA-28-181", "PA-44-180"],
                        help='Filter normalization data to specific aircraft type(s)')
    parser.add_argument('--aircraft_class', type=str, default=None,
                        choices=["single_engine", "multi_engine"],
                        help='Filter normalization data by aircraft class (single_engine=Cessna_172S+PA-28-181, multi_engine=PA-44-180)')

    # Cross-aircraft generalization evaluation
    parser.add_argument('--cross_aircraft_eval', action='store_true',
                        help='Enable cross-aircraft generalization: normalize on single-engine (Cessna_172S, PA-28-181), test on multi-engine (PA-44-180)')

    # Masking parameters (should match training/validation setup)
    parser.add_argument('--masking_ratio', type=float, default=0.6,
                        help='Masking ratio (default: 0.6)')
    parser.add_argument('--mean_mask_length', type=int, default=3,
                        help='Mean mask length (default: 3)')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision (AMP)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile() optimization')

    # Physics evaluation parameters
    parser.add_argument('--physics_eval', action='store_true',
                        help='Enable physics-based evaluation using MPC trajectory optimization')
    parser.add_argument('--aircraft_preset', type=str, default='cessna172s',
                        choices=['cessna172s', 'pa28', 'pa44'],
                        help='Aircraft dynamics preset for physics evaluation')
    parser.add_argument('--feature_map_preset', type=str, default='ngafid_44col',
                        choices=['ngafid_44col', 'input_cols_40', 'toy_ngafid'],
                        help='Feature map preset for physics state extraction')
    parser.add_argument('--lambda_control', type=float, default=0.01,
                        help='Control regularization weight for physics loss')
    parser.add_argument('--physics_mask_ratio', type=float, default=0.3,
                        help='Mask ratio for contiguous physics masking (default: 0.3)')
    parser.add_argument('--max_physics_samples', type=int, default=100,
                        help='Maximum number of flights to evaluate for physics loss (default: 100)')
    parser.add_argument('--skip_gt_physics', action='store_true',
                        help='Skip ground truth physics baseline (2x faster, no ratio metric)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./masked_regression_results',
                        help='Directory to save results (default: ./masked_regression_results)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this evaluation run')

    args = parser.parse_args()

    # Resolve aircraft_class to aircraft_type list
    AIRCRAFT_CLASS_MAP = {
        "single_engine": ["Cessna_172S", "PA-28-181"],
        "multi_engine": ["PA-44-180"],
    }

    # Cross-aircraft evaluation: train on single-engine, test on multi-engine
    if args.cross_aircraft_eval:
        print("\n" + "="*60)
        print("CROSS-AIRCRAFT GENERALIZATION EVALUATION")
        print("  Normalization: single-engine (Cessna_172S, PA-28-181)")
        print("  Test data: multi-engine (PA-44-180)")
        print("="*60 + "\n")
        args.aircraft_type = AIRCRAFT_CLASS_MAP["single_engine"]
        test_aircraft_filter = AIRCRAFT_CLASS_MAP["multi_engine"]
    else:
        test_aircraft_filter = None
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

    # Load test data
    print(f"Loading test data from {args.data_dir}...")
    test_data, flight_ids = load_flight_data(args.data_dir, aircraft_types=test_aircraft_filter)
    feat_dim = test_data.shape[2]
    print(f"  Loaded {len(flight_ids)} flights, shape: {test_data.shape}")

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
        print("  Using automatic mixed precision (bfloat16)")

    # Warmup pass to initialize CUDA kernels
    if torch.cuda.is_available():
        print("  Running warmup pass...")
        seq_len = test_data.shape[1]
        dummy_input = torch.randn(2, seq_len, feat_dim, device=device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
        torch.cuda.synchronize()
        del dummy_input

    # Setup physics evaluation if enabled
    feature_map = None
    dynamics = None
    if args.physics_eval:
        print(f"\nPhysics evaluation enabled:")
        print(f"  Aircraft preset: {args.aircraft_preset}")
        print(f"  Feature map: {args.feature_map_preset}")
        print(f"  Lambda control: {args.lambda_control}")
        print(f"  Physics mask ratio: {args.physics_mask_ratio}")
        print(f"  Max physics samples: {args.max_physics_samples}")

        feature_map = PRESET_FEATURE_MAPS[args.feature_map_preset]
        aircraft_params = AIRCRAFT_PRESETS[args.aircraft_preset]
        dynamics = AircraftDynamics(
            dt=1.0,
            has_fuel=feature_map.has_fuel,
            **aircraft_params
        )

    # Evaluate
    print(f"\nEvaluating with masking_ratio={args.masking_ratio}, mean_mask_length={args.mean_mask_length}...")
    metrics = evaluate_model(
        model,
        test_data,
        flight_ids,
        normalization_params,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        batch_size=args.batch_size,
        device=device,
        physics_eval=args.physics_eval,
        feature_map=feature_map,
        dynamics=dynamics,
        lambda_control=args.lambda_control,
        physics_mask_ratio=args.physics_mask_ratio,
        max_physics_samples=args.max_physics_samples,
        compute_gt_baseline=not args.skip_gt_physics,
        use_amp=use_amp,
    )

    # Print results
    print("\n" + "="*60)
    if args.cross_aircraft_eval:
        print("CROSS-AIRCRAFT MASKED REGRESSION RESULTS")
        print("  (Trained on single-engine, tested on multi-engine)")
    else:
        print("MASKED REGRESSION RESULTS")
    print("="*60)
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Data: {args.data_dir}")
    if args.cross_aircraft_eval:
        print(f"Norm Aircraft: {args.aircraft_type}")
        print(f"Test Aircraft: {test_aircraft_filter}")
    print(f"Masking: ratio={args.masking_ratio}, mean_length={args.mean_mask_length}")
    print("-"*60)
    print("Data Loss - Overall Metrics (all positions):")
    print(f"  MSE:  {metrics['overall_mse']:.6f}")
    print(f"  MAE:  {metrics['overall_mae']:.6f}")
    print(f"  RMSE: {metrics['overall_rmse']:.6f}")
    print("-"*60)
    print("Data Loss - Masked-Only Metrics (masked positions only):")
    print(f"  MSE:  {metrics['masked_mse']:.6f}")
    print(f"  MAE:  {metrics['masked_mae']:.6f}")
    print(f"  RMSE: {metrics['masked_rmse']:.6f}")

    # Print physics metrics if computed
    if 'physics' in metrics:
        print("-"*60)
        print("Physics Loss Metrics (MPC Trajectory Optimization):")
        print(f"  Note: Uses contiguous masking (ratio={metrics['physics']['physics_mask_ratio']})")
        phys = metrics['physics']
        print(f"  Physics Loss (mean):   {phys['physics_loss_mean']:.6f}")
        print(f"  Physics Loss (std):    {phys['physics_loss_std']:.6f}")
        print(f"  Physics Loss (median): {phys['physics_loss_median']:.6f}")
        print(f"  GT Physics Loss:       {phys['gt_physics_loss_mean']:.6f}")
        print(f"  Physics/GT Ratio:      {phys['physics_loss_ratio']:.4f}")
        print(f"  Num physics samples:   {phys['num_physics_samples']}")
        print("  Per-State Breakdown (RMSE in original units, weighted contribution to loss):")
        for state in phys['per_state_rmse'].keys():
            rmse = phys['per_state_rmse'][state]
            weighted = phys['per_state_weighted'][state]
            unit = {'roll': 'deg', 'pitch': 'deg', 'altitude': 'ft', 'heading': 'deg', 'airspeed': 'kts', 'fuel': 'gal'}.get(state, '')
            print(f"    {state:12s}: RMSE={rmse:10.2f} {unit:4s}  weighted={weighted:.4f}")

    print("="*60)

    # Prepare results for saving
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
        'norm_params_source': norm_source,
        'timestamp': timestamp,
        'run_name': run_name,
        'cross_aircraft_eval': args.cross_aircraft_eval,
        'config': {
            'masking_ratio': args.masking_ratio,
            'mean_mask_length': args.mean_mask_length,
            'batch_size': args.batch_size,
            'feat_dim': feat_dim,
            'num_test_flights': len(flight_ids),
            'num_parameters': num_params,
            'norm_aircraft_types': args.aircraft_type,
            'test_aircraft_types': test_aircraft_filter,
        },
        'model_config': model_config,
        'metrics': metrics,
    }

    # Add physics config if enabled
    if args.physics_eval:
        results['physics_config'] = {
            'aircraft_preset': args.aircraft_preset,
            'feature_map_preset': args.feature_map_preset,
            'lambda_control': args.lambda_control,
            'physics_mask_ratio': args.physics_mask_ratio,
            'max_physics_samples': args.max_physics_samples,
        }

    # Save results
    output_file = os.path.join(args.output_dir, f"{run_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
