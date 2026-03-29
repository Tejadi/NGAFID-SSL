#!/usr/bin/env python3
"""
Unified evaluation script for forecasting benchmark.
Supports BERT models with forecasting (end-of-sequence masking).

Unlike masked regression (random masking), forecasting masks the END of each flight,
creating a causal prediction task where the model predicts future timesteps.

Includes optional physics-based evaluation using MPC trajectory optimization.

Usage:
    python eval_forecasting.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data
    python eval_forecasting.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data --forecast_ratio 0.2
    python eval_forecasting.py --model_type bert --checkpoint path/to/model.pt --data_dir path/to/test_data --physics_eval
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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


def load_flight_data(flight_dir):
    """Load flight data from CSV files in a directory."""
    csv_files = list(Path(flight_dir).glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in {flight_dir}")

    flights = []
    flight_ids = []
    sequence_lengths = []

    for path in tqdm(csv_files, desc='Loading flight data'):
        filename = path.name

        # Extract flight ID
        try:
            flight_id = int(filename.split('flight_')[1].split('.csv')[0])
        except:
            flight_id = len(flights)

        flight = pd.read_csv(path, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

        # Select numeric columns and clean
        numeric_cols = flight.select_dtypes(include=[np.number]).columns
        flight_numeric = flight[numeric_cols].ffill().bfill()
        flight_array = flight_numeric.values.astype(np.float32)

        # Store original length before any padding
        original_length = min(flight_array.shape[0], 10000)

        flights.append(flight_array)
        flight_ids.append(flight_id)
        sequence_lengths.append(original_length)

    # Pad/truncate to same length
    max_len = 10000
    feat_dim = flights[0].shape[1]

    padded_flights = []
    for flight in flights:
        if flight.shape[0] > max_len:
            padded_flights.append(flight[:max_len])
        elif flight.shape[0] < max_len:
            padding = np.zeros((max_len - flight.shape[0], feat_dim), dtype=np.float32)
            padded_flights.append(np.vstack([flight, padding]))
        else:
            padded_flights.append(flight)

    flights_array = np.stack(padded_flights, axis=0)
    return flights_array, flight_ids, sequence_lengths


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


def forecast_mask(X, forecast_ratio, sequence_length, min_horizon=100):
    """
    Create a forecasting mask that masks the END of the sequence.

    Args:
        X: Input array of shape (seq_len, feat_dim)
        forecast_ratio: Fraction of original sequence to mask/predict
        sequence_length: Original sequence length before padding
        min_horizon: Minimum number of timesteps to mask

    Returns:
        mask: Boolean array (True = keep/context, False = mask/predict)
        forecast_horizon: Number of timesteps being predicted
    """
    total_len = X.shape[0]

    # Calculate forecast horizon based on original sequence length
    forecast_horizon = max(min_horizon, int(sequence_length * forecast_ratio))

    # Ensure we have enough context (at least 50% for context)
    max_horizon = sequence_length // 2
    forecast_horizon = min(forecast_horizon, max_horizon)

    # Create mask: True for context, False for forecast, True for padding
    mask = np.ones(X.shape, dtype=np.float32)

    # Mask the forecast portion (end of original sequence)
    forecast_start = sequence_length - forecast_horizon
    forecast_end = sequence_length
    mask[forecast_start:forecast_end, :] = 0.0

    return mask, forecast_horizon


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


def load_patchtst_model(checkpoint_path, feat_dim, device):
    """Load PatchTST model from checkpoint."""
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
    return loaders[model_type](checkpoint_path, feat_dim, device)


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
        mask: Mask array where 0 = masked/forecast positions (seq_len, feat_dim)
        feature_map: PhysicsFeatureMap for column mapping
        dynamics: AircraftDynamics model
        lambda_control: Control regularization weight
        min_segment_length: Minimum segment length to evaluate

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


def _physics_worker(args):
    """Worker function for parallel physics evaluation."""
    (original, reconstructed, mask, feature_map_preset, aircraft_preset,
     lambda_control, min_segment_length, compute_gt_baseline) = args

    # Recreate objects in worker process
    feature_map = PRESET_FEATURE_MAPS[feature_map_preset]
    aircraft_params = AIRCRAFT_PRESETS[aircraft_preset]
    dynamics = AircraftDynamics(
        dt=1.0,
        has_fuel=feature_map.has_fuel,
        **aircraft_params
    )

    return compute_physics_loss_for_flight(
        original=original,
        reconstructed=reconstructed,
        mask=mask,
        feature_map=feature_map,
        dynamics=dynamics,
        lambda_control=lambda_control,
        min_segment_length=min_segment_length,
        compute_gt_baseline=compute_gt_baseline,
    )


def evaluate_model(model, test_data, flight_ids, sequence_lengths, normalization_params,
                   forecast_ratio=0.2, min_horizon=100, batch_size=16, device='cuda',
                   physics_eval=False, feature_map=None, dynamics=None, lambda_control=0.01,
                   max_physics_samples=100, compute_gt_baseline=True, physics_workers=8,
                   feature_map_preset='ngafid_44col', aircraft_preset='cessna172s',
                   causal_mask=False, model_type='bert'):
    """
    Evaluate model on forecasting task.

    Args:
        model: Trained model
        test_data: Test data array (num_flights, seq_len, feat_dim)
        flight_ids: List of flight IDs
        sequence_lengths: List of original sequence lengths
        normalization_params: Dict with 'mean' and 'std'
        forecast_ratio: Fraction of sequence to predict
        min_horizon: Minimum timesteps to predict
        batch_size: Batch size for evaluation
        device: Device to use
        physics_eval: Whether to compute physics loss
        feature_map: PhysicsFeatureMap for physics evaluation
        dynamics: AircraftDynamics for physics evaluation
        lambda_control: Control regularization weight for physics
        causal_mask: If True, apply causal attention mask (BERT can only see past)
        model_type: Type of model ('bert', 'lstm', 'mlp')

    Returns:
        Dictionary of metrics
    """
    model.eval()

    mean = normalization_params['mean']
    std = normalization_params['std']

    # Normalize data
    normalized_data = (test_data - mean) / std

    total_mse = 0.0
    total_mae = 0.0
    total_forecast_mse = 0.0
    total_forecast_mae = 0.0
    total_samples = 0
    total_forecast_positions = 0

    # Physics loss accumulators
    physics_results = []
    physics_losses = []
    gt_physics_losses = []
    per_state_mses = []
    physics_tasks = []  # For parallel processing

    num_flights = len(test_data)
    num_batches = (num_flights + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_flights)

            batch_data = normalized_data[start_idx:end_idx]
            batch_original = test_data[start_idx:end_idx]  # Keep original scale for physics
            batch_seq_lens = sequence_lengths[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx

            # Create forecast masks for each sample
            batch_masks = []
            batch_horizons = []
            for i in range(batch_size_actual):
                mask, horizon = forecast_mask(
                    batch_data[i],
                    forecast_ratio=forecast_ratio,
                    sequence_length=batch_seq_lens[i],
                    min_horizon=min_horizon
                )
                batch_masks.append(mask)
                batch_horizons.append(horizon)

            masks = np.stack(batch_masks)

            # Create masked input
            X = torch.tensor(batch_data, dtype=torch.float32).to(device)
            mask_tensor = torch.tensor(masks, dtype=torch.float32).to(device)
            X_masked = X * mask_tensor

            # Create causal attention mask if requested
            # This forces BERT to only attend to past positions (true forecasting)
            attn_mask = None
            if causal_mask:
                seq_len = X_masked.shape[1]
                # Create lower triangular mask: position i can attend to positions 0..i
                # Shape: (1, 1, seq_len, seq_len) for broadcasting
                # 0 = attend, -inf = don't attend (BERT convention after processing)
                causal_attn_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                    diagonal=1
                )
                # Expand for batch: (batch_size, 1, seq_len, seq_len)
                attn_mask = causal_attn_mask.unsqueeze(0).unsqueeze(0).expand(
                    batch_size_actual, 1, seq_len, seq_len
                )

            # Forward pass - only pass attention_mask for BERT models
            if model_type == 'bert':
                reconstruction = model(X_masked, attention_mask=attn_mask)
            else:
                reconstruction = model(X_masked)

            # Compute metrics
            X_np = X.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()
            mask_np = masks

            # Overall MSE/MAE
            mse = np.mean((X_np - recon_np) ** 2)
            mae = np.mean(np.abs(X_np - recon_np))

            # MSE/MAE only on forecast positions (mask == 0)
            forecast_mask_bool = (mask_np == 0)
            num_forecast = forecast_mask_bool.sum()

            if num_forecast > 0:
                forecast_mse = np.sum((X_np[forecast_mask_bool] - recon_np[forecast_mask_bool]) ** 2)
                forecast_mae = np.sum(np.abs(X_np[forecast_mask_bool] - recon_np[forecast_mask_bool]))
            else:
                forecast_mse = 0.0
                forecast_mae = 0.0

            total_mse += mse * batch_size_actual
            total_mae += mae * batch_size_actual
            total_forecast_mse += forecast_mse
            total_forecast_mae += forecast_mae
            total_samples += batch_size_actual
            total_forecast_positions += num_forecast

            # Collect data for parallel physics evaluation
            if physics_eval and feature_map is not None and dynamics is not None:
                # Denormalize reconstruction back to original scale
                recon_original_scale = recon_np * std + mean

                for i in range(batch_size_actual):
                    # Skip if we've already collected enough samples for physics
                    if len(physics_tasks) >= max_physics_samples:
                        break
                    seq_len = batch_seq_lens[i]
                    original_flight = batch_original[i][:seq_len].copy()
                    recon_flight = recon_original_scale[i][:seq_len].copy()
                    mask_flight = mask_np[i][:seq_len].copy()

                    physics_tasks.append((
                        original_flight,
                        recon_flight,
                        mask_flight,
                        feature_map_preset,
                        aircraft_preset,
                        lambda_control,
                        min_horizon // 2,
                        compute_gt_baseline,
                    ))

    # Run physics evaluation in parallel
    if physics_eval and physics_tasks:
        print(f"\nRunning physics evaluation on {len(physics_tasks)} flights with {physics_workers} workers...")
        with Pool(processes=physics_workers) as pool:
            results = list(tqdm(
                pool.imap(_physics_worker, physics_tasks),
                total=len(physics_tasks),
                desc="Physics eval"
            ))

        for phys_result in results:
            if phys_result is not None:
                physics_results.append(phys_result)
                physics_losses.append(phys_result['physics_loss'])
                gt_physics_losses.append(phys_result['gt_physics_loss'])
                per_state_mses.append(phys_result['per_state_mse'])

    # Compute averages
    metrics = {
        'overall_mse': total_mse / total_samples,
        'overall_mae': total_mae / total_samples,
        'overall_rmse': np.sqrt(total_mse / total_samples),
        'forecast_mse': total_forecast_mse / total_forecast_positions if total_forecast_positions > 0 else 0,
        'forecast_mae': total_forecast_mae / total_forecast_positions if total_forecast_positions > 0 else 0,
        'forecast_rmse': np.sqrt(total_forecast_mse / total_forecast_positions) if total_forecast_positions > 0 else 0,
        'num_samples': total_samples,
        'num_forecast_positions': int(total_forecast_positions),
        'forecast_ratio': forecast_ratio,
        'min_horizon': min_horizon,
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
            'per_state_mse': {name: float(per_state_mse_mean[i]) for i, name in enumerate(state_names[:len(per_state_mse_mean)])},
            'per_state_rmse': {name: float(per_state_rmse_mean[i]) for i, name in enumerate(state_names[:len(per_state_rmse_mean)])},
            'per_state_weighted': {name: float(per_state_weighted[i]) for i, name in enumerate(state_names[:len(per_state_weighted)])},
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models on forecasting benchmark',
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

    # Forecasting parameters
    parser.add_argument('--forecast_ratio', type=float, default=0.2,
                        help='Fraction of sequence to predict (default: 0.2 = last 20%%)')
    parser.add_argument('--min_horizon', type=int, default=100,
                        help='Minimum timesteps to predict (default: 100)')
    parser.add_argument('--causal_mask', action='store_true',
                        help='Apply causal attention mask (BERT can only attend to past positions). '
                             'Tests true forecasting without bidirectional "peeking".')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')

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
    parser.add_argument('--max_physics_samples', type=int, default=100,
                        help='Maximum number of flights to evaluate for physics loss (default: 100)')
    parser.add_argument('--skip_gt_physics', action='store_true',
                        help='Skip ground truth physics baseline (2x faster, no ratio metric)')
    parser.add_argument('--physics_workers', type=int, default=8,
                        help='Number of parallel workers for physics evaluation (default: 8)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./forecasting_results',
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

    # Load test data
    print(f"Loading test data from {args.data_dir}...")
    test_data, flight_ids, sequence_lengths = load_flight_data(args.data_dir)
    feat_dim = test_data.shape[2]
    print(f"  Loaded {len(flight_ids)} flights, shape: {test_data.shape}")

    # Load model
    print(f"Loading {args.model_type.upper()} model from {args.checkpoint}...")
    model, model_config = load_model(args.model_type, args.checkpoint, feat_dim, device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Setup physics evaluation if enabled
    feature_map = None
    dynamics = None
    if args.physics_eval:
        print(f"\nPhysics evaluation enabled:")
        print(f"  Aircraft preset: {args.aircraft_preset}")
        print(f"  Feature map: {args.feature_map_preset}")
        print(f"  Lambda control: {args.lambda_control}")
        print(f"  Max physics samples: {args.max_physics_samples}")

        feature_map = PRESET_FEATURE_MAPS[args.feature_map_preset]
        aircraft_params = AIRCRAFT_PRESETS[args.aircraft_preset]
        dynamics = AircraftDynamics(
            dt=1.0,
            has_fuel=feature_map.has_fuel,
            **aircraft_params
        )

    # Evaluate
    print(f"\nEvaluating with forecast_ratio={args.forecast_ratio}, min_horizon={args.min_horizon}...")
    metrics = evaluate_model(
        model,
        test_data,
        flight_ids,
        sequence_lengths,
        normalization_params,
        forecast_ratio=args.forecast_ratio,
        min_horizon=args.min_horizon,
        batch_size=args.batch_size,
        device=device,
        physics_eval=args.physics_eval,
        feature_map=feature_map,
        dynamics=dynamics,
        lambda_control=args.lambda_control,
        max_physics_samples=args.max_physics_samples,
        compute_gt_baseline=not args.skip_gt_physics,
        physics_workers=args.physics_workers,
        feature_map_preset=args.feature_map_preset,
        aircraft_preset=args.aircraft_preset,
        causal_mask=args.causal_mask,
        model_type=args.model_type,
    )

    # Print results
    print("\n" + "=" * 60)
    if args.causal_mask:
        print("FORECASTING RESULTS (CAUSAL ATTENTION)")
        print("  (Model can only attend to past positions - true forecasting)")
    else:
        print("FORECASTING RESULTS (BIDIRECTIONAL ATTENTION)")
        print("  (Model can attend to all positions - masked future reconstruction)")
    print("=" * 60)
    print(f"Model: {args.model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Forecast ratio: {args.forecast_ratio} (predict last {args.forecast_ratio*100:.0f}%)")
    print(f"Min horizon: {args.min_horizon}")
    print(f"Causal mask: {args.causal_mask}")
    print("-" * 60)
    print("Data Loss Metrics:")
    print(f"  Overall MSE:   {metrics['overall_mse']:.6f}")
    print(f"  Overall MAE:   {metrics['overall_mae']:.6f}")
    print(f"  Overall RMSE:  {metrics['overall_rmse']:.6f}")
    print("-" * 60)
    print("Forecast-Only Metrics:")
    print(f"  Forecast MSE:  {metrics['forecast_mse']:.6f}")
    print(f"  Forecast MAE:  {metrics['forecast_mae']:.6f}")
    print(f"  Forecast RMSE: {metrics['forecast_rmse']:.6f}")

    # Print physics metrics if computed
    if 'physics' in metrics:
        print("-" * 60)
        print("Physics Loss Metrics (MPC Trajectory Optimization):")
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

    print("-" * 60)
    print(f"Num samples: {metrics['num_samples']}")
    print(f"Num forecast positions: {metrics['num_forecast_positions']:,}")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_type}_{timestamp}"

    results = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'norm_params_source': args.norm_params if args.norm_params else f"computed from {train_dir}",
        'timestamp': timestamp,
        'run_name': run_name,
        'config': {
            'forecast_ratio': args.forecast_ratio,
            'min_horizon': args.min_horizon,
            'batch_size': args.batch_size,
            'feat_dim': feat_dim,
            'num_test_flights': len(flight_ids),
            'num_parameters': num_params,
            'causal_mask': args.causal_mask,
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
            'max_physics_samples': args.max_physics_samples,
        }

    output_file = os.path.join(args.output_dir, f"{run_name}.json")
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
