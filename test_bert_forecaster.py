#!/usr/bin/env python3
"""
Test script for BERT models on forecasting task.

This script evaluates pretrained BERT models (either masked regressor or forecaster)
on the forecasting task, where the model must predict the END of the flight sequence
given only the beginning.

Can also evaluate on random masking for comparison.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.bert_masked_regressor import BertMaskedRegressor
from ngafid_datasets.transformation_dataset import mask_transform


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def forecast_mask_transform(X, forecast_ratio, sequence_length=None, min_horizon=100):
    """
    Create a forecasting mask that masks the END of the sequence.

    Args:
        X: Input array of shape (seq_len, feat_dim)
        forecast_ratio: Fraction of original sequence to mask/predict
        sequence_length: Original sequence length before padding (if None, uses X.shape[0])
        min_horizon: Minimum number of timesteps to mask

    Returns:
        X: Original tensor
        transformed_X: Masked tensor (forecast portion zeroed out)
        mask: Boolean mask (True = keep, False = mask/predict)
    """
    total_len = X.shape[0]

    if sequence_length is None:
        sequence_length = total_len

    # Calculate forecast horizon based on original sequence length
    forecast_horizon = max(min_horizon, int(sequence_length * forecast_ratio))

    # Ensure we have enough context (at least 50% for context)
    max_horizon = sequence_length // 2
    forecast_horizon = min(forecast_horizon, max_horizon)

    # Create mask: 1 for context, 0 for forecast, 1 for padding
    mask = np.ones_like(X, dtype=bool)

    # Mask the forecast portion (end of original sequence)
    forecast_start = sequence_length - forecast_horizon
    forecast_end = sequence_length
    mask[forecast_start:forecast_end, :] = False

    # Padding (after sequence_length) stays as True (not predicted)

    X_tensor = torch.from_numpy(X)
    mask_tensor = torch.from_numpy(mask)
    transformed_X = X_tensor * mask_tensor

    return X_tensor, transformed_X, mask_tensor, forecast_horizon


def load_bert_model(model_path, device, config_override=None):
    """Load pretrained BERT model from checkpoint."""
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'args' in checkpoint:
        config = checkpoint['args']
    else:
        config = {}

    # Apply overrides
    if config_override:
        config.update(config_override)

    # Get model parameters
    feat_dim = checkpoint.get('feat_dim', config.get('feat_dim', 44))
    hidden_size = config.get('hidden_size', 1024)
    encoder_layers = config.get('encoder_layers', 8)
    decoder_layers = config.get('decoder_layers', 6)
    num_heads = config.get('num_heads', 16)
    max_seq_len = config.get('seq_len', config.get('max_seq_len', 10000))
    dropout = config.get('dropout', 0.1)

    print(f"Model config: {hidden_size}d, {encoder_layers} enc, {decoder_layers} dec, {num_heads} heads")
    print(f"Sequence length: {max_seq_len}, Feature dim: {feat_dim}")

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

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config, feat_dim


def load_flight_data(data_dir, split='test', max_files=None):
    """Load flight data from CSV files."""
    data_path = Path(data_dir) / "preprocessed_data" / split

    if not data_path.exists():
        # Try alternative path
        data_path = Path(data_dir) / split

    if not data_path.exists():
        raise ValueError(f"Data path not found: {data_path}")

    csv_files = sorted(data_path.glob("*.csv"))

    if max_files:
        csv_files = csv_files[:max_files]

    print(f"Loading {len(csv_files)} flights from {data_path}")

    all_data = []
    flight_ids = []
    sequence_lengths = []
    aircraft_types = []

    for csv_file in tqdm(csv_files, desc="Loading flights"):
        try:
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]

            # Handle missing values
            df_clean = df_numeric.ffill().bfill()

            # Convert to numpy
            flight_data = df_clean.to_numpy(dtype=np.float32)

            if flight_data.shape[0] == 0:
                continue

            # Store original length
            original_length = min(flight_data.shape[0], 10000)

            # Extract flight ID and aircraft type from filename
            # Expected format: AircraftType_flight_ID.csv
            name = csv_file.stem
            parts = name.split('_')

            flight_id = None
            aircraft_type = "Unknown"

            for i, part in enumerate(parts):
                if part == 'flight' and i + 1 < len(parts):
                    try:
                        flight_id = int(parts[i + 1])
                    except ValueError:
                        pass

            if flight_id is None:
                import re
                numbers = re.findall(r'\d+', name)
                if numbers:
                    flight_id = int(numbers[-1])
                else:
                    flight_id = len(all_data)

            # Extract aircraft type (everything before _flight_)
            if '_flight_' in name:
                aircraft_type = name.split('_flight_')[0].replace('_', ' ')

            all_data.append(flight_data)
            flight_ids.append(flight_id)
            sequence_lengths.append(original_length)
            aircraft_types.append(aircraft_type)

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    return all_data, flight_ids, sequence_lengths, aircraft_types


def compute_normalization_params(data_list):
    """Compute global mean and std from data."""
    all_data = np.vstack(data_list)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std[std == 0] = 1.0
    return {'mean': mean, 'std': std}


def evaluate_forecasting(model, data_list, flight_ids, sequence_lengths, normalization_params,
                        forecast_ratio=0.2, min_horizon=100, device='cuda'):
    """
    Evaluate model on forecasting task (end masking).
    """
    model.eval()

    mean = normalization_params['mean']
    std = normalization_params['std']

    total_mse = 0
    total_mae = 0
    total_mse_masked = 0
    total_mae_masked = 0
    total_samples = 0
    total_masked_positions = 0

    all_results = []

    with torch.no_grad():
        for flight_data, flight_id, seq_len in tqdm(
            zip(data_list, flight_ids, sequence_lengths),
            total=len(data_list),
            desc=f"Forecasting eval (ratio={forecast_ratio})"
        ):
            # Normalize
            flight_normalized = (flight_data - mean) / std

            # Create forecast mask
            X, X_masked, mask, horizon = forecast_mask_transform(
                flight_normalized,
                forecast_ratio=forecast_ratio,
                sequence_length=seq_len,
                min_horizon=min_horizon
            )

            # Move to device
            X = X.unsqueeze(0).float().to(device)
            X_masked = X_masked.unsqueeze(0).float().to(device)
            mask = mask.unsqueeze(0).to(device)

            # Forward pass
            reconstruction = model(X_masked)

            # Compute metrics on normalized values
            X_np = X.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Overall MSE/MAE
            mse = np.mean((X_np - recon_np) ** 2)
            mae = np.mean(np.abs(X_np - recon_np))

            # MSE/MAE only on masked (forecast) positions
            masked_positions = ~mask_np
            if masked_positions.sum() > 0:
                mse_masked = np.mean((X_np[masked_positions] - recon_np[masked_positions]) ** 2)
                mae_masked = np.mean(np.abs(X_np[masked_positions] - recon_np[masked_positions]))
            else:
                mse_masked = 0
                mae_masked = 0

            total_mse += mse
            total_mae += mae
            total_mse_masked += mse_masked * masked_positions.sum()
            total_mae_masked += mae_masked * masked_positions.sum()
            total_samples += 1
            total_masked_positions += masked_positions.sum()

            all_results.append({
                'flight_id': flight_id,
                'sequence_length': seq_len,
                'forecast_horizon': horizon,
                'mse': mse,
                'mae': mae,
                'mse_masked': mse_masked,
                'mae_masked': mae_masked,
                'num_masked_positions': int(masked_positions.sum())
            })

    # Aggregate metrics
    metrics = {
        'mse_overall': total_mse / total_samples,
        'mae_overall': total_mae / total_samples,
        'rmse_overall': np.sqrt(total_mse / total_samples),
        'mse_forecast_only': total_mse_masked / total_masked_positions if total_masked_positions > 0 else 0,
        'mae_forecast_only': total_mae_masked / total_masked_positions if total_masked_positions > 0 else 0,
        'rmse_forecast_only': np.sqrt(total_mse_masked / total_masked_positions) if total_masked_positions > 0 else 0,
        'total_flights': total_samples,
        'total_masked_positions': int(total_masked_positions),
        'forecast_ratio': forecast_ratio,
    }

    return metrics, all_results


def evaluate_random_masking(model, data_list, flight_ids, sequence_lengths, normalization_params,
                           masking_ratio=0.5, mean_mask_length=60, device='cuda'):
    """
    Evaluate model on random masking task (standard masked reconstruction).
    """
    model.eval()

    mean = normalization_params['mean']
    std = normalization_params['std']

    total_mse = 0
    total_mae = 0
    total_samples = 0

    with torch.no_grad():
        for flight_data, flight_id, seq_len in tqdm(
            zip(data_list, flight_ids, sequence_lengths),
            total=len(data_list),
            desc=f"Random masking eval (ratio={masking_ratio})"
        ):
            # Normalize
            flight_normalized = (flight_data - mean) / std

            # Create random mask
            X, X_masked, mask = mask_transform(
                flight_normalized,
                masking_ratio=masking_ratio,
                mean_mask_length=mean_mask_length,
                mode='separate',
                distribution='geometric',
                random_seed=flight_id
            )

            # Move to device
            X = X.unsqueeze(0).float().to(device)
            X_masked = X_masked.unsqueeze(0).float().to(device)

            # Forward pass
            reconstruction = model(X_masked)

            # Compute metrics
            X_np = X.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()

            mse = np.mean((X_np - recon_np) ** 2)
            mae = np.mean(np.abs(X_np - recon_np))

            total_mse += mse
            total_mae += mae
            total_samples += 1

    metrics = {
        'mse': total_mse / total_samples,
        'mae': total_mae / total_samples,
        'rmse': np.sqrt(total_mse / total_samples),
        'total_flights': total_samples,
        'masking_ratio': masking_ratio,
        'mean_mask_length': mean_mask_length,
    }

    return metrics


def evaluate_by_aircraft_type(model, data_list, flight_ids, sequence_lengths, aircraft_types,
                             normalization_params, forecast_ratio=0.2, min_horizon=100, device='cuda'):
    """
    Evaluate forecasting performance broken down by aircraft type.
    """
    # Group by aircraft type
    type_groups = {}
    for i, atype in enumerate(aircraft_types):
        if atype not in type_groups:
            type_groups[atype] = {'data': [], 'ids': [], 'lengths': []}
        type_groups[atype]['data'].append(data_list[i])
        type_groups[atype]['ids'].append(flight_ids[i])
        type_groups[atype]['lengths'].append(sequence_lengths[i])

    results_by_type = {}

    for atype, group in type_groups.items():
        print(f"\nEvaluating {atype} ({len(group['data'])} flights)...")
        metrics, _ = evaluate_forecasting(
            model, group['data'], group['ids'], group['lengths'],
            normalization_params, forecast_ratio, min_horizon, device
        )
        results_by_type[atype] = metrics

    return results_by_type


def main():
    parser = argparse.ArgumentParser(
        description='Test BERT models on forecasting task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing flight data')

    # Data arguments
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to evaluate')
    parser.add_argument('--norm_params_path', type=str, default=None,
                       help='Path to normalization parameters (if None, computed from data)')

    # Evaluation mode
    parser.add_argument('--eval_mode', type=str, default='both',
                       choices=['forecast', 'random', 'both'],
                       help='Evaluation mode: forecast (end masking), random, or both')

    # Forecasting arguments
    parser.add_argument('--forecast_ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3],
                       help='Forecast ratios to evaluate')
    parser.add_argument('--min_horizon', type=int, default=100,
                       help='Minimum forecast horizon')

    # Random masking arguments
    parser.add_argument('--masking_ratio', type=float, default=0.5,
                       help='Masking ratio for random masking evaluation')
    parser.add_argument('--mean_mask_length', type=int, default=60,
                       help='Mean mask length for random masking')

    # Model config overrides
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Override hidden size from checkpoint')
    parser.add_argument('--encoder_layers', type=int, default=None,
                       help='Override encoder layers from checkpoint')
    parser.add_argument('--decoder_layers', type=int, default=None,
                       help='Override decoder layers from checkpoint')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Override num heads from checkpoint')

    # Output
    parser.add_argument('--output_dir', type=str, default='./forecasting_eval_results',
                       help='Output directory for results')
    parser.add_argument('--by_aircraft', action='store_true',
                       help='Also report results broken down by aircraft type')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build config override
    config_override = {}
    if args.hidden_size:
        config_override['hidden_size'] = args.hidden_size
    if args.encoder_layers:
        config_override['encoder_layers'] = args.encoder_layers
    if args.decoder_layers:
        config_override['decoder_layers'] = args.decoder_layers
    if args.num_heads:
        config_override['num_heads'] = args.num_heads

    # Load model
    model, config, feat_dim = load_bert_model(
        args.checkpoint, device, config_override if config_override else None
    )

    # Load data
    data_list, flight_ids, sequence_lengths, aircraft_types = load_flight_data(
        args.data_dir, args.split, args.max_files
    )

    print(f"Loaded {len(data_list)} flights")
    print(f"Aircraft types: {set(aircraft_types)}")

    # Load or compute normalization params
    if args.norm_params_path:
        normalization_params = np.load(args.norm_params_path, allow_pickle=True).item()
        print(f"Loaded normalization params from {args.norm_params_path}")
    else:
        print("Computing normalization parameters from data...")
        normalization_params = compute_normalization_params(data_list)

    all_results = {}

    # Forecasting evaluation
    if args.eval_mode in ['forecast', 'both']:
        print("\n" + "=" * 70)
        print("FORECASTING EVALUATION (End Masking)")
        print("=" * 70)

        forecast_results = {}

        for ratio in args.forecast_ratios:
            print(f"\n--- Forecast Ratio: {ratio} ({ratio*100:.0f}% of flight) ---")

            metrics, per_flight = evaluate_forecasting(
                model, data_list, flight_ids, sequence_lengths,
                normalization_params, forecast_ratio=ratio,
                min_horizon=args.min_horizon, device=device
            )

            forecast_results[f'ratio_{ratio}'] = metrics

            print(f"\nResults for {ratio*100:.0f}% forecast:")
            print(f"  MSE (overall):       {metrics['mse_overall']:.6f}")
            print(f"  MAE (overall):       {metrics['mae_overall']:.6f}")
            print(f"  RMSE (overall):      {metrics['rmse_overall']:.6f}")
            print(f"  MSE (forecast only): {metrics['mse_forecast_only']:.6f}")
            print(f"  MAE (forecast only): {metrics['mae_forecast_only']:.6f}")
            print(f"  RMSE (forecast only):{metrics['rmse_forecast_only']:.6f}")

        all_results['forecasting'] = forecast_results

        # Per-aircraft evaluation
        if args.by_aircraft:
            print("\n" + "-" * 70)
            print("Results by Aircraft Type")
            print("-" * 70)

            for ratio in args.forecast_ratios:
                print(f"\n--- Forecast Ratio: {ratio} ---")
                by_type = evaluate_by_aircraft_type(
                    model, data_list, flight_ids, sequence_lengths, aircraft_types,
                    normalization_params, ratio, args.min_horizon, device
                )

                for atype, metrics in by_type.items():
                    print(f"  {atype}: MSE={metrics['mse_forecast_only']:.6f}, "
                          f"MAE={metrics['mae_forecast_only']:.6f}")

                all_results[f'forecasting_by_aircraft_ratio_{ratio}'] = by_type

    # Random masking evaluation
    if args.eval_mode in ['random', 'both']:
        print("\n" + "=" * 70)
        print("RANDOM MASKING EVALUATION")
        print("=" * 70)

        metrics = evaluate_random_masking(
            model, data_list, flight_ids, sequence_lengths,
            normalization_params, masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length, device=device
        )

        all_results['random_masking'] = metrics

        print(f"\nResults for random masking (ratio={args.masking_ratio}):")
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")

    # Summary comparison
    if args.eval_mode == 'both':
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        print(f"{'Task':<30} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
        print("-" * 70)

        # Random masking
        rm = all_results['random_masking']
        print(f"{'Random Masking':<30} {rm['mse']:<12.6f} {rm['mae']:<12.6f} {rm['rmse']:<12.6f}")

        # Forecasting at each ratio
        for ratio in args.forecast_ratios:
            fm = all_results['forecasting'][f'ratio_{ratio}']
            label = f"Forecast {ratio*100:.0f}% (overall)"
            print(f"{label:<30} {fm['mse_overall']:<12.6f} {fm['mae_overall']:<12.6f} {fm['rmse_overall']:<12.6f}")
            label = f"Forecast {ratio*100:.0f}% (masked only)"
            print(f"{label:<30} {fm['mse_forecast_only']:<12.6f} {fm['mae_forecast_only']:<12.6f} {fm['rmse_forecast_only']:<12.6f}")

    # Save results (convert numpy types to native Python for JSON serialization)
    output_file = os.path.join(args.output_dir, f"eval_results_{args.split}.json")
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save config
    config_file = os.path.join(args.output_dir, "eval_config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
