#!/usr/bin/env python3
"""
Evaluation script for PatchTST forecaster on the test split.
Evaluates at forecast ratios [0.1, 0.2, 0.3], matching the BERT forecasting
eval protocol (end-of-sequence masking, normalized MSE/MAE).
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from models.patchtst_masked_regressor import PatchTSTMaskedRegressor
from train_full_flights import compute_normalization_parameters


EVAL_FORECAST_RATIOS = [0.1, 0.2, 0.3]
MIN_FORECAST_HORIZON = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchTST Forecaster on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory (containing preprocessed_data/test/)")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_model(checkpoint_path, device):
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

    print(f"Loaded checkpoint from step {checkpoint['global_step']} "
          f"(eval_loss: {checkpoint.get('eval_loss', 'N/A')})")
    return model, saved_args


def load_test_flights(data_dir, seq_len, max_files):
    """Load test flights, returning (padded_array, sequence_lengths).
    Mirrors ForecastFlightDataset file discovery logic.
    """
    data_path = Path(data_dir)
    preprocessed_path = data_path / "preprocessed_data"

    if preprocessed_path.exists():
        test_path = preprocessed_path / "test"
        if test_path.exists():
            csv_files = sorted(test_path.glob("*.csv"))
        else:
            csv_files = sorted([
                f for f in data_path.glob("*.csv")
                if not any(n in f.name.lower()
                           for n in ['aircraft_types', 'events', 'flight_ids', 'splits'])
            ])
    else:
        csv_files = sorted([
            f for f in data_path.glob("*.csv")
            if not any(n in f.name.lower()
                       for n in ['aircraft_types', 'events', 'flight_ids', 'splits'])
        ])

    if max_files is not None:
        csv_files = csv_files[:max_files]

    print(f"Loading {len(csv_files)} test files...")

    flights = []
    sequence_lengths = []

    for f in tqdm(csv_files, desc="Loading test flights"):
        try:
            df = pd.read_csv(f, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
            arr = df.select_dtypes(include=[np.number]).ffill().bfill().to_numpy(dtype=np.float32)
            if arr.shape[0] == 0:
                continue

            original_length = min(arr.shape[0], seq_len)
            sequence_lengths.append(original_length)

            if arr.shape[0] < seq_len:
                padding = np.repeat(arr[-1:], seq_len - arr.shape[0], axis=0)
                arr = np.vstack([arr, padding])
            else:
                arr = arr[:seq_len]

            flights.append(arr)
        except Exception as e:
            print(f"  Warning: {f.name}: {e}")

    return np.stack(flights), np.array(sequence_lengths)


def forecast_mask(seq_len_total, original_length, forecast_ratio, feat_dim, min_horizon=50):
    """Create end-of-sequence forecast mask. Matches eval_forecasting.py protocol."""
    horizon = max(min_horizon, int(original_length * forecast_ratio))
    max_horizon = original_length // 2
    horizon = min(horizon, max_horizon)
    horizon = max(horizon, 1)

    mask = np.ones((seq_len_total, feat_dim), dtype=np.float32)
    forecast_start = original_length - horizon
    mask[forecast_start:original_length, :] = 0.0
    return mask, horizon


def evaluate_at_ratio(model, data_normalized, sequence_lengths, mean, std,
                      forecast_ratio, device, batch_size, seq_len, feat_dim):
    """Evaluate at a single forecast ratio. Matches eval_forecasting.py metric computation."""
    total_forecast_mse_norm = 0.0
    total_forecast_mae_norm = 0.0
    total_forecast_mse_orig = 0.0
    total_forecast_mae_orig = 0.0
    total_forecast_positions = 0

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    n = len(data_normalized)
    for i in tqdm(range(0, n, batch_size), desc=f"  ratio={forecast_ratio}"):
        batch_norm = data_normalized[i:i + batch_size]
        batch_seq_lens = sequence_lengths[i:i + batch_size]
        batch_size_actual = len(batch_norm)

        batch_masks = []
        for j in range(batch_size_actual):
            mask, _ = forecast_mask(seq_len, batch_seq_lens[j], forecast_ratio, feat_dim, MIN_FORECAST_HORIZON)
            batch_masks.append(mask)

        masks_np = np.stack(batch_masks)
        mask_t = torch.tensor(masks_np, dtype=torch.float32, device=device)

        x_original = torch.tensor(batch_norm, dtype=torch.float32, device=device)
        x_masked = x_original * mask_t

        with torch.no_grad():
            reconstructed = model(x_masked)

        # Forecast positions only (mask == 0)
        forecast_pos = (mask_t == 0).float()
        n_forecast = forecast_pos.sum().item()

        if n_forecast > 0:
            total_forecast_mse_norm += ((reconstructed - x_original) ** 2 * forecast_pos).sum().item()
            total_forecast_mae_norm += (torch.abs(reconstructed - x_original) * forecast_pos).sum().item()

            recon_orig = reconstructed * std_t + mean_t
            x_orig_scale = x_original * std_t + mean_t
            total_forecast_mse_orig += ((recon_orig - x_orig_scale) ** 2 * forecast_pos).sum().item()
            total_forecast_mae_orig += (torch.abs(recon_orig - x_orig_scale) * forecast_pos).sum().item()

        total_forecast_positions += n_forecast

    denom = total_forecast_positions if total_forecast_positions > 0 else float('inf')
    return {
        "forecast_ratio": forecast_ratio,
        "forecast_positions": int(total_forecast_positions),
        "normalized": {
            "mse": total_forecast_mse_norm / denom,
            "rmse": (total_forecast_mse_norm / denom) ** 0.5,
            "mae": total_forecast_mae_norm / denom,
        },
        "original_scale": {
            "mse": total_forecast_mse_orig / denom,
            "rmse": (total_forecast_mse_orig / denom) ** 0.5,
            "mae": total_forecast_mae_orig / denom,
        },
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    model, train_args = load_model(args.checkpoint, device)
    feat_dim = train_args['feat_dim']

    print("Computing normalization parameters...")
    normalization_params = compute_normalization_parameters(args.data_dir, max_files=500)
    mean = normalization_params['mean']
    std = normalization_params['std']

    raw_data, sequence_lengths = load_test_flights(args.data_dir, args.seq_len, args.max_files)
    print(f"Test flights: {raw_data.shape}, seq_lens range: [{sequence_lengths.min()}, {sequence_lengths.max()}]")

    # Normalize once
    data_normalized = (raw_data - mean) / std

    all_results = []
    for ratio in EVAL_FORECAST_RATIOS:
        print(f"\nEvaluating forecast_ratio={ratio}...")
        result = evaluate_at_ratio(
            model, data_normalized, sequence_lengths, mean, std,
            ratio, device, args.batch_size, args.seq_len, feat_dim,
        )
        all_results.append(result)
        print(f"  Normalized  — MSE: {result['normalized']['mse']:.6f}  "
              f"RMSE: {result['normalized']['rmse']:.6f}  "
              f"MAE: {result['normalized']['mae']:.6f}")
        print(f"  Orig scale  — MSE: {result['original_scale']['mse']:.6f}  "
              f"RMSE: {result['original_scale']['rmse']:.6f}  "
              f"MAE: {result['original_scale']['mae']:.6f}")

    print("\n=== Summary ===")
    print(f"{'Ratio':<8} {'Norm MSE':<14} {'Norm MAE':<14} {'Orig MSE':<14} {'Orig MAE'}")
    for r in all_results:
        print(f"{r['forecast_ratio']:<8} "
              f"{r['normalized']['mse']:<14.6f} "
              f"{r['normalized']['mae']:<14.6f} "
              f"{r['original_scale']['mse']:<14.6f} "
              f"{r['original_scale']['mae']:.6f}")

    output = {
        "checkpoint": args.checkpoint,
        "total_test_flights": len(raw_data),
        "results_by_forecast_ratio": all_results,
    }

    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_forecast_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
