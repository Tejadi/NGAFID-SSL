#!/usr/bin/env python3
"""
Evaluation script for PatchTST masked regressor on the test split.
Evaluates at masking ratios [0.2, 0.5, 0.8] with mean_mask_length=3,
matching the protocol used for BERT/LSTM/MLP baselines.
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from models.patchtst_masked_regressor import PatchTSTMaskedRegressor
from train_full_flights import compute_normalization_parameters
from ngafid_datasets.transformation_dataset import mask_transform


MASKING_RATIO = 0.6
EVAL_MEAN_MASK_LENGTHS = [5, 60]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchTST Masked Regressor on test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--local_data_dir", type=str, required=True,
                        help="Directory containing flight CSV files")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max test files to evaluate (None for all)")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
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


def load_test_flights(data_dir, seq_len, train_split, val_split, max_files, seed=0):
    """Load test split CSV files and return fixed-length windows.
    Mirrors LocalFlightDataset._find_csv_files / _split_files logic exactly."""
    import pandas as pd

    data_path = Path(data_dir)

    # Mirror LocalFlightDataset: prefer preprocessed_data/test/ directory
    preprocessed_path = data_path / "preprocessed_data"
    if preprocessed_path.exists():
        test_path = preprocessed_path / "test"
        if test_path.exists():
            test_files = sorted(test_path.glob("*.csv"))
            if max_files is not None:
                test_files = test_files[:max_files]
            print(f"Loading {len(test_files)} test files from {test_path}...")
        else:
            test_files = []
    else:
        # Fallback: manual split on flat directory
        all_files = sorted([
            f for f in data_path.glob("*.csv")
            if not any(name in f.name.lower()
                       for name in ['aircraft_types', 'events', 'flight_ids', 'splits', 'sequence_length'])
        ])
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        n_train = int(len(all_files) * train_split)
        n_val = int(len(all_files) * val_split)
        test_files = [all_files[i] for i in indices[n_train + n_val:]]
        if max_files is not None:
            test_files = test_files[:max_files]
        print(f"Loading {len(test_files)} test files...")

    windows = []
    for f in tqdm(test_files, desc="Loading test data"):
        try:
            df = pd.read_csv(f, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
            numeric = df.select_dtypes(include=[np.number])
            arr = numeric.ffill().bfill().to_numpy(dtype=np.float32)
            for start in range(0, len(arr) - seq_len + 1, seq_len):
                windows.append(arr[start:start + seq_len])
        except Exception as e:
            print(f"  Warning: {f.name}: {e}")

    return np.stack(windows)  # (N, seq_len, feat_dim)


def evaluate_at_mask_length(model, data_normalized, mean, std, mean_mask_length, device, batch_size):
    """Evaluate model at a single mean_mask_length with fixed ratio=0.6. Matches BERT eval protocol."""
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_orig = 0.0
    total_mae_orig = 0.0
    total_masked = 0

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    n = len(data_normalized)
    for i in tqdm(range(0, n, batch_size), desc=f"  mask_len={mean_mask_length}"):
        batch = data_normalized[i:i + batch_size]

        masked_batch = []
        mask_batch = []
        for j, seq in enumerate(batch):
            _, masked_seq, mask = mask_transform(
                seq,
                masking_ratio=MASKING_RATIO,
                mean_mask_length=mean_mask_length,
                mode='separate',
                distribution='geometric',
                random_seed=i + j,
            )
            masked_batch.append(masked_seq.numpy())
            mask_batch.append(mask.numpy())

        x_masked = torch.tensor(np.stack(masked_batch), dtype=torch.float32, device=device)
        x_original = torch.tensor(batch, dtype=torch.float32, device=device)
        masks = torch.tensor(np.stack(mask_batch), dtype=torch.float32, device=device)

        with torch.no_grad():
            reconstructed = model(x_masked)

        masked_pos = (masks == 0).float()
        n_masked = masked_pos.sum().item()

        total_mse_norm += ((reconstructed - x_original) ** 2 * masked_pos).sum().item()
        total_mae_norm += (torch.abs(reconstructed - x_original) * masked_pos).sum().item()

        recon_orig = reconstructed * std_t + mean_t
        x_orig_scale = x_original * std_t + mean_t
        total_mse_orig += ((recon_orig - x_orig_scale) ** 2 * masked_pos).sum().item()
        total_mae_orig += (torch.abs(recon_orig - x_orig_scale) * masked_pos).sum().item()

        total_masked += n_masked

    denom = total_masked if total_masked > 0 else float('inf')
    return {
        "masking_ratio": MASKING_RATIO,
        "mean_mask_length": mean_mask_length,
        "masked_positions": int(total_masked),
        "normalized": {
            "mse": total_mse_norm / denom,
            "rmse": (total_mse_norm / denom) ** 0.5,
            "mae": total_mae_norm / denom,
        },
        "original_scale": {
            "mse": total_mse_orig / denom,
            "rmse": (total_mse_orig / denom) ** 0.5,
            "mae": total_mae_orig / denom,
        },
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    model, train_args = load_model(args.checkpoint, device)

    print("Computing normalization parameters...")
    normalization_params = compute_normalization_parameters(args.local_data_dir, max_files=500)
    mean = normalization_params['mean']
    std = normalization_params['std']

    raw_data = load_test_flights(
        args.local_data_dir, args.seq_len,
        args.train_split, args.val_split, args.max_files,
    )
    print(f"Test windows: {raw_data.shape}")

    # Normalize once, same as BERT eval
    data_normalized = (raw_data - mean) / std

    all_results = []
    for mask_len in EVAL_MEAN_MASK_LENGTHS:
        print(f"\nEvaluating masking_ratio={MASKING_RATIO}, mean_mask_length={mask_len}...")
        result = evaluate_at_mask_length(model, data_normalized, mean, std, mask_len, device, args.batch_size)
        all_results.append(result)

        print(f"  Normalized  — MSE: {result['normalized']['mse']:.6f}  "
              f"RMSE: {result['normalized']['rmse']:.6f}  "
              f"MAE: {result['normalized']['mae']:.6f}")
        print(f"  Orig scale  — MSE: {result['original_scale']['mse']:.6f}  "
              f"RMSE: {result['original_scale']['rmse']:.6f}  "
              f"MAE: {result['original_scale']['mae']:.6f}")

    print("\n=== Summary (masking_ratio=0.6) ===")
    print(f"{'MaskLen':<10} {'Norm MSE':<14} {'Norm MAE':<14} {'Orig MSE':<14} {'Orig MAE'}")
    for r in all_results:
        print(f"{r['mean_mask_length']:<10} "
              f"{r['normalized']['mse']:<14.6f} "
              f"{r['normalized']['mae']:<14.6f} "
              f"{r['original_scale']['mse']:<14.6f} "
              f"{r['original_scale']['mae']:.6f}")

    output = {
        "checkpoint": args.checkpoint,
        "total_test_windows": len(raw_data),
        "masking_ratio": MASKING_RATIO,
        "results_by_mean_mask_length": all_results,
    }

    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
