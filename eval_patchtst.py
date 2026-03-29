#!/usr/bin/env python3
"""
Evaluation script for PatchTST masked regressor on the test split.
Reports MSE and MAE on masked positions.
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from models.patchtst_masked_regressor import PatchTSTMaskedRegressor
from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters


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
    parser.add_argument("--masking_ratio", type=float, default=0.5,
                        help="Masking ratio for evaluation")
    parser.add_argument("--mean_mask_length", type=int, default=30,
                        help="Mean mask length for evaluation")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    model = PatchTSTMaskedRegressor(
        feat_dim=args['feat_dim'],
        seq_len=args['seq_len'],
        patch_len=args['patch_len'],
        stride=args['stride'],
        d_model=args['d_model'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        encoder_layers=args['encoder_layers'],
        decoder_layers=args['decoder_layers'],
        dropout=0.0,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['global_step']} "
          f"(eval_loss: {checkpoint.get('eval_loss', 'N/A')})")
    return model, args


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model, train_args = load_model(args.checkpoint, device)

    print("Computing normalization parameters...")
    normalization_params = compute_normalization_parameters(
        args.local_data_dir,
        max_files=500,
    )

    test_dataset = GlobalNormalizedFlightDataset(
        normalization_params=normalization_params,
        data_dir=args.local_data_dir,
        split="test",
        seq_len=args.seq_len,
        max_files=args.max_files,
        seed=0,
        train_split=0.8,
        val_split=0.1,
        use_random_masking=False,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # For denormalization
    mean = torch.tensor(normalization_params['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(normalization_params['std'], dtype=torch.float32, device=device)

    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_orig = 0.0
    total_mae_orig = 0.0
    total_masked = 0
    total_samples = 0

    with torch.no_grad():
        for x_masked, x_original, mask in tqdm(test_loader, desc="Evaluating"):
            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            reconstructed = model(x_masked)

            masked_positions = (mask == 0).float()
            n_masked = masked_positions.sum().item()

            # Normalized space metrics
            total_mse_norm += ((reconstructed - x_original) ** 2 * masked_positions).sum().item()
            total_mae_norm += (torch.abs(reconstructed - x_original) * masked_positions).sum().item()

            # Denormalize and compute in original scale
            recon_orig = reconstructed * std + mean
            x_orig_scale = x_original * std + mean
            total_mse_orig += ((recon_orig - x_orig_scale) ** 2 * masked_positions).sum().item()
            total_mae_orig += (torch.abs(recon_orig - x_orig_scale) * masked_positions).sum().item()

            total_masked += n_masked
            total_samples += x_masked.size(0)

    n = total_masked if total_masked > 0 else float('inf')
    avg_mse_norm = total_mse_norm / n
    avg_mae_norm = total_mae_norm / n
    avg_rmse_norm = avg_mse_norm ** 0.5
    avg_mse_orig = total_mse_orig / n
    avg_mae_orig = total_mae_orig / n
    avg_rmse_orig = avg_mse_orig ** 0.5

    print("\n=== Test Set Results ===")
    print(f"Samples evaluated     : {total_samples}")
    print(f"Masked positions      : {int(total_masked)}")
    print(f"--- Normalized space (z-scored) ---")
    print(f"MSE  : {avg_mse_norm:.6f}")
    print(f"RMSE : {avg_rmse_norm:.6f}")
    print(f"MAE  : {avg_mae_norm:.6f}")
    print(f"--- Original scale (sensor units) ---")
    print(f"MSE  : {avg_mse_orig:.6f}")
    print(f"RMSE : {avg_rmse_orig:.6f}")
    print(f"MAE  : {avg_mae_orig:.6f}")

    results = {
        "checkpoint": args.checkpoint,
        "samples": total_samples,
        "masked_positions": int(total_masked),
        "masking_ratio": args.masking_ratio,
        "mean_mask_length": args.mean_mask_length,
        "normalized": {
            "mse": avg_mse_norm,
            "rmse": avg_rmse_norm,
            "mae": avg_mae_norm,
        },
        "original_scale": {
            "mse": avg_mse_orig,
            "rmse": avg_rmse_orig,
            "mae": avg_mae_orig,
        },
    }

    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
