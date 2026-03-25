#!/usr/bin/env python3
"""
Training script for MLP baseline on masked regression.

This trains a simple MLP that has NO temporal context - serving as
a lower bound baseline for the temporal models (LSTM, BERT).
"""

import argparse
import os
import time
import json
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - skipping W&B logging")

from models.mlp_baseline import MLPBaseline, count_parameters
from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP Baseline")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing flight data")
    parser.add_argument("--seq_len", type=int, default=10000,
                        help="Sequence length")
    parser.add_argument("--max_files_train", type=int, default=None,
                        help="Maximum training files")
    parser.add_argument("--max_files_val", type=int, default=None,
                        help="Maximum validation files")

    # Model arguments
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[256, 512, 256],
                        help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluation interval (steps)")

    # Masking arguments
    parser.add_argument("--masking_ratio", type=float, default=0.15,
                        help="Masking ratio")
    parser.add_argument("--mean_mask_length", type=int, default=10,
                        help="Mean mask length")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./baseline_runs/mlp",
                        help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="ngafid-baselines",
                        help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def create_dataloader(data_dir, split, batch_size, seq_len, masking_ratio,
                      mean_mask_length, normalization_params, max_files=None,
                      num_workers=2, seed=42, use_random_masking=False,
                      masking_ratios=None, mean_mask_lengths=None):
    """Create dataloader for training or validation.

    LocalFlightDataset expects data_dir to be the PARENT of preprocessed_data/.
    If the user passes preprocessed_data/ directly, we go up one level.
    """
    # LocalFlightDataset looks for data_dir/preprocessed_data/train|val|test
    # If data_dir itself ends with preprocessed_data, go up one level
    root_dir = data_dir
    if os.path.basename(os.path.normpath(data_dir)) == "preprocessed_data":
        root_dir = os.path.dirname(os.path.normpath(data_dir))

    # Map split names to what LocalFlightDataset expects
    split_name = "validation" if split == "val" else split

    dataset = GlobalNormalizedFlightDataset(
        data_dir=root_dir,
        split=split_name,
        seq_len=seq_len,
        masking_ratio=masking_ratio,
        mean_mask_length=mean_mask_length,
        normalization_params=normalization_params,
        max_files=max_files,
        seed=seed,
        use_random_masking=use_random_masking,
        masking_ratios=masking_ratios or [0.2, 0.5, 0.8],
        mean_mask_lengths=mean_mask_lengths or [5, 60],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, dataset


def evaluate(model, dataloader, device, max_batches=50):
    """Evaluate model on validation data."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            _, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)

            total_mse += mse_loss.item() * x_masked.size(0)
            total_mae += mae_loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)

    return {
        "eval_mse": total_mse / total_samples if total_samples > 0 else float('inf'),
        "eval_mae": total_mae / total_samples if total_samples > 0 else float('inf'),
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MLP Baseline Training")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"mlp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Compute normalization parameters
    print("Computing normalization parameters...")
    train_dir = os.path.join(args.data_dir, "train")
    normalization_params = compute_normalization_parameters(
        train_dir if os.path.exists(train_dir) else args.data_dir,
        max_files=args.max_files_train or 500
    )

    # Get feature dimension
    import pandas as pd
    import glob
    sample_files = glob.glob(os.path.join(
        train_dir if os.path.exists(train_dir) else args.data_dir,
        "*.csv"
    ))[:1]
    if sample_files:
        df = pd.read_csv(sample_files[0])
        feat_dim = len([c for c in df.columns if c not in ['time', 'timestamp', 'Time']])
    else:
        feat_dim = 44
    print(f"Feature dimension: {feat_dim}")

    # Create dataloaders
    # Training uses random masking (same as BERT) for fair comparison
    # Validation uses fixed masking_ratio=0.6, mean_mask_length=3 (same as BERT eval)
    print("Creating dataloaders...")
    train_loader, train_dataset = create_dataloader(
        args.data_dir, "train", args.batch_size, args.seq_len,
        args.masking_ratio, args.mean_mask_length, normalization_params,
        max_files=args.max_files_train, seed=args.seed,
        use_random_masking=True,
    )

    val_loader, _ = create_dataloader(
        args.data_dir, "val", args.batch_size, args.seq_len,
        0.6, 3, normalization_params,
        max_files=args.max_files_val, seed=args.seed + 1,
        use_random_masking=False,
    )

    print(f"Train samples: {len(train_dataset)}")

    # Create model
    model = MLPBaseline(
        feat_dim=feat_dim,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"mlp_{timestamp}",
            config={
                **vars(args),
                "model": "MLP",
                "feat_dim": feat_dim,
                "num_params": count_parameters(model),
            }
        )

    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_eval_mse = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_mse = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (x_masked, x_original, mask) in enumerate(pbar):
            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_mse += mse_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            pbar.set_postfix({"mse": f"{mse_loss.item():.4f}"})

            if use_wandb:
                wandb.log({
                    "train/mse": mse_loss.item(),
                    "train/mae": mae_loss.item(),
                }, step=global_step)

            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_metrics = evaluate(model, val_loader, device)
                print(f"\nStep {global_step}: Eval MSE = {eval_metrics['eval_mse']:.4f}")

                if use_wandb:
                    wandb.log({
                        "eval/mse": eval_metrics["eval_mse"],
                        "eval/mae": eval_metrics["eval_mae"],
                    }, step=global_step)

                if eval_metrics["eval_mse"] < best_eval_mse:
                    best_eval_mse = eval_metrics["eval_mse"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "eval_mse": best_eval_mse,
                        "config": vars(args),
                        "feat_dim": feat_dim,
                    }, os.path.join(output_dir, "best_model.pt"))
                    print(f"Saved best model (MSE: {best_eval_mse:.4f})")

                model.train()

        avg_epoch_mse = epoch_mse / epoch_samples
        print(f"Epoch {epoch+1} - Avg MSE: {avg_epoch_mse:.4f}, Best Eval: {best_eval_mse:.4f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "config": vars(args),
        "feat_dim": feat_dim,
    }, os.path.join(output_dir, "final_model.pt"))

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print(f"Training complete! Best eval MSE: {best_eval_mse:.4f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
