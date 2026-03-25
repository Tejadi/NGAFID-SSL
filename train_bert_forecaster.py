#!/usr/bin/env python3
"""
Training script for BERT-based flight forecasting.

This script trains a BERT model for next-token prediction on flight data.
Unlike masked regression (which uses random masking), forecasting masks
the END of each flight, creating a causal prediction task.

The model learns to predict future flight states given past context,
which is useful for:
- Flight trajectory prediction
- Anomaly detection (comparing predicted vs actual)
- Pre-training for downstream tasks
"""

import argparse
import os
import time
import json
import gc
from typing import Dict, Any, Optional, List

# Set CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - skipping W&B logging")

# Import models and datasets
try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
    from ngafid_datasets.forecast_flight_dataset import ForecastFlightDataset, create_forecast_dataloader
    from ngafid_datasets.local_flight_dataset import get_feature_dim_from_local_data
    from train_full_flights import compute_normalization_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BERT Forecaster for Flight Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/oscar/data/sbach/shared/ngafid",
                        help="Directory containing flight CSV files")
    parser.add_argument("--seq_len", type=int, default=10000,
                        help="Sequence length (flights padded to this)")
    parser.add_argument("--max_files_train", type=int, default=None,
                        help="Maximum training files")
    parser.add_argument("--max_files_val", type=int, default=100,
                        help="Maximum validation files")

    # Forecasting arguments
    parser.add_argument("--forecast_ratios", type=float, nargs='+', default=[0.1, 0.2, 0.3],
                        help="List of forecast ratios to randomly sample from")
    parser.add_argument("--fixed_forecast_ratio", type=float, default=None,
                        help="Use a single fixed forecast ratio instead of random sampling")
    parser.add_argument("--min_forecast_horizon", type=int, default=100,
                        help="Minimum timesteps to predict")
    parser.add_argument("--max_forecast_horizon", type=int, default=None,
                        help="Maximum timesteps to predict (None for unlimited)")

    # Model arguments (matching train_full_flights.py for fair comparison)
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="BERT hidden size")
    parser.add_argument("--encoder_layers", type=int, default=8,
                        help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=6,
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=16,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluation interval (steps)")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Checkpoint save interval (steps)")
    parser.add_argument("--max_checkpoints", type=int, default=3,
                        help="Maximum number of step checkpoints to keep (best_model.pt always kept)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Logging interval (steps)")

    # System arguments
    parser.add_argument("--num_workers", type=int, default=1,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./bert_forecaster_runs",
                        help="Output directory")
    parser.add_argument("--job_name", type=str, default=None,
                        help="Job name for this run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="bert-flight-forecaster",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # Memory optimization
    parser.add_argument("--use_mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")

    return parser.parse_args()


def create_lr_scheduler(optimizer, num_training_steps, warmup_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, max_batches: int = 50) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_samples = 0
    total_forecast_positions = 0

    with torch.no_grad():
        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)

            # Count forecast positions (mask == 0)
            forecast_positions = (mask == 0).sum().item()

            total_loss += loss.item() * x_masked.size(0)
            total_mse_loss += mse_loss.item() * x_masked.size(0)
            total_mae_loss += mae_loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)
            total_forecast_positions += forecast_positions

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse_loss = total_mse_loss / total_samples if total_samples > 0 else float('inf')
    avg_mae_loss = total_mae_loss / total_samples if total_samples > 0 else float('inf')
    avg_forecast_positions = total_forecast_positions / total_samples if total_samples > 0 else 0

    return {
        "eval_loss": avg_loss,
        "eval_mse_loss": avg_mse_loss,
        "eval_mae_loss": avg_mae_loss,
        "eval_forecast_positions": avg_forecast_positions,
    }


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("BERT Flight Forecaster Training")
    print("=" * 70)
    if args.fixed_forecast_ratio is not None:
        print(f"Forecast ratio (fixed): {args.fixed_forecast_ratio} (predict last {args.fixed_forecast_ratio*100:.0f}% of each flight)")
    else:
        print(f"Forecast ratios (random): {args.forecast_ratios}")
        print(f"  Will randomly sample from: {[f'{r*100:.0f}%' for r in args.forecast_ratios]}")
    print(f"Min forecast horizon: {args.min_forecast_horizon} timesteps")
    print(f"Max forecast horizon: {args.max_forecast_horizon or 'unlimited'}")
    print()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print()

    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        exit(1)

    # Get feature dimension
    print("Detecting feature dimension...")
    try:
        feat_dim = get_feature_dim_from_local_data(args.data_dir)
        print(f"Detected feature dimension: {feat_dim}")
    except Exception as e:
        print(f"Error detecting feature dimension: {e}")
        feat_dim = 44
        print(f"Using default: {feat_dim}")

    # Compute normalization parameters
    print("Computing normalization parameters...")
    try:
        normalization_params = compute_normalization_parameters(
            args.data_dir, max_files=args.max_files_train
        )
    except Exception as e:
        print(f"Error computing normalization: {e}")
        normalization_params = None

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or f"forecaster_r{args.forecast_ratios}_{timestamp}"
    output_dir = os.path.join(args.output_dir, job_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output directory: {output_dir}")

    # Create data loaders
    print("Creating data loaders...")

    # Determine forecast ratio settings
    if args.fixed_forecast_ratio is not None:
        forecast_ratios = None
        forecast_ratio = args.fixed_forecast_ratio
    else:
        forecast_ratios = args.forecast_ratios
        forecast_ratio = 0.2  # Fallback, won't be used

    train_loader = create_forecast_dataloader(
        data_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        forecast_ratios=forecast_ratios,
        forecast_ratio=forecast_ratio,
        min_forecast_horizon=args.min_forecast_horizon,
        max_forecast_horizon=args.max_forecast_horizon,
        normalization_params=normalization_params,
        max_files=args.max_files_train,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Validation data
    val_data_dir = os.path.join(args.data_dir, "preprocessed_data", "val")
    if not os.path.exists(val_data_dir):
        val_data_dir = args.data_dir
        val_split = "val"
    else:
        val_split = "train"  # Use all files in val directory

    val_loader = create_forecast_dataloader(
        data_dir=val_data_dir,
        split=val_split,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        forecast_ratios=forecast_ratios,
        forecast_ratio=forecast_ratio,
        min_forecast_horizon=args.min_forecast_horizon,
        max_forecast_horizon=args.max_forecast_horizon,
        normalization_params=normalization_params,
        max_files=args.max_files_val,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )

    print(f"Train batches: ~{len(train_loader)}")

    # Create model
    print("Creating model...")
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_mixed_precision=args.use_mixed_precision,
    ).to(device)

    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # Setup optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("Using standard AdamW optimizer")

    # Setup scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = create_lr_scheduler(optimizer, total_steps, args.warmup_steps)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_mixed_precision else None

    print(f"\nTraining configuration:")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print()

    # Setup logging
    writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=job_name,
                config={
                    **vars(args),
                    "feat_dim": feat_dim,
                    "total_params": total_params,
                    "task": "forecasting",
                    "use_random_forecast_ratio": args.fixed_forecast_ratio is None,
                    "forecast_ratios_str": str(args.forecast_ratios) if args.fixed_forecast_ratio is None else str(args.fixed_forecast_ratio),
                }
            )
            print(f"W&B tracking: {wandb.run.url}")
        except Exception as e:
            print(f"W&B setup failed: {e}")
            use_wandb = False

    # Training loop
    print("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print(f"  Resumed at epoch {start_epoch}, global_step {global_step}")
        else:
            print(f"Warning: checkpoint not found at {args.resume}, starting fresh")

    # Checkpoints directory
    checkpoint_dir = os.path.join(
        "./checkpoints",
        f"bert_forecaster_{timestamp}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=True,
            dynamic_ncols=True
        )

        accumulation_step = 0
        optimizer.zero_grad()

        for batch_idx, (x_masked, x_original, mask) in enumerate(pbar):
            x_masked = x_masked.to(device, non_blocking=True)
            x_original = x_original.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Forward pass
            if args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, mse_loss, mae_loss = model.compute_loss(
                        x_masked, x_original, mask, scaler=scaler
                    )
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            accumulation_step += 1

            # Optimizer step
            if accumulation_step % args.gradient_accumulation_steps == 0:
                if args.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Update metrics
            actual_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += actual_loss * x_masked.size(0)
            epoch_mse_loss += mse_loss.item() * x_masked.size(0)
            epoch_mae_loss += mae_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Logging - log every step for better wandb graphs
            if accumulation_step % args.gradient_accumulation_steps == 0 and global_step % args.log_interval == 0:
                forecast_positions = (mask == 0).sum().item()
                total_positions = mask.numel()
                effective_forecast_ratio = forecast_positions / total_positions
                current_lr = scheduler.get_last_lr()[0]
                avg_train_loss = epoch_loss / (epoch_samples + 1e-8)
                avg_train_mse = epoch_mse_loss / (epoch_samples + 1e-8)

                writer.add_scalar("train/loss", actual_loss, global_step)
                writer.add_scalar("train/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("train/mae_loss", mae_loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/forecast_positions", forecast_positions, global_step)
                writer.add_scalar("train/forecast_ratio", effective_forecast_ratio, global_step)

                if use_wandb:
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/mse_loss": mse_loss.item(),
                        "train/mae_loss": mae_loss.item(),
                        "train/avg_loss": avg_train_loss,
                        "train/avg_mse": avg_train_mse,
                        "train/lr": current_lr,
                        "train/forecast_positions": forecast_positions,
                        "train/forecast_ratio": effective_forecast_ratio,
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                    }, step=global_step)

            # Progress bar
            pbar.set_postfix({
                "mse": f"{mse_loss.item():.4f}",
                "mae": f"{mae_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            # Evaluation
            if (accumulation_step % args.gradient_accumulation_steps == 0 and
                global_step % args.eval_interval == 0 and global_step > 0):

                pbar.write(f"\nEvaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_loader, device)

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                writer.add_scalar("eval/mse_loss", eval_metrics["eval_mse_loss"], global_step)
                writer.add_scalar("eval/mae_loss", eval_metrics["eval_mae_loss"], global_step)

                if use_wandb:
                    # Log eval metrics with multiple aliases for easy comparison
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/mse_loss": eval_metrics["eval_mse_loss"],
                        "eval/mae_loss": eval_metrics["eval_mae_loss"],
                        "eval/forecast_positions": eval_metrics["eval_forecast_positions"],
                        # Aliases for easier chart creation
                        "val_loss": eval_metrics["eval_loss"],
                        "val_mse": eval_metrics["eval_mse_loss"],
                        "val_mae": eval_metrics["eval_mae_loss"],
                    }, step=global_step)

                    # Log train vs eval gap
                    train_eval_gap = eval_metrics["eval_mse_loss"] - (epoch_mse_loss / (epoch_samples + 1e-8))
                    wandb.log({
                        "gap/train_eval_mse": train_eval_gap,
                    }, step=global_step)

                pbar.write(f"Step {global_step}: Eval MSE = {eval_metrics['eval_mse_loss']:.4f}")

                # Save best model
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'eval_metrics': eval_metrics,
                        'feat_dim': feat_dim,
                        'config': vars(args),
                    }, os.path.join(output_dir, "best_model.pt"))
                    pbar.write(f"Saved best model (loss: {best_eval_loss:.4f})")

                model.train()

            # Save checkpoint
            if (accumulation_step % args.gradient_accumulation_steps == 0 and
                global_step % args.save_interval == 0 and global_step > 0):

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'feat_dim': feat_dim,
                }, os.path.join(output_dir, f"checkpoint_step_{global_step}.pt"))

                # Clean up old checkpoints, keeping only the most recent max_checkpoints
                import glob as glob_module
                import re
                checkpoint_files = glob_module.glob(os.path.join(output_dir, "checkpoint_step_*.pt"))
                if len(checkpoint_files) > args.max_checkpoints:
                    # Sort by step number
                    def get_step(f):
                        match = re.search(r'checkpoint_step_(\d+)\.pt', f)
                        return int(match.group(1)) if match else 0
                    checkpoint_files.sort(key=get_step)
                    # Remove oldest checkpoints
                    for old_ckpt in checkpoint_files[:-args.max_checkpoints]:
                        os.remove(old_ckpt)
                        print(f"Removed old checkpoint: {os.path.basename(old_ckpt)}")

        # End of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mse = epoch_mse_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mae = epoch_mae_loss / epoch_samples if epoch_samples > 0 else 0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average MSE: {avg_epoch_mse:.4f}")
        print(f"  Average MAE: {avg_epoch_mae:.4f}")
        print(f"  Best eval loss: {best_eval_loss:.4f}")

        if use_wandb:
            wandb.log({
                "epoch/loss": avg_epoch_loss,
                "epoch/mse": avg_epoch_mse,
                "epoch/mae": avg_epoch_mae,
                "epoch/number": epoch + 1,
            }, step=global_step)

        # Save at specific epochs
        if (epoch + 1) in save_epochs:
            epoch_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'feat_dim': feat_dim,
                'config': vars(args),
            }, epoch_save_path)
            print(f"Saved model at epoch {epoch+1}")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': args.epochs,
        'global_step': global_step,
        'feat_dim': feat_dim,
        'config': vars(args),
    }, os.path.join(output_dir, "final_model.pt"))

    if use_wandb:
        wandb.finish()

    writer.close()

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
