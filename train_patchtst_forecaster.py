#!/usr/bin/env python3
"""
Training script for PatchTST-based flight forecasting.

Masks the END of each flight (causal prediction), using the same
ForecastFlightDataset as the BERT forecaster for fair comparison.
"""

import argparse
import os
import time
import json
from typing import Dict, Optional

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - skipping W&B logging")

try:
    from models.patchtst_masked_regressor import PatchTSTMaskedRegressor, count_parameters
    from ngafid_datasets.forecast_flight_dataset import create_forecast_dataloader
    from ngafid_datasets.local_flight_dataset import get_feature_dim_from_local_data
    from train_full_flights import compute_normalization_parameters
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchTST Forecaster for Flight Data")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_files_train", type=int, default=None)
    parser.add_argument("--max_files_val", type=int, default=100)

    # Forecasting
    parser.add_argument("--forecast_ratios", type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument("--fixed_forecast_ratio", type=float, default=None)
    parser.add_argument("--min_forecast_horizon", type=int, default=50)
    parser.add_argument("--max_forecast_horizon", type=int, default=None)

    # Model
    parser.add_argument("--feat_dim", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000)

    # System
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="./patchtst_forecaster_runs")
    parser.add_argument("--job_name", type=str, default=None)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="patchtst-flight-forecaster")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def create_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
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

            total_loss += loss.item() * x_masked.size(0)
            total_mse += mse_loss.item() * x_masked.size(0)
            total_mae += mae_loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)
            total_forecast_positions += (mask == 0).sum().item()

    denom = total_samples if total_samples > 0 else 1
    return {
        "eval_loss": total_loss / denom,
        "eval_mse_loss": total_mse / denom,
        "eval_mae_loss": total_mae / denom,
        "eval_forecast_positions": total_forecast_positions / denom,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Output dir
    job_name = args.job_name or f"patchtst_forecaster_{int(time.time())}"
    output_dir = os.path.join(args.output_dir, job_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Output directory: {output_dir}")

    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    # W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or job_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=output_dir,
        )
        print(f"W&B tracking: {wandb.run.url}")

    # Feature dim
    feat_dim = args.feat_dim
    if feat_dim is None:
        try:
            feat_dim = get_feature_dim_from_local_data(args.data_dir)
            print(f"Auto-detected feature dimension: {feat_dim}")
        except Exception:
            feat_dim = 44
            print(f"Using default feature dimension: {feat_dim}")

    # Normalization
    print("Computing normalization parameters...")
    normalization_params = compute_normalization_parameters(args.data_dir, max_files=500)

    # Forecast ratio config
    if args.fixed_forecast_ratio is not None:
        forecast_ratios = None
        forecast_ratio = args.fixed_forecast_ratio
    else:
        forecast_ratios = args.forecast_ratios
        forecast_ratio = 0.2  # unused when forecast_ratios provided

    # Data loaders
    print("Creating data loaders...")
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

    val_data_dir = os.path.join(args.data_dir, "preprocessed_data", "val")
    if not os.path.exists(val_data_dir):
        val_data_dir = args.data_dir
        val_split = "val"
    else:
        val_split = "train"

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

    # Model
    print("Creating model...")
    model = PatchTSTMaskedRegressor(
        feat_dim=feat_dim,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    print("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        for batch_idx, (x_masked, x_original, mask) in enumerate(pbar):
            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * x_masked.size(0)
            epoch_mse += mse_loss.item() * x_masked.size(0)
            epoch_mae += mae_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            if global_step % 100 == 0:
                forecast_positions = (mask == 0).sum().item()
                forecast_ratio_actual = forecast_positions / mask.numel()

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("train/mae_loss", mae_loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/forecast_ratio", forecast_ratio_actual, global_step)

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/mae_loss": mae_loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/forecast_ratio": forecast_ratio_actual,
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                        "train/step": global_step,
                    }, step=global_step)

            pbar.set_postfix({
                "mse": f"{mse_loss.item():.4f}",
                "mae": f"{mae_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            if global_step % args.eval_interval == 0 and global_step > 0:
                pbar.write(f"\nEvaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_loader, device)

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                writer.add_scalar("eval/mse_loss", eval_metrics["eval_mse_loss"], global_step)
                writer.add_scalar("eval/mae_loss", eval_metrics["eval_mae_loss"], global_step)

                if use_wandb:
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/mse_loss": eval_metrics["eval_mse_loss"],
                        "eval/mae_loss": eval_metrics["eval_mae_loss"],
                        "eval/forecast_positions": eval_metrics["eval_forecast_positions"],
                    }, step=global_step)

                pbar.write(f"Step {global_step}: Eval loss = {eval_metrics['eval_loss']:.4f}")

                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    save_args = vars(args).copy()
                    save_args['feat_dim'] = feat_dim
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'eval_loss': eval_metrics["eval_loss"],
                        'args': save_args,
                    }, os.path.join(output_dir, "best_model.pt"))
                    pbar.write(f"Saved best model (eval_loss: {best_eval_loss:.4f})")

                model.train()

            if global_step % args.save_interval == 0 and global_step > 0:
                save_args = vars(args).copy()
                save_args['feat_dim'] = feat_dim
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'args': save_args,
                }, os.path.join(output_dir, f"checkpoint_{global_step}.pt"))

            global_step += 1

        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        pbar.write(f"Epoch {epoch+1} completed - avg loss: {avg_loss:.4f}")

        if use_wandb:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/number": epoch + 1,
            }, step=global_step)

        pbar.close()

    # Final save
    save_args = vars(args).copy()
    save_args['feat_dim'] = feat_dim
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'args': save_args,
    }, os.path.join(output_dir, "final_model.pt"))

    print(f"Training completed! Best eval loss: {best_eval_loss:.4f}")
    print(f"Models saved to: {output_dir}")

    if use_wandb:
        wandb.summary["best_eval_loss"] = best_eval_loss
        wandb.finish()

    writer.close()


if __name__ == "__main__":
    main()
