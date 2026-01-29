#!/usr/bin/env python3
"""
Training script for PatchTST-based masked regression on flight data.
Supports random masking for enhanced training diversity.
"""

import argparse
import os
import time
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
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
    from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchTST Masked Regressor for Flight Data")

    # Data arguments
    parser.add_argument("--local_data_dir", type=str, required=True,
                        help="Local directory containing flight CSV files")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length for flight windows")
    parser.add_argument("--max_files_train", type=int, default=None,
                        help="Maximum training files to use (None for all)")
    parser.add_argument("--max_files_val", type=int, default=None,
                        help="Maximum validation files to use (None for all)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of files for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of files for validation")

    # Model arguments
    parser.add_argument("--feat_dim", type=int, default=None,
                        help="Feature dimension (auto-detected if None)")
    parser.add_argument("--patch_len", type=int, default=16,
                        help="Patch length for PatchTST")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride for PatchTST")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feedforward dimension")
    parser.add_argument("--encoder_layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluation interval in steps")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Model save interval in steps")

    # Random masking arguments (matching BERT setup)
    parser.add_argument("--use_random_masking", action="store_true",
                        help="Use random masking ratios and lengths")
    parser.add_argument("--masking_ratios", type=float, nargs='+', default=[0.2, 0.5, 0.8],
                        help="List of masking ratios to sample from (if using random masking)")
    parser.add_argument("--mean_mask_lengths", type=int, nargs='+', default=[5, 60],
                        help="List of mean mask lengths to sample from (if using random masking)")

    # Fixed masking arguments (fallback if not using random)
    parser.add_argument("--masking_ratio", type=float, default=0.6,
                        help="Fixed masking ratio (used if not using random masking)")
    parser.add_argument("--mean_mask_length", type=int, default=3,
                        help="Fixed mean mask length (used if not using random masking)")

    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('auto', 'cpu', 'cuda', 'cuda:0')")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./patchtst_masked_regressor_runs",
                        help="Output directory for models and logs")
    parser.add_argument("--job_name", type=str, default=None,
                        help="Job name for this run")

    # Weights & Biases arguments
    parser.add_argument("--wandb_project", type=str, default="patchtst-flight-ssl",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (defaults to job_name)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    return parser.parse_args()


def setup_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")

    return device


def setup_output_dir(args) -> str:
    if args.job_name:
        run_name = args.job_name
    else:
        masking_suffix = "random" if args.use_random_masking else "fixed"
        run_name = f"patchtst_d{args.d_model}_l{args.encoder_layers}_{masking_suffix}_{int(time.time())}"

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return output_dir


def create_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, max_batches: int = 100) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_samples = 0
    total_masked_positions = 0

    with torch.no_grad():
        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)

            masked_positions = (mask == 0).sum().item()

            total_loss += loss.item() * x_masked.size(0)
            total_mse_loss += mse_loss.item() * x_masked.size(0)
            total_mae_loss += mae_loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)
            total_masked_positions += masked_positions

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse_loss = total_mse_loss / total_samples if total_samples > 0 else float('inf')
    avg_mae_loss = total_mae_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse_per_position = avg_mse_loss * total_samples / total_masked_positions if total_masked_positions > 0 else float('inf')
    avg_mae_per_position = avg_mae_loss * total_samples / total_masked_positions if total_masked_positions > 0 else float('inf')

    return {
        "eval_loss": avg_loss,
        "eval_mse_loss": avg_mse_loss,
        "eval_mae_loss": avg_mae_loss,
        "eval_mse_per_position": avg_mse_per_position,
        "eval_mae_per_position": avg_mae_per_position,
        "eval_masked_positions": total_masked_positions / total_samples if total_samples > 0 else 0
    }


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = setup_device(args.device)

    # Setup output directory
    output_dir = setup_output_dir(args)
    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    # Initialize Weights & Biases
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb_run_name = args.wandb_run_name or args.job_name or f"patchtst_d{args.d_model}_l{args.encoder_layers}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=output_dir,
        )
        print(f"W&B tracking: {wandb.run.url}")
    else:
        print("Using only TensorBoard logging")

    print("Computing normalization parameters...")
    normalization_params = compute_normalization_parameters(
        args.local_data_dir,
        max_files=args.max_files_train
    )

    if normalization_params is None:
        print("Warning: Could not compute normalization parameters")
    else:
        print(f"Normalization: mean shape={normalization_params['mean'].shape}, std shape={normalization_params['std'].shape}")
        feat_dim = normalization_params['mean'].shape[0]
        print(f"Auto-detected feature dimension: {feat_dim}")

    # Override feat_dim if provided
    if args.feat_dim is not None:
        feat_dim = args.feat_dim
        print(f"Using user-specified feature dimension: {feat_dim}")

    print("Creating data loaders...")

    # Create training dataset
    train_dataset = GlobalNormalizedFlightDataset(
        normalization_params=normalization_params,
        data_dir=args.local_data_dir,
        split="train",
        seq_len=args.seq_len,
        max_files=args.max_files_train,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        use_random_masking=args.use_random_masking,
        masking_ratios=args.masking_ratios if args.use_random_masking else None,
        mean_mask_lengths=args.mean_mask_lengths if args.use_random_masking else None,
        masking_ratio=args.masking_ratio if not args.use_random_masking else 0.6,
        mean_mask_length=args.mean_mask_length if not args.use_random_masking else 3,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create validation dataset
    val_dataset = GlobalNormalizedFlightDataset(
        normalization_params=normalization_params,
        data_dir=args.local_data_dir,
        split="validation",
        seq_len=args.seq_len,
        max_files=args.max_files_val,
        seed=args.seed + 1,
        train_split=args.train_split,
        val_split=args.val_split,
        use_random_masking=args.use_random_masking,
        masking_ratios=args.masking_ratios if args.use_random_masking else None,
        mean_mask_lengths=args.mean_mask_lengths if args.use_random_masking else None,
        masking_ratio=args.masking_ratio if not args.use_random_masking else 0.6,
        mean_mask_length=args.mean_mask_length if not args.use_random_masking else 3,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

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
        wandb.config.update({
            "model_parameters": str(count_parameters(model)),
        })

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Estimate total steps
    total_steps = args.epochs * 1000
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    print("Starting training...")

    global_step = 0
    best_eval_loss = float('inf')
    best_eval_mse_per_position = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch",
            leave=True
        )

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
            epoch_mse_loss += mse_loss.item() * x_masked.size(0)
            epoch_mae_loss += mae_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            if global_step % 100 == 0:
                masked_positions = (mask == 0).sum().item()
                mse_per_position = mse_loss.item() / masked_positions if masked_positions > 0 else 0
                mae_per_position = mae_loss.item() / masked_positions if masked_positions > 0 else 0
                masking_ratio = masked_positions / mask.numel()

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("train/mae_loss", mae_loss.item(), global_step)
                writer.add_scalar("train/mse_per_position", mse_per_position, global_step)
                writer.add_scalar("train/mae_per_position", mae_per_position, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/masking_ratio", masking_ratio, global_step)

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/mae_loss": mae_loss.item(),
                        "train/mse_per_position": mse_per_position,
                        "train/mae_per_position": mae_per_position,
                        "train/masking_ratio": masking_ratio,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                        "train/step": global_step,
                    }, step=global_step)

            pbar.set_postfix({
                "mse": f"{mse_loss.item():.4f}",
                "mae": f"{mae_loss.item():.4f}",
                "avg_mse": f"{epoch_mse_loss/(epoch_samples+1e-8):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            if global_step % args.eval_interval == 0 and global_step > 0:
                pbar.write(f"\nEvaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_loader, device)

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                writer.add_scalar("eval/mse_loss", eval_metrics["eval_mse_loss"], global_step)
                writer.add_scalar("eval/mae_loss", eval_metrics["eval_mae_loss"], global_step)
                writer.add_scalar("eval/mse_per_position", eval_metrics["eval_mse_per_position"], global_step)
                writer.add_scalar("eval/mae_per_position", eval_metrics["eval_mae_per_position"], global_step)
                writer.add_scalar("eval/masked_positions", eval_metrics["eval_masked_positions"], global_step)

                if use_wandb:
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/mse_loss": eval_metrics["eval_mse_loss"],
                        "eval/mae_loss": eval_metrics["eval_mae_loss"],
                        "eval/mse_per_position": eval_metrics["eval_mse_per_position"],
                        "eval/mae_per_position": eval_metrics["eval_mae_per_position"],
                        "eval/masked_positions": eval_metrics["eval_masked_positions"],
                        "eval/step": global_step,
                    }, step=global_step)

                pbar.write(f"Step {global_step}: Eval loss = {eval_metrics['eval_loss']:.4f}, MSE/pos = {eval_metrics['eval_mse_per_position']:.4f}")

                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    best_eval_mse_per_position = eval_metrics["eval_mse_per_position"]
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
    print(f"Best MSE per position: {best_eval_mse_per_position:.4f}")
    print(f"Models saved to: {output_dir}")

    if use_wandb:
        wandb.summary["best_eval_loss"] = best_eval_loss
        wandb.summary["best_eval_mse_per_position"] = best_eval_mse_per_position
        wandb.summary["total_epochs"] = args.epochs
        wandb.summary["total_steps"] = global_step
        wandb.finish()

    writer.close()


if __name__ == "__main__":
    main()
