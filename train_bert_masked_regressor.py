#!/usr/bin/env python3
"""
Training script for BERT-based masked column regression on flight data.

Simple architecture:
1. Load masked flight data from HuggingFace dataset
2. Use BERT encoder to get embeddings from masked data
3. Train decoder to reconstruct original flight data
4. Evaluate on masked positions only
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

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - skipping W&B logging")

# Import our models and dataset
try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
    from ngafid_datasets.bert_flight_dataset import create_dataloader
    from ngafid_datasets.local_flight_dataset import create_local_dataloader, get_feature_dim_from_local_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT Masked Regressor for Flight Data")

    # Data arguments
    parser.add_argument("--repo_id", type=str, default="CDuong04/NGAFID-LOCI-GATS-Data",
                        help="HuggingFace dataset repository ID")
    parser.add_argument("--local_data_dir", type=str, default=None,
                        help="Local directory containing flight CSV files (overrides HuggingFace)")
    parser.add_argument("--subdir", type=str, default="preprocessed_data",
                        help="Subdirectory in the dataset")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length for flight windows")
    parser.add_argument("--max_files_train", type=int, default=None,
                        help="Maximum training files to use (None for all)")
    parser.add_argument("--max_files_val", type=int, default=100,
                        help="Maximum validation files to use")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of files for training (local data only)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of files for validation (local data only)")

    # Model arguments
    parser.add_argument("--feat_dim", type=int, default=None,
                        help="Feature dimension (auto-detected if None)")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="BERT hidden size")
    parser.add_argument("--encoder_layers", type=int, default=6,
                        help="Number of BERT encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
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

    # Masking arguments
    parser.add_argument("--masking_ratio", type=float, default=0.6,
                        help="Ratio of values to mask")
    parser.add_argument("--mean_mask_length", type=int, default=3,
                        help="Average length of masked segments")

    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('auto', 'cpu', 'cuda', 'cuda:0')")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./bert_masked_regressor_runs",
                        help="Output directory for models and logs")
    parser.add_argument("--job_name", type=str, default=None,
                        help="Job name for this run")

    # Weights & Biases arguments
    parser.add_argument("--wandb_project", type=str, default="bert-flight-ssl",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (defaults to job_name)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    return parser.parse_args()


def setup_device(device_str: str) -> torch.device:
    """Setup compute device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")

    return device


def setup_output_dir(args) -> str:
    """Create output directory for this run."""
    if args.job_name:
        run_name = args.job_name
    else:
        run_name = f"bert_h{args.hidden_size}_l{args.encoder_layers}_{int(time.time())}"

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return output_dir


def create_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, max_batches: int = 100) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            loss, _ = model.compute_loss(x_masked, x_original, mask)

            total_loss += loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    return {"eval_loss": avg_loss}


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
        wandb_run_name = args.wandb_run_name or args.job_name or f"bert_h{args.hidden_size}_l{args.encoder_layers}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=output_dir,
        )
        print(f"🪄 W&B tracking: {wandb.run.url}")
    else:
        print("📝 Using only TensorBoard logging")

    print("Creating data loaders...")

    # Determine if we're using local data or HuggingFace
    use_local_data = args.local_data_dir is not None

    if use_local_data:
        print(f"Using local data from: {args.local_data_dir}")

        # Auto-detect feature dimension from local data
        try:
            feat_dim = get_feature_dim_from_local_data(args.local_data_dir)
            if feat_dim:
                print(f"Auto-detected feature dimension: {feat_dim}")
            else:
                feat_dim = args.feat_dim
                if feat_dim is None:
                    print("Could not auto-detect feature dimension. Please specify --feat_dim")
                    exit(1)
        except Exception as e:
            print(f"Error detecting feature dimension: {e}")
            feat_dim = args.feat_dim
            if feat_dim is None:
                print("Please specify --feat_dim")
                exit(1)
    else:
        print(f"Using HuggingFace dataset: {args.repo_id}")

        # Get feature dimension from a sample
        try:
            sample_loader = create_dataloader(
                repo_id=args.repo_id,
                split="train",
                batch_size=1,
                seq_len=args.seq_len,
                max_files=1,
                num_workers=0,
                seed=args.seed,
            )
            x_sample, _, _ = next(iter(sample_loader))
            feat_dim = x_sample.shape[-1]
            print(f"Auto-detected feature dimension: {feat_dim}")
        except Exception as e:
            print(f"Could not auto-detect feature dimension: {e}")
            feat_dim = args.feat_dim
            if feat_dim is None:
                print("Please specify --feat_dim")
                exit(1)

    # Create data loaders
    if use_local_data:
        train_loader = create_local_dataloader(
            data_dir=args.local_data_dir,
            split="train",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            max_files=args.max_files_train,
            num_workers=args.num_workers,
            seed=args.seed,
            train_split=args.train_split,
            val_split=args.val_split,
        )

        val_loader = create_local_dataloader(
            data_dir=args.local_data_dir,
            split="validation",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            max_files=args.max_files_val,
            num_workers=args.num_workers,
            seed=args.seed + 1,
            train_split=args.train_split,
            val_split=args.val_split,
        )
    else:
        train_loader = create_dataloader(
            repo_id=args.repo_id,
            split="train",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            max_files=args.max_files_train,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        val_loader = create_dataloader(
            repo_id=args.repo_id,
            split="validation" if "validation" in ["train", "validation", "test"] else "train",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            masking_ratio=args.masking_ratio,
            mean_mask_length=args.mean_mask_length,
            max_files=args.max_files_val,
            num_workers=args.num_workers,
            seed=args.seed + 1,
        )

    print("Creating model...")
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Log model to W&B
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
        wandb.config.update({
            "model_parameters": count_parameters(model),
            "feat_dim": feat_dim,
        })

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Estimate total steps
    total_steps = args.epochs * 1000  # Rough estimate
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    print("Starting training...")

    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        # Create progress bar for this epoch
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

            # Forward pass
            optimizer.zero_grad()
            loss, _ = model.compute_loss(x_masked, x_original, mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            epoch_loss += loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            # Log training metrics
            if global_step % 100 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                # W&B logging
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                        "train/step": global_step,
                    }, step=global_step)

            # Update progress bar with current metrics
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss/(epoch_samples+1e-8):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            # Evaluation
            if global_step % args.eval_interval == 0 and global_step > 0:
                pbar.write(f"\nEvaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_loader, device)

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)

                # W&B evaluation logging
                if use_wandb:
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/step": global_step,
                    }, step=global_step)

                pbar.write(f"Step {global_step}: Eval loss = {eval_metrics['eval_loss']:.4f}")

                # Save best model
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    # Add feature dimension to args for model loading
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

            # Save checkpoint
            if global_step % args.save_interval == 0 and global_step > 0:
                # Add feature dimension to args for model loading
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

        # End of epoch
        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        pbar.write(f"Epoch {epoch+1} completed - avg loss: {avg_loss:.4f}")

        # Log epoch summary to W&B
        if use_wandb:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/number": epoch + 1,
            }, step=global_step)

        pbar.close()

    # Final save
    # Add feature dimension to args for model loading
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

    # Final W&B summary
    if use_wandb:
        wandb.summary["best_eval_loss"] = best_eval_loss
        wandb.summary["total_epochs"] = args.epochs
        wandb.summary["total_steps"] = global_step
        wandb.finish()

    writer.close()


if __name__ == "__main__":
    main()