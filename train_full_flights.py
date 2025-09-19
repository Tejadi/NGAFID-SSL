#!/usr/bin/env python3
"""
Training script for full-flight BERT masked regressor on Oscar cluster.
Optimized for RTX A5000 with 24-hour training window.
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
    from ngafid_datasets.local_flight_dataset import create_local_dataloader, get_feature_dim_from_local_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def create_lr_scheduler(optimizer, num_training_steps, warmup_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, max_batches: int = 50) -> Dict[str, float]:
    """Evaluate model on validation data."""
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

            # Count masked positions for per-position metrics
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
    # Fixed configuration optimized for Oscar cluster training
    print("🚀 Starting BERT Flight Training on Oscar Cluster")
    print("=" * 60)

    # Dataset and model configuration
    data_dir = "/oscar/data/sbach/shared/ngafid"
    seq_len = 10000  # Full flight sequences
    batch_size = 8   # Conservative for 10k sequence length
    epochs = 18
    learning_rate = 5e-5  # Conservative for large model + long sequences

    # Model architecture (optimized for RTX A5000)
    hidden_size = 1536
    encoder_layers = 12
    decoder_layers = 8
    num_heads = 16
    dropout = 0.1

    # Training settings
    warmup_steps = 2000
    eval_interval = 1000
    save_interval = 3000
    max_files_train = 800  # Conservative for 10k sequences
    max_files_val = 150

    print(f"📊 Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Sequence length: {seq_len:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Model: {hidden_size}d, {encoder_layers}enc, {decoder_layers}dec")
    print()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (GPU not available)")
    print()

    # Check data directory
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Please check the path and try again.")
        exit(1)

    print(f"✅ Found data directory: {data_dir}")

    # Get feature dimension from data
    print("🔍 Detecting feature dimension from data...")
    try:
        feat_dim = get_feature_dim_from_local_data(data_dir)
        print(f"✅ Detected feature dimension: {feat_dim}")
    except Exception as e:
        print(f"❌ Error detecting feature dimension: {e}")
        print("Using default feature dimension: 44")
        feat_dim = 44

    # Create data loaders
    print("📁 Creating data loaders...")
    try:
        train_loader = create_local_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=batch_size,
            seq_len=seq_len,
            max_files=max_files_train,
            num_workers=2,  # Conservative for large sequences
            seed=42
        )

        val_loader = create_local_dataloader(
            data_dir=data_dir,
            split="val",
            batch_size=batch_size,
            seq_len=seq_len,
            max_files=max_files_val,
            num_workers=2,
            seed=42
        )
        print(f"✅ Created data loaders (train: ~{len(train_loader)} batches)")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        exit(1)

    # Create model
    print("🏗️  Creating model...")
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=seq_len,
    ).to(device)

    total_params = count_parameters(model)
    print(f"✅ Model created with {total_params:,} parameters")
    print(f"   Estimated GPU memory: ~{total_params * 4 / 1e9:.1f} GB")
    print()

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    total_steps = len(train_loader) * epochs
    scheduler = create_lr_scheduler(optimizer, total_steps, warmup_steps)

    print(f"📈 Training setup:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    print(f"   Eval interval: {eval_interval:,}")
    print()

    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_name = f"bert_full_flights_{timestamp}"

    # Create output directory
    output_dir = f"./results/{job_name}"
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{output_dir}/tensorboard")

    # W&B
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(
                project="bert-flight-full",
                name=job_name,
                config={
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_size": hidden_size,
                    "encoder_layers": encoder_layers,
                    "decoder_layers": decoder_layers,
                    "epochs": epochs,
                    "total_params": total_params,
                }
            )
            print("✅ W&B logging enabled")
        except Exception as e:
            print(f"⚠️  W&B setup failed: {e}")
            use_wandb = False

    print(f"📊 Results will be saved to: {output_dir}")
    print()

    # Training loop
    print("🎯 Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        epoch_samples = 0

        # Create progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            dynamic_ncols=True
        )

        for batch_idx, (x_masked, x_original, mask) in enumerate(pbar):
            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            # Forward pass
            optimizer.zero_grad()
            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            epoch_loss += loss.item() * x_masked.size(0)
            epoch_mse_loss += mse_loss.item() * x_masked.size(0)
            epoch_mae_loss += mae_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            # Log training metrics
            if global_step % 100 == 0:
                # Compute per-position metrics
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

            # Update progress bar
            pbar.set_postfix({
                "mse": f"{mse_loss.item():.4f}",
                "mae": f"{mae_loss.item():.4f}",
                "avg_mse": f"{epoch_mse_loss/(epoch_samples+1e-8):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            # Evaluation
            if global_step % eval_interval == 0 and global_step > 0:
                pbar.write(f"\n🔍 Evaluating at step {global_step}...")
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

                pbar.write(f"📊 Step {global_step}: Eval MSE = {eval_metrics['eval_mse_loss']:.4f}, MAE = {eval_metrics['eval_mae_loss']:.4f}")

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
                        'config': {
                            'hidden_size': hidden_size,
                            'encoder_layers': encoder_layers,
                            'decoder_layers': decoder_layers,
                            'num_heads': num_heads,
                            'seq_len': seq_len,
                        }
                    }, f"{output_dir}/best_model.pt")
                    pbar.write(f"💾 Saved best model (eval_loss: {best_eval_loss:.4f})")

            # Save checkpoint
            if global_step % save_interval == 0 and global_step > 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'feat_dim': feat_dim,
                }, f"{output_dir}/checkpoint_step_{global_step}.pt")
                pbar.write(f"💾 Saved checkpoint at step {global_step}")

            global_step += 1

        # End of epoch summary
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mse = epoch_mse_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mae = epoch_mae_loss / epoch_samples if epoch_samples > 0 else 0

        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Average MSE: {avg_epoch_mse:.4f}")
        print(f"   Average MAE: {avg_epoch_mae:.4f}")
        print(f"   Best eval loss: {best_eval_loss:.4f}")
        print()

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
        'global_step': global_step,
        'feat_dim': feat_dim,
        'final_config': {
            'hidden_size': hidden_size,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'total_params': total_params,
        }
    }, f"{output_dir}/final_model.pt")

    writer.close()
    if use_wandb:
        wandb.finish()

    print("🎉 Training completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🏆 Best evaluation loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
