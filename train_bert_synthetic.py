#!/usr/bin/env python3
"""
Training script using synthetic flight data to demonstrate the BERT masked regressor.
This version works without requiring the HuggingFace dataset download.
"""

import argparse
import os
import time
import json
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import our model
try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def noise_mask(X, masking_ratio, mean_mask_length, mode='separate', distribution='geometric'):
    """Masking function for synthetic data."""
    seq_len, feat_dim = X.shape
    if distribution == 'geometric':
        if mode == 'separate':
            mask = np.ones((seq_len, feat_dim), dtype=bool)
            for m in range(feat_dim):
                mask[:, m] = geom_noise_mask_single(seq_len, mean_mask_length, masking_ratio)
        else:
            mask_seq = geom_noise_mask_single(seq_len, mean_mask_length, masking_ratio)
            mask = np.tile(mask_seq[:, None], (1, feat_dim))
    else:
        if mode == 'separate':
            mask = np.random.rand(seq_len, feat_dim) > masking_ratio
        else:
            mask_seq = np.random.rand(seq_len) > masking_ratio
            mask = np.tile(mask_seq[:, None], (1, feat_dim))
    return mask


def geom_noise_mask_single(L, avg_mask_len, masking_ratio):
    """Geometric masking for single sequence."""
    mask = np.ones(L, dtype=bool)
    p_m = 1.0 / avg_mask_len
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    state = False if np.random.rand() < masking_ratio else True
    for i in range(L):
        mask[i] = state
        if state and np.random.rand() < p_m:
            state = False
        elif (not state) and np.random.rand() < p_u:
            state = True
    return mask


def create_flight_batch(batch_size: int, seq_len: int, feat_dim: int, masking_ratio: float = 0.6, mean_mask_length: int = 3):
    """Create a batch of synthetic flight data with masking."""
    x_original_list = []
    x_masked_list = []
    mask_list = []

    for _ in range(batch_size):
        # Generate realistic flight data
        base_values = np.random.randn(feat_dim) * 0.5
        noise = np.random.randn(seq_len, feat_dim) * 0.1
        cumulative_noise = np.cumsum(noise, axis=0)
        flight = base_values[None, :] + cumulative_noise

        # Add periodic components
        for i in range(feat_dim):
            if i % 3 == 0:
                flight[:, i] += np.sin(np.linspace(0, 4*np.pi, seq_len)) * 0.5

        flight = flight.astype(np.float32)

        # Apply masking
        mask = noise_mask(flight, masking_ratio=masking_ratio, mean_mask_length=mean_mask_length)
        masked_flight = flight * mask.astype(np.float32)

        x_original_list.append(torch.tensor(flight))
        x_masked_list.append(torch.tensor(masked_flight))
        mask_list.append(torch.tensor(mask.astype(np.float32)))

    return torch.stack(x_masked_list), torch.stack(x_original_list), torch.stack(mask_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT Masked Regressor with Synthetic Flight Data")

    # Model arguments
    parser.add_argument("--feat_dim", type=int, default=15,
                        help="Feature dimension")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
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
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Number of steps per epoch")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluation interval in steps")

    # Masking arguments
    parser.add_argument("--masking_ratio", type=float, default=0.6,
                        help="Ratio of values to mask")
    parser.add_argument("--mean_mask_length", type=int, default=3,
                        help="Average length of masked segments")

    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('auto', 'cpu', 'cuda')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./synthetic_bert_runs",
                        help="Output directory for models and logs")
    parser.add_argument("--job_name", type=str, default=None,
                        help="Job name for this run")

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


def evaluate_model(model, feat_dim: int, seq_len: int, batch_size: int, device, masking_ratio: float, mean_mask_length: int, num_batches: int = 10):
    """Evaluate model on synthetic validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x_masked, x_original, mask = create_flight_batch(
                batch_size, seq_len, feat_dim, masking_ratio, mean_mask_length
            )
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
    if args.job_name:
        run_name = args.job_name
    else:
        run_name = f"synthetic_bert_h{args.hidden_size}_l{args.encoder_layers}_{int(time.time())}"

    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    print("Creating model...")
    model = BertMaskedRegressor(
        feat_dim=args.feat_dim,
        hidden_size=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("Starting training...")

    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")

        for step in pbar:
            # Create synthetic batch
            x_masked, x_original, mask = create_flight_batch(
                args.batch_size, args.seq_len, args.feat_dim,
                args.masking_ratio, args.mean_mask_length
            )
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
            epoch_loss += loss.item()

            # Log training metrics
            if global_step % 25 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Evaluation
            if global_step % args.eval_interval == 0 and global_step > 0:
                print(f"\nEvaluating at step {global_step}...")
                eval_metrics = evaluate_model(
                    model, args.feat_dim, args.seq_len, args.batch_size,
                    device, args.masking_ratio, args.mean_mask_length
                )

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                print(f"Eval loss = {eval_metrics['eval_loss']:.4f}")

                # Save best model
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'eval_loss': eval_metrics["eval_loss"],
                        'args': vars(args),
                    }, os.path.join(output_dir, "best_model.pt"))
                    print(f"Saved best model (eval_loss: {best_eval_loss:.4f})")

                model.train()

            global_step += 1

        # End of epoch
        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'args': vars(args),
    }, os.path.join(output_dir, "final_model.pt"))

    print(f"Training completed! Best eval loss: {best_eval_loss:.4f}")
    print(f"Models saved to: {output_dir}")

    # Final evaluation with larger batch
    print("\nFinal comprehensive evaluation...")
    final_eval = evaluate_model(
        model, args.feat_dim, args.seq_len, args.batch_size * 2,
        device, args.masking_ratio, args.mean_mask_length, num_batches=20
    )
    print(f"Final evaluation loss: {final_eval['eval_loss']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()