#!/usr/bin/env python3
"""
Debug the loss computation to understand why masking ratio doesn't affect loss as expected.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ngafid_datasets.local_flight_dataset import create_local_dataloader
from models.bert_masked_regressor import BertMaskedRegressor

def debug_loss_computation():
    """Debug what's happening in loss computation."""
    print("=" * 60)
    print("Debug Loss Computation")
    print("=" * 60)

    # Test both masking ratios
    masking_ratios = [0.05, 0.6]

    for mask_ratio in masking_ratios:
        print(f"\n--- Testing masking ratio: {mask_ratio} ---")

        # Create dataloader
        dataloader = create_local_dataloader(
            data_dir="./NGAFID-LOCI-GATS-Data",
            split="train",
            batch_size=2,
            seq_len=64,
            masking_ratio=mask_ratio,
            max_files=1,
            num_workers=0,
            seed=42,
        )

        # Get one batch
        x_masked, x_original, mask = next(iter(dataloader))

        print(f"Data shapes: {x_masked.shape}")
        print(f"Total positions: {x_masked.numel()}")

        # Analyze masking
        masked_positions = (mask == 0)
        num_masked = masked_positions.sum().item()
        masking_percentage = num_masked / x_masked.numel() * 100

        print(f"Masked positions: {num_masked}")
        print(f"Actual masking %: {masking_percentage:.1f}%")

        # Manual loss computation
        diff = x_masked - x_original
        squared_diff = diff ** 2

        # Total MSE (all positions)
        total_mse = torch.mean(squared_diff)
        print(f"Total MSE (all positions): {total_mse.item():.2f}")

        # MSE on masked positions only
        if num_masked > 0:
            masked_mse = torch.mean(squared_diff[masked_positions])
            print(f"MSE on masked positions: {masked_mse.item():.2f}")

            # Sum version (like in model)
            masked_sum = torch.sum(squared_diff[masked_positions])
            masked_avg = masked_sum / num_masked
            print(f"Manual sum/count: {masked_avg.item():.2f}")

        # Test model loss computation
        model = BertMaskedRegressor(feat_dim=44, hidden_size=128, encoder_layers=2, decoder_layers=1)
        model.eval()

        with torch.no_grad():
            loss, recon_loss = model.compute_loss(x_masked, x_original, mask)
            print(f"Model loss: {loss.item():.2f}")

        print()

if __name__ == "__main__":
    debug_loss_computation()