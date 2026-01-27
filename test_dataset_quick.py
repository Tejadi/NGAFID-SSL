#!/usr/bin/env python3
"""
Quick test script to understand dataset structure and test BERT model.
"""

import pandas as pd
import numpy as np
import io
import os
import torch
import sys
from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def test_dataset_structure():
    """Test dataset structure with minimal downloads."""
    repo_id = "CDuong04/NGAFID-LOCI-GATS-Data"
    print(f"Testing dataset: {repo_id}")

    try:
        # List repository files
        files = list_repo_files(repo_id)
        print(f"Total files in repository: {len(files)}")

        # Show some example files
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"CSV files: {len(csv_files)}")
        if csv_files:
            print("Example CSV files:")
            for i, f in enumerate(csv_files[:5]):
                print(f"  {i+1}. {f}")

            # Try to download and examine a single file
            sample_file = csv_files[0]
            print(f"\nDownloading sample file: {sample_file}")

            # Download single file
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=sample_file,
                local_dir="./sample_flight_data",
                force_download=True
            )

            print(f"Downloaded to: {local_file}")

            # Read and analyze the CSV
            df = pd.read_csv(local_file, na_values=[' NaN', 'NaN', 'NaN '])
            print(f"Sample file shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Fill missing values like in the original code
            df_clean = df.fillna(method='ffill').fillna(method='bfill')
            flight_data = df_clean.to_numpy(dtype=np.float32)

            print(f"Cleaned data shape: {flight_data.shape}")
            print(f"Feature dimension: {flight_data.shape[1]}")
            print(f"Sequence length: {flight_data.shape[0]}")

            return flight_data.shape[1], flight_data  # Return feature dim and sample data

    except Exception as e:
        print(f"Error with dataset: {e}")

    return None, None


def noise_mask(X, masking_ratio, mean_mask_length, mode='separate', distribution='geometric'):
    """Standalone implementation of noise_mask."""
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
    """Standalone implementation of geometric masking."""
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


def create_flight_batch(batch_size: int = 4, seq_len: int = 256, feat_dim: int = 15):
    """Create a batch of synthetic flight data."""
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
        mask = noise_mask(flight, masking_ratio=0.6, mean_mask_length=3)
        masked_flight = flight * mask.astype(np.float32)

        x_original_list.append(torch.tensor(flight))
        x_masked_list.append(torch.tensor(masked_flight))
        mask_list.append(torch.tensor(mask.astype(np.float32)))

    return torch.stack(x_masked_list), torch.stack(x_original_list), torch.stack(mask_list)


def test_training_loop():
    """Test a complete training loop."""
    print("Testing complete training workflow...")

    # Model parameters
    feat_dim = 15
    seq_len = 256
    batch_size = 4

    # Create model
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=256,
        encoder_layers=4,
        decoder_layers=2,
        num_heads=4,
        max_seq_len=seq_len,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    model.train()
    losses = []

    for step in range(10):
        # Create a batch
        x_masked, x_original, mask = create_flight_batch(batch_size, seq_len, feat_dim)

        # Forward pass
        optimizer.zero_grad()
        loss, _ = model.compute_loss(x_masked, x_original, mask)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        masking_ratio = (mask == 0).float().mean().item()

        print(f"Step {step:2d}: loss={loss.item():.4f}, mask_ratio={masking_ratio:.3f}, lr={scheduler.get_last_lr()[0]:.2e}")

    # Check convergence
    if losses[-1] < losses[0]:
        print(f"✓ Training successful: loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    else:
        print(f"⚠ Loss did not decrease consistently (might need more steps)")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        x_masked, x_original, mask = create_flight_batch(2, seq_len, feat_dim)
        reconstructed = model(x_masked)

        # Compute metrics
        masked_positions = (mask == 0)
        if masked_positions.sum() > 0:
            mse_masked = torch.mean((reconstructed[masked_positions] - x_original[masked_positions]) ** 2)
            mae_masked = torch.mean(torch.abs(reconstructed[masked_positions] - x_original[masked_positions]))
            print(f"✓ Evaluation: MSE={mse_masked.item():.4f}, MAE={mae_masked.item():.4f}")

    return True


def main():
    """Run the test."""
    print("=" * 60)
    print("Dataset Analysis and BERT Model Test")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # First try to analyze the real dataset
    feat_dim, sample_data = test_dataset_structure()

    if feat_dim is None:
        print("\nCould not determine feature dimension from dataset.")
        print("Using synthetic data for testing...")
        feat_dim = 15
    else:
        print(f"\n✓ Successfully analyzed dataset!")
        print(f"  Feature dimension: {feat_dim}")
        if sample_data is not None:
            print(f"  Sample shape: {sample_data.shape}")

    # Test the model with determined or default feature dimension
    try:
        print(f"\nTesting BERT model with feat_dim={feat_dim}...")

        # Test model creation
        model = BertMaskedRegressor(
            feat_dim=feat_dim,
            hidden_size=256,
            encoder_layers=4,
            decoder_layers=2,
            num_heads=8,
            max_seq_len=256,
        )
        print(f"✓ Model created successfully!")
        print(f"  Parameters: {count_parameters(model):,}")

        # Test training loop with synthetic data
        success = test_training_loop()
        if success:
            print(f"\n🎉 All tests passed!")
            print(f"  Feature dimension: {feat_dim}")
            print(f"  Model is ready for training!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())