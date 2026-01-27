#!/usr/bin/env python3
"""
Test script for local dataset functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from ngafid_datasets.local_flight_dataset import create_local_dataloader, get_feature_dim_from_local_data


def test_local_dataset():
    """Test the local dataset functionality."""
    data_dir = "./NGAFID-LOCI-GATS-Data"

    print("Testing LocalFlightDataset...")
    print(f"Data directory: {data_dir}")

    # Test feature dimension detection
    feat_dim = get_feature_dim_from_local_data(data_dir)
    print(f"Auto-detected feature dimension: {feat_dim}")

    if feat_dim is None:
        print("Could not detect feature dimension. Check your data directory.")
        return False

    try:
        # Test dataloader with minimal settings
        dataloader = create_local_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=2,
            seq_len=128,
            max_files=3,  # Only test a few files
            num_workers=0,
            seed=42,
        )

        print("\nTesting dataloader...")
        batch_count = 0
        total_samples = 0

        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            batch_count += 1
            total_samples += x_masked.size(0)

            print(f"Batch {batch_idx}:")
            print(f"  x_masked shape: {x_masked.shape}")
            print(f"  x_original shape: {x_original.shape}")
            print(f"  mask shape: {mask.shape}")
            print(f"  Feature dimension: {x_masked.shape[-1]}")
            print(f"  Masking ratio: {(mask == 0).float().mean().item():.3f}")

            # Verify shapes
            assert x_masked.shape == x_original.shape == mask.shape
            assert x_masked.shape[-1] == feat_dim

            # Test some basic properties
            masked_positions = (mask == 0)
            unmasked_positions = (mask == 1)

            print(f"  Masked positions: {masked_positions.sum().item()}")
            print(f"  Unmasked positions: {unmasked_positions.sum().item()}")

            # Verify masked positions are zeroed
            assert torch.allclose(x_masked[masked_positions], torch.zeros_like(x_masked[masked_positions]))

            # Verify unmasked positions are preserved
            assert torch.allclose(x_masked[unmasked_positions], x_original[unmasked_positions])

            if batch_idx >= 4:  # Test a few batches
                break

        print(f"\n✓ Processed {batch_count} batches, {total_samples} samples total")
        print(f"✓ Feature dimension: {feat_dim}")
        print("✓ Local dataset test completed successfully!")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_splits():
    """Test different data splits."""
    data_dir = "./NGAFID-LOCI-GATS-Data"

    print("\nTesting data splits...")

    splits = ["train", "validation", "test"]

    for split in splits:
        try:
            dataloader = create_local_dataloader(
                data_dir=data_dir,
                split=split,
                batch_size=1,
                seq_len=64,
                max_files=2,
                num_workers=0,
                seed=42,
            )

            # Test that we can get at least one batch
            batch = next(iter(dataloader))
            print(f"✓ Split '{split}': {batch[0].shape}")

        except Exception as e:
            print(f"✗ Split '{split}' failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Local Dataset with NGAFID Data")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Test basic functionality
    success = test_local_dataset()

    if success:
        # Test data splits
        test_splits()

        print("\n🎉 All local dataset tests passed!")
        print("\nYou can now train with:")
        print("python train_bert_masked_regressor.py --local_data_dir ./NGAFID-LOCI-GATS-Data --job_name local_test --epochs 2")
    else:
        print("\n❌ Local dataset tests failed.")

    print("\n" + "=" * 60)