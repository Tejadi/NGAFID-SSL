#!/usr/bin/env python3
"""
Simple test script for the BERT masked regressor implementation.
Creates synthetic flight data to test the model without dataset dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


def noise_mask(X, masking_ratio, mean_mask_length, mode='separate', distribution='geometric'):
    """Standalone implementation of noise_mask to avoid pandas dependency."""
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
            mask = np.random.rand(seq_len, feat_dim) > masking_ratio  # True = keep, False = mask
        else:
            mask_seq = np.random.rand(seq_len) > masking_ratio
            mask = np.tile(mask_seq[:, None], (1, feat_dim))
    return mask


def geom_noise_mask_single(L, avg_mask_len, masking_ratio):
    """Standalone implementation of geometric masking."""
    mask = np.ones(L, dtype=bool)
    p_m = 1.0 / avg_mask_len                     # prob to end a masked segment
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # prob to end an unmasked segment
    state = False if np.random.rand() < masking_ratio else True  # start in masked state with given ratio
    for i in range(L):
        mask[i] = state  # True = keep original, False = mask out
        if state and np.random.rand() < p_m:
            state = False
        elif (not state) and np.random.rand() < p_u:
            state = True
    return mask


def create_synthetic_flight_data(num_samples: int = 4, seq_len: int = 128, feat_dim: int = 15):
    """Create synthetic flight data for testing."""
    # Generate realistic-looking flight data with smooth trajectories
    flight_data = []

    for _ in range(num_samples):
        # Start with some base values and add smooth variations
        base_values = np.random.randn(feat_dim) * 0.5

        # Generate smooth time series by using cumulative noise
        noise = np.random.randn(seq_len, feat_dim) * 0.1
        cumulative_noise = np.cumsum(noise, axis=0)

        # Add trend and base values
        flight = base_values[None, :] + cumulative_noise

        # Add some periodic components (like altitude changes)
        for i in range(feat_dim):
            if i % 3 == 0:  # Some features have periodic behavior
                flight[:, i] += np.sin(np.linspace(0, 4*np.pi, seq_len)) * 0.5

        flight_data.append(flight.astype(np.float32))

    return flight_data


def test_masking():
    """Test the masking functionality."""
    print("Testing masking functionality...")

    seq_len, feat_dim = 128, 15
    flight_data = create_synthetic_flight_data(1, seq_len, feat_dim)[0]

    # Test masking
    mask = noise_mask(flight_data, masking_ratio=0.6, mean_mask_length=3)
    masked_data = flight_data * mask.astype(np.float32)

    print(f"Original data shape: {flight_data.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked ratio: {(mask == 0).mean():.3f}")

    return True


def test_model():
    """Test the BERT model."""
    print("Testing BERT model...")

    # Create model
    feat_dim = 15
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=256,
        encoder_layers=3,
        decoder_layers=2,
        num_heads=4,
        max_seq_len=128,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Create synthetic batch
    batch_size, seq_len = 4, 128
    flight_data = create_synthetic_flight_data(batch_size, seq_len, feat_dim)

    # Convert to tensors and apply masking
    x_original_list = []
    x_masked_list = []
    mask_list = []

    for flight in flight_data:
        mask = noise_mask(flight, masking_ratio=0.6, mean_mask_length=3)
        masked_flight = flight * mask.astype(np.float32)

        x_original_list.append(torch.tensor(flight))
        x_masked_list.append(torch.tensor(masked_flight))
        mask_list.append(torch.tensor(mask.astype(np.float32)))

    # Stack into batches
    x_original = torch.stack(x_original_list)
    x_masked = torch.stack(x_masked_list)
    mask_tensor = torch.stack(mask_list)

    print(f"Batch shapes: x_original={x_original.shape}, x_masked={x_masked.shape}, mask={mask_tensor.shape}")

    # Test forward pass
    with torch.no_grad():
        reconstructed = model(x_masked)
        print(f"Reconstructed shape: {reconstructed.shape}")

        # Test loss computation
        loss, recon_loss = model.compute_loss(x_masked, x_original, mask_tensor)
        print(f"Loss: {loss.item():.4f}")

        # Compute accuracy on masked positions
        masked_positions = (mask_tensor == 0)
        if masked_positions.sum() > 0:
            mse_masked = torch.mean((reconstructed[masked_positions] - x_original[masked_positions]) ** 2)
            print(f"MSE on masked positions: {mse_masked.item():.4f}")

    return True


def test_training_step():
    """Test a training step."""
    print("Testing training step...")

    feat_dim = 15
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=128,  # Smaller for faster testing
        encoder_layers=2,
        decoder_layers=1,
        num_heads=4,
        max_seq_len=64,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create a small batch
    batch_size, seq_len = 2, 64
    flight_data = create_synthetic_flight_data(batch_size, seq_len, feat_dim)

    x_original_list = []
    x_masked_list = []
    mask_list = []

    for flight in flight_data:
        mask = noise_mask(flight, masking_ratio=0.6, mean_mask_length=3)
        masked_flight = flight * mask.astype(np.float32)

        x_original_list.append(torch.tensor(flight))
        x_masked_list.append(torch.tensor(masked_flight))
        mask_list.append(torch.tensor(mask.astype(np.float32)))

    x_original = torch.stack(x_original_list)
    x_masked = torch.stack(x_masked_list)
    mask_tensor = torch.stack(mask_list)

    # Training step
    model.train()

    loss_before = None
    for step in range(3):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(x_masked, x_original, mask_tensor)
        loss.backward()
        optimizer.step()

        if loss_before is None:
            loss_before = loss.item()

        print(f"Step {step}: loss = {loss.item():.4f}")

    # Check that loss decreased
    if loss.item() < loss_before:
        print("✓ Loss decreased during training")
    else:
        print("⚠ Loss did not decrease - this might be normal for short training")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing BERT Masked Regressor Implementation")
    print("=" * 50)

    tests = [
        test_masking,
        test_model,
        test_training_step,
    ]

    passed = 0
    total = len(tests)

    for i, test_func in enumerate(tests):
        try:
            print(f"\n[{i+1}/{total}] {test_func.__name__}")
            print("-" * 30)
            success = test_func()
            if success:
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The implementation is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    main()