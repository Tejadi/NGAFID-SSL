#!/usr/bin/env python3
"""
Test script to verify that the saved BERT model works correctly.
"""

import torch
import numpy as np
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bert_masked_regressor import BertMaskedRegressor


def noise_mask(X, masking_ratio, mean_mask_length, mode='separate', distribution='geometric'):
    """Masking function."""
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
    """Geometric masking."""
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


def test_saved_model(model_path: str):
    """Test a saved model."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']

    print(f"Model trained for {checkpoint.get('global_step', 'unknown')} steps")
    if 'eval_loss' in checkpoint:
        print(f"Best eval loss: {checkpoint['eval_loss']:.4f}")

    # Recreate model
    model = BertMaskedRegressor(
        feat_dim=args['feat_dim'],
        hidden_size=args['hidden_size'],
        encoder_layers=args['encoder_layers'],
        decoder_layers=args['decoder_layers'],
        num_heads=args['num_heads'],
        dropout=args['dropout'],
        max_seq_len=args['seq_len'],
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully")

    # Test with synthetic data
    print("\nTesting model performance...")
    batch_size = 8
    seq_len = args['seq_len']
    feat_dim = args['feat_dim']

    # Generate test data
    x_original_list = []
    x_masked_list = []
    mask_list = []

    for _ in range(batch_size):
        # Generate flight data
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

    x_original = torch.stack(x_original_list)
    x_masked = torch.stack(x_masked_list)
    mask_tensor = torch.stack(mask_list)

    # Evaluate
    with torch.no_grad():
        # Forward pass
        reconstructed = model(x_masked)

        # Compute loss
        loss, _ = model.compute_loss(x_masked, x_original, mask_tensor)

        # Compute detailed metrics
        masked_positions = (mask_tensor == 0)
        unmasked_positions = (mask_tensor == 1)

        if masked_positions.sum() > 0:
            mse_masked = torch.mean((reconstructed[masked_positions] - x_original[masked_positions]) ** 2)
            mae_masked = torch.mean(torch.abs(reconstructed[masked_positions] - x_original[masked_positions]))
        else:
            mse_masked = mae_masked = float('inf')

        if unmasked_positions.sum() > 0:
            mse_unmasked = torch.mean((reconstructed[unmasked_positions] - x_original[unmasked_positions]) ** 2)
            mae_unmasked = torch.mean(torch.abs(reconstructed[unmasked_positions] - x_original[unmasked_positions]))
        else:
            mse_unmasked = mae_unmasked = float('inf')

        masking_ratio = masked_positions.float().mean()

    print(f"✓ Test results:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Masking ratio: {masking_ratio:.3f}")
    print(f"  MSE (masked): {mse_masked.item():.4f}")
    print(f"  MAE (masked): {mae_masked.item():.4f}")
    print(f"  MSE (unmasked): {mse_unmasked.item():.4f}")
    print(f"  MAE (unmasked): {mae_unmasked.item():.4f}")

    # Performance analysis
    if mse_masked < mse_unmasked:
        print("  Note: Model reconstructs masked regions better than preserving unmasked ones")
    else:
        print("  Note: Model preserves unmasked regions better than reconstructing masked ones")

    return True


def main():
    """Test both saved models."""
    print("=" * 60)
    print("Testing Saved BERT Models")
    print("=" * 60)

    model_dir = "bert_masked_regressor_runs/fixed_test"

    models_to_test = [
        (os.path.join(model_dir, "best_model.pt"), "Best Model"),
        (os.path.join(model_dir, "final_model.pt"), "Final Model"),
    ]

    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            print(f"\n{model_name}:")
            print("-" * 40)
            try:
                test_saved_model(model_path)
                print("✓ Test passed!")
            except Exception as e:
                print(f"✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n{model_name}: Model file not found at {model_path}")

    print("\n" + "=" * 60)
    print("🎉 Model verification complete!")

    # Show training arguments
    args_path = os.path.join(model_dir, "args.json")
    if os.path.exists(args_path):
        print("\nTraining configuration:")
        with open(args_path, 'r') as f:
            args = json.load(f)
        for key, value in args.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()