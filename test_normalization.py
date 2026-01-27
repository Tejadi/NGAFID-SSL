#!/usr/bin/env python3
"""
Test script to verify normalization consistency between autoencoder and BERT approaches.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def test_normalization_consistency():
    """Test that both approaches compute the same normalization parameters."""

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 44

    # Generate synthetic flight data similar to real data
    test_data = np.random.randn(n_samples, n_features) * 10 + np.random.rand(n_features) * 100

    print("🧪 Testing normalization consistency...")
    print(f"   Test data shape: {test_data.shape}")

    # Method 1: Autoencoder approach (from train_autoencoder.py)
    print("\n📊 Computing autoencoder normalization...")
    data_reshaped = test_data.reshape(-1, n_features)
    auto_mean = np.mean(data_reshaped, axis=0)
    auto_std = np.std(data_reshaped, axis=0)
    auto_std[auto_std == 0] = 1.0

    print(f"   Autoencoder mean range: [{auto_mean.min():.4f}, {auto_mean.max():.4f}]")
    print(f"   Autoencoder std range: [{auto_std.min():.4f}, {auto_std.max():.4f}]")

    # Method 2: BERT approach (from train_full_flights.py)
    print("\n🤖 Computing BERT normalization...")
    concatenated_data = test_data  # Same data, just different variable name
    bert_mean = np.mean(concatenated_data, axis=0)
    bert_std = np.std(concatenated_data, axis=0)
    bert_std[bert_std == 0] = 1.0

    print(f"   BERT mean range: [{bert_mean.min():.4f}, {bert_mean.max():.4f}]")
    print(f"   BERT std range: [{bert_std.min():.4f}, {bert_std.max():.4f}]")

    # Verify consistency
    print("\n✅ Checking consistency...")
    mean_diff = np.abs(auto_mean - bert_mean).max()
    std_diff = np.abs(auto_std - bert_std).max()

    print(f"   Max difference in means: {mean_diff:.10f}")
    print(f"   Max difference in stds: {std_diff:.10f}")

    tolerance = 1e-10
    if mean_diff < tolerance and std_diff < tolerance:
        print(f"✅ PASS: Normalization parameters are identical (within {tolerance})")
        return True
    else:
        print(f"❌ FAIL: Normalization parameters differ by more than {tolerance}")
        return False

def test_normalized_data():
    """Test that normalized data has the expected properties."""

    print("\n🧪 Testing normalized data properties...")

    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 44
    test_data = np.random.randn(n_samples, n_features) * 10 + np.random.rand(n_features) * 100

    # Compute normalization parameters
    data_mean = np.mean(test_data, axis=0)
    data_std = np.std(test_data, axis=0)
    data_std[data_std == 0] = 1.0

    # Apply normalization
    normalized_data = (test_data - data_mean) / data_std

    # Check properties
    normalized_mean = np.mean(normalized_data, axis=0)
    normalized_std = np.std(normalized_data, axis=0)

    print(f"   Original data mean range: [{data_mean.min():.4f}, {data_mean.max():.4f}]")
    print(f"   Original data std range: [{data_std.min():.4f}, {data_std.max():.4f}]")
    print(f"   Normalized data mean range: [{normalized_mean.min():.6f}, {normalized_mean.max():.6f}]")
    print(f"   Normalized data std range: [{normalized_std.min():.6f}, {normalized_std.max():.6f}]")

    # Should be approximately 0 mean, 1 std
    mean_tolerance = 1e-10
    std_tolerance = 1e-10

    if (np.abs(normalized_mean).max() < mean_tolerance and
        np.abs(normalized_std - 1.0).max() < std_tolerance):
        print("✅ PASS: Normalized data has ~0 mean and ~1 std")
        return True
    else:
        print("❌ FAIL: Normalized data doesn't have expected properties")
        return False

if __name__ == "__main__":
    print("🚀 Testing BERT normalization implementation")
    print("=" * 60)

    test1_passed = test_normalization_consistency()
    test2_passed = test_normalized_data()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Normalization implementation is correct.")
    else:
        print("❌ Some tests failed. Please check the implementation.")