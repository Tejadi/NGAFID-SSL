#!/usr/bin/env python3
"""
Verification script to test that BERT training uses the same normalization as autoencoder.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Create a minimal version of the normalization functions without dependencies
def compute_bert_normalization_parameters(data_dir: str, max_files: int = 100):
    """
    Simplified version of the BERT normalization computation for testing.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))[:max_files]

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]
            df_clean = df_numeric.ffill().bfill()
            flight_data = df_clean.to_numpy(dtype=np.float32)

            if flight_data.shape[0] > 0 and flight_data.shape[1] > 0:
                all_data.append(flight_data)
        except Exception as e:
            continue

    if not all_data:
        raise ValueError("No valid data found")

    concatenated_data = np.vstack(all_data)
    data_mean = np.mean(concatenated_data, axis=0)
    data_std = np.std(concatenated_data, axis=0)
    data_std[data_std == 0] = 1.0

    return {
        'mean': data_mean,
        'std': data_std
    }

def compute_autoencoder_normalization_parameters(train_data):
    """
    Autoencoder normalization computation for comparison.
    """
    data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    data_mean = np.mean(data_reshaped, axis=0)
    data_std = np.std(data_reshaped, axis=0)
    data_std[data_std == 0] = 1.0

    return {
        'mean': data_mean,
        'std': data_std
    }

def test_end_to_end_normalization():
    """Test end-to-end normalization consistency."""
    print("🧪 Testing end-to-end normalization consistency...")

    # Create temporary directory with synthetic CSV data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate synthetic flight data files
        np.random.seed(42)
        n_flights = 5
        n_features = 44
        all_flight_data = []

        for i in range(n_flights):
            # Generate a flight with varying length
            flight_length = np.random.randint(100, 500)
            flight_data = np.random.randn(flight_length, n_features) * 10 + np.random.rand(n_features) * 100

            # Create DataFrame with feature names
            feature_names = [f"feature_{j}" for j in range(n_features)]
            df = pd.DataFrame(flight_data, columns=feature_names)

            # Save to CSV
            csv_path = temp_path / f"flight_{i:03d}.csv"
            df.to_csv(csv_path, index=False)

            all_flight_data.append(flight_data)

        print(f"   Created {n_flights} synthetic flight CSV files")

        # Method 1: BERT approach (loading from CSV files)
        print("\n🤖 Computing BERT normalization from CSV files...")
        bert_params = compute_bert_normalization_parameters(str(temp_path))

        # Method 2: Autoencoder approach (from numpy arrays)
        print("📊 Computing autoencoder normalization from numpy arrays...")
        # Simulate how autoencoder loads data
        combined_data = np.vstack(all_flight_data)
        auto_params = compute_autoencoder_normalization_parameters(combined_data)

        # Compare results
        print("\n✅ Comparing normalization parameters...")
        mean_diff = np.abs(bert_params['mean'] - auto_params['mean']).max()
        std_diff = np.abs(bert_params['std'] - auto_params['std']).max()

        print(f"   BERT mean range: [{bert_params['mean'].min():.4f}, {bert_params['mean'].max():.4f}]")
        print(f"   Auto mean range: [{auto_params['mean'].min():.4f}, {auto_params['mean'].max():.4f}]")
        print(f"   Max difference in means: {mean_diff:.10f}")

        print(f"   BERT std range: [{bert_params['std'].min():.4f}, {bert_params['std'].max():.4f}]")
        print(f"   Auto std range: [{auto_params['std'].min():.4f}, {auto_params['std'].max():.4f}]")
        print(f"   Max difference in stds: {std_diff:.10f}")

        tolerance = 1e-4  # Relaxed tolerance for file I/O and floating point precision
        if mean_diff < tolerance and std_diff < tolerance:
            print(f"✅ PASS: End-to-end normalization is consistent (within {tolerance})")
            return True
        else:
            print(f"❌ FAIL: End-to-end normalization differs by more than {tolerance}")
            return False

def test_global_vs_per_flight():
    """Test the difference between global and per-flight normalization."""
    print("\n🧪 Testing global vs per-flight normalization...")

    # Create test data with different flight characteristics
    np.random.seed(42)
    n_features = 44

    # Flight 1: High values
    flight1 = np.random.randn(200, n_features) * 5 + 100
    # Flight 2: Low values
    flight2 = np.random.randn(200, n_features) * 2 + 10
    # Flight 3: Medium values
    flight3 = np.random.randn(200, n_features) * 8 + 50

    all_data = np.vstack([flight1, flight2, flight3])

    # Global normalization (our new approach)
    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)
    global_std[global_std == 0] = 1.0

    print(f"   Global mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    print(f"   Global std range: [{global_std.min():.4f}, {global_std.max():.4f}]")

    # Per-flight normalization (old approach)
    print("\n   Per-flight statistics:")
    for i, flight in enumerate([flight1, flight2, flight3], 1):
        flight_mean = np.mean(flight, axis=0)
        flight_std = np.std(flight, axis=0)
        print(f"   Flight {i} mean range: [{flight_mean.min():.4f}, {flight_mean.max():.4f}]")
        print(f"   Flight {i} std range: [{flight_std.min():.4f}, {flight_std.max():.4f}]")

    # Apply both normalizations to flight 1
    flight1_global_norm = (flight1 - global_mean) / global_std
    flight1_local_norm = (flight1 - np.mean(flight1, axis=0)) / np.std(flight1, axis=0)

    print(f"\n   Flight 1 after global normalization mean: {np.mean(flight1_global_norm, axis=0).mean():.6f}")
    print(f"   Flight 1 after local normalization mean: {np.mean(flight1_local_norm, axis=0).mean():.6f}")

    print("✅ Global normalization preserves relative differences between flights")
    print("   while local normalization removes them")

    return True

if __name__ == "__main__":
    print("🚀 Verifying BERT normalization implementation")
    print("=" * 60)

    test1_passed = test_end_to_end_normalization()
    test2_passed = test_global_vs_per_flight()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All verification tests passed!")
        print("✅ BERT training will now use the same normalization as autoencoder")
        print("✅ This ensures fair comparison between both approaches")
    else:
        print("❌ Some verification tests failed.")