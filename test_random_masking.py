#!/usr/bin/env python3
"""
Test script to verify that random masking is working correctly in BERT training.
"""

import numpy as np
import pandas as pd
import tempfile
import torch
from pathlib import Path
from collections import Counter
import sys

# Add current directory to path to import our modules
sys.path.append('.')

def create_test_data(temp_dir, n_flights=3, n_features=44):
    """Create synthetic CSV files for testing."""
    temp_path = Path(temp_dir)

    # Create train directory structure
    train_dir = temp_path / "preprocessed_data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    feature_names = [f"feature_{j}" for j in range(n_features)]

    for i in range(n_flights):
        # Generate a flight with varying length
        flight_length = np.random.randint(100, 300)
        flight_data = np.random.randn(flight_length, n_features) * 10 + np.random.rand(n_features) * 100

        # Create DataFrame
        df = pd.DataFrame(flight_data, columns=feature_names)

        # Save to CSV
        csv_path = train_dir / f"flight_{i:03d}.csv"
        df.to_csv(csv_path, index=False)

    return str(temp_path)

def test_random_masking():
    """Test that random masking produces different parameter combinations."""
    print("🧪 Testing random masking implementation...")

    # Import our modules
    try:
        from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters
        from ngafid_datasets.masked_flight_dataset import noise_mask
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data_dir = create_test_data(temp_dir)
        print(f"   Created test data in: {data_dir}")

        # Compute normalization parameters
        try:
            normalization_params = compute_normalization_parameters(data_dir, max_files=10)
            print(f"   ✅ Computed normalization parameters")
        except Exception as e:
            print(f"   ❌ Error computing normalization: {e}")
            normalization_params = None

        # Test dataset with random masking
        masking_ratios = [0.2, 0.5, 0.8]
        mean_mask_lengths = [5, 60]

        dataset = GlobalNormalizedFlightDataset(
            normalization_params=normalization_params,
            data_dir=data_dir,
            split="train",
            seq_len=100,  # Short sequences for fast testing
            max_files=3,
            seed=42,
            use_random_masking=True,
            masking_ratios=masking_ratios,
            mean_mask_lengths=mean_mask_lengths,
        )

        print(f"   ✅ Created dataset with random masking")

        # Sample multiple batches and track masking parameters
        masking_combinations = []
        masking_ratios_observed = []

        sample_count = 0
        max_samples = 50  # Test with 50 samples

        for x_masked, x_original, mask in dataset:
            # Calculate actual masking ratio
            actual_masking_ratio = (mask == 0).float().mean().item()
            masking_ratios_observed.append(actual_masking_ratio)

            sample_count += 1
            if sample_count >= max_samples:
                break

        print(f"   📊 Collected {sample_count} samples")

        # Analyze masking ratio distribution
        ratio_counter = Counter()
        for ratio in masking_ratios_observed:
            # Round to nearest expected ratio for counting
            closest_ratio = min(masking_ratios, key=lambda x: abs(x - ratio))
            if abs(ratio - closest_ratio) < 0.05:  # Within 5% tolerance
                ratio_counter[closest_ratio] += 1

        print(f"   🎲 Observed masking ratio distribution:")
        total_samples = sum(ratio_counter.values())
        for ratio in sorted(masking_ratios):
            count = ratio_counter[ratio]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"      {ratio:.1f}: {count}/{total_samples} ({percentage:.1f}%)")

        # Check if all expected ratios were used
        expected_ratios_used = set(ratio_counter.keys())
        expected_ratios = set(masking_ratios)

        if expected_ratios_used == expected_ratios:
            print(f"   ✅ All expected masking ratios were used")
        else:
            missing = expected_ratios - expected_ratios_used
            print(f"   ⚠️  Missing ratios: {missing}")

        # Check for reasonable distribution (not all samples using same ratio)
        if len(ratio_counter) > 1:
            print(f"   ✅ Random masking is working - multiple ratios observed")
            return True
        else:
            print(f"   ❌ Random masking may not be working - only one ratio observed")
            return False

def test_fixed_vs_random():
    """Compare fixed masking vs random masking behavior."""
    print("\n🧪 Testing fixed vs random masking behavior...")

    try:
        from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data_dir = create_test_data(temp_dir)

        # Compute normalization parameters
        try:
            normalization_params = compute_normalization_parameters(data_dir, max_files=10)
        except Exception as e:
            normalization_params = None

        # Test fixed masking dataset
        fixed_dataset = GlobalNormalizedFlightDataset(
            normalization_params=normalization_params,
            data_dir=data_dir,
            split="train",
            seq_len=100,
            max_files=3,
            seed=42,
            use_random_masking=False,  # Fixed masking
            masking_ratio=0.6,
            mean_mask_length=3,
        )

        # Test random masking dataset
        random_dataset = GlobalNormalizedFlightDataset(
            normalization_params=normalization_params,
            data_dir=data_dir,
            split="train",
            seq_len=100,
            max_files=3,
            seed=42,
            use_random_masking=True,  # Random masking
            masking_ratios=[0.2, 0.5, 0.8],
            mean_mask_lengths=[5, 60],
        )

        # Sample from both datasets
        fixed_ratios = []
        random_ratios = []

        # Get 10 samples from each
        for i, (x_masked, x_original, mask) in enumerate(fixed_dataset):
            if i >= 10: break
            ratio = (mask == 0).float().mean().item()
            fixed_ratios.append(ratio)

        for i, (x_masked, x_original, mask) in enumerate(random_dataset):
            if i >= 10: break
            ratio = (mask == 0).float().mean().item()
            random_ratios.append(ratio)

        # Analyze variance
        fixed_variance = np.var(fixed_ratios)
        random_variance = np.var(random_ratios)

        print(f"   Fixed masking ratios: {[f'{r:.3f}' for r in fixed_ratios[:5]]}...")
        print(f"   Random masking ratios: {[f'{r:.3f}' for r in random_ratios[:5]]}...")
        print(f"   Fixed masking variance: {fixed_variance:.6f}")
        print(f"   Random masking variance: {random_variance:.6f}")

        if random_variance > fixed_variance * 2:  # Random should have higher variance
            print(f"   ✅ Random masking shows higher variance as expected")
            return True
        else:
            print(f"   ⚠️  Random masking variance not significantly higher than fixed")
            return False

if __name__ == "__main__":
    print("🚀 Testing BERT random masking implementation")
    print("=" * 60)

    test1_passed = test_random_masking()
    test2_passed = test_fixed_vs_random()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All random masking tests passed!")
        print("✅ BERT training will use diverse masking patterns")
        print("✅ 6 combinations: {0.2, 0.5, 0.8} × {5, 60} = 6 discrete pairs")
    else:
        print("❌ Some random masking tests failed.")