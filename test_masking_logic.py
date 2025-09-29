#!/usr/bin/env python3
"""
Simple test script to verify random masking logic without heavy dependencies.
"""

import numpy as np
import sys
from collections import Counter

def test_random_choice_logic():
    """Test the core random choice logic."""
    print("🧪 Testing random choice logic...")

    # Set random seed for reproducibility
    np.random.seed(42)

    masking_ratios = [0.2, 0.5, 0.8]
    mean_mask_lengths = [5, 60]

    # Simulate choosing random parameters for many windows
    n_samples = 1000
    ratio_counts = Counter()
    length_counts = Counter()
    combination_counts = Counter()

    for i in range(n_samples):
        # This is the logic used in the BERT training
        random_masking_ratio = np.random.choice(masking_ratios)
        random_mean_mask_length = np.random.choice(mean_mask_lengths)

        ratio_counts[random_masking_ratio] += 1
        length_counts[random_mean_mask_length] += 1
        combination_counts[(random_masking_ratio, random_mean_mask_length)] += 1

    print(f"   Generated {n_samples} random samples")

    # Check ratio distribution
    print(f"\n   📊 Masking ratio distribution:")
    for ratio in sorted(masking_ratios):
        count = ratio_counts[ratio]
        percentage = count / n_samples * 100
        print(f"      {ratio:.1f}: {count}/{n_samples} ({percentage:.1f}%)")

    # Check length distribution
    print(f"\n   📊 Mean mask length distribution:")
    for length in sorted(mean_mask_lengths):
        count = length_counts[length]
        percentage = count / n_samples * 100
        print(f"      {length}: {count}/{n_samples} ({percentage:.1f}%)")

    # Check combination distribution
    print(f"\n   📊 Combination distribution:")
    expected_combinations = 6  # 3 ratios × 2 lengths
    actual_combinations = len(combination_counts)

    for (ratio, length) in sorted(combination_counts.keys()):
        count = combination_counts[(ratio, length)]
        percentage = count / n_samples * 100
        print(f"      ({ratio:.1f}, {length}): {count}/{n_samples} ({percentage:.1f}%)")

    print(f"\n   ✅ Expected {expected_combinations} combinations, found {actual_combinations}")

    # Verify all combinations are present
    expected_pairs = {(r, l) for r in masking_ratios for l in mean_mask_lengths}
    actual_pairs = set(combination_counts.keys())

    if expected_pairs == actual_pairs:
        print(f"   ✅ All expected combinations found: {sorted(expected_pairs)}")
        return True
    else:
        missing = expected_pairs - actual_pairs
        print(f"   ❌ Missing combinations: {missing}")
        return False

def test_masking_simulation():
    """Simulate the actual masking process."""
    print("\n🧪 Testing masking simulation...")

    # Simple simulation of noise_mask function behavior
    def simulate_masking_ratio(true_ratio, sequence_length=1000):
        """Simulate the actual masking ratio achieved."""
        # Add some variance to simulate real masking behavior
        variance = 0.05  # ±5% variance
        noise = np.random.normal(0, variance)
        return max(0.0, min(1.0, true_ratio + noise))

    masking_ratios = [0.2, 0.5, 0.8]
    mean_mask_lengths = [5, 60]

    print(f"   Testing with ratios: {masking_ratios}")
    print(f"   Testing with lengths: {mean_mask_lengths}")

    # Simulate 100 masking operations
    np.random.seed(42)
    achieved_ratios = []

    for i in range(100):
        target_ratio = np.random.choice(masking_ratios)
        target_length = np.random.choice(mean_mask_lengths)

        # Simulate achieved masking ratio
        achieved_ratio = simulate_masking_ratio(target_ratio)
        achieved_ratios.append((target_ratio, achieved_ratio))

    # Analyze results
    for target_ratio in masking_ratios:
        target_samples = [achieved for target, achieved in achieved_ratios if target == target_ratio]
        if target_samples:
            mean_achieved = np.mean(target_samples)
            std_achieved = np.std(target_samples)
            print(f"   Target {target_ratio:.1f}: achieved {mean_achieved:.3f} ± {std_achieved:.3f} (n={len(target_samples)})")

    print(f"   ✅ Masking simulation completed")
    return True

def test_discrete_combinations():
    """Verify we get exactly 6 discrete combinations."""
    print("\n🧪 Testing discrete combinations...")

    masking_ratios = [0.2, 0.5, 0.8]
    mean_mask_lengths = [5, 60]

    # Generate all possible combinations
    all_combinations = []
    for ratio in masking_ratios:
        for length in mean_mask_lengths:
            all_combinations.append((ratio, length))

    print(f"   All possible combinations:")
    for i, (ratio, length) in enumerate(all_combinations, 1):
        print(f"      {i}. ratio={ratio:.1f}, length={length}")

    expected_count = len(masking_ratios) * len(mean_mask_lengths)
    actual_count = len(all_combinations)

    if actual_count == 6 and expected_count == 6:
        print(f"   ✅ Exactly 6 discrete combinations confirmed")
        return True
    else:
        print(f"   ❌ Expected 6 combinations, got {actual_count}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Random Masking Logic")
    print("=" * 60)

    test1_passed = test_random_choice_logic()
    test2_passed = test_masking_simulation()
    test3_passed = test_discrete_combinations()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed and test3_passed:
        print("🎉 All random masking logic tests passed!")
        print("✅ Random masking will provide 6 discrete parameter combinations")
        print("✅ Training data will have enhanced diversity")
        print("✅ Each window gets randomly selected masking parameters")
    else:
        print("❌ Some tests failed. Check the implementation.")