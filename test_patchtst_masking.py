#!/usr/bin/env python3
"""
Quick test to verify PatchTST training setup with random masking.
"""

import sys
import argparse

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from models.patchtst_masked_regressor import PatchTSTMaskedRegressor, count_parameters
        from train_full_flights import GlobalNormalizedFlightDataset, compute_normalization_parameters
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_random_masking_args():
    """Test argument parsing for random masking."""
    print("\nTesting argument parsing...")
    sys.argv = [
        'test',
        '--local_data_dir', './NGAFID-LOCI-GATS-Data',
        '--use_random_masking',
        '--masking_ratios', '0.2', '0.5', '0.8',
        '--mean_mask_lengths', '5', '60',
    ]

    try:
        from train_patchtst_masked_regressor import parse_args
        args = parse_args()

        assert args.use_random_masking == True
        assert args.masking_ratios == [0.2, 0.5, 0.8]
        assert args.mean_mask_lengths == [5, 60]

        print(f"✅ Random masking enabled: {args.use_random_masking}")
        print(f"✅ Masking ratios: {args.masking_ratios}")
        print(f"✅ Mean mask lengths: {args.mean_mask_lengths}")
        print(f"   Expected combinations: {len(args.masking_ratios)} × {len(args.mean_mask_lengths)} = {len(args.masking_ratios) * len(args.mean_mask_lengths)}")
        return True
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False

def main():
    print("🚀 Testing PatchTST Random Masking Setup")
    print("=" * 60)

    test1 = test_imports()
    test2 = test_random_masking_args()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("🎉 All tests passed!")
        print("✅ PatchTST is ready for training with random masking")
        print("✅ Matching BERT setup: ratios [0.2, 0.5, 0.8] × lengths [5, 60]")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
