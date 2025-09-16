#!/usr/bin/env python3
"""
Example usage of the BERT Masked Regressor for flight data.

This shows how to use the model with the HuggingFace dataset.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="BERT Masked Regressor Example")
    parser.add_argument("--test_only", action="store_true",
                        help="Run quick test with synthetic data")
    args = parser.parse_args()

    if args.test_only:
        print("Running synthetic data test...")
        import subprocess
        result = subprocess.run([
            "python", "test_bert_implementation.py"
        ])
        return result.returncode

    print("Training BERT Masked Regressor on flight data...")
    print("\nTo train the model, run:")
    print("python train_bert_masked_regressor.py --job_name my_experiment --epochs 5")
    print("\nTo test with synthetic data, run:")
    print("python example_usage.py --test_only")

    print("\nExample training command with options:")
    print("python train_bert_masked_regressor.py \\")
    print("    --job_name bert_flight_experiment \\")
    print("    --hidden_size 512 \\")
    print("    --encoder_layers 6 \\")
    print("    --decoder_layers 3 \\")
    print("    --batch_size 16 \\")
    print("    --learning_rate 1e-4 \\")
    print("    --epochs 10 \\")
    print("    --masking_ratio 0.6 \\")
    print("    --seq_len 256")


if __name__ == "__main__":
    main()