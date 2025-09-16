#!/usr/bin/env python3
"""
Debug script to understand the data range and identify scaling issues.
"""

import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ngafid_datasets.local_flight_dataset import create_local_dataloader


def analyze_data_ranges():
    """Analyze the range and statistics of the flight data."""
    print("Analyzing flight data ranges...")

    # Read a few sample files
    sample_files = [
        "NGAFID-LOCI-GATS-Data/preprocessed_data/train/Cessna_172S_flight_100.csv",
        "NGAFID-LOCI-GATS-Data/preprocessed_data/train/PA-28-181_flight_6918.csv"
    ]

    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"\n--- {os.path.basename(file_path)} ---")
            df = pd.read_csv(file_path, na_values=[' NaN', 'NaN', 'NaN ', 'nan', ''])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

            print(f"Shape: {df_numeric.shape}")
            print(f"Columns: {len(numeric_cols)}")

            # Statistics for each column
            stats = df_numeric.describe()
            print("\nData ranges:")
            for col in df_numeric.columns:
                min_val = stats.loc['min', col]
                max_val = stats.loc['max', col]
                mean_val = stats.loc['mean', col]
                std_val = stats.loc['std', col]
                print(f"  {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, std={std_val:.2f}")

            # Look for extremely large values
            max_abs = np.abs(df_numeric.values).max()
            print(f"\nMaximum absolute value: {max_abs:.2e}")

            # Check for common flight data issues
            very_large_cols = []
            for col in df_numeric.columns:
                if np.abs(df_numeric[col]).max() > 10000:
                    very_large_cols.append(col)

            if very_large_cols:
                print(f"Columns with very large values (>10000): {very_large_cols}")

            break


def test_dataloader_values():
    """Test the actual values coming from the dataloader."""
    print("\n" + "="*60)
    print("Testing dataloader values...")

    dataloader = create_local_dataloader(
        data_dir="./NGAFID-LOCI-GATS-Data",
        split="train",
        batch_size=2,
        seq_len=64,
        max_files=1,
        num_workers=0,
        seed=42,
    )

    for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Shapes: {x_masked.shape}")

        # Statistics
        print(f"  Original data range: [{x_original.min().item():.2e}, {x_original.max().item():.2e}]")
        print(f"  Original data mean: {x_original.mean().item():.2e}")
        print(f"  Original data std: {x_original.std().item():.2e}")

        print(f"  Masked data range: [{x_masked.min().item():.2e}, {x_masked.max().item():.2e}]")
        print(f"  Masked data mean: {x_masked.mean().item():.2e}")
        print(f"  Masked data std: {x_masked.std().item():.2e}")

        # Check for inf/nan
        print(f"  Has inf: {torch.isinf(x_original).any().item()}")
        print(f"  Has nan: {torch.isnan(x_original).any().item()}")

        # Test a simple loss
        mse_loss = torch.nn.functional.mse_loss(x_masked, x_original)
        print(f"  Simple MSE loss: {mse_loss.item():.2e}")

        if batch_idx >= 1:
            break


def main():
    print("=" * 60)
    print("Debugging Flight Data Issues")
    print("=" * 60)

    analyze_data_ranges()
    test_dataloader_values()

    print("\n" + "=" * 60)
    print("Debug complete.")


if __name__ == "__main__":
    main()