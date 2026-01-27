#!/usr/bin/env python3
"""
Standalone script to compute normalization parameters from training data.
This replicates the normalization computation from train_full_flights.py
without running the actual training loop.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def compute_normalization_parameters(data_dir: str, max_files: int = None):
    """
    Compute global normalization parameters from training data.
    Uses the same approach as the BERT training script.

    Args:
        data_dir: Directory containing CSV flight data files
        max_files: Maximum number of files to use (None = use all)

    Returns:
        Dictionary containing 'mean' and 'std' arrays
    """
    print("Computing global normalization parameters...")

    # Find training data files
    data_path = Path(data_dir)
    train_files = []

    # Look for preprocessed training data
    preprocessed_path = data_path / "preprocessed_data" / "train"
    if preprocessed_path.exists():
        print(f"Using preprocessed training data from: {preprocessed_path}")
        train_files = list(preprocessed_path.glob("*.csv"))
    else:
        # Fallback to all CSV files
        print(f"Using all CSV files from: {data_path}")
        train_files = list(data_path.glob("*.csv"))
        # Filter out metadata files
        train_files = [f for f in train_files if not any(name in f.name.lower()
                      for name in ['aircraft_types', 'events', 'flight_ids', 'splits'])]

    if max_files is not None:
        train_files = train_files[:max_files]

    if not train_files:
        raise ValueError(f"No training CSV files found in {data_dir}")

    print(f"Using {len(train_files)} files to compute normalization parameters")

    # Collect all data for computing global statistics
    all_data = []

    for csv_file in tqdm(train_files, desc="Loading training data"):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]

            # Handle missing values (forward fill then backward fill)
            df_clean = df_numeric.ffill().bfill()

            # Convert to numpy
            flight_data = df_clean.to_numpy(dtype=np.float32)

            if flight_data.shape[0] > 0 and flight_data.shape[1] > 0:
                all_data.append(flight_data)

        except Exception as e:
            print(f"Warning: Error processing {csv_file.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data found for computing normalization parameters")

    # Concatenate all data
    print("Concatenating all flight data...")
    concatenated_data = np.vstack(all_data)
    print(f"Total data shape: {concatenated_data.shape}")
    print(f"  - Total timesteps: {concatenated_data.shape[0]:,}")
    print(f"  - Number of features: {concatenated_data.shape[1]}")

    # Compute global mean and std (same as autoencoder)
    print("Computing mean and standard deviation...")
    data_mean = np.mean(concatenated_data, axis=0)
    data_std = np.std(concatenated_data, axis=0)

    # Avoid division by zero (same as autoencoder)
    data_std[data_std == 0] = 1.0

    print(f"\nNormalization parameters computed for {len(data_mean)} features:")
    print(f"  - Mean range: [{data_mean.min():.4f}, {data_mean.max():.4f}]")
    print(f"  - Std range: [{data_std.min():.4f}, {data_std.max():.4f}]")

    # Check for any unusual values
    zero_std_features = np.sum(data_std == 1.0)
    if zero_std_features > 0:
        print(f"  - Warning: {zero_std_features} features had zero std (set to 1.0)")

    return {
        'mean': data_mean,
        'std': data_std
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save normalization parameters from flight data"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/ngafid',
        help='Path to directory containing flight CSV files (default: /data/ngafid)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to use for computing stats (default: use all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save normalization parameters (default: ./results)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output filename (default: bert_normalization_params_TIMESTAMP.npy)'
    )

    args = parser.parse_args()

    # Compute normalization parameters
    norm_params = compute_normalization_parameters(
        data_dir=args.data_dir,
        max_files=args.max_files
    )

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"bert_normalization_params_{timestamp}.npy"
    else:
        output_name = args.output_name

    output_path = output_dir / output_name

    # Save normalization parameters
    print(f"\nSaving normalization parameters to: {output_path}")
    np.save(output_path, norm_params)

    print("\nDone! You can load these parameters with:")
    print(f"  norm_params = np.load('{output_path}', allow_pickle=True).item()")
    print("\nThe dictionary contains:")
    print("  - 'mean': array of feature means")
    print("  - 'std': array of feature standard deviations")


if __name__ == "__main__":
    main()
