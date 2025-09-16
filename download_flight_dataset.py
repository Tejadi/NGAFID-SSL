#!/usr/bin/env python3
"""
Download and prepare NGAFID flight data for training.
This script handles downloading from multiple sources and formats.
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import argparse


def download_kaggle_dataset(dataset_name: str, output_dir: str) -> bool:
    """Download dataset from Kaggle using kaggle API."""
    try:
        import kaggle

        # Set up kaggle API
        kaggle.api.authenticate()

        print(f"Downloading {dataset_name} from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )

        print(f"✓ Successfully downloaded to {output_dir}")
        return True

    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False


def download_zenodo_dataset(zenodo_id: str, output_dir: str) -> bool:
    """Download dataset from Zenodo."""
    try:
        # Zenodo API URL for the record
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"

        print(f"Fetching metadata from Zenodo record {zenodo_id}...")
        response = requests.get(api_url)
        response.raise_for_status()

        data = response.json()
        files = data['files']

        os.makedirs(output_dir, exist_ok=True)

        for file_info in files:
            file_url = file_info['links']['self']
            filename = file_info['key']
            file_size = file_info['size']

            print(f"Downloading {filename} ({file_size / 1024 / 1024:.1f} MB)...")

            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()

            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"✓ Downloaded {filename}")

            # If it's a zip file, extract it
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                os.remove(output_path)  # Remove zip after extraction

        return True

    except Exception as e:
        print(f"Error downloading from Zenodo: {e}")
        return False


def analyze_flight_data(data_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Analyze flight data to determine feature dimension and structure."""

    data_path = Path(data_dir)

    # Look for CSV files
    csv_files = list(data_path.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None

    print(f"Found {len(csv_files)} CSV files")

    # Analyze a few sample files to understand structure
    sample_files = csv_files[:5]  # Check first 5 files

    for i, csv_file in enumerate(sample_files):
        try:
            print(f"\nAnalyzing file {i+1}: {csv_file.name}")

            # Try to read the CSV
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Column names: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

            # Check data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(f"  Numeric columns: {len(numeric_cols)}")

            # Check for missing data
            missing_pct = (df.isnull().sum() / len(df) * 100).mean()
            print(f"  Average missing data: {missing_pct:.1f}%")

            # This looks like flight data if it has a reasonable number of numeric columns
            if len(numeric_cols) >= 10:  # Flight data typically has many sensor readings
                print(f"  ✓ This looks like flight data!")

                # Clean and check final feature dimension
                df_numeric = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                flight_data = df_numeric.to_numpy(dtype=np.float32)

                print(f"  Final feature dimension: {flight_data.shape[1]}")
                print(f"  Sample sequence length: {flight_data.shape[0]}")

                return flight_data.shape[1], str(csv_file.parent)

        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
            continue

    return None, None


def create_synthetic_flight_dataset(output_dir: str,
                                    num_flights: int = 100,
                                    feat_dim: int = 20,
                                    min_seq_len: int = 200,
                                    max_seq_len: int = 1000):
    """Create synthetic flight dataset for testing."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {num_flights} synthetic flight files...")

    np.random.seed(42)

    for i in range(num_flights):
        # Random sequence length
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)

        # Create realistic flight data
        flight_data = []

        # Base values (aircraft state at start)
        base_values = np.random.randn(feat_dim) * 10

        # Generate time series with realistic flight patterns
        for t in range(seq_len):
            # Add some realistic trends and noise
            trend = base_values + 0.01 * t * np.random.randn(feat_dim)
            noise = np.random.randn(feat_dim) * 0.5

            # Add some periodic components (like engine cycles, etc.)
            periodic = np.sin(2 * np.pi * t / 50) * np.random.randn(feat_dim) * 2

            row = trend + noise + periodic
            flight_data.append(row)

        # Convert to DataFrame with realistic column names
        columns = [
            'altitude', 'airspeed', 'ground_speed', 'vertical_speed', 'heading',
            'pitch', 'roll', 'yaw_rate', 'engine_rpm', 'fuel_flow',
            'oil_pressure', 'oil_temp', 'cylinder_head_temp', 'exhaust_temp',
            'manifold_pressure', 'throttle_pos', 'mixture_pos', 'flap_pos',
            'gear_pos', 'nav_freq'
        ][:feat_dim]

        if len(columns) < feat_dim:
            # Add generic sensor columns if we need more
            columns.extend([f'sensor_{j}' for j in range(len(columns), feat_dim)])

        df = pd.DataFrame(flight_data, columns=columns)

        # Save to CSV
        flight_file = os.path.join(output_dir, f"synthetic_flight_{i:04d}.csv")
        df.to_csv(flight_file, index=False)

        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{num_flights} files...")

    print(f"✓ Created {num_flights} synthetic flight files in {output_dir}")
    return feat_dim, output_dir


def main():
    parser = argparse.ArgumentParser(description="Download and prepare flight dataset")
    parser.add_argument("--output_dir", type=str, default="./flight_data",
                        help="Output directory for dataset")
    parser.add_argument("--source", type=str, choices=['kaggle', 'zenodo', 'synthetic'],
                        default='synthetic', help="Data source")
    parser.add_argument("--kaggle_dataset", type=str,
                        default="hooong/aviation-maintenance-dataset-from-the-ngafid",
                        help="Kaggle dataset name")
    parser.add_argument("--zenodo_id", type=str, default="6624956",
                        help="Zenodo record ID")
    parser.add_argument("--num_synthetic", type=int, default=100,
                        help="Number of synthetic flights to generate")

    args = parser.parse_args()

    print("=" * 60)
    print("Flight Dataset Preparation")
    print("=" * 60)

    success = False

    if args.source == 'kaggle':
        success = download_kaggle_dataset(args.kaggle_dataset, args.output_dir)
    elif args.source == 'zenodo':
        success = download_zenodo_dataset(args.zenodo_id, args.output_dir)
    elif args.source == 'synthetic':
        feat_dim, data_dir = create_synthetic_flight_dataset(
            args.output_dir,
            num_flights=args.num_synthetic
        )
        success = True
        print(f"\n✓ Synthetic dataset ready!")
        print(f"  Feature dimension: {feat_dim}")
        print(f"  Data directory: {data_dir}")
        print(f"  Files: {args.num_synthetic}")
        return

    if success:
        print(f"\n📊 Analyzing downloaded data...")
        feat_dim, data_dir = analyze_flight_data(args.output_dir)

        if feat_dim:
            print(f"\n✓ Dataset ready for training!")
            print(f"  Feature dimension: {feat_dim}")
            print(f"  Data directory: {data_dir}")
            print(f"\nTo train the model, run:")
            print(f"python train_bert_masked_regressor.py --local_data_dir {data_dir} --feat_dim {feat_dim}")
        else:
            print("❌ Could not determine dataset structure")
    else:
        print("❌ Failed to download dataset")
        print("Try using synthetic data: python download_flight_dataset.py --source synthetic")


if __name__ == "__main__":
    main()