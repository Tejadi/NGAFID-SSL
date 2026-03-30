#!/usr/bin/env python3
"""
Local flight dataset loader for BERT masked regression.
Loads CSV files directly from local directory instead of streaming from HuggingFace.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Iterator
import glob
import os
from .masked_flight_dataset import noise_mask

DERVIED_COLS = ['stallindex', 'aoasimple', 'densityratio', 'trueairspeed(ft/min)', 'vspdcalculated']

class LocalFlightDataset(IterableDataset):
    """
    Loads flight data from local CSV files for BERT masked regression.

    Each sample contains:
    - x_masked: Flight data with masked values set to 0
    - x_original: Original unmasked flight data
    - mask: Binary mask (1=keep, 0=masked)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 256,
        step: int = 256,
        masking_ratio: float = 0.6,
        mean_mask_length: int = 3,
        max_files: Optional[int] = None,
        seed: int = 42,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.step = step
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.max_files = max_files
        self.seed = seed
        self.train_split = train_split
        self.val_split = val_split

        # Initialize random state for reproducible masking
        self.rng = np.random.RandomState(seed)

        # Find all CSV files
        self.csv_files = self._find_csv_files()

        # Split files into train/val/test
        self.split_files = self._split_files()

        print(f"Found {len(self.csv_files)} total CSV files")
        print(f"Split '{split}': {len(self.split_files)} files")

    def _find_csv_files(self) -> List[Path]:
        """Find all CSV files in the data directory."""
        csv_files = []

        # Look for preprocessed data first (these are the flight data files)
        preprocessed_path = self.data_dir / "preprocessed_data"
        if preprocessed_path.exists():
            for split_dir in ["train", "val", "test"]:
                split_path = preprocessed_path / split_dir
                if split_path.exists():
                    csv_files.extend(list(split_path.glob("*.csv")))

        # Fallback to any CSV files
        if not csv_files:
            for pattern in ["*.csv", "**/*.csv"]:
                csv_files.extend(self.data_dir.glob(pattern))
            # Filter out metadata files
            csv_files = [f for f in csv_files if not any(name in f.name.lower() for name in ['aircraft_types', 'events', 'flight_ids', 'splits'])]

        # Sort for reproducible ordering
        csv_files.sort()

        return csv_files

    def _split_files(self) -> List[Path]:
        """Split files into train/validation/test sets."""
        total_files = len(self.csv_files)

        if total_files == 0:
            return []

        # Check if we have pre-split directories
        preprocessed_path = self.data_dir / "preprocessed_data"
        if preprocessed_path.exists():
            # Use existing directory structure
            split_mapping = {
                "train": "train",
                "validation": "val",
                "val": "val",
                "test": "test"
            }

            if self.split in split_mapping:
                split_dir = split_mapping[self.split]
                split_path = preprocessed_path / split_dir

                if split_path.exists():
                    # Get files from specific split directory
                    split_files = list(split_path.glob("*.csv"))
                    split_files.sort()

                    # Apply max_files limit
                    if self.max_files is not None:
                        split_files = split_files[:self.max_files]

                    return split_files

        # Fallback to random splitting if no pre-split directories
        # Use seed for reproducible splits
        np.random.seed(self.seed)
        indices = np.random.permutation(total_files)

        # Calculate split points
        train_end = int(self.train_split * total_files)
        val_end = train_end + int(self.val_split * total_files)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "validation" or self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            # Default to all files
            split_indices = indices

        # Apply max_files limit
        if self.max_files is not None:
            split_indices = split_indices[:self.max_files]

        return [self.csv_files[i] for i in split_indices]

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize flight data using robust statistics.

        Args:
            data: Flight data array of shape (seq_len, feat_dim)

        Returns:
            Normalized data array
        """
        normalized_data = np.copy(data)

        for i in range(data.shape[1]):  # For each feature
            feature_data = data[:, i]

            # Use median and IQR for robust normalization
            median = np.median(feature_data)
            q75, q25 = np.percentile(feature_data, [75, 25])
            iqr = q75 - q25

            # Avoid division by zero
            if iqr > 1e-6:
                normalized_data[:, i] = (feature_data - median) / iqr
            else:
                # If no variation, center around median
                normalized_data[:, i] = feature_data - median

        return normalized_data

    def __len__(self) -> int:
        return len(self.split_files)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset, yielding windowed and masked flight data.

        Yields:
            Tuple of (x_masked, x_original, mask) tensors
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Multi-worker: split data across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Set per-worker random seed
            per_worker_seed = self.seed + worker_id
            np.random.seed(per_worker_seed)

            # Determine which files this worker handles
            start_idx = worker_id
            step_size = num_workers
        else:
            # Single worker
            np.random.seed(self.seed)
            start_idx = 0
            step_size = 1

        for idx in range(start_idx, len(self.split_files), step_size):
            csv_file = self.split_files[idx]

            try:
                # Read CSV file
                df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
                df = df.drop(columns=DERVIED_COLS)

                # Select only numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_numeric = df[numeric_cols]

                # Handle missing values
                df_clean = df_numeric.ffill().bfill()

                # Convert to numpy
                flight_data = df_clean.to_numpy(dtype=np.float32)

                if flight_data.shape[0] == 0 or flight_data.shape[1] == 0:
                    continue

                # Normalize the data to prevent extremely large loss values
                # Use robust normalization (median and IQR) to handle outliers
                flight_data = self._normalize_data(flight_data)

                # Handle case where flight is shorter than seq_len
                if len(flight_data) < self.seq_len:
                    # Pad with last values
                    pad_length = self.seq_len - len(flight_data)
                    if len(flight_data) > 0:
                        last_row = flight_data[-1:]
                    else:
                        last_row = np.zeros((1, flight_data.shape[1]))
                    padding = np.repeat(last_row, pad_length, axis=0)
                    flight_data = np.vstack([flight_data, padding])

                # Slide windows over the flight data
                for start in range(0, len(flight_data) - self.seq_len + 1, self.step):
                    window = flight_data[start:start + self.seq_len]

                    if window.shape[0] != self.seq_len:
                        continue

                    # Apply masking using existing noise_mask function
                    mask = noise_mask(
                        window,
                        self.masking_ratio,
                        self.mean_mask_length,
                        mode='separate',
                        distribution='geometric'
                    )

                    # Create masked version (0 out masked positions)
                    masked_window = window * mask.astype(np.float32)

                    # Convert to tensors
                    x_masked = torch.tensor(masked_window, dtype=torch.float32)
                    x_original = torch.tensor(window, dtype=torch.float32)
                    mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32)

                    yield x_masked, x_original, mask_tensor

            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                continue


def create_local_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 16,
    seq_len: int = 256,
    masking_ratio: float = 0.6,
    mean_mask_length: int = 3,
    max_files: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> DataLoader:
    """
    Create a DataLoader for local flight data.

    Args:
        data_dir: Directory containing CSV flight data files
        split: Dataset split to use ('train', 'validation', 'test')
        batch_size: Batch size for training
        seq_len: Sequence length for windows
        masking_ratio: Ratio of values to mask
        mean_mask_length: Average length of masked segments
        max_files: Maximum number of files to process (None for all)
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
        train_split: Fraction of files for training
        val_split: Fraction of files for validation

    Returns:
        DataLoader yielding batches of (x_masked, x_original, mask)
    """
    dataset = LocalFlightDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        masking_ratio=masking_ratio,
        mean_mask_length=mean_mask_length,
        max_files=max_files,
        seed=seed,
        train_split=train_split,
        val_split=val_split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def get_feature_dim_from_local_data(data_dir: str) -> Optional[int]:
    """Get feature dimension from a sample of local flight data."""
    data_path = Path(data_dir)

    # Find CSV files, prioritizing preprocessed_data
    csv_files = []

    # Look for preprocessed data first (these are the flight data files)
    preprocessed_path = data_path / "preprocessed_data"
    if preprocessed_path.exists():
        for split_dir in ["train", "val", "test"]:
            split_path = preprocessed_path / split_dir
            if split_path.exists():
                csv_files.extend(list(split_path.glob("*.csv")))

    # Fallback to any CSV files
    if not csv_files:
        csv_files = list(data_path.glob("*.csv")) + list(data_path.glob("**/*.csv"))
        # Filter out metadata files
        csv_files = [f for f in csv_files if not any(name in f.name.lower() for name in ['aircraft_types', 'events', 'flight_ids', 'splits'])]

    if not csv_files:
        return None

    # Try to read the first few files to determine feature dimension
    for csv_file in csv_files[:5]:
        try:
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
            df = df.drop(columns=DERVIED_COLS)

            # Skip files that look like metadata (very few rows)
            if len(df) < 10:
                continue

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 5:  # Should have many features for flight data
                print(f"Feature dim detected from: {csv_file.name}")
                return len(numeric_cols)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    return None


if __name__ == "__main__":
    # Test the local dataset
    print("Testing LocalFlightDataset...")

    # Create synthetic test data first
    import sys
    sys.path.append('..')

    # Run the download script to create test data
    os.system("python download_flight_dataset.py --source synthetic --num_synthetic 10 --output_dir ./test_flight_data")

    try:
        # Test feature dimension detection
        feat_dim = get_feature_dim_from_local_data("./test_flight_data")
        print(f"Detected feature dimension: {feat_dim}")

        # Test dataloader
        dataloader = create_local_dataloader(
            data_dir="./test_flight_data",
            split="train",
            batch_size=2,
            seq_len=128,
            max_files=3,
            num_workers=0,
        )

        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  x_masked shape: {x_masked.shape}")
            print(f"  x_original shape: {x_original.shape}")
            print(f"  mask shape: {mask.shape}")
            print(f"  Masking ratio: {(mask == 0).float().mean().item():.3f}")

            if batch_idx >= 2:  # Only test a few batches
                break

        print("Local dataset test completed successfully!")

    except Exception as e:
        print(f"Local dataset test failed: {e}")
        import traceback
        traceback.print_exc()
