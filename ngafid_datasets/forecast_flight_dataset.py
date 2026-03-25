#!/usr/bin/env python3
"""
Forecast flight dataset for BERT-based next-token prediction.

This dataset masks the END of each flight sequence instead of random positions,
creating a forecasting task where the model must predict future timesteps
using only past context.

Key difference from masked regression:
- Masked regression: Random masking throughout sequence (bidirectional context)
- Forecasting: Mask only the end of sequence (causal/unidirectional context)
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Dict
import os


class ForecastFlightDataset(IterableDataset):
    """
    Dataset for flight forecasting - masks the END of each flight.

    The forecast horizon is calculated as a ratio of the ORIGINAL (pre-padded)
    flight length, ensuring consistent forecasting difficulty across flights
    of different lengths.

    Supports random sampling from multiple forecast ratios for training diversity.

    Mask structure for a flight of length N padded to seq_len:
        [1, 1, ..., 1,   0, 0, ..., 0,   1, 1, ..., 1]
         └─ context ─┘   └─ forecast ─┘   └─ padding ─┘
         (N - horizon)    (horizon)        (seq_len - N)

    Setting padding mask to 1 means loss won't be computed there
    (since loss computation uses mask == 0 to find target positions).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 10000,
        forecast_ratios: Optional[List[float]] = None,
        forecast_ratio: float = 0.2,
        min_forecast_horizon: int = 100,
        max_forecast_horizon: Optional[int] = None,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
        max_files: Optional[int] = None,
        seed: int = 42,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        """
        Initialize the forecast dataset.

        Args:
            data_dir: Directory containing CSV flight data files
            split: Dataset split ('train', 'validation', 'test')
            seq_len: Sequence length to pad flights to (default 10000)
            forecast_ratios: List of forecast ratios to randomly sample from.
                             If provided, forecast_ratio is ignored.
                             Default: [0.1, 0.2, 0.3]
            forecast_ratio: Fixed forecast ratio (only used if forecast_ratios is None)
            min_forecast_horizon: Minimum number of timesteps to predict
            max_forecast_horizon: Maximum number of timesteps to predict (None for no limit)
            normalization_params: Dict with 'mean' and 'std' arrays for global normalization
            max_files: Maximum number of files to process (None for all)
            seed: Random seed for reproducibility
            train_split: Fraction of files for training
            val_split: Fraction of files for validation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len

        # Random forecast ratio sampling (default behavior)
        if forecast_ratios is not None:
            self.forecast_ratios = forecast_ratios
            self.use_random_ratio = True
        else:
            # Default to random sampling from [0.1, 0.2, 0.3]
            self.forecast_ratios = [0.1, 0.2, 0.3]
            self.use_random_ratio = True

        self.forecast_ratio = forecast_ratio  # Fallback for single ratio mode
        self.min_forecast_horizon = min_forecast_horizon
        self.max_forecast_horizon = max_forecast_horizon
        self.normalization_params = normalization_params
        self.max_files = max_files
        self.seed = seed
        self.train_split = train_split
        self.val_split = val_split

        # Find and split CSV files
        self.csv_files = self._find_csv_files()
        self.split_files = self._split_files()

        print(f"ForecastFlightDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Files: {len(self.split_files)}")
        print(f"  Sequence length: {seq_len}")
        if self.use_random_ratio:
            print(f"  Forecast ratios (random): {self.forecast_ratios}")
        else:
            print(f"  Forecast ratio (fixed): {forecast_ratio}")
        print(f"  Min forecast horizon: {min_forecast_horizon}")
        print(f"  Max forecast horizon: {max_forecast_horizon or 'unlimited'}")

    def _find_csv_files(self) -> List[Path]:
        """Find all CSV files in the data directory."""
        csv_files = []

        # Look for preprocessed data first
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
            csv_files = [f for f in csv_files if not any(
                name in f.name.lower()
                for name in ['aircraft_types', 'events', 'flight_ids', 'splits']
            )]

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
                    split_files = list(split_path.glob("*.csv"))
                    split_files.sort()

                    if self.max_files is not None:
                        split_files = split_files[:self.max_files]

                    return split_files

        # Fallback to random splitting
        np.random.seed(self.seed)
        indices = np.random.permutation(total_files)

        train_end = int(self.train_split * total_files)
        val_end = train_end + int(self.val_split * total_files)

        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "validation" or self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            split_indices = indices

        if self.max_files is not None:
            split_indices = split_indices[:self.max_files]

        return [self.csv_files[i] for i in split_indices]

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply global normalization using precomputed mean and std.
        Falls back to per-flight robust normalization if no global params.
        """
        if self.normalization_params is None:
            # Per-flight robust normalization (median and IQR)
            normalized_data = np.copy(data)
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                median = np.median(feature_data)
                q75, q25 = np.percentile(feature_data, [75, 25])
                iqr = q75 - q25
                if iqr > 1e-6:
                    normalized_data[:, i] = (feature_data - median) / iqr
                else:
                    normalized_data[:, i] = feature_data - median
            return normalized_data

        # Global normalization
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']

        if mean.shape[0] != data.shape[1] or std.shape[0] != data.shape[1]:
            print(f"Warning: Dimension mismatch. Data: {data.shape[1]}, Params: {mean.shape[0]}")
            # Fallback to per-flight normalization
            return self._normalize_data_robust(data)

        return ((data - mean) / std).astype(np.float32)

    def _normalize_data_robust(self, data: np.ndarray) -> np.ndarray:
        """Per-flight robust normalization using median and IQR."""
        normalized_data = np.copy(data)
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            median = np.median(feature_data)
            q75, q25 = np.percentile(feature_data, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-6:
                normalized_data[:, i] = (feature_data - median) / iqr
            else:
                normalized_data[:, i] = feature_data - median
        return normalized_data

    def _calculate_forecast_horizon(self, original_length: int, forecast_ratio: Optional[float] = None) -> int:
        """
        Calculate forecast horizon based on original flight length.

        Args:
            original_length: Length of the flight before padding
            forecast_ratio: Ratio to use (if None, samples from self.forecast_ratios)

        Returns:
            Number of timesteps to forecast
        """
        # Use provided ratio or sample from list
        if forecast_ratio is None:
            if self.use_random_ratio:
                forecast_ratio = np.random.choice(self.forecast_ratios)
            else:
                forecast_ratio = self.forecast_ratio

        # Calculate horizon as ratio of original length
        horizon = int(original_length * forecast_ratio)

        # Apply minimum
        horizon = max(horizon, self.min_forecast_horizon)

        # Apply maximum if specified
        if self.max_forecast_horizon is not None:
            horizon = min(horizon, self.max_forecast_horizon)

        # Ensure we have enough context (at least 50% of flight for context)
        max_allowed = original_length // 2
        if horizon > max_allowed:
            horizon = max_allowed

        # Ensure horizon is at least 1
        horizon = max(horizon, 1)

        return horizon

    def __len__(self) -> int:
        return len(self.split_files)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset, yielding forecast-masked flight data.

        Yields:
            Tuple of (x_masked, x_original, mask) tensors where:
            - x_masked: Flight data with forecast portion zeroed out
            - x_original: Original flight data (padded to seq_len)
            - mask: Binary mask (1=context/padding, 0=forecast targets)
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            np.random.seed(self.seed + worker_id)
            start_idx = worker_id
            step_size = num_workers
        else:
            np.random.seed(self.seed)
            start_idx = 0
            step_size = 1

        for idx in range(start_idx, len(self.split_files), step_size):
            csv_file = self.split_files[idx]

            try:
                # Read CSV file
                df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

                # Select only numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_numeric = df[numeric_cols]

                # Handle missing values
                df_clean = df_numeric.ffill().bfill()

                # Convert to numpy
                flight_data = df_clean.to_numpy(dtype=np.float32)

                if flight_data.shape[0] == 0 or flight_data.shape[1] == 0:
                    continue

                # Store original length BEFORE padding
                original_length = flight_data.shape[0]
                feat_dim = flight_data.shape[1]

                # Skip flights that are too short
                if original_length < self.min_forecast_horizon * 2:
                    continue

                # Apply normalization BEFORE padding
                flight_data = self._normalize_data(flight_data)

                # Calculate forecast horizon based on ORIGINAL length
                forecast_horizon = self._calculate_forecast_horizon(original_length)

                # Pad flight to seq_len if needed
                if original_length < self.seq_len:
                    pad_length = self.seq_len - original_length
                    if original_length > 0:
                        last_row = flight_data[-1:]
                    else:
                        last_row = np.zeros((1, feat_dim))
                    padding = np.repeat(last_row, pad_length, axis=0)
                    flight_data_padded = np.vstack([flight_data, padding])
                else:
                    # Truncate if longer than seq_len
                    flight_data_padded = flight_data[:self.seq_len]
                    original_length = min(original_length, self.seq_len)

                # Create forecast mask
                # 1 = keep (context + padding), 0 = predict (forecast targets)
                mask = np.ones((self.seq_len, feat_dim), dtype=np.float32)

                # Calculate forecast region indices
                forecast_start = original_length - forecast_horizon
                forecast_end = original_length

                # Mask the forecast portion (end of original flight)
                mask[forecast_start:forecast_end, :] = 0

                # Padding (after original_length) stays as 1 - won't contribute to loss

                # Create masked input (zero out forecast portion)
                x_masked = flight_data_padded.copy()
                x_masked[forecast_start:forecast_end, :] = 0

                # Convert to tensors
                x_masked_tensor = torch.tensor(x_masked, dtype=torch.float32)
                x_original_tensor = torch.tensor(flight_data_padded, dtype=torch.float32)
                mask_tensor = torch.tensor(mask, dtype=torch.float32)

                yield x_masked_tensor, x_original_tensor, mask_tensor

            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                continue


def create_forecast_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 4,
    seq_len: int = 10000,
    forecast_ratios: Optional[List[float]] = None,
    forecast_ratio: float = 0.2,
    min_forecast_horizon: int = 100,
    max_forecast_horizon: Optional[int] = None,
    normalization_params: Optional[Dict[str, np.ndarray]] = None,
    max_files: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> DataLoader:
    """
    Create a DataLoader for flight forecasting.

    Args:
        data_dir: Directory containing CSV flight data files
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        seq_len: Sequence length to pad flights to
        forecast_ratios: List of forecast ratios to randomly sample from (default [0.1, 0.2, 0.3])
        forecast_ratio: Fixed forecast ratio (only used if forecast_ratios is None)
        min_forecast_horizon: Minimum timesteps to predict
        max_forecast_horizon: Maximum timesteps to predict
        normalization_params: Global normalization parameters
        max_files: Maximum files to process
        num_workers: DataLoader workers
        seed: Random seed
        train_split: Fraction for training
        val_split: Fraction for validation

    Returns:
        DataLoader yielding (x_masked, x_original, mask) batches
    """
    dataset = ForecastFlightDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        forecast_ratios=forecast_ratios,
        forecast_ratio=forecast_ratio,
        min_forecast_horizon=min_forecast_horizon,
        max_forecast_horizon=max_forecast_horizon,
        normalization_params=normalization_params,
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


if __name__ == "__main__":
    # Test the forecast dataset
    print("Testing ForecastFlightDataset...")

    # Test with synthetic data
    import tempfile
    import os

    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple test CSV
        test_data = np.random.randn(500, 10).astype(np.float32)  # 500 timesteps, 10 features
        df = pd.DataFrame(test_data, columns=[f"feat_{i}" for i in range(10)])

        os.makedirs(os.path.join(tmpdir, "preprocessed_data", "train"), exist_ok=True)
        df.to_csv(os.path.join(tmpdir, "preprocessed_data", "train", "test_flight.csv"), index=False)

        # Test dataset
        dataset = ForecastFlightDataset(
            data_dir=tmpdir,
            split="train",
            seq_len=1000,
            forecast_ratio=0.2,
            min_forecast_horizon=50,
        )

        # Test iteration
        for x_masked, x_original, mask in dataset:
            print(f"  x_masked shape: {x_masked.shape}")
            print(f"  x_original shape: {x_original.shape}")
            print(f"  mask shape: {mask.shape}")

            # Check masking
            masked_positions = (mask == 0).sum().item()
            total_positions = mask.numel()
            print(f"  Masked positions: {masked_positions} / {total_positions}")
            print(f"  Forecast ratio (actual): {masked_positions / total_positions:.3f}")

            # Verify forecast is at the end of original data (before padding)
            # Original length was 500, so forecast should be around position 400-500
            masked_rows = (mask[:, 0] == 0).numpy()
            if masked_rows.any():
                first_masked = np.argmax(masked_rows)
                last_masked = len(masked_rows) - np.argmax(masked_rows[::-1]) - 1
                print(f"  Forecast region: rows {first_masked} to {last_masked}")

            break

    print("\nForecastFlightDataset test completed!")
