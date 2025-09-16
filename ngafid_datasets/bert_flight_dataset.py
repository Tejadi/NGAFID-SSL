#!/usr/bin/env python3
"""
Dataset class for BERT-based masked flight regression.
Uses HuggingFace dataset and existing masking functions.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset
from datasets import load_dataset
from huggingface_hub import list_repo_files
from typing import List, Optional, Tuple, Iterator
import io
from .masked_flight_dataset import noise_mask


class BertFlightDataset(IterableDataset):
    """
    Streams flight data from HuggingFace dataset for BERT masked regression.

    Each sample contains:
    - x_masked: Flight data with masked values set to 0
    - x_original: Original unmasked flight data
    - mask: Binary mask (1=keep, 0=masked)
    """

    def __init__(
        self,
        repo_id: str = "CDuong04/NGAFID-LOCI-GATS-Data",
        split: str = "train",
        subdir: str = "preprocessed_data",
        seq_len: int = 256,
        step: int = 256,
        masking_ratio: float = 0.6,
        mean_mask_length: int = 3,
        max_files: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.subdir = subdir
        self.seq_len = seq_len
        self.step = step
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.max_files = max_files
        self.seed = seed

        # Initialize random state for reproducible masking
        self.rng = np.random.RandomState(seed)

        # Load dataset
        try:
            # Try with subdir first, fall back to default
            try:
                self.dataset = load_dataset(repo_id, subdir)
            except ValueError:
                print(f"Config '{subdir}' not found, using default config")
                self.dataset = load_dataset(repo_id)

            if split not in self.dataset:
                raise ValueError(f"Split '{split}' not found in dataset. Available: {list(self.dataset.keys())}")
            self.data = self.dataset[split]
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {repo_id}: {e}")

        print(f"Loaded dataset with {len(self.data)} samples")

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

        file_count = 0
        max_files = self.max_files or len(self.data)

        for idx in range(start_idx, min(len(self.data), max_files), step_size):
            if file_count >= (max_files // (step_size if step_size > 1 else 1)):
                break

            try:
                # Get the flight data - assuming it's stored as a CSV string or file path
                sample = self.data[idx]

                # Handle different data formats
                if 'csv_data' in sample:
                    # If CSV data is stored directly as string
                    flight_df = pd.read_csv(io.StringIO(sample['csv_data']))
                elif 'file_path' in sample:
                    # If file path is provided
                    flight_df = pd.read_csv(sample['file_path'], na_values=[' NaN', 'NaN', 'NaN '])
                elif isinstance(sample, dict) and 'data' in sample:
                    # If data is stored as dict/array
                    flight_df = pd.DataFrame(sample['data'])
                else:
                    # Try to convert directly to dataframe
                    flight_df = pd.DataFrame(sample)

                # Clean and prepare data
                flight_df = flight_df.fillna(method='ffill').fillna(method='bfill')
                flight_data = flight_df.to_numpy(dtype=np.float32)

                # Handle case where flight is shorter than seq_len
                if len(flight_data) < self.seq_len:
                    # Pad with last values
                    pad_length = self.seq_len - len(flight_data)
                    last_row = flight_data[-1:] if len(flight_data) > 0 else np.zeros((1, flight_data.shape[1]))
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

                file_count += 1

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue


def create_dataloader(
    repo_id: str = "CDuong04/NGAFID-LOCI-GATS-Data",
    split: str = "train",
    batch_size: int = 16,
    seq_len: int = 256,
    masking_ratio: float = 0.6,
    mean_mask_length: int = 3,
    max_files: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for BERT flight regression training.

    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split to use ('train', 'validation', 'test')
        batch_size: Batch size for training
        seq_len: Sequence length for windows
        masking_ratio: Ratio of values to mask
        mean_mask_length: Average length of masked segments
        max_files: Maximum number of files to process (None for all)
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility

    Returns:
        DataLoader yielding batches of (x_masked, x_original, mask)
    """
    dataset = BertFlightDataset(
        repo_id=repo_id,
        split=split,
        seq_len=seq_len,
        masking_ratio=masking_ratio,
        mean_mask_length=mean_mask_length,
        max_files=max_files,
        seed=seed,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing BertFlightDataset...")

    try:
        dataloader = create_dataloader(
            split="train",
            batch_size=2,
            seq_len=128,
            max_files=1,
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

        print("Dataset test completed successfully!")

    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()