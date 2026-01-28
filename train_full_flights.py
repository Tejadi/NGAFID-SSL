#!/usr/bin/env python3
"""
Training script for full-flight BERT masked regressor on Oscar cluster.
Optimized for RTX A5000 with 24-hour training window.
"""

import argparse
import os
import time
import json
import gc
from typing import Dict, Any, Optional, List

# Set CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - skipping W&B logging")

# Import our models and dataset
try:
    from models.bert_masked_regressor import BertMaskedRegressor, count_parameters
    from ngafid_datasets.local_flight_dataset import LocalFlightDataset, get_feature_dim_from_local_data
    from ngafid_datasets.masked_flight_dataset import noise_mask
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def create_lr_scheduler(optimizer, num_training_steps, warmup_steps):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_normalization_parameters(data_dir: str, max_files: int = 100) -> Dict[str, np.ndarray]:
    """
    Compute global normalization parameters from training data.
    Uses the same approach as the autoencoder benchmark.

    Args:
        data_dir: Directory containing CSV flight data files
        max_files: Maximum number of files to use for computing statistics

    Returns:
        Dictionary containing 'mean' and 'std' arrays
    """
    print("📊 Computing global normalization parameters...")

    # Find training data files
    data_path = Path(data_dir)
    train_files = []

    # Look for preprocessed training data
    preprocessed_path = data_path / "preprocessed_data" / "train"
    if preprocessed_path.exists():
        train_files = list(preprocessed_path.glob("*.csv"))[:max_files]
    else:
        # Fallback to all CSV files
        train_files = list(data_path.glob("*.csv"))[:max_files]
        train_files = [f for f in train_files if not any(name in f.name.lower()
                      for name in ['aircraft_types', 'events', 'flight_ids', 'splits'])]

    if not train_files:
        raise ValueError(f"No training CSV files found in {data_dir}")

    print(f"   Using {len(train_files)} files to compute normalization parameters")

    # Collect all data for computing global statistics
    all_data = []

    for csv_file in tqdm(train_files, desc="Loading training data"):
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

            if flight_data.shape[0] > 0 and flight_data.shape[1] > 0:
                all_data.append(flight_data)

        except Exception as e:
            print(f"   Warning: Error processing {csv_file.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data found for computing normalization parameters")

    # Concatenate all data
    concatenated_data = np.vstack(all_data)
    print(f"   Total data shape: {concatenated_data.shape}")

    # Compute global mean and std (same as autoencoder)
    data_mean = np.mean(concatenated_data, axis=0)
    data_std = np.std(concatenated_data, axis=0)

    # Avoid division by zero (same as autoencoder)
    data_std[data_std == 0] = 1.0

    print(f"   Computed normalization parameters for {len(data_mean)} features")
    print(f"   Mean range: [{data_mean.min():.4f}, {data_mean.max():.4f}]")
    print(f"   Std range: [{data_std.min():.4f}, {data_std.max():.4f}]")

    return {
        'mean': data_mean,
        'std': data_std
    }


class GlobalNormalizedFlightDataset(LocalFlightDataset):
    """
    Extended LocalFlightDataset that applies global normalization
    instead of per-flight robust normalization to match autoencoder approach.
    Also supports random masking parameters for enhanced training diversity.
    """

    def __init__(
        self,
        normalization_params: Optional[Dict[str, np.ndarray]] = None,
        use_random_masking: bool = False,
        masking_ratios: List[float] = [0.2, 0.5, 0.8],
        mean_mask_lengths: List[int] = [5, 60],
        aircraft_types: Optional[List[str]] = None,
        data_scale: float = 1.0,
        **kwargs
    ):
        # Remove masking parameters from kwargs if using random masking
        if use_random_masking:
            kwargs.pop('masking_ratio', None)
            kwargs.pop('mean_mask_length', None)
            # Set default values for parent class initialization
            kwargs['masking_ratio'] = masking_ratios[0]
            kwargs['mean_mask_length'] = mean_mask_lengths[0]

        super().__init__(**kwargs)
        self.normalization_params = normalization_params
        self.use_random_masking = use_random_masking
        self.masking_ratios = masking_ratios
        self.mean_mask_lengths = mean_mask_lengths

        # Apply aircraft type filter
        if aircraft_types is not None:
            before = len(self.split_files)
            self.split_files = filter_files_by_aircraft(self.split_files, aircraft_types)
            print(f"Aircraft filter {aircraft_types}: {before} -> {len(self.split_files)} files")

        # Apply data scaling
        if data_scale < 1.0:
            before = len(self.split_files)
            self.split_files = apply_data_scale(self.split_files, data_scale, seed=kwargs.get('seed', 42))
            print(f"Data scale {data_scale:.0%}: {before} -> {len(self.split_files)} files")

        if use_random_masking:
            print(f"✨ Using random masking with {len(masking_ratios)} ratios × {len(mean_mask_lengths)} lengths = {len(masking_ratios) * len(mean_mask_lengths)} combinations")
            print(f"   Masking ratios: {masking_ratios}")
            print(f"   Mean mask lengths: {mean_mask_lengths}")

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply global normalization using precomputed mean and std.
        If normalization_params is None, falls back to original robust normalization.

        Args:
            data: Flight data array of shape (seq_len, feat_dim)

        Returns:
            Normalized data array
        """
        if self.normalization_params is None:
            # Fallback to original robust normalization
            return super()._normalize_data(data)

        # Apply global normalization (same as autoencoder)
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']

        # Ensure dimensions match
        if mean.shape[0] != data.shape[1] or std.shape[0] != data.shape[1]:
            print(f"Warning: Dimension mismatch in normalization. "
                  f"Data: {data.shape[1]}, Mean: {mean.shape[0]}, Std: {std.shape[0]}")
            return super()._normalize_data(data)

        # Apply normalization: (data - mean) / std
        normalized_data = (data - mean) / std

        return normalized_data.astype(np.float32)

    def __iter__(self):
        """
        Override iterator to support random masking parameters.
        """
        if not self.use_random_masking:
            # Use parent's iterator if not using random masking
            yield from super().__iter__()
            return

        # Custom iterator with random masking
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
                # Read CSV file (same as parent)
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

                # Apply normalization
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

                    # RANDOM MASKING: Choose random parameters for this window
                    random_masking_ratio = np.random.choice(self.masking_ratios)
                    random_mean_mask_length = np.random.choice(self.mean_mask_lengths)

                    # Apply masking with random parameters
                    mask = noise_mask(
                        window,
                        random_masking_ratio,
                        random_mean_mask_length,
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


def create_global_normalized_dataloader(
    data_dir: str,
    normalization_params: Optional[Dict[str, np.ndarray]] = None,
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
    use_random_masking: bool = False,
    masking_ratios: List[float] = [0.2, 0.5, 0.8],
    mean_mask_lengths: List[int] = [5, 60],
    aircraft_types: Optional[List[str]] = None,
    data_scale: float = 1.0,
) -> DataLoader:
    """
    Create a DataLoader for local flight data with global normalization and optional random masking.

    Args:
        data_dir: Directory containing CSV flight data files
        normalization_params: Global normalization parameters (mean, std)
        split: Dataset split to use ('train', 'validation', 'test')
        batch_size: Batch size for training
        seq_len: Sequence length for windows
        masking_ratio: Ratio of values to mask (ignored if use_random_masking=True)
        mean_mask_length: Average length of masked segments (ignored if use_random_masking=True)
        max_files: Maximum number of files to process (None for all)
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
        train_split: Fraction of files for training
        val_split: Fraction of files for validation
        use_random_masking: Whether to use random masking parameters
        masking_ratios: List of masking ratios to randomly choose from
        mean_mask_lengths: List of mean mask lengths to randomly choose from
        aircraft_types: List of aircraft type prefixes to filter by (None for all)
        data_scale: Fraction of data to use (0.0-1.0)

    Returns:
        DataLoader yielding batches of (x_masked, x_original, mask)
    """
    dataset = GlobalNormalizedFlightDataset(
        normalization_params=normalization_params,
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        masking_ratio=masking_ratio,
        mean_mask_length=mean_mask_length,
        max_files=max_files,
        seed=seed,
        train_split=train_split,
        val_split=val_split,
        use_random_masking=use_random_masking,
        masking_ratios=masking_ratios,
        mean_mask_lengths=mean_mask_lengths,
        aircraft_types=aircraft_types,
        data_scale=data_scale,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def evaluate_model(model, dataloader, device, max_batches: int = 50) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_samples = 0
    total_masked_positions = 0

    with torch.no_grad():
        for batch_idx, (x_masked, x_original, mask) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)

            loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)

            # Count masked positions for per-position metrics
            masked_positions = (mask == 0).sum().item()

            total_loss += loss.item() * x_masked.size(0)
            total_mse_loss += mse_loss.item() * x_masked.size(0)
            total_mae_loss += mae_loss.item() * x_masked.size(0)
            total_samples += x_masked.size(0)
            total_masked_positions += masked_positions

    # Clear CUDA cache after evaluation to prevent memory buildup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse_loss = total_mse_loss / total_samples if total_samples > 0 else float('inf')
    avg_mae_loss = total_mae_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse_per_position = avg_mse_loss * total_samples / total_masked_positions if total_masked_positions > 0 else float('inf')
    avg_mae_per_position = avg_mae_loss * total_samples / total_masked_positions if total_masked_positions > 0 else float('inf')

    return {
        "eval_loss": avg_loss,
        "eval_mse_loss": avg_mse_loss,
        "eval_mae_loss": avg_mae_loss,
        "eval_mse_per_position": avg_mse_per_position,
        "eval_mae_per_position": avg_mae_per_position,
        "eval_masked_positions": total_masked_positions / total_samples if total_samples > 0 else 0
    }


def filter_files_by_aircraft(file_list, aircraft_types):
    """Filter a list of file paths to only include specified aircraft types.

    Args:
        file_list: List of Path objects (CSV flight files)
        aircraft_types: List of aircraft type prefixes (e.g., ["Cessna_172S", "PA-28-181"])

    Returns:
        Filtered list of Path objects
    """
    filtered = [f for f in file_list if any(f.name.startswith(at) for at in aircraft_types)]
    return filtered


def apply_data_scale(file_list, scale, seed=42):
    """Subsample a file list to a given fraction for data scaling experiments.

    Args:
        file_list: List of Path objects
        scale: Fraction of data to keep (0.0-1.0)
        seed: Random seed for reproducible subsampling

    Returns:
        Subsampled list of Path objects
    """
    if scale >= 1.0:
        return file_list
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(len(file_list) * scale))
    indices = rng.choice(len(file_list), size=n_keep, replace=False)
    indices.sort()
    return [file_list[i] for i in indices]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train BERT masked regressor on full flights")
    parser.add_argument("--use_fixed_masking", action="store_true",
                        help="Use fixed masking ratio and mean mask length instead of random sampling")
    parser.add_argument("--masking_ratio", type=float, default=0.5,
                        help="Fixed masking ratio (default: 0.5, only used with --use_fixed_masking)")
    parser.add_argument("--mean_mask_length", type=int, default=60,
                        help="Fixed mean mask length (default: 60, only used with --use_fixed_masking)")
    parser.add_argument("--wandb_project", type=str, default="bert-flight-full",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # Data filtering arguments
    parser.add_argument("--aircraft_type", type=str, nargs='+', default=None,
                        choices=["Cessna_172S", "PA-28-181", "PA-44-180"],
                        help="Filter training data to specific aircraft type(s)")
    parser.add_argument("--aircraft_class", type=str, default=None,
                        choices=["single_engine", "multi_engine"],
                        help="Filter training data by aircraft class (single_engine=Cessna_172S+PA-28-181, multi_engine=PA-44-180)")
    parser.add_argument("--data_scale", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0) for data scaling experiments")
    args = parser.parse_args()

    # Resolve aircraft_class to aircraft_type list
    AIRCRAFT_CLASS_MAP = {
        "single_engine": ["Cessna_172S", "PA-28-181"],
        "multi_engine": ["PA-44-180"],
    }
    if args.aircraft_class is not None:
        if args.aircraft_type is not None:
            print("Warning: --aircraft_class overrides --aircraft_type")
        args.aircraft_type = AIRCRAFT_CLASS_MAP[args.aircraft_class]

    # Memory-optimized configuration for Oscar cluster training
    print("🚀 Starting Memory-Optimized BERT Flight Training")
    print("=" * 60)

    # Dataset and model configuration
    data_dir = "/oscar/data/sbach/shared/ngafid"
    seq_len = 10000  # Full flight sequences (non-negotiable)
    batch_size = 4   # Ultra-conservative for seq_len=10000
    gradient_accumulation_steps = 8  # Effective batch size = 1 * 8 = 8
    epochs = 50
    learning_rate = 1e-4  # Slightly higher due to smaller batch size

    # Model architecture (memory-optimized)
    hidden_size = 1024  # Reduced from 1536
    encoder_layers = 8  # Reduced from 12
    decoder_layers = 6  # Reduced from 8
    num_heads = 16
    dropout = 0.1

    # Training settings
    warmup_steps = 1000  # Reduced proportionally
    eval_interval = 500  # More frequent evaluation
    save_interval = 2000
    max_files_train = 400  # Reduced for faster epochs
    max_files_val = 100

    # Memory optimization settings
    use_mixed_precision = True
    use_gradient_checkpointing = True
    use_memory_efficient_optimizer = True

    print(f"📊 Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Sequence length: {seq_len:,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Model: {hidden_size}d, {encoder_layers}enc, {decoder_layers}dec")
    print()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (GPU not available)")
    print()

    # Check data directory
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Please check the path and try again.")
        exit(1)

    print(f"✅ Found data directory: {data_dir}")

    # Get feature dimension from data
    print("🔍 Detecting feature dimension from data...")
    try:
        feat_dim = get_feature_dim_from_local_data(data_dir)
        print(f"✅ Detected feature dimension: {feat_dim}")
    except Exception as e:
        print(f"❌ Error detecting feature dimension: {e}")
        print("Using default feature dimension: 44")
        feat_dim = 44

    # Compute global normalization parameters (same as autoencoder)
    print("🔧 Computing global normalization parameters...")
    try:
        normalization_params = compute_normalization_parameters(data_dir, max_files=max_files_train)
        print(f"✅ Computed normalization parameters")

        # Save normalization parameters for consistency with autoencoder
        norm_params_path = f"./results/bert_normalization_params_{time.strftime('%Y%m%d_%H%M%S')}.npy"
        os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)
        np.save(norm_params_path, normalization_params)
        print(f"💾 Saved normalization parameters to: {norm_params_path}")

    except Exception as e:
        print(f"❌ Error computing normalization parameters: {e}")
        print("⚠️  Training will continue without global normalization")
        normalization_params = None

    # Create data loaders with global normalization
    if args.use_fixed_masking:
        print(f"📁 Creating data loaders with FIXED masking...")
        print(f"   Masking ratio: {args.masking_ratio}")
        print(f"   Mean mask length: {args.mean_mask_length}")
        use_random_masking = False
        masking_ratio = args.masking_ratio
        mean_mask_length = args.mean_mask_length
        masking_ratios = [args.masking_ratio]
        mean_mask_lengths = [args.mean_mask_length]
    else:
        print("📁 Creating data loaders with RANDOM masking...")
        masking_ratios = [0.2, 0.5, 0.8]
        mean_mask_lengths = [5, 60]
        print(f"🎲 Training will use random masking: {len(masking_ratios)} ratios × {len(mean_mask_lengths)} lengths = {len(masking_ratios) * len(mean_mask_lengths)} combinations")
        print(f"   Masking ratios: {masking_ratios}")
        print(f"   Mean mask lengths: {mean_mask_lengths}")
        use_random_masking = True
        masking_ratio = 0.6  # Default for fallback
        mean_mask_length = 3  # Default for fallback

    # Print data filtering info
    if args.aircraft_type:
        print(f"Aircraft filter: {args.aircraft_type}")
    if args.data_scale < 1.0:
        print(f"Data scale: {args.data_scale:.0%}")

    try:
        train_loader = create_global_normalized_dataloader(
            data_dir=data_dir,
            normalization_params=normalization_params,
            split="train",
            batch_size=batch_size,
            seq_len=seq_len,
            max_files=max_files_train,
            num_workers=1,  # Reduced for memory efficiency
            seed=42,
            use_random_masking=use_random_masking,
            masking_ratio=masking_ratio,
            mean_mask_length=mean_mask_length,
            masking_ratios=masking_ratios,
            mean_mask_lengths=mean_mask_lengths,
            aircraft_types=args.aircraft_type,
            data_scale=args.data_scale,
        )

        # Use fixed masking for validation (for consistent evaluation)
        # Use dedicated validation directory if it exists
        val_data_dir = "/oscar/data/sbach/shared/ngafid/preprocessed_data/val"
        if not os.path.exists(val_data_dir):
            print(f"⚠️  Validation directory {val_data_dir} not found, using splits from main data")
            val_data_dir = data_dir
            val_split = "val"
        else:
            print(f"✅ Using dedicated validation directory: {val_data_dir}")
            val_split = "train"  # Use all files in val directory

        val_loader = create_global_normalized_dataloader(
            data_dir=val_data_dir,
            normalization_params=normalization_params,
            split=val_split,
            batch_size=batch_size,
            seq_len=seq_len,
            max_files=max_files_val,
            num_workers=1,  # Reduced for memory efficiency
            seed=42,
            use_random_masking=False,  # Fixed masking for validation
            masking_ratio=0.6,  # Standard masking ratio for evaluation
            mean_mask_length=3   # Standard mean mask length for evaluation
        )
        print(f"✅ Created data loaders (train: ~{len(train_loader)} batches)")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        exit(1)

    # Create memory-optimized model
    print("🏗️  Creating memory-optimized model...")
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_mixed_precision=use_mixed_precision,
    ).to(device)

    total_params = count_parameters(model)
    print(f"✅ Model created with {total_params:,} parameters")
    print(f"   Estimated GPU memory: ~{total_params * 4 / 1e9:.1f} GB")
    print()

    # Setup memory-efficient optimizer and scheduler
    if use_memory_efficient_optimizer:
        # Use 8-bit AdamW optimizer to reduce memory by ~50%
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            print("✅ Using 8-bit AdamW optimizer (50% memory reduction)")
        except ImportError:
            print("⚠️  bitsandbytes not available, using standard AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

    # Calculate total steps with gradient accumulation
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * epochs
    scheduler = create_lr_scheduler(optimizer, total_steps, warmup_steps)

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

    print(f"📈 Memory-optimized training setup:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    print(f"   Mixed precision: {use_mixed_precision}")
    print(f"   Gradient checkpointing: {use_gradient_checkpointing}")
    print(f"   Memory-efficient optimizer: {use_memory_efficient_optimizer}")
    print()

    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_name = f"bert_full_flights_{timestamp}"

    # Create output directory
    output_dir = f"./results/{job_name}"
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{output_dir}/tensorboard")

    # W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=job_name,
                config={
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "hidden_size": hidden_size,
                    "encoder_layers": encoder_layers,
                    "decoder_layers": decoder_layers,
                    "epochs": epochs,
                    "total_params": total_params,
                    "normalization": "global_mean_std" if normalization_params else "per_flight_robust",
                    "feat_dim": feat_dim,
                    "use_fixed_masking": args.use_fixed_masking,
                    "random_masking": use_random_masking,
                    "masking_ratios": masking_ratios,
                    "mean_mask_lengths": mean_mask_lengths,
                    "num_masking_combinations": len(masking_ratios) * len(mean_mask_lengths),
                    "train_masking_ratio": masking_ratio if args.use_fixed_masking else "random",
                    "train_mean_mask_length": mean_mask_length if args.use_fixed_masking else "random",
                    "val_masking_ratio": 0.6,
                    "val_mean_mask_length": 3,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "use_mixed_precision": use_mixed_precision,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "memory_efficient_optimizer": use_memory_efficient_optimizer,
                    "val_data_dir": val_data_dir,
                    "val_split": val_split,
                    "max_files_train": max_files_train,
                    "max_files_val": max_files_val,
                    "aircraft_type": args.aircraft_type,
                    "aircraft_class": args.aircraft_class,
                    "data_scale": args.data_scale,
                    "warmup_steps": warmup_steps,
                    "eval_interval": eval_interval,
                    "save_interval": save_interval,
                }
            )
            print("✅ W&B logging enabled")
        except Exception as e:
            print(f"⚠️  W&B setup failed: {e}")
            use_wandb = False

    print(f"📊 Results will be saved to: {output_dir}")
    print()

    # Create save directory for this training run in local checkpoints folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"/oscar/home/cduong5/NGAFID-SSL/checkpoints/bert_models_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Models will be saved to: {save_dir}")

    # Training loop
    print("🎯 Starting training...")
    global_step = 0
    best_eval_loss = float('inf')

    # Epochs to save models at (every 5 epochs)
    save_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_mae_loss = 0.0
        epoch_samples = 0

        # Create progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            dynamic_ncols=True
        )

        accumulation_step = 0
        optimizer.zero_grad()

        for batch_idx, (x_masked, x_original, mask) in enumerate(pbar):
            x_masked = x_masked.to(device, non_blocking=True)
            x_original = x_original.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Forward pass with mixed precision
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask, scaler=scaler)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

            accumulation_step += 1

            # Perform optimizer step every gradient_accumulation_steps
            if accumulation_step % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    # Gradient clipping with mixed precision
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Update metrics (unscale loss for logging)
            actual_loss = loss.item() * gradient_accumulation_steps
            epoch_loss += actual_loss * x_masked.size(0)
            epoch_mse_loss += mse_loss.item() * x_masked.size(0)
            epoch_mae_loss += mae_loss.item() * x_masked.size(0)
            epoch_samples += x_masked.size(0)

            # Clear cache periodically to prevent memory buildup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Log training metrics (only after actual optimizer steps)
            if accumulation_step % gradient_accumulation_steps == 0 and global_step % 50 == 0:
                # Compute per-position metrics
                masked_positions = (mask == 0).sum().item()
                mse_per_position = mse_loss.item() / masked_positions if masked_positions > 0 else 0
                mae_per_position = mae_loss.item() / masked_positions if masked_positions > 0 else 0
                masking_ratio = masked_positions / mask.numel()

                # Compute additional metrics for more comprehensive logging
                current_lr = scheduler.get_last_lr()[0]
                avg_train_mse = epoch_mse_loss / (epoch_samples + 1e-8)
                avg_train_mae = epoch_mae_loss / (epoch_samples + 1e-8)
                avg_train_loss = epoch_loss / (epoch_samples + 1e-8)

                writer.add_scalar("train/loss", actual_loss, global_step)
                writer.add_scalar("train/mse_loss", mse_loss.item(), global_step)
                writer.add_scalar("train/mae_loss", mae_loss.item(), global_step)
                writer.add_scalar("train/mse_per_position", mse_per_position, global_step)
                writer.add_scalar("train/mae_per_position", mae_per_position, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/masking_ratio", masking_ratio, global_step)
                writer.add_scalar("train/avg_mse", avg_train_mse, global_step)
                writer.add_scalar("train/avg_mae", avg_train_mae, global_step)
                writer.add_scalar("train/avg_loss", avg_train_loss, global_step)

                if use_wandb:
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/mse_loss": mse_loss.item(),
                        "train/mae_loss": mae_loss.item(),
                        "train/mse_per_position": mse_per_position,
                        "train/mae_per_position": mae_per_position,
                        "train/masking_ratio": masking_ratio,
                        "train/lr": current_lr,
                        "train/avg_mse": avg_train_mse,
                        "train/avg_mae": avg_train_mae,
                        "train/avg_loss": avg_train_loss,
                        "train/epoch": epoch + (batch_idx / len(train_loader)),
                        "train/step": global_step,
                        "train/masked_positions": masked_positions,
                        "train/total_positions": mask.numel(),
                    }, step=global_step)

            # Update progress bar
            pbar.set_postfix({
                "mse": f"{mse_loss.item():.4f}",
                "mae": f"{mae_loss.item():.4f}",
                "avg_mse": f"{epoch_mse_loss/(epoch_samples+1e-8):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
                "acc_step": f"{accumulation_step % gradient_accumulation_steps + 1}/{gradient_accumulation_steps}",
            })

            # Evaluation (only after actual optimizer steps)
            if accumulation_step % gradient_accumulation_steps == 0 and global_step % eval_interval == 0 and global_step > 0:
                pbar.write(f"\n🔍 Evaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_loader, device)

                writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                writer.add_scalar("eval/mse_loss", eval_metrics["eval_mse_loss"], global_step)
                writer.add_scalar("eval/mae_loss", eval_metrics["eval_mae_loss"], global_step)
                writer.add_scalar("eval/mse_per_position", eval_metrics["eval_mse_per_position"], global_step)
                writer.add_scalar("eval/mae_per_position", eval_metrics["eval_mae_per_position"], global_step)
                writer.add_scalar("eval/masked_positions", eval_metrics["eval_masked_positions"], global_step)

                if use_wandb:
                    # Log all evaluation metrics with detailed names
                    wandb_eval_metrics = {
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/mse_loss": eval_metrics["eval_mse_loss"],
                        "eval/mae_loss": eval_metrics["eval_mae_loss"],
                        "eval/mse_per_position": eval_metrics["eval_mse_per_position"],
                        "eval/mae_per_position": eval_metrics["eval_mae_per_position"],
                        "eval/masked_positions": eval_metrics["eval_masked_positions"],
                        "eval/step": global_step,
                        "eval/epoch": epoch + (batch_idx / len(train_loader)),
                        # Additional aliases for common metric names
                        "validation_loss": eval_metrics["eval_loss"],
                        "validation_mse": eval_metrics["eval_mse_loss"],
                        "validation_mae": eval_metrics["eval_mae_loss"],
                        "val_loss": eval_metrics["eval_loss"],
                        "val_mse": eval_metrics["eval_mse_loss"],
                        "val_mae": eval_metrics["eval_mae_loss"],
                        "val_mse_per_pos": eval_metrics["eval_mse_per_position"],
                        "val_mae_per_pos": eval_metrics["eval_mae_per_position"],
                    }

                    # Add improvement metrics if we have best loss tracking
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        wandb_eval_metrics.update({
                            "eval/is_best": 1,
                            "eval/improvement": best_eval_loss - eval_metrics["eval_loss"],
                        })
                    else:
                        wandb_eval_metrics.update({
                            "eval/is_best": 0,
                            "eval/improvement": 0,
                        })

                    wandb.log(wandb_eval_metrics, step=global_step)

                pbar.write(f"📊 Step {global_step}: Eval MSE = {eval_metrics['eval_mse_loss']:.4f}, MAE = {eval_metrics['eval_mae_loss']:.4f}")

                # Return to training mode
                model.train()

                # Save best model
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'eval_metrics': eval_metrics,
                        'feat_dim': feat_dim,
                        'config': {
                            'hidden_size': hidden_size,
                            'encoder_layers': encoder_layers,
                            'decoder_layers': decoder_layers,
                            'num_heads': num_heads,
                            'seq_len': seq_len,
                        }
                    }, f"{output_dir}/best_model.pt")
                    pbar.write(f"💾 Saved best model (eval_loss: {best_eval_loss:.4f})")

            # Save checkpoint (only after actual optimizer steps)
            if accumulation_step % gradient_accumulation_steps == 0 and global_step % save_interval == 0 and global_step > 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'feat_dim': feat_dim,
                }, f"{output_dir}/checkpoint_step_{global_step}.pt")
                pbar.write(f"💾 Saved checkpoint at step {global_step}")

            # Clear gradients and cache if not accumulating
            if accumulation_step % gradient_accumulation_steps != 0:
                # Don't increment global_step here - only after actual optimizer step
                pass

        # Clear CUDA cache at the end of each epoch to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # End of epoch summary
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mse = epoch_mse_loss / epoch_samples if epoch_samples > 0 else 0
        avg_epoch_mae = epoch_mae_loss / epoch_samples if epoch_samples > 0 else 0

        # Log epoch summary to wandb
        if use_wandb:
            wandb.log({
                "epoch/loss": avg_epoch_loss,
                "epoch/mse": avg_epoch_mse,
                "epoch/mae": avg_epoch_mae,
                "epoch/number": epoch + 1,
                "epoch/best_eval_loss": best_eval_loss,
                "epoch/samples_processed": epoch_samples,
                "epoch/lr": scheduler.get_last_lr()[0],
            }, step=global_step)

        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        print(f"   Average MSE: {avg_epoch_mse:.4f}")
        print(f"   Average MAE: {avg_epoch_mae:.4f}")
        print(f"   Best eval loss: {best_eval_loss:.4f}")

        # Save model at specific epochs
        if (epoch + 1) in save_epochs:
            epoch_save_path = f"{save_dir}/model_epoch_{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'feat_dim': feat_dim,
                'best_eval_loss': best_eval_loss,
                'epoch_metrics': {
                    'avg_loss': avg_epoch_loss,
                    'avg_mse': avg_epoch_mse,
                    'avg_mae': avg_epoch_mae,
                },
                'config': {
                    'hidden_size': hidden_size,
                    'encoder_layers': encoder_layers,
                    'decoder_layers': decoder_layers,
                    'num_heads': num_heads,
                    'seq_len': seq_len,
                    'total_params': total_params,
                }
            }, epoch_save_path)
            print(f"💾 Saved model at epoch {epoch+1} to: {epoch_save_path}")

        print()

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
        'global_step': global_step,
        'feat_dim': feat_dim,
        'final_config': {
            'hidden_size': hidden_size,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'total_params': total_params,
        }
    }, f"{output_dir}/final_model.pt")

    # Final summary logging
    if use_wandb:
        wandb.log({
            "training/completed": True,
            "training/total_epochs": epochs,
            "training/total_steps": global_step,
            "training/best_eval_loss": best_eval_loss,
            "training/final_lr": scheduler.get_last_lr()[0],
        })
        wandb.finish()

    writer.close()

    print("🎉 Training completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🏆 Best evaluation loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
