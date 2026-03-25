#!/usr/bin/env python3
"""
Linear probe evaluation for anomaly classification.

Evaluates representation quality of pre-trained models (SimCLR, BERT, LSTM, MLP)
by training a logistic regression on frozen representations to classify
whether a flight contains any anomaly event.
"""

import argparse
import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# SimCLR uses 41 specific features (from benchmarks/conv_mhsa/flight.py)
SIMCLR_INPUT_COLS = [
    'vspdg', 'e1egtdivergence', 'crs', 'vspdcalculated', 'trk', 'normac',
    'altmsl', 'vspd', 'oat', 'hplwas', 'baroa', 'e1oilp', 'ias', 'latac',
    'e1egt1', 'densityratio', 'e1oilt', 'altmsllagdiff', 'pitch', 'tas',
    'fqtyr', 'totalfuel', 'trueairspeed(ft/min)', 'hplfd', 'magvar',
    'e1egt2', 'altgps', 'amp1', 'fqtyl', 'volt1', 'e1fflow', 'altagl',
    'altb', 'roll', 'stallindex', 'e1egt3', 'e1rpm', 'e1egt4', 'hdg',
    'aoasimple', 'gndspd',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe anomaly classification")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["simclr", "bert", "lstm", "mlp"],
                        help="Type of pre-trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default="/oscar/data/sbach/shared/ngafid",
                        help="Root data directory")
    parser.add_argument("--events_file", type=str,
                        default="/oscar/data/sbach/bats/projects/ngafid/events.csv",
                        help="Path to events.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./linear_probe_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max files per split (for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile() optimization")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy single-sample extraction (slower)")
    return parser.parse_args()


def extract_flight_id(file_path: Path) -> Optional[int]:
    """Extract flight ID from filename like Cessna_172S_flight_100.csv."""
    name = file_path.stem  # e.g., Cessna_172S_flight_100
    match = re.search(r'_flight_(\d+)', name)
    if match:
        return int(match.group(1))
    # Fallback: last number in filename
    numbers = re.findall(r'\d+', name)
    if numbers:
        return int(numbers[-1])
    return None


def extract_aircraft_type(file_path: Path) -> Optional[str]:
    """Extract aircraft type from filename like Cessna_172S_flight_100.csv."""
    name = file_path.stem
    # Match known aircraft types
    if name.startswith("Cessna_172S"):
        return "Cessna_172S"
    elif name.startswith("PA-28-181"):
        return "PA-28-181"
    elif name.startswith("PA-44-180"):
        return "PA-44-180"
    return None


AIRCRAFT_TYPES = ["Cessna_172S", "PA-28-181", "PA-44-180"]
AIRCRAFT_TYPE_TO_IDX = {at: i for i, at in enumerate(AIRCRAFT_TYPES)}


def load_event_flight_ids(events_file: str) -> Set[int]:
    """Load set of flight IDs that have any anomaly event."""
    df = pd.read_csv(events_file)
    flight_ids = set(df['flight_id'].astype(int).unique())
    print(f"Loaded {len(flight_ids)} flights with events from {events_file}")
    return flight_ids


def load_event_labels(events_file: str) -> Tuple[Set[int], Dict[str, Set[int]], List[str]]:
    """Load binary and per-event-type labels.

    Returns:
        event_flight_ids: Set of all flight IDs with any event
        event_type_flight_ids: Dict mapping event_name -> set of flight IDs
        event_types: Sorted list of all event type names
    """
    df = pd.read_csv(events_file)
    df['flight_id'] = df['flight_id'].astype(int)

    event_flight_ids = set(df['flight_id'].unique())
    event_types = sorted(df['name'].str.strip().unique())
    event_type_flight_ids = {}
    for et in event_types:
        fids = set(df[df['name'].str.strip() == et]['flight_id'].unique())
        event_type_flight_ids[et] = fids

    print(f"Loaded {len(event_flight_ids)} flights with events, {len(event_types)} event types")
    return event_flight_ids, event_type_flight_ids, event_types


def get_split_files(data_dir: str, split: str) -> List[Path]:
    """Get list of flight CSV files for a given split."""
    split_dir = Path(data_dir) / "preprocessed_data" / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*.csv"))
    return files


def compute_normalization_params(data_dir: str, model_type: str, max_files: int = 500) -> Dict[str, np.ndarray]:
    """Compute normalization parameters from training data."""
    train_files = get_split_files(data_dir, "train")[:max_files]
    all_data = []

    for csv_file in tqdm(train_files, desc="Computing normalization"):
        try:
            df = pd.read_csv(csv_file, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])
            if model_type == "simclr":
                available = [c for c in SIMCLR_INPUT_COLS if c in df.columns]
                for c in SIMCLR_INPUT_COLS:
                    if c not in df.columns:
                        df[c] = 0.0
                df = df[SIMCLR_INPUT_COLS]
            else:
                df = df.select_dtypes(include=[np.number])
            df = df.ffill().bfill().fillna(0)
            all_data.append(df.to_numpy(dtype=np.float32))
        except Exception as e:
            continue

    concatenated = np.vstack(all_data)
    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    std[std == 0] = 1.0
    print(f"Normalization params computed over {len(all_data)} files, {concatenated.shape[1]} features")
    return {'mean': mean, 'std': std}


def load_flight_data(file_path: Path, model_type: str,
                     normalization_params: Dict[str, np.ndarray]) -> np.ndarray:
    """Load and preprocess a single flight for the given model type."""
    df = pd.read_csv(file_path, na_values=[' NaN', 'NaN', 'NaN ', 'nan'])

    if model_type == "simclr":
        for c in SIMCLR_INPUT_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[SIMCLR_INPUT_COLS]
        seq_len = 4096
    else:
        df = df.select_dtypes(include=[np.number])
        seq_len = 10000

    df = df.ffill().bfill().fillna(0)
    data = df.to_numpy(dtype=np.float32)

    # Normalize
    mean = normalization_params['mean']
    std = normalization_params['std']
    if mean.shape[0] == data.shape[1]:
        data = (data - mean) / std

    # Pad or truncate
    if len(data) < seq_len:
        if len(data) > 0:
            pad = np.repeat(data[-1:], seq_len - len(data), axis=0)
            data = np.vstack([data, pad])
        else:
            data = np.zeros((seq_len, data.shape[1]), dtype=np.float32)
    else:
        data = data[:seq_len]

    return data


# --- Model loading ---

def load_simclr_model(checkpoint_path: str, device: torch.device):
    from models.resnet_simclr import ResNetSimCLR
    model = ResNetSimCLR("resnet18", out_dim=128)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.remove_projector()
    model.to(device)
    model.eval()
    return model


def load_bert_model(checkpoint_path: str, device: torch.device):
    from models.bert_masked_regressor import BertMaskedRegressor
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    feat_dim = ckpt.get('feat_dim', 44)
    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=config['hidden_size'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        num_heads=config['num_heads'],
        dropout=0.0,
        max_seq_len=config.get('seq_len', 10000),
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
    )
    # Handle torch.compile() prefix in state dict keys
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_lstm_model(checkpoint_path: str, device: torch.device):
    from models.lstm_baseline import LSTMBaseline, LSTMBaselineChunked
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    feat_dim = ckpt.get('feat_dim', 44)
    use_chunked = config.get('use_chunked', False)
    ModelClass = LSTMBaselineChunked if use_chunked else LSTMBaseline

    kwargs = dict(
        feat_dim=feat_dim,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 2),
        dropout=0.0,
        bidirectional=config.get('bidirectional', True),
    )
    if use_chunked:
        kwargs['chunk_size'] = config.get('chunk_size', 2000)

    model = ModelClass(**kwargs)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_mlp_model(checkpoint_path: str, device: torch.device):
    from models.mlp_baseline import MLPBaseline
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    feat_dim = ckpt.get('feat_dim', 44)
    model = MLPBaseline(
        feat_dim=feat_dim,
        hidden_sizes=config.get('hidden_sizes', [256, 512, 256]),
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    loaders = {
        "simclr": load_simclr_model,
        "bert": load_bert_model,
        "lstm": load_lstm_model,
        "mlp": load_mlp_model,
    }
    return loaders[model_type](checkpoint_path, device)


# --- Representation extraction ---

def extract_representation_batch(model, model_type: str, flight_data_batch: np.ndarray,
                                  device: torch.device, use_amp: bool = True,
                                  mlp_intermediate: nn.Module = None) -> np.ndarray:
    """Extract flight-level representations for a batch of flights.

    Args:
        model: The pre-trained model
        model_type: Type of model (simclr, bert, lstm, mlp)
        flight_data_batch: numpy array of shape (batch_size, seq_len, feat_dim)
        device: torch device
        use_amp: Whether to use automatic mixed precision
        mlp_intermediate: Pre-built intermediate MLP layers (for efficiency)

    Returns:
        numpy array of shape (batch_size, repr_dim)
    """
    # Handle torch.compile() wrapped models
    base_model = getattr(model, '_orig_mod', model)

    with torch.no_grad():
        # Use autocast for AMP when on CUDA
        amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (use_amp and device.type == 'cuda') else torch.inference_mode()

        with amp_context:
            if model_type == "simclr":
                # Input: (B, 1, 4096, 41) — 2D image-like
                x = torch.from_numpy(flight_data_batch).to(device, dtype=torch.float32, non_blocking=True)
                x = x.unsqueeze(1)  # (B, 1, 4096, 41)
                rep = model(x)  # (B, 512) after GAP + Identity projector
                return rep.float().cpu().numpy()

            elif model_type == "bert":
                x = torch.from_numpy(flight_data_batch).to(device, dtype=torch.float32, non_blocking=True)
                encoded = base_model.encoder(x)  # (B, seq_len, hidden_size)
                rep = encoded.mean(dim=1)  # (B, hidden_size)
                return rep.float().cpu().numpy()

            elif model_type == "lstm":
                x = torch.from_numpy(flight_data_batch).to(device, dtype=torch.float32, non_blocking=True)
                projected = base_model.input_proj(x)
                # Process in chunks to avoid memory issues
                chunk_size = getattr(base_model, 'chunk_size', 2000) or 2000
                chunks = []
                for start in range(0, projected.shape[1], chunk_size):
                    chunk = projected[:, start:start + chunk_size, :]
                    chunk_out, _ = base_model.lstm(chunk)
                    chunks.append(chunk_out)
                lstm_out = torch.cat(chunks, dim=1)  # (B, seq_len, hidden)
                rep = lstm_out.mean(dim=1)  # (B, hidden)
                return rep.float().cpu().numpy()

            elif model_type == "mlp":
                x = torch.from_numpy(flight_data_batch).to(device, dtype=torch.float32, non_blocking=True)
                batch_size, seq_len, feat_dim = x.shape
                x_flat = x.reshape(-1, feat_dim)  # (B * seq_len, feat_dim)
                hidden = mlp_intermediate(x_flat)  # (B * seq_len, hidden_dim)
                hidden = hidden.reshape(batch_size, seq_len, -1)  # (B, seq_len, hidden_dim)
                rep = hidden.mean(dim=1)  # (B, hidden_dim)
                return rep.float().cpu().numpy()


def extract_representation(model, model_type: str, flight_data: np.ndarray,
                           device: torch.device) -> np.ndarray:
    """Extract a flight-level representation vector from a pre-trained model.

    This is a convenience wrapper around extract_representation_batch for single samples.
    """
    # Add batch dimension and call batched version
    flight_data_batch = flight_data[np.newaxis, ...]  # (1, seq_len, feat_dim)
    rep_batch = extract_representation_batch(model, model_type, flight_data_batch, device)
    return rep_batch.squeeze(0)  # Remove batch dimension


class FlightDataset(Dataset):
    """PyTorch Dataset for flight data with parallel loading support."""

    def __init__(
        self,
        files: List[Path],
        model_type: str,
        normalization_params: Dict[str, np.ndarray],
        event_flight_ids: Set[int],
        event_type_flight_ids: Dict[str, Set[int]],
        event_types: List[str],
    ):
        self.files = files
        self.model_type = model_type
        self.normalization_params = normalization_params
        self.event_flight_ids = event_flight_ids
        self.event_type_flight_ids = event_type_flight_ids
        self.event_types = event_types

        # Pre-filter files to only include valid ones (with extractable flight_id and aircraft_type)
        self.valid_files = []
        for f in files:
            fid = extract_flight_id(f)
            aircraft_type = extract_aircraft_type(f)
            if fid is not None and aircraft_type is not None:
                self.valid_files.append((f, fid, aircraft_type))

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path, fid, aircraft_type = self.valid_files[idx]

        # Load flight data
        flight_data = load_flight_data(file_path, self.model_type, self.normalization_params)

        # Compute labels
        binary_label = 1 if fid in self.event_flight_ids else 0
        multi_label = np.array([1 if fid in self.event_type_flight_ids[et] else 0
                                for et in self.event_types], dtype=np.int64)
        aircraft_idx = AIRCRAFT_TYPE_TO_IDX[aircraft_type]

        return {
            'flight_data': flight_data.astype(np.float32),
            'flight_id': fid,
            'binary_label': binary_label,
            'multi_label': multi_label,
            'aircraft_idx': aircraft_idx,
        }


def flight_collate_fn(batch):
    """Custom collate function for FlightDataset."""
    flight_data = np.stack([item['flight_data'] for item in batch])
    flight_ids = [item['flight_id'] for item in batch]
    binary_labels = np.array([item['binary_label'] for item in batch])
    multi_labels = np.stack([item['multi_label'] for item in batch])
    aircraft_indices = np.array([item['aircraft_idx'] for item in batch])

    return {
        'flight_data': flight_data,
        'flight_ids': flight_ids,
        'binary_labels': binary_labels,
        'multi_labels': multi_labels,
        'aircraft_indices': aircraft_indices,
    }


def extract_all_representations(
    model, model_type: str, data_dir: str, split: str,
    event_flight_ids: Set[int], event_type_flight_ids: Dict[str, Set[int]],
    event_types: List[str], normalization_params: Dict,
    device: torch.device, max_files: Optional[int] = None,
    batch_size: int = 16, num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Extract representations and labels for all flights in a split.

    Uses batched inference and parallel data loading for improved performance.

    Args:
        model: Pre-trained model
        model_type: Type of model (simclr, bert, lstm, mlp)
        data_dir: Root data directory
        split: Data split (train, test, val)
        event_flight_ids: Set of flight IDs with any anomaly
        event_type_flight_ids: Dict mapping event type -> set of flight IDs
        event_types: List of event type names
        normalization_params: Dict with 'mean' and 'std' arrays
        device: torch device
        max_files: Optional limit on number of files to process
        batch_size: Batch size for inference (default: 16)
        num_workers: Number of parallel data loading workers (default: 4)

    Returns:
        X: representations (N, repr_dim)
        y_binary: binary anomaly labels (N,)
        y_multilabel: multi-label matrix (N, num_event_types)
        y_aircraft: aircraft type labels (N,) - indices into AIRCRAFT_TYPES
        flight_ids: list of flight IDs
    """
    files = get_split_files(data_dir, split)
    if max_files is not None:
        files = files[:max_files]

    # Create dataset and dataloader
    dataset = FlightDataset(
        files, model_type, normalization_params,
        event_flight_ids, event_type_flight_ids, event_types
    )

    # Use pin_memory for faster CPU->GPU transfer when using CUDA
    pin_memory = device.type == 'cuda'

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=flight_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    representations = []
    binary_labels = []
    multilabel_rows = []
    aircraft_labels = []
    flight_ids = []

    total_batches = len(dataloader)
    print(f"Processing {len(dataset)} valid flights in {total_batches} batches "
          f"(batch_size={batch_size}, num_workers={num_workers})")

    # Pre-build MLP intermediate layers once (avoid recreating every batch)
    mlp_intermediate = None
    if model_type == "mlp":
        # Handle torch.compile() wrapped models
        base_model = getattr(model, '_orig_mod', model)
        mlp_intermediate = nn.Sequential(*list(base_model.mlp.children())[:-1])
        mlp_intermediate.to(device)
        mlp_intermediate.eval()

    for batch in tqdm(dataloader, desc=f"Extracting {split} representations"):
        # Extract representations for this batch
        flight_data_batch = batch['flight_data']  # (B, seq_len, feat_dim)
        reps = extract_representation_batch(model, model_type, flight_data_batch, device,
                                            mlp_intermediate=mlp_intermediate)

        representations.append(reps)
        binary_labels.append(batch['binary_labels'])
        multilabel_rows.append(batch['multi_labels'])
        aircraft_labels.append(batch['aircraft_indices'])
        flight_ids.extend(batch['flight_ids'])

    # Concatenate all batches
    X = np.concatenate(representations, axis=0)
    y_binary = np.concatenate(binary_labels, axis=0)
    y_multilabel = np.concatenate(multilabel_rows, axis=0)
    y_aircraft = np.concatenate(aircraft_labels, axis=0)

    skipped = len(files) - len(dataset)
    if skipped > 0:
        print(f"Skipped {skipped} flights in {split} (invalid flight_id or aircraft_type)")

    return X, y_binary, y_multilabel, y_aircraft, flight_ids


def extract_all_representations_legacy(
    model, model_type: str, data_dir: str, split: str,
    event_flight_ids: Set[int], event_type_flight_ids: Dict[str, Set[int]],
    event_types: List[str], normalization_params: Dict,
    device: torch.device, max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Legacy single-sample extraction (kept for debugging/comparison).

    Extract representations and labels for all flights in a split.

    Returns:
        X: representations (N, repr_dim)
        y_binary: binary anomaly labels (N,)
        y_multilabel: multi-label matrix (N, num_event_types)
        y_aircraft: aircraft type labels (N,) - indices into AIRCRAFT_TYPES
        flight_ids: list of flight IDs
    """
    files = get_split_files(data_dir, split)
    if max_files is not None:
        files = files[:max_files]

    representations = []
    binary_labels = []
    multilabel_rows = []
    aircraft_labels = []
    flight_ids = []
    skipped = 0

    for file_path in tqdm(files, desc=f"Extracting {split} representations (legacy)"):
        fid = extract_flight_id(file_path)
        aircraft_type = extract_aircraft_type(file_path)
        if fid is None or aircraft_type is None:
            skipped += 1
            continue

        try:
            flight_data = load_flight_data(file_path, model_type, normalization_params)
            rep = extract_representation(model, model_type, flight_data, device)
            binary_label = 1 if fid in event_flight_ids else 0
            multi_label = [1 if fid in event_type_flight_ids[et] else 0 for et in event_types]
            aircraft_idx = AIRCRAFT_TYPE_TO_IDX[aircraft_type]

            representations.append(rep)
            binary_labels.append(binary_label)
            multilabel_rows.append(multi_label)
            aircraft_labels.append(aircraft_idx)
            flight_ids.append(fid)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            skipped += 1
            continue

        # Clear GPU cache periodically
        if len(representations) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped > 0:
        print(f"Skipped {skipped} flights in {split}")

    X = np.stack(representations)
    y_binary = np.array(binary_labels)
    y_multilabel = np.array(multilabel_rows)
    y_aircraft = np.array(aircraft_labels)
    return X, y_binary, y_multilabel, y_aircraft, flight_ids


def train_and_evaluate_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42,
) -> Dict:
    """Train logistic regression probe and evaluate."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=seed,
        class_weight='balanced',
        C=1.0,
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    return metrics


def evaluate_per_event_type(
    X_train: np.ndarray, y_train_multi: np.ndarray,
    X_test: np.ndarray, y_test_multi: np.ndarray,
    event_types: List[str], seed: int = 42,
    min_positive_samples: int = 5,
) -> Dict:
    """Train one binary classifier per event type and evaluate each."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    per_event_results = {}
    for i, et in enumerate(event_types):
        y_tr = y_train_multi[:, i]
        y_te = y_test_multi[:, i]

        # Skip event types with too few positive samples
        if y_tr.sum() < min_positive_samples or y_te.sum() < min_positive_samples:
            per_event_results[et] = {
                "skipped": True,
                "reason": f"too few positives (train={int(y_tr.sum())}, test={int(y_te.sum())})",
            }
            continue

        clf = LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=seed,
            class_weight='balanced', C=1.0,
        )
        clf.fit(X_train_scaled, y_tr)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        per_event_results[et] = {
            "skipped": False,
            "roc_auc": float(roc_auc_score(y_te, y_prob)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "train_positives": int(y_tr.sum()),
            "test_positives": int(y_te.sum()),
        }

    # Compute macro-average ROC-AUC over non-skipped events
    aucs = [v["roc_auc"] for v in per_event_results.values() if not v.get("skipped")]
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    return {
        "macro_roc_auc": macro_auc,
        "num_evaluated": len(aucs),
        "num_skipped": len(event_types) - len(aucs),
        "per_event": per_event_results,
    }


def train_and_evaluate_aircraft_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42,
) -> Dict:
    """Train multi-class logistic regression for aircraft classification."""
    from sklearn.metrics import confusion_matrix

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multi-class classifier (lbfgs uses multinomial by default for multi-class)
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=seed,
        class_weight='balanced',
        C=1.0,
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics
    per_class_metrics = {}
    for i, aircraft in enumerate(AIRCRAFT_TYPES):
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        y_prob_class = y_prob[:, i]

        per_class_metrics[aircraft] = {
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_binary, y_prob_class)) if y_true_binary.sum() > 0 else 0.0,
            "support": int(y_true_binary.sum()),
        }

    # Macro-averaged metrics
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    macro_precision = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average='macro', zero_division=0))

    # One-vs-rest ROC-AUC
    try:
        macro_roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
    except ValueError:
        macro_roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": AIRCRAFT_TYPES,
    }

    return metrics


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print("=" * 60)
    print(f"Linear Probe Evaluation: {args.model_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Load event labels
    event_flight_ids, event_type_flight_ids, event_types = load_event_labels(args.events_file)

    # Compute normalization parameters
    print("Computing normalization parameters...")
    norm_params = compute_normalization_params(args.data_dir, args.model_type)

    # Load model
    print(f"Loading {args.model_type} model...")
    model = load_model(args.model_type, args.checkpoint, device)
    print("Model loaded.")

    # Enable cudnn benchmark for consistent input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Compile model for faster inference (PyTorch 2.0+)
    if not args.no_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')

    # Warmup pass to initialize CUDA kernels
    if torch.cuda.is_available():
        print("Running warmup pass...")
        seq_len = 4096 if args.model_type == "simclr" else 10000
        feat_dim = len(norm_params['mean'])
        dummy_batch = np.random.randn(2, seq_len, feat_dim).astype(np.float32)
        mlp_intermediate = None
        if args.model_type == "mlp":
            # Handle torch.compile() wrapped models
            base_model = getattr(model, '_orig_mod', model)
            mlp_intermediate = nn.Sequential(*list(base_model.mlp.children())[:-1])
            mlp_intermediate.to(device)
            mlp_intermediate.eval()
        _ = extract_representation_batch(model, args.model_type, dummy_batch, device,
                                         mlp_intermediate=mlp_intermediate)
        torch.cuda.synchronize()
        del dummy_batch

    # Select extraction function
    if args.legacy:
        print("\nUsing legacy single-sample extraction (--legacy flag set)")
        extract_fn = extract_all_representations_legacy
        extract_kwargs = {}
    else:
        print(f"\nUsing batched extraction (batch_size={args.batch_size}, num_workers={args.num_workers})")
        extract_fn = extract_all_representations
        extract_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers}

    # Extract representations
    print("\nExtracting train representations...")
    X_train, y_train, y_train_multi, y_train_aircraft, train_ids = extract_fn(
        model, args.model_type, args.data_dir, "train",
        event_flight_ids, event_type_flight_ids, event_types,
        norm_params, device, args.max_files, **extract_kwargs,
    )
    print(f"Train: {len(y_train)} flights, {y_train.sum()} with events "
          f"({y_train.mean():.1%}), repr dim: {X_train.shape[1]}")
    for i, at in enumerate(AIRCRAFT_TYPES):
        print(f"  {at}: {(y_train_aircraft == i).sum()} flights")

    print("\nExtracting test representations...")
    X_test, y_test, y_test_multi, y_test_aircraft, test_ids = extract_fn(
        model, args.model_type, args.data_dir, "test",
        event_flight_ids, event_type_flight_ids, event_types,
        norm_params, device, args.max_files, **extract_kwargs,
    )
    print(f"Test: {len(y_test)} flights, {y_test.sum()} with events "
          f"({y_test.mean():.1%}), repr dim: {X_test.shape[1]}")
    for i, at in enumerate(AIRCRAFT_TYPES):
        print(f"  {at}: {(y_test_aircraft == i).sum()} flights")

    # Binary classification
    print("\nTraining binary linear probe...")
    binary_metrics = train_and_evaluate_probe(X_train, y_train, X_test, y_test, args.seed)

    print("\n" + "=" * 60)
    print("Binary Anomaly Classification Results:")
    print(f"  ROC-AUC:   {binary_metrics['roc_auc']:.4f}")
    print(f"  Accuracy:  {binary_metrics['accuracy']:.4f}")
    print(f"  Precision: {binary_metrics['precision']:.4f}")
    print(f"  Recall:    {binary_metrics['recall']:.4f}")
    print(f"  F1:        {binary_metrics['f1']:.4f}")
    print("=" * 60)

    # Per-event-type classification
    print("\nTraining per-event-type classifiers...")
    multilabel_results = evaluate_per_event_type(
        X_train, y_train_multi, X_test, y_test_multi,
        event_types, args.seed,
    )

    print(f"\nPer-Event-Type Results (macro ROC-AUC: {multilabel_results['macro_roc_auc']:.4f}):")
    print(f"  Evaluated: {multilabel_results['num_evaluated']}, "
          f"Skipped: {multilabel_results['num_skipped']}")
    print(f"  {'Event Type':<35} {'ROC-AUC':>8} {'F1':>8} {'Train+':>7} {'Test+':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for et in event_types:
        r = multilabel_results['per_event'][et]
        if r.get('skipped'):
            print(f"  {et:<35} {'SKIP':>8} {'':>8} {'':>7} {'':>7}  ({r['reason']})")
        else:
            print(f"  {et:<35} {r['roc_auc']:>8.4f} {r['f1']:>8.4f} {r['train_positives']:>7} {r['test_positives']:>7}")

    # Aircraft classification
    print("\n" + "=" * 60)
    print("Aircraft Classification")
    print("=" * 60)
    print("\nTraining aircraft classification probe...")
    aircraft_metrics = train_and_evaluate_aircraft_probe(
        X_train, y_train_aircraft, X_test, y_test_aircraft, args.seed
    )

    print(f"\nAircraft Classification Results:")
    print(f"  Accuracy:       {aircraft_metrics['accuracy']:.4f}")
    print(f"  Macro ROC-AUC:  {aircraft_metrics['macro_roc_auc']:.4f}")
    print(f"  Macro F1:       {aircraft_metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:{aircraft_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:   {aircraft_metrics['macro_recall']:.4f}")
    print(f"\n  Per-Class Results:")
    print(f"  {'Aircraft':<15} {'ROC-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for aircraft in AIRCRAFT_TYPES:
        m = aircraft_metrics['per_class'][aircraft]
        print(f"  {aircraft:<15} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['support']:>8}")
    print("=" * 60)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "representation_dim": int(X_train.shape[1]),
        "train_flights": int(len(y_train)),
        "test_flights": int(len(y_test)),
        "anomaly_classification": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "binary_metrics": binary_metrics,
            "multilabel_results": multilabel_results,
            "event_types": event_types,
        },
        "aircraft_classification": {
            "metrics": aircraft_metrics,
            "train_distribution": {at: int((y_train_aircraft == i).sum()) for i, at in enumerate(AIRCRAFT_TYPES)},
            "test_distribution": {at: int((y_test_aircraft == i).sum()) for i, at in enumerate(AIRCRAFT_TYPES)},
        },
        "classifier_config": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_file = os.path.join(args.output_dir, f"{args.model_type}_linear_probe.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
