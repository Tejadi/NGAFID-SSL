#!/usr/bin/env python3
import argparse
import math
import os
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset
from huggingface_hub import list_repo_files

# === Your model ===
try:
    from models.bart_autoencoder import BartAutoencoder
except Exception:
    from bart_autoencoder import BartAutoencoder

# === Masking utilities (user-provided package) ===
try:
    from ngafid_datasets.transformation_dataset import mask_transform, sequential_mask_transform
except Exception as e:
    raise RuntimeError(
        "Couldn't import mask utilities from ngafid_datasets.transformation_dataset. "
        "Please ensure this module is available in your environment."
    ) from e


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HFStreamedMaskedWindows(IterableDataset):
    """
    Streams CSV time-series from a Hugging Face dataset repo (without storing on disk),
    slices each flight into fixed-length windows, applies a masking transform, and yields
    tensors suitable for the BART autoencoder.

    Each item yielded is a tuple of (X_masked, X_orig, mask) with shapes [S, F].
    Batched by the DataLoader into [B, S, F].
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        subdir: str = "preprocessed_data",
        seq_len: int = 256,
        step: int = 256,
        masking: str = "random",
        masking_ratio: float = 0.6,
        mean_mask_length: int = 3,
        start_point: float = 0.5,
        mask_length: int = 10,
        max_files: Optional[int] = None,
        revision: Optional[str] = None,
        seed: int = 0,
        show_file_progress: bool = True,   # shows a tqdm over files
        normalize: str = "file",           # {"none","file"}  z-score per CSV
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.subdir = subdir
        self.seq_len = int(seq_len)
        self.step = int(step)
        self.masking = masking
        self.masking_ratio = masking_ratio
        self.mean_mask_length = int(mean_mask_length)
        self.start_point = start_point
        self.mask_length = int(mask_length)
        self.max_files = max_files
        self.revision = revision or "main"
        self.seed = seed
        self.show_file_progress = show_file_progress
        self.normalize = normalize
        self.norm_eps = norm_eps

        # List files in the HF dataset repo under the given split folder
        prefix = f"{self.subdir}/{self.split}/"
        all_files = list_repo_files(repo_id=self.repo_id, repo_type="dataset", revision=self.revision)
        self.file_paths: List[str] = sorted([fp for fp in all_files if fp.startswith(prefix) and fp.endswith(".csv")])
        if not self.file_paths:
            raise RuntimeError(
                f"No CSV files found under '{prefix}' in dataset repo '{self.repo_id}' "
                f"(revision={self.revision})."
            )
        if self.max_files is not None:
            self.file_paths = self.file_paths[: self.max_files]

        # Probe the first file to discover numeric feature columns and count (F)
        self.feature_columns, self.n_features = self._probe_feature_columns(self.file_paths[0])
        if self.n_features <= 0:
            raise RuntimeError("Failed to detect any numeric feature columns from the first CSV file.")

    def _probe_feature_columns(self, rel_path: str) -> Tuple[List[str], int]:
        """Open a single CSV via streaming Dataset and return the ordered numeric columns we will use."""
        uri = f"hf://datasets/{self.repo_id}/{rel_path}"
        ds = load_dataset("csv", data_files=uri, split="train", streaming=True)
        first = next(iter(ds))
        cols = [k for k, v in first.items() if _is_number(v)]
        return cols, len(cols)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        S = self.seq_len

        pbar_files = tqdm(
            total=len(self.file_paths),
            desc=f"Streaming {self.split} files",
            position=0,
            leave=True,
            disable=not self.show_file_progress,
        )

        try:
            for rel_path in self.file_paths:
                uri = f"hf://datasets/{self.repo_id}/{rel_path}"
                ds = load_dataset("csv", data_files=uri, split="train", streaming=True)

                rows: List[List[float]] = []
                for ex in ds:
                    rows.append([_safe_float(ex.get(col, 0.0)) for col in self.feature_columns])
                if not rows:
                    pbar_files.update(1)
                    continue

                X = np.asarray(rows, dtype=np.float32)  # [T, F]

                # --- NEW: per-file z-score normalization (keeps MSE numerically sane) ---
                if self.normalize == "file":
                    mu = X.mean(axis=0, keepdims=True)
                    sd = X.std(axis=0, keepdims=True)
                    sd = np.maximum(sd, self.norm_eps)
                    X = (X - mu) / sd
                # -----------------------------------------------------------------------

                T = X.shape[0]

                # Yield fixed-length windows with the chosen masking strategy
                for start in range(0, max(0, T - S + 1), self.step):
                    window_np = X[start : start + S, :]  # [S, F]
                    if window_np.shape[0] != S:
                        continue

                    if self.masking == "random":
                        X_orig, X_masked, mask = mask_transform(
                            window_np,
                            masking_ratio=self.masking_ratio,
                            mean_mask_length=self.mean_mask_length,
                            mode="separate",
                            distribution="geometric",
                        )
                    else:
                        X_orig, X_masked, mask = sequential_mask_transform(
                            window_np,
                            starting_point=self.start_point,
                            n=self.mask_length,
                            sequence_length=S,
                        )
                    yield (X_masked, X_orig, mask.float())

                pbar_files.update(1)
        finally:
            pbar_files.close()


def main():
    ap = argparse.ArgumentParser(description="Train BART Autoencoder on streamed HF dataset (no local download).")
    # --- HF streaming options ---
    ap.add_argument("--hf_repo", type=str, default="CDuong04/NGAFID-LOCI-GATS-Data")
    ap.add_argument("--hf_subdir", type=str, default="preprocessed_data")
    ap.add_argument("--hf_split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--hf_revision", type=str, default="main")
    ap.add_argument("--max_files", type=int, default=None)

    # --- Windowing & masking ---
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--masking", type=str, default="random", choices=["random", "sequential"])
    ap.add_argument("--masking_ratio", type=float, default=0.6)
    ap.add_argument("--mean_mask_length", type=int, default=3)
    ap.add_argument("--start_point", type=float, default=0.5)
    ap.add_argument("--mask_length", type=int, default=10)

    # --- Training hyperparams ---
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)  # slightly gentler LR

    # --- Model hyperparams ---
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_encoder_layers", type=int, default=4)
    ap.add_argument("--num_decoder_layers", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=10000)
    ap.add_argument("--bart_name", type=str, default="facebook/bart-base")
    ap.add_argument("--latent_dim", type=int, default=128)

    # --- NEW: loss + normalization controls ---
    ap.add_argument("--loss_on", type=str, default="masked", choices=["masked", "unmasked", "all"],
                    help="Where to compute reconstruction loss. 'masked' is typical for MAE-style training.")
    ap.add_argument("--normalize", type=str, default="file", choices=["none", "file"],
                    help="Per-file z-score normalization before masking.")
    ap.add_argument("--norm_eps", type=float, default=1e-6)

    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[info] device: {device}")

    # Build the streaming dataset (no files saved locally)
    stream_ds = HFStreamedMaskedWindows(
        repo_id=args.hf_repo,
        split=args.hf_split,
        subdir=args.hf_subdir,
        seq_len=args.seq_len,
        step=args.step,
        masking=args.masking,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        start_point=args.start_point,
        mask_length=args.mask_length,
        max_files=args.max_files,
        revision=args.hf_revision,
        show_file_progress=True,
        normalize=args.normalize,
        norm_eps=args.norm_eps,
    )
    print(f"[info] streaming from repo='{args.hf_repo}' split='{args.hf_split}' subdir='{args.hf_subdir}' "
          f"revision='{args.hf_revision}'  |  files: ~{len(stream_ds.file_paths)}")
    print(f"[info] detected numeric feature_count (F) = {stream_ds.n_features}  |  seq_len (S) = {args.seq_len}")

    loader = DataLoader(
        stream_ds,
        batch_size=args.batch_size,
        shuffle=False,          # IterableDataset does not support shuffle=True
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = BartAutoencoder(
        input_dim=stream_ds.n_features,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        bart_model_name=args.bart_name,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"[info] model params (trainable): {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # We’ll compute masked average MSE manually for clarity
    for epoch in range(1, args.epochs + 1):
        print(f"\n[info] ===== Epoch {epoch}/{args.epochs} =====")
        model.train()
        running_num = 0.0   # sum of squared errors over targeted elements
        running_den = 0.0   # number of targeted elements

        pbar = tqdm(loader, desc=f"Epoch {epoch} / train", leave=False, position=1)
        for step_i, batch in enumerate(pbar, 1):
            X_masked, X_orig, mask = batch  # [B, S, F]
            X_masked = X_masked.to(device)
            X_orig   = X_orig.to(device)
            mask     = mask.to(device).float()

            # Decide where to apply the loss
            if args.loss_on == "masked":
                weight = mask
            elif args.loss_on == "unmasked":
                weight = 1.0 - mask
            else:  # "all"
                weight = torch.ones_like(mask)

            opt.zero_grad(set_to_none=True)
            y = model(X_masked)  # [B, S, F]

            sq_err = (y - X_orig) ** 2                      # elementwise
            num = (sq_err * weight).sum()                   # sum over targeted elements
            den = weight.sum().clamp_min(1.0)               # count of targeted elements
            loss = num / den                                # average per targeted element

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_num += float(num.detach().item())
            running_den += float(den.detach().item())

            if step_i % 50 == 0:
                pbar.set_postfix({
                    "mse(target)": f"{(running_num/max(running_den,1.0)):.6f}",
                    "pct_mask": f"{mask.mean().item():.2f}"
                })

        epoch_mse = running_num / max(running_den, 1.0)
        print(f"[info] epoch {epoch:02d}  MSE({args.loss_on}) = {epoch_mse:.6f}")

    print("[info] training finished")


if __name__ == "__main__":
    main()