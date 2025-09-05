#!/usr/bin/env python3
import argparse
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# your model
try:
    from models.bert_autoencoder import BertAutoencoder
except Exception:
    from bert_autoencoder import BertAutoencoder

#  your masking functions (same ones used in train_autoencoder.py)
# mask_transform returns (X, masked_X, mask) as torch tensors
# sequential_mask_transform returns (X, masked_X, mask) as torch tensors
from transformation_dataset import mask_transform, sequential_mask_transform  # :contentReference[oaicite:2]{index=2}

# -----------------------
# Toy data (shape-correct)
# -----------------------
def make_toy_data(n_flights: int, timesteps: int, n_features: int, seed: int = 0) -> np.ndarray:
    """
    Returns float32 array [n_flights, timesteps, n_features]
    Random-walk-ish values (no file I/O).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_flights, timesteps, n_features)).cumsum(axis=1) / 50.0
    return X.astype(np.float32)

# -----------------------
# Windowed toy dataset
# -----------------------
class ToyMaskedWindowDataset(Dataset):
    """
    Takes flights [N, T, F] and yields (X_masked, X_orig, mask) with shape [1, S, F]
    so your old squeeze(1) in the train loop still works.
    Masking uses your functions from transformation_dataset.py.
    """
    def __init__(
        self,
        flights: np.ndarray,
        seq_len: int = 256,
        step: int = 256,
        masking: str = "random",              # "random" (mask_transform) or "sequential"
        masking_ratio: float = 0.6,           # for random masking
        mean_mask_length: int = 3,            # for random masking
        start_point: float = 0.5,             # for sequential masking
        mask_length: int = 10,                # for sequential masking
        seed: int = 0,
    ):
        self.X = flights                                    # [N, T, F]
        self.N, self.T, self.F = flights.shape
        self.S = seq_len
        self.P = step
        self.masking = masking
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.start_point = start_point
        self.mask_length = mask_length
        self.seed = seed

        # build (flight_idx, start_idx) list
        self.index: List[Tuple[int, int]] = []
        for i in range(self.N):
            if self.T >= self.S:
                num = 1 + (self.T - self.S) // self.P
                self.index += [(i, j * self.P) for j in range(num)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, s = self.index[k]
        e = s + self.S
        window_np = self.X[i, s:e, :]  # [S, F] numpy

        # --- use YOUR masking funcs, exactly like in train_autoencoder.py ---
        # random/geometric masking
        if self.masking == "random":
            X, masked_X, mask = mask_transform(
                window_np,
                masking_ratio=self.masking_ratio,
                mean_mask_length=self.mean_mask_length,
                mode="separate",
                distribution="geometric",
            )  # -> torch tensors (S,F)  :contentReference[oaicite:3]{index=3}
        else:
            # sequential masking
            X, masked_X, mask = sequential_mask_transform(
                window_np,
                starting_point=self.start_point,
                n=self.mask_length,
                sequence_length=self.S,
            )  # -> torch tensors (S,F)  :contentReference[oaicite:4]{index=4}

        # Keep the extra dim at axis 0 to match old code path ([1,S,F])
        return masked_X.unsqueeze(0), X.unsqueeze(0), mask.unsqueeze(0)

# -------------
# Training loop
# -------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ap = argparse.ArgumentParser()
    # data + windows
    ap.add_argument("--n_flights",   type=int, default=8)
    ap.add_argument("--timesteps",   type=int, default=3000)
    ap.add_argument("--n_features",  type=int, default=44)      # set 58 to match discovered columns
    ap.add_argument("--seq_len",     type=int, default=256)
    ap.add_argument("--step",        type=int, default=256)
    ap.add_argument("--seed",        type=int, default=0)

    # masking (choose one)
    ap.add_argument("--masking",     type=str, default="random", choices=["random", "sequential"])
    ap.add_argument("--masking_ratio", type=float, default=0.6)   # random
    ap.add_argument("--mean_mask_length", type=int, default=3)     # random
    ap.add_argument("--start_point", type=float, default=0.5)      # sequential
    ap.add_argument("--mask_length", type=int, default=10)         # sequential

    # training
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=3)
    ap.add_argument("--lr",          type=float, default=1e-3)

    # model hyperparams
    ap.add_argument("--d_model",     type=int, default=128)
    ap.add_argument("--num_heads",   type=int, default=8)
    ap.add_argument("--num_encoder_layers", type=int, default=4)
    ap.add_argument("--num_decoder_layers", type=int, default=4)
    ap.add_argument("--ff_dim",      type=int, default=256)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--max_len",     type=int, default=10000)
    ap.add_argument("--bert_name",   type=str,  default="bert-base-uncased")
    ap.add_argument("--latent_dim",  type=int,  default=128)

    args = ap.parse_args()

    # device (CPU / Apple Silicon MPS / CUDA)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[info] device: {device}")

    # builds toy data
    print("[info] generating toy data…")
    flights = make_toy_data(args.n_flights, args.timesteps, args.n_features, seed=args.seed)
    N, T, F = flights.shape
    print(f"[info] data shape = [n_flights={N}, timesteps={T}, n_features={F}]")

    # dataset / loader
    ds = ToyMaskedWindowDataset(
        flights,
        seq_len=args.seq_len,
        step=args.step,
        masking=args.masking,
        masking_ratio=args.masking_ratio,
        mean_mask_length=args.mean_mask_length,
        start_point=args.start_point,
        mask_length=args.mask_length,
        seed=args.seed,
    )
    num_windows = len(ds)
    num_batches = (num_windows + args.batch_size - 1) // args.batch_size
    print(f"[info] windows/epoch: {num_windows}  |  batch_size: {args.batch_size}  |  ~batches/epoch: {num_batches}")

    num_workers = 0 if device.type in ("cpu", "mps") else 2
    pin_memory = device.type == "cuda"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)

    # model
    model = BertAutoencoder(
        input_dim=args.n_features,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        bert_model_name=args.bert_name,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"[info] model params (trainable): {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="sum")

    # train
    for epoch in range(1, args.epochs + 1):
        print(f"\n[info] ===== Epoch {epoch}/{args.epochs} =====")
        model.train()
        total_loss = 0.0
        total_elems = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} / train", leave=False)
        for step_i, (X_masked, X_orig, mask) in enumerate(pbar, 1):
            # shapes from dataset: [B, 1, S, F]
            X_masked = X_masked.to(device)
            X_orig   = X_orig.to(device)
            mask     = mask.to(device)

            # your old code squeezes this extra dim; keep that behavior
            X_masked = X_masked.squeeze(1)  # [B, S, F]
            X_orig   = X_orig.squeeze(1)    # [B, S, F]
            mask     = mask.squeeze(1).float()  # [B, S, F], 1=keep, 0=masked

            opt.zero_grad(set_to_none=True)
            y = model(X_masked)  # [B, S, F]

            # compute MSE only on masked positions (mask==0), same idea as your earlier script
            weight = (1.0 - mask)
            loss = criterion(y * weight, X_orig * weight)

            loss.backward()
            opt.step()

            total_loss  += loss.item()
            total_elems += weight.sum().item()

            if step_i % 25 == 0:
                pbar.set_postfix({"running_mse(masked)": total_loss / max(total_elems, 1)})

        avg = total_loss / max(total_elems, 1)
        print(f"[info] epoch {epoch:02d}  MSE(masked) = {avg:.6f}")

    print("[info] training finished")

if __name__ == "__main__":
    main()
