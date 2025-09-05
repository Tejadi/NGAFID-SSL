#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- model import (works whether bert_autoencoder.py is in models/ or same dir) ---
try:
    from models.bert_autoencoder import BertAutoencoder
except Exception:
    from bert_autoencoder import BertAutoencoder

# --- optional wandb ---
def maybe_init_wandb(use_wandb: bool, **cfg):
    if not use_wandb:
        class _Noop:
            def log(self, *_, **__): pass
        return _Noop()
    import wandb
    wandb.init(project=cfg.pop("project", "ngafid-toy"),
               name=cfg.pop("name", "BERT-AE-toy"),
               config=cfg)
    return wandb

# -----------------------
# Toy data (shape-correct)
# -----------------------
def make_toy_data(n_flights: int, timesteps: int, n_features: int, seed: int = 0) -> np.ndarray:
    """
    returns float32 array [n_flights, timesteps, n_features]
    random-walk-ish values to look time-seriesy (no file IO)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_flights, timesteps, n_features)).cumsum(axis=1) / 50.0
    return X.astype(np.float32)

# ---------------
# Masking dataset
# ---------------
@dataclass
class MaskCfg:
    seq_len: int = 512         # window length
    step: int = 512            # stride between windows
    mask_ratio: float = 0.3    # fraction of timesteps to mask (0..1)
    per_feature: bool = False  # if True, mask each (t,f) independently

class MaskedWindowDataset(Dataset):
    """
    Takes flights [N, T, F] and yields (x_masked, x_orig, mask) shaped [S, F]
    where mask==1 marks masked elements.
    """
    def __init__(self, flights: np.ndarray, cfg: MaskCfg):
        self.cfg = cfg
        self.X = flights                                      # [N, T, F]
        self.N, self.T, self.F = flights.shape
        self.index: List[Tuple[int, int]] = []

        S, P = cfg.seq_len, cfg.step
        for i in range(self.N):
            if self.T >= S:
                num = 1 + (self.T - S) // P
                self.index += [(i, j * P) for j in range(num)]

    def __len__(self): return len(self.index)

    def __getitem__(self, k: int):
        i, s = self.index[k]
        e = s + self.cfg.seq_len
        x = self.X[i, s:e, :]                                  # [S, F]

        if self.cfg.per_feature:
            mask = (np.random.rand(*x.shape) < self.cfg.mask_ratio).astype(np.float32)
        else:
            # mask whole time-steps and broadcast over features
            tmask = (np.random.rand(x.shape[0], 1) < self.cfg.mask_ratio).astype(np.float32)
            mask = np.repeat(tmask, x.shape[1], axis=1).astype(np.float32)

        x_masked = x.copy()
        x_masked[mask.astype(bool)] = 0.0                      # zero-mask; your model can learn mask token

        return (
            torch.tensor(x_masked, dtype=torch.float32),       # [S, F]
            torch.tensor(x,        dtype=torch.float32),       # [S, F]
            torch.tensor(mask,     dtype=torch.float32),       # [S, F]
        )

# -------------
# Training loop
# -------------
def train_one_epoch(model, loader, opt, device, wb, epoch_idx):
    model.train()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_elems = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx} / Train", leave=False)
    for step, (x_masked, x_orig, mask) in enumerate(pbar, 1):
        x_masked = x_masked.to(device)           # [B, S, F]
        x_orig   = x_orig.to(device)
        mask     = mask.to(device)

        opt.zero_grad(set_to_none=True)
        y = model(x_masked)                      # -> [B, S, F]
        inv = (1.0 - mask)                       # unmasked = 1
        loss = mse(y * inv, x_orig * inv)
        loss.backward()
        opt.step()

        total_loss  += loss.item()
        total_elems += inv.sum().item()

        if step % 25 == 0:
            pbar.set_postfix({"running_mse_unmasked": total_loss / max(total_elems, 1)})

    avg = total_loss / max(total_elems, 1)
    wb.log({"mse_unmasked": avg, "epoch": epoch_idx})
    return avg

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ap = argparse.ArgumentParser()
    # data shape
    ap.add_argument("--n_flights",   type=int, default=10)
    ap.add_argument("--timesteps",   type=int, default=4000)
    ap.add_argument("--n_features",  type=int, default=44)     # set 58 to match your discovered cols
    # windowing / masking
    ap.add_argument("--seq_len",     type=int, default=256)
    ap.add_argument("--step",        type=int, default=256)
    ap.add_argument("--mask_ratio",  type=float, default=0.3)
    ap.add_argument("--per_feature", action="store_true")
    # training
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=3)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--use_wandb",   action="store_true")
    ap.add_argument("--wandb_project", type=str, default="ngafid-toy")
    # model hyperparams (match your BertAutoencoder signature)
    ap.add_argument("--d_model",     type=int, default=128)
    ap.add_argument("--num_heads",   type=int, default=8)
    ap.add_argument("--dec_layers",  type=int, default=4)
    ap.add_argument("--ff_dim",      type=int, default=256)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--max_len",     type=int, default=10000)
    ap.add_argument("--bert_name",   type=str,  default="bert-base-uncased")
    ap.add_argument("--latent_dim",  type=int,  default=128)
    args = ap.parse_args()

    # wandb
    wb = maybe_init_wandb(
        args.use_wandb,
        project=args.wandb_project,
        name=f"toy-BERTAE nf={args.n_features} S={args.seq_len} mr={args.mask_ratio}",
        learning_rate=args.lr, epochs=args.epochs
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # (optional) if you're on CPU, fewer workers & no pin_memory usually helps
    num_workers = 0 if device.type == "cpu" else 2
    pin_memory = device.type != "cpu"

    # make toy data
    print("[info] generating toy data...")
    flights = make_toy_data(args.n_flights, args.timesteps, args.n_features)
    N, T, F = flights.shape
    print(f"[info] data shape = [n_flights={N}, timesteps={T}, n_features={F}]")

    # dataset / loader
    ds = MaskedWindowDataset(
        flights,
        MaskCfg(seq_len=args.seq_len, step=args.step,
                mask_ratio=args.mask_ratio, per_feature=args.per_feature)
    )
    num_windows = len(ds)
    num_batches = (num_windows + args.batch_size - 1) // args.batch_size
    print(f"[info] windows per epoch: {num_windows}  |  batch_size: {args.batch_size}  |  ~batches/epoch: {num_batches}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

    # model
    model = BertAutoencoder(
        input_dim=args.n_features,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        bert_model_name=args.bert_name,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"[info] model params (trainable): {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train
    for epoch in range(1, args.epochs + 1):
        print(f"\n[info] ===== Epoch {epoch}/{args.epochs} =====")
        mse_unmasked = train_one_epoch(model, dl, opt, device, wb, epoch)
        print(f"[info] epoch {epoch:02d}  MSE(unmasked) = {mse_unmasked:.6f}")

    print("[info] training finished ✅")

if __name__ == "__main__":
    main()
