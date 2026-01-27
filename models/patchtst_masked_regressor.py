#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    def __init__(self, feat_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.patch_len = patch_len
        self.d_model = d_model

        self.linear = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.shape

        num_patches = seq_len // self.patch_len
        actual_len = num_patches * self.patch_len
        x = x[:, :actual_len, :]

        x = x.reshape(batch_size, num_patches, self.patch_len, feat_dim)
        x = x.permute(0, 3, 1, 2)

        x = x.reshape(batch_size * feat_dim, num_patches, self.patch_len)

        x = self.linear(x)

        return x, num_patches


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_embedding = PatchEmbedding(feat_dim, patch_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        batch_size, seq_len, feat_dim = x.shape

        x, num_patches = self.patch_embedding(x)

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        return x, num_patches


class PatchTSTDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        patch_len: int,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.patch_len = patch_len
        self.d_model = d_model

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor, num_patches: int, feat_dim: int, seq_len: int) -> torch.Tensor:
        batch_size = x.size(0) // feat_dim

        x = self.transformer_decoder(x)

        x = self.output_projection(x)

        x = x.reshape(batch_size, feat_dim, num_patches, self.patch_len)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, num_patches * self.patch_len, feat_dim)

        if x.size(1) < seq_len:
            padding = torch.zeros(batch_size, seq_len - x.size(1), feat_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > seq_len:
            x = x[:, :seq_len, :]

        return x


class PatchTSTMaskedRegressor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.encoder = PatchTSTEncoder(
            feat_dim=feat_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=encoder_layers,
            dropout=dropout,
            activation=activation,
        )

        self.decoder = PatchTSTDecoder(
            feat_dim=feat_dim,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=decoder_layers,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.shape

        encoded, num_patches = self.encoder(x)

        reconstructed = self.decoder(encoded, num_patches, feat_dim, seq_len)

        return reconstructed

    def compute_loss(
        self,
        x_masked: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstructed = self.forward(x_masked)

        masked_positions = (mask == 0)

        mse_loss = F.mse_loss(reconstructed[masked_positions], x_original[masked_positions])
        mae_loss = F.l1_loss(reconstructed[masked_positions], x_original[masked_positions])

        loss = mse_loss

        return loss, mse_loss, mae_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing PatchTSTMaskedRegressor...")

    batch_size = 4
    seq_len = 256
    feat_dim = 44
    patch_len = 16
    d_model = 512

    model = PatchTSTMaskedRegressor(
        feat_dim=feat_dim,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=8,
        d_model=d_model,
        n_heads=8,
        d_ff=2048,
        encoder_layers=6,
        decoder_layers=3,
        dropout=0.1,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    x = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.ones_like(x)
    mask[:, :100, :] = 0

    x_masked = x * mask

    with torch.no_grad():
        output = model(x_masked)
        loss, mse_loss, mae_loss = model.compute_loss(x_masked, x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"MAE Loss: {mae_loss.item():.4f}")
    print("Model test passed!")
