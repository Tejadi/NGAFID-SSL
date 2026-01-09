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
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.patch_len = patch_len
        self.d_model = d_model

        layers = []
        current_dim = d_model

        for i in range(num_layers - 1):
            next_dim = d_model * 2 if i == 0 else d_model
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, patch_len))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, num_patches: int, batch_size: int) -> torch.Tensor:
        x = self.decoder(x)

        x = x.reshape(batch_size, self.feat_dim, num_patches, self.patch_len)

        x = x.permute(0, 2, 3, 1)

        x = x.reshape(batch_size, num_patches * self.patch_len, self.feat_dim)

        return x


class PatchTSTMaskedRegressor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        seq_len: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
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
        )

        self.decoder = PatchTSTDecoder(
            feat_dim=feat_dim,
            patch_len=patch_len,
            d_model=d_model,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def forward(self, x_masked: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x_masked.size(0)

        encoded, num_patches = self.encoder(x_masked, attention_mask)

        reconstructed = self.decoder(encoded, num_patches, batch_size)

        if reconstructed.size(1) < self.seq_len:
            pad_len = self.seq_len - reconstructed.size(1)
            padding = reconstructed[:, -1:, :].repeat(1, pad_len, 1)
            reconstructed = torch.cat([reconstructed, padding], dim=1)
        elif reconstructed.size(1) > self.seq_len:
            reconstructed = reconstructed[:, :self.seq_len, :]

        return reconstructed

    def compute_loss(
        self,
        x_masked: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_mae_loss: bool = False,
        loss_weight_mse: float = 1.0,
        loss_weight_mae: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstructed = self.forward(x_masked, attention_mask)

        masked_positions = (mask == 0).float()

        mse_loss = F.mse_loss(
            reconstructed * masked_positions,
            x_original * masked_positions,
            reduction='sum'
        )

        mae_loss = F.l1_loss(
            reconstructed * masked_positions,
            x_original * masked_positions,
            reduction='sum'
        )

        num_masked = masked_positions.sum()
        if num_masked > 0:
            mse_loss = mse_loss / num_masked
            mae_loss = mae_loss / num_masked

        if use_mae_loss:
            total_loss = loss_weight_mse * mse_loss + loss_weight_mae * mae_loss
        else:
            total_loss = mse_loss

        return total_loss, mse_loss, mae_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    batch_size, seq_len, feat_dim = 2, 256, 44

    model = PatchTSTMaskedRegressor(
        feat_dim=feat_dim,
        seq_len=seq_len,
        patch_len=16,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        encoder_layers=6,
        decoder_layers=3,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    x_original = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len, feat_dim)).float()
    x_masked = x_original * mask

    with torch.no_grad():
        total_loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
        print(f"Test MSE loss: {mse_loss.item():.4f}")
        print(f"Test MAE loss: {mae_loss.item():.4f}")
        print(f"Test total loss: {total_loss.item():.4f}")
        print("Model test passed!")
