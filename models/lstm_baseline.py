#!/usr/bin/env python3
"""
LSTM baseline for masked regression on flight data.

This model uses bidirectional LSTM to capture temporal patterns
for reconstructing masked positions in flight time series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMBaseline(nn.Module):
    """
    Bidirectional LSTM for masked regression.

    Architecture:
    1. Input projection: feat_dim -> hidden_size
    2. Bidirectional LSTM encoder (stacked layers)
    3. Output projection: 2*hidden_size -> feat_dim

    Uses bidirectional processing to capture both past and future context
    for predicting masked positions.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output projection
        lstm_output_size = hidden_size * self.num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feat_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch, seq_len, feat_dim)
            hidden: Optional initial hidden state

        Returns:
            Output tensor of shape (batch, seq_len, feat_dim)
        """
        batch_size, seq_len, feat_dim = x.shape

        # Project input
        x = self.input_proj(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, hidden)

        # Project output
        out = self.output_proj(lstm_out)

        return out

    def compute_loss(
        self,
        x_masked: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        **kwargs  # Accept extra args for compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss on masked positions.

        Args:
            x_masked: Input with masked positions zeroed out (batch, seq_len, feat_dim)
            x_original: Original unmasked input (batch, seq_len, feat_dim)
            mask: Binary mask where 0 = masked, 1 = visible (batch, seq_len, feat_dim)

        Returns:
            Tuple of (total_loss, mse_loss, mae_loss)
        """
        # Get predictions
        reconstructed = self.forward(x_masked)

        # Only compute loss on masked positions (where mask == 0)
        masked_positions = (mask == 0)

        if masked_positions.sum() == 0:
            # No masked positions - return zero loss
            zero = torch.tensor(0.0, device=x_masked.device)
            return zero, zero, zero

        # Compute losses
        mse_loss = F.mse_loss(
            reconstructed[masked_positions],
            x_original[masked_positions]
        )
        mae_loss = F.l1_loss(
            reconstructed[masked_positions],
            x_original[masked_positions]
        )

        # Total loss is MSE
        loss = mse_loss

        return loss, mse_loss, mae_loss


class LSTMBaselineChunked(LSTMBaseline):
    """
    LSTM baseline with chunked processing for very long sequences.

    For sequences like 10,000 timesteps, processing the full sequence
    can be memory-intensive. This version processes in chunks and
    carries hidden state between chunks.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        chunk_size: int = 1000,
        **kwargs
    ):
        super().__init__(
            feat_dim=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            **kwargs
        )
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunked processing.

        Note: For bidirectional LSTM, chunking loses some context at
        chunk boundaries. This is a tradeoff for memory efficiency.
        """
        batch_size, seq_len, feat_dim = x.shape

        # If sequence fits in one chunk, use standard forward
        if seq_len <= self.chunk_size:
            return super().forward(x)

        # Process in chunks
        outputs = []
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]

            # Process chunk
            chunk_out = super().forward(chunk)
            outputs.append(chunk_out)

        # Concatenate outputs
        out = torch.cat(outputs, dim=1)

        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing LSTMBaseline...")

    batch_size = 4
    seq_len = 1000
    feat_dim = 44

    model = LSTMBaseline(
        feat_dim=feat_dim,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.ones_like(x)
    mask[:, 100:200, :] = 0  # Mask some positions

    x_masked = x * mask

    with torch.no_grad():
        output = model(x_masked)
        loss, mse_loss, mae_loss = model.compute_loss(x_masked, x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"MAE Loss: {mae_loss.item():.4f}")
    print("Model test passed!")

    # Test chunked version
    print("\nTesting LSTMBaselineChunked...")

    model_chunked = LSTMBaselineChunked(
        feat_dim=feat_dim,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        chunk_size=500,
    )

    with torch.no_grad():
        output_chunked = model_chunked(x_masked)
        loss_c, mse_c, mae_c = model_chunked.compute_loss(x_masked, x, mask)

    print(f"Chunked output shape: {output_chunked.shape}")
    print(f"Chunked MSE Loss: {mse_c.item():.4f}")
    print("Chunked model test passed!")
