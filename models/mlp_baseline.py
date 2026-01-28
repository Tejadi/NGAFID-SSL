#!/usr/bin/env python3
"""
Simple MLP baseline for masked regression on flight data.

This model has NO temporal context - it predicts each masked timestep
independently using only the features at that timestep. This serves as
a lower bound baseline to show that temporal modeling is valuable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLPBaseline(nn.Module):
    """
    Simple MLP that predicts masked values without temporal context.

    For masked regression, this model:
    1. Takes the masked input (with zeros at masked positions)
    2. Passes each timestep through an MLP independently
    3. Predicts the original values at masked positions

    This is intentionally limited - it cannot use temporal patterns,
    making it a lower bound for models that do use sequence context.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_sizes: list = [256, 512, 256],
        dropout: float = 0.1,
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()

        self.feat_dim = feat_dim

        # Build MLP layers
        layers = []
        in_dim = feat_dim

        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Output projection
        layers.append(nn.Linear(in_dim, feat_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - apply MLP to each timestep independently.

        Args:
            x: Input tensor of shape (batch, seq_len, feat_dim)

        Returns:
            Output tensor of shape (batch, seq_len, feat_dim)
        """
        batch_size, seq_len, feat_dim = x.shape

        # Reshape to (batch * seq_len, feat_dim) for MLP
        x_flat = x.reshape(-1, feat_dim)

        # Apply MLP
        out_flat = self.mlp(x_flat)

        # Reshape back to (batch, seq_len, feat_dim)
        out = out_flat.reshape(batch_size, seq_len, feat_dim)

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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing MLPBaseline...")

    batch_size = 4
    seq_len = 1000
    feat_dim = 44

    model = MLPBaseline(
        feat_dim=feat_dim,
        hidden_sizes=[256, 512, 256],
        dropout=0.1,
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
