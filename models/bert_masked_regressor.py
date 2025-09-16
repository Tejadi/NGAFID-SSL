#!/usr/bin/env python3
"""
Simple BERT Encoder + Decoder for Masked Column Regression on Flight Data

Architecture:
1. Input: Masked flight data (seq_len, feat_dim)
2. BERT Encoder: Extract embeddings from masked sequences
3. Custom Decoder: Reconstruct original flight data
4. Loss: Compare reconstructed vs original on masked positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Optional, Tuple


class FlightBertEncoder(nn.Module):
    """BERT encoder for flight time series data."""

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size

        # Project flight features to BERT embedding dimension
        self.feature_projection = nn.Linear(feat_dim, hidden_size)

        # BERT configuration
        config = BertConfig(
            vocab_size=1,  # Not used since we project features directly
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            use_cache=False,
        )

        # Initialize BERT without embeddings (we'll use our own projection)
        self.bert = BertModel(config, add_pooling_layer=False)
        # Remove the word embeddings since we project features directly
        del self.bert.embeddings.word_embeddings

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BERT encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feat_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Encoded features of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, feat_dim = x.shape

        # Project features to BERT embedding dimension
        embeddings = self.feature_projection(x)  # (batch_size, seq_len, hidden_size)

        # Add position embeddings
        position_embeddings = self.bert.embeddings.position_embeddings(
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        embeddings += position_embeddings

        # Add token type embeddings (all zeros for flight data)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings

        # Apply layer norm and dropout
        embeddings = self.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.bert.embeddings.dropout(embeddings)

        # Pass through BERT encoder layers
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return encoder_outputs.last_hidden_state


class FlightDecoder(nn.Module):
    """Decoder to reconstruct flight data from BERT embeddings."""

    def __init__(
        self,
        hidden_size: int,
        feat_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.feat_dim = feat_dim

        # Decoder layers
        layers = []
        current_dim = hidden_size

        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer outputs original feature dimension
                layers.extend([
                    nn.Linear(current_dim, feat_dim),
                ])
            else:
                # Intermediate layers
                next_dim = hidden_size // (2 ** (i + 1))
                next_dim = max(next_dim, feat_dim * 2)  # Don't go too small
                layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                current_dim = next_dim

        self.decoder = nn.Sequential(*layers)

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Decode BERT embeddings back to flight features.

        Args:
            encoded_features: Encoded features (batch_size, seq_len, hidden_size)

        Returns:
            Reconstructed flight data (batch_size, seq_len, feat_dim)
        """
        return self.decoder(encoded_features)


class BertMaskedRegressor(nn.Module):
    """Complete BERT-based masked regression model for flight data."""

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 768,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size

        # BERT encoder
        self.encoder = FlightBertEncoder(
            feat_dim=feat_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_position_embeddings=max_seq_len,
        )

        # Decoder
        self.decoder = FlightDecoder(
            hidden_size=hidden_size,
            feat_dim=feat_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_masked: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: encode masked input and decode to reconstruct original.

        Args:
            x_masked: Masked input data (batch_size, seq_len, feat_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Reconstructed data (batch_size, seq_len, feat_dim)
        """
        # Encode
        encoded = self.encoder(x_masked, attention_mask)

        # Decode
        reconstructed = self.decoder(encoded)

        return reconstructed

    def compute_loss(
        self,
        x_masked: torch.Tensor,
        x_original: torch.Tensor,
        mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss on masked positions only.

        Args:
            x_masked: Masked input (batch_size, seq_len, feat_dim)
            x_original: Original unmasked data (batch_size, seq_len, feat_dim)
            mask: Binary mask (1=keep, 0=masked) (batch_size, seq_len, feat_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Tuple of (total_loss, reconstruction_loss)
        """
        # Forward pass
        reconstructed = self.forward(x_masked, attention_mask)

        # Compute loss only on masked positions (where mask == 0)
        masked_positions = (mask == 0).float()

        # MSE loss on masked positions only
        reconstruction_loss = F.mse_loss(
            reconstructed * masked_positions,
            x_original * masked_positions,
            reduction='sum'
        )

        # Normalize by number of masked elements
        num_masked = masked_positions.sum()
        if num_masked > 0:
            reconstruction_loss = reconstruction_loss / num_masked

        return reconstruction_loss, reconstruction_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    batch_size, seq_len, feat_dim = 4, 128, 20

    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=256,
        encoder_layers=4,
        decoder_layers=2,
        num_heads=8,
        max_seq_len=seq_len,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Create dummy data
    x_original = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len, feat_dim)).float()
    x_masked = x_original * mask

    # Test forward pass
    with torch.no_grad():
        loss, recon_loss = model.compute_loss(x_masked, x_original, mask)
        print(f"Test loss: {loss.item():.4f}")
        print("Model test passed!")