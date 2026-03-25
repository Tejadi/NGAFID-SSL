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
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple


class FlightBertEncoder(nn.Module):
    """Memory-optimized BERT encoder for flight time series data with gradient checkpointing."""

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        use_gradient_checkpointing: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention

        # Project flight features to BERT embedding dimension
        self.feature_projection = nn.Linear(feat_dim, hidden_size)

        # Check if Flash Attention is available
        attn_implementation = "eager"
        if use_flash_attention:
            # Try flash_attn package first, then fall back to PyTorch SDPA
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print(f"  Using Flash Attention 2 for O(n) memory attention")
            except ImportError:
                # PyTorch 2.0+ has native SDPA with flash attention support
                import torch
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    attn_implementation = "sdpa"
                    print(f"  Using PyTorch SDPA (native memory-efficient attention)")
                else:
                    print(f"  Flash Attention not available, using standard attention")
                    attn_implementation = "eager"

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
        # Note: attn_implementation requires transformers >= 4.36.0
        try:
            self.bert = BertModel(config, add_pooling_layer=False, attn_implementation=attn_implementation)
        except TypeError:
            # Older transformers version - fall back to default attention
            if attn_implementation != "eager":
                print(f"  Warning: transformers version too old for {attn_implementation}, using default attention")
            self.bert = BertModel(config, add_pooling_layer=False)
        # Remove the word embeddings since we project features directly
        del self.bert.embeddings.word_embeddings

        # Enable gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()

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
    """Memory-optimized decoder with gradient checkpointing, skip connections and layer norm."""

    def __init__(
        self,
        hidden_size: int,
        feat_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_skip_connections: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Build decoder layers with skip connections and layer norm
        self.layers = nn.ModuleList()
        current_dim = hidden_size

        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer outputs original feature dimension
                layer = nn.Sequential(
                    nn.Linear(current_dim, feat_dim),
                )
            else:
                # Sophisticated decoder architecture for large models
                if i == 0:
                    # First layer - expand significantly for large hidden sizes
                    next_dim = hidden_size * 2  # e.g., 1536 -> 3072
                elif i == 1:
                    # Second layer - maintain high capacity
                    next_dim = hidden_size  # e.g., 3072 -> 1536
                elif i == 2:
                    # Third layer - start reducing
                    next_dim = hidden_size // 2  # e.g., 1536 -> 768
                elif i == 3:
                    # Fourth layer - continue reducing
                    next_dim = hidden_size // 4  # e.g., 768 -> 384
                else:
                    # Later layers - gradual reduction but maintain reasonable size
                    reduction_factor = 2 ** (i - 2)
                    next_dim = max(hidden_size // reduction_factor, feat_dim * 8)

                layer = nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                current_dim = next_dim

            self.layers.append(layer)

        # Skip connection projections (when dimensions don't match)
        self.skip_projections = nn.ModuleList()
        if self.use_skip_connections:
            # Calculate dimensions for each layer based on the logic above
            skip_dims = [hidden_size]  # Input dimension
            for i in range(num_layers - 1):  # Exclude final layer
                if i == 0:
                    skip_dims.append(hidden_size * 2)
                elif i == 1:
                    skip_dims.append(hidden_size)
                elif i == 2:
                    skip_dims.append(hidden_size // 2)
                elif i == 3:
                    skip_dims.append(hidden_size // 4)
                else:
                    reduction_factor = 2 ** (i - 2)
                    skip_dims.append(max(hidden_size // reduction_factor, feat_dim * 8))

            for i in range(num_layers - 1):  # No skip for final layer
                # Create projection if dimensions differ, otherwise identity
                if skip_dims[i] != skip_dims[i + 1]:
                    self.skip_projections.append(nn.Linear(skip_dims[i], skip_dims[i + 1]))
                else:
                    self.skip_projections.append(nn.Identity())

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Decode BERT embeddings back to flight features with skip connections and optional checkpointing.

        Args:
            encoded_features: Encoded features (batch_size, seq_len, hidden_size)

        Returns:
            Reconstructed flight data (batch_size, seq_len, feat_dim)
        """
        x = encoded_features

        for i, layer in enumerate(self.layers[:-1]):  # All but final layer
            residual = x

            # Use gradient checkpointing if enabled and training
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

            # Add skip connection if enabled and dimensions allow
            if self.use_skip_connections and i < len(self.skip_projections):
                skip_proj = self.skip_projections[i]
                if self.use_gradient_checkpointing and self.training:
                    x = x + checkpoint(skip_proj, residual, use_reentrant=False)
                else:
                    x = x + skip_proj(residual)

        # Final layer (no skip connection or checkpointing)
        x = self.layers[-1](x)

        return x


class BertMaskedRegressor(nn.Module):
    """Memory-optimized BERT-based masked regression model for flight data."""

    def __init__(
        self,
        feat_dim: int,
        hidden_size: int = 768,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_gradient_checkpointing: bool = True,
        use_mixed_precision: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_flash_attention = use_flash_attention

        # BERT encoder with gradient checkpointing
        self.encoder = FlightBertEncoder(
            feat_dim=feat_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_position_embeddings=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention=use_flash_attention,
        )

        # Decoder with gradient checkpointing
        self.decoder = FlightDecoder(
            hidden_size=hidden_size,
            feat_dim=feat_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def forward(
        self,
        x_masked: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Memory-optimized forward pass with optional checkpointing.

        Args:
            x_masked: Masked input data (batch_size, seq_len, feat_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Reconstructed data (batch_size, seq_len, feat_dim)
        """
        # Encode with optional mixed precision
        if self.use_mixed_precision and self.training:
            with torch.cuda.amp.autocast():
                encoded = self.encoder(x_masked, attention_mask)
                reconstructed = self.decoder(encoded)
        else:
            encoded = self.encoder(x_masked, attention_mask)
            reconstructed = self.decoder(encoded)

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
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss with optional mixed precision.

        Args:
            x_masked: Masked input (batch_size, seq_len, feat_dim)
            x_original: Original unmasked data (batch_size, seq_len, feat_dim)
            mask: Binary mask (1=keep, 0=masked) (batch_size, seq_len, feat_dim)
            attention_mask: Optional attention mask (batch_size, seq_len)
            use_mae_loss: Whether to include MAE in the training loss
            loss_weight_mse: Weight for MSE loss component
            loss_weight_mae: Weight for MAE loss component (if used)
            scaler: Optional GradScaler for mixed precision

        Returns:
            Tuple of (total_loss, mse_loss, mae_loss)
        """
        # Forward pass with optional mixed precision
        if self.use_mixed_precision and self.training and scaler is not None:
            with torch.cuda.amp.autocast():
                reconstructed = self.forward(x_masked, attention_mask)
        else:
            reconstructed = self.forward(x_masked, attention_mask)

        # Compute loss only on masked positions (where mask == 0)
        masked_positions = (mask == 0).float()

        # MSE loss on masked positions only
        mse_loss = F.mse_loss(
            reconstructed * masked_positions,
            x_original * masked_positions,
            reduction='sum'
        )

        # MAE loss on masked positions only
        mae_loss = F.l1_loss(
            reconstructed * masked_positions,
            x_original * masked_positions,
            reduction='sum'
        )

        # Normalize by number of masked elements
        num_masked = masked_positions.sum()
        if num_masked > 0:
            mse_loss = mse_loss / num_masked
            mae_loss = mae_loss / num_masked

        # Combine losses
        if use_mae_loss:
            total_loss = loss_weight_mse * mse_loss + loss_weight_mae * mae_loss
        else:
            total_loss = mse_loss

        return total_loss, mse_loss, mae_loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the memory-optimized model
    batch_size, seq_len, feat_dim = 2, 256, 44  # Realistic flight data dimensions

    model = BertMaskedRegressor(
        feat_dim=feat_dim,
        hidden_size=1536,
        encoder_layers=12,
        decoder_layers=8,
        num_heads=16,
        max_seq_len=seq_len,
        use_gradient_checkpointing=True,
        use_mixed_precision=True,
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Create dummy data
    x_original = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len, feat_dim)).float()
    x_masked = x_original * mask

    # Test forward pass
    with torch.no_grad():
        total_loss, mse_loss, mae_loss = model.compute_loss(x_masked, x_original, mask)
        print(f"Test MSE loss: {mse_loss.item():.4f}")
        print(f"Test MAE loss: {mae_loss.item():.4f}")
        print(f"Test total loss: {total_loss.item():.4f}")
        print("Model test passed!")