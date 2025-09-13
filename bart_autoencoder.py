import math
import torch
import torch.nn as nn
from transformers import AutoModel


def _extend_bart_positional_embeddings(model: AutoModel, new_max_len: int):
    """
    Safely extend BART's learned positional embeddings in-place, for both encoder
    and decoder, without replacing the module class (keeps BART's custom forward).
    """
    def _grow_inplace(pos_module: nn.Embedding, target_seq_len: int):
        # BART uses an offset (usually 2) so total table = seq_len + offset
        offset = getattr(pos_module, "offset", 2)
        target_n = target_seq_len + offset

        old_n, dim = pos_module.weight.shape
        if target_n <= old_n:
            return  # already big enough

        # Create a bigger weight tensor on the same device/dtype
        with torch.no_grad():
            new_weight = pos_module.weight.new_empty(target_n, dim)
            new_weight[:old_n].copy_(pos_module.weight)
            # Repeat the last vector for new positions
            new_weight[old_n:].copy_(pos_module.weight[-1:].expand(target_n - old_n, -1))
            # Assign as a Parameter to keep it part of the module
            pos_module.weight = nn.Parameter(new_weight, requires_grad=pos_module.weight.requires_grad)

        # Keep metadata consistent with nn.Embedding expectations
        pos_module.num_embeddings = target_n
        pos_module.embedding_dim = dim

    # Extend encoder positions
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_positions"):
        _grow_inplace(model.encoder.embed_positions, new_max_len)

    # Extend decoder positions (safe even if you don't call decoder)
    if hasattr(model, "decoder") and hasattr(model.decoder, "embed_positions"):
        _grow_inplace(model.decoder.embed_positions, new_max_len)

    # Reflect the new logical max sequence length in config
    if hasattr(model, "config"):
        current = getattr(model.config, "max_position_embeddings", new_max_len)
        model.config.max_position_embeddings = max(current, new_max_len)


class FrozenBartBlock(nn.Module):
    def __init__(self, bart_model_name: str, max_len: int):
        super().__init__()
        self.bart = AutoModel.from_pretrained(bart_model_name)

        # Grow learned positional embeddings to max_len (e.g., 10000)
        _extend_bart_positional_embeddings(self.bart, max_len)

        # Freeze all params *after* extension so the new bigger table is frozen too
        for p in self.bart.parameters():
            p.requires_grad = False
        self.bart.eval()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        # No truncation: rely on extended positions. Beware of O(S^2) attention cost for very large S.
        with torch.no_grad():
            out = self.bart.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            ).last_hidden_state
        return out


class BartAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_len=10000,
        bart_model_name="facebook/bart-base",
        latent_dim=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim or d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pre_enc = nn.Linear(d_model, 768)
        self.post_enc = nn.Linear(768, d_model)
        self.to_latent = nn.Linear(d_model, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, d_model)
        self.pre_dec = nn.Linear(d_model, 768)
        self.post_dec = nn.Linear(768, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)
        self.act = nn.GELU()
        self.encoder_bart = FrozenBartBlock(bart_model_name, max_len)
        self.decoder_bart = FrozenBartBlock(bart_model_name, max_len)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        b, s, f = x.shape
        attn = torch.ones(b, s, dtype=torch.long, device=x.device)

        enc_in = self.input_proj(x) * math.sqrt(self.d_model)
        enc_in = self.pre_enc(enc_in)
        enc_out = self.encoder_bart(enc_in, attn)
        enc_out = self.post_enc(enc_out)

        z = self.act(self.to_latent(enc_out))

        dec_in = self.from_latent(z)
        dec_in = self.pre_dec(dec_in)
        dec_out = self.decoder_bart(dec_in, attn)
        dec_out = self.post_dec(dec_out)

        y = self.output_proj(dec_out)
        return y
