import math
import torch
import torch.nn as nn
from transformers import AutoModel

def _extend_bert_positional_embeddings(bert_model: AutoModel, new_max_len: int):
    old = bert_model.embeddings.position_embeddings
    old_n, dim = old.num_embeddings, old.embedding_dim
    if new_max_len <= old_n:
        return
    new = nn.Embedding(new_max_len, dim)
    with torch.no_grad():
        new.weight[:old_n].copy_(old.weight)
        new.weight[old_n:].copy_(old.weight[-1:].repeat(new_max_len - old_n, 1))
    bert_model.embeddings.position_embeddings = new
    bert_model.config.max_position_embeddings = new_max_len

class FrozenBertBlock(nn.Module):
    def __init__(self, bert_model_name: str, max_len: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        _extend_bert_positional_embeddings(self.bert, max_len)
        self.bert.eval()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        return out

class BertAutoencoder(nn.Module):
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
        bert_model_name="bert-base-uncased",
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
        self.encoder_bert = FrozenBertBlock(bert_model_name, max_len)
        self.decoder_bert = FrozenBertBlock(bert_model_name, max_len)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        b, s, f = x.shape
        attn = torch.ones(b, s, dtype=torch.long, device=x.device)
        enc_in = self.input_proj(x) * math.sqrt(self.d_model)
        enc_in = self.pre_enc(enc_in)
        enc_out = self.encoder_bert(enc_in, attn)
        enc_out = self.post_enc(enc_out)
        z = self.act(self.to_latent(enc_out))
        dec_in = self.from_latent(z)
        dec_in = self.pre_dec(dec_in)
        dec_out = self.decoder_bert(dec_in, attn)
        dec_out = self.post_dec(dec_out)
        y = self.output_proj(dec_out)
        return y
