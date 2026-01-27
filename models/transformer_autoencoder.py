# models/transformer_autoencoder.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        # Add positional encoding up to seq_len
        x = x + self.pe[:seq_len, :]
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=256, dropout=0.1, max_len=10000):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))  # will be learned
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        batch_size, seq_len, feat_dim = x.shape

        # Prepare encoder input
        # Shape for transformer: (seq_len, batch, d_model)
        enc_in = self.input_proj(x)                              
        enc_in = enc_in.permute(1, 0, 2) * math.sqrt(self.d_model) 
        enc_in = self.pos_encoder(enc_in)                         
        # Pass through Transformer encoder
        memory = self.encoder(enc_in)  # memory: (seq_len, batch, d_model)

        mask_tokens = self.mask_token.expand(seq_len, batch_size, self.d_model)
        dec_in = self.pos_decoder(mask_tokens)  # add positional encoding to decoder input
        output = self.decoder(dec_in, memory)   # output: (seq_len, batch, d_model)
        output = output.permute(1, 0, 2)        # (batch, seq_len, d_model)
        output = self.output_proj(output)       # (batch, seq_len, input_dim)
        return output
