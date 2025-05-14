import torch
import torch.nn as nn

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(2*hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return dec_out
