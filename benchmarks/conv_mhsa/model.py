import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (B, H, L, D)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))

        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
        return self.out(attn_output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class ConvMHSAClassifier(nn.Module):
    def __init__(self, input_channels=1, last_token=False, n_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, 7, stride=2, padding=3),
            nn.ReLU(),
        )

        self.encoder_layers = nn.Sequential(
            EncoderLayer(512, 8, 512),
            EncoderLayer(512, 8, 512),
            EncoderLayer(512, 8, 512),
            EncoderLayer(512, 8, 512),
        )

        self.last_token = last_token
        self.final_layer = nn.Linear(512, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv(x)      # (B, 512, L')
        x = x.transpose(1, 2)  # (B, L', 512)

        x = self.encoder_layers(x)

        if self.last_token:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        return torch.sigmoid(self.final_layer(x))


def main():
    model = ConvMHSAClassifier(input_channels=23, last_token=False)
    x = torch.randn(32, 4096, 23)  # (batch_size, sequence_length, channels)
    output = model(x)

    print(output)

if __name__ == "__main__":
    main()
