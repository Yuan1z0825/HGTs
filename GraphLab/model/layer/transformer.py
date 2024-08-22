import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x





class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=cfg.transformer.num_layers, num_heads=cfg.transformer.num_heads, feedforward_size=cfg.transformer.feedforward_size, dropout_rate=cfg.transformer.dropout_rate):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout_rate)

        # 使用PyTorch的内置TransformerEncoder模块
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=feedforward_size, dropout=dropout_rate),
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)

        # 转换为 (S, N, E) 格式
        src = src.permute(1, 0, 2)

        # 使用TransformerEncoder
        output = self.encoder(src, src_key_padding_mask=src_mask)

        # 恢复为 (N, S, E) 格式
        output = output.permute(1, 0, 2)

        output = self.output_layer(output)
        output = torch.mean(output, dim=1, keepdim=True)  # 按第一维求平均，保留第一维
        return output
