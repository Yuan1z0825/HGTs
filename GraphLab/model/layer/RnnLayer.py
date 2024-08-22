import torch.nn as nn
import torch.nn.functional as F

from GraphLab import register
from GraphLab.config import cfg
from GraphLab.model.layer.HyperGraph import HGNNPConv
from GraphLab.model.layer.transformer import Transformer
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[-1]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[-1]


class LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attention_scores = F.softmax(self.fc(out), dim=-1)  # 计算自注意力权重
        context_vector = attention_scores * out  # 使用权重和输出相乘，得到上下文向量
        return context_vector[-1]


rnn_layer = {
    'LSTM': LSTM,
    'LSTM_ATTENTION': LSTM_Attention,
    'GRU': GRU,
    'Transformer': Transformer,
    'HyperGraph': HGNNPConv,
}

rnn_layer = {**register.rnn_layer, **rnn_layer}
