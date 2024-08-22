from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[-1]
