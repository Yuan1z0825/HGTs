import DeepHypergraph.dhg as dhg
import torch
import torch.nn as nn

from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
class HGNNPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        X_ = hg.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_