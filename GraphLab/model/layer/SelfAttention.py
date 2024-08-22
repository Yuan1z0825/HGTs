import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import scatter
from torch_geometric.nn.inits import glorot, zeros

from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything
from torch_scatter import scatter_mean
seed_anything(cfg.seed)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.score_layer = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Softmax(dim=0)
        )
        self.score_layer.apply(initialize_weights)

    def forward(self, x):
        scores = self.score_layer(x)
        return scores

# class Attn_Net_Gated(nn.Module):
#     def __init__(self, dim=256, dropout=True, out_dim=1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [
#             nn.Linear(dim, dim),
#             nn.Tanh()]
#
#         self.attention_b = [nn.Linear(dim, dim),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(cfg.gnn.dropout))
#             self.attention_b.append(nn.Dropout(cfg.gnn.dropout))
#
#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#
#         self.attention_c = nn.Linear(dim, out_dim)
#
#     def forward(self, x, node_id_index):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#
#         # Compute the similarity matrix
#         similarity_matrix = torch.mm(A, A.t())
#         print(torch.unique(node_id_index).size(0))  # Unique class count
#
#         # Aggregate rows
#         aggregated_rows = scatter_mean(similarity_matrix, index=node_id_index, dim=0)
#
#         # Aggregate columns
#         aggregated_columns = scatter_mean(aggregated_rows.t(), index=node_id_index, dim=0).t()
#         required_columns = 8
#
#         # Check if padding is needed
#         if aggregated_columns.size(1) < required_columns:
#             padding = torch.zeros(aggregated_columns.size(0), required_columns - aggregated_columns.size(1),
#                                   device=aggregated_columns.device)
#             aggregated_columns = torch.cat([aggregated_columns, padding], dim=1)
#
#             # Check if row padding is needed
#             if aggregated_columns.size(0) < required_columns:
#                 row_padding = torch.zeros(required_columns - aggregated_columns.size(0), aggregated_columns.size(1),
#                                           device=aggregated_columns.device)
#                 aggregated_columns = torch.cat([aggregated_columns, row_padding], dim=0)
#
#         print(aggregated_columns.size(), '----')
#
#         # Continue with forward propagation
#         A = self.attention_c(A)  # N x out_dim
#         return A, aggregated_columns


class Attn_Net_Gated(nn.Module):
    def __init__(self, dim=256, dropout=True, out_dim=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(dim, dim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(dim, dim),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(cfg.gnn.dropout))
            self.attention_b.append(nn.Dropout(cfg.gnn.dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(dim, out_dim)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A