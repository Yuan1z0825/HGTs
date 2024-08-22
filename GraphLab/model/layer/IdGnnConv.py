import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (add_remaining_self_loops, remove_self_loops, add_self_loops, softmax)
from torch_scatter import scatter_add

from GraphLab.config import cfg
from GraphLab.register import register_layer
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
class GATIDConvLayer(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        super(GATIDConvLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        # 不同的参数
        self.weight_id = nn.ParameterList(Parameter(torch.Tensor(in_channels, out_channels)) for _ in
                          range(7))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        for item in self.weight_id:
            glorot(item)
        zeros(self.bias)
        glorot(self.att)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, id=None, node_label=None, patch_id=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            # x=torch.matmul(x,self.weight)
            ox = x.clone()
            for i in range(7):
                index = torch.nonzero(node_label == i + 1)[:, 0]
                if len(index) != 0:
                    cell = torch.index_select(x, dim=0, index=index)
                    cell = torch.matmul(cell, self.weight_id[i])
                    ox.index_add_(0, index, cell)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))
        return self.propagate(edge_index, size=size, x=ox)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        result = x_j * alpha.view(-1, self.heads, 1)
        result = result.view(-1, self.out_channels)
        return result

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GeneralIDConvLayer(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(GeneralIDConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.heads = 1
        self.normalize = cfg.gnn.normalize_adj

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        # 不同的参数
        self.weight_id = nn.ParameterList([nn.Parameter(torch.Tensor(in_channels, out_channels)) for _ in range(7)])
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        for item in self.weight_id:
            glorot(item)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index,
             num_nodes,
             edge_weight=None,
             improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, id=None, node_label=None, patch_id=None, edge_weight=None):
        '''

        :param x: 节点的特征向量二维
        :param edge_index: 边的索引
        :param id: 中心节点的索引列表
        :param node_label: 每个节点的标签
        :param edge_weight: 边的权值
        :return:
        '''

        # if torch.any(torch.isnan(x)):
        #     print("x contains NaN values")
        # else:
        #     print("x does not contain NaN values")

        # print(x.shape,id,id.shape)
        if id is not None:
            x_id = torch.index_select(x, dim=0, index=id)
            x_id = torch.matmul(x_id, self.weight)
            x.index_add_(0, id, x_id)

        ox = x.clone()

        # ox = torch.matmul(ox, self.weight)

        for i in range(7):
            index = torch.nonzero(node_label == i + 1)[:, 0]
            if len(index) != 0:
                cell = torch.index_select(x, dim=0, index=index)
                cell = torch.matmul(cell, self.weight_id[i])
                ox.index_add_(0, index, cell)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, ox.size(self.node_dim),
                                             edge_weight, self.improved,
                                             ox.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=ox, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GeneralIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralIDConv, self).__init__()
        self.model = GeneralIDConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if cfg.dataset.transform == 'ego':
            batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                            id=batch.node_id_index, node_label=batch.node_label,
                                            patch_id=batch.patch_id)
        else:
            batch.node_feature = self.model(batch.node_feature, batch.edge_index, id=None, node_label=batch.node_label)

        # 保存batch信息到文件
        # torch.save(batch, 'batch.pt')
        return batch


class GATIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATIDConv, self).__init__()
        self.model = GATIDConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):

        if cfg.dataset.format == 'dglmulty':
            batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                            id=batch.node_id_index, node_label=batch.node_label,
                                            patch_id=batch.patch_id)
        else:
            batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                            id=batch.node_id_index, node_label=batch.node_label)
        return batch


register_layer('idconv', GeneralIDConv)
register_layer('gatidconv', GATIDConv)
