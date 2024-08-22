import torch
import torch.nn as nn
import torch.nn.functional as F

import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.model.activation.act import act_dict
from GraphLab.model.layer.IdGnnConv import  *
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
# register additional convs
# General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self,
                 name,
                 dim_in,
                 dim_out,
                 has_act=True,
                 has_bn=True,
                 has_l2norm=False,
                 **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in,
                                      dim_out,
                                      bias=not has_bn,
                                      **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature,
                                                 p=2,
                                                 dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''

    def __init__(self,
                 name,
                 num_layers,
                 dim_in,
                 dim_out,
                 dim_inner=None,
                 final_act=True,
                 **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.edge_feature = self.bn(batch.edge_feature)
        return batch


class MLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 bias=True,
                 dim_inner=32,
                 num_layers=2,
                 **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear',
                                  num_layers - 1,
                                  dim_in,
                                  dim_inner,
                                  dim_inner,
                                  final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
}
layer_dict = {**register.layer_dict, **layer_dict}
