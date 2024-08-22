import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, PairNorm

import GraphLab.register as register
from GraphLab.DeepLoss import create_Loss_model
from GraphLab.config import cfg
from GraphLab.init import init_weights
from GraphLab.model.activation.act import act_dict
from GraphLab.model.feature_process.feature_augment import Preprocess
from GraphLab.model.head.head import head_dict
from GraphLab.model.layer.IdGnnConv import *
from GraphLab.model.layer.IdGnnLayer import (GeneralMultiLayer, GeneralLayer)
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)

# layer
def GNNLayer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer('linear',
                             cfg.gnn.layers_pre_mp,
                             dim_in,
                             dim_out,
                             dim_inner=32,
                             final_act=True)


class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers, scale=1, use_pairnorm=cfg.model.use_pairnorm):
        super(GNNSkipBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            d_in = dim_in if i == 0 else dim_out
            self.layers.append(GNNLayer(d_in, dim_out))
            if use_pairnorm: # 判断是否使用PairNorm
                self.layers.append(PairNorm(scale=scale))
        d_in = dim_in if num_layers == 1 else dim_out
        self.layers.append(GNNLayer(d_in, dim_out, has_act=False))
        if num_layers > 1 and use_pairnorm: # 判断是否使用PairNorm
            self.layers.append(PairNorm(scale=scale))
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        for layer in self.layers:
            if isinstance(layer, PairNorm):
                batch.node_feature = layer(batch.node_feature)
            else:
                batch = layer(batch)
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = node_feature + batch.node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = torch.cat((node_feature, batch.node_feature), 1)
        else:
            raise ValueError('cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch




# Stage: NN except start and head
class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, dim_in, dim_out, num_layers, scale=1, use_pairnorm=cfg.model.use_pairnorm):
        super(GNNStackStage, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            self.layers.append(GNNLayer(d_in, dim_out))
            if use_pairnorm: # 判断是否使用PairNorm
                self.layers.append(PairNorm(scale=scale))
        self.dim_out = dim_out

    def forward(self, batch):

        # 保存经过GNN后的节点特征和节点类别
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, PairNorm):
                batch.node_feature = layer(batch.node_feature)
            else:
                batch = layer(batch)
            # Save the node features and labels after each layer
        #     torch.save(batch.node_feature, f'/data0/yuanyz/NewGraph/results/node_feature/node_features_layer_{idx}.pt')
        #     torch.save(batch.node_label, f'/data0/yuanyz/NewGraph/results/node_feature/node_labels_layer_{idx}.pt')
        # print("node_feature and node_label saved successfully")
        #
        # raise ValueError("node_feature and node_label saved successfully")

        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)

        # if torch.any(torch.isnan(batch.node_feature)):
        #     print("batch.node_feature contains NaN values")
        # else:
        #     print("batch.node_feature does not contain NaN values")
        return batch



class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers, scale=1):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


# Model: start + stage + head
class GNN(nn.Module):
    '''General GNN model'''

    def __init__(self, dim_in, dim_out, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]
        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        # self.loss_weight = nn.Parameter(torch.tensor(0.9999))
        # self.scale = nn.Parameter(torch.tensor(1.0))
        # self.loss_weight.data = torch.clamp(self.loss_weight, 0.9990, 0.9999)
        # d_in=dim_in
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp,
                               # scale=self.scale
                               )
            d_in = self.mp.dim_out
        if cfg.gnn.DeepsurvUse:
            self.Deepsurv = create_Loss_model(d_in)
            d_in = self.Deepsurv.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)
        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch.node_feature = F.dropout(batch.node_feature, p=cfg.gnn.dropout, training=self.training, inplace=False)
            batch = module(batch)
        return batch
