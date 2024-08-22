""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""
import timm
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.model.layer.IdGnnLayer import MLP
from GraphLab.model.layer.RnnLayer import rnn_layer
from GraphLab.model.pooling.pooling import pooling_dict, GATModel
from torchvision.models import VisionTransformer, ViT_L_16_Weights
from torchvision.models import resnet50, ResNet50_Weights
from GraphLab.model.layer.SelfAttention import SelfAttention, Attn_Net_Gated
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)

# Head

class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''

    def __init__(self, dim_in, dim_out):
        super(GNNGraphHead, self).__init__()
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)

        self.similarity_matrices = []

        if cfg.dataset.name != 'visualization' and cfg.model.rnn_layer not in ['', 'GAT', 'Cluster'] and cfg.dataset.format != 'LoadImg':
            self.model = rnn_layer[cfg.model.rnn_layer](dim_in, dim_in)
        elif cfg.model.rnn_layer == 'GAT' and cfg.dataset.format != 'LoadImg':
            self.model = GATModel(dim_in, dim_in)
        elif cfg.dataset.format == 'LoadImg':
            self.model1 = rnn_layer[cfg.model.rnn_layer](2 * dim_in, dim_in)
            weights = ResNet50_Weights.DEFAULT
            self.model2 = resnet50(weights=weights)
            # 将模型参数设置为不参与训练
            for param in self.model2.parameters():
                param.requires_grad = False
            self.model2.fc = nn.Linear(self.model2.fc.in_features, dim_in)
        else:
            self.model = None
        if cfg.model.attention:
            self.attention_score = Attn_Net_Gated(dim_in)
        else:
            self.attention_score = None
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]
        if cfg.dataset.multitasking:
            self.node_level_cls = MLP(dim_in, cfg.dataset.subtaskdim, num_layers=cfg.gnn.layers_post_mp, bias=True)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.graph_label

    def forward(self, batch):

        if self.attention_score is not None:
            # 得到每个节点的注意力分数
            # print(batch.node_feature.shape, batch.node_label.shape)
            # batch.node_att_score, similarity_matrix = self.attention_score(batch.node_feature, batch.node_label)
            batch.node_att_score = self.attention_score(batch.node_feature)
            # self.similarity_matrices.append(similarity_matrix.detach().cpu().numpy())  # 确保将Tensor从GPU移动到CPU并转换为NumPy数组
            # 根据注意力分数更新节点的特征
            batch.node_feature = batch.node_feature * batch.node_att_score

        if cfg.dataset.transform == 'ego' and cfg.dataset.format != 'dglmulty':
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch,
                                         batch.node_id_index)
        elif (cfg.dataset.format == 'dglmulty' or cfg.dataset.format == 'deepsnap') and cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(self.model, batch.node_feature, batch.batch, batch.patch_id,
                                         batch.node_id_index)
        elif cfg.dataset.format == 'dglmulty' or cfg.dataset.format == 'deepsnap':
            graph_emb = self.pooling_fun(self.model, batch.node_feature, batch.batch, batch.patch_id)
        elif (cfg.dataset.format == 'LoadImg') and cfg.dataset.transform == 'ego':
            graph_emb = self.pooling_fun(self.model1, batch.node_feature, batch.batch, batch.patch_id,
                                         batch.node_id_index, self.model2,
                                         batch.img)
        elif (cfg.dataset.format == 'LoadImg'):
            graph_emb = self.pooling_fun(self.model1, batch.node_feature, batch.batch, batch.patch_id, None, None,
                                         self.model2, batch.img)
        else:
            graph_emb = self.pooling_fun(batch.node_feature, batch.batch)
        if cfg.dataset.multitasking:
            batch.node_level_feature = self.node_level_cls(batch.node_feature)

        print("graph_emb", graph_emb)
        graph_emb = self.layer_post_mp(graph_emb)
        # graph_emb = F.relu(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        if self.attention_score is not None:
            return pred, label, batch.node_att_score
        else:
            return pred, label


class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''

    def __init__(self, dim_in, dim_out):
        super(GNNNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=cfg.gnn.layers_post_mp,
                                 bias=True)

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return batch.node_feature[batch.node_label_index], \
                batch.node_label[batch.node_label_index]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


# Head models for external interface
head_dict = {
    'graph': GNNGraphHead,
    'node': GNNNodeHead
}

head_dict = {**register.head_dict, **head_dict}
