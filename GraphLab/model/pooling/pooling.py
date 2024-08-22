import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter
import GraphLab.register as register
from DeepHypergraph import dhg
from GraphLab.config import cfg
import timm

from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
from GraphLab.model.layer.MIL import MIL


# Pooling options (pool nodes into graph representations)
# pooling function takes in node embedding [num_nodes x emb_dim] and
# batch (indices) and outputs graph embedding [num_graphs x emb_dim].
def build_adjacency_matrix(tensor, similarity_threshold=0.5):
    similarity_matrix = cosine_similarity(tensor)
    adjacency_matrix = (similarity_matrix > similarity_threshold).astype(float)
    return torch.tensor(adjacency_matrix)


def adjacency_to_edge_index(adjacency_matrix):
    edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t().contiguous()
    return edge_index


class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat_conv1 = GATConv(in_channels, 16, heads=1, dropout=0.1)
        self.gat_conv2 = GATConv(16, 32, heads=1, dropout=0.1)
        self.gat_conv3 = GATConv(16, out_channels, heads=1, concat=False, dropout=0.1)
        self.TopKPooling = TopKPooling(out_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                glorot(m.parameters())

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.gat_conv1(x, edge_index))
        x = F.normalize(x)
        # x = F.relu(self.gat_conv2(x, edge_index))
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.gat_conv3(x, edge_index)
        x = F.normalize(x)

        x, _, _, _, _, _ = self.TopKPooling(x, edge_index)
        return x


def global_add_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        # if torch.any(torch.isnan(batch)):
        #     batch[torch.isnan(batch)] = 0  # 替换 batch 中的 NaN 值为 0
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


def patch_max_pool(model, x, batch, patch_id=None, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
        patch_id = torch.index_select(patch_id, dim=0, index=id)

    bias = 0
    batch_feature = torch.tensor([]).to(torch.device(cfg.device))
    if patch_id is not None:
        for i in range(size):
            index = torch.where(batch == i)
            patient = torch.index_select(x, dim=0, index=index[0])
            patch_size = patch_id[bias:bias + patient.shape[0], 0].long().max().item() + 1
            patch_feature = scatter(patient, patch_id[bias:bias + patient.shape[0], 0].long(), dim=0,
                                    dim_size=patch_size, reduce='max')
            bias += patient.shape[0]
            # patch_feature=model(patch_feature)
            patch_feature = torch.sum(patch_feature, dim=0)
            patch_feature = patch_feature.view(1, -1)
            batch_feature = torch.cat((batch_feature, patch_feature), dim=0)
        return batch_feature
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


def patch_add_pool(x, batch, patch_id=None, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
        patch_id = torch.index_select(patch_id, dim=0, index=id)
        print(patch_id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


def patch_mean_pool(model, x, batch, patch_id=None, id=None, size=None, model2=None, img=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
        patch_id = torch.index_select(patch_id, dim=0, index=id)
        if img is not None:
            img = torch.index_select(img, dim=0, index=id)
    bias = 0
    img_bias = 0
    batch_feature = torch.tensor([]).to(torch.device(cfg.device))
    if patch_id is not None:
        # 遍历所有的病人
        for i in range(size):

            # 选择出单个病人对应的索引
            index = torch.where(batch == i)

            # 提取单个病人所有patch的特征
            patient = torch.index_select(x, dim=0, index=index[0])

            # 找到patch的数目
            patch_size = patch_id[bias:bias + patient.shape[0], 0].long().max().item() + 1
            # if model is not None and model.training :
            #     num_features = random.randint(1, patient.shape[0])
            #     # 生成一个随机数，表示要选择哪些特征
            #     start_idx = random.randint(0, patient.shape[0] - num_features)
            #     # 从 patch_id 中取出一段连续的小块的 ID，并将其转换为整型
            #     selected_features = patch_id[bias:bias + num_features, 0].long()
            #     # 从上面的一段小块中选出随机选择的特征
            #     patient = patient[start_idx:start_idx + num_features]
            #
            #     #更新patch_size的大小
            #     patch_size = selected_features.max().item()+1
            #
            #     patch_feature = scatter(patient, selected_features, dim=0,
            #                             dim_size=patch_size, reduce='mean')
            # else:
            #     patch_feature=scatter(patient,patch_id[bias:bias+patient.shape[0],0].long(),dim=0,dim_size=patch_size,reduce='mean')

            # patient_img

            # 对每个patch内的细胞进行mean pooling
            patch_feature = scatter(patient, patch_id[bias:bias + patient.shape[0], 0].long(), dim=0,
                                    dim_size=patch_size, reduce='mean')

            if cfg.model.Cluster:
                patch_feature = MIL(patch_feature,
                                    patch_feature.shape[1],
                                    patch_feature.shape[1],
                                    cfg.model.n_Cluster,
                                    cfg.model.Cluster_layer_num,
                                    cfg.model.Cluster_epoch)

            if cfg.dataset.format == 'LoadImg':
                patient_imgs = img[img_bias:img_bias + patch_size]
                img_bias += patch_size
                assert patch_feature.shape[0] == patient_imgs.shape[0]
                # print(patch_feature.shape)
                # img_feature .shape  [batch_size,dim_in]
                img_feature = model2(patient_imgs)
                # patch_feature = patch_feature * 0.0
                # img_feature = img_feature*0.0
                patch_feature = torch.cat((patch_feature, img_feature), dim=1)
            bias += patient.shape[0]
            # 计算邻接矩阵并将其转换为边缘索引
            if cfg.model.rnn_layer == 'HyperGraph':
                hg = dhg.Hypergraph.from_feature_kNN(patch_feature, k=5)
                patch_feature = model(patch_feature, hg)
                hg = dhg.Hypergraph.from_feature_kNN(patch_feature, k=5)
                patch_feature = model(patch_feature, hg)
                # patch_feature = model(patch_feature, hg)
                patch_feature = torch.mean(patch_feature, dim=0)
            elif cfg.model.rnn_layer == 'GAT':
                adjacency_matrix = build_adjacency_matrix(patch_feature.detach().cpu())
                edge_index = adjacency_to_edge_index(adjacency_matrix)
                data = Data(x=patch_feature, edge_index=edge_index)
                data.to(torch.device(cfg.device))
                # model = (1, patch_feature.shape[1])
                model.to(torch.device(cfg.device))
                patch_feature = model(data.x, data.edge_index)
                patch_feature = patch_feature.mean(dim=0)
            elif model is not None:
                patch_feature = patch_feature.unsqueeze(0)
                patch_feature = model(patch_feature)
            else:
                patch_feature = torch.mean(patch_feature, dim=0)
            patch_feature = patch_feature.view(1, -1)
            batch_feature = torch.cat((batch_feature, patch_feature), dim=0)
        return batch_feature
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


pooling_dict = {
    'add': global_add_pool,
    'mean': global_mean_pool,
    'max': global_max_pool,
    'patch_add': patch_add_pool,
    'patch_max': patch_max_pool,
    'patch_mean': patch_mean_pool
}

pooling_dict = {**register.pooling_dict, **pooling_dict}
