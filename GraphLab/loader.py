import logging
import os
import pickle
import random
import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from PIL import Image
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph as DeepSnapG
from dgl import load_graphs
from histocartography.visualization import OverlayGraphVisualization
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset)
from torchvision import transforms
import torchstain
import cv2
from torchvision import transforms
import GraphLab.model.feature_process.feature_augment as preprocess
import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.model.transform.transform import (edge_nets, ego_nets)
from sklearn.decomposition import PCA

from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
pca = None
if cfg.train.pca > 0:
    pca = PCA(n_components=cfg.train.pca)

def one_hot_encode(x):
    one_hot = np.eye(7)[x]
    return one_hot


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLHeteroGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 定义预处理转换
PreprocessImg = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch, type):
    return COLLATE_FN[type]([example for example in batch])


def load_cg(cg_path):
    """
    Load cell graphs
    """
    # Cell_Graph = load_graphs(cg_path)
    # paths=(os.path.join(cg_path,"AllCell.bin"))
    # print(paths)
    Cell_Graph = load_graphs((os.path.join(cg_path, "AllCell.bin")))

    return Cell_Graph


def load_pyg(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name, dataset_dir):
    '''
    load networkx format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except Exception:
        graphs = nx.read_gpickle('{}/{}.gpickle'.format(dataset_dir, name))

        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset(form=None):
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
        # for graph in graphs:
        #     nx.draw(graph)
    # Load from OGB formatted data
    elif format == 'dgl':
        graphs = load_dgl(form)
    elif format == 'deepsnap':
        graphs = load_deepsnap(form)
    elif format == 'dglbatch':
        graphs, labels = load_dgl_new(form)
        return graphs, labels
    elif format == 'dglmulty' or format == 'LoadImg':
        graphs, labels, patch_path = load_dgl_Multy(form)
        return graphs, labels, patch_path
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 100
    else:
        min_node = 5
    return min_node


def transform_before_split(dataset, form=None):
    '''
    Dataset transformation before train/val/test split
    :param dataset: A DeepSNAP dataset object
    :return: A transformed DeepSNAP dataset object
    '''
    augmentation = preprocess.FeatureAugment()
    actual_feat_dims = augmentation.augment(dataset)
    # Update augmented feature/label dims by real dims (user specified dims
    # may not be realized)
    if form == 'test':
        cfg.dataset.augment_feature_dims = actual_feat_dims
    return dataset


def transform_after_split(datasets):
    '''
    Dataset transformation after train/val/test split
    :param dataset: A list of DeepSNAP dataset objects
    :return: A list of transformed DeepSNAP dataset objects
    '''
    if cfg.dataset.transform == 'ego':
        for split_dataset in datasets:
            split_dataset.apply_transform(ego_nets,
                                          radius=cfg.gnn.layers_mp,
                                          update_tensor=True,
                                          update_graph=False)
    elif cfg.dataset.transform == 'edge':
        for split_dataset in datasets:
            split_dataset.apply_transform(edge_nets,
                                          update_tensor=True,
                                          update_graph=False)
            split_dataset.task = 'node'
        cfg.dataset.task = 'node'
    return datasets


def set_dataset_info(datasets):
    r"""
    Set global dataset information

    Args:
        datasets: List of dataset object

    """
    # get dim_in and dim_out
    try:
        cfg.share.dim_in = datasets[0].num_node_features
    except Exception:
        cfg.share.dim_in = 1
    try:
        cfg.share.dim_out = 1
        # cfg.share.dim_out = datasets[0].num_labels
        # if 'classification' in cfg.dataset.task_type and \
        #         cfg.share.dim_out == 2:
        #     cfg.share.dim_out = 1
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    cfg.share.num_splits = len(datasets)


def stack_patch(id, num):
    temp = []
    for i in range(num):
        temp.append(torch.tensor(id))
    ans = torch.stack(temp, dim=0)
    return ans


count = 0


# 定义替换函数
def replace_tensor(x):
    if random.random() < cfg.model.p_value:
        return torch.zeros_like(x)
    else:
        return x


from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ProcessPoolExecutor
# 将处理单个图的部分封装为一个函数
def process_graphs(graphs, labels, patch_path):
    temp_graphs = []
    for ind, graph_info in enumerate(graphs):
        patch_id = torch.ones((graph_info.num_nodes(), 1)).mul(ind)
        edge_id = torch.ones((graph_info.num_edges(), 1)).mul(ind)
        graph_info.ndata['patch_id'] = patch_id
        graph_info.edata['gid'] = edge_id
        temp_graphs.append(graph_info)
    temp_graphs = dgl.batch(temp_graphs)
    src, dst = temp_graphs.edges()
    G = nx.DiGraph(zip(src.tolist(), dst.tolist()))

    node_features1 = temp_graphs.ndata['feat'][:, 2:23 - 2 - 7]
    node_features2 = temp_graphs.ndata['feat'][:, 21:23]
    node_features = torch.cat((node_features1, node_features2), dim=1)
    node_labels = temp_graphs.ndata['name']
    patch_id = temp_graphs.ndata['patch_id']
    eid = temp_graphs.edata['gid']
    graph_labels = labels
    path = patch_path

    graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                              graph_label=graph_labels, patch_id=patch_id, path=path, eid=eid)
    return graph

# 使用 ThreadPoolExecutor 并发地处理图
def parallel_processing(Graphs, labels, patch_path):
    with ProcessPoolExecutor(max_workers=4) as executor:
        #func = partial(process_graphs, labels=labels, patch_path=patch_path)

        DeepSnap_Graph = executor.map(process_graphs, Graphs,labels,patch_path)
    for item in DeepSnap_Graph:
        print(item)
    return DeepSnap_Graph

def transform_to_DeepSnap(Graphs, labels=None, imgs=None, Flag=1, form='train', patch_path=[]):
    '''

    :param Graphs: 表示带有标签的DGL图
    :Flag: 2代表dglmulty  1代表 dglbatch   0代表dgl不需要labels 3代表LoadImg
    :return: DeepSnap Graph
    '''

    if Flag == 3:
        DeepSnap_Graph = []
        index = 0
        from tqdm import tqdm
        path = os.path.join('/data0/yuanyz/NewGraph', cfg.dataset.name, form)
        if not os.path.exists(path):
            os.makedirs(path)
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open(path + '/no.pickle', 'wb') as f:
            for graphs in tqdm(Graphs):
                temp_graphs = []
                for ind, graph_info in enumerate(graphs):
                    patch_id = torch.ones((graph_info.num_nodes(), 1)).mul(ind)
                    edge_id = torch.ones((graph_info.num_edges(), 1)).mul(ind)
                    graph_info.ndata['patch_id'] = patch_id
                    graph_info.edata['gid'] = edge_id
                    temp_graphs.append(graph_info)
                temp_graphs = dgl.batch(temp_graphs)
                #src, dst = temp_graphs.edges()
                G = nx.DiGraph(temp_graphs.to_networkx())
                node_features1 = temp_graphs.ndata['feat'][:, :24]
                node_features2 = temp_graphs.ndata['feat'][:, 31:33]
                node_features = torch.cat((node_features1, node_features2), dim=1)
                node_labels = temp_graphs.ndata['name']
                patch_id = temp_graphs.ndata['patch_id']
                eid = temp_graphs.edata['gid']
                graph_labels = labels[index]
                path = patch_path[index]
                assert len(path) == len(graphs)
                imgs=[]
                for idx, patch in enumerate(path):
                    input_image = Image.open(patch + "/wsi.png").convert('RGB')
                    input_image = T(input_image)

                    # img = input_image.copy()
                    # input_image = np.array(input_image)[:,:,:3]
                    # input_image = Image.fromarray(input_image)
                    # input_image = PreprocessImg(input_image)

                    imgs.append(input_image)

                    # visualizer = OverlayGraphVisualization(
                    #     node_style='fill',
                    #     node_radius=3,
                    #     edge_thickness=1,
                    #     colormap='jet',
                    #     show_colormap=True,
                    #     min_max_color_normalize=True,
                    # )
                    # canvas = visualizer.process(img, graphs[idx])
                    # # canvas = visualizer.process(img, graph)
                    # canvas.save(
                    #     os.path.join('/home/yuanyz/Run/test_sny/'+str(index)+"_"+str(idx) + ".png"),
                    #     quality=100
                    # )

                imgs = torch.stack(imgs)
                assert len(imgs) == len(graphs)
                index += 1
                graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                                  graph_label=graph_labels, patch_id=patch_id, path=path, eid=eid,img=imgs)
                DeepSnap_Graph.append(graph)

    elif Flag == 2:
        DeepSnap_Graph = []
        index = 0
        # 示例：使用 process_graphs_concurrent 函数处理图列表
        # DeepSnap_Graph = parallel_processing(Graphs, labels, patch_path)
        from tqdm import  tqdm
        path = os.path.join('/data0/yuanyz/NewGraph', cfg.dataset.name, form)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/no.pickle', 'wb') as f:
            for graphs in tqdm(Graphs):
                temp_graphs = []
                for ind, graph_info in enumerate(graphs):
                    patch_id = torch.ones((graph_info.num_nodes(), 1)).mul(ind)
                    edge_id = torch.ones((graph_info.num_edges(), 1)).mul(ind)
                    graph_info.ndata['patch_id'] = patch_id
                    graph_info.edata['gid'] = edge_id
                    temp_graphs.append(graph_info)
                temp_graphs = dgl.batch(temp_graphs)

                nodes = temp_graphs.nodes().tolist()
                edges = temp_graphs.edges()
                # 创建一个新的networkx DiGraph对象
                G = nx.DiGraph()
                # 添加节点和边
                G.add_nodes_from(nodes)
                G.add_edges_from(list(zip(edges[0].tolist(), edges[1].tolist())))
                #src,dst = temp_graphs.edges()
                #G = nx.DiGraph(zip(src.tolist(), dst.tolist()))
                # G = nx.DiGraph(temp_graphs.to_networkx())
                # if form!='test':
                #     for i in range(int(1/cfg.model.p_value)+1):
                #         node_features = temp_graphs.ndata['feat']
                #         # 把每一行替换成全为0的tensor
                #         node_features = torch.stack(list(map(replace_tensor, node_features)))
                #         node_labels = temp_graphs.ndata['name']
                #         patch_id = temp_graphs.ndata['patch_id']
                #         graph_labels = labels[index]
                #         graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                #                   graph_label=graph_labels,patch_id=patch_id)
                #         DeepSnap_Graph.append(graph)
                #     index += 1
                # else:
                # print(temp_graphs.ndata['feat'].shape)
                # node_features3 = temp_graphs.ndata['feat'][:, :6]
                # node_features4 = temp_graphs.ndata['feat'][:, 18:24]
                node_features1 = temp_graphs.ndata['feat'][:, 40:41]
                node_features2 = temp_graphs.ndata['feat'][:, 48:]

                # node_features1 = temp_graphs.ndata['feat'][:, :24]
                # node_features2 = temp_graphs.ndata['feat'][:, 31:33]
                # node_features = temp_graphs.ndata['centroid']
                # node_features = temp_graphs.ndata['feat'][:,1:]
                # node_labels = temp_graphs.ndata['feat'][:, 0:1]
                # one_dimensional_features = torch.argmax(node_features1, dim=1).view(-1, 1)

                node_features = torch.cat((node_features1, node_features2), dim=1)

                # node_features = node_features1
                # node_features = node_features[:, 41:]
                # node_features = node_features[:, [2, 20]]
                # node_features = temp_graphs.ndata['feat'][:, temp_graphs.ndata['feat'].shape[1]-2:]
                # node_features = node_features1

                if pca is not None:
                    node_features = torch.from_numpy(pca.fit_transform(node_features))
                # node_features = node_features[:, cfg.dataset.feature_dim:cfg.dataset.feature_dim+1]
                # node_features = torch.cat((node_features, node_features2), dim=1)
                # node_features = temp_graphs.ndata['feat']
                # if form=='train':
                #    node_features = torch.stack(list(map(replace_tensor, node_features)))
                node_labels = temp_graphs.ndata['name']
                patch_id = temp_graphs.ndata['patch_id']
                eid = temp_graphs.edata['gid']
                graph_labels = labels[index]
                path = patch_path[index]
                index += 1
                graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                                  graph_label=graph_labels, patch_id=patch_id, path=path, eid=eid)
                # deepsnap.save_graphs("example_graph.pt", [graph])
                #pickle.dump(graph, f)
                DeepSnap_Graph.append(graph)
    elif Flag == 1:
        DeepSnap_Graph = []
        index = 0
        from tqdm import tqdm
        for graph_info in tqdm(Graphs):
            G = nx.Graph(graph_info.to_networkx())
            # nx.
            # node_features=torch.tensor(one_hot_encode(graph_info.ndata['name']-1))
            # node_features=node_features.squeeze(1)
            # node_features = graph_info.ndata['feat']
            # node_features = node_features[:, 2:node_features.shape[1] - 5]
            # node_features = node_features[:,0:2]
            # node_features = torch.cat((node_features,graph_info.ndata['name']),dim=1)
            node_features = graph_info.ndata['feat']
            node_labels = graph_info.ndata['feat'][:, 0:1]
            print(node_labels.max())
            graph_labels = labels[index]
            index += 1
            graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                              graph_label=graph_labels)
            # graph.save("mygraph.json")
            # graph = DeepSnapG(G, netlib=None, node_feature=node_features,graph_label=graph_labels)
            print(graph)
            DeepSnap_Graph.append(graph)
    else:
        DeepSnap_Graph = []
        from tqdm import tqdm
        for graph_info in tqdm(Graphs):
            # graph_info[0][0]=graph_info[0][0].subgraph(range(min(50,graph_info[0][0].number_of_nodes())))
            # 获取出度并筛选
            # out_degrees = graph_info[0][0].out_degrees().flatten()
            # top_512_nodes = out_degrees.topk(min(512,graph_info[0][0].number_of_nodes()), sorted=True)[1]
            # graph_info[0][0] = graph_info[0][0].subgraph(top_512_nodes)
            # # print(top_50_nodes)
            # # 选择子图
            # print(graph_info[0][0])
            # 计算图中每个节点的度数
            # degrees = graph_info[0][0].in_degrees().float().clamp(min=1)
            # # 对图进行归一化
            # graph_info[0][0].ndata['norm'] = torch.reciprocal(degrees)
            # graph_info[0][0].apply_nodes(lambda nodes: {'norm': degrees[nodes] * graph_info[0][0].ndata['norm'][nodes]})
            # print(graph_info)
            if graph_info[0][0].number_of_edges() < 1:
                global count
                count += 1
                print(count)
                continue
            G = nx.Graph(graph_info[0][0].to_networkx())
            node_features1 = graph_info[0][0].ndata['feat'][:, 0:23 - 2 - 7]
            node_features2 = graph_info[0][0].ndata['feat'][:, 21:23]
            node_features = torch.cat((node_features1, node_features2), dim=1)
            # node_features = graph_info[0][0].ndata['feat']
            # node_features = node_features[:,0:node_features.shape[1]-2]
            # print(node_features.shape)
            node_labels = graph_info[0][0].ndata['name']
            graph_labels = graph_info[1]['CoxLabel']
            # graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
            #                   graph_label=graph_labels)
            graph = DeepSnapG(G, netlib=None, node_feature=node_features, node_label=node_labels,
                              graph_label=graph_labels)
            DeepSnap_Graph.append(graph)
    return DeepSnap_Graph


def load_split_dataset(form):
    """

    :param name: train / val / test
    :return:
    """
    graphs, labels = load_dataset(form)
    return graphs, labels


def create_dataset(splits=None):
    # Load dataset
    if splits is None:
        splits = ['train', 'val', 'test']
    time1 = time.time()
    # splits=[None]
    cur_dataset_form = ['dgl', 'dglbatch', 'dglmulty', 'LoadImg', 'deepsnap']
    Flag = cur_dataset_form.index(cfg.dataset.format)
    datasets = []
    for form in splits:
        if Flag == 0:
            graphs = load_dataset(form)
            if graphs is not None:
                graphs = transform_to_DeepSnap(graphs, labels=None, Flag=Flag, form=form)
        elif Flag == 2 or Flag == 3:
            graphs, labels, patch_path = load_dataset(form)
            if graphs is not None:
                graphs = transform_to_DeepSnap(graphs, labels=labels, Flag=Flag, form=form, patch_path=patch_path)
        elif Flag == 4:
            graphs = load_dataset(form)
        print("加载{}完成".format(form))
        # Filter graphs
        time2 = time.time()
        min_node = filter_graphs()
        # Create whole dataset
        if graphs is not None:
            dataset = GraphDataset(
                graphs,
                task=cfg.dataset.task,
                edge_train_mode=cfg.dataset.edge_train_mode,
                edge_message_ratio=cfg.dataset.edge_message_ratio,
                edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
                resample_disjoint=cfg.dataset.resample_disjoint,
                minimum_node_per_graph=min_node
            )
            # Transform the whole dataset
            dataset = transform_before_split(dataset, form)
            # print(form)
            datasets.append(dataset)

    if len(datasets) == 2:
        temp_datasets = datasets[0].split(transductive=cfg.dataset.transductive,
                                          split_ratio=[0.75, 0.25],
                                          shuffle=cfg.dataset.shuffle_split)
        datasets[0] = temp_datasets[0]
        datasets.insert(1, temp_datasets[1])

    # for i in range(3):
    #     print(datasets[i])
    time3 = time.time()
    # We only change the training negative sampling ratio
    # for i in range(1, len(datasets)):
    #     datasets[i].edge_negative_sampling_ratio = 1

    print("加载数据集完成")
    # Transform each split dataset
    time4 = time.time()
    datasets = transform_after_split(datasets)
    set_dataset_info(datasets)
    print("Ego变换完成")
    time5 = time.time()
    logging.info('Load: {:.4}s, Before split: {:.4}s, '
                 'Split: {:.4}s, After split: {:.4}s'.format(
        time2 - time1, time3 - time2, time4 - time3,
        time5 - time4))

    return datasets


def create_loader(datasets, batch_size=cfg.train.batch_size):
    seed_anything(cfg.seed)
    if len(datasets) == 1:
        loader_test = DataLoader(datasets[0],
                                 collate_fn=Batch.collate(),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=16,
                                 pin_memory=False)
        return [loader_test]
    loader_train = DataLoader(datasets[0],
                              collate_fn=Batch.collate(),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(
            DataLoader(datasets[i],
                       collate_fn=Batch.collate(),
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=16,
                       pin_memory=False))
    return loaders


def load_dgl(form=None):
    '''
    load raw datasets.
    :return: a list of DGL graphs, plus additional info if needed
    '''
    # format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = ''
    if form is None:
        dataset_dir = cfg.dataset.dir + "/" + name
    else:
        dataset_dir = cfg.dataset.dir + "/" + name + "/" + form
    graphs = []
    if not os.path.exists(dataset_dir):
        return None
    patients = os.listdir(dataset_dir)  # CellGraph/patient
    for i in range(len(patients)):
        path = os.path.join(dataset_dir, patients[i])  # 所有patch所在的路径
        if os.path.isfile(path):
            continue
        patches = os.listdir(path)
        for patch in patches:
            graph = load_cg(os.path.join(path, patch))
            if graph is not None:
                graphs.append(graph)
    return graphs


def load_deepsnap(form=None):
    name = cfg.dataset.name
    if form is None:
        dataset_dir = '/data/yuanyz' + "/" + name
    else:
        dataset_dir = '/data/yuanyz' + "/" + name + "/" + form

    if not os.path.exists(dataset_dir):
        return None, None
    graphs = []
    print(dataset_dir)
    cnt = 0
    if form != 'val':
        with open(os.path.join(dataset_dir, 'graph_data.pickle'), 'rb') as f:
            while True:
                try:
                    graph = pickle.load(f)
                    graphs.append(graph)
                except EOFError:
                    break
                cnt += 1
                # if cnt>2:
                #     break
                print(f"loading the {cnt}.th patient")

    else:
        with open(os.path.join(dataset_dir, 'graph_data_val.pickle'), 'rb') as f:
            while True:
                try:
                    graph = pickle.load(f)
                    graphs.append(graph)
                except EOFError:
                    break
                cnt += 1
                # if cnt>2:
                #     break
                print(f"loading the {cnt}.th patient")

    return graphs

def contains_no_bin_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.bin'):
            return False
    return True

def load_dgl_Multy(form=None):
    '''
    load raw datasets.
    :return: a list of DGL graphs, plus additional info if needed
    '''
    # format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = ''
    if form is None:
        dataset_dir = cfg.dataset.dir + "/" + name
    else:
        dataset_dir = cfg.dataset.dir + "/" + name + "/" + form

    if not os.path.exists(dataset_dir):
        return None, None
    graphs = []
    labels = []
    all_paths = []
    print(dataset_dir)
    patients = os.listdir(dataset_dir)  # CellGraph/
    # patients=patients[2:4]
    print(len(patients))
    risk_csv = pd.read_csv('/data0/yuanyz/NewGraph/censor_with_normalized_risk_tcga.csv')
    # risk_csv = pd.read_csv('/data0/yuanyz/NewGraph/censor_with_normalized_risk_own.csv')

    cnt = 0
    for i in tqdm(range(len(patients))):
        # if i>1:
        #     break
        path = os.path.join(dataset_dir, patients[i])  # 所有patch所在的路径

        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            temp_graph = []
            label = None
            patch_name = os.listdir(path)
            paths = []
            for item in patch_name:
                patch_path = os.path.join(path, item)
                if not os.path.isdir(patch_path):
                    print(patch_path)
                    continue
                if contains_no_bin_files(patch_path):
                    for filename in os.listdir(patch_path):
                        os.remove(os.path.join(patch_path, filename))
                    os.rmdir(patch_path)
                    print(patch_path, " has been removed")
                    continue
                graph = load_cg(patch_path)
                if graph is not None:
                    a = torch.tensor(risk_csv[risk_csv['标本号'] == patients[i]]['风险系数'].values,
                                     dtype=torch.float32).unsqueeze(0)
                    graph[1]['CoxLabel'] = graph[1]['CoxLabel'].masked_fill(torch.isnan(graph[1]['CoxLabel']), 0)
                    label = torch.cat((graph[1]['CoxLabel'], a), dim=1)
                    temp_graph.append(graph[0][0])
                    paths.append(patch_path)
            # temp_graph=temp_graph[:32]
            bias = 0
            n_split = min(cfg.dataset.augment_split, len(temp_graph))
            if form == 'train':
                for i in range(n_split):
                    graphs.append(temp_graph[bias:bias + int(len(temp_graph) / n_split)])
                    all_paths.append(paths[bias:bias + int(len(temp_graph) / n_split)])
                    bias += int(len(temp_graph) / n_split)
                    labels.append(label)
                if bias < len(temp_graph):
                    graphs.append(temp_graph[bias:])
                    all_paths.append(paths[bias:])
                    labels.append(label)
            else:
                graphs.append(temp_graph)
                labels.append(label)
                all_paths.append(paths)
    return graphs, labels, all_paths


def load_Img(form=None):
    '''
    load raw datasets.
    :return: a list of DGL graphs, plus additional info if needed
    '''
    # format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = ''
    if form is None:
        dataset_dir = cfg.dataset.dir + "/" + name
    else:
        dataset_dir = cfg.dataset.dir + "/" + name + "/" + form

    if not os.path.exists(dataset_dir):
        return None, None
    graphs = []
    labels = []
    imgs = []
    print(dataset_dir)
    patients = os.listdir(dataset_dir)  # CellGraph/
    print(len(patients))
    cnt = 0
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for i in range(len(patients)):
        path = os.path.join(dataset_dir, patients[i])  # 所有patch所在的路径

        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            temp_graph = []
            label = None
            patch_name = os.listdir(path)
            for item in patch_name:
                print(item)
                patch_path = os.path.join(path, item)
                input_image = Image.open(patch_path + "/wsi.png")
                input_image = np.array(input_image)
                input_image = Image.fromarray(input_image)
                input_image = PreprocessImg(input_image)

                graph = load_cg(patch_path)
                if graph is not None:
                    label = graph[1]['CoxLabel']
                    temp_graph.append(graph[0][0])
                    imgs.append(input_image)
            # temp_graph=temp_graph[:32]
            bias = 0
            n_split = min(cfg.dataset.augment_split, len(temp_graph))
            if form == 'train':
                for i in range(n_split):
                    graphs.append(temp_graph[bias:bias + int(len(temp_graph) / n_split)])
                    bias += int(len(temp_graph) / n_split)
                    labels.append(label)
                if bias < len(temp_graph):
                    graphs.append(temp_graph[bias:])
                    labels.append(label)
            else:
                graphs.append(temp_graph)
                labels.append(label)
    return graphs, labels, imgs
