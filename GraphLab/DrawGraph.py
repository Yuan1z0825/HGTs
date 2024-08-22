import pickle
import os
import pickle
import time

import dgl
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph as DeepSnapG
from dgl import load_graphs
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import DataLoader
from torch_geometric.datasets import (PPI, Amazon, Coauthor, KarateClub,
                                      MNISTSuperpixels, Planetoid, QM7b,
                                      TUDataset)
from tqdm import tqdm

import GraphLab.model.feature_process.feature_augment as preprocess
import GraphLab.register as register
from GraphLab.cmd_args import parse_args
from GraphLab.config import cfg, load_cfg, set_out_dir, dump_cfg
from GraphLab.model.transform.transform import (edge_nets, ego_nets)


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


def collate_fn(batch, type):
    return COLLATE_FN[type]([example for example in batch])


def load_cg(cg_path):
    """
    Load cell graphs
    """
    # cg_fnames = []
    # cg_fnames = cg_fnames + glob(os.path.join(cg_path, '*.bin'))
    # cg_fnames.sort()
    # num_cg = len(cg_fnames)
    # if num_cg==0:
    #     return None
    # Cell_Graph = load_graphs(cg_path)
    # paths=(os.path.join(cg_path,"AllCell.bin"))
    # print(paths)
    Cell_Graph = load_graphs((os.path.join(cg_path, "AllCell.bin")))

    # Cell_Graphs=[load_graphs(fname) for fname in cg_fnames]
    # cell_graphs = [entry[0][0] for entry in Cell_Graphs]
    # cell_graphs = [collate_fn(cell_graphs, type(cell_graphs[0]).__name__)]
    # cell_graph_labels =[entry[1]['CoxLabel'] for entry in Cell_Graphs]
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
    elif format == 'dglbatch':
        graphs, labels = load_dgl_new(form)
        return graphs, labels
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


def transform_to_DeepSnap(Graphs, labels=None, Flag=1):
    '''

    :param Graphs: 表示带有标签的DGL图
    :Flag: 2代表dglmulty  1代表 dglbatch   0代表dgl不需要labels
    :return: DeepSnap Graph
    '''
    if Flag == 1:
        DeepSnap_Graph = []
        index = 0
        for graph_info in tqdm(Graphs):
            G = nx.draw(graph_info.to_networkx())
            # nx.Draw(G)
            # plt.show()
            print(G)
            node_features = graph_info.ndata['name']
            node_labels = graph_info.ndata['name']
            graph_labels = labels[index]
            index += 1
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


def create_dataset():
    # Load dataset
    time1 = time.time()
    splits = ['train', 'val', 'test']
    Flag = 1
    datasets = []
    for form in splits:
        if Flag == 0:
            graphs = load_dataset(form)
            if graphs is not None:
                graphs = transform_to_DeepSnap(graphs, labels=None, Flag=Flag)
        else:
            graphs, labels = load_dataset(form)
            if graphs is not None:
                graphs = transform_to_DeepSnap(graphs, labels=labels, Flag=Flag)
        print("加载{}完成".format(form))
    return datasets


def create_loader(datasets):
    loader_train = DataLoader(datasets[0],
                              collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(
            DataLoader(datasets[i],
                       collate_fn=Batch.collate(),
                       batch_size=cfg.train.batch_size,
                       shuffle=False,
                       num_workers=cfg.num_workers,
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
        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            patch_name = os.listdir(path)
            for item in patch_name:
                # print(item)
                patch_path = os.path.join(path, item)
                graph = load_cg(patch_path)
                if graph is not None:
                    graphs.append(graph)
    return graphs


def load_dgl_new(form=None):
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
    labels = []
    if not os.path.exists(dataset_dir):
        return None, None
    patients = os.listdir(dataset_dir)  # CellGraph/patient
    for i in range(len(patients)):
        path = os.path.join(dataset_dir, patients[i])  # 所有patch所在的路径
        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            temp_graph = []
            label = None
            patch_name = os.listdir(path)
            for item in patch_name:
                patch_path = os.path.join(path, item)
                graph = load_cg(patch_path)
                if graph is not None:
                    label = graph[1]['CoxLabel']
                    temp_graph.append(graph[0][0])
            batchgraph = dgl.batch(temp_graph)
            graphs.append(batchgraph)
            labels.append(label)
    return graphs, labels


# Load cmd line args
args = parse_args()
# Load configs file
load_cfg(cfg, args)
set_out_dir(cfg.out_dir, args.cfg_file)
# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)
dump_cfg(cfg)
create_dataset()
