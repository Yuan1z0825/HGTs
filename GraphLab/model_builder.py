import torch

import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.model.gnn.gnn import GNN
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
network_dict = {
    'gnn': GNN,
}
network_dict = {**register.network_dict, **network_dict}


def create_model(to_device=True, dim_in=None, dim_out=None):
    r"""
    Create model for graph machine learning

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1
    elif 'regression' in cfg.dataset.task_type and dim_out == 2 and 'cox' in cfg.model.loss_fun:
        dim_out = 1
    elif 'classification_multi' in cfg.dataset.task_type and dim_out == 1:
        dim_out = 7
    elif 'CensoredCrossEntropyLoss' == cfg.model.loss_fun:
        dim_out = 5
    elif 'multi_task' == cfg.model.loss_fun:
        dim_out = 7
    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
        print(cfg.device)
    return model
