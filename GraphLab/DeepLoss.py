from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from GraphLab.config import cfg
from GraphLab.model.layer.CoxLossLayer import DeepSurv
from GraphLab.model.layer.CoxLossLayer import NegativeLogLikelihood
from GraphLab.utils.utils import read_config

configs = read_config("../Run/configs/Loss/loss.ini")
# builds network|criterion|optimizer based on configuration
criterion = None


def create_Loss_model(dim_in):
    configs['network']['dims'][0] = dim_in
    model = DeepSurv(configs['network']).to(torch.device(cfg.device))
    global criterion
    criterion = NegativeLogLikelihood(configs['network']).to(torch.device(cfg.device))
    return model


def compute_DSLoss(risk_pred, y, e, model):
    loss = criterion(risk_pred, y, e, model)
    return loss
