import torch.nn as nn

from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the ebatch.node_featureperiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, batch):
        # print(batch.node_feature)
        if self.mode == 'None':
            return batch

        col_mean = batch.node_feature.mean(dim=0)
        if self.mode == 'PN':
            batch.node_feature = batch.node_feature - col_mean
            rownorm_mean = (1e-6 + batch.node_feature.pow(2).sum(dim=1).mean()).sqrt()
            batch.node_feature = self.scale * batch.node_feature / rownorm_mean

        if self.mode == 'PN-SI':
            batch.node_feature = batch.node_feature - col_mean
            rownorm_individual = (1e-6 + batch.node_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            batch.node_feature = self.scale * batch.node_feature / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + batch.node_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            batch.node_feature = self.scale * batch.node_feature / rownorm_individual - col_mean

        return batch
