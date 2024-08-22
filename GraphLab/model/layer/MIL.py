import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)

class DifferentiableKMeans(nn.Module):
    def __init__(self, n_clusters, input_dim, device):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, input_dim, device=device))

    def forward(self, x):
        pairwise_distances = torch.cdist(x, self.centroids)
        gumbel_noise = torch.distributions.gumbel.Gumbel(0, 1).sample(pairwise_distances.shape).to(x.device)
        softmax_inputs = -pairwise_distances + gumbel_noise
        cluster_assignments = softmax_inputs.softmax(dim=1)
        return cluster_assignments


class ClusterThenMlp(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_clusters=3, layer_num=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.clustering = DifferentiableKMeans(n_clusters, input_dim, torch.device(cfg.device))
        layers = [nn.Linear(input_dim, hidden_dim, device=torch.device(cfg.device))]
        for i in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, device=torch.device(cfg.device)))
        layers.append(nn.Linear(hidden_dim, n_clusters, device=torch.device(cfg.device)))
        self.mlp = nn.Sequential(*layers)
        self.transform = nn.Linear(n_clusters, input_dim, device=torch.device(cfg.device))

    def forward(self, x):
        cluster_assignments = self.clustering(x)
        output = self.mlp(x)
        return output, cluster_assignments


def MIL(x, input_dim, hidden_dim, n_clusters, layer_num, epochs=10):
    pre_train = '/data/yuanyz/HccGraph/Run/configs/mil_model.pth'
    model = ClusterThenMlp(input_dim=input_dim, hidden_dim=hidden_dim, n_clusters=n_clusters, layer_num=layer_num)
    if os.path.exists(pre_train):
        model.load_state_dict(torch.load(pre_train))
        # print("加载mil的预训练参数以便进一步训练")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if x.requires_grad:
        for epoch in range(epochs):
            # Create a new x for each loop
            x_new = x.detach().clone().requires_grad_()
            output, cluster_assignments = model(x_new)
            labels = torch.argmax(cluster_assignments, dim=1)

            loss = nn.CrossEntropyLoss()(output, labels)

            acc = accuracy_score(torch.max(output, dim=1)[1].detach().cpu().numpy(), labels.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update x using model.transform and the new output
            x = model.transform(output.detach())
            # if (epochs + 1) % 50 == 0:
            #     print(f'Epoch: {epoch + 1}  Accuracy: {acc}  Loss: {loss}')
    else:
        model.eval()
        output, cluster_assignments = model(x)
        labels = torch.argmax(cluster_assignments, dim=1)
        loss = nn.CrossEntropyLoss()(output, labels)
        acc = accuracy_score(torch.max(output, dim=1)[1].detach().cpu().numpy(), labels.detach().cpu().numpy())
        x = model.transform(output)
        # print(f'Testing ...    Accuracy: {acc}  Loss: {loss}')

    # 保存模型
    torch.save(model.state_dict(), '/data/yuanyz/HccGraph/Run/configs/mil_model.pth')

    return x
