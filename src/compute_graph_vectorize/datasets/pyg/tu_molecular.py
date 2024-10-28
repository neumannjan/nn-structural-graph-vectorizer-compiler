from typing import Literal, Protocol

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool


class _ModuleFactoryProtocol(Protocol):
    def __call__(self, activation: str, output_size: int, num_features: int, dim: int = 10) -> torch.nn.Module:
        ...


class NetGCN(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features: int, dim: int = 10):
        super(NetGCN, self).__init__()

        self.conv1 = GCNConv(num_features, dim, normalize=False, cached=False, bias=False)
        self.conv2 = GCNConv(dim, dim, normalize=False, cached=False, bias=False)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

        self.fc1 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        return self.activation(x)


class NetGraphSage(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features: int, dim: int = 10):
        super(NetGraphSage, self).__init__()
        self.conv1 = SAGEConv(num_features, dim, normalize=False, bias=False)
        self.conv2 = SAGEConv(dim, dim, normalize=False, bias=False)

        self.fc1 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)

        return self.activation(x)


class NetGIN(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features: int, dim: int = 10):
        super(NetGIN, self).__init__()

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)

        self.l1 = Linear(dim, output_size, bias=False)
        self.l2 = Linear(dim, output_size, bias=False)
        self.l3 = Linear(dim, output_size, bias=False)
        self.l4 = Linear(dim, output_size, bias=False)
        self.l5 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        x4 = F.relu(self.conv4(x3, edge_index))
        x5 = F.relu(self.conv5(x4, edge_index))

        m1 = global_mean_pool(x1, batch)
        m2 = global_mean_pool(x2, batch)
        m3 = global_mean_pool(x3, batch)
        m4 = global_mean_pool(x4, batch)
        m5 = global_mean_pool(x5, batch)

        stacked = torch.stack([self.l1(m1), self.l2(m2), self.l3(m3), self.l4(m4), self.l5(m5)], dim=0)
        x = torch.sum(stacked, dim=0)

        return self.activation(x)


TUDatasetTemplate = Literal["gcn", "gin", "gsage"]

_TEMPLATE_MAP: dict[TUDatasetTemplate, _ModuleFactoryProtocol] = {
    "gcn": NetGCN,
    "gin": NetGIN,
    "gsage": NetGraphSage,
}


def build_pyg_module(
    template: TUDatasetTemplate, activation: str, output_size: int, num_features: int, dim: int = 10
) -> torch.nn.Module:
    return _TEMPLATE_MAP[template](activation=activation, output_size=output_size, num_features=num_features, dim=dim)
