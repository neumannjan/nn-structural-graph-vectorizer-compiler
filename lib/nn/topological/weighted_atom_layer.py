import torch

from lib.nn.sources.source import NeuralNetworkDefinition, Neurons
from lib.nn.topological.settings import Settings

from .linear import build_optimal_linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self,
        network: NeuralNetworkDefinition,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.linear = build_optimal_linear(
            network,
            neurons,
            period=None,
            settings=settings,
            gather_unique_first=True,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.tanh(y)
        return y
