import torch

from lib.nn.sources.source import Neurons
from lib.nn.topological.settings import Settings

from .linear import Linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.linear = Linear(
            neurons,
            period=None,
            settings=settings,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.tanh(y)
        return y
