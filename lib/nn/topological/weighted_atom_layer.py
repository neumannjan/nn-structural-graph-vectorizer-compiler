import torch

from lib.nn.sources.source import Neurons
from lib.nn.topological.settings import Settings

from .linear import UniqueLinearAndCollect


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()
        # TODO: write a heuristic to choose optimal linear layer implementation
        self.linear = UniqueLinearAndCollect(
            neurons,
            period=None,
            settings=settings,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('WEIGHTED_ATOM_LINEAR'):
            y = self.linear(layer_values)
        with torch.profiler.record_function('WEIGHTED_TANH'):
                y = torch.tanh(y)
        return y
