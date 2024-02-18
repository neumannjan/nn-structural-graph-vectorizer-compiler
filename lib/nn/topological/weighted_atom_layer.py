import torch

from lib.nn.topological.layers import Ordinals

from .linear import Linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(self, layer_neurons: list, neuron_ordinals: Ordinals) -> None:
        super().__init__()
        self.linear = Linear(layer_neurons, neuron_ordinals, period=None)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('WEIGHTED_ATOM_LINEAR'):
            y = self.linear(layer_values)
        with torch.profiler.record_function('WEIGHTED_TANH'):
                y = torch.tanh(y)
        return y
