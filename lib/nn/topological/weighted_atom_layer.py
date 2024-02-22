import torch

from lib.nn.topological.layers import Ordinals

from .linear import Linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(self, layer_neurons: list, neuron_ordinals: Ordinals, optimize_linear_gathers=True) -> None:
        super().__init__()
        self.linear = Linear(layer_neurons, neuron_ordinals, period=None, optimize_gathers=optimize_linear_gathers)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.tanh(y)
        return y
