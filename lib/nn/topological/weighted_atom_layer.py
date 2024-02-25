import torch

from lib.nn.sources.source import Neurons

from .linear import Linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(self, neurons: Neurons, optimize_linear_gathers=True) -> None:
        super().__init__()
        self.linear = Linear(neurons, period=None, optimize_gathers=optimize_linear_gathers)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.tanh(y)
        return y
