import torch

from .weighted_layer import WeightedLayer


class WeightedAtomLayer(WeightedLayer):
    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = super().forward(layer_values)
        y = torch.tanh(y)
        return y
