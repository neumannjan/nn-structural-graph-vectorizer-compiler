from typing import Mapping

import torch

from lib.nn.sources.source import NeuralNetworkDefinition, Neurons
from lib.nn.topological.settings import Settings
from lib.nn.transformation import build_transformation
from lib.utils import head_and_rest

from .linear import build_optimal_linear


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self,
        network: NeuralNetworkDefinition,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()

        transformation, tr_rest = head_and_rest(neurons.get_transformations())

        # TODO assumption: all neurons have the same transformation
        for t in tr_rest:
            assert t == transformation

        self.linear = build_optimal_linear(
            network,
            neurons,
            period=None,
            settings=settings,
            gather_unique_first=True,
        )

        self.transformation = build_transformation(transformation)

    def forward(self, layer_values: Mapping[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = self.transformation(y)
        return y
