from typing import Mapping

import torch

from lib.nn.sources.base import Network, Neurons
from lib.nn.topological.linear import build_optimal_linear
from lib.nn.topological.settings import Settings
from lib.nn.transformation import build_transformation
from lib.utils import head_and_rest


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        network: Network,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()

        head_len, rest_lengths = head_and_rest(neurons.input_lengths)

        # TODO: assumption: all neurons have the same no. of inputs (and weights)
        for l in rest_lengths:
            assert head_len == l

        transformation, tr_rest = head_and_rest(neurons.get_transformations())

        # TODO assumption: all neurons have the same transformation
        for t in tr_rest:
            assert t == transformation

        self.linear = build_optimal_linear(
            network,
            neurons,
            period=head_len,
            settings=settings,
            gather_unique_first=False,
        )

        self.transformation = build_transformation(transformation)

    def forward(self, layer_values: Mapping[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.sum(y, 1)
        y = self.transformation(y)
        return y
