import torch

from lib.nn.sources.source import Neurons
from lib.nn.topological.linear import Linear
from lib.nn.topological.settings import Settings
from lib.utils import head_and_rest


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
    ) -> None:
        super().__init__()

        head_len, rest_lengths = head_and_rest(neurons.input_lengths)

        for l in rest_lengths:
            assert head_len == l

        self.linear = Linear(
            neurons,
            period=head_len,
            settings=settings,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.sum(y, 1)
        y = torch.tanh(y)
        return y
