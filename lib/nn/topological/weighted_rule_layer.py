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

        # TODO: assumption: all neurons have the same no. of inputs (and weights)
        for l in rest_lengths:
            assert head_len == l

        # TODO: write a heuristic to choose optimal linear layer implementation
        self.linear = Linear(
            neurons,
            period=head_len,
            settings=settings,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('WEIGHTED_RULE_LINEAR'):
            y = self.linear(layer_values)
        with torch.profiler.record_function('WEIGHTED_RULE_SUM'):
            y = torch.sum(y, 1)
        with torch.profiler.record_function('WEIGHTED_TANH'):
            y = torch.tanh(y)
        return y
