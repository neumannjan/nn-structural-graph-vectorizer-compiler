import torch

from lib.nn.sources.source import Neurons
from lib.nn.topological.linear import Linear
from lib.utils import head_and_rest


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        check_same_inputs_dim_assumption=True,
        optimize_linear_gathers=True,
    ) -> None:
        super().__init__()

        head_len, rest_lengths = head_and_rest(neurons.input_lengths)

        if check_same_inputs_dim_assumption:
            for l in rest_lengths:
                assert head_len == l

        self.linear = Linear(
            neurons, period=head_len, optimize_gathers=optimize_linear_gathers
        )

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = torch.sum(y, 1)
        y = torch.tanh(y)
        return y
