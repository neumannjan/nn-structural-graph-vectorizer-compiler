import torch

from lib.nn.topological.layers import Ordinals
from lib.nn.topological.linear import Linear


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        check_same_inputs_dim_assumption=True,
    ) -> None:
        super().__init__()

        neuron = layer_neurons[0]

        inputs_dim = len(neuron.getInputs())

        if check_same_inputs_dim_assumption:
            for n in layer_neurons:
                assert inputs_dim == len(n.getInputs())

        self.linear = Linear(layer_neurons, neuron_ordinals, period=inputs_dim)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('WEIGHTED_RULE_LINEAR'):
            y = self.linear(layer_values)
        with torch.profiler.record_function('WEIGHTED_RULE_SUM'):
            y = torch.sum(y, 1)
        with torch.profiler.record_function('WEIGHTED_TANH'):
            y = torch.tanh(y)
        return y
