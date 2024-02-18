import torch
from lib.nn.topological.layers import Ordinals

from .weighted_layer import WeightedLayer


class WeightedRuleLayer(WeightedLayer):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        check_same_inputs_dim_assumption=True,
    ) -> None:
        super().__init__(layer_neurons, neuron_ordinals)

        neuron = layer_neurons[0]

        self.inputs_dim = len(neuron.getInputs())

        if check_same_inputs_dim_assumption:
            for n in layer_neurons:
                assert self.inputs_dim == len(n.getInputs())

    def forward(self, layer_values: dict[int, torch.Tensor]):
        y = super().forward(layer_values)
        y = torch.reshape(y, [-1, self.inputs_dim, *y.shape[1:]])
        y = torch.sum(y, 1)
        y = torch.tanh(y)
        return y
