import torch
from lib.nn.gather import build_optimal_gather_module
from lib.nn.linear import Linear
from lib.nn.topological.layers import Ordinals
from tqdm.auto import tqdm


class WeightedRuleLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        assume_rule_weights_same=True,
        check_same_inputs_dim_assumption=True,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        neuron = layer_neurons[0]

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
        )

        self.linear = Linear(layer_neurons, assume_all_weights_same=assume_rule_weights_same)

        self.inputs_dim = neuron.getInputs().size()

        if check_same_inputs_dim_assumption:
            for n in layer_neurons:
                assert self.inputs_dim == n.getInputs().size()

        self.assume_rule_weights_same = assume_rule_weights_same

        if assume_rule_weights_same:  # check
            layer_index = neuron.getLayer()

            for n in tqdm(layer_neurons[1:], desc="Verifying neurons"):
                assert n.getLayer() == layer_index

                assert len(self.linear.weight_indices) == len(n.getWeights())
                for our_widx, weight in zip(self.linear.weight_indices, n.getWeights()):
                    assert our_widx == weight.index

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)

        if self.assume_rule_weights_same:
            input_values = torch.reshape(input_values, [-1, self.inputs_dim, *input_values.shape[1:]])
            y = self.linear(input_values)
        else:
            y = self.linear(input_values)
            y = torch.reshape(y, [-1, self.inputs_dim, *y.shape[1:]])

        # TODO: parameterize
        y = torch.sum(y, 1)
        y = torch.tanh(y)
        return y
