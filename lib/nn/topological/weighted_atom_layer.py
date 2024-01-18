import torch
from lib.nn.gather import GatherModule
from lib.nn.linear import Linear
from lib.nn.topological.layers import Ordinals


class WeightedAtomLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.gather = GatherModule(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
            allow_merge_on_all_inputs_same=True,
        )

        self.linear = Linear(layer_neurons, assume_all_weights_same=False)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        # TODO reshape ?
        y = self.linear(input_values)
        y = torch.tanh(y)
        return y
