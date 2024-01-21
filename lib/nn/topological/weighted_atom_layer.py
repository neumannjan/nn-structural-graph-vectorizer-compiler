import torch
from lib.nn.gather import build_optimal_gather_module
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

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
        )

        self.linear = Linear(layer_neurons, assume_all_weights_same=False)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('WEIGHTED_ATOM_GATHER'):
            input_values = self.gather(layer_values)
        # TODO reshape ?
        with torch.profiler.record_function('WEIGHTED_ATOM_LINEAR'):
            y = self.linear(input_values)
        with torch.profiler.record_function('WEIGHTED_ATOM_TANH'):
            y = torch.tanh(y)
        return y
