import torch
from lib.nn.gather import build_optimal_gather_module
from lib.nn.topological.layers import Ordinals
from lib.utils import atleast_3d_rev


class AggregationLayer(torch.nn.Module):
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

        self.inputs_dims = [n.getInputs().size() for n in layer_neurons]
        self.inputs_dims_match = all((self.inputs_dims[0] == d for d in self.inputs_dims[1:]))

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)

        if self.inputs_dims_match:
            input_values = torch.reshape(input_values, [-1, self.inputs_dims[0], *input_values.shape[1:]])
            # TODO: parameterize
            y = input_values.mean(dim=1)
        else:
            input_values = list(torch.split(input_values, self.inputs_dims, dim=0))
            input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

            # TODO: parameterize
            # (mean)
            y = torch.sum(input_values, dim=1) / atleast_3d_rev(torch.tensor(self.inputs_dims))

        return y

    @property
    def padding_needed(self) -> bool:
        return not self.inputs_dims_match

    def extra_repr(self) -> str:
        return f"padding_needed={self.padding_needed},"
