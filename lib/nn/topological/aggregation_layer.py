from typing import Literal

import torch
from lib.nn.gather import build_optimal_gather_module
from lib.nn.topological.layers import Ordinals
from lib.utils import atleast_3d_rev

AggregationType = Literal['mean', 'sum']


def get_aggregation_func(agg: AggregationType):
    if agg == 'mean':
        return torch.Tensor.mean
    elif agg == 'sum':
        return torch.Tensor.sum
    else:
        raise ValueError()


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        aggregation: AggregationType = 'mean',
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
        )

        self.inputs_dims = [len(n.getInputs()) for n in layer_neurons]
        self.inputs_dims_tensor = torch.nn.Parameter(
            atleast_3d_rev(torch.tensor(self.inputs_dims, dtype=torch.int)), requires_grad=False
        )
        self.inputs_dims_match = torch.all(self.inputs_dims_tensor == self.inputs_dims[0])
        self.aggregation = aggregation
        self.aggregation_func = get_aggregation_func(aggregation)

        if aggregation == 'sum':
            def padded_aggregation_func(tensor: torch.Tensor):
                return torch.sum(tensor, dim=1)
        elif aggregation == 'mean':
            def padded_aggregation_func(tensor: torch.Tensor):
                return torch.sum(tensor, dim=1) / self.inputs_dims_tensor

        self.padded_aggregation_func = padded_aggregation_func

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)

        if self.inputs_dims_match:
            input_values = torch.reshape(input_values, [-1, self.inputs_dims[0], *input_values.shape[1:]])
            y = self.aggregation_func(input_values, dim=1)
        else:
            input_values = list(torch.split(input_values, self.inputs_dims, dim=0))
            input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

            y = self.padded_aggregation_func(input_values)

        return y

    def extra_repr(self) -> str:
        return f"aggregation={self.aggregation}, inputs_dims_match={self.inputs_dims_match},"
