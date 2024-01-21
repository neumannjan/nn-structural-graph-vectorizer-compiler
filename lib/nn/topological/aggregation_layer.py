from typing import Literal

import torch
from lib.nn.gather import build_optimal_gather_module
from lib.nn.topological.layers import Ordinals
from torch_scatter import scatter_add, scatter_mean

AggregationType = Literal["mean", "sum"]


def get_aggregation_func(agg: AggregationType):
    if agg == "mean":
        return torch.Tensor.mean
    elif agg == "sum":
        return torch.Tensor.sum
    else:
        raise ValueError()


def get_scatter_func(agg: AggregationType):
    if agg == "mean":
        return scatter_mean
    elif agg == "sum":
        return scatter_add
    else:
        raise ValueError()


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        aggregation: AggregationType = "mean",
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
        )

        self.inputs_dims = torch.nn.Parameter(
            torch.tensor([len(n.getInputs()) for n in layer_neurons], dtype=torch.int32), requires_grad=False
        )

        self.inputs_dims_match = torch.all(self.inputs_dims == self.inputs_dims[0])
        self.aggregation = aggregation

        if self.inputs_dims_match:
            self.aggregation_func = get_aggregation_func(aggregation)
        else:
            self.index = torch.nn.Parameter(
                torch.repeat_interleave(torch.arange(0, self.inputs_dims.shape[0]), repeats=self.inputs_dims),
                requires_grad=False,
            )
            self.scatter_func = get_scatter_func(aggregation)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('AGG_GATHER'):
            input_values = self.gather(layer_values)

        if self.inputs_dims_match:
            with torch.profiler.record_function('AGG_RESHAPE'):
                input_values = torch.reshape(input_values, [-1, self.inputs_dims[0], *input_values.shape[1:]])
            with torch.profiler.record_function('AGG_AGGREGATION'):
                y = self.aggregation_func(input_values, dim=1)
        else:
            with torch.profiler.record_function('AGG_SCATTER_REDUCE'):
                y = self.scatter_func(input_values, self.index, dim=0)

        return y

    def extra_repr(self) -> str:
        return f"aggregation={self.aggregation}, inputs_dims_match={self.inputs_dims_match},"
