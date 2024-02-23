import torch

from lib.nn.aggregation import AggregationType, ReshapeAndAggregate, build_optimal_reshape_aggregate
from lib.nn.gather import build_optimal_multi_layer_gather, build_optimal_multi_layer_gather_and_reshape
from lib.nn.topological.layers import Ordinals


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        aggregation_type: AggregationType = "mean",
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        inputs_dims = torch.tensor([len(n.getInputs()) for n in layer_neurons], dtype=torch.int32)

        reshape_agg = build_optimal_reshape_aggregate(aggregation_type, counts=inputs_dims)

        input_layer_ordinal_pairs = [
            neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()
        ]

        if reshape_agg.is_matching_dimension:
            self.gather = build_optimal_multi_layer_gather_and_reshape(
                input_layer_ordinal_pairs, dim=reshape_agg.get_reshape().period
            )
            self.aggregate = reshape_agg.get_aggregate()
        else:
            self.gather = build_optimal_multi_layer_gather(input_layer_ordinal_pairs)
            self.aggregate = reshape_agg

    def forward(self, layer_values: dict[int, torch.Tensor]):
        x = self.gather(layer_values)
        x = self.aggregate(x)
        return x
