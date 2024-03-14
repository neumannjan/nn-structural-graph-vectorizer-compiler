import torch

from lib.nn.aggregation import AggregationType, build_optimal_reshape_aggregate
from lib.nn.gather import build_optimal_multi_layer_gather, build_optimal_multi_layer_gather_and_reshape
from lib.nn.sources.source import Neurons
from lib.nn.topological.settings import Settings


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
        aggregation_type: AggregationType = "mean",
    ) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        inputs_dims = torch.tensor(list(neurons.input_lengths), dtype=torch.int32)

        reshape_agg = build_optimal_reshape_aggregate(
            aggregation_type, counts=inputs_dims, allow_non_builtin_torch_ops=settings.allow_non_builtin_torch_ops
        )

        inputs_ordinals = list(neurons.inputs.ordinals)

        if reshape_agg.is_matching_dimension:
            self.gather = build_optimal_multi_layer_gather_and_reshape(
                inputs_ordinals, period=reshape_agg.get_reshape().period
            )
            self.aggregate = reshape_agg.get_aggregate()
        else:
            self.gather = build_optimal_multi_layer_gather(inputs_ordinals)
            self.aggregate = reshape_agg

    def forward(self, layer_values: dict[int, torch.Tensor]):
        x = self.gather(layer_values)
        x = self.aggregate(x)
        return x
