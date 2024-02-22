import torch

from lib.nn.aggregation import AggregationType, ReshapeAndAggregate, build_optimal_aggregation
from lib.nn.gather import build_optimal_gather_module
from lib.nn.topological.layers import Ordinals


class AggregationLayer(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        aggregation: AggregationType = "mean",
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        inputs_dims = torch.tensor([len(n.getInputs()) for n in layer_neurons], dtype=torch.int32)

        aggregation_module = build_optimal_aggregation(aggregation, counts=inputs_dims)

        if isinstance(aggregation_module, ReshapeAndAggregate):
            period = aggregation_module.period
            aggregation_module = aggregation_module.aggregation
        else:
            period = None

        self.gather = build_optimal_gather_module(
            [neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()],
            period=period,
        )

        self.aggregation = aggregation_module

    def forward(self, layer_values: dict[int, torch.Tensor]):
        x = self.gather(layer_values)
        x = self.aggregation(x)
        return x
