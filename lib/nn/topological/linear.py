import warnings

import torch

from lib.nn.gather import (
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
)
from lib.nn.sources.source import Neurons
from lib.nn.topological.settings import Settings
from lib.nn.weight import create_weights_and_gather


class Linear(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
        period: int | None = None,
    ) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        self.period = period

        inputs_ordinals = list(neurons.inputs.ordinals)

        if period is None:
            gather = build_optimal_multi_layer_gather(inputs_ordinals)
        else:
            gather = build_optimal_multi_layer_gather_and_reshape(inputs_ordinals, period=period)
        self.gather = gather

        weight_definitions = list(neurons.input_weights)
        self.weight, self.gather_weights = create_weights_and_gather(
            weight_definitions,
            period=period,
            group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
        )

        if period is not None and settings.optimize_linear_gathers:
            # can we simplify?
            can_simplify_inputs = self.gather.optimal_period == period and not self.gather.is_optimal
            can_simplify_weights = self.gather_weights.optimal_period == period and not self.gather_weights.is_optimal

            if can_simplify_inputs and can_simplify_weights:
                # TODO
                warnings.warn(
                    "Can simplify both weights and inputs due to matching dimensionality, "
                    "but it is not implemented yet, so the run will be slow."
                )
            elif can_simplify_inputs:
                self.gather = gather.get_optimal()
            elif can_simplify_weights:
                self.gather_weights = self.gather_weights.get_optimal()

        # TODO invert gather+linear to gather_set+linear+gather if beneficial

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        w = self.gather_weights(w)
        y = w @ input_values
        return y

    def extra_repr(self) -> str:
        return f"period={self.period},"
