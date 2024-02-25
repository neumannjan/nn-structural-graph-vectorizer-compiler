import warnings

import torch

from lib.nn.gather import (
    build_optimal_gather,
    build_optimal_gather_and_reshape,
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
)
from lib.nn.sources.source import Neurons
from lib.nn.weight import create_weights


class Linear(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        period: int | None = None,
        optimize_gathers: bool = True,
    ) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        self.period = period

        inputs_ordinals = list(neurons.inputs.ordinals)

        if period is None:
            gather = build_optimal_multi_layer_gather(inputs_ordinals)
        else:
            gather = build_optimal_multi_layer_gather_and_reshape(inputs_ordinals, dim=period)

        self.weight, weight_idx_map = create_weights(neurons.input_weights)

        weight_idxs: list[int] = [weight_idx_map[str(w.id)] for w in neurons.input_weights]

        if period is None:
            gather_weights = build_optimal_gather(weight_idxs)
        else:
            gather_weights = build_optimal_gather_and_reshape(weight_idxs, dim=period)

        if period is not None and optimize_gathers:
            # can we simplify?
            can_simplify_inputs = gather.optimal_period == period and not gather.is_optimal
            can_simplify_weights = gather_weights.optimal_period == period and not gather_weights.is_optimal

            if can_simplify_inputs and can_simplify_weights:
                # TODO
                warnings.warn(
                    "Can simplify both weights and inputs due to matching dimensionality, "
                    "but it is not implemented yet, so the run will be slow."
                )
            elif can_simplify_inputs:
                gather = gather.get_optimal()
            elif can_simplify_weights:
                gather_weights = gather_weights.get_optimal()

        self.gather = gather
        self.gather_weights = gather_weights

        # TODO invert gather+linear to gather_set+linear+gather if beneficial

    def forward(self, layer_values: dict[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        w = self.gather_weights(w)
        y = w @ input_values
        return y

    def extra_repr(self) -> str:
        return f"period={self.period},"
