import warnings
from itertools import product

import torch

from lib.nn.gather import (
    build_optimal_gather,
    build_optimal_gather_and_reshape,
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
)
from lib.nn.topological.layers import Ordinals
from lib.nn.weight import build_weights_from_java, get_weight_shape_single


class Linear(torch.nn.Module):
    def __init__(
        self,
        layer_neurons: list,
        neuron_ordinals: Ordinals,
        period: int | None = None,
        optimize_gathers: bool = True,
    ) -> None:
        super().__init__()

        self.neuron_ids = [n.getIndex() for n in layer_neurons]

        self.period = period

        input_layer_ordinal_pairs = [
            neuron_ordinals[int(inp.getIndex())] for n in layer_neurons for inp in n.getInputs()
        ]

        if period is None:
            gather = build_optimal_multi_layer_gather(input_layer_ordinal_pairs)
        else:
            gather = build_optimal_multi_layer_gather_and_reshape(input_layer_ordinal_pairs, dim=period)

        self.weight, weight_idx_map = build_weights_from_java(layer_neurons)

        def _iter_all_weights():
            return (w for n in layer_neurons for w in n.getWeights())

        weight_idxs: list[int] = [weight_idx_map[str(w.index)] for w in _iter_all_weights()]

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
