import warnings
from typing import OrderedDict

import numpy as np
import torch

from lib.nn.gather import (
    build_optimal_gather,
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
)
from lib.nn.sources.source import NeuralNetworkDefinition, Neurons
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

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('LINEAR__GATHER_INPS'):
            input_values = self.gather(layer_values)
        with torch.profiler.record_function('LINEAR__WEIGHT'):
            w = self.weight()
        with torch.profiler.record_function('LINEAR__GATHER_WEIGHTS'):
            w = self.gather_weights(w)
        with torch.profiler.record_function('LINEAR__LINEAR'):
            y = w @ input_values
        return y

    def extra_repr(self) -> str:
        return f"period={self.period},"


class UniqueLinearAndCollect(torch.nn.Module):
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
        weight_definitions = list(neurons.input_weights)

        inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
        inputs_weights_pairs_unique = list(OrderedDict.fromkeys(inputs_weights_pairs).keys())

        inputs_ordinals_unique = [o for o, _ in inputs_weights_pairs_unique]
        weight_definitions_unique = [wd for _, wd in inputs_weights_pairs_unique]

        to_order_idxs = [inputs_weights_pairs_unique.index(key) for key in inputs_weights_pairs]

        if period is None:
            gather = build_optimal_multi_layer_gather(inputs_ordinals_unique)
        else:
            gather = build_optimal_multi_layer_gather_and_reshape(inputs_ordinals_unique, period=period)
        self.gather = gather

        self.weight, self.gather_weights = create_weights_and_gather(
            weight_definitions_unique,
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

        self.post_linear_gather = build_optimal_gather(to_order_idxs)

    def forward(self, layer_values: dict[int, torch.Tensor]):
        with torch.profiler.record_function('LINEAR__GATHER_INPS_UNIQ'):
            input_values = self.gather(layer_values)
        with torch.profiler.record_function('LINEAR__WEIGHT'):
            w = self.weight()
        with torch.profiler.record_function('LINEAR__GATHER_WEIGHTS_UNIQ'):
            w = self.gather_weights(w)
        with torch.profiler.record_function('LINEAR__LINEAR'):
            y = w @ input_values
        with torch.profiler.record_function('LINEAR__POST_GATHER'):
            y = self.post_linear_gather(y)
        return y

    def extra_repr(self) -> str:
        return f"period={self.period},"


class GatherWeights(torch.nn.Module):
    def __init__(
        self,
        neurons: Neurons,
        settings: Settings,
        period: int | None = None,
    ) -> None:
        super().__init__()

        weight_definitions = list(neurons.input_weights)
        self.weight, self.gather_weights = create_weights_and_gather(
            weight_definitions,
            period=period,
            group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
        )

    def forward(self, x):
        with torch.profiler.record_function('LINEAR__WEIGHT'):
            w = self.weight()
        with torch.profiler.record_function('LINEAR__GATHER_WEIGHTS'):
            w = self.gather_weights(w)
        return w


def build_optimal_linear(
    network: NeuralNetworkDefinition,
    neurons: Neurons,
    settings: Settings,
    gather_unique_first: bool,
    period: int | None = None,
):
    inp_layers = set((l for l, _ in neurons.inputs.ordinals))
    all_inputs_facts = all((network.layers[l].type == "FactLayer" for l in inp_layers))

    # TODO: we can also replace matmul with piecewise multiplication whenever possible,
    # and that way cover even more use-cases with a simpler operation
    if all_inputs_facts:
        all_inputs_ones = all((np.all(inp == 1.0) for inp in neurons.inputs.get_values_numpy()))

        if all_inputs_ones:
            # can skip the matmul entirely and can just return the weights
            return GatherWeights(neurons, settings, period)

    # TODO: guess best `gather_unique_first` value automatically using heuristics
    if gather_unique_first:
        return UniqueLinearAndCollect(neurons, settings, period)
    else:
        return Linear(neurons, settings, period)
