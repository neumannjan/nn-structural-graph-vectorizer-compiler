import warnings
from typing import Collection, Mapping, OrderedDict

import numpy as np
import torch

from lib.nn.gather import (
    GatherModuleLike,
    LayerGatherModuleLike,
    NoopGather,
    build_optimal_gather,
    build_optimal_gather_and_reshape,
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
    get_optimal_gather_for_period,
    get_optimal_layer_gather_for_period,
)
from lib.nn.sources.base import LayerOrdinal, Network, Neurons, WeightDefinition
from lib.nn.topological.settings import Settings
from lib.nn.weight import create_weights_and_gather


class Linear(torch.nn.Module):
    def __init__(
        self,
        inputs_gather: LayerGatherModuleLike,
        weight: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.gather = inputs_gather
        self.weight = weight

    def forward(self, layer_values: Mapping[int, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        y = w @ input_values
        return y


class ReturnWeights(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, layer_values: Mapping[int, torch.Tensor]):
        return self.weight()


def _build_optimal_linear(
    inputs_ordinals: list[LayerOrdinal],
    weight_definitions: Collection[WeightDefinition],
    period: int | None,
    group_learnable_weight_parameters=True,
    optimize_linear_gathers=True,
):
    if period is None:
        gather = build_optimal_multi_layer_gather(inputs_ordinals)
    else:
        gather = build_optimal_multi_layer_gather_and_reshape(inputs_ordinals, period=period)

    weight = create_weights_and_gather(
        weight_definitions,
        period=period,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
    )

    if period is not None and optimize_linear_gathers:
        # can we simplify?
        gather_optimal = get_optimal_layer_gather_for_period(gather, period=period)
        weight_optimal = get_optimal_gather_for_period(weight, period=period)

        can_simplify_inputs = gather_optimal != gather
        can_simplify_weight = weight_optimal != weight

        if can_simplify_inputs and can_simplify_weight:
            # TODO
            warnings.warn(
                "Can simplify both weights and inputs due to matching dimensionality, "
                "but it is not implemented yet, so the run will be slow."
            )
        elif can_simplify_inputs:
            gather = gather_optimal
        elif can_simplify_weight:
            weight = weight_optimal

    return Linear(gather, weight)


class LinearAndGather(torch.nn.Module, LayerGatherModuleLike):
    def __init__(
        self,
        linear: Linear,
        gather: GatherModuleLike,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.gather = gather

    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @property
    def optimal_period(self) -> int:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    def get_optimal(self):
        gather = self.gather.get_optimal()

        if isinstance(gather, NoopGather):
            return self.linear

        if gather == self.gather:
            return self

        return LinearAndGather(self.linear, gather)

    def forward(self, layer_values: Mapping[int, torch.Tensor]):
        y = self.linear(layer_values)
        y = self.gather(y)
        return y


def _build_optimal_linear_unique_and_gather(
    inputs_ordinals: list[LayerOrdinal],
    weight_definitions: Collection[WeightDefinition],
    period: int | None,
    group_learnable_weight_parameters=True,
    optimize_linear_gathers=True,
):
    inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
    inputs_weights_pairs_unique = list(OrderedDict.fromkeys(inputs_weights_pairs).keys())

    inputs_ordinals_unique = [o for o, _ in inputs_weights_pairs_unique]
    weight_definitions_unique = [wd for _, wd in inputs_weights_pairs_unique]

    linear_unique = _build_optimal_linear(
        inputs_ordinals_unique,
        weight_definitions_unique,
        period=None,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
        optimize_linear_gathers=optimize_linear_gathers,
    )

    if len(inputs_weights_pairs_unique) == len(inputs_weights_pairs):
        return linear_unique

    to_order_idxs = [inputs_weights_pairs_unique.index(key) for key in inputs_weights_pairs]

    if period is None:
        post_linear_gather = build_optimal_gather(to_order_idxs)
    else:
        post_linear_gather = build_optimal_gather_and_reshape(to_order_idxs, period=period)

    return LinearAndGather(linear_unique, post_linear_gather)


def build_optimal_linear(
    network: Network,
    neurons: Neurons,
    period: int | None,
    settings: Settings,
    gather_unique_first: bool,
):
    inp_layers = set((l for l, _ in neurons.inputs.ordinals))
    all_inputs_facts = all((network.layers[l].type == "FactLayer" for l in inp_layers))

    weight_definitions = list(neurons.input_weights)

    # TODO: we can also replace matmul with piecewise multiplication whenever possible,
    # and that way cover even more use-cases with a simpler operation
    if all_inputs_facts:
        all_inputs_ones = all((np.all(inp == 1.0) for inp in neurons.inputs.get_values_numpy()))

        if all_inputs_ones:
            # can skip the matmul entirely and can just return the weights
            return ReturnWeights(
                create_weights_and_gather(
                    weight_definitions,
                    period=period,
                    group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
                )
            )

    inputs_ordinals = list(neurons.inputs.ordinals)

    # TODO: guess best `gather_unique_first` value automatically using heuristics
    if gather_unique_first:
        func = _build_optimal_linear_unique_and_gather
    else:
        func = _build_optimal_linear

    return func(
        inputs_ordinals=inputs_ordinals,
        weight_definitions=weight_definitions,
        period=period,
        group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
        optimize_linear_gathers=settings.optimize_linear_gathers,
    )
