import warnings
from typing import Collection, OrderedDict

import numpy as np
import torch
from torch.jit import unused

from lib.nn.gather import (
    GatherModuleLike,
    LayerGatherModuleLike,
    NoopGather,
    Periodic,
    build_optimal_gather,
    build_optimal_gather_and_reshape,
    build_optimal_multi_layer_gather,
    build_optimal_multi_layer_gather_and_reshape,
    get_optimal_gather_for_period,
    get_optimal_layer_gather_for_period,
)
from lib.nn.sources.base import LayerOrdinal, Network, Neurons, WeightDefinition
from lib.nn.weight import create_weights_and_gather


class Linear(torch.nn.Module, Periodic):
    def __init__(
        self,
        inputs_gather: LayerGatherModuleLike,
        weight: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.gather = inputs_gather
        self.weight = weight

    @unused
    def get_period(self) -> int | None:
        if isinstance(self.gather, Periodic) and isinstance(self.weight, Periodic):
            p1, p2 = self.gather.get_period(), self.weight.get_period()
            if p1 == p2:
                return p1
            else:
                warnings.warn(
                    f"Unexpected situation in linear layer: Gather has period {p1} and Weight has period {p2}.\n"
                    f"Gather:\n{self.gather}\n\nWeight:\n{self.weight}\n"
                )

        return None

    def unwrap_final_gather(self) -> None:
        return None

    def forward(self, layer_values: dict[str, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        y = w @ input_values
        return y


class ReturnWeights(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, layer_values: dict[str, torch.Tensor]):
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

    @unused
    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @unused
    @property
    def optimal_period(self) -> int:
        return self.gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    @unused
    def get_period(self) -> int | None:
        return self.gather.get_period() or self.linear.get_period()

    @unused
    def get_optimal(self):
        gather = self.gather.get_optimal()

        if isinstance(gather, NoopGather):
            return self.linear

        if gather == self.gather:
            return self

        return LinearAndGather(self.linear, gather)

    @unused
    def unwrap_final_gather(self) -> tuple[torch.nn.Module, dict[int, int]] | None:
        tpl = self.gather.unwrap_final_gather()
        if tpl is None:
            return None

        gather2, idx_map = tpl
        if isinstance(gather2, NoopGather):
            return self.linear, idx_map
        else:
            return LinearAndGather(self.linear, self.gather2), idx_map

    def forward(self, layer_values: dict[str, torch.Tensor]):
        y = self.linear(layer_values)
        y = self.gather(y)
        return y


def _count_unique(inputs_ordinals: list[LayerOrdinal], weight_definitions: Collection[WeightDefinition]):
    arrs = [(io.layer, io.ordinal, wd.id, int(wd.learnable)) for io, wd in zip(inputs_ordinals, weight_definitions)]
    arrs.sort()

    a = np.array(arrs)
    out = a.shape[0] - (a[:-1] == a[1:]).all(axis=1).sum()
    return out


def _build_optimal_linear_unique_and_gather(
    inputs_ordinals: list[LayerOrdinal],
    weight_definitions: Collection[WeightDefinition],
    period: int | None,
    group_learnable_weight_parameters=True,
    optimize_linear_gathers=True,
):
    n_unique = _count_unique(inputs_ordinals, weight_definitions)
    inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
    n_total = len(inputs_ordinals)

    # check if it is worth it to do it like this
    if n_unique == n_total:
        return _build_optimal_linear(
            inputs_ordinals, weight_definitions, period, group_learnable_weight_parameters, optimize_linear_gathers
        )

    inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
    inputs_weights_pairs_unique = list(OrderedDict.fromkeys(inputs_weights_pairs).keys())
    assert len(inputs_weights_pairs_unique) == n_unique

    inputs_ordinals_unique = [o for o, _ in inputs_weights_pairs_unique]
    weight_definitions_unique = [wd for _, wd in inputs_weights_pairs_unique]

    linear_unique = _build_optimal_linear(
        inputs_ordinals_unique,
        weight_definitions_unique,
        period=None,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
        optimize_linear_gathers=optimize_linear_gathers,
    )

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
    group_learnable_weight_parameters: bool,
    optimize_linear_gathers: bool,
):
    inputs_ordinals = list(neurons.inputs.ordinals)

    weight_definitions = list(neurons.input_weights)

    if len(weight_definitions) == 0:
        # skip a linear layer and just build a gather layer
        if period is None:
            return build_optimal_multi_layer_gather(inputs_ordinals)
        else:
            return build_optimal_multi_layer_gather_and_reshape(inputs_ordinals, period=period)

    assert len(weight_definitions) == len(inputs_ordinals)

    # TODO: allow skipping matmul when possible again
    # TODO: we can also replace matmul with piecewise multiplication whenever possible,
    # and that way cover even more use-cases with a simpler operation
    # inp_layers = set((l for l, _ in neurons.inputs.ordinals))
    # all_inputs_facts = all((network.layers[l].type == "FactLayer" for l in inp_layers))
    # if all_inputs_facts:
    #     all_inputs_ones = all((np.all(inp == 1.0) for inp in neurons.inputs.get_values_numpy()))

    #     if all_inputs_ones:
    #         # can skip the matmul entirely and can just return the weights
    #         return ReturnWeights(
    #             create_weights_and_gather(
    #                 weight_definitions,
    #                 period=period,
    #                 group_learnable_weight_parameters=group_learnable_weight_parameters,
    #             )
    #         )

    return _build_optimal_linear_unique_and_gather(
        inputs_ordinals=inputs_ordinals,
        weight_definitions=weight_definitions,
        period=period,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
        optimize_linear_gathers=optimize_linear_gathers,
    )
