import warnings
from typing import Collection, Mapping, OrderedDict

import numpy as np
import torch
from torch.jit import unused

from lib.nn.gather import (
    GatherModuleLike,
    NoopGather,
    build_optimal_gather_and_reshape,
    build_optimal_multi_layer_gather_and_reshape,
    get_optimal_gather_for_periodic_gather,
)
from lib.nn.sources.base import LayerOrdinal, Network, Neurons, WeightDefinition
from lib.nn.utils import Sequential, ShapeTransformable
from lib.nn.weight import WeightModuleLike, create_weights_and_gather


class Linear(torch.nn.Module, ShapeTransformable):
    def __init__(
        self,
        inputs_gather: GatherModuleLike,
        weight: GatherModuleLike | WeightModuleLike,
    ) -> None:
        super().__init__()
        self.gather = inputs_gather
        self.weight = weight

    @unused
    def compute_output_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        weight_shape = self.weight.compute_output_shape()
        gather_shape = self.gather.compute_output_shape(layer_shapes)
        assert weight_shape[-1] == gather_shape[-2]

        begin_shape = [max(a, b) for a, b in zip(weight_shape[:-2], gather_shape[:-2])]
        return [*begin_shape, weight_shape[-2], gather_shape[-1]]

    def unwrap_final_gather(self, shape_like) -> tuple["Linear", dict[int, int]]:
        return self, {}

    def forward(self, layer_values: dict[str, torch.Tensor]):
        input_values = self.gather(layer_values)
        w = self.weight()
        y = w @ input_values
        return y


def _build_optimal_linear(
    inputs_ordinals: list[LayerOrdinal],
    layer_shapes: Mapping[str, list[int]],
    weight_definitions: Collection[WeightDefinition],
    period: int | None,
    group_learnable_weight_parameters: bool,
    optimize_linear_gathers: bool,
    use_unique_pre_gathers: bool,
):
    # will not reshape if period is None
    gather = build_optimal_multi_layer_gather_and_reshape(
        inputs_ordinals,
        layer_shapes,
        use_unique_pre_gathers=use_unique_pre_gathers,
        period=period,
    )

    weight = create_weights_and_gather(
        weight_definitions,
        period=period,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
    )

    if period is not None and optimize_linear_gathers:
        # can we simplify?
        gather_optimal = get_optimal_gather_for_periodic_gather(gather, period=period, input_shape_like=layer_shapes)
        weight_optimal = get_optimal_gather_for_periodic_gather(weight, period=period, input_shape_like=None)

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


class LinearAndGather(torch.nn.Module, GatherModuleLike):
    def __init__(
        self,
        linear: Linear,
        gather: GatherModuleLike,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.gather = gather

    @unused
    def compute_output_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        shape = self.linear.compute_output_shape(layer_shapes)
        shape = self.gather.compute_output_shape(shape)
        return shape

    @unused
    def compute_optimal_shape(self, layer_shapes: dict[str, list[int]]) -> list[int]:
        shape = self.linear.compute_output_shape(layer_shapes)
        shape = self.gather.compute_optimal_shape(shape)
        return shape

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    @unused
    def get_optimal(self):
        gather = self.gather.get_optimal()

        if gather == self.gather:
            return self

        return LinearAndGather(self.linear, gather)

    @unused
    def unwrap_final_gather(self, shape) -> tuple[ShapeTransformable, dict[int, int]]:
        gather2, idx_map = self.gather.unwrap_final_gather(shape)
        if isinstance(gather2, NoopGather):
            return self.linear, idx_map

        if gather2 == self.gather:
            return self, {}

        return Sequential([self.linear, gather2]), idx_map

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
    layer_shapes: Mapping[str, list[int]],
    weight_definitions: Collection[WeightDefinition],
    period: int | None,
    group_learnable_weight_parameters: bool,
    optimize_linear_gathers: bool,
    use_unique_pre_gathers: bool,
):
    n_unique = _count_unique(inputs_ordinals, weight_definitions)
    inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
    n_total = len(inputs_ordinals)

    # check if it is worth it to do it like this
    if n_unique == n_total:
        return _build_optimal_linear(
            inputs_ordinals=inputs_ordinals,
            layer_shapes=layer_shapes,
            weight_definitions=weight_definitions,
            period=period,
            group_learnable_weight_parameters=group_learnable_weight_parameters,
            optimize_linear_gathers=optimize_linear_gathers,
            use_unique_pre_gathers=use_unique_pre_gathers,
        )

    inputs_weights_pairs = list(zip(inputs_ordinals, weight_definitions))
    inputs_weights_pairs_unique = list(OrderedDict.fromkeys(inputs_weights_pairs).keys())
    assert len(inputs_weights_pairs_unique) == n_unique

    inputs_ordinals_unique = [o for o, _ in inputs_weights_pairs_unique]
    weight_definitions_unique = [wd for _, wd in inputs_weights_pairs_unique]

    linear_unique = _build_optimal_linear(
        inputs_ordinals=inputs_ordinals_unique,
        layer_shapes=layer_shapes,
        weight_definitions=weight_definitions_unique,
        period=None,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
        optimize_linear_gathers=optimize_linear_gathers,
        use_unique_pre_gathers=use_unique_pre_gathers,
    )

    to_order_idxs = [inputs_weights_pairs_unique.index(key) for key in inputs_weights_pairs]

    # will not reshape if period is None
    post_linear_gather = build_optimal_gather_and_reshape(to_order_idxs, period=period)

    return LinearAndGather(linear_unique, post_linear_gather)


def build_optimal_linear(
    network: Network,
    neurons: Neurons,
    layer_shapes: Mapping[str, list[int]],
    period: int | None,
    group_learnable_weight_parameters: bool,
    optimize_linear_gathers: bool,
    use_unique_pre_gathers: bool,
):
    inputs_ordinals = list(neurons.inputs.ordinals)

    weight_definitions = list(neurons.input_weights)

    if len(weight_definitions) == 0:
        # skip a linear layer and just build a gather layer
        # will not reshape if period is None
        return build_optimal_multi_layer_gather_and_reshape(
            inputs_ordinals,
            layer_shapes,
            use_unique_pre_gathers=use_unique_pre_gathers,
            period=period,
        )

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
        layer_shapes=layer_shapes,
        weight_definitions=weight_definitions,
        period=period,
        group_learnable_weight_parameters=group_learnable_weight_parameters,
        optimize_linear_gathers=optimize_linear_gathers,
        use_unique_pre_gathers=use_unique_pre_gathers,
    )
