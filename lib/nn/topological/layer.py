from collections.abc import Iterable
from typing import Collection, Mapping, TypeVar

import torch
from torch.jit import unused
from torch.nn import Identity

from lib.nn.aggregation.fixed_count import FixedCountAggregation, build_fixed_count_aggregate
from lib.nn.aggregation.universal import ScatterAggregate, ViewAndAggregate, build_optimal_reshape_aggregate
from lib.nn.gather import GatherModuleLike
from lib.nn.sources.base import LayerNeurons, LayerOrdinal, Network
from lib.nn.topological.linear import Linear, build_optimal_linear
from lib.nn.topological.settings import Settings
from lib.nn.transformation import build_transformation
from lib.utils import addindent, atleast_3d_rev, head_and_rest

_T = TypeVar("_T")


def _get_single_if_all_same(source: Collection[_T]) -> _T | Collection[_T]:
    first, rest = head_and_rest(source)

    for v in rest:
        if first != v:
            return source

    # all are the same
    return first


def _assert_all_same(what_plural: str, source: Iterable[_T]) -> _T:
    first, rest = head_and_rest(source)

    for v in rest:
        assert first == v, f"Assertion failed: found {what_plural} {first} and {v}"

    return first


def _assert_all_same_ignore_none(what_plural: str, source: Iterable[_T]) -> _T | None:
    first = None

    for v in source:
        if v is None:
            continue

        if first is None:
            first = v
            continue

        assert first == v, f"Assertion failed: found {what_plural} {first} and {v}"

    return first


class FactLayer(torch.nn.Module):
    def __init__(self, out_to: int, neurons: LayerNeurons, settings: Settings) -> None:
        super().__init__()

        self.neuron_ids = neurons.ids

        # TODO: assumption: FactLayer has no transformation
        transformation = _assert_all_same("transformations", neurons.get_transformations())
        assert transformation in (None, "identity")

        # TODO assumption: FactLayer has no weights
        assert len(list(neurons.input_weights)) == 0

        self.value = torch.nn.Parameter(
            atleast_3d_rev(torch.stack(list(neurons.get_values_torch()))),
            requires_grad=False,
        )
        self.out_to = out_to

    def extra_repr(self) -> str:
        return f"out_to={self.out_to}, len={self.value.shape[0]}"

    def forward(self, layer_values: dict[str, torch.Tensor]):
        layer_values[str(self.out_to)] = self.value
        return layer_values

    @unused
    @property
    def total_items(self) -> int:
        return self.value.shape[0]

    @unused
    def unwrap_final_gather(self) -> None:
        return None


class Layer(torch.nn.Module):
    @staticmethod
    def from_network(
        out_to: int,
        network: Network,
        neurons: LayerNeurons,
        layer_sizes: Mapping[int, int],
        settings: Settings,
    ):
        if neurons.layer.type == "FactLayer":
            return FactLayer(out_to, neurons, settings)

        period_or_counts = _get_single_if_all_same(neurons.input_lengths)

        if isinstance(period_or_counts, int):
            period = period_or_counts
        else:
            period = None

        linear = build_optimal_linear(
            network,
            neurons,
            layer_sizes=layer_sizes,
            period=period if period != 1 else None,  # do not reshape if period == 1
            group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
            optimize_linear_gathers=settings.optimize_linear_gathers,
            use_unique_pre_gathers=settings.use_unique_pre_gathers,
        )

        if period == 1:
            # no need to aggregate, already aggregated (no reshape needed either)
            aggregate = Identity()
        else:
            # uneven period or period greater than 1
            aggregation = _assert_all_same_ignore_none("aggregations", neurons.get_aggregations())
            aggregation = aggregation or "sum"

            if period is None:
                # uneven period
                assert linear.get_period() is None
                counts = torch.tensor(list(neurons.input_lengths), dtype=torch.int32)

                aggregate = build_optimal_reshape_aggregate(
                    aggregation, counts, allow_non_builtin_torch_ops=settings.allow_non_builtin_torch_ops
                )
            else:
                # even period, already reshaped by linear layer
                # must only aggregate
                aggregate = build_fixed_count_aggregate(aggregation)

        transformation = _assert_all_same(
            "transformations", (None if t == "identity" else t for t in neurons.get_transformations())
        )
        transform = build_transformation(transformation)

        return Layer(
            out_to=out_to,
            linear=linear,
            aggregate=aggregate,
            transform=transform,
        )

    def __init__(
        self,
        out_to: int,
        linear: GatherModuleLike | Linear,
        aggregate: Identity | ViewAndAggregate | ScatterAggregate | FixedCountAggregation,
        transform: torch.nn.Module,
    ):
        super().__init__()
        self.out_to = out_to
        self.linear = linear
        self.aggregate = aggregate
        self.transform = transform

    @unused
    @property
    def total_items(self) -> int:
        return self.linear.total_items

    @unused
    def unwrap_final_gather(self) -> tuple["Layer", dict[LayerOrdinal, LayerOrdinal]] | None:
        if not isinstance(self.aggregate, Identity):
            return None

        tpl = self.linear.unwrap_final_gather()
        if tpl is None:
            return None

        mdl, idx_map = tpl
        new_layer = Layer(
            out_to=self.out_to,
            linear=mdl,
            aggregate=self.aggregate,
            transform=self.transform,
        )

        l = self.out_to
        ord_map = {LayerOrdinal(l, a): LayerOrdinal(l, b) for a, b in idx_map.items()}

        return new_layer, ord_map

    def extra_repr(self) -> str:
        return f"out_to={self.out_to},"

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            # hide identity submodules
            if isinstance(module, Identity):
                continue

            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines[1:] + child_lines

        main_str = self._get_name() + "("
        if lines:
            main_str += extra_lines[0]
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def forward(self, layer_values: dict[str, torch.Tensor]):
        x = self.linear(layer_values)
        x = self.aggregate(x)
        x = self.transform(x)
        layer_values[str(self.out_to)] = x
        return layer_values
