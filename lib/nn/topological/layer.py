from collections.abc import Iterable
from typing import Collection, TypeVar

import torch
from torch.nn import Identity

from lib.nn.aggregation.fixed_count import build_fixed_count_aggregate
from lib.nn.aggregation.universal import build_optimal_reshape_aggregate
from lib.nn.sources.base import LayerNeurons, Network
from lib.nn.topological.linear import build_optimal_linear
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


class Layer(torch.nn.Module):
    def __new__(
        cls,
        out_to: int,
        network: Network,
        neurons: LayerNeurons,
        settings: Settings,
    ):
        if neurons.layer.type == "FactLayer":
            return FactLayer(out_to, neurons, settings)

        return super().__new__(cls)

    def __init__(
        self,
        out_to: int,
        network: Network,
        neurons: LayerNeurons,
        settings: Settings,
    ) -> None:
        super().__init__()

        period_or_counts = _get_single_if_all_same(neurons.input_lengths)

        if isinstance(period_or_counts, int):
            period = period_or_counts
        else:
            period = None

        self.linear = build_optimal_linear(
            network,
            neurons,
            period=period if period != 1 else None,  # do not reshape if period == 1
            group_learnable_weight_parameters=settings.group_learnable_weight_parameters,
            optimize_linear_gathers=settings.optimize_linear_gathers,
        )

        if period == 1:
            # no need to aggregate, already aggregated (no reshape needed either)
            self.aggregate = Identity()
        else:
            # uneven period or period greater than 1
            aggregation = _assert_all_same("aggregations", neurons.get_aggregations())
            aggregation = aggregation or "sum"

            if period is None:
                # uneven period
                assert self.linear.get_period() is None
                counts = torch.tensor(list(neurons.input_lengths), dtype=torch.int32)

                self.aggregate = build_optimal_reshape_aggregate(
                    aggregation, counts, allow_non_builtin_torch_ops=settings.allow_non_builtin_torch_ops
                )
            else:
                # even period, already reshaped by linear layer
                # must only aggregate
                self.aggregate = build_fixed_count_aggregate(aggregation)

        transformation = _assert_all_same("transformations", neurons.get_transformations())
        self.transform = build_transformation(transformation)

        self.out_to = out_to

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
