from collections import defaultdict
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from lib.nn.sources.source import LayerOrdinal
from lib.utils import detect_repeating_sequence_in_list


class GatherLike(Protocol):
    @property
    def total_items(self) -> int:
        ...

    @property
    def optimal_period(self) -> int:
        ...

    @property
    def is_optimal(self) -> bool:
        ...


@runtime_checkable
class GatherModuleLike(GatherLike, Protocol):
    def get_optimal(self) -> "GatherModuleLike":
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class NoopGather(torch.nn.Module, GatherModuleLike):
    def __init__(self, input_length: int) -> None:
        super().__init__()
        self._input_length = input_length

    @property
    def total_items(self) -> int:
        return self._input_length

    @property
    def optimal_period(self) -> int:
        return self._input_length

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "NoopGather":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TakeValue(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def extra_repr(self) -> str:
        return f"ordinal={self.ordinal}"

    @property
    def total_items(self) -> int:
        return 1

    @property
    def optimal_period(self) -> int:
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "TakeValue":
        return self

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.ordinal].unsqueeze(0)


class SliceValues(torch.nn.Module, GatherModuleLike):
    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"start={self.start}, end={self.end}"

    @property
    def total_items(self) -> int:
        return self.end - self.start

    @property
    def optimal_period(self) -> int:
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "SliceValues":
        return self

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start: self.end]


class TakeEachNth(torch.nn.Module, GatherModuleLike):
    def __init__(self, step: int, start: int, end: int) -> None:
        super().__init__()
        self.step = step
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"step={self.step}, start={self.start}, end={self.end}"

    @property
    def total_items(self) -> int:
        len = self.end - self.start
        return -(-len // self.step)

    @property
    def optimal_period(self) -> int:
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "TakeEachNth":
        return self

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start: self.end: self.step]


class GenericGather(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinals: Sequence[int]) -> None:
        super().__init__()
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def extra_repr(self) -> str:
        return f"ordinals=(list of size {self.ordinals.shape[0]})"

    @property
    def total_items(self) -> int:
        return len(self.ordinals)

    @property
    def optimal_period(self) -> int:
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "GatherModuleLike":
        return self

    def forward(self, layer_input: torch.Tensor):
        return torch.index_select(layer_input, 0, self.ordinals)


class Repeat(torch.nn.Module, GatherModuleLike):
    def __init__(self, input_length: int, repeats: int, total_length: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.total_length = total_length
        self._input_length = input_length

    def extra_repr(self) -> str:
        return f"repeats={self.repeats}, total_length={self.total_length}"

    @property
    def total_items(self) -> int:
        return self.total_length

    @property
    def optimal_period(self) -> int:
        return self._input_length

    @property
    def is_optimal(self) -> bool:
        return False

    def get_optimal(self) -> "GatherModuleLike":
        return NoopGather(input_length=self._input_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x


class GatherAndRepeat(torch.nn.Module, GatherModuleLike):
    def __init__(self, the_gather_module: GatherModuleLike, repeats: int, total_length: int) -> None:
        super().__init__()
        self.gather = the_gather_module
        # Repeat is slow !!
        self.repeat = Repeat(the_gather_module.total_items, repeats, total_length)

    @property
    def total_items(self) -> int:
        return self.repeat.total_length

    @property
    def optimal_period(self) -> int:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return False

    def get_optimal(self) -> "GatherModuleLike":
        return self.gather.get_optimal()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.repeat(x)
        return x


def build_optimal_gather(
    ordinals: Sequence[int],
    allow_subseq=True,
) -> TakeValue | SliceValues | TakeEachNth | GenericGather | GatherAndRepeat:
    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        return TakeValue(ordinals[0])

    step = ordinals[1] - ordinals[0]
    all_ordinals_differ_by_step = all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_step:
        if step == 1:
            return SliceValues(ordinals[0], ordinals[-1] + 1)

        return TakeEachNth(step=step, start=ordinals[0], end=ordinals[-1] + 1)

    if allow_subseq:
        subseq = detect_repeating_sequence_in_list(ordinals, allow_last_incomplete=True)
        if subseq is not None and len(subseq) <= len(ordinals) // 2:
            subseq_gather = build_optimal_gather(subseq.tolist(), allow_subseq=False)
            repeats = -(-len(ordinals) // subseq_gather.optimal_period)
            total_length = len(ordinals)

            if total_length == subseq_gather.total_items:
                return subseq_gather

            return GatherAndRepeat(subseq_gather, repeats=repeats, total_length=total_length)

    return GenericGather(ordinals)


@runtime_checkable
class LayerGatherModuleLike(GatherLike, Protocol):
    def get_optimal(self) -> "LayerGatherModuleLike":
        ...

    def __call__(self, layer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        ...


class SingleLayerGather(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, input_layer: int, the_gather_module: GatherModuleLike) -> None:
        super().__init__()
        self.input_layer = input_layer
        self.delegate = the_gather_module

    @property
    def total_items(self) -> int:
        return self.delegate.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.delegate.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.delegate.is_optimal

    def get_optimal(self) -> "SingleLayerGather":
        if self.is_optimal:
            return self

        optimal_delegate = self.delegate.get_optimal()
        return SingleLayerGather(input_layer=self.input_layer, the_gather_module=optimal_delegate)

    def extra_repr(self) -> str:
        return f"layer={self.input_layer},"

    def forward(self, layer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        x = layer_values[self.input_layer]
        x = self.delegate(x)
        return x


def build_optimal_single_layer_gather(input_layer: int, ordinals: list[int]):
    """Build the optimal gather network module when inputs are guaranteed to come from a single layer."""
    gather = build_optimal_gather(ordinals)
    return SingleLayerGather(input_layer, gather)


class LayersGatherConcat(torch.nn.Module, LayerGatherModuleLike):
    """
    Gather for inputs coming from multiple layers.

    First performs individual gathers for each input layer. Then concatenates the result to a single tensor.
    Provides a mapping to remember which value contains which neuron output.
    """

    def __init__(self, per_layer_ordinals: dict[int, list[int]]) -> None:
        super().__init__()
        self.layers = sorted(per_layer_ordinals.keys(), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        ### setup single-layer gathers ###

        # each gather is a set gather, so there is no optimal period for any of them
        self.layer_gathers = torch.nn.ModuleDict(
            {str(l): build_optimal_gather(per_layer_ordinals[l]) for l in self.layers}
        )

        ### setup idx map ###

        layer_sizes = [len(per_layer_ordinals[layer]) for layer in self.layers]
        layer_prefixes = np.concatenate([[0], np.cumsum(layer_sizes)[:-1]])

        self.idx_map: dict[LayerOrdinal, int] = {
            LayerOrdinal(l, o): layer_prefixes[layers_map[l]] + i
            for l in self.layers
            for i, o in enumerate(per_layer_ordinals[l])
        }

    @property
    def total_items(self) -> int:
        return sum((g.total_items for g in self.layer_gathers.values()))

    @property
    def optimal_period(self) -> int:
        # each gather is a set gather, so there is no optimal period for any of them
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "LayersGatherConcat":
        return self

    def forward(self, layer_values: dict[int, torch.Tensor]):
        xs = [self.layer_gathers[str(layer)](layer_values[layer]) for layer in self.layers]

        # TODO: Do we really need broadcasting here? Can we resolve this in a simpler way
        # (such as ideally by knowing the input dimensions)?
        shapes = [t.shape[1:] for t in xs]

        shape_hull = [-1]
        shape_hull.extend(torch.broadcast_shapes(*shapes))

        xs = [t.expand(*shape_hull) for t in xs]

        x = torch.concatenate(xs)
        return x


class MultiLayerGather(torch.nn.Module, LayerGatherModuleLike):
    """
    Gather for inputs coming from multiple layers.

    First performs individual gathers for each input layer. Then concatenates the result to a single tensor. This is
    done by a MultiLayerSetGather.

    Finally, performs the non-set gather using a final single Gather.
    """

    def __init__(self, multi_layer_set_gather: LayersGatherConcat, final_gather: GatherModuleLike) -> None:
        super().__init__()
        self.multi_layer_set_gather = multi_layer_set_gather

        # ordinals = [multi_layer_set_gather.idx_map[l, o] for l, o in input_layer_ordinal_pairs]
        self.final_gather = final_gather

    @property
    def total_items(self) -> int:
        return self.final_gather.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.final_gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.final_gather.is_optimal

    def get_optimal(self) -> "MultiLayerGather":
        if self.is_optimal:
            return self

        final_optimal = self.final_gather.get_optimal()
        return MultiLayerGather(self.multi_layer_set_gather, final_optimal)

    def forward(self, layer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.multi_layer_set_gather(layer_values)
        x = self.final_gather(x)
        return x


def build_optimal_multi_layer_gather(inputs_ordinals: list[LayerOrdinal]):
    layer0, _ = inputs_ordinals[0]
    is_single_layer = all((layer0 == l for l, _ in inputs_ordinals[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather(layer0, [o for _, o in inputs_ordinals])

    per_layer_ordinals_set: dict[int, set[int]] = defaultdict(lambda: set())

    for l, o in inputs_ordinals:
        per_layer_ordinals_set[l].add(o)

    per_layer_ordinals = {l: sorted(o) for l, o in per_layer_ordinals_set.items()}

    multi_layer_set_gather = LayersGatherConcat(per_layer_ordinals)
    final_ordinals = [multi_layer_set_gather.idx_map[p] for p in inputs_ordinals]
    final_gather = build_optimal_gather(final_ordinals)

    return MultiLayerGather(multi_layer_set_gather, final_gather)


class ViewWithPeriod(torch.nn.Module, GatherModuleLike):
    def __init__(self, input_length: int, period: int) -> None:
        super().__init__()
        self.input_length = input_length
        self.period = period

    def extra_repr(self) -> str:
        return f"period={self.period}"

    @property
    def total_items(self) -> int:
        return self.input_length // self.period

    @property
    def optimal_period(self) -> int:
        return self.period

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "GatherModuleLike":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view([-1, self.period, *x.shape[1:]])


class GatherAndView(torch.nn.Module, GatherModuleLike):
    def __init__(self, gather: GatherModuleLike, view: ViewWithPeriod) -> None:
        super().__init__()
        self.gather = gather
        self.view = view

    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    def get_optimal(self) -> "GatherAndView | ViewWithPeriod":
        if self.is_optimal:
            return self

        gather_optimal = self.gather.get_optimal()

        if isinstance(gather_optimal, NoopGather):
            return self.view

        return GatherAndView(
            gather=gather_optimal,
            view=self.view,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.view(x)
        return x


class LayerGatherAndView(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, gather: LayerGatherModuleLike, view: ViewWithPeriod) -> None:
        super().__init__()
        self.gather = gather
        self.view = view

    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    def get_optimal(self) -> "LayerGatherAndView":
        if self.is_optimal:
            return self

        gather_optimal = self.gather.get_optimal()

        return LayerGatherAndView(
            gather=gather_optimal,
            view=self.view,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.gather(layer_values)
        x = self.view(x)
        return x


def build_optimal_gather_and_reshape(ordinals: list[int], period: int):
    gather = build_optimal_gather(ordinals)
    reshape = ViewWithPeriod(input_length=gather.total_items, period=period)
    return GatherAndView(gather, reshape)


def build_optimal_multi_layer_gather_and_reshape(inputs_ordinals: list[LayerOrdinal], period: int):
    gather = build_optimal_multi_layer_gather(inputs_ordinals)
    reshape = ViewWithPeriod(input_length=gather.total_items, period=period)
    return LayerGatherAndView(gather, reshape)
