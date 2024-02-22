from typing import Protocol, Sequence, TypeVar, overload, runtime_checkable

import numpy as np
import torch

from lib.nn.topological.layers import LayerOrdinal
from lib.nn.utils.utils import Repeat, ViewWithPeriod
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
    def optimal_period(self) -> int | None:
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
    def optimal_period(self) -> int | None:
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
    def optimal_period(self) -> int | None:
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
    def optimal_period(self) -> int | None:
        return self.total_items

    @property
    def is_optimal(self) -> bool:
        return True

    def get_optimal(self) -> "GatherModuleLike":
        return self

    def forward(self, layer_input: torch.Tensor):
        return torch.index_select(layer_input, 0, self.ordinals)


class GatherAndRepeatNonOptimal(torch.nn.Module, GatherModuleLike):
    def __init__(self, the_gather_module: GatherModuleLike, repeats: int, total_length: int) -> None:
        super().__init__()
        self.gather = the_gather_module
        # Repeat is slow !!
        self.repeat = Repeat(repeats, total_length)

    @property
    def total_items(self) -> int:
        return self.repeat.total_length

    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return False

    def get_optimal(self) -> "GatherModuleLike":
        return self.gather

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.repeat(x)
        return x


def build_optimal_gather(
    ordinals: Sequence[int],
    allow_subseq=True,
):
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
        if subseq is not None:
            subseq_gather = build_optimal_gather(subseq.tolist(), allow_subseq=False)
            repeats = -(-len(ordinals) // subseq_gather.optimal_period)
            total_length = len(ordinals)

            if total_length == subseq_gather.total_items:
                return subseq_gather

            return GatherAndRepeatNonOptimal(subseq_gather, repeats=repeats, total_length=total_length)

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


class MultiLayerSetGather(torch.nn.Module, LayerGatherModuleLike):
    """
    Gather for inputs coming from multiple layers.

    First performs individual gathers for each input layer. Then concatenates the result to a single tensor.
    Provides a mapping to remember which value contains which neuron output.
    """

    def __init__(self, input_layer_ordinal_pairs: Sequence[LayerOrdinal]) -> None:
        super().__init__()
        self.input_layer_ordinal_pairs = input_layer_ordinal_pairs

        self.layers = sorted(set((l for l, _ in input_layer_ordinal_pairs)), reverse=True)
        layers_map = {l: i for i, l in enumerate(self.layers)}

        ### setup single-layer gathers ###

        per_layer_ordinals: dict[int, list[int]] = {
            layer: sorted(set((o for l, o in input_layer_ordinal_pairs if l == layer))) for layer in self.layers
        }

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

        # concatenated_ordinals = [
        #     layer_prefixes[layers_map[l]] + per_layer_ordinals2_map[l][o] for l, o in input_layer_ordinal_pairs
        # ]

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

    def get_optimal(self) -> "MultiLayerSetGather":
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

    def __init__(self, multi_layer_set_gather: MultiLayerSetGather, final_gather: GatherModuleLike) -> None:
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


def build_optimal_inputs_gather(input_layer_ordinal_pairs: list[LayerOrdinal]):
    layer0, _ = input_layer_ordinal_pairs[0]
    is_single_layer = all((layer0 == l for l, _ in input_layer_ordinal_pairs[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather(layer0, [o for _, o in input_layer_ordinal_pairs])

    multi_layer_set_gather = MultiLayerSetGather(input_layer_ordinal_pairs)
    final_ordinals = [multi_layer_set_gather.idx_map[p] for p in input_layer_ordinal_pairs]
    final_gather = build_optimal_gather(final_ordinals)

    return MultiLayerGather(multi_layer_set_gather, final_gather)


class GatherAndReshape(torch.nn.Module, GatherModuleLike):
    def __init__(self, gather: GatherModuleLike, reshape: torch.nn.Module) -> None:
        super().__init__()
        self.gather = gather
        self.reshape = reshape

    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    def get_optimal(self) -> "GatherAndReshape":
        if self.is_optimal:
            return self

        return GatherAndReshape(
            gather=self.gather.get_optimal(),
            reshape=self.reshape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.reshape(x)
        return x


class LayerGatherAndReshape(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, gather: LayerGatherModuleLike, reshape: torch.nn.Module) -> None:
        super().__init__()
        self.gather = gather
        self.reshape = reshape

    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    def get_optimal(self) -> "LayerGatherAndReshape":
        if self.is_optimal:
            return self

        return LayerGatherAndReshape(
            gather=self.gather.get_optimal(),
            reshape=self.reshape,
        )

    def forward(self, layer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.gather(layer_values)
        x = self.reshape(x)
        return x


def build_optimal_gather_and_reshape(ordinals: list[int], dim: int):
    gather = build_optimal_gather(ordinals)
    reshape = ViewWithPeriod(dim)
    return GatherAndReshape(gather, reshape)


def build_optimal_inputs_gather_and_reshape(input_layer_ordinal_pairs: list[LayerOrdinal], dim: int):
    gather = build_optimal_inputs_gather(input_layer_ordinal_pairs)
    reshape = ViewWithPeriod(dim)
    return LayerGatherAndReshape(gather, reshape)
