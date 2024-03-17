from collections import defaultdict
from typing import Protocol, Sequence, Union, runtime_checkable

import numpy as np
import torch
from torch.jit import unused
from torch.nn import Identity

from lib.nn.sources.base import LayerOrdinal
from lib.nn.utils import broadcast_shapes_compiled
from lib.utils import detect_repeating_K_sequence_in_list, detect_repeating_sequence_in_list


@runtime_checkable
class Periodic(Protocol):
    def get_period(self) -> int | None: ...


class GatherLike(Periodic, Protocol):
    @property
    def total_items(self) -> int: ...

    @property
    def optimal_period(self) -> int: ...

    @property
    def is_optimal(self) -> bool: ...


@runtime_checkable
class GatherModuleLike(GatherLike, Protocol):
    def get_optimal(self) -> "GatherModuleLike": ...

    def unwrap_final_gather(self) -> tuple["GatherModuleLike", dict[int, int]] | None: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class NoopGather(torch.nn.Module, GatherModuleLike):
    def __init__(self, input_length: int | None) -> None:
        super().__init__()
        self._input_length = input_length

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        if self._input_length is None:
            raise ValueError("Cannot infer NoopGather total_items")

        return self._input_length

    @unused
    @property
    def optimal_period(self) -> int:
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "NoopGather":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TakeValue(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinal: int) -> None:
        super().__init__()
        self.ordinal = ordinal

    def extra_repr(self) -> str:
        return f"ordinal={self.ordinal}"

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return 1

    @unused
    @property
    def optimal_period(self) -> int:
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "TakeValue":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        if self.ordinal == 0:
            return NoopGather(None), {}

        return NoopGather(None), {0: self.ordinal}

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.ordinal].unsqueeze(0)


class SliceValues(torch.nn.Module, GatherModuleLike):
    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"start={self.start}, end={self.end}"

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.end - self.start

    @unused
    @property
    def optimal_period(self) -> int:
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "SliceValues":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        ord_map = {i: j for i, j in enumerate(range(self.start, self.end)) if i != j}
        return NoopGather(None), ord_map

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start : self.end]


class TakeEachNth(torch.nn.Module, GatherModuleLike):
    def __init__(self, step: int, start: int, end: int) -> None:
        super().__init__()
        self.step = step
        self.start = start
        self.end = end

    def extra_repr(self) -> str:
        return f"step={self.step}, start={self.start}, end={self.end}"

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        len = self.end - self.start
        return -(-len // self.step)

    @unused
    @property
    def optimal_period(self) -> int:
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "TakeEachNth":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        ord_map = {i: j for i, j in enumerate(range(self.start, self.end, self.step)) if i != j}
        return NoopGather(None), ord_map

    def forward(self, layer_input: torch.Tensor):
        return layer_input[self.start : self.end : self.step]


class GenericGather(torch.nn.Module, GatherModuleLike):
    def __init__(self, ordinals: Sequence[int]) -> None:
        super().__init__()
        self.ordinals = torch.nn.Parameter(torch.tensor(ordinals, dtype=torch.int32), requires_grad=False)

    def extra_repr(self) -> str:
        return f"ordinals=(list of size {self.ordinals.shape[0]})"

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return len(self.ordinals)

    @unused
    @property
    def optimal_period(self) -> int:
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        ord_map = {i: j for i, j in enumerate(self.ordinals.cpu().tolist()) if i != j}
        return NoopGather(None), ord_map

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

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.total_length

    @unused
    @property
    def optimal_period(self) -> int:
        return self._input_length

    @unused
    @property
    def is_optimal(self) -> bool:
        return False

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return NoopGather(input_length=self._input_length)

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        ord_map = {}
        i = 0
        for _ in range(self.repeats):
            for j in range(self._input_length):
                if i != j:
                    ord_map[i] = j
                i += 1
                if i >= self.total_length:
                    return Identity(), ord_map
        return Identity(), ord_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(self.repeats, *([1] * (x.dim() - 1)))
        x = x[: self.total_length]
        return x


class _GatherWrapper(torch.nn.Module, GatherModuleLike):
    def __init__(self, gather: torch.nn.Module) -> None:
        super().__init__()
        self._GatherWrapper__delegate = gather

    @unused
    def get_period(self) -> int | None:
        raise ValueError(f"Cannot infer get_period of {repr(self)}")

    @unused
    @property
    def total_items(self) -> int:
        raise ValueError(f"Cannot infer total_items of {repr(self)}")

    @unused
    @property
    def optimal_period(self) -> int:
        raise ValueError(f"Cannot infer optimal_period of {repr(self)}")

    @unused
    @property
    def is_optimal(self) -> bool:
        raise ValueError(f"Cannot infer is_optimal of {repr(self)}")

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        raise ValueError(f"Cannot infer get_optimal of {repr(self)}")

    @unused
    def unwrap_final_gather(self) -> tuple["GatherModuleLike", dict[int, int]] | None:
        raise ValueError(f"Cannot infer unwrap_final_gather of {repr(self)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._GatherWrapper__delegate(x)

    def repr(self) -> str:
        return f"{self.__class__.__name__}({self._GatherWrapper__delegate})"

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
        delegate = super().__getattr__("_GatherWrapper__delegate")

        if name == "_GatherWrapper__delegate":
            return delegate

        return self.delegate.__getattr__(name)


@unused
def _unwrap_gathers_sequential(
    *modules: "GatherModuleLike | LayerGatherModuleLike",
) -> tuple[torch.nn.Module, dict[int, int]] | None:
    out_modules, out_idx_map = [], {}

    for module in modules:
        tpl = module.unwrap_final_gather()
        if tpl is None:
            continue

        module, idx_map = tpl

        # combine out_modules and module
        out_modules.append(module)

        # combine out_idx_map and idx_map
        for a, b in out_idx_map.items():
            c = idx_map.get(b, b)
            if a == c:
                if a in out_idx_map:
                    del out_idx_map[a]
            else:
                out_idx_map[a] = c

    if len(out_modules) == 0:
        return None

    out_modules = [m for m in out_modules if not isinstance(m, NoopGather)]

    if len(out_modules) == 0:
        return Identity(), out_idx_map
    elif len(out_modules) == 1:
        return out_modules[0], out_idx_map
    else:
        return torch.nn.Sequential(*out_modules), out_idx_map


@unused
def unwrap_gathers_sequential(*modules: GatherModuleLike) -> tuple[GatherModuleLike, dict[int, int]] | None:
    tpl = _unwrap_gathers_sequential(*modules)
    if tpl is None:
        return None
    a, b = tpl
    if isinstance(a, torch.nn.Sequential):
        return _GatherWrapper(a), b


@unused
def unwrap_layer_gathers_sequential(
    first_module: "LayerGatherModuleLike", *modules: "LayerGatherModuleLike | GatherModuleLike"
) -> tuple["LayerGatherModuleLike", dict[int, int]] | None:
    tpl = _unwrap_gathers_sequential(first_module, *modules)
    if tpl is None:
        return None
    a, b = tpl
    if isinstance(a, torch.nn.Sequential):
        return _LayerGatherWrapper(a), b


class GatherAndRepeat(torch.nn.Module, GatherModuleLike):
    def __init__(self, the_gather_module: GatherModuleLike, repeats: int, total_length: int) -> None:
        super().__init__()
        self.gather = the_gather_module
        # Repeat is slow !!
        self.repeat = Repeat(the_gather_module.total_items, repeats, total_length)

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.repeat.total_length

    @unused
    @property
    def optimal_period(self) -> int:
        return self.gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return False

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self.gather.get_optimal()

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        return unwrap_gathers_sequential(self.gather, self.repeat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.repeat(x)
        return x


@runtime_checkable
class LayerGatherModuleLike(GatherLike, Protocol):
    def get_optimal(self) -> "LayerGatherModuleLike": ...

    def unwrap_final_gather(self) -> tuple["LayerGatherModuleLike", dict[int, int]] | None: ...

    def __call__(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor: ...


class _LayerGatherWrapper(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, gather: torch.nn.Module) -> None:
        super().__init__()
        self._GatherWrapper__delegate = gather

    @unused
    def get_period(self) -> int | None:
        raise ValueError(f"Cannot infer get_period of {repr(self)}")

    @unused
    @property
    def total_items(self) -> int:
        raise ValueError(f"Cannot infer total_items of {repr(self)}")

    @unused
    @property
    def optimal_period(self) -> int:
        raise ValueError(f"Cannot infer optimal_period of {repr(self)}")

    @unused
    @property
    def is_optimal(self) -> bool:
        raise ValueError(f"Cannot infer is_optimal of {repr(self)}")

    @unused
    def get_optimal(self) -> "LayerGatherModuleLike":
        raise ValueError(f"Cannot infer get_optimal of {repr(self)}")

    @unused
    def unwrap_final_gather(self) -> tuple["LayerGatherModuleLike", dict[int, int]] | None:
        raise ValueError(f"Cannot infer unwrap_final_gather of {repr(self)}")

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._GatherWrapper__delegate(layer_values)

    def repr(self) -> str:
        return f"{self.__class__.__name__}({self._GatherWrapper__delegate})"

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.Module]:
        delegate = super().__getattr__("_GatherWrapper__delegate")

        if name == "_GatherWrapper__delegate":
            return delegate

        return self.delegate.__getattr__(name)


class LayerGatherAndRepeat(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, the_gather_module: LayerGatherModuleLike, repeats: int, total_length: int) -> None:
        super().__init__()
        self.gather = the_gather_module
        # Repeat is slow !!
        self.repeat = Repeat(the_gather_module.total_items, repeats, total_length)

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.repeat.total_length

    @unused
    @property
    def optimal_period(self) -> int:
        return self.gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return False

    @unused
    def get_optimal(self) -> "LayerGatherModuleLike":
        return self.gather.get_optimal()

    @unused
    def unwrap_final_gather(self) -> tuple[LayerGatherModuleLike, dict[int, int]] | None:
        return unwrap_layer_gathers_sequential(self.gather, self.repeat)

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.gather(layer_values)
        x = self.repeat(x)
        return x


def build_optimal_gather(
    ordinals: Sequence[int],
    allow_subseq=True,
    period_hint: int | None = None,
) -> TakeValue | SliceValues | TakeEachNth | GenericGather | GatherAndRepeat:
    ###### simple retrieval ######

    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        return TakeValue(ordinals[0])

    ###### simple slicing #######

    step = ordinals[1] - ordinals[0]
    all_ordinals_differ_by_step = all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_step:
        if step == 1:
            return SliceValues(ordinals[0], ordinals[-1] + 1)

        return TakeEachNth(step=step, start=ordinals[0], end=ordinals[-1] + 1)

    ###### subsequence with (optimizable) repeat: #######

    if allow_subseq:
        subseq = None
        if period_hint is not None:
            # try the hint first
            subseq = detect_repeating_K_sequence_in_list(ordinals, period=period_hint, allow_last_incomplete=True)

        if subseq is None:
            subseq = detect_repeating_sequence_in_list(ordinals, allow_last_incomplete=True)

        if subseq is not None and len(subseq) <= len(ordinals) // 2:
            subseq_gather = build_optimal_gather(subseq.tolist(), allow_subseq=False)
            repeats = -(-len(ordinals) // subseq_gather.optimal_period)
            total_length = len(ordinals)

            if total_length == subseq_gather.total_items:
                return subseq_gather

            return GatherAndRepeat(subseq_gather, repeats=repeats, total_length=total_length)

    ###### generic fallback implementation ######

    return GenericGather(ordinals)


class SingleLayerGather(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, input_layer: int, the_gather_module: GatherModuleLike) -> None:
        super().__init__()
        self.input_layer = input_layer
        self.delegate = the_gather_module

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.delegate.total_items

    @unused
    @property
    def optimal_period(self) -> int | None:
        return self.delegate.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.delegate.is_optimal

    @unused
    def get_optimal(self) -> "SingleLayerGather":
        if self.is_optimal:
            return self

        optimal_delegate = self.delegate.get_optimal()
        return SingleLayerGather(input_layer=self.input_layer, the_gather_module=optimal_delegate)

    @unused
    def unwrap_final_gather(self) -> tuple[LayerGatherModuleLike, dict[int, int]] | None:
        tpl = self.delegate.unwrap_final_gather()
        if tpl is None:
            return None

        mdl, idx_map = tpl
        return SingleLayerGather(self.input_layer, mdl), idx_map

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}[{self.input_layer}]({repr(self.delegate)})"

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = layer_values[str(self.input_layer)]
        x = self.delegate(x)
        return x


def build_optimal_single_layer_gather(input_layer: int, ordinals: list[int], period_hint: int | None = None):
    """Build the optimal gather network module when inputs are guaranteed to come from a single layer."""
    gather = build_optimal_gather(ordinals, period_hint=period_hint)
    return SingleLayerGather(input_layer, gather)


def _expand_concat_tensors(xs: list[torch.Tensor]) -> torch.Tensor:
    # TODO: Do we really need broadcasting here? Can we resolve this in a simpler way
    # (such as ideally by knowing the input dimensions)?
    shapes = [list(t.shape[1:]) for t in xs]

    shape_hull = [-1]
    shape_hull.extend(broadcast_shapes_compiled(shapes))

    xs = [t.expand(shape_hull) for t in xs]

    x = torch.concatenate(xs)
    return x


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

    def __getitem__(self, layer_id: int):
        return self.layer_gathers[str(layer_id)]

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return sum((g.total_items for g in self.layer_gathers.values()))

    @unused
    @property
    def optimal_period(self) -> int:
        # each gather is a set gather, so there is no optimal period for any of them
        return self.total_items

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "LayersGatherConcat":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[LayerGatherModuleLike, dict[int, int]] | None:
        # cannot unwrap a concat of multiple layers! :(
        return None

    def forward(self, layer_values: dict[str, torch.Tensor]):
        xs = []

        for str_layer, gather in self.layer_gathers.items():
            x_this = gather(layer_values[str_layer])
            xs.append(x_this)

        x = _expand_concat_tensors(xs)
        return x

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({repr(self.layer_gathers)})"


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

    @unused
    def get_period(self) -> int | None:
        return None

    @unused
    @property
    def total_items(self) -> int:
        return self.final_gather.total_items

    @unused
    @property
    def optimal_period(self) -> int | None:
        return self.final_gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.final_gather.is_optimal

    @unused
    def get_optimal(self) -> "MultiLayerGather":
        if self.is_optimal:
            return self

        final_optimal = self.final_gather.get_optimal()
        return MultiLayerGather(self.multi_layer_set_gather, final_optimal)

    @unused
    def unwrap_final_gather(self) -> tuple[LayerGatherModuleLike, dict[int, int]] | None:
        tpl = self.final_gather.unwrap_final_gather()
        if tpl is None:
            return None

        mdl, idx_map = tpl
        if isinstance(mdl, NoopGather):
            out_mdl = self.multi_layer_set_gather
        else:
            out_mdl = MultiLayerGather(self.multi_layer_set_gather, mdl)

        return out_mdl, idx_map

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.multi_layer_set_gather(layer_values)
        x = self.final_gather(x)
        return x


def _build_multi_layer_gather(inputs_ordinals: list[LayerOrdinal], period_hint: int | None = None):
    per_layer_ordinals_set: dict[int, set[int]] = defaultdict(lambda: set())

    for l, o in inputs_ordinals:
        per_layer_ordinals_set[l].add(o)

    per_layer_ordinals = {l: sorted(o) for l, o in per_layer_ordinals_set.items()}

    multi_layer_set_gather = LayersGatherConcat(per_layer_ordinals)
    final_ordinals = [multi_layer_set_gather.idx_map[p] for p in inputs_ordinals]
    final_gather = build_optimal_gather(final_ordinals, period_hint=period_hint)

    return MultiLayerGather(multi_layer_set_gather, final_gather)


def build_optimal_multi_layer_gather(inputs_ordinals: list[LayerOrdinal], period_hint: int | None = None):
    layer0, _ = inputs_ordinals[0]
    is_single_layer = all((layer0 == l for l, _ in inputs_ordinals[1:]))

    if is_single_layer:
        return build_optimal_single_layer_gather(layer0, [o for _, o in inputs_ordinals], period_hint=period_hint)

    return _build_multi_layer_gather(inputs_ordinals, period_hint)


class ViewWithPeriod(torch.nn.Module, GatherModuleLike):
    def __init__(self, input_length: int, period: int) -> None:
        super().__init__()
        self.input_length = input_length
        self.period = period

    def extra_repr(self) -> str:
        return f"period={self.period}"

    @unused
    def get_period(self) -> int | None:
        return self.period

    @unused
    @property
    def total_items(self) -> int:
        return self.input_length // self.period

    @unused
    @property
    def optimal_period(self) -> int:
        return self.period

    @unused
    @property
    def is_optimal(self) -> bool:
        return True

    @unused
    def get_optimal(self) -> "GatherModuleLike":
        return self

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [-1, self.period]
        shape.extend(x.shape[1:])
        return x.view(shape)


class GatherAndView(torch.nn.Module, GatherModuleLike):
    def __init__(self, gather: GatherModuleLike, view: ViewWithPeriod) -> None:
        super().__init__()
        self.gather = gather
        self.view = view

    @unused
    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @unused
    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    @unused
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

    @unused
    def unwrap_final_gather(self) -> tuple[GatherModuleLike, dict[int, int]] | None:
        return None

    @unused
    def get_period(self) -> int | None:
        return self.view.get_period()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        x = self.view(x)
        return x


class LayerGatherAndView(torch.nn.Module, LayerGatherModuleLike):
    def __init__(self, gather: LayerGatherModuleLike, view: ViewWithPeriod) -> None:
        super().__init__()
        self.gather = gather
        self.view = view

    @unused
    @property
    def total_items(self) -> int:
        return self.gather.total_items

    @unused
    @property
    def optimal_period(self) -> int | None:
        return self.gather.optimal_period

    @unused
    @property
    def is_optimal(self) -> bool:
        return self.gather.is_optimal

    @unused
    def get_optimal(self) -> "LayerGatherAndView":
        if self.is_optimal:
            return self

        gather_optimal = self.gather.get_optimal()

        return LayerGatherAndView(
            gather=gather_optimal,
            view=self.view,
        )

    @unused
    def unwrap_final_gather(self) -> tuple[LayerGatherModuleLike, dict[int, int]] | None:
        return None

    @unused
    def get_period(self) -> int | None:
        return self.view.get_period()

    def forward(self, layer_values: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.gather(layer_values)
        x = self.view(x)
        return x


def build_optimal_gather_and_reshape(ordinals: list[int], period: int):
    gather = build_optimal_gather(ordinals, period_hint=period)
    reshape = ViewWithPeriod(input_length=gather.total_items, period=period)
    return GatherAndView(gather, reshape)


def build_optimal_multi_layer_gather_and_reshape(inputs_ordinals: list[LayerOrdinal], period: int):
    gather = build_optimal_multi_layer_gather(inputs_ordinals, period_hint=period)
    reshape = ViewWithPeriod(input_length=gather.total_items, period=period)
    return LayerGatherAndView(gather, reshape)


def get_optimal_gather_for_period(gather: GatherModuleLike, period: int) -> GatherModuleLike:
    if gather.optimal_period == period:
        return gather if gather.is_optimal else gather.get_optimal()
    elif period % gather.optimal_period == 0:
        opt = gather.get_optimal()
        return GatherAndRepeat(opt, repeats=period // opt.total_items, total_length=period)

    return gather


def get_optimal_layer_gather_for_period(gather: LayerGatherModuleLike, period: int) -> LayerGatherModuleLike:
    if gather.optimal_period == period:
        return gather if gather.is_optimal else gather.get_optimal()
    elif period % gather.optimal_period == 0:
        opt = gather.get_optimal()
        return LayerGatherAndRepeat(opt, repeats=period // opt.total_items, total_length=period)

    return gather
