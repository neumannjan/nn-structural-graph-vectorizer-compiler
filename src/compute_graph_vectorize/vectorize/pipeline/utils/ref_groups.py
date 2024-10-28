from abc import ABC, abstractmethod
from typing import Hashable, Iterable, Literal, OrderedDict, Protocol, Sequence, TypeVar, overload

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts

# retrieval


@overload
def get_refs(counts: ComputeLayerCounts, batch: int, source: Refs) -> Refs: ...


@overload
def get_refs(counts: ComputeLayerCounts, batch: int, source: GenericGather | GatheredLayers) -> Sequence[int]: ...


def get_refs(
    counts: ComputeLayerCounts, batch: int, source: Refs | GenericGather | GatheredLayers
) -> Refs | Sequence[int]:
    match source:
        case Refs():
            return source
        case GenericGather():
            return source.ordinals
        case GatheredLayers(gather=GenericGather(ordinals)):
            return ordinals
        case GatheredLayers(refs=refs, gather=NoopGather()):
            return range(counts.compute_layer_refs_count(batch, refs))
        case _:
            assert False, f"{source}"


# transformation


_THashable = TypeVar("_THashable", bound=Hashable)


class GrouperError(Exception):
    pass


class SeqGrouper(Protocol):
    def group(self, seq: Sequence[_THashable]) -> Iterable[tuple[_THashable, ...]]: ...

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]: ...


class EvenGrouper(SeqGrouper):
    def __init__(self, period: int) -> None:
        self.period = period

    def group(self, seq: Iterable[_THashable]) -> Iterable[tuple[_THashable, ...]]:
        it = iter(seq)
        while True:
            try:
                vals = [next(it)]
            except StopIteration:
                break

            try:
                for _ in range(self.period - 1):
                    vals.append(next(it))
            except StopIteration:
                raise GrouperError()

            yield tuple(vals)

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]:
        return [val for tpl in seq for val in tpl]


class UnevenGrouper(SeqGrouper):
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def group(self, seq: Iterable[_THashable]) -> Iterable[tuple[_THashable, ...]]:
        it = iter(seq)

        for c in self.counts:
            if c == 0:
                raise ValueError()

            try:
                vals = [next(it)]
            except StopIteration:
                break

            try:
                for _ in range(c - 1):
                    vals.append(next(it))
            except StopIteration:
                raise GrouperError()

            yield tuple(vals)

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]:
        return [val for tpl in seq for val in tpl]


class NoopGrouper(SeqGrouper):
    def group(self, seq: Sequence[_THashable]) -> Iterable[tuple[_THashable, ...]]:
        return ((v,) for v in seq)

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]:
        return [val for tpl in seq for val in tpl]


class EvenTransposedGrouper(SeqGrouper):
    def __init__(self, period: int) -> None:
        self.period = period
        self._delegate = EvenGrouper(period)

    def _transpose(self, seq: Sequence[_THashable], period: int) -> Iterable[_THashable]:
        for i in range(period):
            yield from seq[i::period]

    def group(self, seq: Sequence[_THashable]) -> Iterable[tuple[_THashable, ...]]:
        return self._delegate.group(self._transpose(seq, period=len(seq) // self.period))

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]:
        return self._transpose(list(self._delegate.ungroup(seq)), period=self.period)


def build_grouper_for_aggregate(aggregate: Reduce) -> SeqGrouper:
    match aggregate:
        case Noop():
            return NoopGrouper()
        case FixedCountReduce(period=period, dim=1):
            return EvenGrouper(period=period)
        case FixedCountReduce(period=period, dim=0):
            return EvenTransposedGrouper(period=period)
        case UnevenReduce(counts=counts):
            return UnevenGrouper(counts)
        case _:
            assert False, f"{aggregate}"


def get_unique_ref_groups(ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]]:
    # out = sorted(set(ref_groups))
    out = sorted(OrderedDict.fromkeys(ref_groups))
    return out


def get_ord_map_for_ref_groups(
    ref_groups: list[tuple[_THashable, ...]], ref_groups_new: list[tuple[_THashable, ...]]
) -> dict[int, int]:
    group_ord_map = {group_ref: o_group_new for o_group_new, group_ref in enumerate(ref_groups_new)}
    ord_map = {}

    for o_group, group_ref in enumerate(ref_groups):
        o_group_new = group_ord_map[group_ref]
        ord_map[o_group] = o_group_new

    return ord_map


def get_unique_ref_groups_with_gather(
    ref_groups: list[tuple[_THashable, ...]],
    gather_per: Literal["per_group", "per_ordinal"],
) -> tuple[list[tuple[_THashable, ...]], GenericGather]:
    ref_groups_uniq = get_unique_ref_groups(ref_groups)

    if gather_per == "per_group":
        group_ord_map = {group_ref: o_group_new for o_group_new, group_ref in enumerate(ref_groups_uniq)}
        final_gather = GenericGather([group_ord_map[group_ref] for group_ref in ref_groups])
    elif gather_per == "per_ordinal":
        group_ord_map = {
            (ref, i): o_new + i for o_new, group_ref in enumerate(ref_groups_uniq) for i, ref in enumerate(group_ref)
        }
        final_gather = GenericGather(
            [group_ord_map[ref, i] for group_ref in ref_groups for i, ref in enumerate(group_ref)]
        )
    else:
        raise ValueError(gather_per)

    return ref_groups_uniq, final_gather


class RefsTransform(Protocol):
    @overload
    def __call__(self, refs: Refs) -> list[tuple[int, str, int]] | None: ...

    @overload
    def __call__(self, refs: Sequence[int]) -> list[int] | None: ...


class SimpleRefGroupsMapTransform(ABC, RefsTransform):
    def __init__(self, grouper: SeqGrouper, with_ord_map: bool) -> None:
        self.grouper = grouper
        self._last_groups: list[tuple] | None = None
        self._ord_map: dict[int, int] | None = None
        self.with_ord_map = with_ord_map

    @abstractmethod
    def get_new_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]] | None:
        pass

    @overload
    def __call__(self, refs: Refs) -> list[tuple[int, str, int]] | None: ...

    @overload
    def __call__(self, refs: Sequence[int]) -> list[int] | None: ...

    def __call__(self, refs: Sequence[_THashable]) -> list[_THashable] | None:
        self._last_groups = None
        self._ord_map = None
        ref_groups = list(self.grouper.group(refs))
        ref_groups_new = self.get_new_ref_groups(ref_groups)

        if ref_groups_new is None:
            return None

        if self.with_ord_map:
            self._ord_map = get_ord_map_for_ref_groups(ref_groups, ref_groups_new)

        refs_out = list(self.grouper.ungroup(ref_groups_new))
        self._last_groups = ref_groups_new
        return refs_out

    @property
    def last_groups(self):
        if self._last_groups is None:
            raise ValueError()
        return self._last_groups

    @property
    def ord_map(self) -> dict[int, int]:
        if self._ord_map is None:
            raise ValueError()
        return self._ord_map


class SimpleUniqueRefsTransform(SimpleRefGroupsMapTransform):
    def get_new_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]] | None:
        ref_groups_new = get_unique_ref_groups(ref_groups)

        if len(ref_groups) == len(ref_groups_new):
            return None

        return ref_groups_new


class SimpleOrderRefsTransform(SimpleRefGroupsMapTransform):
    def __init__(self, grouper: SeqGrouper, with_ord_map: bool) -> None:
        super().__init__(grouper, with_ord_map)

    def get_new_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]] | None:
        out = sorted(ref_groups)
        return out


class CustomRefsTransform(SimpleRefGroupsMapTransform):
    def __init__(
        self, grouper: SeqGrouper, new_refs: Sequence[tuple[int, str, int]] | Sequence[int], with_ord_map: bool
    ) -> None:
        super().__init__(grouper, with_ord_map)
        self._new_refs = new_refs

    def get_new_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]] | None:
        try:
            return list(self.grouper.group(self._new_refs))
        except GrouperError:
            return None

    def __call__(self, refs: Sequence[_THashable]) -> list[_THashable] | None:
        try:
            return super().__call__(refs)  # pyright: ignore
        except KeyError:
            return None


class GatherRefsTransform(RefsTransform):
    def __init__(self, grouper: SeqGrouper, ordinals: list[int]) -> None:
        self.grouper = grouper
        self.ordinals = ordinals
        self._last_groups: list[tuple] | None = None

    def __call__(self, refs: Sequence[_THashable]) -> list[_THashable] | None:
        ref_groups = list(self.grouper.group(refs))
        ref_groups_filtered = [ref_groups[o] for o in self.ordinals]

        refs_out = list(self.grouper.ungroup(ref_groups_filtered))
        self._last_groups = ref_groups_filtered
        return refs_out

    @property
    def last_groups(self):
        if self._last_groups is None:
            raise ValueError()
        return self._last_groups


# assignment


@overload
def apply_refs_to_target(refs: Iterable[tuple[int, str, int]], target: Refs) -> None: ...


@overload
def apply_refs_to_target(refs: Sequence[int], target: GenericGather | GatheredLayers) -> None: ...


@overload
def apply_refs_to_target(
    refs: Iterable[tuple[int, str, int]] | Sequence[int], target: Refs | GenericGather | GatheredLayers
) -> None: ...


def apply_refs_to_target(refs, target: Refs | GenericGather | GatheredLayers) -> None:
    match target:
        case Refs():
            target.set(refs)
        case GenericGather():
            target.ordinals = refs
        case GatheredLayers(gather=GenericGather() as gather):
            gather.ordinals = refs
        case GatheredLayers():
            target.gather = GenericGather(refs)
        case _:
            assert False, f"{target}"


@overload
def get_remapped_refs(
    counts: ComputeLayerCounts, batch: int, source: Refs, transform: RefsTransform
) -> list[tuple[int, str, int]] | None: ...


@overload
def get_remapped_refs(
    counts: ComputeLayerCounts, batch: int, source: GenericGather | GatheredLayers, transform: RefsTransform
) -> list[int] | None: ...


@overload
def get_remapped_refs(
    counts: ComputeLayerCounts, batch: int, source: Refs | GenericGather | GatheredLayers, transform: RefsTransform
) -> list[tuple[int, str, int]] | list[int] | None: ...


def get_remapped_refs(
    counts: ComputeLayerCounts, batch: int, source: Refs | GenericGather | GatheredLayers, transform: RefsTransform
) -> list[tuple[int, str, int]] | list[int] | None:
    refs = get_refs(counts, batch, source)
    refs_transformed = transform(refs)
    return refs_transformed


def remap_refs(
    counts: ComputeLayerCounts, batch: int, source: Refs | GenericGather | GatheredLayers, transform: RefsTransform
) -> bool:
    match source:
        case Refs():
            refs = get_refs(counts, batch, source)
            refs_transformed = transform(refs)
            if refs_transformed is not None:
                apply_refs_to_target(refs_transformed, source)
                return True
            else:
                return False
        case _:
            refs = get_refs(counts, batch, source)
            refs_transformed = transform(refs)
            if refs_transformed is not None:
                apply_refs_to_target(refs_transformed, source)
                return True
            else:
                return False
