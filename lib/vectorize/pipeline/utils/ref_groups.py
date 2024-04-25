from typing import Hashable, Iterable, Literal, OrderedDict, Protocol, Sequence, TypeVar, overload

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts

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
                yield tuple([next(it) for _ in range(self.period)])
            except StopIteration:
                break

    def ungroup(self, seq: Sequence[tuple[_THashable, ...]]) -> Iterable[_THashable]:
        return [val for tpl in seq for val in tpl]


class UnevenGrouper(SeqGrouper):
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def group(self, seq: Iterable[_THashable]) -> Iterable[tuple[_THashable, ...]]:
        it = iter(seq)

        for c in self.counts:
            yield tuple((next(it) for _ in range(c)))

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


def get_unique_ref_groups_with_map(
    ref_groups: list[tuple[_THashable, ...]],
) -> tuple[list[tuple[_THashable, ...]], dict[int, int]]:
    ref_groups_uniq = get_unique_ref_groups(ref_groups)
    group_ord_map = {group_ref: o_group_new for o_group_new, group_ref in enumerate(ref_groups_uniq)}
    ord_map = {}

    for o_group, group_ref in enumerate(ref_groups):
        o_group_new = group_ord_map[group_ref]
        ord_map[o_group] = o_group_new

    return ref_groups_uniq, ord_map


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


class SimpleUniqueRefsTransform(RefsTransform):
    def __init__(self, grouper: SeqGrouper) -> None:
        self.grouper = grouper
        self._last_groups: list[tuple] | None = None

    def get_unique_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]]:
        return get_unique_ref_groups(ref_groups)

    @overload
    def __call__(self, refs: Refs) -> list[tuple[int, str, int]] | None: ...

    @overload
    def __call__(self, refs: Sequence[int]) -> list[int] | None: ...

    def __call__(self, refs: Sequence[_THashable]) -> list[_THashable] | None:
        self._last_groups = None
        ref_groups = list(self.grouper.group(refs))
        ref_groups_uniq = self.get_unique_ref_groups(ref_groups)

        if len(ref_groups) == len(ref_groups_uniq):
            return None

        refs_out = list(self.grouper.ungroup(ref_groups_uniq))
        self._last_groups = ref_groups_uniq
        return refs_out

    @property
    def last_groups(self):
        if self._last_groups is None:
            raise ValueError()
        return self._last_groups


class SimpleUniqueRefsMappableTransform(SimpleUniqueRefsTransform):
    def __init__(self, grouper: SeqGrouper) -> None:
        self.grouper = grouper
        self._ord_map: dict[int, int] | None = None

    def get_unique_ref_groups(self, ref_groups: list[tuple[_THashable, ...]]) -> list[tuple[_THashable, ...]]:
        out, self._ord_map = get_unique_ref_groups_with_map(ref_groups)
        return out

    def __call__(self, refs):
        self._ord_map = None
        return super().__call__(refs)

    @property
    def ord_map(self) -> dict[int, int]:
        if self._ord_map is None:
            raise ValueError()
        return self._ord_map


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
def apply_refs_to_target(refs: Iterable[tuple[int, str, int]], target: Refs): ...


@overload
def apply_refs_to_target(refs: Sequence[int], target: GenericGather | GatheredLayers): ...


def apply_refs_to_target(refs, target: Refs | GenericGather | GatheredLayers):
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
