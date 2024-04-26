import itertools
import math
from typing import Collection, Iterable, Iterator, Literal, Protocol, Sequence, TypeVar

from lib.utils import head_and_rest
from lib.vectorize.model import *
from lib.vectorize.model.settings import LinearsPadForSymmetriesOption
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation
from lib.vectorize.pipeline.lift_symmetrical_linears import LiftSymmetricalLinears
from lib.vectorize.pipeline.utils.gather import combine_gathers_
from lib.vectorize.pipeline.utils.ref_groups import apply_refs_to_target

_T = TypeVar("_T")


def _interleave(refs: Iterable[_T], repeats: int):
    for ref in refs:
        for i in range(repeats):
            yield ref


def _repeat(refs: Iterable[_T], repeats: int):
    for i in range(repeats):
        for ref in refs:
            yield ref


def _extend_mirror(refs: Iterable[_T], repeats: int, transpose: bool):
    if transpose:
        return _interleave(refs, repeats)
    else:
        return _repeat(refs, repeats)


def _extend(refs: Iterable[_T], total: int):
    head, refs = head_and_rest(refs)

    i = 0
    yield head
    i += 1

    for ref in refs:
        yield ref
        i += 1

    while i < total:
        yield head
        i += 1


def _extend_for_each(
    refs_uniq: Iterable[_T], unique_pairs: Collection[tuple[_T, _T]], total_each: int, key_axis: int, transpose: bool
):
    iters = [
        (lambda ref: _extend((pair[1 - key_axis] for pair in unique_pairs if pair[key_axis] == ref), total=total_each))(
            ref
        )
        for ref in refs_uniq
    ]

    if transpose:
        for it in iters:
            for i in range(total_each):
                yield next(it)
    else:
        for i in range(total_each):
            for it in iters:
                yield next(it)


_POS_WEIGHT_REFS = 0
_POS_REFS = 1


def get_pairs(refs: Iterable[_T], weight_refs: Iterable[_T]):
    return zip(weight_refs, refs)


def get_unique(values: Iterable[_T]) -> list[_T]:
    # out = list(OrderedDict.fromkeys(values))
    out = sorted(set(values))  # pyright: ignore
    return out


def _refs_group_by_counts(sources: Sequence[Iterator[_T]], counts: Sequence[int]) -> Iterator[_T]:
    last_value = None
    while True:
        for src, cnt in zip(sources, counts):
            vs = list(itertools.islice(src, cnt))
            if len(vs) > 0:
                last_value = vs[-1]
            yield from vs
            if len(vs) < cnt:
                for _ in range(cnt - len(vs)):
                    assert last_value is not None
                    yield last_value


def _refs_repeat_by_counts(source_values: Sequence[_T], counts: Sequence[int]) -> Iterator[_T]:
    while True:
        for src, cnt in zip(source_values, counts):
            for _ in range(cnt):
                yield src


class _PadStrategy(Protocol):
    @property
    def computational_cost(self) -> tuple[int, int, int]: ...

    @property
    def is_applicable(self) -> bool: ...

    def build_refs(self) -> tuple[Refs, Refs]: ...


_Ref = tuple[int, str, int]


class _BucketPadStrategy(_PadStrategy):
    def __init__(
        self,
        unique_pairs: list[tuple[_Ref, _Ref]],
        bucketing_refs_uniq: list[_Ref],
        bucketing_refs_axis: Literal[0, 1],
        max_refs_uniq: int,
    ) -> None:
        self.unique_pairs = unique_pairs
        self.bucketing_refs_uniq = bucketing_refs_uniq
        self.axis: Literal[0, 1] = bucketing_refs_axis
        self.max_refs_uniq = max_refs_uniq

    def _get_bref_grp_counts(self):
        refs_per_bref = [
            [p[1 - self.axis] for p in self.unique_pairs if p[self.axis] == bref] for bref in self.bucketing_refs_uniq
        ]

        min_bref_refs_len = min((len(v) for v in refs_per_bref))
        bref_grp_counts = [len(refs) // min_bref_refs_len for refs in refs_per_bref]

        min_bref_refs_len_padded = min_bref_refs_len + max(
            math.ceil((len(refs_this) - min_bref_refs_len * cnt) / cnt)
            for refs_this, cnt in zip(refs_per_bref, bref_grp_counts)
        )

        total_values = sum(bref_grp_counts) * min_bref_refs_len_padded
        return refs_per_bref, bref_grp_counts, total_values

    @property
    def computational_cost(self) -> tuple[int, int, int]:
        _, bref_grp_counts, total_values = self._get_bref_grp_counts()

        gather1 = sum(bref_grp_counts)
        gather2 = total_values
        linear = total_values

        return gather1, gather2, linear

    @property
    def is_applicable(self) -> bool:
        return len(self.bucketing_refs_uniq) <= self.max_refs_uniq

    def build_refs(self) -> tuple[Refs, Refs]:
        refs_per_bref, bref_grp_counts, total_values = self._get_bref_grp_counts()

        refs_out = Refs.from_iter(
            itertools.islice(
                _refs_group_by_counts(
                    [iter(refs_this) for refs_this in refs_per_bref],
                    bref_grp_counts,
                ),
                total_values,
            )
        )

        brefs_out = Refs.from_iter(
            itertools.islice(
                _refs_repeat_by_counts(
                    self.bucketing_refs_uniq,
                    bref_grp_counts,
                ),
                total_values,
            )
        )

        # TODO do something with the aggregate as well?

        if self.axis == _POS_WEIGHT_REFS:
            return refs_out, brefs_out
        elif self.axis == _POS_REFS:
            return brefs_out, refs_out
        else:
            raise ValueError(self.axis)


class _FullPadStrategy(_PadStrategy):
    def __init__(self, refs_uniq: list[_Ref], wrefs_uniq: list[_Ref], transpose: bool) -> None:
        self.refs_uniq = refs_uniq
        self.wrefs_uniq = wrefs_uniq
        self.transpose = transpose

    @property
    def computational_cost(self) -> tuple[int, int, int]:
        gather1 = len(self.refs_uniq)
        gather2 = len(self.wrefs_uniq)
        linear = gather1 * gather2
        return gather1, gather2, linear

    @property
    def is_applicable(self) -> bool:
        return True

    def build_refs(self) -> tuple[Refs, Refs]:
        if self.transpose:
            refs_out = Refs.from_iter(_repeat(self.refs_uniq, repeats=len(self.wrefs_uniq)))
            wrefs_out = Refs.from_iter(_interleave(self.wrefs_uniq, repeats=len(self.refs_uniq)))
        else:
            refs_out = Refs.from_iter(_interleave(self.refs_uniq, repeats=len(self.wrefs_uniq)))
            wrefs_out = Refs.from_iter(_repeat(self.wrefs_uniq, repeats=len(self.refs_uniq)))

        return refs_out, wrefs_out


class _FullOneSidePadStrategy(_PadStrategy):
    def __init__(
        self,
        unique_pairs: list[tuple[_Ref, _Ref]],
        bucketing_refs_uniq: list[_Ref],
        bucketing_refs_axis: Literal[0, 1],
        transpose: bool,
    ) -> None:
        self.unique_pairs = unique_pairs
        self.bucketing_refs_uniq = bucketing_refs_uniq
        self.axis: Literal[0, 1] = bucketing_refs_axis
        self.transpose = transpose

    @property
    def computational_cost(self) -> tuple[int, int, int]:
        brefs_reshaped_uses_max = max(
            sum((1 for p in self.unique_pairs if p[self.axis] == bref)) for bref in self.bucketing_refs_uniq
        )

        gather1 = len(self.bucketing_refs_uniq) * brefs_reshaped_uses_max
        gather2 = len(self.bucketing_refs_uniq)
        linear = gather1

        return gather1, gather2, linear

    @property
    def is_applicable(self) -> bool:
        return True

    def build_refs(self) -> tuple[Refs, Refs]:
        brefs_reshaped_uses_max = max(
            sum((1 for p in self.unique_pairs if p[self.axis] == bref)) for bref in self.bucketing_refs_uniq
        )

        refs_out = Refs.from_iter(
            _extend_for_each(
                self.bucketing_refs_uniq,
                self.unique_pairs,
                total_each=brefs_reshaped_uses_max,
                key_axis=self.axis,
                transpose=self.transpose,
            )
        )

        brefs_out = Refs.from_iter(
            _extend_mirror(self.bucketing_refs_uniq, repeats=brefs_reshaped_uses_max, transpose=self.transpose)
        )

        if self.axis == _POS_WEIGHT_REFS:
            return refs_out, brefs_out
        elif self.axis == _POS_REFS:
            return brefs_out, refs_out
        else:
            raise ValueError(self.axis)


class OptimizeLinearsPadForSymmetries(LayerwiseOperation):
    def __init__(
        self,
        network: VectorizedLayerNetwork,
        how: LinearsPadForSymmetriesOption,
        transpose: bool,
        max_refs_nogather_uniq: int,
    ) -> None:
        self.network = network
        self.how: LinearsPadForSymmetriesOption = how
        self.transpose = transpose
        self.max_refs_nogather_uniq = max_refs_nogather_uniq
        self._counts = ComputeLayerCounts(network)
        self._dim_lift = LiftSymmetricalLinears(network)

    def _get_padded_refs(
        self, batch: int, layer_id: str, refs: Refs, wrefs: Refs, expected_period: int | None
    ) -> tuple[Refs, Refs] | None:
        if self.how == "never":
            return None

        assert len(refs) == len(wrefs)

        refs_uniq = get_unique(refs)
        wrefs_uniq = get_unique(wrefs)

        unique_pairs = get_unique(get_pairs(refs, wrefs))

        if len(unique_pairs) < len(refs):
            # The references aren't unique. No point in padding this.
            return None

        if len(unique_pairs) >= len(refs_uniq) * len(wrefs_uniq):
            # The references are already full (we have each with each at least). No padding needed.
            return None

        assert len(wrefs_uniq) < len(refs_uniq)  # TODO generalize

        # inputs gather + weights gather + linear
        original_cost: Iterable[int] = len(refs), len(refs), len(refs)

        strategies: Sequence[tuple[Iterable[int], _PadStrategy]] = [
            (strategy.computational_cost, strategy)
            for strategy in (
                _FullPadStrategy(refs_uniq, wrefs_uniq, transpose=self.transpose) if self.how != "sided_only" else None,
                _FullOneSidePadStrategy(
                    unique_pairs,
                    bucketing_refs_uniq=wrefs_uniq,
                    bucketing_refs_axis=_POS_WEIGHT_REFS,
                    transpose=self.transpose,
                ) if self.how != "full_only" else None,
                _FullOneSidePadStrategy(
                    unique_pairs,
                    bucketing_refs_uniq=refs_uniq,
                    bucketing_refs_axis=_POS_REFS,
                    transpose=self.transpose,
                ) if self.how != "full_only" else None,
                _BucketPadStrategy(
                    unique_pairs,
                    bucketing_refs_uniq=wrefs_uniq,
                    bucketing_refs_axis=_POS_WEIGHT_REFS,
                    max_refs_uniq=self.max_refs_nogather_uniq,
                ) if self.how != "full_only" else None,
                _BucketPadStrategy(
                    unique_pairs,
                    bucketing_refs_uniq=refs_uniq,
                    bucketing_refs_axis=_POS_REFS,
                    max_refs_uniq=self.max_refs_nogather_uniq,
                ) if self.how != "full_only" else None,
            )
            if strategy is not None and strategy.is_applicable
        ]

        min_strategy_cost, min_strategy = min(strategies, key=lambda v: sum(v[0]))

        print(f"PADDING {layer_id} strategies:", [(s.__class__.__name__, cost) for cost, s in strategies])
        print("ORIGINAL cost:", original_cost)

        _, (refs_exp_orig, wrefs_exp_orig) = self._dim_lift.detect_dim_lift(
            batch, refs, wrefs, existing_lifts=None, preferred_period=expected_period
        )

        expected_original_cost = len(refs_exp_orig), len(wrefs_exp_orig), len(refs_exp_orig) * len(wrefs_exp_orig)

        if sum(min_strategy_cost) > sum(expected_original_cost):
            return None

        print(f"PADDING {layer_id} using {min_strategy.__class__.__name__}:")
        refs_out, wrefs_out = min_strategy.build_refs()
        print(f"{len(refs)} -> {len(refs_out)}")
        print(f"{len(wrefs)} -> {len(wrefs_out)}")
        return refs_out, wrefs_out

    def _remap(
        self, batch: int, layer_id: str, refs: Refs, wrefs: Refs, expected_period: int | None
    ) -> GenericGather | None:
        padded = self._get_padded_refs(batch, layer_id, refs, wrefs, expected_period)

        if padded is None:
            return None

        new_refs, new_wrefs = padded

        new_pairs_idx_map = {}

        for i, pair in enumerate(zip(new_refs, new_wrefs)):
            if pair not in new_pairs_idx_map:
                new_pairs_idx_map[pair] = i

        final_gather = GenericGather([new_pairs_idx_map[pair] for pair in zip(refs, wrefs)])

        apply_refs_to_target(new_refs, refs)
        apply_refs_to_target(new_wrefs, wrefs)
        return final_gather

    def _get_expected_period(self, aggregate: Reduce) -> int | None:
        match aggregate:
            case FixedCountReduce(period=period):
                return period
            case UnevenReduce():
                return None
            case Noop():
                return None
            case _:
                assert False, f"{aggregate}"

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if self.how == "never":
            return layer

        match layer:
            case Layer(base=InputLayerBase(input=input)):
                # nothing to do
                return layer
            case Layer(
                base=(
                    LinearLayerBase(input=Refs() as input, weight=Refs() as weight, lifts=lifts)
                    | LinearGatherLayerBase(
                        input=Refs() as input, weight=Refs() as weight, gather=NoopGather(), lifts=lifts
                    )
                )
            ):
                if lifts is not None:
                    return layer

                period = self._get_expected_period(layer.aggregate)

                final_gather = self._remap(batch, layer_id, input, weight, expected_period=period)
                if final_gather is not None:
                    layer.base = LinearGatherLayerBase(input=input, weight=weight, gather=final_gather, lifts=None)
                return layer
            case Layer(
                base=LinearGatherLayerBase(
                    input=Refs() as input, weight=Refs() as weight, gather=GenericGather() as gather2, lifts=lifts
                )
            ):
                if lifts is not None:
                    return layer

                period = self._get_expected_period(layer.aggregate)

                gather1 = self._remap(batch, layer_id, input, weight, expected_period=period)
                if gather1 is not None:
                    combine_gathers_(gather1, gather2)
                return layer
            case _:
                raise ValueError(layer.base)


def build_optimize_linears_pad_for_symmetries(
    how: LinearsPadForSymmetriesOption, transpose: bool, max_refs_nogather_uniq: int
):
    def _init(network: VectorizedLayerNetwork):
        return OptimizeLinearsPadForSymmetries(
            network,
            how=how,
            transpose=transpose,
            max_refs_nogather_uniq=max_refs_nogather_uniq,
        )

    return _init
