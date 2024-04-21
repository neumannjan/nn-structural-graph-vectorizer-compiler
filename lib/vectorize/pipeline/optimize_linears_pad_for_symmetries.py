from typing import Callable, Collection, Iterable, OrderedDict, TypeVar

from lib.utils import head_and_rest
from lib.vectorize.model import *
from lib.vectorize.model.settings import LinearsPadForSymmetriesOption
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation
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


def _extend_for_each(refs_uniq: Iterable[_T], unique_pairs: Collection[tuple[_T, _T]], total_each: int, key_axis: int):
    iters = [
        (lambda ref: _extend((pair[1 - key_axis] for pair in unique_pairs if pair[key_axis] == ref), total=total_each))(
            ref
        )
        for ref in refs_uniq
    ]

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


class OptimizeLinearsPadForSymmetries(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork, how: LinearsPadForSymmetriesOption) -> None:
        self.network = network
        self.how: LinearsPadForSymmetriesOption = how
        self._counts = ComputeLayerCounts(network)

    def _get_padded_refs(self, batch: int, refs: Refs, wrefs: Refs) -> tuple[Refs, Refs] | None:
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

        # gather + gather + linear

        refs_reshaped_uses_max = max(sum((1 for pair in unique_pairs if pair[_POS_REFS] == ref)) for ref in refs_uniq)
        wrefs_reshaped_uses_max = max(
            sum((1 for pair in unique_pairs if pair[_POS_WEIGHT_REFS] == wref)) for wref in wrefs_uniq
        )

        # inputs gather + weights gather + linear
        original_compute = len(refs) * 3
        full_reshaped_compute = len(refs_uniq) + len(wrefs_uniq) + len(refs_uniq) * len(wrefs_uniq)
        refs_reshaped_compute = (
            len(refs_uniq) + len(refs_uniq) * refs_reshaped_uses_max + len(refs_uniq) * refs_reshaped_uses_max
        )
        wrefs_reshaped_compute = (
            len(wrefs_uniq) * wrefs_reshaped_uses_max + len(wrefs_uniq) + len(wrefs_uniq) * wrefs_reshaped_uses_max
        )

        variants: list[tuple[int, Callable[[], tuple[Refs, Refs]]]] = [
            (
                full_reshaped_compute,
                lambda: (
                    Refs.from_iter(_interleave(refs_uniq, repeats=len(wrefs_uniq))),
                    Refs.from_iter(_repeat(wrefs_uniq, repeats=len(refs_uniq))),
                ),
            ),
            (
                refs_reshaped_compute,
                lambda: (
                    Refs.from_iter(_repeat(refs_uniq, repeats=refs_reshaped_uses_max)),
                    Refs.from_iter(
                        _extend_for_each(refs_uniq, unique_pairs, total_each=refs_reshaped_uses_max, key_axis=_POS_REFS)
                    ),
                ),
            ),
            (
                wrefs_reshaped_compute,
                lambda: (
                    Refs.from_iter(
                        _extend_for_each(
                            wrefs_uniq, unique_pairs, total_each=wrefs_reshaped_uses_max, key_axis=_POS_WEIGHT_REFS
                        )
                    ),
                    Refs.from_iter(_repeat(wrefs_uniq, repeats=wrefs_reshaped_uses_max)),
                ),
            ),
        ]

        if self.how == "always_full":
            return variants[0][1]()
        elif self.how == "always_inputs_only":
            return variants[1][1]()
        elif self.how == "always_weights_only":
            return variants[2][1]()

        variant_compute, variant = min(variants, key=lambda v: v[0])

        if self.how == "by_count" and variant_compute > original_compute:
            return None

        return variant()

    def _remap(self, how: LinearsPadForSymmetriesOption, batch: int, refs: Refs, wrefs: Refs) -> GenericGather | None:
        padded = self._get_padded_refs(batch, refs, wrefs)

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

                final_gather = self._remap(self.how, batch, input, weight)
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

                gather1 = self._remap(self.how, batch, input, weight)
                if gather1 is not None:
                    combine_gathers_(gather1, gather2)
                return layer
            case _:
                raise ValueError(layer.base)


def build_optimize_linears_pad_for_symmetries(how: LinearsPadForSymmetriesOption):
    def _init(network: VectorizedLayerNetwork):
        return OptimizeLinearsPadForSymmetries(network, how=how)

    return _init
