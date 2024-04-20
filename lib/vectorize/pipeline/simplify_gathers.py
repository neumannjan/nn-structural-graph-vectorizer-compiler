from typing import Container, Generic, Sequence, Type, TypeVar, overload

from lib.utils import detect_repeating_interleaved_sequence_in_list, detect_repeating_sequence_in_list
from lib.vectorize.model import *
from lib.vectorize.model.gather import OneGather
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation

_T = TypeVar("_T")


class _AnyWhitelist(Container[_T], Generic[_T]):
    def __contains__(self, x: object, /) -> bool:
        return True


_ANY_WHITELIST: Container[Type[Gather]] = _AnyWhitelist()


_TGather = TypeVar("_TGather", bound=Gather)


@overload
def build_optimal_gather(
    ordinals: Sequence[int], total_refs_count: int | None, *, whitelist: Container[Type[_TGather]], allow_subseq=True
) -> _TGather: ...


@overload
def build_optimal_gather(ordinals: Sequence[int], total_refs_count: int | None, *, allow_subseq=True) -> Gather: ...


def build_optimal_gather(
    ordinals: Sequence[int],
    total_refs_count: int | None,
    *,
    whitelist: Container[Type[Gather]] | None = None,
    allow_subseq=True,
) -> Gather:
    if whitelist is None:
        whitelist = _ANY_WHITELIST
    assert GenericGather in whitelist

    if len(ordinals) == 0:
        return GenericGather(list(ordinals))

    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        if total_refs_count == 1:
            if NoopGather in whitelist and len(ordinals) == 1:
                return NoopGather()
            elif RepeatInterleave in whitelist:
                return RepeatInterleave(times=len(ordinals), total_length=len(ordinals))
        else:
            if TakeSingleValue in whitelist and len(ordinals) == 1:
                return TakeSingleValue(ordinals[0])
            elif RepeatInterleave in whitelist:
                return RepeatInterleave(times=len(ordinals), total_length=len(ordinals))

    ###### simple slicing #######

    if len(ordinals) >= 2:
        step = ordinals[1] - ordinals[0]
        all_ordinals_differ_by_step = step > 0 and all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

        if all_ordinals_differ_by_step:
            start = ordinals[0]
            end = ordinals[-1] + 1
            if NoopGather in whitelist and step == 1 and start == 0 and end == total_refs_count:
                return NoopGather()
            elif SliceValues in whitelist:
                return SliceValues(step=step, start=start, end=end)

    ###### subsequence with (optimizable) repeat: #######

    if allow_subseq:
        subseq_len = detect_repeating_sequence_in_list(ordinals, allow_last_incomplete=True)
        subseq = ordinals[:subseq_len]

        if subseq_len is not None and subseq_len <= len(ordinals) // 2:
            subseq_gather = build_optimal_gather(
                subseq, total_refs_count=total_refs_count, allow_subseq=False, whitelist=whitelist
            )
            repeats = -(-len(ordinals) // subseq_len)
            total_length = len(ordinals)

            if total_length == subseq_len:
                return subseq_gather

            if Repeat in whitelist:
                match subseq_gather:
                    case NoopGather():
                        return Repeat(times=repeats, total_length=total_length)
                    case _ if GatherPair in whitelist:
                        return GatherPair(subseq_gather, Repeat(times=repeats, total_length=total_length))

        if RepeatInterleave in whitelist:
            repeats = detect_repeating_interleaved_sequence_in_list(ordinals, allow_last_incomplete=True)

            if repeats is not None:
                assert repeats > 1
                subseq = ordinals[::repeats]
                subseq_gather = build_optimal_gather(
                    subseq, total_refs_count=total_refs_count, allow_subseq=False, whitelist=whitelist
                )
                total_length = len(ordinals)

                match subseq_gather:
                    case NoopGather():
                        return RepeatInterleave(times=repeats, total_length=total_length)
                    case _ if GatherPair in whitelist:
                        return GatherPair(subseq_gather, RepeatInterleave(times=repeats, total_length=total_length))

    ###### generic fallback implementation ######
    return GenericGather(list(ordinals))


class SimplifyGathers(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._compute_layer_counts = ComputeLayerCounts(network)

    @overload
    def simplify_gather(self, gather: OneGather, total_refs_count: int | None) -> OneGather: ...

    @overload
    def simplify_gather(self, gather: Gather, total_refs_count: int | None) -> Gather: ...

    def simplify_gather(self, gather: Gather, total_refs_count: int | None) -> Gather:
        match gather:
            case GenericGather(ordinals=ordinals):
                return build_optimal_gather(ordinals, total_refs_count=total_refs_count)
            case GatherPair(a, b):
                a = self.simplify_gather(a, total_refs_count=total_refs_count)
                b = self.simplify_gather(b, total_refs_count=total_refs_count)

                match (a, b):
                    case (NoopGather(), _):
                        return b
                    case (_, NoopGather()):
                        return a
                    case _:
                        return GatherPair(a, b)
            case TakeSingleValue():
                return gather
            case NoopGather():
                return gather
            case SliceValues(start=_, end=_, step=_):
                return gather
            case Repeat(times=_, total_length=_):
                return gather
            case RepeatInterleave(times=_, total_length=_):
                return gather
            case _:
                assert False, f"{gather}"

    def _reorder_refs(self, refs: LayerRefs, ordinals: list[int]):
        refs.types = [refs.types[o] for o in ordinals]
        refs.layer_ids = [refs.layer_ids[o] for o in ordinals]

    def _for_input(self, batch: int, input: Input):
        match input:
            case GatheredLayers(refs=refs):
                total_count = self._compute_layer_counts.compute_layer_refs_count(batch, refs)

                input.gather = self.simplify_gather(input.gather, total_refs_count=total_count)

                # if the gather remained non-simple (generic), then if the total no. of ordinals
                # is low and the references are trivial, we may just preorder the references themselves
                # TODO parametrize the len() threshold
                if isinstance(input.gather, GenericGather) and len(input.gather.ordinals) <= 20:
                    refs_len1 = all((v == 1 for v in self._compute_layer_counts.iter_layer_refs_counts(batch, refs)))
                    if refs_len1:
                        self._reorder_refs(refs, input.gather.ordinals)
                        input.gather = NoopGather()
            case _:
                assert False, f"{input}"

    def _for_layer_base(self, batch: int, base: LayerBase) -> LayerBase:
        match base:
            case InputLayerBase(input=input):
                self._for_input(batch, input)
                return base
            case LinearLayerBase(input=input, weight=weight):
                self._for_input(batch, input)
                self._for_input(batch, weight)
                return base
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather, lifts=lifts):
                self._for_input(batch, input)
                self._for_input(batch, weight)
                total_count = self._compute_layer_counts.compute_linear_count(batch, input, weight, lifts)
                gather = self.simplify_gather(gather, total_refs_count=total_count)
                if isinstance(gather, NoopGather):
                    return LinearLayerBase(input, weight, lifts=lifts)
                else:
                    return LinearGatherLayerBase(input, weight, gather, lifts=lifts)
            case _:
                assert False

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.base = self._for_layer_base(batch, layer.base)
        return layer

    def simplify_gathers(self):
        for bid, batch in self.network.batches.items():
            for layer in batch.layers.values():
                layer.base = self._for_layer_base(bid, layer.base)


def simplify_gathers(network: VectorizedLayerNetwork):
    SimplifyGathers(network).simplify_gathers()
    return network
