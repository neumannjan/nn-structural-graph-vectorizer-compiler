from typing import overload

from lib.utils import detect_repeating_sequence_in_list
from lib.vectorize.model import *
from lib.vectorize.model.gather import OneGather
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


def build_optimal_gather(ordinals: list[int], total_size: int | None, allow_subseq=True) -> Gather:
    all_inputs_the_same = all((ordinals[0] == o for o in ordinals[1:]))

    if all_inputs_the_same:
        if total_size == 1:
            return NoopGather()
        return TakeSingleValue(ordinals[0])

    ###### simple slicing #######

    step = ordinals[1] - ordinals[0]
    all_ordinals_differ_by_step = all((b - a == step for a, b in zip(ordinals[:-1], ordinals[1:])))

    if all_ordinals_differ_by_step:
        start = ordinals[0]
        end = ordinals[-1] + 1
        if step == 1 and start == 0 and end == total_size:
            return NoopGather()
        return SliceValues(step=step, start=start, end=end)

    ###### subsequence with (optimizable) repeat: #######

    if allow_subseq:
        subseq = None
        if subseq is None:
            subseq = detect_repeating_sequence_in_list(ordinals, allow_last_incomplete=True)

        if subseq is not None and len(subseq) <= len(ordinals) // 2:
            subseq_gather = build_optimal_gather(subseq.tolist(), total_size=total_size, allow_subseq=False)
            repeats = -(-len(ordinals) // len(subseq))
            total_length = len(ordinals)

            if total_length == len(subseq):
                return subseq_gather

            return GatherPair(subseq_gather, Repeat(times=repeats, total_length=total_length))

    ###### generic fallback implementation ######

    return GenericGather(ordinals)


class SimplifyGathers(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._compute_layer_counts = ComputeLayerCounts(network)

    @overload
    def simplify_gather(self, gather: OneGather, total_size: int | None) -> OneGather: ...

    @overload
    def simplify_gather(self, gather: Gather, total_size: int | None) -> Gather: ...

    def simplify_gather(self, gather: Gather, total_size: int | None) -> Gather:
        match gather:
            case GenericGather(ordinals=ordinals):
                return build_optimal_gather(ordinals, total_size=total_size)
            case GatherPair(a, b):
                a = self.simplify_gather(a, total_size=total_size)
                b = self.simplify_gather(b, total_size=total_size)
                return GatherPair(a, b)
            case TakeSingleValue():
                return gather
            case NoopGather():
                return gather
            case SliceValues(start=start, end=end, step=step):
                return gather
            case Repeat(times=_, total_length=total_length):
                return gather
            case _:
                assert False, f"{gather}"

    def _get_lrefs_size(self, batch: int, lrefs: LayerRefs):
        return self._compute_layer_counts.compute_layer_refs_count(batch, lrefs)

    def _for_input(self, batch: int, input: Input):
        match input:
            case GatheredLayers(refs=refs, gather=gather):
                total_size = self._get_lrefs_size(batch, refs)
                input.gather = self.simplify_gather(gather, total_size=total_size)
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
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                self._for_input(batch, input)
                self._for_input(batch, weight)
                total_size = self._compute_layer_counts.compute_linear_count(batch, input, weight)
                return LinearGatherLayerBase(input, weight, self.simplify_gather(gather, total_size=total_size))
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
