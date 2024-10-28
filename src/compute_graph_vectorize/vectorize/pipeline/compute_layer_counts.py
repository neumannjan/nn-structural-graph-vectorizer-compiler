import functools
import math
from operator import mul
from typing import Collection, Iterable, Iterator, Mapping

from compute_graph_vectorize.utils import MapMapping
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.model.layer import DimensionLifts, lifts_dimension_match
from compute_graph_vectorize.vectorize.pipeline.layerwise import LayerwiseOperation


def compute_fact_count(fact: Fact) -> int:
    match fact:
        case UnitFact():
            return 1
        case EyeFact():
            return 1
        case ValueFact(value=value):
            return value.shape[0]
        case _:
            assert False


def compute_weight_count(weight: LearnableWeight) -> int:
    return weight.value.shape[0]


def compute_gather_count(in_count: int, gather: Gather) -> int:
    match gather:
        case GenericGather(ordinals=ordinals):
            return len(ordinals)
        case TakeSingleValue(ordinal=_):
            return 1
        case NoopGather():
            return in_count
        case SliceValues(start=start, end=end, step=step):
            return -(-(end - start) // step)
        case Repeat(times=_, total_length=total_length):
            return total_length
        case RepeatInterleave(times=_, total_length=total_length):
            return total_length
        case GatherPair(a, b):
            count = in_count
            count = compute_gather_count(count, a)
            count = compute_gather_count(count, b)
            return count
        case _:
            assert False


def compute_aggregation_count(in_count: int, agg: Reduce) -> int:
    match agg:
        case FixedCountReduce(period=period, reduce=_):
            return in_count // period
        case UnevenReduce(counts=counts, reduce=_):
            return len(counts)
        case Noop():
            return in_count
        case _:
            assert False


def compute_facts_count(facts: Collection[Fact]) -> int:
    return sum((compute_fact_count(f) for f in facts))


def compute_ref_count(network: VectorizedLayerNetwork | VectorizedOpSeqNetwork, ref: tuple[int, str, int]):
    t, l, o = ref
    if t == Refs.TYPE_FACT:
        return compute_fact_count(network.fact_layers[l].facts[o])
    elif t == Refs.TYPE_WEIGHT:
        return compute_weight_count(network.weights[l])
    else:
        return 1


def iter_ref_counts(network: VectorizedLayerNetwork | VectorizedOpSeqNetwork, refs: Refs) -> Iterable[int]:
    return (compute_ref_count(network, ref) for ref in refs)


def compute_refs_count(network: VectorizedLayerNetwork | VectorizedOpSeqNetwork, refs: Refs) -> int:
    return sum(iter_ref_counts(network, refs))


def compute_layer_ref_count(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    type: int,
    id: str,
) -> int:
    match type:
        case LayerRefs.TYPE_FACT:
            cnt = network.fact_layers[id].count
            assert cnt is not None
            return cnt
        case LayerRefs.TYPE_WEIGHT:
            return compute_weight_count(network.weights[id])
        case LayerRefs.TYPE_LAYER:
            if batch_layer_counts is not None:
                cnt = batch_layer_counts[id]
                if cnt is not None:
                    return cnt

            match network:
                case VectorizedLayerNetwork():
                    cnt = compute_layer_count(network, batch, batch_layer_counts, network.batches[batch].layers[id])
                case VectorizedOpSeqNetwork():
                    cnt = compute_operation_seq_count(
                        network, batch, batch_layer_counts, network.batches[batch].layers[id]
                    )
                case _:
                    raise ValueError(network)
            return cnt
        case _:
            raise ValueError(type)


def iter_layer_refs_counts(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    refs: LayerRefs,
) -> Iterable[int]:
    for t, id in refs:
        yield compute_layer_ref_count(network, batch, batch_layer_counts, t, id)


def compute_layer_refs_count(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    refs: LayerRefs,
) -> int:
    return sum(iter_layer_refs_counts(network, batch, batch_layer_counts, refs))


def compute_input_count(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    input: Input,
) -> int:
    match input:
        case Refs() as refs:
            count = compute_refs_count(network, refs)
        case GatheredLayers(refs=refs, gather=gather):
            count = compute_layer_refs_count(network, batch, batch_layer_counts, refs)
            count = compute_gather_count(count, gather)
        case _:
            assert False, f"{input}"
    return count


def compute_lifted_count(input_count: int, weight_count: int, lifts: DimensionLifts) -> int:
    if lifts is None or lifts_dimension_match(lifts):
        return math.lcm(weight_count, input_count)

    (a0, a1), (b0, b1) = lifts
    return max(a0, b0) * max(a1, b1)


def compute_linear_count(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    input: Input,
    weight: Input,
    lifts: DimensionLifts,
) -> int:
    weight_count = compute_input_count(network, batch, batch_layer_counts, weight)
    input_count = compute_input_count(network, batch, batch_layer_counts, input)
    return compute_lifted_count(input_count, weight_count, lifts)


def compute_layer_base_count(
    network: VectorizedLayerNetwork | VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    base: LayerBase,
) -> int:
    match base:
        case InputLayerBase(input=gathered_source):
            return compute_input_count(network, batch, batch_layer_counts, gathered_source)
        case LinearLayerBase(input=input, weight=weight, lifts=lifts):
            return compute_linear_count(network, batch, batch_layer_counts, input, weight, lifts)
        case LinearGatherLayerBase(input=input, weight=weight, gather=gather, lifts=lifts):
            count = compute_linear_count(network, batch, batch_layer_counts, input, weight, lifts)
            count = compute_gather_count(count, gather)
            return count
        case _:
            assert False


def compute_operation_shape(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    in_shape: tuple[int, ...],
    skip_dims: int,
    op: Operation,
) -> tuple[int, ...]:
    match op:
        case Linear(weight_ops=weight_ops):
            w_shp = compute_operation_seq_shape(network, batch, batch_layer_counts, weight_ops)
            assert len(in_shape) == len(w_shp)
            return tuple((max(a, b) for a, b in zip(in_shape, w_shp)))
        case _ if isinstance(op, Gather):
            return compute_gather_count(in_shape[0], op), *in_shape[1:]
        case Transform():
            return in_shape
        case DimReduce(dim=dim):
            return *in_shape[:dim], *in_shape[dim + 1 :]
        case UnevenReduce(counts=counts):
            return len(counts), *in_shape[1:]
        case View(shape=shape):
            out_shape = shape.dims[:-skip_dims]
            last_dim_cnt = functools.reduce(mul, in_shape, 1)
            for v in out_shape:
                if v > 0:
                    last_dim_cnt //= v

            out_shape = tuple((last_dim_cnt if v == -1 else v for v in out_shape))
            return out_shape
        case _:
            assert False, f"{op}"


def iter_operation_seq_shapes(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    seq: OperationSeq,
) -> Iterator[tuple[int, ...]]:
    skip_dims = 2

    init_cnt = compute_layer_refs_count(network, batch, batch_layer_counts, seq.layer_refs)
    shp = (init_cnt,)
    yield shp

    for op in seq:
        shp = compute_operation_shape(network, batch, batch_layer_counts, in_shape=shp, skip_dims=skip_dims, op=op)
        yield shp


def compute_operation_seq_shape(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    seq: OperationSeq,
) -> tuple[int, ...]:
    for shp in iter_operation_seq_shapes(network, batch, batch_layer_counts, seq):
        pass

    return shp


def compute_operation_seq_count(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    seq: OperationSeq,
) -> int:
    return compute_operation_seq_shape(network, batch, batch_layer_counts, seq)[0]


def compute_layer_count(
    network: VectorizedLayerNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    layer: Layer,
) -> int:
    count = compute_layer_base_count(network, batch, batch_layer_counts, layer.base)
    count = compute_aggregation_count(count, layer.aggregate)
    return count


def compute_op_seq_network_layer_counts(network: VectorizedOpSeqNetwork, batch_id: int):
    batch = network.batches[batch_id]

    batch_layer_counts: dict[str, int] = {}

    for layer_id, op_seq in batch.layers.items():
        cnt = compute_operation_seq_count(network, batch_id, batch_layer_counts, op_seq)
        batch_layer_counts[layer_id] = cnt

    return batch_layer_counts


class ComputeLayerCounts(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def compute_fact_count(self, fact: Fact) -> int:
        return compute_fact_count(fact)

    def compute_weight_count(self, weight: LearnableWeight) -> int:
        return compute_weight_count(weight)

    def compute_gather_count(self, in_count: int, gather: Gather) -> int:
        return compute_gather_count(in_count, gather)

    def compute_aggregation_count(self, in_count: int, agg: Reduce) -> int:
        return compute_aggregation_count(in_count, agg)

    def compute_facts_count(self, facts: Collection[Fact]) -> int:
        return compute_facts_count(facts)

    def compute_ref_count(self, ref: tuple[int, str, int]):
        return compute_ref_count(self.network, ref)

    def iter_ref_counts(self, refs: Refs) -> Iterable[int]:
        return iter_ref_counts(self.network, refs)

    def compute_refs_count(self, refs: Refs) -> int:
        return compute_refs_count(self.network, refs)

    def _batch_layer_counts(self, batch_id: int) -> Mapping[str, int | None]:
        batch = self.network.batches[batch_id]
        out = MapMapping(lambda v: v.count, batch.layers)
        return out

    def compute_layer_ref_count(
        self,
        batch: int,
        type: int,
        id: str,
        layers_fresh=False,
    ) -> int:
        return compute_layer_ref_count(
            self.network, batch, None if layers_fresh else self._batch_layer_counts(batch), type, id
        )

    def iter_layer_refs_counts(self, batch: int, refs: LayerRefs, layers_fresh=False) -> Iterable[int]:
        return iter_layer_refs_counts(
            self.network, batch, None if layers_fresh else self._batch_layer_counts(batch), refs
        )

    def compute_layer_refs_count(self, batch: int, refs: LayerRefs) -> int:
        return compute_layer_refs_count(self.network, batch, self._batch_layer_counts(batch), refs)

    def compute_input_count(self, batch: int, input: Input) -> int:
        return compute_input_count(self.network, batch, self._batch_layer_counts(batch), input)

    def compute_lifted_count(self, input_count: int, weight_count: int, lifts: DimensionLifts) -> int:
        return compute_lifted_count(input_count, weight_count, lifts)

    def compute_linear_count(self, batch: int, input: Input, weight: Input, lifts: DimensionLifts) -> int:
        return compute_linear_count(self.network, batch, self._batch_layer_counts(batch), input, weight, lifts)

    def compute_layer_base_count(self, batch: int, base: LayerBase) -> int:
        return compute_layer_base_count(self.network, batch, self._batch_layer_counts(batch), base)

    def compute_layer_count(self, batch: int, layer: Layer) -> int:
        return compute_layer_count(self.network, batch, self._batch_layer_counts(batch), layer)

    def compute_counts(self):
        for layer in self.network.fact_layers.values():
            layer.count = sum((compute_fact_count(f) for f in layer.facts))

        for bid, batch in self.network.batches.items():
            try:
                for lid, layer in batch.layers.items():
                    layer.count = self.compute_layer_count(bid, layer)
            except Exception as e:
                raise Exception(f"Exception in batch {bid}, layer {lid}") from e

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        layer.count = self.compute_layer_count(batch, layer)
        return layer


def compute_layer_counts(network: VectorizedLayerNetwork):
    ComputeLayerCounts(network).compute_counts()
    return network
