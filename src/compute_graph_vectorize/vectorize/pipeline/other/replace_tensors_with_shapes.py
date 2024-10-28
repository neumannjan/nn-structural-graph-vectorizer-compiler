import copy
from collections import OrderedDict
from typing import TypeVar, overload

import numpy as np

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import compute_layer_refs_count


def _for_fact_layer(fact_layer: FactLayer) -> FactLayer:
    return FactLayer([], count=fact_layer.count, shape=copy.deepcopy(fact_layer.shape))


def _for_weight(weight: LearnableWeight) -> LearnableWeight:
    return LearnableWeight(value=np.array(weight.value.shape))


_TOp = TypeVar("_TOp", bound=Operation)


@overload
def _for_op(network: VectorizedOpSeqNetwork, batch: int, op: _TOp) -> _TOp: ...


@overload
def _for_op(network: VectorizedOpSeqNetwork, batch: int, op: Operation) -> Operation: ...


def _for_op(network: VectorizedOpSeqNetwork, batch: int, op: Operation) -> Operation:
    match op:
        case Linear(weight_ops=weight_ops):
            return Linear(_for_op_seq(network, batch, weight_ops))
        case GenericGather(ordinals=ordinals):
            return GenericGather(ordinals=[len(ordinals)])
        case TakeSingleValue():
            return TakeSingleValue(0)
        case NoopGather():
            return NoopGather()
        case SliceValues(start=start, end=end, step=step):
            return SliceValues(0, (end - start) // step, 1)
        case Repeat():
            return copy.deepcopy(op)
        case RepeatInterleave():
            return copy.deepcopy(op)
        case GatherPair(a, b):
            return GatherPair(
                _for_op(network, batch, a),
                _for_op(network, batch, b),
            )
        case Transform():
            return copy.deepcopy(op)
        case DimReduce():
            return copy.deepcopy(op)
        case UnevenReduce(counts=counts, reduce=reduce):
            return UnevenReduce(counts=[len(counts)], reduce=reduce)
        case View():
            return copy.deepcopy(op)
        case _:
            assert False, f"{op}"


def _for_op_seq(network: VectorizedOpSeqNetwork, batch: int, seq: OperationSeq) -> OperationSeq:
    return OperationSeq(
        layer_refs=LayerRefs.from_iter([(compute_layer_refs_count(network, batch, None, seq.layer_refs), "")]),
        operations=[_for_op(network, batch, op) for op in seq.operations],
        count=seq.expected_count,
    )


def _for_batch(network: VectorizedOpSeqNetwork, batch_id: int, batch: OpSeqBatch) -> OpSeqBatch:
    return OpSeqBatch(
        OrderedDict(((str(i), _for_op_seq(network, batch_id, ops)) for i, (k, ops) in enumerate(batch.layers.items())))
    )


def replace_tensors_with_shapes(network: VectorizedOpSeqNetwork) -> VectorizedOpSeqNetwork:
    return VectorizedOpSeqNetwork(
        fact_layers={str(i): _for_fact_layer(v) for i, (k, v) in enumerate(network.fact_layers.items())},
        weights={str(i): _for_weight(v) for i, (k, v) in enumerate(network.weights.items())},
        batches=OrderedDict(((i, _for_batch(network, k, v)) for i, (k, v) in enumerate(network.batches.items()))),
    )
