import itertools
from typing import Iterator, Mapping

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import (
    compute_layer_refs_count,
    compute_op_seq_network_layer_counts,
    compute_operation_shape,
    iter_operation_seq_shapes,
)


def compute_operation_estimate(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    in_shape: tuple[int, ...],
    skip_dims: int,
    op: Operation,
) -> int:
    match op:
        case Linear(weight_ops=weight_ops):
            out = compute_operation_seq_estimate(network, batch, batch_layer_counts, weight_ops)
            out += compute_operation_shape(network, batch, batch_layer_counts, in_shape, skip_dims, op)[0]
            return out
        case _ if isinstance(op, Gather):
            return compute_operation_shape(network, batch, batch_layer_counts, in_shape, skip_dims, op)[0]
        case Transform():
            return in_shape[0]
        case DimReduce(dim=dim):
            if dim in (0, 1):
                return in_shape[1 - dim]
            else:
                raise ValueError(op)
        case UnevenReduce(counts=counts):
            return sum(counts)
        case View():
            return 0
        case _:
            assert False, f"{op}"


def compute_layer_refs_perf_estimate(network: VectorizedOpSeqNetwork, batch: int, batch_layer_counts: Mapping[str, int |
    None] | None, refs: LayerRefs):
    if len(refs) <= 1:
        return 0

    return compute_layer_refs_count(network, batch, batch_layer_counts, refs)


def iter_operation_seq_estimates(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    seq: OperationSeq,
) -> Iterator[int]:
    shapes_iter = iter_operation_seq_shapes(network, batch, batch_layer_counts, seq)

    yield compute_layer_refs_perf_estimate(network, batch, batch_layer_counts, seq.layer_refs)

    for in_shp, op in zip(itertools.islice(shapes_iter, len(seq.operations)), seq):
        yield compute_operation_estimate(network, batch, batch_layer_counts, in_shp, skip_dims=2, op=op)

def compute_operation_seq_estimate(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    seq: OperationSeq,
) -> int:
    return sum(iter_operation_seq_estimates(network, batch, batch_layer_counts, seq))


def compute_batch_perf_estimate(network: VectorizedOpSeqNetwork, batch: int) -> int:
    batch_layer_counts = compute_op_seq_network_layer_counts(network, batch)
    b = network.batches[batch]

    return sum(compute_operation_seq_estimate(network, batch, batch_layer_counts, seq) for seq in b.layers.values())


def compute_op_seq_network_perf_estimate(network: VectorizedOpSeqNetwork):
    return sum(compute_batch_perf_estimate(network, batch) for batch in network.batches)
