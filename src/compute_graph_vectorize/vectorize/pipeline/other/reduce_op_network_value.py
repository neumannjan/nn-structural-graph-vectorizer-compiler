import itertools
from typing import Iterator, Mapping, Protocol

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import (
    compute_layer_refs_count,
    compute_op_seq_network_layer_counts,
    iter_operation_seq_shapes,
)


class OpNetworkOperationComputeFunc(Protocol):
    def __call__(
        self,
        *,
        network: VectorizedOpSeqNetwork,
        batch: int,
        batch_layer_counts: Mapping[str, int | None] | None,
        in_shape: tuple[int, ...],
        skip_dims: int,
        op: SimpleOperation,
    ) -> int: ...


def count_gathers(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    in_shape: tuple[int, ...],
    skip_dims: int,
    op: SimpleOperation,
):
    return 1 if isinstance(op, Gather) else 0


def count_gather_items(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    in_shape: tuple[int, ...],
    skip_dims: int,
    op: SimpleOperation,
):
    match op:
        case GenericGather(ordinals=ordinals):
            return len(ordinals)
        case TakeSingleValue():
            return 1
        case NoopGather():
            return 0
        case SliceValues(start=start, end=end, step=step):
            return (end - start) // step
        case Repeat(times=_, total_length=total_length):
            return total_length
        case RepeatInterleave(times=_, total_length=total_length):
            return total_length
        case Transform():
            return 0
        case DimReduce():
            return 0
        case UnevenReduce():
            return 0
        case View():
            return 0
        case _:
            assert False, f"{op}"


_MAP: list[OpNetworkOperationComputeFunc] = [
    count_gathers,
    count_gather_items,
]

def sum_operation_values(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    func: OpNetworkOperationComputeFunc,
    in_shape: tuple[int, ...],
    skip_dims: int,
    op: Operation,
):
    match op:
        case Linear(weight_ops=weight_ops):
            return sum_operation_seq_values(network, batch, batch_layer_counts, func, weight_ops)
        case GatherPair(a, b):
            a_val = sum_operation_values(network, batch, batch_layer_counts, func, in_shape, skip_dims, a)
            b_val = sum_operation_values(network, batch, batch_layer_counts, func, in_shape, skip_dims, b)
            return a_val + b_val
        case _:
            return func(
                network=network,
                batch=batch,
                batch_layer_counts=batch_layer_counts,
                in_shape=in_shape,
                skip_dims=skip_dims,
                op=op,
            )


def iter_operation_seq_values(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    func: OpNetworkOperationComputeFunc,
    seq: OperationSeq,
) -> Iterator[int]:
    shapes_iter = iter_operation_seq_shapes(network, batch, batch_layer_counts, seq)

    for in_shp, op in zip(itertools.islice(shapes_iter, len(seq.operations)), seq):
        yield sum_operation_values(network, batch, batch_layer_counts, func, in_shp, skip_dims=2, op=op)


def sum_operation_seq_values(
    network: VectorizedOpSeqNetwork,
    batch: int,
    batch_layer_counts: Mapping[str, int | None] | None,
    func: OpNetworkOperationComputeFunc,
    seq: OperationSeq,
) -> int:
    return sum(iter_operation_seq_values(network, batch, batch_layer_counts, func, seq))


def sum_batch_values(
    network: VectorizedOpSeqNetwork,
    batch: int,
    func: OpNetworkOperationComputeFunc,
) -> int:
    batch_layer_counts = compute_op_seq_network_layer_counts(network, batch)
    b = network.batches[batch]

    return sum(sum_operation_seq_values(network, batch, batch_layer_counts, func, seq) for seq in b.layers.values())


def sum_op_network_values(network: VectorizedOpSeqNetwork, func: OpNetworkOperationComputeFunc):
    return sum(sum_batch_values(network, batch, func) for batch in network.batches)
