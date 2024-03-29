from typing import OrderedDict

from lib.vectorize.model import *
from lib.vectorize.model.op_network import DimReduce
from lib.vectorize.pipeline.compute_layer_shapes import ComputeLayerShapes


class ToSeqNetwork:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._compute_shapes = ComputeLayerShapes(network)

    def _get_aggregate_period(self, aggregate: Reduce) -> int | None:
        match aggregate:
            case FixedCountReduce(period=period, reduce=_):
                return period
            case UnevenReduce(counts=_, reduce=_):
                return None
            case Noop():
                return None
            case _:
                assert False, f"{aggregate}"

    def _add_linear_ops(
        self,
        batch_id: int,
        id: str,
        aggregate: Reduce,
        input: GatheredLayers,
        weight: GatheredLayers,
        out: list[Operation],
    ) -> None:
        out.append(input.refs)
        if not isinstance(input.gather, NoopGather):
            out.append(input.gather)
        out.append(Linear(weight))

        period = self._get_aggregate_period(aggregate)
        if period is not None:
            shape = self._compute_shapes.compute_linear_shape(batch_id, input, weight)
            assert isinstance(shape, ConcreteShape)
            out.append(View(ConcreteShape([-1, period, *shape])))

    def _map_layer(self, batch_id: int, id: str, layer: Layer) -> OperationSeq:
        try:
            out: list[Operation] = []

            match layer.base:
                case InputLayerBase(input=GatheredLayers() as input):
                    out.append(input.refs)
                    if not isinstance(input.gather, NoopGather):
                        out.append(input.gather)
                case LinearLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight):
                    self._add_linear_ops(
                        batch_id=batch_id, id=id, aggregate=layer.aggregate, input=input, weight=weight, out=out
                    )
                case LinearGatherLayerBase(
                    input=GatheredLayers() as input,
                    weight=GatheredLayers() as weight,
                    gather=gather,
                ):
                    self._add_linear_ops(
                        batch_id=batch_id, id=id, aggregate=layer.aggregate, input=input, weight=weight, out=out
                    )
                    out.append(gather)
                case _:
                    assert False

            match layer.aggregate:
                case FixedCountReduce(period=_, reduce=r):
                    out.append(DimReduce(dim=1, reduce=r))
                case UnevenReduce():
                    out.append(layer.aggregate)
                case Noop():
                    pass
                case _:
                    assert False, f"{layer.aggregate}"

            if layer.transform.transform != "identity":
                out.append(layer.transform)

            return OperationSeq(out)
        except Exception as e:
            raise Exception(f"Exception in layer {id} (batch {batch_id})") from e

    def _map_batch(self, batch_id: int, batch: Batch) -> OpSeqBatch:
        layers = OrderedDict(((id, self._map_layer(batch_id, id, layer)) for id, layer in batch.layers.items()))
        return OpSeqBatch(layers=layers)

    def to_seq_network(self) -> VectorizedOpSeqNetwork:
        return VectorizedOpSeqNetwork(
            fact_layers=self.network.fact_layers,
            weights=self.network.weights,
            batches={id: self._map_batch(id, batch) for id, batch in self.network.batches.items()},
            ref_pool=self.network.ref_pool,
        )


def to_seq_network(network: VectorizedLayerNetwork) -> VectorizedOpSeqNetwork:
    return ToSeqNetwork(network).to_seq_network()
