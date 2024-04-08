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

    def _build_period_view(self, shape: Shape, period: int) -> View:
        assert isinstance(shape, ConcreteShape)
        return View(ConcreteShape([-1, period, *shape]))

    def _add_linear_ops(
        self,
        batch_id: int,
        id: str,
        period: int | None,
        input: GatheredLayers,
        weight: GatheredLayers,
        out: list[Operation],
    ) -> None:
        if not isinstance(input.gather, NoopGather):
            out.append(input.gather)

        weight_ops = OperationSeq(layer_refs=weight.refs, operations=[])

        if not isinstance(weight.gather, NoopGather):
            weight_ops.operations.append(weight.gather)

        if period is not None:
            input_shape = self._compute_shapes.compute_input_shape(batch_id, input)
            out.append(self._build_period_view(input_shape, period))

            weight_shape = self._compute_shapes.compute_input_shape(batch_id, weight)
            weight_ops.operations.append(self._build_period_view(weight_shape, period))

        out.append(Linear(weight_ops))

    def _map_layer(self, batch_id: int, id: str, layer: Layer) -> OperationSeq:
        try:
            layer_refs: LayerRefs | None = None
            out: list[Operation] = []

            period = self._get_aggregate_period(layer.aggregate)

            match layer.base:
                case InputLayerBase(input=GatheredLayers() as input):
                    layer_refs = input.refs

                    if not isinstance(input.gather, NoopGather):
                        out.append(input.gather)

                    if period is not None:
                        shape = self._compute_shapes.compute_layer_base_shape(batch_id, layer.base)
                        out.append(self._build_period_view(shape, period))
                case LinearLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight):
                    layer_refs = input.refs
                    self._add_linear_ops(batch_id=batch_id, id=id, period=period, input=input, weight=weight, out=out)

                case LinearGatherLayerBase(
                    input=GatheredLayers() as input,
                    weight=GatheredLayers() as weight,
                    gather=gather,
                ):
                    layer_refs = input.refs
                    self._add_linear_ops(batch_id=batch_id, id=id, period=None, input=input, weight=weight, out=out)

                    out.append(gather)

                    if period is not None:
                        shape = self._compute_shapes.compute_layer_base_shape(batch_id, layer.base)
                        out.append(self._build_period_view(shape, period))

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

            return OperationSeq(layer_refs, out)
        except Exception as e:
            raise Exception(f"Exception in layer {id} (batch {batch_id})") from e

    def _map_batch(self, batch_id: int, batch: Batch) -> OpSeqBatch:
        layers = OrderedDict(((id, self._map_layer(batch_id, id, layer)) for id, layer in batch.layers.items()))
        return OpSeqBatch(layers=layers)

    def to_seq_network(self) -> VectorizedOpSeqNetwork:
        return VectorizedOpSeqNetwork(
            fact_layers=self.network.fact_layers,
            weights=self.network.weights,
            batches=OrderedDict(((id, self._map_batch(id, batch)) for id, batch in self.network.batches.items())),
        )


def to_seq_network(network: VectorizedLayerNetwork) -> VectorizedOpSeqNetwork:
    return ToSeqNetwork(network).to_seq_network()
