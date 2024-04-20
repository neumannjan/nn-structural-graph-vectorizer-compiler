from typing import OrderedDict

from lib.vectorize.model import *
from lib.vectorize.model.op_network import DimReduce
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.compute_layer_shapes import ComputeLayerShapes


class ToSeqNetwork:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._compute_shapes = ComputeLayerShapes(network)
        self._compute_counts = ComputeLayerCounts(network)

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

    def _build_view(self, shape: Shape, lift: tuple[int, ...]) -> View:
        assert isinstance(shape, ConcreteShape)

        return View(ConcreteShape([*lift, *shape]))

    def _is_simple_lift(self, lift: tuple[int, int]):
        match lift:
            case (-1, 1) | (1, -1):
                return True
            case _:
                return False

    def _add_linear_ops(
        self,
        batch_id: int,
        id: str,
        agg_period: int | None,
        input: GatheredLayers,
        weight: GatheredLayers,
        lifts: tuple[tuple[int, int], tuple[int, int]] | None,
        out: list[Operation],
    ) -> None:
        if not isinstance(input.gather, NoopGather):
            out.append(input.gather)

        weight_ops = OperationSeq(layer_refs=weight.refs, operations=[], count=None)

        if not isinstance(weight.gather, NoopGather):
            weight_ops.operations.append(weight.gather)

        if lifts is not None and self._is_simple_lift(lifts[0]) and self._is_simple_lift(lifts[1]):
            lifts = None

        if lifts is not None:
            input_shape = self._compute_shapes.compute_input_shape(batch_id, input)
            weight_shape = self._compute_shapes.compute_input_shape(batch_id, weight)

            input_lift, weight_lift = lifts

            out.append(self._build_view(input_shape, input_lift))
            weight_ops.operations.append(self._build_view(weight_shape, weight_lift))

            out.append(Linear(weight_ops))

            shape = self._compute_shapes.compute_linear_shape_from_shapes(input_shape, weight_shape)

            if agg_period is None:
                out.append(self._build_view(shape, (-1,)))
            elif agg_period is not None and (input_lift != (-1, agg_period) or weight_lift != (-1, agg_period)):
                out.append(self._build_view(shape, (-1, agg_period)))
        else:
            out.append(Linear(weight_ops))

            if agg_period is not None:
                shape = self._compute_shapes.compute_linear_shape(batch_id, input, weight)
                out.append(self._build_view(shape, (-1, agg_period)))

    def _map_layer(self, batch_id: int, id: str, layer: Layer) -> OperationSeq:
        try:
            layer_refs: LayerRefs | None = None
            out: list[Operation] = []

            agg_period = self._get_aggregate_period(layer.aggregate)

            match layer.base:
                case InputLayerBase(input=GatheredLayers() as input):
                    layer_refs = input.refs

                    if not isinstance(input.gather, NoopGather):
                        out.append(input.gather)

                    if agg_period is not None:
                        shape = self._compute_shapes.compute_layer_base_shape(batch_id, layer.base)
                        out.append(self._build_view(shape, (-1, agg_period)))
                case LinearLayerBase(input=GatheredLayers() as input, weight=GatheredLayers() as weight, lifts=lifts):
                    layer_refs = input.refs
                    self._add_linear_ops(
                        batch_id=batch_id,
                        id=id,
                        agg_period=agg_period,
                        input=input,
                        weight=weight,
                        lifts=lifts,
                        out=out,
                    )

                case LinearGatherLayerBase(
                    input=GatheredLayers() as input,
                    weight=GatheredLayers() as weight,
                    gather=gather,
                    lifts=lifts,
                ):
                    layer_refs = input.refs
                    self._add_linear_ops(
                        batch_id=batch_id,
                        id=id,
                        agg_period=None,
                        input=input,
                        weight=weight,
                        lifts=lifts,
                        out=out,
                    )

                    out.append(gather)

                    if agg_period is not None:
                        shape = self._compute_shapes.compute_layer_base_shape(batch_id, layer.base)
                        out.append(self._build_view(shape, (-1, agg_period)))
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

            return OperationSeq(layer_refs, out, count=layer.count)
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
