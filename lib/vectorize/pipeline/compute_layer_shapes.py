from typing import Callable, Iterable

from lib.vectorize.model import *

VARIOUS_SHAPE = VariousShape()
ANY_SHAPE = AnyShape()


def _compute_fact_shape(fact: Fact) -> Shape:
    match fact:
        case UnitFact():
            return ANY_SHAPE
        case ValueFact(value=value):
            return ConcreteShape(value.shape[1:])
        case _:
            assert False


def _compute_weight_shape(weight: LearnableWeight) -> Shape:
    return ConcreteShape(weight.value.shape[1:])


def reduce_shapes(shapes: Iterable[Shape], func: Callable[[ConcreteShape, ConcreteShape], Shape]) -> Shape:
    acc = ANY_SHAPE
    for s in shapes:
        match s:
            case AnyShape():
                continue
            case VariousShape():
                return VARIOUS_SHAPE
            case ConcreteShape(_) as shp2:
                match acc:
                    case AnyShape():
                        acc = ConcreteShape(shp2)
                    case VariousShape():
                        return VARIOUS_SHAPE
                    case ConcreteShape(_) as shp:
                        acc = func(shp, shp2)
                    case _:
                        assert False
            case _:
                assert False

    return acc


class ComputeLayerShapes:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    def compute_gather_shape(self, in_shape: Shape, gather: Gather) -> Shape:
        match in_shape:
            case ConcreteShape():
                match gather:
                    case ViewWithPeriod(period=period):
                        return ConcreteShape([period, *in_shape[0:]])
                    case GatherPair(a, b):
                        shape = in_shape
                        shape = self.compute_gather_shape(shape, a)
                        shape = self.compute_gather_shape(shape, b)
                        return shape
                    case _:
                        return in_shape
            case _:
                return in_shape

    def compute_aggregation_shape(self, shp: Shape, agg: Reduce) -> Shape:
        match (shp, agg):
            case (ConcreteShape(_), DimReduce(dim=dim)):
                return ConcreteShape([*shp[: dim - 1], *shp[dim:]])
            case _:
                return shp

    def compute_ref_shape(self, batch: int, ref: Ref) -> Shape:
        match ref:
            case FactRef(id=id, ordinal=ordinal):
                return _compute_fact_shape(self.network.fact_layers[id].facts[ordinal])
            case WeightRef(id=id):
                return _compute_weight_shape(self.network.weights[id])
            case NeuronRef(id=id, ordinal=ordinal):
                match self.network.batches[batch].layers[id].shape:
                    case ConcreteShape(_) as shp:
                        return shp
                    case AnyShape():
                        return ANY_SHAPE
                    case VariousShape():
                        return VARIOUS_SHAPE
                    case _:
                        assert False
            case _:
                assert False, f"{ref}"

    def compute_refs_shape(self, batch: int, refs: Refs) -> Shape:
        return reduce_shapes(
            (self.compute_ref_shape(batch, ref) for ref in refs.refs),
            lambda a, b: VARIOUS_SHAPE if a != b else a,
        )

    def compute_layer_ref_shape(self, batch: int, ref: LayerRef) -> Shape:
        match ref:
            case FactLayerRef(id=id):
                return self.network.fact_layers[id].shape
            case NeuronLayerRef(id=id):
                return self.network.batches[batch].layers[id].shape
            case WeightRef(id=id):
                return _compute_weight_shape(self.network.weights[id])
            case _:
                assert False, f"{ref}"

    def compute_layer_refs_shape(self, batch: int, refs: LayerRefs) -> Shape:
        return reduce_shapes(
            (self.compute_layer_ref_shape(batch, ref) for ref in refs.refs),
            lambda a, b: VARIOUS_SHAPE if a != b else a,
        )

    def compute_input_shape(self, batch: int, input: Input) -> Shape:
        match input:
            case Refs(_) as refs:
                shape = self.compute_refs_shape(batch, refs)
            case GatheredLayers(refs=refs, gather=gather):
                shape = self.compute_layer_refs_shape(batch, refs)
                shape = self.compute_gather_shape(shape, gather)
            case _:
                assert False, f"{input}"
        return shape

    def compute_linear_shape(self, batch: int, input: Input, weight: Input) -> Shape:
        weight_shape = self.compute_input_shape(batch, weight)
        input_shape = self.compute_input_shape(batch, input)

        match (weight_shape, input_shape):
            case (ConcreteShape(_), ConcreteShape(_)):
                begin_shape = [max(a, b) for a, b in zip(weight_shape[:-2], input_shape[:-2])]
                return ConcreteShape([*begin_shape, weight_shape[-2], input_shape[-1]])
            case (VariousShape(), _):
                return VARIOUS_SHAPE
            case (_, VariousShape()):
                return VARIOUS_SHAPE
            case (ConcreteShape(_), _):
                return weight_shape
            case (_, ConcreteShape(_)):
                return input_shape
            case _:
                return ANY_SHAPE

    def compute_layer_base_shape(self, batch: int, base: LayerBase) -> Shape:
        match base:
            case InputLayerBase(input=input):
                return self.compute_input_shape(batch, input)
            case LinearLayerBase(input=input, weight=weight):
                return self.compute_linear_shape(batch, input, weight)
            case LinearGatherLayerBase(input=input, weight=weight, gather=gather):
                shape = self.compute_linear_shape(batch, input, weight)
                shape = self.compute_gather_shape(shape, gather)
                return shape
            case _:
                assert False

    def compute_layer_shape(self, batch: int, layer: Layer) -> Shape:
        shape = self.compute_layer_base_shape(batch, layer.base)
        shape = self.compute_aggregation_shape(shape, layer.aggregate)
        return shape

    def compute_shapes(self):
        for layer in self.network.fact_layers.values():
            layer.shape = reduce_shapes(
                ((_compute_fact_shape(f) for f in layer.facts)),
                lambda a, b: VARIOUS_SHAPE if a != b else ConcreteShape(a),
            )

        for bid, batch in self.network.batches.items():
            try:
                for lid, layer in batch.layers.items():
                    layer.shape = self.compute_layer_shape(bid, layer)
            except Exception as e:
                raise Exception(f"Exception in batch {bid}, layer {lid}") from e


def compute_layer_shapes(network: VectorizedNetwork):
    ComputeLayerShapes(network).compute_shapes()
    return network
