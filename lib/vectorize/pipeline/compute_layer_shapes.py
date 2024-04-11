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
    for shp in shapes:
        match shp:
            case AnyShape():
                continue
            case VariousShape():
                return VARIOUS_SHAPE
            case ConcreteShape(_):
                match acc:
                    case AnyShape():
                        acc = ConcreteShape(shp)
                    case VariousShape():
                        return VARIOUS_SHAPE
                    case ConcreteShape(_):
                        acc = func(acc, shp)
                    case _:
                        assert False
            case _:
                assert False

    return acc


class ComputeLayerShapes:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def compute_gather_shape(self, in_shape: Shape, gather: Gather) -> Shape:
        match in_shape:
            case ConcreteShape():
                match gather:
                    case GatherPair(a, b):
                        shape = in_shape
                        shape = self.compute_gather_shape(shape, a)
                        shape = self.compute_gather_shape(shape, b)
                        return shape
                    case _:
                        return in_shape
            case _:
                return in_shape

    def compute_ref_shape(self, batch: int, type: int, id: str, ordinal: int) -> Shape:
        match type:
            case Refs.TYPE_FACT:
                return _compute_fact_shape(self.network.fact_layers[id].facts[ordinal])
            case Refs.TYPE_WEIGHT:
                return _compute_weight_shape(self.network.weights[id])
            case Refs.TYPE_LAYER:
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
                assert False, f"{type}"

    def compute_refs_shape(self, batch: int, refs: Refs) -> Shape:
        return reduce_shapes(
            (self.compute_ref_shape(batch, t, l, o) for t, l, o in zip(refs.types, refs.layer_ids, refs.ordinals)),
            lambda a, b: VARIOUS_SHAPE if a != b else a,
        )

    def iter_layer_refs_shapes(self, batch: int, refs: LayerRefs) -> Iterable[Shape]:
        for t, id in refs:
            match t:
                case LayerRefs.TYPE_FACT:
                    yield self.network.fact_layers[id].shape
                case LayerRefs.TYPE_WEIGHT:
                    yield _compute_weight_shape(self.network.weights[id])
                case LayerRefs.TYPE_LAYER:
                    yield self.network.batches[batch].layers[id].shape
                case _:
                    assert False, f"{t}"

    def compute_layer_refs_shape(self, batch: int, refs: LayerRefs) -> Shape:
        return reduce_shapes(
            self.iter_layer_refs_shapes(batch, refs),
            lambda a, b: VARIOUS_SHAPE if a != b else a,
        )

    def compute_input_shape(self, batch: int, input: Input) -> Shape:
        match input:
            case Refs() as refs:
                shape = self.compute_refs_shape(batch, refs)
            case GatheredLayers(refs=refs, gather=gather):
                shape = self.compute_layer_refs_shape(batch, refs)
                shape = self.compute_gather_shape(shape, gather)
            case _:
                assert False, f"{input}"
        return shape

    def compute_linear_shape_from_shapes(self, input_shape: Shape, weight_shape: Shape) -> Shape:
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

    def compute_linear_shape(self, batch: int, input: Input, weight: Input) -> Shape:
        weight_shape = self.compute_input_shape(batch, weight)
        input_shape = self.compute_input_shape(batch, input)

        return self.compute_linear_shape_from_shapes(input_shape, weight_shape)

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


def compute_layer_shapes(network: VectorizedLayerNetwork):
    ComputeLayerShapes(network).compute_shapes()
    return network
