from typing import Callable, Iterable

from lib.vectorize.model import *


def _compute_fact_shape(fact: Fact) -> Shape:
    match fact:
        case UnitFact():
            return AnyShape()
        case ValueFact(value=value):
            return ConcreteShape(value.shape)
        case _:
            assert False


def reduce_shapes(shapes: Iterable[Shape], func: Callable[[ConcreteShape, ConcreteShape], Shape]) -> Shape:
    acc = AnyShape()
    for s in shapes:
        match s:
            case AnyShape():
                continue
            case VariousShape():
                return VariousShape()
            case ConcreteShape(_) as shp2:
                match acc:
                    case AnyShape():
                        acc = ConcreteShape(shp2)
                    case VariousShape():
                        return VariousShape()
                    case ConcreteShape(_) as shp:
                        acc = func(shp, shp2)
                    case _:
                        assert False
            case _:
                assert False

    return acc


def _remap_shape(func: Callable[[ConcreteShape], Iterable[int]], shape: Shape) -> Shape:
    match shape:
        case ConcreteShape(_) as shp:
            return ConcreteShape(func(shp))
        case _:
            return shape


class ComputeLayerShapes:
    def __init__(self, network: VectorizedNetwork) -> None:
        self.network = network

    def compute_weight_shape(self, weight: LearnableWeight) -> Shape:
        return ConcreteShape(list(weight.value.shape))

    def compute_gather_shape(self, in_shape: Shape, gather: Gather) -> Shape:
        match gather:
            case GenericGather(ordinals=ordinals):
                return _remap_shape(lambda shp: [len(ordinals), *shp[1:]], in_shape)
            case TakeSingleValue(ordinal=_):
                return _remap_shape(lambda shp: [1, *shp[1:]], in_shape)
            case NoopGather():
                return in_shape
            case SliceValues(start=start, end=end, step=step):
                return _remap_shape(lambda shp: [-(-(end - start) // step), *shp[1:]], in_shape)
            case Repeat(times=_, total_length=total_length):
                return _remap_shape(lambda shp: [total_length, *shp[1:]], in_shape)
            case ViewWithPeriod(period=period):
                return _remap_shape(lambda shp: [shp[0] // period, period, *shp[1:]], in_shape)
            case GatherSequence(gathers=gathers):
                return reduce_shapes(
                    (self.compute_gather_shape(in_shape, g) for g in gathers),
                    lambda a, b: ConcreteShape([a[0] + b[0], *a[1:]]) if a[1:] == b[1:] else VariousShape(),
                )
            case _:
                assert False

    def compute_aggregation_shape(self, in_shape: Shape, agg: Reduce) -> Shape:
        match in_shape:
            case ConcreteShape(shp):
                match agg:
                    case FixedCountReduce(period=period, reduce=_):
                        return ConcreteShape([shp[0] // period, *shp[1:]])
                    case DimReduce(dim=dim, reduce=_):
                        return ConcreteShape([*shp[:dim], *shp[dim + 1 :]])
                    case UnevenReduce(counts=counts, reduce=_):
                        return ConcreteShape([len(counts), *shp[1:]])
                    case Noop():
                        return in_shape
                    case _:
                        assert False
            case _:
                return in_shape

    def compute_ref_shape(self, batch: int, ref: Ref) -> Shape:
        match ref:
            case FactRef(id=id, ordinal=ordinal):
                return _compute_fact_shape(self.network.fact_layers[id].facts[ordinal])
            case NeuronRef(id=id, ordinal=ordinal):
                match self.network.batches[batch].layers[id].shape:
                    case ConcreteShape(_) as shp:
                        return ConcreteShape([1, *shp[1:]])
                    case AnyShape():
                        return AnyShape()
                    case VariousShape():
                        return VariousShape()
                    case _:
                        assert False
            case WeightRef(id=id):
                return self.compute_weight_shape(self.network.weights[id])
            case _:
                assert False, f"{ref}"

    def compute_refs_shape(self, batch: int, refs: Refs) -> Shape:
        return reduce_shapes(
            (self.compute_ref_shape(batch, ref) for ref in refs.refs),
            lambda a, b: VariousShape() if a[1:] != b[1:] else ConcreteShape([a[0] + b[0], *a[1:]]),
        )

    def compute_layer_ref_shape(self, batch: int, ref: LayerRef) -> Shape:
        match ref:
            case FactLayerRef(id=id):
                return reduce_shapes(
                    (_compute_fact_shape(fact) for fact in self.network.fact_layers[id].facts),
                    lambda a, b: VariousShape() if a[1:] != b[1:] else ConcreteShape([a[0] + b[0], *a[1:]]),
                )
            case NeuronLayerRef(id=id):
                return self.network.batches[batch].layers[id].shape
            case WeightRef(id=id):
                return self.compute_weight_shape(self.network.weights[id])
            case _:
                assert False, f"{ref}"

    def compute_layer_refs_shape(self, batch: int, refs: LayerRefs) -> Shape:
        return reduce_shapes(
            (self.compute_layer_ref_shape(batch, ref) for ref in refs.refs),
            lambda a, b: VariousShape() if a[1:] != b[1:] else ConcreteShape([a[0] + b[0], *a[1:]]),
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

        def _reduce_shapes(weight_shape: ConcreteShape, input_shape: ConcreteShape):
            assert len(weight_shape) == len(input_shape)
            assert weight_shape[-1] == input_shape[-2]

            begin_shape = [max(a, b) for a, b in zip(weight_shape[:-2], input_shape[:-2])]
            return ConcreteShape([*begin_shape, weight_shape[-2], input_shape[-1]])

        return reduce_shapes([weight_shape, input_shape], _reduce_shapes)

    def compute_layer_base_shape(self, batch: int, base: LayerBase) -> Shape:
        match base:
            case InputLayerBase(input=gathered_source):
                return self.compute_input_shape(batch, gathered_source)
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

    def compute_shapes(self) -> VectorizedNetwork:
        for layer in self.network.fact_layers.values():
            layer.shape = reduce_shapes(
                ((_compute_fact_shape(f) for f in layer.facts)),
                lambda a, b: VariousShape() if a != b else ConcreteShape(a),
            )

        for bid, batch in self.network.batches.items():
            try:
                for lid, layer in batch.layers.items():
                    layer.shape = self.compute_layer_shape(bid, layer)
            except Exception as e:
                raise Exception(f"Exception in batch {bid}, layer {lid}") from e

        return self.network


def compute_shapes(network: VectorizedNetwork):
    return ComputeLayerShapes(network).compute_shapes()
