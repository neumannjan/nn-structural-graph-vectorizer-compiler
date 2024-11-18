import itertools
from collections import OrderedDict
from typing import Callable, Collection, Generic, Iterable, Mapping, Protocol, TypeVar, overload

from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_shapes import ComputeLayerShapes

_Ref = tuple[int, str, int]


_T = TypeVar("_T", covariant=True)


class LayerIndexer(Protocol, Generic[_T]):
    def for_refs(self, batch_id: int, refs: Refs) -> Iterable[_T]: ...
    def for_linear(self, batch_id: int, irefs: Refs, wrefs: Refs) -> Iterable[_T]: ...


@overload
def _iter_indexer(
    base: LinearLayerBase | LinearGatherLayerBase, batch_id: int, indexer: LayerIndexer[_T]
) -> Iterable[tuple[tuple[_Ref, _Ref], _T]]: ...


@overload
def _iter_indexer(base: InputLayerBase, batch_id: int, indexer: LayerIndexer[_T]) -> Iterable[tuple[_Ref, _T]]: ...


def _iter_indexer(
    base: LayerBase, batch_id: int, indexer: LayerIndexer[_T]
) -> Iterable[tuple[tuple[_Ref, _Ref] | _Ref, _T]]:
    match base:
        case InputLayerBase(input=Refs() as refs):
            yield from zip(refs, indexer.for_refs(batch_id, refs))
        case (
            LinearLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
            | LinearGatherLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
        ):
            yield from zip(zip(irefs, wrefs), indexer.for_linear(batch_id, irefs, wrefs))
        case _:
            raise ValueError(base)


class SeparateInputRefs:
    def __init__(
        self,
        network: VectorizedLayerNetwork,
        indexer_factory: Callable[[VectorizedLayerNetwork], LayerIndexer[str]],
    ) -> None:
        self.network = network
        self.indexer = indexer_factory(network)
        self._shapes = ComputeLayerShapes(network)

    def _get_layer_id_for_grp(self, layers: Collection[str], orig_layer_id: str, grp: str) -> str:
        name = orig_layer_id + "__" + grp

        if name not in layers:
            return name

        for i in itertools.count():
            name2 = name + str(i)

            if name2 not in layers:
                return name2

        assert False

    def _for_layer(self, layers: Collection[str], batch_id: int, layer_id: str, layer: Layer) -> Mapping[str, Layer]:
        new_layers: OrderedDict[str, Layer] = OrderedDict()

        @overload
        def _get_new_layer_refs(grp: str, old_base: InputLayerBase) -> Refs: ...

        @overload
        def _get_new_layer_refs(grp: str, old_base: LinearLayerBase | LinearGatherLayerBase) -> tuple[Refs, Refs]: ...

        def _get_new_layer_refs(grp: str, old_base: LayerBase) -> Refs | tuple[Refs, Refs]:
            if grp not in new_layers:
                match old_base:
                    case InputLayerBase():
                        refs = Refs([], [], [])
                        new_base = InputLayerBase(refs)
                    case LinearLayerBase() | LinearGatherLayerBase():
                        refs = Refs([], [], []), Refs([], [], [])
                        new_base = LinearLayerBase(*refs, lifts=old_base.lifts)
                    case _:
                        assert False

                layer = Layer(new_base, Noop(), Transform("identity"))
                new_layers[grp] = layer
                return refs
            else:
                new_layer = new_layers[grp]
                match new_layer.base:
                    case InputLayerBase(input=Refs() as refs):
                        return refs
                    case (
                        LinearLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
                        | LinearGatherLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
                    ):
                        return irefs, wrefs
                    case _:
                        assert False

        new_grps = set((grp for _, grp in _iter_indexer(layer.base, batch_id, self.indexer)))

        if len(new_grps) <= 1:
            return {}

        grp_to_lid = {grp: self._get_layer_id_for_grp(layers, orig_layer_id=layer_id, grp=grp) for grp in new_grps}

        match layer.base:
            case InputLayerBase(input=Refs() as refs):
                for i, (ref, grp) in enumerate(_iter_indexer(layer.base, batch_id, self.indexer)):
                    new_layer_refs = _get_new_layer_refs(grp, layer.base)
                    new_o = len(new_layer_refs)
                    new_layer_refs.append(ref)

                    lid = grp_to_lid[grp]
                    refs[i] = Refs.TYPE_LAYER, lid, new_o

                for grp, new_layer in new_layers.items():
                    new_layer_refs = _get_new_layer_refs(grp, layer.base)
                    new_layer.shape = self._shapes.compute_refs_shape(batch_id, new_layer_refs)
            case (
                LinearLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
                | LinearGatherLayerBase(input=Refs() as irefs, weight=Refs() as wrefs)
            ):
                if len(irefs) != len(wrefs):
                    raise ValueError("Not supported")

                new_orig_refs = Refs([], [], [])

                for i, ((iref, wref), grp) in enumerate(_iter_indexer(layer.base, batch_id, self.indexer)):
                    (new_layer_irefs, new_layer_wrefs) = _get_new_layer_refs(grp, layer.base)
                    new_o = len(new_layer_irefs)
                    new_layer_irefs.append(iref)
                    new_layer_wrefs.append(wref)

                    lid = grp_to_lid[grp]
                    new_orig_refs.append((Refs.TYPE_LAYER, lid, new_o))

                match layer.base:
                    case LinearLayerBase() | LinearGatherLayerBase(gather=NoopGather()):
                        pass
                    case LinearGatherLayerBase(gather=GenericGather(ordinals)):
                        new_orig_refs2 = Refs([], [], [])
                        for o in ordinals:
                            new_orig_refs2.append(new_orig_refs[o])
                        new_orig_refs = new_orig_refs2

                for grp, new_layer in new_layers.items():
                    new_layer_irefs, new_layer_wrefs = _get_new_layer_refs(grp, layer.base)
                    i_shape = self._shapes.compute_refs_shape(batch_id, new_layer_irefs)
                    w_shape = self._shapes.compute_refs_shape(batch_id, new_layer_wrefs)
                    new_layer.shape = self._shapes.compute_linear_shape_from_shapes(i_shape, w_shape)

                layer.base = InputLayerBase(input=new_orig_refs)
            case _:
                assert False

        out = {grp_to_lid[grp]: layer for grp, layer in new_layers.items()}
        return out

    def separate_input_refs(self):
        for batch_id, batch in self.network.batches.items():
            layer_ids = list(batch.layers.keys())
            for layer_id, layer in list(batch.layers.items()):
                try:
                    new_layers_this = self._for_layer(layer_ids, batch_id, layer_id, layer)
                    batch.layers.update(new_layers_this)  # TODO only run all this crap if new_layers_this is not None
                    layer_ord = layer_ids.index(layer_id)
                    layer_ids[layer_ord:layer_ord] = new_layers_this.keys()
                    layer.shape = self._shapes.compute_layer_shape(batch_id, layer)
                except Exception:
                    raise Exception(f"Failure in layer {layer_id}: {batch}")

            batch.layers = OrderedDict(((k, batch.layers[k]) for k in layer_ids))


class _ShapeLayerIndexer(LayerIndexer[tuple[ConcreteShape | AnyShape, ...]]):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self._shapes = ComputeLayerShapes(network)

    def for_refs(self, batch_id: int, refs: Refs) -> Iterable[tuple[ConcreteShape | AnyShape, ...]]:
        shapes_map = {ref: self._for_ref_shape(batch_id, ref) for ref in refs}
        concrete_shapes = set([s for s in shapes_map.values() if not isinstance(s, AnyShape)])
        if len(concrete_shapes) == 1:
            shp = next(iter(concrete_shapes))
            return [(shp,)] * len(refs)

        return ((shapes_map[ref],) for ref in refs)

    def for_linear(self, batch_id: int, irefs: Refs, wrefs: Refs) -> Iterable[tuple[ConcreteShape | AnyShape, ...]]:
        ishapes = self.for_refs(batch_id, irefs)
        wshapes = self.for_refs(batch_id, wrefs)

        return ((wshp, ishp) for ((wshp,), (ishp,)) in zip(wshapes, ishapes))

    def _for_ref_shape(self, batch_id: int, ref: _Ref) -> ConcreteShape | AnyShape:
        out_shape = self._shapes.compute_ref_shape(batch_id, *ref)
        match out_shape:
            case ConcreteShape() | AnyShape():
                return out_shape
            case _:
                raise ValueError(out_shape)


class ShapeLayerIndexer(LayerIndexer[str]):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self._delegate = _ShapeLayerIndexer(network)

    def _get_shape_id(self, shape: ConcreteShape | AnyShape) -> str:
        match shape:
            case ConcreteShape(dims):
                return "_".join((str(d) for d in dims))
            case AnyShape():
                return "any"
            case _:
                assert False, f"{shape}"

    def _get_shape_ids(self, shapes: Iterable[ConcreteShape | AnyShape]) -> str:
        return "__".join((self._get_shape_id(sh) for sh in shapes))

    def for_refs(self, batch_id: int, refs: Refs) -> Iterable[str]:
        return (self._get_shape_ids(shps) for shps in self._delegate.for_refs(batch_id, refs))

    def for_linear(self, batch_id: int, irefs: Refs, wrefs: Refs) -> Iterable[str]:
        return (self._get_shape_ids(shps) for shps in self._delegate.for_linear(batch_id, irefs, wrefs))


class WeightLayerIndexer(LayerIndexer[str]):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self._network = network

    def for_refs(self, batch_id: int, refs: Refs) -> Iterable[str]:
        return (self.for_ref(batch_id, ref) for ref in refs)

    def for_linear(self, batch_id: int, irefs: Refs, wrefs: Refs) -> Iterable[str]:
        return (self._for_linear(batch_id, iref, wref) for iref, wref in zip(irefs, wrefs))

    def for_ref(self, batch_id: int, ref: _Ref) -> str:
        t, l, _ = ref
        if t == Refs.TYPE_WEIGHT:
            return l
        else:
            return ""

    def _for_linear(self, batch_id: int, iref: _Ref, wref: _Ref) -> str:
        t1, l1, _ = iref
        t2, l2, _ = wref
        match t1, t2:
            case (Refs.TYPE_WEIGHT, Refs.TYPE_WEIGHT):
                return l1 + "_" + l2
            case (Refs.TYPE_WEIGHT, _):
                return l1
            case (_, Refs.TYPE_WEIGHT):
                return l2
            case _:
                return ""


class CombinedLayerIndexer(LayerIndexer[str]):
    def __init__(self, *indexers: LayerIndexer[str]) -> None:
        self.indexers = indexers

    def for_refs(self, batch_id: int, refs: Refs) -> Iterable[str]:
        return ("__".join(strs) for strs in zip(*(idx.for_refs(batch_id, refs) for idx in self.indexers)))

    def for_linear(self, batch_id: int, irefs: Refs, wrefs: Refs) -> Iterable[str]:
        return ("__".join(strs) for strs in zip(*(idx.for_linear(batch_id, irefs, wrefs) for idx in self.indexers)))


def build_combined_layer_indexer_factory(
    *indexers: Callable[[VectorizedLayerNetwork], LayerIndexer[str]],
) -> Callable[[VectorizedLayerNetwork], CombinedLayerIndexer]:
    def _f(network: VectorizedLayerNetwork):
        return CombinedLayerIndexer(*[f(network) for f in indexers])

    return _f


def build_separate_input_refs(indexer_factory: Callable[[VectorizedLayerNetwork], LayerIndexer[str]]):
    def separate_input_refs(network: VectorizedLayerNetwork):
        SeparateInputRefs(network, indexer_factory).separate_input_refs()
        return network

    return separate_input_refs
