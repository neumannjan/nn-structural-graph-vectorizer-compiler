import copy
import itertools
from collections import OrderedDict, deque
from typing import Collection, Generator, Iterable, Mapping

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_shapes import ComputeLayerShapes


class MaterializeUnitTransforms:
    def __init__(self, network: VectorizedLayerNetwork):
        self.network = network
        self._compute_shapes = ComputeLayerShapes(network)

    def _iter_layer_refs(self, input: Input) -> Iterable[tuple[LayerRefs | Refs, tuple[int, str]]]:
        match input:
            case Refs():
                for ref in set((t, l) for t, l, _ in input):
                    yield input, ref
            case GatheredLayers(refs=LayerRefs() as refs):
                for ref in set(refs):
                    yield refs, ref
            case _:
                assert False, f"{input}"

    def _find_layer_inputs_for_linear(
        self, batch_id: int, layer_id: str, input: Input, weight: Input, out_shape: ConcreteShape
    ) -> Iterable[tuple[LayerRefs | Refs, tuple[int, str], ConcreteShape]]:
        input_shape = self._compute_shapes.compute_input_shape(batch_id, input)
        weight_shape = self._compute_shapes.compute_input_shape(batch_id, weight)

        if not isinstance(input_shape, ConcreteShape) and not isinstance(weight_shape, ConcreteShape):
            raise ValueError("Simplify unit linears first, and perhaps remove identity layers first.")

        if not isinstance(input_shape, ConcreteShape):
            assert isinstance(weight_shape, ConcreteShape)
            assert len(weight_shape.dims) == 2
            assert len(out_shape.dims) == 2

            a, b = weight_shape.dims
            a2, c = out_shape.dims
            assert a == a2

            input_shape = ConcreteShape((b, c))

        if not isinstance(weight_shape, ConcreteShape):
            assert isinstance(input_shape, ConcreteShape)
            assert len(input_shape.dims) == 2
            assert len(out_shape.dims) == 2

            b, c = input_shape.dims
            a, c2 = out_shape.dims
            assert c == c2

            weight_shape = ConcreteShape((a, b))

        for owner, ref in self._iter_layer_refs(input):
            yield owner, ref, input_shape

        for owner, ref in self._iter_layer_refs(weight):
            yield owner, ref, weight_shape

    def _find_layer_inputs(
        self, batch_id: int, layer_id: str, layer: Layer, out_shape: ConcreteShape
    ) -> Iterable[tuple[LayerRefs | Refs, tuple[int, str], ConcreteShape]]:
        if isinstance(layer.shape, ConcreteShape):
            assert layer.shape.dims == out_shape.dims, f"{batch_id}, {layer_id}: {layer.shape} != {out_shape}"

        match layer.base:
            case InputLayerBase(input=input):
                for owner, ref in self._iter_layer_refs(input):
                    yield owner, ref, out_shape
            case LinearLayerBase(input=input, weight=weight):
                yield from self._find_layer_inputs_for_linear(batch_id, layer_id, input, weight, out_shape)
            case LinearGatherLayerBase(input=input, weight=weight):
                yield from self._find_layer_inputs_for_linear(batch_id, layer_id, input, weight, out_shape)
            case _:
                assert False, f"{layer.base}"

    def _backiter_layers(
        self, batch_id: int, batch: Batch
    ) -> Generator[tuple[LayerRefs | Refs, tuple[int, str], ConcreteShape], tuple[tuple[int, str], Layer | None], None]:
        layer_id, layer = next(reversed(batch.layers.items()))

        if not isinstance(layer.shape, ConcreteShape):
            raise ValueError(f"The output shape of the entire network (batch {batch_id}) is unknown. Cannot continue.")

        out_shape = layer.shape

        queue = deque(self._find_layer_inputs(batch_id, layer_id, layer, out_shape))

        while len(queue) > 0:
            owner, layer_inp_ref, layer_inp_shape = queue.popleft()
            layer_inp_type, layer_inp_id = layer_inp_ref

            if layer_inp_type == LayerRefs.TYPE_LAYER:
                layer_inp_ref, inp_layer = yield owner, (layer_inp_type, layer_inp_id), layer_inp_shape
                assert inp_layer is not None
                layer_inp_type, layer_inp_id = layer_inp_ref
                queue.extend(self._find_layer_inputs(batch_id, layer_inp_id, inp_layer, layer_inp_shape))
            elif layer_inp_type == LayerRefs.TYPE_FACT:
                _, _ = yield owner, (layer_inp_type, layer_inp_id), layer_inp_shape
            elif layer_inp_type == LayerRefs.TYPE_WEIGHT:
                pass
            else:
                assert False

    def _get_current_shape(self, ref: tuple[int, str], layers: Mapping[str, Layer]) -> Shape:
        t, l = ref

        if t == LayerRefs.TYPE_LAYER:
            return layers[l].shape
        elif t == LayerRefs.TYPE_FACT:
            return self.network.fact_layers[l].shape
        else:
            assert False

    def _get_unused_name(self, all_names: Collection[str], desired_name: str) -> str:
        if desired_name not in all_names:
            return desired_name

        for i in itertools.count():
            desired_name2 = desired_name + str(i)
            if desired_name2 not in all_names:
                return desired_name2

        assert False

    def _replace_refs(self, refs: LayerRefs | Refs, ref_from: tuple[int, str], ref_to: tuple[int, str]):
        t, l = ref_to

        for i, ref in enumerate(refs):
            if ref[:2] == ref_from:
                refs.types[i] = t
                refs.layer_ids[i] = l

    def materialize_unit_transforms(self):
        materialized_layers: dict[tuple[tuple[int, str], ConcreteShape], tuple[int, str]] = {}

        for batch_id, batch in self.network.batches.items():
            new_layers_ordered = list(batch.layers.keys())
            new_layers_dict = batch.layers

            _gen = self._backiter_layers(batch_id, batch)
            try:
                owner, ref, shape = next(_gen)
                while True:
                    t, l = ref

                    # we need to establish the proper shape of this layer!
                    # if not already in reshaped_layers, create a new layer/fact with known shape
                    if (ref, shape) not in materialized_layers:
                        # First check if the shape is already known. If it is, escape early.
                        curr_shape = self._get_current_shape(ref, new_layers_dict)
                        if isinstance(curr_shape, ConcreteShape):
                            if t == LayerRefs.TYPE_FACT:
                                owner, ref, shape = _gen.send((ref, None))
                            elif t == LayerRefs.TYPE_LAYER:
                                owner, ref, shape = _gen.send((ref, new_layers_dict[l]))
                            else:
                                assert False
                            continue

                        if t == LayerRefs.TYPE_FACT:
                            orig_layer = self.network.fact_layers[l]
                            new_layer_id = self._get_unused_name(
                                self.network.fact_layers, l + "__" + "_".join((str(v) for v in shape))
                            )

                            eye_shape: bool = len(shape) == 2 and shape[0] == shape[1]

                            new_layer = FactLayer(
                                facts=[
                                    EyeFact(dim=shape[0]) if eye_shape and isinstance(f, UnitFact) else f
                                    for f in orig_layer.facts
                                ],
                                count=orig_layer.count,
                                shape=shape,
                            )
                        elif t == LayerRefs.TYPE_LAYER:
                            orig_layer = new_layers_dict[l]
                            new_layer_id = self._get_unused_name(
                                new_layers_ordered, l + "__" + "_".join((str(v) for v in shape))
                            )

                            new_layer = copy.deepcopy(orig_layer)
                            new_layer.shape = shape
                        else:
                            assert False

                        materialized_layers[ref, shape] = new_layer_ref = (t, new_layer_id)

                        if t == LayerRefs.TYPE_FACT:
                            assert isinstance(new_layer, FactLayer)
                            self.network.fact_layers[new_layer_id] = new_layer
                        elif t == LayerRefs.TYPE_LAYER:
                            assert isinstance(new_layer, Layer)
                            new_layers_ordered.insert(new_layers_ordered.index(l), new_layer_id)
                            new_layers_dict[new_layer_id] = new_layer
                        else:
                            assert False
                    else:
                        new_layer_ref = materialized_layers[ref, shape]
                        _, new_layer_id = new_layer_ref

                    # replace references in owner
                    self._replace_refs(owner, ref, new_layer_ref)

                    if t == LayerRefs.TYPE_FACT:
                        owner, ref, shape = _gen.send((new_layer_ref, None))
                    elif t == LayerRefs.TYPE_LAYER:
                        owner, ref, shape = _gen.send((new_layer_ref, new_layers_dict[new_layer_id]))
                    else:
                        assert False
            except StopIteration:
                pass

            # replace the layers
            batch.layers = OrderedDict(((k, new_layers_dict[k]) for k in new_layers_ordered))


def materialize_unit_transforms(network: VectorizedLayerNetwork):
    MaterializeUnitTransforms(network).materialize_unit_transforms()
    return network
