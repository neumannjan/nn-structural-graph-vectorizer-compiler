import itertools
from typing import Iterable, Literal

from lib.utils import head_and_rest
from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts


class PreDissolveIdentityLayers:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _get_identity_refs_origin(self, batch: int, refs: Refs) -> str | None:
        (t0, l0, o0), rest = head_and_rest(refs)

        if t0 != Refs.TYPE_LAYER:
            return None

        if o0 != 0:
            return None

        o = o0
        for i, (t, l, o) in enumerate(rest, 1):
            if i != o or t != t0 or l != l0:
                return None

        if self._counts.compute_layer_ref_count(batch, t0, l0) != o + 1:
            return None

        return l0

    def _get_refs(self, layer: Layer) -> Refs:
        match layer:
            case Layer(base=InputLayerBase(input=Refs() as refs)):
                return refs
            case Layer(base=LinearLayerBase(input=Refs() as refs)):
                return refs
            case Layer(base=LinearGatherLayerBase(input=Refs() as refs)):
                return refs
            case _:
                raise ValueError(layer)

    def _get_layer_identity_origin(self, batch: int, layer: Layer) -> str | None:
        return self._get_identity_refs_origin(batch, self._get_refs(layer))

    def _get_layer_refs_set(self, layer: Layer) -> set[str]:
        return set(l for t, l, _ in self._get_refs(layer) if t == Refs.TYPE_LAYER)

    def _get_layer_level_top(self, layer: Layer) -> int:
        if layer.transform.transform != "identity":
            return 0b111

        match layer.aggregate:
            case Noop():
                pass
            case _:
                return 0b110

        match layer.base:
            case LinearLayerBase() | LinearGatherLayerBase():
                return 0b100

        return 0b000

    def _get_layer_level_bottom(self, layer: Layer) -> int:
        match layer.base:
            case LinearLayerBase() | LinearGatherLayerBase():
                return 0b111

        match layer.aggregate:
            case Noop():
                pass
            case _:
                return 0b011

        if layer.transform.transform != "identity":
            return 0b001

        return 0b000

    def _levels_mergable(self, origin: int, level: int) -> bool:
        return origin & level == 0

    def _merge_bases(self, layer: Layer, origin_layer: Layer):
        match (layer.base, origin_layer.base):
            case (InputLayerBase(), _):
                layer.base = origin_layer.base
            case (_, InputLayerBase()):
                layer.base.input = origin_layer.base.input
            case (_, _):
                raise ValueError(layer.base, origin_layer.base)

    def _merge_layers(self, layer: Layer, origin_layer: Layer):
        self._merge_bases(layer, origin_layer)
        if not isinstance(origin_layer.aggregate, Noop):
            layer.aggregate = origin_layer.aggregate
        if origin_layer.transform.transform != "identity":
            layer.transform = origin_layer.transform

    def _get_layer_uses(self, layers: Iterable[Layer]) -> dict[str, int]:
        out: dict[str, int] = {}

        for layer in layers:
            for ref_layer_id in self._get_layer_refs_set(layer):
                out[ref_layer_id] = out.get(ref_layer_id, 0) + 1

        return out

    def _merge_all_layers(self, batch: int, layers: dict[str, Layer]):
        layer_id_uses = self._get_layer_uses(layers.values())

        for layer_id, layer in list(layers.items()):
            level = self._get_layer_level_bottom(layer)

            origin_id = self._get_layer_identity_origin(batch, layer)
            if origin_id is not None and layer_id_uses[origin_id] == 1:
                origin_level = self._get_layer_level_top(layers[origin_id])
                if self._levels_mergable(origin_level, level):
                    self._merge_layers(layer, layers[origin_id])
                    del layers[origin_id]

    def predissolve_identity_layers(self):
        for batch_id, batch in self.network.batches.items():
            self._merge_all_layers(batch_id, batch.layers)


def predissolve_identity_layers(network: VectorizedLayerNetwork):
    PreDissolveIdentityLayers(network).predissolve_identity_layers()
    return network


class DissolveIdentityLayers:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._ref_remap: dict[str, LayerRefs] = {}

    def _is_identity_layer(self, layer: Layer) -> bool:
        match layer:
            case Layer(
                base=InputLayerBase(input=GatheredLayers(gather=NoopGather())),
                aggregate=Noop(),
                transform=Transform(transform="identity"),
            ):
                return True
            case _:
                return False

    def _get_identity_refs(self, layer: Layer) -> LayerRefs:
        match layer.base:
            case InputLayerBase(input=GatheredLayers(refs=refs)):
                return refs
            case _:
                assert False, f"{layer.base}"

    def _remap_refs(self, refs: LayerRefs):
        for i in itertools.count():
            if i >= len(refs):
                break

            t, l = refs.types[i], refs.layer_ids[i]

            if t == LayerRefs.TYPE_LAYER:
                refs_sub = self._ref_remap.get(l, None)

                if refs_sub is not None:
                    refs.types[i : i + 1] = refs_sub.types
                    refs.layer_ids[i : i + 1] = refs_sub.layer_ids

    def _remap_refs_in_layer(self, layer: Layer):
        match layer.base:
            case InputLayerBase(input=GatheredLayers(refs=refs)):
                self._remap_refs(refs)
            case LinearLayerBase(input=GatheredLayers(refs=refs), weight=GatheredLayers(refs=wrefs)):
                self._remap_refs(refs)
                self._remap_refs(wrefs)
            case LinearGatherLayerBase(input=GatheredLayers(refs=refs), weight=GatheredLayers(refs=wrefs)):
                self._remap_refs(refs)
                self._remap_refs(wrefs)

    def dissolve_identity_layers(self):
        for batch_id, batch in self.network.batches.items():
            for layer_id in list(batch.layers.keys()):
                layer = batch.layers[layer_id]
                self._remap_refs_in_layer(layer)
                if self._is_identity_layer(layer):
                    refs = self._get_identity_refs(layer)
                    del batch.layers[layer_id]
                    self._ref_remap[layer_id] = refs


def dissolve_identity_layers(network: VectorizedLayerNetwork):
    DissolveIdentityLayers(network).dissolve_identity_layers()
    return network
