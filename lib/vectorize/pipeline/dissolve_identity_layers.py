import itertools

from lib.vectorize.model import *


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
