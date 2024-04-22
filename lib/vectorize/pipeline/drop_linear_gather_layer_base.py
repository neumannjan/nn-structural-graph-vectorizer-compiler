from typing import OrderedDict, Sequence

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts


class DropLinearGatherLayerBase:
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _for_layer(self, batch: int, layer_id: str, layer: Layer) -> Sequence[tuple[str, Layer]] | None:
        match layer.base:
            case LinearGatherLayerBase(input=input, weight=weight, gather=GenericGather(ordinals), lifts=lifts):
                prev_layer_id = layer_id + "_pregather"

                refs2 = Refs(
                    types=[Refs.TYPE_LAYER] * len(ordinals),
                    layer_ids=[prev_layer_id] * len(ordinals),
                    ordinals=[o for o in ordinals],
                )

                layer1 = Layer(
                    base=LinearLayerBase(
                        input=input,
                        weight=weight,
                        lifts=lifts,
                    ),
                    aggregate=Noop(),
                    transform=Transform("identity"),
                    count=self._counts.compute_layer_base_count(batch, layer.base),
                    shape=layer.shape,
                    compilable=layer.compilable,
                )

                layer2 = Layer(
                    base=InputLayerBase(input=refs2),
                    aggregate=layer.aggregate,
                    transform=layer.transform,
                    count=layer.count,
                    shape=layer.shape,
                    compilable=layer.compilable,
                )

                return [(prev_layer_id, layer1), (layer_id, layer2)]
            case LinearGatherLayerBase():
                raise ValueError(layer.base)
            case _:
                return None

    def drop_linear_gather_layer_base(self):
        for batch_id, batch in self.network.batches.items():
            layer_ids_out = list(batch.layers)

            for layer_id, layer in list(batch.layers.items()):
                out_layers = self._for_layer(batch_id, layer_id, layer)

                if out_layers is not None:
                    idx = layer_ids_out.index(layer_id)
                    layer_ids_out[idx : idx + 1] = [lid for lid, _ in out_layers]
                    for lid2, layer2 in out_layers:
                        batch.layers[lid2] = layer2

            batch.layers = OrderedDict(((k, batch.layers[k]) for k in layer_ids_out))


def drop_linear_gather_layer_base(network: VectorizedLayerNetwork):
    DropLinearGatherLayerBase(network).drop_linear_gather_layer_base()
    return network
