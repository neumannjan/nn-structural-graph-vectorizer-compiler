from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation


class ConvertLinearsToUnique(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _for_ordinal_pairs(
        self,
        ordinals: list[int],
        weight_ordinals: list[int],
    ) -> tuple[list[int], list[int], GenericGather | None]:
        ordinal_pairs = list(zip(ordinals, weight_ordinals))
        ordinal_pairs_set = set(ordinal_pairs)

        if len(ordinal_pairs) == len(ordinal_pairs_set):
            return ordinals, weight_ordinals, None

        new_ordinal_pairs = sorted(ordinal_pairs_set)

        new_ordinals = [o for o, _ in new_ordinal_pairs]
        new_weight_ordinals = [o for _, o in new_ordinal_pairs]

        new_ordinal_pairs_idx_map = {pair: i for i, pair in enumerate(new_ordinal_pairs)}

        final_gather = GenericGather([new_ordinal_pairs_idx_map[pair] for pair in ordinal_pairs])
        return new_ordinals, new_weight_ordinals, final_gather

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer.base:
            case InputLayerBase(input=input):
                # nothing to do
                return layer
            case LinearLayerBase(
                input=GatheredLayers(refs=_, gather=GenericGather(ordinals)) as input,
                weight=GatheredLayers(refs=_, gather=GenericGather(weight_ordinals)) as weight,
            ):
                ordinals, weight_ordinals, final_gather = self._for_ordinal_pairs(ordinals, weight_ordinals)
                if final_gather is not None:
                    input.gather = GenericGather(ordinals)
                    weight.gather = GenericGather(weight_ordinals)

                    layer.base = LinearGatherLayerBase(input=input, weight=weight, gather=final_gather)

                return layer
            case LinearGatherLayerBase():
                raise NotImplementedError("Concatenation of multiple gathers not yet implemented.")
            case _:
                assert False, f"{layer.base}"

    def convert_linears_to_unique(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                batch.layers[lid] = self(bid, lid, layer)


def convert_linears_to_unique(network: VectorizedLayerNetwork):
    ConvertLinearsToUnique(network).convert_linears_to_unique()
    return network
