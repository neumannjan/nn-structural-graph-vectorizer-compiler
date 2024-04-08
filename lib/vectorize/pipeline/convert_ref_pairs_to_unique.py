from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation

_Refs = Refs | list[int]


class ConvertRefPairsToUnique(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _for_pairs(
        self,
        refs: _Refs,
        weight_refs: _Refs,
    ) -> tuple[_Refs, _Refs, GenericGather | None]:
        pairs = list(
            zip(
                refs,
                weight_refs,
            )
        )
        pairs_set = set(pairs)

        if len(pairs) == len(pairs_set):
            return refs, weight_refs, None

        new_pairs = sorted(pairs_set)

        if isinstance(refs, Refs):
            new_refs = Refs(
                types=[p[0] for p, _ in new_pairs],  # pyright: ignore
                layer_ids=[p[1] for p, _ in new_pairs],  # pyright: ignore
                ordinals=[p[2] for p, _ in new_pairs],  # pyright: ignore
            )
        else:
            new_refs = [p for p, _ in new_pairs]

        if isinstance(weight_refs, Refs):
            new_weight_refs = Refs(
                types=[p[0] for _, p in new_pairs],  # pyright: ignore
                layer_ids=[p[1] for _, p in new_pairs],  # pyright: ignore
                ordinals=[p[2] for _, p in new_pairs],  # pyright: ignore
            )
        else:
            new_weight_refs = [p for _, p in new_pairs]

        new_ordinal_pairs_idx_map = {pair: i for i, pair in enumerate(new_pairs)}

        final_gather = GenericGather([new_ordinal_pairs_idx_map[pair] for pair in pairs])
        return new_refs, new_weight_refs, final_gather  # pyright: ignore

    def _extract_refs(self, input: Input):
        match input:
            case Refs():
                return input
            case GatheredLayers(refs=_, gather=GenericGather(ordinals)):
                return ordinals
            case _:
                assert False, f"{input}"

    def _apply_refs(self, refs: _Refs, input: Input):
        match (refs, input):
            case Refs(), Refs():
                return refs
            case list(), GatheredLayers(refs=_, gather=GenericGather() as gather):
                gather.ordinals = refs
                return input
            case _:
                assert False, f"{refs, input}"

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        match layer.base:
            case InputLayerBase(input=input):
                # nothing to do
                return layer
            case LinearLayerBase(input=input, weight=weight):
                refs, weight_refs, final_gather = self._for_pairs(self._extract_refs(input), self._extract_refs(weight))
                if final_gather is not None:
                    input = self._apply_refs(refs, input)
                    weight = self._apply_refs(weight_refs, weight)
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
    ConvertRefPairsToUnique(network).convert_linears_to_unique()
    return network
