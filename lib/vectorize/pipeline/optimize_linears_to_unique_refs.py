from collections.abc import Iterable

import numpy as np

from lib.vectorize.model import *
from lib.vectorize.pipeline.layerwise import LayerwiseOperation

_Refs = Refs | list[int]
_Ref = int | tuple[int, str, int]


_POS_REFS = 1
_POS_WEIGHT_REFS = 0


def _get_pairs(refs: _Refs, weight_refs: _Refs) -> list[tuple[_Ref, _Ref]]:
    return list(zip(weight_refs, refs))


# def _sort_pairs_by_weight_counts(pairs: Iterable[tuple[_Ref, _Ref]]) -> list[tuple[_Ref, _Ref]]:
#     weight_refs = [p[_POS_WEIGHT_REFS] for p in pairs]
#     _, uniq_idx, counts = np.unique(weight_refs, axis=0, return_index=True, return_counts=True)
#     _idx = np.argsort(counts, axis=0)
#     uniq_idx = uniq_idx[_idx]
#     uniq_map = {weight_refs[v_idx]: i for i, v_idx in enumerate(uniq_idx)}
#     out = sorted(pairs, key=lambda p: (-uniq_map[p[_POS_WEIGHT_REFS]], p[_POS_REFS]))
#     return out


class OptimizeLinearsUniqueRefPairs(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network

    def _for_pairs(
        self,
        refs: _Refs,
        weight_refs: _Refs,
    ) -> tuple[_Refs, _Refs, GenericGather | None]:
        pairs = _get_pairs(refs, weight_refs)
        # new_pairs = list(OrderedDict.fromkeys(pairs))
        new_pairs = sorted(set(pairs))

        if len(pairs) == len(new_pairs):
            return refs, weight_refs, None

        if isinstance(refs, Refs):
            new_refs = Refs(
                types=[p[_POS_REFS][0] for p in new_pairs],  # pyright: ignore
                layer_ids=[p[_POS_REFS][1] for p in new_pairs],  # pyright: ignore
                ordinals=[p[_POS_REFS][2] for p in new_pairs],  # pyright: ignore
            )
        else:
            new_refs = [p[_POS_REFS] for p in new_pairs]

        if isinstance(weight_refs, Refs):
            new_weight_refs = Refs(
                types=[p[_POS_WEIGHT_REFS][0] for p in new_pairs],  # pyright: ignore
                layer_ids=[p[_POS_WEIGHT_REFS][1] for p in new_pairs],  # pyright: ignore
                ordinals=[p[_POS_WEIGHT_REFS][2] for p in new_pairs],  # pyright: ignore
            )
        else:
            new_weight_refs = [p[_POS_WEIGHT_REFS] for p in new_pairs]

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
        match layer:
            case Layer(base=InputLayerBase(input=input)):
                # nothing to do
                return layer
            case Layer(base=LinearLayerBase(input=input, weight=weight)):
                refs, weight_refs, final_gather = self._for_pairs(self._extract_refs(input), self._extract_refs(weight))
                if final_gather is not None:
                    input = self._apply_refs(refs, input)
                    weight = self._apply_refs(weight_refs, weight)
                    layer.base = LinearGatherLayerBase(input=input, weight=weight, gather=final_gather)
                return layer
            case Layer(base=LinearGatherLayerBase()):
                raise NotImplementedError("Concatenation of multiple gathers not yet implemented.")
            case _:
                assert False, f"{layer.base}"

    def convert_linears_to_unique(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                batch.layers[lid] = self(bid, lid, layer)


def convert_linears_to_unique(network: VectorizedLayerNetwork):
    OptimizeLinearsUniqueRefPairs(network).convert_linears_to_unique()
    return network
