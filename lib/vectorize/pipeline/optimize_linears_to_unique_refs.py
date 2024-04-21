from typing import Iterable, OrderedDict, TypeVar

from lib.vectorize.model import *
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts
from lib.vectorize.pipeline.layerwise import LayerwiseOperation
from lib.vectorize.pipeline.utils.gather import combine_gathers_
from lib.vectorize.pipeline.utils.ref_groups import (
    apply_refs_to_target,
)

# def _sort_pairs_by_weight_counts(pairs: Iterable[tuple[_Ref, _Ref]]) -> list[tuple[_Ref, _Ref]]:
#     weight_refs = [p[_POS_WEIGHT_REFS] for p in pairs]
#     _, uniq_idx, counts = np.unique(weight_refs, axis=0, return_index=True, return_counts=True)
#     _idx = np.argsort(counts, axis=0)
#     uniq_idx = uniq_idx[_idx]
#     uniq_map = {weight_refs[v_idx]: i for i, v_idx in enumerate(uniq_idx)}
#     out = sorted(pairs, key=lambda p: (-uniq_map[p[_POS_WEIGHT_REFS]], p[_POS_REFS]))
#     return out


_T = TypeVar("_T")


_POS_WEIGHT_REFS = 0
_POS_REFS = 1


def get_pairs(refs: Iterable[_T], weight_refs: Iterable[_T]):
    return zip(weight_refs, refs)


def get_unique(values: Iterable[_T]) -> list[_T]:
    # out = list(OrderedDict.fromkeys(values))
    out = sorted(set(values))  # pyright: ignore
    return out


class OptimizeLinearsUniqueRefPairs(LayerwiseOperation):
    def __init__(self, network: VectorizedLayerNetwork) -> None:
        self.network = network
        self._counts = ComputeLayerCounts(network)

    def _remap_ref_pairs(self, batch: int, refs: Refs, wrefs: Refs) -> GenericGather | None:
        new_pairs = get_unique(get_pairs(refs, wrefs))

        if new_pairs is None:
            return None

        if max(len(refs), len(wrefs)) <= len(new_pairs):
            return None

        out_refs = Refs.from_iter((p[_POS_REFS] for p in new_pairs))
        out_weight_refs = Refs.from_iter((p[_POS_WEIGHT_REFS] for p in new_pairs))

        new_ordinal_pairs_idx_map = {pair: i for i, pair in enumerate(new_pairs)}
        final_gather = GenericGather([new_ordinal_pairs_idx_map[pair] for pair in get_pairs(refs, wrefs)])

        apply_refs_to_target(out_refs, refs)
        apply_refs_to_target(out_weight_refs, wrefs)
        return final_gather

    def __call__(self, batch: int, layer_id: str, layer: Layer) -> Layer:
        if layer_id == "l1_embed__wa":
            # TODO remove
            pass

        match layer:
            case Layer(base=InputLayerBase(input=input)):
                # nothing to do
                return layer
            case Layer(
                base=(
                    LinearLayerBase(input=Refs() as input, weight=Refs() as weight, lifts=lifts)
                    | LinearGatherLayerBase(
                        input=Refs() as input, weight=Refs() as weight, gather=NoopGather(), lifts=lifts
                    )
                )
            ):
                if lifts is not None:
                    # skipping
                    return layer

                final_gather = self._remap_ref_pairs(batch, input, weight)
                if final_gather is not None:
                    layer.base = LinearGatherLayerBase(input=input, weight=weight, gather=final_gather, lifts=None)
                return layer
            case Layer(
                base=LinearGatherLayerBase(
                    input=Refs() as input, weight=Refs() as weight, gather=GenericGather() as gather2, lifts=lifts
                )
            ):
                if lifts is not None:
                    # skipping
                    return layer

                gather1 = self._remap_ref_pairs(batch, input, weight)
                if gather1 is not None:
                    combine_gathers_(gather1, gather2)
                return layer
            case _:
                raise ValueError(layer.base)

    def convert_linears_to_unique(self):
        for bid, batch in self.network.batches.items():
            for lid, layer in batch.layers.items():
                batch.layers[lid] = self(bid, lid, layer)


def convert_linears_to_unique(network: VectorizedLayerNetwork):
    OptimizeLinearsUniqueRefPairs(network).convert_linears_to_unique()
    return network
