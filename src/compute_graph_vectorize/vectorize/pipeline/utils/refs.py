from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts


def build_refs_for_layer(counts: ComputeLayerCounts, batch: int, layer_id: str, layer: Layer):
    count = counts.compute_layer_count(batch, layer)

    return Refs(
        types=[Refs.TYPE_LAYER] * count,
        layer_ids=[layer_id] * count,
        ordinals=list(range(count)),
    )
