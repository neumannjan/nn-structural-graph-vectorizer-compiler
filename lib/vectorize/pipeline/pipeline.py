from lib.sources.base import Network
from lib.vectorize.model.network import VectorizedNetwork
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.layerwise import LayerwiseSeq
from lib.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from lib.vectorize.pipeline.simplify_pure_unit_fact_linears import (
    SimplifyPureUnitFactLinears,
)
from lib.vectorize.pipeline.utils.pipe import PIPE


def _print(a):
    print(a)
    return a


def build_vectorized_network(network: Network) -> VectorizedNetwork:
    compile = (
        PIPE  #
        + build_initial_network
        + _print
        + merge_unit_facts
        + _print
        # + drop_unused_neurons  # TODO
        + compute_layer_counts
        # + simplify_gathers
        # + transpose_fixed_count_linears  # <- optional
        # + extract_unit_ordinals
        + LayerwiseSeq(
            SimplifyPureUnitFactLinears,
            ConcatInputsLayers,  # <- gathers are expected starting here
        )
        # + convert_linears_to_unique  # compute just unique pairs, and add final gather
        # + simplify_linears  # <- 'optimize' gather pairs on K_subseq == period
        # + optimize_tail_gathers  # <- those with view at the end might require some special treatment?
        + compute_layer_shapes  # <- shapes are expected starting here
        # + merge_weights
        # + materialize_unit_facts
        # + precompute_pure_fact_layers
    )

    return compile(network)
