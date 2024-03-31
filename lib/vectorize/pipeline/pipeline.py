from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.layerwise import Layerwise, LayerwiseSeq
from lib.vectorize.pipeline.materialize_unit_facts import materialize_unit_facts
from lib.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from lib.vectorize.pipeline.simplify_gathers import SimplifyGathers
from lib.vectorize.pipeline.simplify_linears import SimplifyLinears
from lib.vectorize.pipeline.simplify_pure_unit_fact_linears import (
    SimplifyPureUnitFactLinears,
)
from lib.vectorize.pipeline.to_seq_network import to_seq_network
from lib.vectorize.pipeline.utils.pipe import PIPE


def _print(a):
    print(a)
    return a


build_vectorized_network = (
    PIPE  #
    + build_initial_network
    + merge_unit_facts
    # + drop_unused_neurons  # TODO
    + compute_layer_counts
    + _print
    # + transpose_fixed_count_linears  # <- optional
    # + extract_unit_ordinals
    + LayerwiseSeq(
        SimplifyPureUnitFactLinears,
        ConcatInputsLayers,  # <- gathers are expected starting here
    )
    + _print
    # + convert_linears_to_unique  # compute just unique pairs, and add final gather
    + Layerwise(SimplifyLinears)  # <- 'optimize' gather pairs on K_subseq == period
    # + optimize_tail_gathers  # <- those with view at the end might require some special treatment?
    + _print
    + Layerwise(SimplifyGathers)
    + compute_layer_shapes  # <- shapes are expected starting here
    # + merge_weights
    + materialize_unit_facts
    # + precompute_pure_fact_layers
    + give_unique_names
    + _print
    + to_seq_network
    + _print
)
