from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.convert_ref_pairs_to_unique import ConvertRefPairsToUnique
from lib.vectorize.pipeline.convert_refs_to_unique import ClearOrdinalsMap, ConvertRefsToUniqueNoOrdRemap, RemapOrdinals
from lib.vectorize.pipeline.dissolve_identity_layers import dissolve_identity_layers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.join_simple_layer_chains import join_simple_layer_chains
from lib.vectorize.pipeline.layerwise import Layerwise, LayerwisePrint, LayerwiseSeq
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
    # + transpose_fixed_count_linears  # <- optional
    # + extract_unit_ordinals
    + _print
    + LayerwiseSeq(
        RemapOrdinals,  # this is 'optimize_tail_gathers'
        SimplifyPureUnitFactLinears,
        ConvertRefPairsToUnique,
        ConvertRefsToUniqueNoOrdRemap,  # this is 'optimize_tail_gathers'
    )
    + compute_layer_counts
    + _print
    + Layerwise(ClearOrdinalsMap)
    + LayerwiseSeq(
        ConcatInputsLayers,  # <- gathers are expected starting here
        SimplifyLinears,  # <- 'optimize' gather pairs on K_subseq == period
    )
    + Layerwise(SimplifyGathers)
    + _print
    + dissolve_identity_layers
    + _print
    + compute_layer_shapes  # <- shapes are expected starting here
    + _print
    # + merge_weights
    + materialize_unit_facts
    # + precompute_pure_fact_layers
    + give_unique_names
    + to_seq_network
    + _print
    + join_simple_layer_chains
    + _print
)
