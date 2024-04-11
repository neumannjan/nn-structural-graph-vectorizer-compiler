from typing import Callable, TypeVar

from lib.sources.base import Network
from lib.vectorize.model import VectorizedOpSeqNetwork
from lib.vectorize.model.settings import VectorizeSettings
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.dissolve_identity_layers import dissolve_identity_layers
from lib.vectorize.pipeline.drop_unused_layers import drop_unused_layers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.join_simple_layer_chains import join_simple_layer_chains
from lib.vectorize.pipeline.layerwise import Layerwise
from lib.vectorize.pipeline.materialize_unit_transforms import materialize_unit_transforms
from lib.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from lib.vectorize.pipeline.optimize_k_sequence_refs_in_linears import OptimizeKSeqRefsInLinears
from lib.vectorize.pipeline.optimize_linears_to_unique_refs import OptimizeLinearsUniqueRefPairs
from lib.vectorize.pipeline.optimize_single_use_gathers import (
    build_optimize_single_use_gathers,
)
from lib.vectorize.pipeline.optimize_tail_refs_to_unique import (
    ClearOrdinalsMap,
    OptimizeTailRefsToUniqueNoOrdRemap,
    RemapOrdinals,
)
from lib.vectorize.pipeline.simplify_gathers import SimplifyGathers
from lib.vectorize.pipeline.simplify_pure_unit_fact_linears import (
    SimplifyPureUnitFactLinears,
)
from lib.vectorize.pipeline.to_seq_network import to_seq_network
from lib.vectorize.pipeline.utils.pipe import PIPE

_T = TypeVar("_T")


def _create_printer(enabled: bool):
    if enabled:

        def _print(a: _T) -> _T:
            print(a)
            return a

        return _print

    return PIPE


def create_vectorized_network_compiler(
    settings: VectorizeSettings, debug_prints: bool = False
) -> Callable[[Network], VectorizedOpSeqNetwork]:
    _debug = _create_printer(debug_prints)

    # ------

    build_vectorized_network = (
        PIPE  #
        + build_initial_network
        + merge_unit_facts
        # + drop_unused_neurons  # TODO
        # + transpose_fixed_count_linears  # <- optional
        # + extract_unit_ordinals
        + _debug
        + Layerwise(SimplifyPureUnitFactLinears)
    )

    # ------

    if settings.linears_optimize_repeating_seq and not settings.linears_optimize_unique_ref_pairs_aggressively:
        # 'optimize' gather pairs on K_subseq == period
        build_vectorized_network += Layerwise(OptimizeKSeqRefsInLinears)

    # ------

    if settings.linears_optimize_unique_ref_pairs and settings.optimize_tail_refs:
        build_vectorized_network += Layerwise(
            RemapOrdinals,  # this is 'optimize_tail_refs'
            OptimizeLinearsUniqueRefPairs,
            OptimizeTailRefsToUniqueNoOrdRemap,  # this is 'optimize_tail_refs'
        )

    elif settings.optimize_tail_refs:
        build_vectorized_network += Layerwise(
            RemapOrdinals,  # this is 'optimize_tail_refs'
            OptimizeTailRefsToUniqueNoOrdRemap,  # this is 'optimize_tail_refs'
        )
    elif settings.linears_optimize_unique_ref_pairs:
        build_vectorized_network += Layerwise(OptimizeLinearsUniqueRefPairs)

    # ------

    if settings.linears_optimize_repeating_seq and settings.linears_optimize_unique_ref_pairs_aggressively:
        build_vectorized_network += Layerwise(OptimizeKSeqRefsInLinears)

    # ------

    build_vectorized_network += (
        PIPE
        + compute_layer_counts
        + _debug
        + Layerwise(ClearOrdinalsMap)
        + Layerwise(ConcatInputsLayers)  # <- gathers are expected starting here
        + Layerwise(SimplifyGathers)
        + _debug
        + dissolve_identity_layers
        + _debug
        # + precompute_pure_fact_layers
        # + preorder_single_use_outputs
        + compute_layer_shapes  # <- shapes are expected starting here
        + _debug
    )

    if settings.optimize_single_use_gathers:
        build_vectorized_network += (
            PIPE
            + build_optimize_single_use_gathers(
                max_chain_length=settings.optimize_single_use_gathers_aggressive_max_chain_length,
                debug=debug_prints,
            )
            + _debug
            + Layerwise(SimplifyGathers)
            + _debug
            + dissolve_identity_layers
            + _debug
        )

    build_vectorized_network += (
        PIPE
        + materialize_unit_transforms
        + drop_unused_layers
        + _debug
        # + merge_weights
        + give_unique_names
        + to_seq_network
        + _debug
        + join_simple_layer_chains
        + _debug
    )

    return build_vectorized_network
