from typing import Callable, TypeVar

from lib.sources.base import Network
from lib.vectorize.model import VectorizedOpSeqNetwork
from lib.vectorize.model.settings import VectorizeSettings
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts, compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.dissolve_identity_layers import dissolve_identity_layers, predissolve_identity_layers
from lib.vectorize.pipeline.drop_unused_layers import drop_unused_layers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.join_simple_layer_chains import join_simple_layer_chains
from lib.vectorize.pipeline.layerwise import Layerwise, LayerwisePrint
from lib.vectorize.pipeline.lift_symmetrical_linears import LiftSymmetricalLinears
from lib.vectorize.pipeline.materialize_unit_transforms import materialize_unit_transforms
from lib.vectorize.pipeline.merge_same_value_facts import merge_same_value_facts
from lib.vectorize.pipeline.merge_trivial_layer_concats import merge_trivial_layer_concats
from lib.vectorize.pipeline.merge_unit_facts import merge_unit_facts
from lib.vectorize.pipeline.optimize_linears_pad_for_symmetries import build_optimize_linears_pad_for_symmetries
from lib.vectorize.pipeline.optimize_linears_to_unique_refs import OptimizeLinearsUniqueRefPairs
from lib.vectorize.pipeline.optimize_single_use_gathers import (
    build_optimize_single_use_gathers,
)
from lib.vectorize.pipeline.optimize_tail_refs_to_unique import (
    ClearOrdinalsMap,
    OptimizeTailRefsToUniqueNoOrdRemap,
    RemapOrdinals,
)
from lib.vectorize.pipeline.separate_input_refs import (
    ShapeLayerIndexer,
    WeightLayerIndexer,
    build_combined_layer_indexer_factory,
    build_separate_input_refs,
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

    if settings.granularize_by_weight:
        indexer_factory = build_combined_layer_indexer_factory(ShapeLayerIndexer, WeightLayerIndexer)
    else:
        indexer_factory = ShapeLayerIndexer

    build_vectorized_network = (
        PIPE  #
        + build_initial_network
        + _debug
        + merge_unit_facts
        + merge_same_value_facts
        # + drop_unused_neurons  # TODO
        # + transpose_fixed_count_linears  # <- optional
        # + extract_unit_ordinals
        + _debug
        + compute_layer_counts
        + _debug
        + predissolve_identity_layers
        + _debug
        + compute_layer_shapes  # <- shapes are expected starting here
        + _debug
        + build_separate_input_refs(indexer_factory)
        + _debug
        + Layerwise(SimplifyPureUnitFactLinears)
        + _debug
        + compute_layer_counts
        + _debug
    )

    # ------

    remaps = Layerwise()

    if settings.optimize_tail_refs:
        remaps += RemapOrdinals
        if debug_prints:
            remaps += LayerwisePrint

    if settings.linears_optimize_unique_ref_pairs:
        remaps += OptimizeLinearsUniqueRefPairs
        if debug_prints:
            remaps += LayerwisePrint

    if settings.linears_symmetries:
        if settings.linears_pad_for_symmetries != "never":
            remaps += build_optimize_linears_pad_for_symmetries(settings.linears_pad_for_symmetries)
            if debug_prints:
                remaps += LayerwisePrint

    if settings.optimize_tail_refs:
        remaps += OptimizeTailRefsToUniqueNoOrdRemap
        if debug_prints:
            remaps += LayerwisePrint

    if len(remaps) > 0:
        remaps += ComputeLayerCounts
        if debug_prints:
            remaps += LayerwisePrint

    # ------

    build_vectorized_network += (
        PIPE  #
        + remaps
        + _debug
        + Layerwise(ClearOrdinalsMap)
        + Layerwise(ConcatInputsLayers)  # <- gathers are expected starting here
        + _debug
        # + Layerwise(MarkCompilableLayers)
        # + _debug
    )

    if settings.optimize_single_use_gathers and settings.optimize_single_use_gathers_before_symmetries:
        build_vectorized_network += (
            PIPE  #
            + build_optimize_single_use_gathers(
                max_chain_length=settings.optimize_single_use_gathers_aggressive_max_chain_length,
                propagate_through_symmetries=settings.optimize_single_use_gathers_aggressive_through_symmetries,
                debug=debug_prints,
            )
            + compute_layer_counts
            + _debug
        )

    if settings.linears_symmetries:
        remaps2 = Layerwise(LiftSymmetricalLinears)

        if debug_prints:
            remaps2 += LayerwisePrint

        build_vectorized_network += (
            PIPE  #
            + remaps2
            + _debug
        )

    if settings.optimize_single_use_gathers and not settings.optimize_single_use_gathers_before_symmetries:
        build_vectorized_network += (
            PIPE  #
            + build_optimize_single_use_gathers(
                max_chain_length=settings.optimize_single_use_gathers_aggressive_max_chain_length,
                propagate_through_symmetries=settings.optimize_single_use_gathers_aggressive_through_symmetries,
                debug=debug_prints,
            )
            + compute_layer_counts
            + _debug
        )

    build_vectorized_network += (
        PIPE  #
        + Layerwise(SimplifyGathers)
        + _debug
        + dissolve_identity_layers
        + _debug
        + materialize_unit_transforms
        + _debug
    )

    if settings.merge_trivial_layer_concats:
        build_vectorized_network += (
            PIPE  #
            + merge_trivial_layer_concats
            + _debug
        )

    build_vectorized_network += (
        PIPE
        + drop_unused_layers
        + _debug
        + give_unique_names
        + to_seq_network
        + _debug
        + join_simple_layer_chains
        + _debug
    )

    return build_vectorized_network
