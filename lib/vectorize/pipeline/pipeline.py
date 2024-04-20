from typing import Callable, TypeVar

from lib.sources.base import Network
from lib.vectorize.model import VectorizedOpSeqNetwork
from lib.vectorize.model.settings import VectorizeSettings
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts, compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers
from lib.vectorize.pipeline.dissolve_identity_layers import dissolve_identity_layers
from lib.vectorize.pipeline.drop_unused_layers import drop_unused_layers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.join_simple_layer_chains import join_simple_layer_chains
from lib.vectorize.pipeline.layerwise import Layerwise, LayerwisePrint
from lib.vectorize.pipeline.lift_symmetrical_linears import LiftSymmetricalLinears
from lib.vectorize.pipeline.materialize_unit_transforms import materialize_unit_transforms
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
from lib.vectorize.pipeline.separate_input_refs import ShapeLayerIndexer, build_separate_input_refs
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
        + _debug
        + merge_unit_facts
        # + drop_unused_neurons  # TODO
        # + transpose_fixed_count_linears  # <- optional
        # + extract_unit_ordinals
        + _debug
        + compute_layer_shapes  # <- shapes are expected starting here
        + _debug
        + build_separate_input_refs(ShapeLayerIndexer)
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

        remaps += LiftSymmetricalLinears
        if debug_prints:
            remaps += LayerwisePrint

    if settings.optimize_tail_refs:
        remaps += OptimizeTailRefsToUniqueNoOrdRemap
        if debug_prints:
            remaps += LayerwisePrint

    if len(remaps) > 0:
        remaps += ComputeLayerCounts

    # ------

    build_vectorized_network += (
        PIPE  #
        + remaps
        + _debug
        + Layerwise(ClearOrdinalsMap)
        + Layerwise(ConcatInputsLayers)  # <- gathers are expected starting here
        + _debug
        + compute_layer_counts
        + _debug
    )

    if settings.optimize_single_use_gathers:
        build_vectorized_network += (
            PIPE  #
            + build_optimize_single_use_gathers(
                max_chain_length=settings.optimize_single_use_gathers_aggressive_max_chain_length,
                debug=debug_prints,
            )
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
