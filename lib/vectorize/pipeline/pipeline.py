import copy
from typing import Callable, TypeVar

from lib.sources.base import Network
from lib.utils import Blacklist
from lib.vectorize.model import VectorizedOpSeqNetwork
from lib.vectorize.model.gather import Repeat, RepeatInterleave
from lib.vectorize.model.network import VectorizedLayerNetwork
from lib.vectorize.model.settings import VectorizeSettings
from lib.vectorize.pipeline.build_initial_network import build_initial_network
from lib.vectorize.pipeline.compute_layer_counts import ComputeLayerCounts, compute_layer_counts
from lib.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes
from lib.vectorize.pipeline.concat_inputs_layers import ConcatInputsLayers, concat_inputs_layers
from lib.vectorize.pipeline.dissolve_identity_layers import dissolve_identity_layers, predissolve_identity_layers
from lib.vectorize.pipeline.drop_linear_gather_layer_base import drop_linear_gather_layer_base
from lib.vectorize.pipeline.drop_redundant_views import drop_redundant_views
from lib.vectorize.pipeline.drop_unused_layers import drop_unused_layers
from lib.vectorize.pipeline.give_unique_names import give_unique_names
from lib.vectorize.pipeline.iso_compress import ForwardPassRunner, build_iso_compression_factory
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
    OptimizeTailRefsToUniqueNoOrdRemap,
    RemapOrdinals,
    build_optimize_tail_refs_to_unique_no_ord_remap_factory,
)
from lib.vectorize.pipeline.separate_input_refs import (
    ShapeLayerIndexer,
    WeightLayerIndexer,
    build_combined_layer_indexer_factory,
    build_separate_input_refs,
)
from lib.vectorize.pipeline.simplify_gathers import build_simplify_gathers_factory
from lib.vectorize.pipeline.simplify_pure_unit_fact_linears import (
    SimplifyPureUnitFactLinears,
)
from lib.vectorize.pipeline.to_seq_network import to_seq_network
from lib.vectorize.pipeline.transpose_fixed_reduce_layers import TransposeFixedCountReduceLayers
from lib.vectorize.pipeline.utils.pipe import PIPE

_T = TypeVar("_T")


def _create_printer(enabled: bool):
    if enabled:

        def _print(a: _T) -> _T:
            print(a)
            return a

        return _print

    return PIPE


def _deepcopy_vectorized(network: VectorizedLayerNetwork) -> VectorizedLayerNetwork:
    return copy.deepcopy(network)


_simple_tail = (
    PIPE
    + _deepcopy_vectorized
    + compute_layer_counts
    + compute_layer_shapes
    + concat_inputs_layers
    + materialize_unit_transforms
    + drop_unused_layers
    + give_unique_names
    + to_seq_network
)


def create_vectorized_network_compiler(
    settings: VectorizeSettings, forward_pass_runner: ForwardPassRunner, debug_prints: bool = False
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
    )

    if settings.transpose_fixed_count_reduce:
        build_vectorized_network += (
            PIPE  #
            + Layerwise(TransposeFixedCountReduceLayers)
            + _debug
        )

    build_vectorized_network += (
        PIPE  #
        + build_separate_input_refs(indexer_factory)
        + _debug
        + compute_layer_counts
        + _debug
        + Layerwise(SimplifyPureUnitFactLinears)
        + _debug
    )

    if settings.iso_compression:
        build_vectorized_network += (
            PIPE  #
            + Layerwise(
                build_iso_compression_factory(_simple_tail, forward_pass_runner),
            )
            + Layerwise(
                RemapOrdinals,
            )
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
            remaps += build_optimize_linears_pad_for_symmetries(
                settings.linears_pad_for_symmetries,
                transpose=settings.transpose_fixed_count_reduce,
                max_refs_nogather_uniq=settings.max_nogather_simple_layer_refs_length,
            )
            if debug_prints:
                remaps += LayerwisePrint

    if settings.optimize_tail_refs:
        remaps += build_optimize_tail_refs_to_unique_no_ord_remap_factory(
            unique_margin_rate=settings.optimize_tail_refs_unique_margin_rate
        )
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
        + drop_linear_gather_layer_base
        + _debug
        # + drop_unused_neurons_no_ord_remap
        # + _debug
        # + Layerwise(RemapOrdinals)
        # + _debug
        + Layerwise(ConcatInputsLayers)  # <- gathers are expected starting here
        + _debug
        # + Layerwise(MarkCompilableLayers)
        # + _debug
    )

    if settings.optimize_single_use_gathers and settings.optimize_single_use_gathers_before_symmetries:
        build_vectorized_network += (
            PIPE  #
            + build_optimize_single_use_gathers(
                margin=settings.optimize_single_use_gathers_margin,
                margin_rate=settings.optimize_single_use_gathers_margin_rate,
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
                margin=settings.optimize_single_use_gathers_margin,
                margin_rate=settings.optimize_single_use_gathers_margin_rate,
                max_chain_length=settings.optimize_single_use_gathers_aggressive_max_chain_length,
                propagate_through_symmetries=settings.optimize_single_use_gathers_aggressive_through_symmetries,
                debug=debug_prints,
            )
            + compute_layer_counts
            + _debug
        )

    build_vectorized_network += (
        PIPE  #
        + Layerwise(
            build_simplify_gathers_factory(
                max_nogather_simple_layer_refs_length=settings.max_nogather_simple_layer_refs_length,
                whitelist=None if settings.allow_repeat_gathers else Blacklist({Repeat, RepeatInterleave}),
            )
        )
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
        + drop_redundant_views
        + _debug
    )

    return build_vectorized_network
