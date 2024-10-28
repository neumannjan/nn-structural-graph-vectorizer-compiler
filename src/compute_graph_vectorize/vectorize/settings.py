from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Literal, TypedDict

from typing_extensions import Unpack

from compute_graph_vectorize.utils import serialize_dataclass

LinearsPadForSymmetriesOption = Literal["never", "sided_only", "full_only", "any"]


class _LinearsSymmetriesSettingsPartial(TypedDict, total=False):
    pad: LinearsPadForSymmetriesOption


@dataclass(frozen=True)
class LinearsSymmetriesSettings:
    pad: LinearsPadForSymmetriesOption = "any"
    """
    Enables the usage of various strategies of reordering + padding of linears' inputs (inputs and weights)
    such that there are more symmetries available to be taken advantage of.

    Chooses the padding/ordering that minimizes the total no. of items gathered for and multiplied in the linear.

    (Symmetry usually significantly reduces the no. of items gathered, but may slightly increase the no. of items
    multiplied due to the added padding.)
    """

    def serialize(self) -> _LinearsSymmetriesSettingsPartial:
        out = _LinearsSymmetriesSettingsPartial(asdict(self))  # pyright: ignore
        return out


class _OptimizeTailRefsSettingsPartial(TypedDict, total=False):
    unique_margin_rate: float


@dataclass(frozen=True)
class OptimizeTailRefsSettings:
    unique_margin_rate: float = 0.05
    """
    To prevent the optimization from ruining some nice existing properties of the network (and producing some
    redundant gathers), this value controls the threshold of significance of the operation count reduction.

    Specifically, will only use the modified (reduced) version of the operation (layer) when
    `1. - (new_count / old_count) > unique_margin_rate`.

    E.g., for `unique_margin_rate == 0.05`, performs the optimization only if the layer size (i.e. the no.
    of neurons in the layer) decreases by at least 5% as a result of the optimization.

    Furthermore, this threshold also controls attempts to remove some gathers added by earlier optimizations, such as
    `linears_optimize_unique_ref_pairs`. When the above inequality evaluates to true for a gather found directly after
    a linear operation, will attempt to remove the gather by asking the following layer to gather the inputs instead.

    This may not be possible if the gather to be removed is immediately followed by an aggregation, where the
    linear operation preceding the gather may not have the results in the correct order needed for the aggregation.
    In such case, the gather can only be propagated upwards (through the linear operation), not downwards. However, this
    is done by the `optimize_single_use_gathers` optimization, not by this one.
    """

    def serialize(self) -> _OptimizeTailRefsSettingsPartial:
        out = _OptimizeTailRefsSettingsPartial(asdict(self))  # pyright: ignore
        return out


OptimizeSingleUseGathersPreset = Literal[
    "free_only",
    "free_only_presym",
    "margin_S",
    "margin_M_presym",
    "margin_M_keepsym",
    "margin_M",
    "margin_L_presym",
    "margin_L_keepsym",
    "margin_L",
    "margin_XL_presym",
    "margin_XL_keepsym",
    "margin_XL",
    "margin_XXL_presym",
    "margin_XXL_keepsym",
    "margin_XXL",
    "agg_max_keepsym",
    "agg_max_propsym",
    "agg_true_unlimited",
]


class _OptimizeSingleUseGathersSettingsPartial(TypedDict, total=False):
    run_before_symmetries: bool
    margin: int
    margin_rate: float
    aggressive_max_chain_depth: int | Literal["unlimited"]
    propagate_through_symmetries: bool


@dataclass(frozen=True)
class OptimizeSingleUseGathersSettings:
    run_before_symmetries: bool = False
    """
    When True, runs this optimization before the `linears_symmetries` optimization. This will allow this optimization to
    be more 'eager', removing more gathers. However, this will likely produce less symmetries, especially those that
    were created by adding extra padding (`linears_symmetries.pad` optimization).
    """

    margin: int = 50
    """
    Will only propagate the gather upwards through the layer if `input_count_after <= input_count_before + margin`.

    E.g., for `margin == 50`, propagates the gather upwards only if the layer size (i.e. the no.
    of neurons in the layer) increases by at most 50 neurons/ordinals as a result of the propagation.

    This setting contributes to the threshold for the memory vs. algorithmic complexity tradeoff,
    i.e. which gathers are worth removing vs. keeping.
    """

    margin_rate: float = 0.05
    """
    Will only propagate the gather upwards through the layer if `1. - (input_count_after / input_count_before) <=
    margin_rate`. Intuitively speaking, the value is the layer count (size) decrease rate (percentage).

    E.g., for `margin_rate == 0.05`, propagates the gather upwards only if the layer size (i.e. the no.
    of neurons in the layer) increases by at most 5% as a result of the propagation.

    This setting contributes to the threshold for the memory vs. algorithmic complexity tradeoff,
    i.e. which gathers are worth removing vs. keeping.
    """

    aggressive_max_chain_depth: int | Literal["unlimited"] = 0
    """
    For values greater than 0, enables aggressivity. This will propagate gathers upwards regardless of how much this
    will increase memory usage (the sizes/counts of layers and fact layers).

    The number is an upper limit on the depth
    of how many layers a single propagation chain is willing to go through. For a value of N, will essentially keep
    each N-th gather (gathers that would be removed irrespectively of aggressivity are removed anyway, and not counted
    into this value).

    When this value is set to a sufficiently high value (or to 'unlimited'), configuration options such as `margin` or
    `margin_rate` lose effect.
    """

    propagate_through_symmetries: bool = True
    """
    Value of `False` will keep any gather immediately following a symmetry intact,
    so as not to break or modify the symmetry in any way. If true, propagates through symmetries that can be propagated
    through without removing them entirely.

    Only effective when `run_before_symmetries` is set to `False` (i.e. when symmetries are already used when running
    the optimization).
    """

    def __post_init__(self):
        assert 0 <= self.margin
        assert 0.0 <= self.margin_rate <= 1.0
        assert self.aggressive_max_chain_depth in ("unlimited",) or 0 <= self.aggressive_max_chain_depth

    def updated(self, changes: _OptimizeSingleUseGathersSettingsPartial) -> "OptimizeSingleUseGathersSettings":
        vals: _OptimizeSingleUseGathersSettingsPartial = asdict(self)  # pyright: ignore
        vals.update(changes)
        return OptimizeSingleUseGathersSettings(**vals)

    @staticmethod
    def preset(
        preset: OptimizeSingleUseGathersPreset,
        **kwargs: Unpack[_OptimizeSingleUseGathersSettingsPartial],
    ) -> "OptimizeSingleUseGathersSettings":
        out = _OPTIMIZE_SINGLE_USE_GATHERS_PRESET_BUILDER_MAP[preset]()

        if len(kwargs) == 0:
            return out

        return out.updated(kwargs)

    def serialize(self) -> _OptimizeSingleUseGathersSettingsPartial:
        out = _OptimizeSingleUseGathersSettingsPartial(asdict(self))  # pyright: ignore
        return out


_OPTIMIZE_SINGLE_USE_GATHERS_PRESET_BUILDER_MAP: dict[
    OptimizeSingleUseGathersPreset, Callable[[], OptimizeSingleUseGathersSettings]
] = {
    "free_only": lambda: OptimizeSingleUseGathersSettings(margin=0, margin_rate=0.0),
    "free_only_presym": lambda: OptimizeSingleUseGathersSettings(margin=0, margin_rate=0.0, run_before_symmetries=True),
    "margin_S": lambda: OptimizeSingleUseGathersSettings(margin=10, margin_rate=0.05),
    "margin_M_presym": lambda: OptimizeSingleUseGathersSettings(
        margin=30, margin_rate=0.05, run_before_symmetries=True
    ),
    "margin_M_keepsym": lambda: OptimizeSingleUseGathersSettings(
        margin=30, margin_rate=0.05, propagate_through_symmetries=False
    ),
    "margin_M": lambda: OptimizeSingleUseGathersSettings(margin=30, margin_rate=0.05),
    "margin_L_presym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.05, run_before_symmetries=True
    ),
    "margin_L_keepsym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.05, propagate_through_symmetries=False
    ),
    "margin_L": lambda: OptimizeSingleUseGathersSettings(margin=50, margin_rate=0.05),
    "margin_XL_presym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.2, run_before_symmetries=True
    ),
    "margin_XL_keepsym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.2, propagate_through_symmetries=False
    ),
    "margin_XL": lambda: OptimizeSingleUseGathersSettings(margin=50, margin_rate=0.2),
    "margin_XXL_presym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.5, run_before_symmetries=True
    ),
    "margin_XXL_keepsym": lambda: OptimizeSingleUseGathersSettings(
        margin=50, margin_rate=0.5, propagate_through_symmetries=False
    ),
    "margin_XXL": lambda: OptimizeSingleUseGathersSettings(margin=50, margin_rate=0.5),
    "agg_max_keepsym": lambda: OptimizeSingleUseGathersSettings(
        aggressive_max_chain_depth="unlimited", propagate_through_symmetries=False
    ),
    "agg_max_propsym": lambda: OptimizeSingleUseGathersSettings(
        aggressive_max_chain_depth="unlimited", propagate_through_symmetries=True
    ),
    "agg_true_unlimited": lambda: OptimizeSingleUseGathersSettings(
        aggressive_max_chain_depth="unlimited", run_before_symmetries=True
    ),
}


class VectorizeSettingsPartial(TypedDict, total=False):
    transpose_fixed_count_reduce: bool
    iso_compression: bool
    linears_optimize_unique_ref_pairs: bool
    linears_symmetries: LinearsSymmetriesSettings | Literal[False]
    optimize_tail_refs: OptimizeTailRefsSettings | Literal[False]
    optimize_single_use_gathers: OptimizeSingleUseGathersSettings | Literal[False]
    allow_repeat_gathers: bool
    merge_trivial_layer_concats: bool
    max_nogather_simple_layer_refs_length: int
    granularize_by_weight: bool


@dataclass(frozen=True)
class VectorizeSettings:
    transpose_fixed_count_reduce: bool = True
    """
    Optimization made specifically for layers involving both a linear operation and an aggregation operation, where
    the aggregation exactly matches the no. of weights being repeatedly applied. Performs a transposition of the
    linear layer from one where the weights are repeated (e.g. `[w0, w1, w2, w0, w1, w2, ...]`) to one where
    the weights are interleaved (e.g. `[w0, ..., w0, w1, ..., w1, w2, ..., w2]`).

    When symmetries optimization is enabled, this will result in a transposed ordering of memory when performing
    the linear operation, as the aggregation will be applied on tensor dimension 0 instead of dimension 1.

    Applied prior to most other optimizations.

    The effect of this optimization may be overridden by following optimizations, such as
    `linears_optimize_unique_ref_pairs`, `linear_symmetries`, `optimize_tail_refs` or `optimize_single_use_gathers`,
    as these optimizations override original weights/inputs orderings as well.

    The `linear_symmetries` padding optimization will honor this configuration option when applicable,
    such that when this option is enabled, the padding will be performed in a way that weights will be
    interleaved and inputs repeated, and the other way around when this option is disabled.
    """

    iso_compression: bool = True
    """
    Enables ISO compression optimization.

    This optimization removes the total no. of (output) neurons in every layer by only keeping neurons
    that produce unique outputs. This is done by a stochastic algorithm that performs multiple forward passes
    with different weight initializations and compares the resulting neuron values.

    This is NOT a lossy optimization! In other words, with probability nearing 100%, this optimization will not change
    the network outputs.

    Applied prior to most other optimizations.
    """

    linears_optimize_unique_ref_pairs: bool = True
    """
    Optimization that deduplicates linear operations.

    Optimizes linear operations such that they never multiply the same input/weight value pairs more than once.

    This is not a subset nor a superset of the ISO compression optimization.
    While ISO compression may merge more neurons than just those consisting of inputs known beforehand to be the same,
    this optimization runs on linear neuron inputs, not outputs, meaning that when aggregation is involved,
    this operation can place an extra gather in-between the linear operation and the aggregation,
    which is something that ISO compression currently is not designed to do.

    This operation may produce extra gather operations in-between linear and aggregation operations. In situations where
    this is computationally unfavorable, `optimize_tail_refs` and/or `optimize_single_use_gathers` optimizations
    may again remove such gathers, while keeping other, favorable results produced by this optimization.
    """

    linears_symmetries: LinearsSymmetriesSettings | Literal[False] = field(default_factory=LinearsSymmetriesSettings)
    """
    Optimization that finds linear inputs/weights that repeat and removes memory copying when possible.

    Finds 'symmetries' in linears, i.e. linears that either require inputs (or weights, or both) to repeat or
    interleave. When possible, replaces such repeats with views (reshapes), so that the underlying input/weight tensors
    do not have to be copied/duplicated to achieve the correct result of the linear operation.

    Also involves the padding optimization, which adds layer reordering and/or padding to 'create' more such symmetries.
    This is further configurable.

    This operation may produce extra gather operations in-between linear and aggregation operations. In situations where
    this is computationally unfavorable, `optimize_tail_refs` and/or `optimize_single_use_gathers` optimizations
    may again remove such gathers, while keeping other, favorable results produced by this optimization.
    """

    optimize_tail_refs: OptimizeTailRefsSettings | Literal[False] = field(default_factory=OptimizeTailRefsSettings)
    """
    Optimization that propagates gathers downwards when possible, to simplify computations in layers.

    Modifies layers/operations such that they do not contain repeated computations of the same (pairs/tuples of)
    input values. Puts the burden of correctly ordering/gathering inputs onto later layers/operations.

    This reduces computational and memory complexity in the earlier layers, but may also increase
    computational complexity by potentially requiring more gather operations throughout the network.

    For example, an unwanted property of this optimization may be that an earlier layer performs 999 unique operations
    instead of 1000 non-unique ones, but a following layer performs aggregations that, even when unique, require
    1000 inputs (including some input value repeats). In such case, the result of this optimization would produce
    a gather in between of the two layers, even in the case where the gather wasn't necessary and the naurons were
    already in the correct order prior to the optimization.

    To prevent the above from happening, there are some controls in place as part of the optimization, as well as other
    optimizations that perform the opposite of this optimization (such as `optimize_single_use_gathers` optimization).
    """

    optimize_single_use_gathers: OptimizeSingleUseGathersSettings | Literal[False] = field(
        default_factory=OptimizeSingleUseGathersSettings
    )
    """
    Optimization that propagates gathers upwards to remove them and lower algorithmic complexity.

    When configured more aggressively, will increase memory usage by duplicating neurons in earlier layers in order for
    later gathers to not be needed.
    """

    allow_repeat_gathers: bool = False
    """
    When enabled, will replace gathers that gather (or interleave) the same repeated pattern of ordinals by a (optional)
    short gather followed by a repeat (or an interleave) operation.

    It may be computationally more beneficial to not do this and instead perform the gather as a single large operation,
    instead of a concatenation of two operations.

    A better approach for taking advantage of such repetitive patterns in gathers towards optimizing the network is to
    use the `linears_symmetries` optimization, with padding enabled (both enabled by default).

    This option may be useful for debugging purposes, e.g. to find symmetries in the unoptimized network,
    or to find symmetries that the padded `linears_symmetries` optimization failed to take advantage of, in order to
    improve the padding algorithm.
    """

    merge_trivial_layer_concats: bool = True
    """
    When enabled, any weights/facts that are trivial, i.e. of size (count) of 1, and are also always used in the same
    order together with some other weights/facts, are merged together into a single non-trivial weight/fact
    (respectively).

    This is to avoid repeated concatenation of such values in every forward pass, in the (potentially somewhat rare)
    instances where this can be done beforehand.

    Note that learnable weights cannot be merged with non-learnable facts.

    Note that repeats of the same learnable weight(s) cannot be pre-concatenated using this method because doing so
    would cause duplication of the learnable parameters of said weight(s).
    """

    max_nogather_simple_layer_refs_length: int = 24
    """
    Threshold for the approach to concatenating weights/facts/inputs when all are trivial, i.e. of size (count) 1.

    For concatenations of trivial weights/facts/inputs of total no. below this threshold, the concatenation is performed
    as-is in the desired order, including repeated uses of the inputs.

    For concatenations of total no. above the threshold, the concatenation is performed on unique references, followed
    by a gather operation to reach the desired order.

    This is to ensure that gather operations are meaningful, as well as that reference concatenations are simple enough.
    """

    granularize_by_weight: bool = False
    """
    Splits layers into multiple layers based on weights. Produces individual layers per each weight applied in linears.

    This granularizes the network such that there are more layers (some of which can be run in parallel), but each layer
    consists of simpler linear operations.

    Please note that this optimization is not recommended at the time of writing, as it produces some gather operations
    that can only be optimized by performing additional layer splitting and/or reordering optimizations which currently
    aren't implemented. Your mileage thus may vary.
    """

    def updated(self, **changes: Unpack[VectorizeSettingsPartial]) -> "VectorizeSettings":
        vals: VectorizeSettingsPartial = asdict(self)  # pyright: ignore
        vals.update(changes)
        return VectorizeSettings(**vals)

    def serialize(self) -> "VectorizeSettingsPartial":
        return VectorizeSettingsPartial(serialize_dataclass(self, call_self=False))  # pyright: ignore

    @staticmethod
    def deserialize(d: dict[str, Any]):
        linears_symmetries = d.get("linears_symmetries", None)
        optimize_tail_refs = d.get("optimize_tail_refs", None)
        optimize_single_use_gathers = d.get("optimize_single_use_gathers", None)
        if linears_symmetries:
            d["linears_symmetries"] = LinearsSymmetriesSettings(**linears_symmetries)
        if optimize_tail_refs:
            d["optimize_tail_refs"] = OptimizeTailRefsSettings(**optimize_tail_refs)
        if optimize_single_use_gathers:
            d["optimize_single_use_gathers"] = OptimizeSingleUseGathersSettings(**optimize_single_use_gathers)

        return VectorizeSettings(**d)
