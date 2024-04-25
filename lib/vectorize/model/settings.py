from dataclasses import dataclass
from typing import Literal

LinearsPadForSymmetriesOption = Literal[
    "never", "by_count", "always", "always_full", "always_inputs_only", "always_weights_only"
]


@dataclass
class VectorizeSettings:
    linears_optimize_unique_ref_pairs: bool = True

    linears_symmetries: bool = True

    linears_pad_for_symmetries: LinearsPadForSymmetriesOption = "by_count"

    iso_compression: bool = True

    optimize_tail_refs: bool = True

    optimize_single_use_gathers: bool = True

    optimize_single_use_gathers_before_symmetries: bool = False

    optimize_single_use_gathers_margin: int = 50

    optimize_single_use_gathers_margin_rate: float = 0.05

    optimize_single_use_gathers_aggressive_max_chain_length: int | Literal["unlimited"] = 0

    optimize_single_use_gathers_aggressive_through_symmetries: bool = True

    merge_trivial_layer_concats: bool = True

    granularize_by_weight: bool = False

    transpose_fixed_count_reduce: bool = False

    def __post_init__(self):
        assert 0. <= self.optimize_single_use_gathers_margin_rate <= 100.
