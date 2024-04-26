from dataclasses import dataclass
from typing import Literal

LinearsPadForSymmetriesOption = Literal["never", "sided_only", "full_only", "any"]


@dataclass
class VectorizeSettings:
    linears_optimize_unique_ref_pairs: bool = True

    linears_symmetries: bool = True

    linears_pad_for_symmetries: LinearsPadForSymmetriesOption = "any"

    iso_compression: bool = True

    optimize_tail_refs: bool = True

    optimize_tail_refs_unique_margin_rate: float = 0.05

    optimize_single_use_gathers: bool = True

    optimize_single_use_gathers_before_symmetries: bool = False

    optimize_single_use_gathers_margin: int = 50

    optimize_single_use_gathers_margin_rate: float = 0.05

    optimize_single_use_gathers_aggressive_max_chain_length: int | Literal["unlimited"] = 0

    optimize_single_use_gathers_aggressive_through_symmetries: bool = True

    allow_repeat_gathers: bool = False

    # TODO: fix bug where weights get duplicated
    merge_trivial_layer_concats: bool = False

    granularize_by_weight: bool = False

    transpose_fixed_count_reduce: bool = False

    max_nogather_simple_layer_refs_length: int = 24

    def __post_init__(self):
        assert 0.0 <= self.optimize_single_use_gathers_margin_rate <= 100.0
