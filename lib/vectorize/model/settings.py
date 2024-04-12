from dataclasses import dataclass
from typing import Literal


@dataclass
class VectorizeSettings:
    # TODO: reintroduce?
    # group_learnable_weight_parameters: bool = True

    linears_optimize_repeating_seq: bool = True

    linears_optimize_unique_ref_pairs: bool = True

    linears_optimize_unique_ref_pairs_aggressively: bool = False

    optimize_tail_refs: bool = True

    optimize_single_use_gathers: bool = True

    optimize_single_use_gathers_aggressive_max_chain_length: int | Literal["unlimited"] = 0
