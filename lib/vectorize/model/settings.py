from dataclasses import dataclass


@dataclass
class VectorizeSettings:
    # TODO: reintroduce?
    # group_learnable_weight_parameters: bool = True

    linears_optimize_repeating_seq: bool = True

    linears_optimize_unique_ref_pairs: bool = True

    linears_optimize_unique_ref_pairs_aggressively: bool = False

    optimize_tail_refs: bool = True
