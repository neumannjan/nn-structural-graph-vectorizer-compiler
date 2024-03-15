from dataclasses import dataclass


@dataclass
class Settings:
    merge_same_facts: bool = True

    # TODO: ASSUMPTION: all samples have the same layer layout
    check_same_layers_assumption: bool = False

    optimize_linear_gathers: bool = True

    # TODO: must be disabled (?) when batch is smaller than full (due to weights sharing)
    group_learnable_weight_parameters: bool = True

    allow_non_builtin_torch_ops: bool = True
