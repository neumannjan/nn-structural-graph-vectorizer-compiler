from dataclasses import dataclass


@dataclass
class Settings:
    # TODO: ASSUMPTION: all facts have the same value
    assume_facts_same: bool = True
    # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same weights
    assume_rule_weights_same: bool = True

    # TODO: ASSUMPTION: all samples have the same layer layout
    check_same_layers_assumption: bool = True

    optimize_linear_gathers: bool = True

    # TODO: must be disabled (?) when batch is smaller than full (due to weights sharing)
    group_learnable_weight_parameters: bool = True
