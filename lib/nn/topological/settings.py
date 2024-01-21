from dataclasses import dataclass


@dataclass
class Settings:
    # TODO: ASSUMPTION: all facts have the same value
    assume_facts_same: bool = True
    # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same weights
    assume_rule_weights_same: bool = True

    # TODO: ASSUMPTION: all neurons have the same layer layout
    check_same_layers_assumption: bool = True
    # TODO: ASSUMPTION: all neurons in a given WeightedRuleLayer have the same number of inputs
    check_same_inputs_dim_assumption: bool = True
