from dataclasses import dataclass


@dataclass
class Settings:
    # TODO: ASSUMPTION: all facts have the same value
    assume_facts_same: bool = True
    # TODO: ASSUMPTION: all neurons have the same layer layout
    check_same_layers_assumption: bool = True
    # TODO: ASSUMPTION: all neurons in a given WeightedLayer have the same number of inputs
    check_same_weights_assumption: bool = True
