import itertools

import pytest
import torch
from lib.nn.gather import GatherModule
from lib.nn.topological.layers import TopologicalNetwork, compute_neuron_ordinals
from lib.tests.utils.network_mock import generate_example_network
from lib.utils import atleast_3d_rev


def _do_the_test(gather_module: GatherModule, network: TopologicalNetwork):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {l: atleast_3d_rev(torch.tensor([n.getIndex() for n in neurons])) for l, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    expected = torch.tensor([network[l][o].getIndex() for l, o in gather_module.input_layer_ordinal_pairs])

    # actual output:
    actual = torch.squeeze(gather_module(layer_values))

    print("Expected:", expected)
    print("Actual:", actual)

    # assert
    assert (actual == expected).all()


GATHER_TEST_PARAMS = list(
    itertools.product(
        [False, True],
        [False, True],
        [False, True],
        list(range(5)),
    )
)


@pytest.mark.parametrize(
    ["inputs_from_previous_layer_only", "assume_facts_same", "allow_merge_on_all_inputs_same", "execution_number"],
    GATHER_TEST_PARAMS,
)
def test_gather_module(
    inputs_from_previous_layer_only: bool,
    assume_facts_same: bool,
    allow_merge_on_all_inputs_same: bool,
    execution_number: int,
):
    layers, network = generate_example_network(inputs_from_previous_layer_only=inputs_from_previous_layer_only)
    ordinals_per_layer, ordinals = compute_neuron_ordinals(layers, network, assume_facts_same=assume_facts_same)
    for l in layers[1:]:
        # gather inputs from previous layer
        gather = GatherModule(
            [ordinals[inp.getIndex()] for n in network[l.index] for inp in n.getInputs()],
            allow_merge_on_all_inputs_same=allow_merge_on_all_inputs_same,
        )

        print("Layer", l.index)
        _do_the_test(gather, network)
