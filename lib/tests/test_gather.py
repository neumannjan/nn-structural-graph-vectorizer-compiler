import itertools

import pytest
import torch
from lib.nn.gather import TakeLayerSlice, build_optimal_gather_module
from lib.nn.topological.layers import LayerOrdinal, TopologicalNetwork, compute_neuron_ordinals
from lib.nn.topological.settings import Settings
from lib.tests.utils.network_mock import generate_example_network
from lib.utils import atleast_3d_rev


def _do_the_test(
    gather_module: torch.nn.Module,
    input_layer_ordinal_pairs: list[LayerOrdinal],
    network: TopologicalNetwork,
    all_same: bool,
):
    # input: indices of the neurons (so that for each neuron, its index is in its position)
    layer_values = {l: atleast_3d_rev(torch.tensor([n.getIndex() for n in neurons])) for l, neurons in network.items()}

    # expected output: list of the neuron indices that the module is supposed to gather
    expected = torch.tensor([network[l][o].getIndex() for l, o in input_layer_ordinal_pairs])

    # actual output:
    actual: torch.Tensor = torch.squeeze(gather_module(layer_values))

    if all_same:
        # if all inputs are the same,
        # the gather module takes advantage of that, but doesn't expand it by default
        expected = expected[0]

    print("Expected:", expected)
    print("Actual:", actual, flush=True)

    # assert
    assert (actual == expected).all()


GATHER_TEST_PARAMS = list(
    itertools.product(
        [False, True],
        [False, True],
        list(range(5)),
    )
)


@pytest.mark.parametrize(
    ["inputs_from_previous_layer_only", "assume_facts_same", "execution_number"],
    GATHER_TEST_PARAMS,
)
def test_gather_module(
    inputs_from_previous_layer_only: bool,
    assume_facts_same: bool,
    execution_number: int,
):
    settings = Settings(assume_facts_same=assume_facts_same)

    layers, network = generate_example_network(inputs_from_previous_layer_only=inputs_from_previous_layer_only)
    _, ordinals = compute_neuron_ordinals(layers, network, settings=settings)
    for l in layers[1:]:
        input_layer_ordinal_pairs = [ordinals[inp.getIndex()] for n in network[l.index] for inp in n.getInputs()]

        gather = build_optimal_gather_module(input_layer_ordinal_pairs)

        print("Layer", l.index)
        print(gather)
        _do_the_test(gather, input_layer_ordinal_pairs, network, all_same=(assume_facts_same and l == 1))


def test_slice_gather():
    slice_start = 10
    slice_end = 100
    ordinals = list(range(slice_start, slice_end))
    total = 400
    input_layer_ordinal_pairs = [LayerOrdinal(0, i) for i in ordinals]
    inputs = {0: torch.tensor(list(range(total)))}

    gather_module = build_optimal_gather_module(input_layer_ordinal_pairs)
    actual = gather_module(inputs)

    assert isinstance(gather_module, TakeLayerSlice)
    expected = torch.tensor(ordinals)
    assert (actual == expected).all()

