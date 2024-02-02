import itertools

import pytest
import torch
from lib.nn.topological.aggregation_layer import AggregationLayer
from lib.nn.topological.layers import LayerDefinition, compute_neuron_ordinals, get_neurons_per_layer
from lib.nn.topological.settings import Settings
from lib.tests.utils.network_mock import MockJavaNeuron
from lib.tests.utils.test_params import SETTINGS_PARAMS

INDEX_FACTORY = iter(itertools.count())


def _n():
    return next(INDEX_FACTORY)


LAYERS = [
    LayerDefinition("", 16),
    LayerDefinition("", 13),
    LayerDefinition("", 12),
]

SAMPLE1 = MockJavaNeuron(
    _n(), 16, [MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(4)]) for _ in range(3)]
)

SAMPLE2 = MockJavaNeuron(
    _n(),
    16,
    [
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(4)]),
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(3)]),
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(3)]),
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(4)]),
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(2)]),
        MockJavaNeuron(_n(), 13, [MockJavaNeuron(_n(), 12, []) for _ in range(4)]),
    ],
)


@pytest.mark.parametrize(
    ["settings"],
    [[settings] for settings in SETTINGS_PARAMS],
)
def test_same_no_of_inputs(settings: Settings):
    inputs = {12: torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5])}
    network = get_neurons_per_layer([SAMPLE1])
    _, ordinals = compute_neuron_ordinals(LAYERS, network, settings)
    layer = AggregationLayer(
        layer_neurons=SAMPLE1.getInputs(),
        neuron_ordinals=ordinals,
        aggregation="sum",
    )
    expected = torch.tensor([8, 12, 20])

    actual = layer(inputs)

    assert (expected == actual).all()


@pytest.mark.parametrize(
    ["settings"],
    [[settings] for settings in SETTINGS_PARAMS],
)
def test_variable_no_of_inputs(settings: Settings):
    inputs = {12: torch.tensor([2, 2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7, 11, 11, 13, 13, 13, 13])}
    network = get_neurons_per_layer([SAMPLE2])
    _, ordinals = compute_neuron_ordinals(LAYERS, network, settings)
    layer = AggregationLayer(
        layer_neurons=SAMPLE2.getInputs(),
        neuron_ordinals=ordinals,
        aggregation="sum",
    )
    expected = torch.tensor([8, 9, 15, 28, 22, 52])

    actual = layer(inputs)

    assert (expected == actual).all()
