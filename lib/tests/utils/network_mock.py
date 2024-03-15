import random
from typing import Sequence

from lib.nn.sources.dict_source import NeuralNetworkDefinitionDict, Neuron
from lib.nn.sources.source import LayerDefinition, LayerType, NeuralNetworkDefinition
from lib.tests.utils.neuron_mock import NeuronTestFactory

EXAMPLE_LAYER_IDS: list[int] = [16, 13, 12, 11, 10, 9, 8, 7]

_A: list[LayerType] = ["AggregationLayer"]
EXAMPLE_LAYER_TYPES: Sequence[LayerType] = ["FactLayer", *(_A * 7)]


def generate_example_network(inputs_from_previous_layer_only=False) -> NeuralNetworkDefinition:
    neurons: dict[int, list[Neuron]] = {}

    layers = [LayerDefinition(id=l, type=t) for l, t in zip(EXAMPLE_LAYER_IDS, EXAMPLE_LAYER_TYPES)]

    factory = NeuronTestFactory(layers=layers)

    layer = EXAMPLE_LAYER_IDS[0]

    neurons[layer] = [factory.create(layer) for _ in range(random.randint(1, 30))]

    if inputs_from_previous_layer_only:
        prev_neurons = neurons[layer]
    else:
        prev_neurons = [*neurons[layer]]

    for layer in EXAMPLE_LAYER_IDS[1:]:
        neurons[layer] = [
            factory.create(layer, inputs=random.choices([n.id for n in prev_neurons], k=random.randint(1, 300)))
            for _ in range(random.randint(1, 300))
        ]

        if inputs_from_previous_layer_only:
            prev_neurons = neurons[layer]
        else:
            prev_neurons += neurons[layer]

    return NeuralNetworkDefinitionDict(layers=layers, neurons=neurons)
