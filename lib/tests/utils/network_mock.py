import itertools
import random
from typing import Any, Sequence

import numpy as np
from lib.interfaces import JavaNeuron, JavaValue, JavaWeight
from lib.nn.topological.layers import LayerDefinition, TopologicalNetwork


class MockJavaValue(JavaValue):
    def __init__(self, arr: np.ndarray) -> None:
        super().__init__()
        self._arr = arr

    def getAsArray(self) -> np.ndarray:
        return self._arr.copy()

    def size(self) -> Sequence[int]:
        return self._arr.shape


class MockJavaWeight(JavaWeight):
    def __init__(self, index: int, value: np.ndarray, learnable: bool) -> None:
        super().__init__()
        self._index = index
        self._value = MockJavaValue(value)
        self._learnable = learnable

    @property
    def value(self) -> JavaValue:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    def isLearnable(self) -> bool:
        return self._learnable


class MockJavaNeuron(JavaNeuron):
    def __init__(
        self,
        index: int,
        layer: int,
        inputs: Sequence[JavaNeuron] | None = None,
        weights: Sequence[JavaWeight] | None = None,
    ) -> None:
        self._index = index

        if inputs is None:
            inputs = []

        if weights is None:
            weights = []

        self._inputs = inputs
        self._layer = layer
        self._weights = weights

    def __repr__(self) -> str:
        return f"({self._layer}:{self._index})"

    def getIndex(self) -> int:
        return self._index

    def getInputs(self) -> list[JavaNeuron]:
        return self._inputs

    def getLayer(self) -> int:
        return self._layer

    def getRawState(self) -> Any:
        raise NotImplementedError()

    def getClass(self) -> Any:
        raise NotImplementedError()

    def getWeights(self) -> list[JavaWeight]:
        return self._weights


EXAMPLE_LAYERS = [16, 13, 12, 11, 10, 9, 8, 7]
EXAMPLE_LAYER_TYPES = ["FactNeuron", *(["AggregateNeuron"] * 7)]


def generate_example_network(inputs_from_previous_layer_only=False) -> tuple[list[LayerDefinition], TopologicalNetwork]:
    network: TopologicalNetwork = {}

    index_factory = iter(itertools.count())

    layer = EXAMPLE_LAYERS[0]

    network[layer] = [MockJavaNeuron(index=next(index_factory), layer=layer) for _ in range(random.randint(1, 30))]

    if inputs_from_previous_layer_only:
        prev_neurons = network[layer]
    else:
        prev_neurons = [*network[layer]]

    for layer in EXAMPLE_LAYERS[1:]:
        network[layer] = [
            MockJavaNeuron(
                index=next(index_factory), layer=layer, inputs=random.choices(prev_neurons, k=random.randint(1, 300))
            )
            for _ in range(random.randint(1, 300))
        ]

        if inputs_from_previous_layer_only:
            prev_neurons = network[layer]
        else:
            prev_neurons += network[layer]

    layers = [LayerDefinition(type=t, index=l) for l, t in zip(EXAMPLE_LAYERS, EXAMPLE_LAYER_TYPES)]

    return layers, network
