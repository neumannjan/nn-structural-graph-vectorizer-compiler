from typing import Sequence

import numpy as np
from neuralogic.core.builder.builder import NeuralSample

from lib.interfaces import JavaClass, JavaNeuron, JavaNeuronQuery, JavaRawState, JavaSample, JavaValue, JavaWeight


class MockJavaNeuronQuery(JavaNeuronQuery):
    def __init__(self, neuron: JavaNeuron) -> None:
        self._neuron = neuron

    @property
    def neuron(self) -> JavaNeuron:
        return self._neuron


class MockJavaSample(JavaSample):
    def __init__(self, neuron: JavaNeuron) -> None:
        self._query = MockJavaNeuronQuery(neuron)

    @property
    def query(self) -> JavaNeuronQuery:
        return self._query


class MockNeuralSample(NeuralSample):
    def __init__(self, sample: JavaSample):
        super().__init__(sample=sample, grounding=None)


class MockJavaClass(JavaClass):
    def __init__(self, simple_name: str) -> None:
        self._simple_name = simple_name

    def getSimpleName(self) -> str:
        return self._simple_name


class MockJavaValue(JavaValue):
    def __init__(self, arr: np.ndarray) -> None:
        super().__init__()
        self._arr = arr

    def getAsArray(self) -> np.ndarray:
        return self._arr.copy()

    def size(self) -> Sequence[int]:
        return self._arr.shape


class MockJavaWeight(JavaWeight):
    def __init__(self, index: int, value: np.ndarray) -> None:
        super().__init__()
        self._index = index
        self._value = MockJavaValue(value)

    @property
    def value(self) -> JavaValue:
        return self._value

    @property
    def index(self) -> int:
        return self._index


class FullMockJavaNeuron(JavaNeuron):
    def __init__(
        self,
        index: int,
        layer: int,
        cls: JavaClass,
        inputs: list[JavaNeuron] | None = None,
        weights: list[JavaWeight] | None = None,
    ) -> None:
        self._index = index

        if inputs is None:
            inputs = []

        if weights is None:
            weights = []

        self._inputs = inputs
        self._weights = weights
        self._layer = layer
        self._cls = cls

    def __repr__(self) -> str:
        return f"({self._layer}:{self._index})"

    def getIndex(self) -> int:
        return self._index

    def getInputs(self) -> list[JavaNeuron]:
        return self._inputs

    def getLayer(self) -> int:
        return self._layer

    def getRawState(self) -> JavaRawState:
        raise NotImplementedError()

    def getClass(self) -> MockJavaClass:
        return self._cls

    def getWeights(self) -> list[JavaWeight]:
        return self._weights
