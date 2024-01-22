from typing import Protocol, Sequence

import numpy as np


class JavaValue(Protocol):
    def getAsArray(self) -> np.ndarray:
        ...

    def size(self) -> Sequence[int]:
        ...


class JavaWeight(Protocol):
    @property
    def value(self) -> JavaValue:
        ...

    @property
    def index(self) -> int:
        ...


class JavaRawState(Protocol):
    def getValue(self) -> JavaValue:
        ...


class JavaClass(Protocol):
    def getSimpleName(self) -> str:
        ...


class JavaNeuron(Protocol):
    def getIndex(self) -> int:
        ...

    def getInputs(self) -> list["JavaNeuron"]:
        ...

    def getRawState(self) -> JavaRawState:
        ...

    def getClass(self) -> JavaClass:
        ...

    def getLayer(self) -> int:
        ...

    def getWeights(self) -> list[JavaWeight]:
        ...


class JavaNeuronQuery(Protocol):
    @property
    def neuron(self) -> JavaNeuron:
        ...


class JavaSample(Protocol):
    @property
    def query(self) -> JavaNeuronQuery:
        ...
