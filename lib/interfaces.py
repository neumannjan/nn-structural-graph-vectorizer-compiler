from typing import Any, Protocol, Sequence

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

    def isLearnable(self) -> bool:
        ...


class JavaNeuron(Protocol):
    def getIndex(self) -> int:
        ...

    def getInputs(self) -> list["JavaNeuron"]:
        ...

    def getRawState(self) -> Any:
        ...

    def getClass(self) -> Any:
        ...

    def getLayer(self) -> int:
        ...

    def getWeights(self) -> Any:
        ...
