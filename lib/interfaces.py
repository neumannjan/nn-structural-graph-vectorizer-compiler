from typing import Any, Protocol


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
