from collections.abc import Collection, Sequence
from typing import Generic, Iterable, Mapping, Protocol, TypeVar

import numpy as np
import torch

from lib.nn.definitions.ops import TransformationDef
from lib.nn.sources.base import LayerDefinition, WeightDefinition

_TNeurons = TypeVar("_TNeurons")


class MinimalAPINetwork(Generic[_TNeurons], Protocol):
    """
    Minimal (i.e. easy to implement) API for a neural network definition.

    `neurons`, passed as parameter to most methods, can be of an arbitrary type, whichever is most convenient for
    whoever is implementing the API.

    For the end user of this API, the actual value of `neurons` doesn't hold any meaning other than
    "it is a black box container for some neurons".

    The end user of this API is meant to interact with `neurons` only
    through the methods in this class, e.g. `get_ids(neurons)` or `get_inputs(neurons)`.

    For convenience, the end user is advised to use the full API, for which there exists a universal implementation
    for an arbitrary `MinimalAPINetwork`.

    The idea is as follows:
    1) The author of a neural network definition implements the minimal API, i.e. subclasses of `MinimalAPINetwork` and
    `MinimalAPIOrdinals`.
    2) The end-user uses NetworkImpl(TheMinimalAPINetwork(), TheMinimalAPIOrdinals()), which provides a more convenient
    and more feature-complete API.

    (Note: Universal implementation `MinimalAPIOrdinalsImpl` should suffice, so all that needs to be implemented is
    `MinimalAPINetwork`.)
    """

    def get_layers(self) -> Sequence[LayerDefinition]:
        """Return all layer definitions in order from input to output."""
        ...

    def get_layers_map(self) -> Mapping[int, LayerDefinition]:
        """Return a mapping from layer IDs to layer definitions."""
        ...

    def get_layer_neurons(self, layer_id: int) -> _TNeurons:
        """Return a representation of all neurons from a particular layer."""
        ...

    def get_ids(self, neurons: _TNeurons) -> Sequence[int]:
        """Return neuron IDs for a given representation of neurons."""
        ...

    def get_inputs(self, neurons: _TNeurons) -> _TNeurons:
        """Return a representation of all inputs of a given representation of neurons."""
        ...

    def get_input_lengths(self, neurons: _TNeurons) -> Sequence[int]:
        """Return input lengths for a given representation of neurons."""
        ...

    def get_input_weights(self, neurons: _TNeurons) -> Iterable[WeightDefinition]:
        """Return input weights for a given representation of neurons."""
        ...

    def get_biases(self, neurons: _TNeurons) -> Sequence[WeightDefinition]:
        """Return bias (offset) weights for a given representation of neurons."""
        ...

    def get_values_numpy(self, neurons: _TNeurons) -> Collection[np.ndarray]:
        """Return values for a given representation of neurons as numpy arrays."""
        ...

    def get_values_torch(self, neurons: _TNeurons) -> Collection[torch.Tensor]:
        """Return values for a given representation of neurons as torch tensors."""
        ...

    def get_transformations(self, neurons: _TNeurons) -> Sequence[TransformationDef | None]:
        """Return transformations from a given representation of neurons."""
        ...

    def slice(self, neurons: _TNeurons, sl: slice) -> _TNeurons:
        """Return a slice of a given representation of neurons."""
        ...

    def select_ids(self, neurons: _TNeurons, ids: Sequence[int]) -> _TNeurons:
        """Return a representation of an arbitrary subset of given neurons based on requested IDs."""
        ...
