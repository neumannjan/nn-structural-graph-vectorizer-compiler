from collections.abc import Collection
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    OrderedDict,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import torch

from compute_graph_vectorize.model.ops import AggregationDef, TransformationDef

if TYPE_CHECKING:
    from compute_graph_vectorize.sources.minimal_api.base import MinimalAPINetwork
    from compute_graph_vectorize.sources.minimal_api.ordinals import MinimalAPIOrdinals

LayerType = Literal["FactLayer", "WeightedAtomLayer", "WeightedRuleLayer", "AtomLayer", "RuleLayer", "AggregationLayer"]


def is_weighted(layer_type: LayerType):
    return layer_type in ("FactLayer", "WeightedAtomLayer", "WeightedRuleLayer")


@dataclass(frozen=True)
class LayerDefinition:
    """Definition of a neural network layer. Contains the layer ID and the layer type."""

    id: str
    type: LayerType


def get_layer_id(layer: str | LayerDefinition):
    if isinstance(layer, LayerDefinition):
        layer_id = layer.id
    else:
        layer_id = layer
    return layer_id


class LayerOrdinal(NamedTuple):
    """Representation of a position of a neuron within a network."""

    layer: str
    ordinal: int

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer={self.layer}, ordinal={self.ordinal})"


class Ordinals(Protocol):
    """
    A container for `LayerOrdinal`s of an arbitrary set of neurons.

    If you want to write a custom implementation of this, please take a look at the minimal API first.
    (Classes `MinimalAPINetwork` and `MinimalAPIOrdinals`, convertible to `Network` via `from_minimal_api`.)
    """

    def ids(self) -> Collection[int]:
        """Get IDs of the corresponding neurons in this view."""
        ...

    def items(self) -> Collection[tuple[int, LayerOrdinal]]:
        """Get (id, LayerOrdinal) pairs of all corresponding neurons in this view."""
        ...

    def values(self) -> Collection[LayerOrdinal]:
        """Get all `LayerOrdinal`s of all corresponding neurons in this view."""
        ...

    def __iter__(self) -> Iterator[LayerOrdinal]:
        """Iterate all `LayerOrdinal`s of all corresponding neurons in this view."""
        ...

    def __len__(self) -> int:
        """Get the total number of neurons in this view."""
        ...

    def __contains__(self, o: object) -> bool: ...

    def __getitem__(self, id: int) -> LayerOrdinal:
        """Get the `LayerOrdinal` of a given neuron ID, if present in this view."""
        ...


class LayerDefinitions(Protocol):
    """A container for all `LayerDefinition`s of a network."""

    def __len__(self) -> int:
        """Get total no. of layers."""
        ...

    def __getitem__(self, layer_id: str) -> LayerDefinition:
        """Get the layer definition for a given layer ID."""
        ...

    def __iter__(self) -> Iterator[LayerDefinition]:
        """Iterate all layer definitions in order from input to output."""
        ...

    def __contains__(self, ld: object) -> bool: ...

    def as_list(self) -> list[LayerDefinition]: ...

    def as_dict(self) -> OrderedDict[str, LayerDefinition]: ...


@runtime_checkable
class WeightDefinition(Protocol):
    """A definition of a (learnable?) weight. Provides its network-unique ID and the `learnable` flag."""

    @property
    def learnable(self) -> bool:
        """Whether the weight is learnable."""
        ...

    @property
    def id(self) -> int:
        """Unique identifier of the weight (unique in a given Network)."""
        ...

    def get_value_numpy(self) -> np.ndarray:
        """Get the initial value of the weight, as a numpy array."""
        ...

    def get_value_torch(self) -> torch.Tensor:
        """Get the initial value of the weight, as a PyTorch tensor."""
        ...

    def __hash__(self) -> int: ...

    def __eq__(self, value: object) -> bool: ...


class Neurons(Protocol):
    """Represents a sequence of neurons. Provides an easy-to-use API for traversal of the network.

    If you want to write a custom implementation of this, please take a look at the minimal API first.
    (Classes `MinimalAPINetwork` and `MinimalAPIOrdinals`, convertible to `Network` via `from_minimal_api`.)
    """

    @property
    def ordinals(self) -> Ordinals:
        """The neuron ordinals (their positions in the network)."""
        ...

    @property
    def ids(self) -> Sequence[int]:
        """The unique neuron IDs."""
        ...

    @property
    def input_lengths(self) -> Sequence[int]:
        """Input counts per neuron."""
        ...

    @property
    def inputs(self) -> "Neurons":
        """Traverse the input neurons of the neurons in this container."""
        ...

    def __len__(self) -> int:
        """Get the total neurons in this container."""
        ...

    @property
    def input_weights(self) -> Iterable[WeightDefinition]:
        """Iterable over weights attached to all neuron inputs of the neurons in this container."""
        ...

    @property
    def biases(self) -> Sequence[WeightDefinition]:
        """Get the neuron bias (offset) weights of the neurons in this container."""
        ...

    def get_values_numpy(self) -> Collection[np.ndarray]:
        """Get the values of the neurons in this container (if applicable) as numpy arrays."""
        ...

    def get_values_torch(self) -> Collection[torch.Tensor]:
        """Get the values of the neurons in this container (if applicable) as PyTorch tensors."""
        ...

    def get_transformations(self) -> Sequence[TransformationDef | None]:
        """Get the transformation function definitions attached to the neurons in this container."""
        ...

    def get_aggregations(self) -> Sequence[AggregationDef | None]:
        """Get the aggregation function definitions attached to the neurons in this container."""
        ...

    def slice(self, sl: slice) -> "Neurons":
        """Get a slice of this neurons container."""
        ...

    def select_ids(self, ids: Sequence[int]) -> "Neurons":
        """View an (ordered) subset of this neurons container based on a sequence of neuron IDs."""
        ...

    def select_ord(self, ords: Sequence[LayerOrdinal]) -> "Neurons":
        """View an (ordered) subset of this neurons container based on a sequence of neuron ordinals."""
        ...


class LayerNeurons(Neurons, Protocol):
    """Represents a layer of neurons. Provides an easy-to-use API for traversal of the network.

    If you want to write a custom implementation of this, please take a look at the minimal API first.
    (Classes `MinimalAPINetwork` and `MinimalAPIOrdinals`, convertible to `Network` via `from_minimal_api`.)
    """

    @property
    def layer(self) -> LayerDefinition:
        """Get the layer definition of this collection of neurons."""
        ...


class Network(Protocol):
    """Represents a neural network, down to a per-neuron granularity.

    Provides an easy-to-use API for traversal of the network.

    If you want to write a custom implementation of this, please take a look at the minimal API first.
    (Classes `MinimalAPINetwork` and `MinimalAPIOrdinals`, convertible to `Network` via `from_minimal_api`.)
    """

    @property
    def ordinals(self) -> Ordinals:
        """Get the ordinals of all neurons in the network. Useful as an `id` -> `LayerOrdinal` mapping."""
        ...

    def __getitem__(self, layer: str | LayerDefinition) -> LayerNeurons:
        """View the neurons contained in a specific layer."""
        ...

    def __len__(self) -> int:
        """Get the total no. of neurons in the network."""
        ...

    @property
    def layers(self) -> LayerDefinitions:
        """Get all layer definitions, ordered from input to output, indexable by its layer IDs."""
        ...

    def items(self) -> Collection[tuple[LayerDefinition, LayerNeurons]]:
        """Get `LayerDefinition`, `LayerNeurons` pairs."""
        ...

    def __iter__(self) -> Iterator[LayerNeurons]:
        """Iterate over the neurons containers in individual layers in the order from input to output."""
        ...

    def __contains__(self, o: object) -> bool: ...

    @property
    def minimal_api(self) -> "MinimalAPINetwork | None":
        """Access the underlying 'minimal API' implementation of the network definition, if it is used."""
        ...

    @property
    def minimal_api_ordinals(self) -> "MinimalAPIOrdinals | None":
        """Access the underlying 'minimal API' implementation of the ordinals definition, if it is used."""
        ...
