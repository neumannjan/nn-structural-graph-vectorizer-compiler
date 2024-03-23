from collections import OrderedDict
from typing import Collection, Iterable, Sequence, TypeGuard, TypeVar, get_args

import numpy as np

from lib.nn.definitions.ops import AggregationDef, ReductionDef, TransformationDef
from lib.sources.base import LayerNeurons, Network, Neurons, Ordinals
from lib.utils import atleast_2d_shape, head_and_rest
from lib.vectorize.model import *

_REDUCTIONS: set[ReductionDef] = set(get_args(ReductionDef))


def _is_reduction_def(agg: AggregationDef | None) -> TypeGuard[ReductionDef | None]:
    return agg is None or agg in _REDUCTIONS


def _is_reduction_defs(aggs: Sequence[AggregationDef | None]) -> TypeGuard[ReductionDef | None]:
    return all((_is_reduction_def(a) for a in aggs))


_T = TypeVar("_T")


def _assert_all_same_ignore_none(what_plural: str, source: Iterable[_T]) -> _T | None:
    first = None

    for v in source:
        if v is None:
            continue

        if first is None:
            first = v
            continue

        assert first == v, f"Assertion failed: found {what_plural} {first} and {v}"

    return first


def _build_gather(fact_layers: Collection[str], layer_sizes: dict[str, int], input_ordinals: Ordinals):
    refs: list[Ref] = []

    for ord in input_ordinals:
        layer = str(ord.layer)
        if layer in fact_layers:
            refs.append(FactRef(layer, ord.ordinal))
        else:
            refs.append(NeuronRef(layer, ord.ordinal))

    return Refs(refs)


def _build_weights(
    neurons: Neurons,
    weights_out: dict[str, LearnableWeight],
    fact_weights_out: list[Fact],
    facts_map: dict[str, int],
    fact_weights_layer: str,
):
    weights = list(neurons.input_weights)
    ids = [str(w.id) for w in weights]

    if len(ids) == 0:
        return None

    for w in weights:
        w_id = str(w.id)

        if w_id not in weights_out and w_id not in facts_map:
            val = w.get_value_numpy()
            val = np.reshape(val, [1, *atleast_2d_shape(val.shape)])
            if not w.learnable:
                if np.all(val == 1.0):
                    fact = UnitFact()
                else:
                    fact = ValueFact(val)
                fact_weights_out.append(fact)
                facts_map[w_id] = len(fact_weights_out) - 1
            else:
                weights_out[w_id] = LearnableWeight(val)

    weight_sources = [FactRef(fact_weights_layer, facts_map[id]) if id in facts_map else WeightRef(id) for id in ids]
    return Refs(weight_sources)


def _build_reduce(neurons: Neurons):
    aggregation_def = _assert_all_same_ignore_none("aggregations", neurons.get_aggregations()) or "sum"
    assert _is_reduction_def(aggregation_def)

    input_counts = neurons.input_lengths

    if aggregation_def is not None:
        first, rest = head_and_rest(input_counts)

        if all((r == first for r in rest)):
            if first == 1:
                return Noop()
            else:
                return FixedCountReduce(period=first, reduce=aggregation_def)
        else:
            return UnevenReduce(counts=list(input_counts), reduce=aggregation_def)
    else:
        return Noop()


def _build_transform(neurons: Neurons):
    transforms: list[TransformationDef] = ["identity" if tr is None else tr for tr in neurons.get_transformations()]

    first, rest = head_and_rest(transforms)

    if all((r == first for r in rest)):
        return Transform(first)

    raise Exception("Multiple transforms found in a single layer")


def _build_layer_base(gather: Input, weights: Input | None):
    if weights is None:
        return InputLayerBase(gather)
    else:
        return LinearLayerBase(gather, weights)


def _build_layer(base: LayerBase, aggregate: Reduce, transform: Transform):
    return Layer(base=base, aggregate=aggregate, transform=transform)


def _build_fact_layer(neurons: LayerNeurons):
    out: list[Fact] = []
    for value in neurons.get_values_numpy():
        if np.all(value == 1.0):
            out.append(UnitFact())
        else:
            out.append(ValueFact(np.reshape(value, [1, *atleast_2d_shape(value.shape)])))

    return FactLayer(out)


def build_initial_network(network: Network) -> VectorizedNetwork:
    layer_sizes: dict[str, int] = {}

    fact_layers: dict[str, FactLayer] = {}
    weights: dict[str, LearnableWeight] = {}
    layers: OrderedDict[str, Layer] = OrderedDict()

    fact_weights: list[Fact] = []
    fact_weight_map: dict[str, int] = {}

    FACT_WEIGHTS_LAYER_KEY = "w"

    for layer, neurons in network.items():
        try:
            layer_id = str(layer.id)

            if layer.type == "FactLayer":
                fact_layers[layer_id] = _build_fact_layer(neurons)
            else:
                # TODO: support multiple different transforms/aggregations/shapes here ?
                transform = _build_transform(neurons)
                reduce = _build_reduce(neurons)

                gather_source = _build_gather(fact_layers, layer_sizes, neurons.inputs.ordinals)
                weight_source = _build_weights(
                    neurons,
                    weights_out=weights,
                    fact_weights_out=fact_weights,
                    facts_map=fact_weight_map,
                    fact_weights_layer=FACT_WEIGHTS_LAYER_KEY,
                )

                layer_base = _build_layer_base(gather_source, weight_source)
                the_layer = _build_layer(layer_base, reduce, transform)

                layers[layer_id] = the_layer

            layer_sizes[layer_id] = len(neurons)
        except Exception as e:
            raise Exception(f"Exception in layer {layer.id}") from e

    fact_weights_layer = FactLayer(fact_weights)
    fact_layers[FACT_WEIGHTS_LAYER_KEY] = fact_weights_layer

    vectorized_net = VectorizedNetwork(fact_layers=fact_layers, weights=weights, batches={0: Batch(layers)})
    return vectorized_net
