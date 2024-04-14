from collections import OrderedDict
from typing import Collection, Iterable, TypeGuard, TypeVar, get_args

import numpy as np

from lib.model.ops import AggregationDef, ReductionDef, TransformationDef
from lib.sources.base import LayerNeurons, Network, Neurons, Ordinals
from lib.utils import atleast_2d_shape, head_and_rest
from lib.vectorize.model import *

_REDUCTIONS: set[ReductionDef] = set(get_args(ReductionDef))


UNIT_FACT = UnitFact()


def _is_reduction_def(agg: AggregationDef | None) -> TypeGuard[ReductionDef | None]:
    return agg is None or agg in _REDUCTIONS


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


def _get_str_id_from_int(layer: int) -> str:
    return "%03d" % layer


def _build_gather(fact_layers: Collection[str], input_ordinals: Ordinals):
    ordinals: list[int] = []
    types: list[int] = []
    layer_ids: list[str] = []

    for ord in input_ordinals:
        ordinals.append(ord.ordinal)
        layer_ids.append(ord.layer)
        if ord.layer in fact_layers:
            types.append(Refs.TYPE_FACT)
        else:
            types.append(Refs.TYPE_LAYER)

    return Refs(
        types=types,
        layer_ids=layer_ids,
        ordinals=ordinals,
    )


def _build_weights(
    neurons: Neurons,
    weights_out: dict[str, LearnableWeight],
    fact_weights_out: list[Fact],
    facts_map: dict[str, int],
    fact_weights_layer: str,
):
    weights = list(neurons.input_weights)
    ids = [_get_str_id_from_int(w.id) for w in weights]

    if len(ids) == 0:
        return None

    for w, w_id in zip(weights, ids):
        if w_id not in weights_out and w_id not in facts_map:
            val = w.get_value_numpy()
            val = np.reshape(val, [1, *atleast_2d_shape(val.shape)])
            if not w.learnable:
                if np.all(val == 1.0):
                    fact = UNIT_FACT
                else:
                    fact = ValueFact(val)
                fact_weights_out.append(fact)
                facts_map[w_id] = len(fact_weights_out) - 1
            else:
                weights_out[w_id] = LearnableWeight(val)

    ordinals: list[int] = []
    types: list[int] = []
    layer_ids: list[str] = []

    for id in ids:
        if id in facts_map:
            ordinals.append(facts_map[id])
            types.append(Refs.TYPE_FACT)
            layer_ids.append(fact_weights_layer)
        else:
            ordinals.append(0)
            types.append(Refs.TYPE_WEIGHT)
            layer_ids.append(id)

    return Refs(
        types=types,
        layer_ids=layer_ids,
        ordinals=ordinals,
    )


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
            out.append(UNIT_FACT)
        else:
            out.append(ValueFact(np.reshape(value, [1, *atleast_2d_shape(value.shape)])))

    return FactLayer(out)


def build_initial_network(network: Network) -> VectorizedLayerNetwork:
    fact_layers: dict[str, FactLayer] = {}
    weights: dict[str, LearnableWeight] = {}
    layers: OrderedDict[str, Layer] = OrderedDict()

    fact_weights: list[Fact] = []
    fact_weight_map: dict[str, int] = {}

    FACT_WEIGHTS_LAYER_KEY = "w"

    for layer, neurons in network.items():
        try:
            if layer.type == "FactLayer":
                fact_layers[layer.id] = _build_fact_layer(neurons)
            else:
                # TODO: support multiple different transforms/aggregations/shapes here ?
                transform = _build_transform(neurons)
                reduce = _build_reduce(neurons)

                gather_source = _build_gather(fact_layers, neurons.inputs.ordinals)
                weight_source = _build_weights(
                    neurons,
                    weights_out=weights,
                    fact_weights_out=fact_weights,
                    facts_map=fact_weight_map,
                    fact_weights_layer=FACT_WEIGHTS_LAYER_KEY,
                )

                layer_base = _build_layer_base(gather_source, weight_source)
                the_layer = _build_layer(layer_base, reduce, transform)

                layers[layer.id] = the_layer
        except Exception as e:
            raise Exception(f"Exception in layer {layer.id}") from e

    fact_weights_layer = FactLayer(fact_weights)
    fact_layers[FACT_WEIGHTS_LAYER_KEY] = fact_weights_layer

    vectorized_net = VectorizedLayerNetwork(
        fact_layers=fact_layers, weights=weights, batches=OrderedDict([(0, Batch(layers))])
    )
    return vectorized_net
