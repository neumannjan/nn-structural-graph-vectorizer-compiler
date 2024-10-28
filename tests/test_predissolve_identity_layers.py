from typing import OrderedDict

import numpy as np
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.dissolve_identity_layers import predissolve_identity_layers

WEIGHTS = [
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 5, 1)),
    np.random.random_sample((1, 2, 10)),
    np.random.random_sample((1, 1, 5)),
]

NETWORK_INPUT = VectorizedLayerNetwork(
    fact_layers={
        "a__f": FactLayer(
            facts=[ValueFact(value=np.array([1.0, 0.0, 0.0]).reshape([1, 3, 1]))],
            count=None,
            shape=VariousShape(),
        ),
        "b__f": FactLayer(
            facts=[ValueFact(value=np.array([0.0, 1.0, 0.0]).reshape([1, 3, 1]))],
            count=None,
            shape=VariousShape(),
        ),
    },
    weights={
        "001": LearnableWeight(WEIGHTS[0]),
        "002": LearnableWeight(WEIGHTS[1]),
        "004": LearnableWeight(WEIGHTS[2]),
        "000": LearnableWeight(WEIGHTS[3]),
        "003": LearnableWeight(WEIGHTS[4]),
    },
    batches=OrderedDict(
        [
            (
                0,
                Batch(
                    layers=OrderedDict(
                        [
                            (
                                "l1__wr",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=LinearLayerBase(
                                        lifts=None,
                                        input=Refs(
                                            types=[0, 0, 0, 0, 0, 0],
                                            layer_ids=["a__f", "b__f", "a__f", "a__f", "b__f", "a__f"],
                                            ordinals=[0, 0, 0, 0, 0, 0],
                                        ),
                                        weight=Refs(
                                            types=[1, 1, 1, 1, 1, 1],
                                            layer_ids=["001", "002", "004", "001", "002", "004"],
                                            ordinals=[0, 0, 0, 0, 0, 0],
                                        ),
                                    ),
                                    aggregate=UnevenReduce(counts=[2, 1, 2, 1], reduce="sum"),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "l1__ag",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=InputLayerBase(
                                        input=Refs(
                                            types=[2, 2, 2, 2],
                                            layer_ids=["l1__wr", "l1__wr", "l1__wr", "l1__wr"],
                                            ordinals=[0, 1, 2, 3],
                                        ),
                                    ),
                                    aggregate=Noop(),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "l1__wa",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=LinearLayerBase(
                                        lifts=None,
                                        input=Refs(
                                            types=[2, 2, 2, 2],
                                            layer_ids=["l1__ag", "l1__ag", "l1__ag", "l1__ag"],
                                            ordinals=[0, 1, 2, 3],
                                        ),
                                        weight=Refs(
                                            types=[1, 1, 1, 1],
                                            layer_ids=["000", "003", "000", "003"],
                                            ordinals=[0, 0, 0, 0],
                                        ),
                                    ),
                                    aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "predict__r",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=InputLayerBase(
                                        input=Refs(types=[2, 2], layer_ids=["l1__wa", "l1__wa"], ordinals=[0, 1]),
                                    ),
                                    aggregate=FixedCountReduce(period=1, reduce="sum", dim=0),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "predict__ag",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=InputLayerBase(
                                        input=Refs(
                                            types=[2, 2], layer_ids=["predict__r", "predict__r"], ordinals=[0, 1]
                                        ),
                                    ),
                                    aggregate=Noop(),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "predict__a",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=InputLayerBase(
                                        input=Refs(
                                            types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]
                                        ),
                                    ),
                                    aggregate=Noop(),
                                    transform=Transform(transform="relu"),
                                    shape=VariousShape(),
                                ),
                            ),
                        ]
                    )
                ),
            ),
        ]
    ),
)


NETWORK_EXPECTED = VectorizedLayerNetwork(
    fact_layers={
        "a__f": FactLayer(
            facts=[ValueFact(value=np.array([1.0, 0.0, 0.0]).reshape([1, 3, 1]))],
            count=None,
            shape=VariousShape(),
        ),
        "b__f": FactLayer(
            facts=[ValueFact(value=np.array([0.0, 1.0, 0.0]).reshape([1, 3, 1]))],
            count=None,
            shape=VariousShape(),
        ),
    },
    weights={
        "001": LearnableWeight(WEIGHTS[0]),
        "002": LearnableWeight(WEIGHTS[1]),
        "004": LearnableWeight(WEIGHTS[2]),
        "000": LearnableWeight(WEIGHTS[3]),
        "003": LearnableWeight(WEIGHTS[4]),
    },
    batches=OrderedDict(
        [
            (
                0,
                Batch(
                    layers=OrderedDict(
                        [
                            (
                                "l1__ag",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=LinearLayerBase(
                                        lifts=None,
                                        input=Refs(
                                            types=[0, 0, 0, 0, 0, 0],
                                            layer_ids=["a__f", "b__f", "a__f", "a__f", "b__f", "a__f"],
                                            ordinals=[0, 0, 0, 0, 0, 0],
                                        ),
                                        weight=Refs(
                                            types=[1, 1, 1, 1, 1, 1],
                                            layer_ids=["001", "002", "004", "001", "002", "004"],
                                            ordinals=[0, 0, 0, 0, 0, 0],
                                        ),
                                    ),
                                    aggregate=UnevenReduce(counts=[2, 1, 2, 1], reduce="sum"),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "l1__wa",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=LinearLayerBase(
                                        lifts=None,
                                        input=Refs(
                                            types=[2, 2, 2, 2],
                                            layer_ids=["l1__ag", "l1__ag", "l1__ag", "l1__ag"],
                                            ordinals=[0, 1, 2, 3],
                                        ),
                                        weight=Refs(
                                            types=[1, 1, 1, 1],
                                            layer_ids=["000", "003", "000", "003"],
                                            ordinals=[0, 0, 0, 0],
                                        ),
                                    ),
                                    aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
                                    transform=Transform(transform="identity"),
                                    shape=VariousShape(),
                                ),
                            ),
                            (
                                "predict__a",
                                Layer(
                                    count=None,
                                    compilable=False,
                                    base=InputLayerBase(
                                        input=Refs(types=[2, 2], layer_ids=["l1__wa", "l1__wa"], ordinals=[0, 1]),
                                    ),
                                    aggregate=FixedCountReduce(period=1, reduce="sum", dim=0),
                                    transform=Transform(transform="relu"),
                                    shape=VariousShape(),
                                ),
                            ),
                        ]
                    )
                ),
            ),
        ]
    ),
)


def test_predissolve_identity_layers():
    actual = predissolve_identity_layers(NETWORK_INPUT)
    print(actual)
    print(NETWORK_EXPECTED)
    assert actual == NETWORK_EXPECTED
