import copy
from typing import OrderedDict

import numpy as np
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.separate_input_refs import ShapeLayerIndexer, build_separate_input_refs

WEIGHTS = [
    np.random.random_sample((1, 10, 3)),
    np.random.random_sample((1, 5, 3)),
    np.random.random_sample((1, 10, 10)),
    np.random.random_sample((1, 10, 5)),
    np.random.random_sample((1, 1, 10)),
]

NETWORK_INPUT = VectorizedLayerNetwork(
  fact_layers={
    "a__f": FactLayer(
      facts=[ValueFact(value=np.array([1.0, 0.0, 0.0]).reshape([1, 3, 1]))], count=None,
      shape=ConcreteShape(dims=[3, 1]),
    ),
    "b__f": FactLayer(
      facts=[ValueFact(value=np.array([0.0, 1.0, 0.0]).reshape([1, 3, 1]))], count=None,
      shape=ConcreteShape(dims=[3, 1]),
    ),
  }, weights={
    "000": LearnableWeight(WEIGHTS[0]),
    "001": LearnableWeight(WEIGHTS[1]),
    "003": LearnableWeight(WEIGHTS[2]),
    "004": LearnableWeight(WEIGHTS[3]),
    "002": LearnableWeight(WEIGHTS[4]),
  }, batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("l1_a__ag", Layer(
          count=None, shape=ConcreteShape(dims=[3, 1]), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[0, 0, 0, 0], layer_ids=["a__f", "b__f", "a__f", "b__f"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__wa", Layer(
          count=None, shape=ConcreteShape(dims=[10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__ag", "l1_a__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["000", "000"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=ConcreteShape(dims=[5, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0], layer_ids=["a__f", "a__f"], ordinals=[0, 0]),
            weight=Refs(types=[1, 1], layer_ids=["001", "001"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2], layer_ids=["l1_a__wa", "l1_b__wa", "l1_a__wa", "l1_b__wa"], ordinals=[0, 0, 1, 1]),
            weight=Refs(types=[1, 1, 1, 1], layer_ids=["003", "004", "003", "004"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("predict__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["002", "002"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
      ])
    )),
  ])
)



NETWORK_EXPECTED = VectorizedLayerNetwork(
  fact_layers={
    "a__f": FactLayer(
      facts=[ValueFact(value=np.array([1.0, 0.0, 0.0]).reshape([1, 3, 1]))], count=None,
      shape=ConcreteShape(dims=[3, 1]),
    ),
    "b__f": FactLayer(
      facts=[ValueFact(value=np.array([0.0, 1.0, 0.0]).reshape([1, 3, 1]))], count=None,
      shape=ConcreteShape(dims=[3, 1]),
    ),
  }, weights={
    "000": LearnableWeight(WEIGHTS[0]),
    "001": LearnableWeight(WEIGHTS[1]),
    "003": LearnableWeight(WEIGHTS[2]),
    "004": LearnableWeight(WEIGHTS[3]),
    "002": LearnableWeight(WEIGHTS[4]),
  }, batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("l1_a__ag", Layer(
          count=None, shape=ConcreteShape(dims=[3, 1]), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[0, 0, 0, 0], layer_ids=["a__f", "b__f", "a__f", "b__f"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__wa", Layer(
          count=None, shape=ConcreteShape(dims=[10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__ag", "l1_a__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["000", "000"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=ConcreteShape(dims=[5, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0], layer_ids=["a__f", "a__f"], ordinals=[0, 0]),
            weight=Refs(types=[1, 1], layer_ids=["001", "001"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag__10_10__10_1", Layer(
          count=None, shape=ConcreteShape(dims=[10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__wa", "l1_a__wa"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["003", "003"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag__10_5__5_1", Layer(
          count=None, shape=ConcreteShape(dims=[10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_b__wa", "l1_b__wa"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["004", "004"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag", Layer(
          count=None, shape=ConcreteShape(dims=[10, 1]), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 2, 2],
                       layer_ids=["predict__ag__10_10__10_1", "predict__ag__10_5__5_1", "predict__ag__10_10__10_1", "predict__ag__10_5__5_1"],
                       ordinals=[0, 0, 1, 1]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("predict__wa", Layer(
          count=None, shape=ConcreteShape(dims=[1, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["002", "002"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
      ])
    )),
  ])
)


def test_separate_input_refs():
    actual = build_separate_input_refs(ShapeLayerIndexer)(copy.deepcopy(NETWORK_INPUT))
    print(actual)
    print(NETWORK_EXPECTED)
    assert actual == NETWORK_EXPECTED
