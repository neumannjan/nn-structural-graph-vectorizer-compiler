import copy
from typing import OrderedDict

import numpy as np
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.compute_layer_counts import compute_layer_counts
from compute_graph_vectorize.vectorize.pipeline.compute_layer_shapes import compute_layer_shapes

WEIGHTS = {
    "000": LearnableWeight(value=np.random.random_sample([1, 10, 1])),
    "001": LearnableWeight(value=np.random.random_sample([1, 10, 1])),
    "002": LearnableWeight(value=np.random.random_sample([1, 10, 1])),
    "003": LearnableWeight(value=np.random.random_sample((1, 10, 10))),
    "004": LearnableWeight(value=np.random.random_sample((1, 5, 10))),
    "006": LearnableWeight(value=np.random.random_sample((1, 10, 10))),
    "007": LearnableWeight(value=np.random.random_sample((1, 10, 5))),
    "005": LearnableWeight(value=np.random.random_sample([1, 1, 10])),
  }


NETWORK_INPUT = VectorizedLayerNetwork(
  fact_layers={
    "unit": FactLayer(
      facts=[UnitFact()], count=None,
      shape=VariousShape(),
    ),
  }, weights=WEIGHTS,
  batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("emb__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0, 0, 0, 0], layer_ids=["unit", "unit", "unit", "unit", "unit"], ordinals=[0, 0, 0, 0, 0]),
            weight=Refs(types=[1, 1, 1, 1, 1], layer_ids=["000", "001", "002", "000", "001"], ordinals=[0, 0, 0, 0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__r", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 0, 2, 2, 0, 2, 2, 0],
                       layer_ids=["emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit"],
                       ordinals=[0, 1, 0, 0, 2, 0, 3, 4, 0]),
          ),
          aggregate=FixedCountReduce(period=3, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__ag", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 2], layer_ids=["l1_a__r", "l1_a__r", "l1_a__r"], ordinals=[0, 1, 2]),
          ),
          aggregate=UnevenReduce(counts=[2, 1], reduce="average"),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__ag", "l1_a__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["003", "003"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["emb__wa", "emb__wa"], ordinals=[0, 3]),
            weight=Refs(types=[1, 1], layer_ids=["004", "004"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2], layer_ids=["l1_a__wa", "l1_b__wa", "l1_a__wa", "l1_b__wa"], ordinals=[0, 0, 1, 1]),
            weight=Refs(types=[1, 1, 1, 1], layer_ids=["006", "007", "006", "007"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("predict__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["005", "005"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
      ])
    )),
  ])
)


NETWORK_EXPECTED_COUNTS = VectorizedLayerNetwork(
  fact_layers={
    "unit": FactLayer(
      facts=[UnitFact()], count=1,
      shape=VariousShape(),
    ),
  }, weights=WEIGHTS,
  batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("emb__wa", Layer(
          count=5, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0, 0, 0, 0], layer_ids=["unit", "unit", "unit", "unit", "unit"], ordinals=[0, 0, 0, 0, 0]),
            weight=Refs(types=[1, 1, 1, 1, 1], layer_ids=["000", "001", "002", "000", "001"], ordinals=[0, 0, 0, 0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__r", Layer(
          count=3, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 0, 2, 2, 0, 2, 2, 0],
                       layer_ids=["emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit"],
                       ordinals=[0, 1, 0, 0, 2, 0, 3, 4, 0]),
          ),
          aggregate=FixedCountReduce(period=3, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__ag", Layer(
          count=2, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 2], layer_ids=["l1_a__r", "l1_a__r", "l1_a__r"], ordinals=[0, 1, 2]),
          ),
          aggregate=UnevenReduce(counts=[2, 1], reduce="average"),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__wa", Layer(
          count=2, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__ag", "l1_a__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["003", "003"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=2, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["emb__wa", "emb__wa"], ordinals=[0, 3]),
            weight=Refs(types=[1, 1], layer_ids=["004", "004"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag", Layer(
          count=2, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2], layer_ids=["l1_a__wa", "l1_b__wa", "l1_a__wa", "l1_b__wa"], ordinals=[0, 0, 1, 1]),
            weight=Refs(types=[1, 1, 1, 1], layer_ids=["006", "007", "006", "007"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("predict__wa", Layer(
          count=2, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["005", "005"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
      ])
    )),
  ])
)


NETWORK_EXPECTED_SHAPES = VectorizedLayerNetwork(
  fact_layers={
    "unit": FactLayer(
      facts=[UnitFact()], count=None,
      shape=AnyShape(),
    ),
  }, weights=WEIGHTS,
  batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("emb__wa", Layer(
          count=None, shape=ConcreteShape([10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0, 0, 0, 0], layer_ids=["unit", "unit", "unit", "unit", "unit"], ordinals=[0, 0, 0, 0, 0]),
            weight=Refs(types=[1, 1, 1, 1, 1], layer_ids=["000", "001", "002", "000", "001"], ordinals=[0, 0, 0, 0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__r", Layer(
          count=None, shape=ConcreteShape([10, 1]), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 0, 2, 2, 0, 2, 2, 0],
                       layer_ids=["emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit", "emb__wa", "emb__wa", "unit"],
                       ordinals=[0, 1, 0, 0, 2, 0, 3, 4, 0]),
          ),
          aggregate=FixedCountReduce(period=3, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__ag", Layer(
          count=None, shape=ConcreteShape([10, 1]), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[2, 2, 2], layer_ids=["l1_a__r", "l1_a__r", "l1_a__r"], ordinals=[0, 1, 2]),
          ),
          aggregate=UnevenReduce(counts=[2, 1], reduce="average"),
          transform=Transform(transform="identity"),
        )),
        ("l1_a__wa", Layer(
          count=None, shape=ConcreteShape([10, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["l1_a__ag", "l1_a__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["003", "003"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=ConcreteShape([5, 1]), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["emb__wa", "emb__wa"], ordinals=[0, 3]),
            weight=Refs(types=[1, 1], layer_ids=["004", "004"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("predict__ag", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2], layer_ids=["l1_a__wa", "l1_b__wa", "l1_a__wa", "l1_b__wa"], ordinals=[0, 0, 1, 1]),
            weight=Refs(types=[1, 1, 1, 1], layer_ids=["006", "007", "006", "007"], ordinals=[0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=2, reduce="sum", dim=1),
          transform=Transform(transform="identity"),
        )),
        ("predict__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2], layer_ids=["predict__ag", "predict__ag"], ordinals=[0, 1]),
            weight=Refs(types=[1, 1], layer_ids=["005", "005"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
      ])
    )),
  ])
)


def test_compute_layer_counts():
    actual = compute_layer_counts(copy.deepcopy(NETWORK_INPUT))
    assert actual == NETWORK_EXPECTED_COUNTS


def test_compute_layer_shapes():
    actual = compute_layer_shapes(copy.deepcopy(NETWORK_INPUT))
    assert actual == NETWORK_EXPECTED_SHAPES
