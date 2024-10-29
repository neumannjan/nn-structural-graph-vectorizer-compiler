import copy
from typing import OrderedDict

import numpy as np
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.simplify_pure_unit_fact_linears import simplify_pure_unit_fact_linears

WEIGHTS = {
  "000": LearnableWeight(value=np.random.random_sample((1, 10, 10))),
  "001": LearnableWeight(value=np.random.random_sample((1, 5, 10))),
  "003": LearnableWeight(value=np.random.random_sample((1, 10, 10))),
  "004": LearnableWeight(value=np.random.random_sample((1, 10, 5))),
  "002": LearnableWeight(value=np.random.random_sample([1, 1, 10])),
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
        ("l1_a__r", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[0, 0, 0, 0, 0, 0, 0, 0, 0], layer_ids=["unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit"], ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0]),
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
            weight=Refs(types=[1, 1], layer_ids=["000", "000"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0], layer_ids=["unit", "unit"], ordinals=[0, 0]),
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
    "unit": FactLayer(
      facts=[UnitFact()], count=None,
      shape=VariousShape(),
    ),
  }, weights=WEIGHTS,
    batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("l1_a__r", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[0, 0, 0, 0, 0, 0, 0, 0, 0], layer_ids=["unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit"], ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0]),
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
            weight=Refs(types=[1, 1], layer_ids=["000", "000"], ordinals=[0, 0]),
          ),
          aggregate=Noop(),
          transform=Transform(transform="identity"),
        )),
        ("l1_b__wa", Layer(
          count=None, shape=VariousShape(), compilable=False,
          base=InputLayerBase(
            input=Refs(types=[1, 1], layer_ids=["001", "001"], ordinals=[0, 0]),
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


def test_simplify_pure_unit_fact_linears():
  actual = simplify_pure_unit_fact_linears(copy.deepcopy(NETWORK_INPUT))
  assert actual == NETWORK_EXPECTED
