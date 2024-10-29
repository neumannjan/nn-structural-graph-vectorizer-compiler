import copy
from typing import OrderedDict

import numpy as np
from compute_graph_vectorize.vectorize.model import *
from compute_graph_vectorize.vectorize.pipeline.layerwise import Layerwise
from compute_graph_vectorize.vectorize.pipeline.transpose_fixed_reduce_layers import TransposeFixedCountReduceLayers

WEIGHTS = [
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 5, 1)),
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 10, 1)),
    np.random.random_sample((1, 2, 10)),
    np.random.random_sample((1, 1, 5)),
    np.random.random_sample((1, 2, 10)),
]


NETWORK_INPUT = VectorizedLayerNetwork(
  fact_layers={
    "unit": FactLayer(
      facts=[UnitFact()], count=None,
      shape=VariousShape(),
    ),
  }, weights={
    "001": LearnableWeight(WEIGHTS[0]),
    "002": LearnableWeight(WEIGHTS[1]),
    "004": LearnableWeight(WEIGHTS[2]),
    "006": LearnableWeight(WEIGHTS[3]),
    "007": LearnableWeight(WEIGHTS[4]),
    "000": LearnableWeight(WEIGHTS[5]),
    "003": LearnableWeight(WEIGHTS[6]),
    "005": LearnableWeight(WEIGHTS[7]),
  }, batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("l1__ag", Layer(
          count=None, compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       layer_ids=["unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit"],
                       ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            weight=Refs(types=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                        layer_ids=["001", "002", "004", "006", "007", "unit", "001", "002", "004", "006", "007", "unit"],
                        ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          ),
          aggregate=UnevenReduce(counts=[2, 1, 3, 2, 1, 3], reduce="sum"),
          transform=Transform(transform="identity"),
          shape=VariousShape(),
        )),
        ("predict__a", Layer(
          count=None, compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2, 2, 2],
                       layer_ids=["l1__ag", "l1__ag", "l1__ag", "l1__ag", "l1__ag", "l1__ag"],
                       ordinals=[0, 1, 2, 3, 4, 5]),
            weight=Refs(types=[1, 1, 1, 1, 1, 1],
                        layer_ids=["000", "003", "005", "000", "003", "005"],
                        ordinals=[0, 0, 0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=3, reduce="sum", dim=1),
          transform=Transform(transform="relu"),
          shape=VariousShape(),
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
  }, weights={
    "001": LearnableWeight(WEIGHTS[0]),
    "002": LearnableWeight(WEIGHTS[1]),
    "004": LearnableWeight(WEIGHTS[2]),
    "006": LearnableWeight(WEIGHTS[3]),
    "007": LearnableWeight(WEIGHTS[4]),
    "000": LearnableWeight(WEIGHTS[5]),
    "003": LearnableWeight(WEIGHTS[6]),
    "005": LearnableWeight(WEIGHTS[7]),
  }, batches=OrderedDict([
    (0, Batch(
      layers=OrderedDict([
        ("l1__ag", Layer(
          count=None, compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       layer_ids=["unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit", "unit"],
                       ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            weight=Refs(types=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                        layer_ids=["001", "002", "004", "006", "007", "unit", "001", "002", "004", "006", "007", "unit"],
                        ordinals=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          ),
          aggregate=UnevenReduce(counts=[2, 1, 3, 2, 1, 3], reduce="sum"),
          transform=Transform(transform="identity"),
          shape=VariousShape(),
        )),
        ("predict__a", Layer(
          count=None, compilable=False,
          base=LinearLayerBase(
            lifts=None,
            input=Refs(types=[2, 2, 2, 2, 2, 2],
                       layer_ids=["l1__ag", "l1__ag", "l1__ag", "l1__ag", "l1__ag", "l1__ag"],
                       ordinals=[0, 3, 1, 4, 2, 5]),
            weight=Refs(types=[1, 1, 1, 1, 1, 1],
                        layer_ids=["000", "000", "003", "003", "005", "005"],
                        ordinals=[0, 0, 0, 0, 0, 0]),
          ),
          aggregate=FixedCountReduce(period=3, reduce="sum", dim=0),
          transform=Transform(transform="relu"),
          shape=VariousShape(),
        )),
      ])
    )),
  ])
)


def test_transpose_fixed_count_reduce():
    actual = Layerwise(TransposeFixedCountReduceLayers)(copy.deepcopy(NETWORK_INPUT))
    assert actual == NETWORK_EXPECTED
