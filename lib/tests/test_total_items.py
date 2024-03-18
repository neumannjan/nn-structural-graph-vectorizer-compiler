import random
from typing import Callable

import pytest
import torch
from lib.nn.gather import GatherModuleLike, GenericGather, SliceValues, TakeEachNth, TakeValue

CLASSES = []


N_ORDINALS = 1000


SIMPLE_GATHER_FACTORIES: list[tuple[Callable[[], GatherModuleLike], int]] = [
    (lambda: TakeValue(ordinal=10), 1),
    (lambda: SliceValues(start=10, end=100), 90),
    (
        lambda: TakeEachNth(step=4, start=10, end=100),
        23,
    ),
    (lambda: GenericGather([random.randint(0, N_ORDINALS - 1) for _ in range(182)]), 182),
]

# TODO some actual important tests


@pytest.mark.parametrize(["gather_factory", "expected"], SIMPLE_GATHER_FACTORIES)
def test_simple_gather(gather_factory: Callable[[], GatherModuleLike], expected: int):
    inp = torch.tensor(list(range(N_ORDINALS)))
    gather = gather_factory()
    actual = gather.total_items

    output = gather(inp)
    assert len(output) == expected
    assert actual == expected
