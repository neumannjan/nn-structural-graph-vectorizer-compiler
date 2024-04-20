import copy

from lib.vectorize.model import *


def combine_gathers_(gather1: GenericGather | NoopGather, out: GenericGather):
    match gather1:
        case NoopGather():
            # nothing to do
            pass
        case GenericGather(ordinals1):
            for i in range(len(out.ordinals)):
                out.ordinals[i] = ordinals1[out.ordinals[i]]
        case _:
            raise ValueError(gather1)


def combine_gathers(gather1: GenericGather | NoopGather, gather2: GenericGather | NoopGather):
    match gather2:
        case NoopGather():
            return gather1

    gather2 = copy.deepcopy(gather2)
    combine_gathers_(gather1, gather2)
    return gather2
