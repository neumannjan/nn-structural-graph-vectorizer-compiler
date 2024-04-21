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


def combine_ord_maps_(ord_map1_out: dict[int, int], ord_map2: dict[int, int]):
    # combine out_idx_map and idx_map
    for a, b in list(ord_map1_out.items()):
        c = ord_map2.get(b, b)
        if a == c:
            if a in ord_map1_out:
                del ord_map1_out[a]
        else:
            ord_map1_out[a] = c


def combine_ord_maps(ord_map1: dict[int, int], ord_map2: dict[int, int]):
    out = copy.deepcopy(ord_map1)
    combine_ord_maps_(out, ord_map2)
    return out
