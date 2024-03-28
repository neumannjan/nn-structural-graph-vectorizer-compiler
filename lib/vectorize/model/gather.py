from lib.vectorize.model.repr import repr_slots


class GenericGather:
    __slots__ = ("ordinals",)
    __repr__ = repr_slots

    def __init__(self, ordinals: list[int]) -> None:
        self.ordinals = ordinals


class TakeSingleValue:
    __slots__ = ("ordinal",)
    __repr__ = repr_slots

    def __init__(self, ordinal: int) -> None:
        self.ordinal = ordinal


class NoopGather:
    __slots__ = ()
    __repr__ = repr_slots


class SliceValues:
    __slots__ = ("start", "end", "step")
    __repr__ = repr_slots

    def __init__(self, start: int, end: int, step: int) -> None:
        self.start = start
        self.end = end
        self.step = step


class Repeat:
    __slots__ = ("times", "total_length")
    __repr__ = repr_slots

    def __init__(self, times: int, total_length: int) -> None:
        self.times = times
        self.total_length = total_length


class ViewWithPeriod:
    __slots__ = ("period",)
    __repr__ = repr_slots

    def __init__(self, period: int) -> None:
        self.period = period


OneGather = GenericGather | TakeSingleValue | NoopGather | SliceValues | Repeat | ViewWithPeriod


class GatherPair:
    __slots__ = ("a", "b")
    __match_args__ = ("a", "b")
    __repr__ = repr_slots

    def __init__(self, a: "Gather", b: "OneGather") -> None:
        self.a = a
        self.b = b


Gather = GenericGather | TakeSingleValue | NoopGather | SliceValues | Repeat | ViewWithPeriod | GatherPair


def _match_all(gather: Gather):
    match gather:
        case GenericGather(ordinals=ordinals):
            ...
        case TakeSingleValue(ordinal=ordinal):
            ...
        case NoopGather():
            ...
        case SliceValues(start=start, end=end, step=step):
            ...
        case Repeat(times=_, total_length=total_length):
            ...
        case ViewWithPeriod(period=period):
            ...
        case GatherPair(a, b):
            ...
        case _:
            assert False, f"{gather}"
