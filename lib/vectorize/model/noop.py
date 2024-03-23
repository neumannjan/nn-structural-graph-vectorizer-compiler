from lib.vectorize.model.repr import repr_slots


class Noop:
    __slots__ = ()
    __repr__ = repr_slots
