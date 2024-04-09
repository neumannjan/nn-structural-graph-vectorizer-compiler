from lib.model.ops import TransformationDef
from lib.vectorize.model.repr import repr_slots


class Transform:
    __slots__ = ("transform",)
    __match_args__ = ("transform",)
    __repr__ = repr_slots

    def __init__(self, transform: TransformationDef) -> None:
        self.transform: TransformationDef = transform
