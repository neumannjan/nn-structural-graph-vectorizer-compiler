from compute_graph_vectorize.model.ops import TransformationDef
from compute_graph_vectorize.vectorize.model.repr import repr_slots


class Transform:
    __slots__ = ("transform",)
    __match_args__ = ("transform",)
    __repr__ = repr_slots

    def __init__(self, transform: TransformationDef) -> None:
        self.transform: TransformationDef = transform

    def __hash__(self) -> int:
        return hash(self.transform)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, Transform) and self.transform == value.transform
