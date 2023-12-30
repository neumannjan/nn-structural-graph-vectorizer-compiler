from neuralogic.core import R, Template, V
from neuralogic.utils.data import Mutagenesis

from .dataset import MyDataset


class MyMutagenesis(MyDataset):
    def __init__(self) -> None:
        _, dataset = Mutagenesis()
        super().__init__('mutagenesis', Template(), dataset)

        self.template.add_rules(
            [(R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]]
        )

        self.template.add_rules(
            [(R.bond_embed(V.B)[3,] <= R.get(bond)(V.B)) for bond in ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]]
        )

        self.template += R.layer_1(V.X) <= (
            R.atom_embed(V.X)[3, 3],
            R.atom_embed(V.Y)[3, 3],
            R.bond(V.X, V.Y, V.B),
            R.bond_embed(V.B),
        )
        self.template += R.layer_2(V.X) <= (
            R.layer_1(V.X)[3, 3],
            R.layer_1(V.Y)[3, 3],
            R.bond(V.X, V.Y, V.B),
            R.bond_embed(V.B),
        )
        self.template += R.layer_3(V.X) <= (
            R.layer_2(V.X)[3, 3],
            R.layer_2(V.Y)[3, 3],
            R.bond(V.X, V.Y, V.B),
            R.bond_embed(V.B),
        )
        self.template += R.predict[1, 3] <= R.layer_3(V.X)


