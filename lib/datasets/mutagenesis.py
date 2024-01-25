from pathlib import Path

from neuralogic.core import R, Template, V
from neuralogic.dataset.file import FileDataset

from .dataset import MyDataset


def simple_template():
    template = Template()

    template.add_rules(
        [(R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]]
    )

    template.add_rules(
        [(R.bond_embed(V.B)[3,] <= R.get(bond)(V.B)) for bond in ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]]
    )

    template += R.layer_1(V.X) <= (
        R.atom_embed(V.X)[3, 3],
        R.atom_embed(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
        R.bond_embed(V.B),
    )
    template += R.layer_2(V.X) <= (
        R.layer_1(V.X)[3, 3],
        R.layer_1(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
        R.bond_embed(V.B),
    )
    template += R.layer_3(V.X) <= (
        R.layer_2(V.X)[3, 3],
        R.layer_2(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
        R.bond_embed(V.B),
    )
    template += R.predict[1, 3] <= R.layer_3(V.X)

    return template


class MyMutagenesis(MyDataset):
    def __init__(self) -> None:
        directory = Path(".") / "dataset" / "mutagenesis"

        dataset = FileDataset(
            examples_file=str(directory / "examples.txt"),
            queries_file=str(directory / "queries.txt"),
        )

        template = simple_template()

        super().__init__("mutagenesis", template, dataset)


class MyMutagenesisMultip(MyDataset):
    def __init__(self) -> None:
        directory = Path(".") / "dataset" / "mutagenesis_multip"

        dataset = FileDataset(
            examples_file=str(directory / "examples.txt"),
            queries_file=str(directory / "queries.txt"),
        )

        template = simple_template()

        super().__init__("mutagenesis", template, dataset)
