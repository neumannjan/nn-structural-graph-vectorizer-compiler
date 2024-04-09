from collections.abc import Callable
from pathlib import Path
from typing import Literal

from neuralogic.core import R, Template, V
from neuralogic.dataset.file import FileDataset

from lib.sources.neuralogic_settings import NeuralogicSettings

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

    print(template)

    # GCN se zapnutejma self-loopama a bez normalizace ?

    return template


def simple_template_no_bond_embed():
    template = Template()

    template.add_rules(
        [(R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]]
    )

    template += R.layer_1(V.X) <= (
        R.atom_embed(V.X)[3, 3],
        R.atom_embed(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
    )
    template += R.layer_2(V.X) <= (
        R.layer_1(V.X)[3, 3],
        R.layer_1(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
    )
    template += R.layer_3(V.X) <= (
        R.layer_2(V.X)[3, 3],
        R.layer_2(V.Y)[3, 3],
        R.bond(V.X, V.Y, V.B),
    )
    template += R.predict[1, 3] <= R.layer_3(V.X)

    return template


MutagenesisTemplate = Literal["simple", "simple_nobond"]


TEMPLATE_MAP: dict[MutagenesisTemplate, Callable[[], Template]] = {
    "simple": simple_template,
    "simple_nobond": simple_template_no_bond_embed,
}


MutagenesisSource = Literal["original", "10x"]


SOURCE_DIRECTORY_MAP: dict[MutagenesisSource, Path] = {
    "original": Path(".") / "datasets_preserved" / "mutagenesis",
    "10x": Path(".") / "datasets" / "mutagenesis_multip",
}


class MyMutagenesis(MyDataset):
    def __init__(
        self,
        settings: NeuralogicSettings,
        template: MutagenesisTemplate = "simple",
        source: MutagenesisSource = "original",
    ) -> None:
        self.source_name = source
        self.template_name = template
        directory = SOURCE_DIRECTORY_MAP[source]

        dataset = FileDataset(
            examples_file=str(directory / "examples.txt"),
            queries_file=str(directory / "queries.txt"),
        )

        the_template = TEMPLATE_MAP[template]

        super().__init__("mutagenesis", the_template, dataset, settings)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_name}, template={self.template_name})"
