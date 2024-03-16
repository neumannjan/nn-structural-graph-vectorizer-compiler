import argparse
from dataclasses import dataclass
from typing import Literal
from typing import get_args as t_get_args

from lib.datasets.mutagenesis import MutagenesisSource, MutagenesisTemplate, MyMutagenesis
from lib.datasets.tu_molecular import MyTUDataset, TUDatasetSource, TUDatasetTemplate
from lib.nn.topological.settings import Settings


@dataclass
class MutagenesisConfig:
    template: MutagenesisTemplate
    source: MutagenesisSource


@dataclass
class TUMolecularConfig:
    template: TUDatasetTemplate
    source: TUDatasetSource


DatasetIdentifier = Literal["mutagenesis", "tu_molecular"]


DatasetInfo = tuple[Literal["mutagenesis"], MutagenesisConfig] | tuple[Literal["tu_molecular"], TUMolecularConfig]


def add_parser_args_for_dataset(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(title="dataset", required=True, dest="dataset")

    mutagenesis = subparsers.add_parser("mutagenesis")
    mutagenesis.add_argument("--source", "-s", choices=t_get_args(MutagenesisSource), default="original")
    mutagenesis.add_argument("--template", "-t", choices=t_get_args(MutagenesisTemplate), default="simple")

    tu = subparsers.add_parser("tu_molecular")
    tu.add_argument("--source", "-s", choices=t_get_args(TUDatasetSource), default="mutag")
    tu.add_argument("--template", "-t", choices=t_get_args(TUDatasetTemplate), default="gcn")


def get_dataset_info_from_args(args: argparse.Namespace) -> DatasetInfo:
    dataset: DatasetIdentifier = args.dataset

    if dataset == "mutagenesis":
        return dataset, MutagenesisConfig(template=args.template, source=args.source)
    elif dataset == "tu_molecular":
        return dataset, TUMolecularConfig(template=args.template, source=args.source)
    else:
        raise ValueError()


def build_dataset(dataset_info: DatasetInfo, settings: Settings):
    if dataset_info[0] == "mutagenesis":
        return MyMutagenesis(settings, template=dataset_info[1].template, source=dataset_info[1].source)
    elif dataset_info[0] == "tu_molecular":
        return MyTUDataset(settings, source=dataset_info[1].source, template=dataset_info[1].template)
    else:
        raise ValueError()
