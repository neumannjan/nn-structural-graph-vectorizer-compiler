import pytest
from compute_graph_vectorize.datasets.dataset import BuiltDatasetInstance
from compute_graph_vectorize.datasets.mutagenesis import MyMutagenesis
from compute_graph_vectorize.datasets.tu_molecular import MyTUDataset
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def nsettings() -> NeuralogicSettings:
    n_settings = NeuralogicSettings()
    n_settings.iso_value_compression = False
    n_settings.chain_pruning = False

    return n_settings


@pytest.fixture
def tu_mutag_gsage(device: str, nsettings: NeuralogicSettings) -> BuiltDatasetInstance:
    dataset = MyTUDataset(nsettings, "mutag", "gsage")
    built_dataset_inst = dataset.build(sample_run=False)
    return built_dataset_inst


@pytest.fixture
def mutag(device: str, nsettings: NeuralogicSettings) -> BuiltDatasetInstance:
    dataset = MyMutagenesis(nsettings, "original", "simple")
    built_dataset_inst = dataset.build(sample_run=False)
    return built_dataset_inst
