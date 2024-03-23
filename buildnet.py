from lib.benchmarks.utils.timer import Timer
from lib.datasets.mutagenesis import MyMutagenesis
from lib.datasets.tu_molecular import MyTUDataset
from lib.nn.definitions.settings import Settings
from lib.sources.builders import from_java
from lib.vectorize.pipeline.pipeline import build_vectorized_network

if __name__ == "__main__":
    settings = Settings()
    settings.neuralogic.iso_value_compression = False
    dataset = MyMutagenesis(settings, "simple", "original")
    # dataset = MyTUDataset(settings, "mutag", "gsage")

    print("Dataset:", dataset)
    print("Settings:", settings)

    print("Building dataset...")
    built_dataset = dataset.build(sample_run=False)

    network = from_java(built_dataset.samples, settings)

    timer = Timer(device="cpu", agg_skip_first=0)
    with timer:
        vectorized_network = build_vectorized_network(network)
    print(vectorized_network)
    print(timer.get_result())
