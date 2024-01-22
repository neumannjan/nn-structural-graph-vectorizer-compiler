from lib.benchmarks.runnables.neuralogic_cpu_runnable import NeuraLogicCPURunnable
from lib.benchmarks.runnables.torch_gather_runnable import TorchGatherRunnable
from lib.benchmarks.runner import MultiRunner
from lib.datasets.mutagenesis import MyMutagenesis
from lib.nn.utils.samples import SampleDuplicator
from tqdm.auto import tqdm

if __name__ == "__main__":
    runnables = {
        'torch_gather_mps': TorchGatherRunnable(device='mps'),
        'torch_gather_cpu': TorchGatherRunnable(device='cpu'),
        'neuralogic_cpu': NeuraLogicCPURunnable(),
    }

    runner = MultiRunner(n_repeats=100)

    dataset = MyMutagenesis().build()

    duplicator = SampleDuplicator(dataset.samples)

    samples = duplicator.extend_dataset(times=7)

    for runnable_name, runnable in tqdm(runnables.items(), desc="Runners"):
        runner.measure(runnable_name, runnable, dataset, samples)

    print(runner.get_result())
