import itertools
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence, overload

from neuralogic.core.builder.builder import NeuralSample
from tqdm.auto import tqdm

from lib.interfaces import JavaNeuron
from lib.nn.utils.mocks import FullMockJavaNeuron, MockJavaSample, MockNeuralSample


@dataclass
class IndexProviders:
    neuron: Iterator[int]
    weight: Iterator[int]


def _get_neuron_from_sample(sample: NeuralSample) -> JavaNeuron:
    return sample.java_sample.query.neuron


def build_new_index_providers(samples: Sequence[NeuralSample]) -> IndexProviders:
    max_existing_index = -1
    max_existing_weight_index = -1
    queue = deque([])

    for sample in tqdm(samples, desc="Finding max indices"):
        neuron = sample.java_sample.query.neuron
        queue.append(neuron)
        while len(queue) > 0:
            neuron: JavaNeuron = queue.popleft()

            idx = neuron.getIndex()
            if max_existing_index < idx:
                max_existing_index = idx

            if str(neuron.getClass().getSimpleName()).startswith('Weighted'):
                max_w_idx = max((w.index for w in neuron.getWeights()))
                if max_existing_weight_index < max_w_idx:
                    max_existing_weight_index = max_w_idx

            queue.extend((inp for inp in neuron.getInputs()))

    return IndexProviders(
        neuron=iter(itertools.count(max_existing_index + 1)),
        weight=iter(itertools.count(max_existing_weight_index + 1)),
    )


class SampleDuplicator:
    def __init__(self, all_samples: Sequence[NeuralSample]) -> None:
        self._all_samples = all_samples
        self._index_providers = build_new_index_providers(all_samples)

    def _duplicate_sample_no_inputs(self, neuron: JavaNeuron) -> FullMockJavaNeuron:
        return FullMockJavaNeuron(
            index=next(self._index_providers.neuron),
            layer=neuron.getLayer(),
            cls=neuron.getClass(),
            weights=neuron.getWeights() if str(neuron.getClass().getSimpleName()).startswith('Weighted') else [],
        )

    @overload
    def duplicate_sample(self, neuron: JavaNeuron) -> FullMockJavaNeuron:
        ...

    @overload
    def duplicate_sample(self, neuron: NeuralSample) -> MockNeuralSample:
        ...

    def duplicate_sample(self, neuron: JavaNeuron | NeuralSample) -> MockNeuralSample | FullMockJavaNeuron:
        is_sample = isinstance(neuron, NeuralSample)

        if is_sample:
            neuron = _get_neuron_from_sample(neuron)

        out = self._duplicate_sample_no_inputs(neuron)

        queue = deque([(neuron, out, inp) for inp in neuron.getInputs()])

        while len(queue) > 0:
            neuron, this_out, inp = queue.popleft()
            inp_out = self._duplicate_sample_no_inputs(inp)
            this_out.getInputs().append(inp_out)
            queue.extend(((inp, inp_out, iinp) for iinp in inp.getInputs()))

        if is_sample:
            out = MockNeuralSample(MockJavaSample(out))

        return out

    @overload
    def duplicate_samples(self, samples: Sequence[JavaNeuron], times: int = 1) -> Sequence[FullMockJavaNeuron]:
        ...

    @overload
    def duplicate_samples(self, samples: Sequence[NeuralSample], times: int = 1) -> Sequence[MockNeuralSample]:
        ...

    @overload
    def duplicate_samples(
        self, samples: Sequence[JavaNeuron | NeuralSample], times: int = 1
    ) -> Sequence[FullMockJavaNeuron | MockNeuralSample]:
        ...

    def duplicate_samples(
        self,
        samples: Sequence[JavaNeuron | NeuralSample],
        times: int = 1,
    ) -> Sequence[FullMockJavaNeuron | MockNeuralSample]:
        if times > 1:
            samples = list(samples) * times

        return list(tqdm((self.duplicate_sample(s) for s in samples), desc="Duplicating samples", total=len(samples)))

    @overload
    def extend_samples(self, samples: Sequence[JavaNeuron], times: int = 1) -> Sequence[FullMockJavaNeuron]:
        ...

    @overload
    def extend_samples(self, samples: Sequence[NeuralSample], times: int = 1) -> Sequence[MockNeuralSample]:
        ...

    @overload
    def extend_samples(
        self, samples: Sequence[JavaNeuron | NeuralSample], times: int = 1
    ) -> Sequence[FullMockJavaNeuron | MockNeuralSample]:
        ...

    def extend_samples(
        self,
        samples: Sequence[JavaNeuron | NeuralSample],
        times: int = 1,
    ) -> Sequence[FullMockJavaNeuron | MockNeuralSample]:
        assert times >= 2
        out = []
        out.extend(samples)
        out.extend(self.duplicate_samples(samples, times=times - 1))
        return out

    def duplicate_dataset(self, times: int = 1) -> Sequence[MockNeuralSample]:
        return self.duplicate_samples(self._all_samples, times=times)

    def extend_dataset(self, times: int = 2) -> Sequence[NeuralSample]:
        return self.extend_samples(self._all_samples, times=times)
