import time

import torch


class TimerResult:
    def __init__(self, accumulated_time_ns: int, runs: int) -> None:
        self._accumulated_time_ns = accumulated_time_ns
        self._runs = runs

    @property
    def result_ns(self) -> int:
        return self._accumulated_time_ns // self._runs

    @property
    def result_s(self) -> float:
        return (self._accumulated_time_ns / self._runs) * 1e-9

    def __repr__(self) -> str:
        return "TimerResult<<avg time: %.04f s>>" % (self.result_s)


class Timer(TimerResult):
    def __init__(self, device) -> None:
        super().__init__(accumulated_time_ns=0, runs=0)
        self._start_time: int | None = None
        self._device = device

    def _get_time_ns(self):
        return time.perf_counter_ns()

    def _synchronize(self):
        device = str(self._device).lower()

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elif device.startswith('mps'):
            torch.mps.synchronize()

    def __enter__(self) -> "Timer":
        assert self._start_time is None
        self._synchronize()
        self._start_time = self._get_time_ns()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._start_time is not None

        self._synchronize()
        self._accumulated_time_ns += self._get_time_ns() - self._start_time
        self._runs += 1

        self._start_time = None

    def get_result(self) -> TimerResult:
        return TimerResult(accumulated_time_ns=self._accumulated_time_ns, runs=self._runs)
