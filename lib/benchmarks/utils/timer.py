import math
import time

import torch


class TimerResult:
    def __init__(self, count: int, mean_ns: float, m2: float) -> None:
        self._count = count
        self._mean_ns = mean_ns
        self._m2_ns2 = m2

    @property
    def result_mean_ns(self) -> float:
        return self._mean_ns

    @property
    def result_mean_s(self) -> float:
        return self._mean_ns * 1e-9

    @property
    def result_variance_ns2(self) -> float:
        return self._m2_ns2 / (self._count - 1)

    @property
    def result_std_ns(self) -> float:
        return math.sqrt(self.result_variance_ns2)

    @property
    def result_std_s(self) -> float:
        return self.result_std_ns * 1e-9

    def __repr__(self) -> str:
        return "TimerResult<<%.04fs ± %.04fs>>" % (
            self.result_mean_s,
            self.result_std_s,
        )


class Timer:
    def __init__(self, device) -> None:
        self._start_time: int | None = None
        self._device = device
        self._result: TimerResult | None = None

    def _get_time_ns(self):
        return time.perf_counter_ns()

    def _synchronize(self):
        device = str(self._device).lower()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device.startswith("mps"):
            torch.mps.synchronize()

    def __enter__(self) -> "Timer":
        assert self._start_time is None
        self._synchronize()
        self._start_time = self._get_time_ns()

        return self

    def _register_run_time(self, time_ns: int):
        if self._result is None:
            self._result = TimerResult(count=1, mean_ns=float(time_ns), m2=0.0)
            return

        # Welford's online algorithm
        self._result._count += 1
        delta = time_ns - self._result._mean_ns
        self._result._mean_ns += delta / self._result._count
        delta2 = time_ns - self._result._mean_ns
        self._result._m2_ns2 += delta * delta2

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._start_time is not None

        self._synchronize()
        self._register_run_time(self._get_time_ns() - self._start_time)
        self._start_time = None

    def get_result(self) -> TimerResult:
        assert self._result is not None
        return self._result
