import time

import numpy as np
import torch


class TimerResult:
    def __init__(self, times: list[int], agg_skip_first=2) -> None:
        self._times = np.array(times)
        self._agg_skip_first = 2

    @property
    def result_mean_ns(self) -> float:
        return self._times[self._agg_skip_first :].mean()

    @property
    def result_mean_s(self) -> float:
        return self.result_mean_ns * 1e-9

    @property
    def result_std_ns(self) -> float:
        return self._times[self._agg_skip_first :].std()

    @property
    def result_std_s(self) -> float:
        return self.result_std_ns * 1e-9

    @property
    def times_ns(self):
        return self._times

    @property
    def times_s(self):
        return self._times * 1e-9

    def __repr__(self) -> str:
        return "TimerResult<<%.04fs ± %.04fs>>" % (
            self.result_mean_s,
            self.result_std_s,
        )


class Timer:
    def __init__(self, device) -> None:
        self._start_time: int | None = None
        self._device = device
        self._times: list[int] = []

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
        self._times.append(time_ns)

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._start_time is not None

        self._synchronize()
        self._register_run_time(self._get_time_ns() - self._start_time)
        self._start_time = None

    def get_result(self) -> TimerResult:
        return TimerResult(self._times)
