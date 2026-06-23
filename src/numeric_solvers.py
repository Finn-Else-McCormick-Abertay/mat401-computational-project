import math
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from src.numeric_types import Motion


def simulate(
    solver: Callable,
    initial: NDArray[np.number] | Motion,
    context: Any,
    duration: float,
    step_duration: float = 0.1,
    array_name: Optional[str] = None,
    extra_recorder: Optional[Callable] = None,
):
    data = {}

    def record_data(data_point: NDArray[np.number] | Motion, time: float):
        nonlocal data
        nonlocal array_name
        if "time" not in data:
            data["time"] = []
        data["time"].append(time)

        def record_array(arr: NDArray[np.number]):
            nonlocal data
            nonlocal array_name
            prefix = "" if array_name is None else array_name + "_"

            index_names = {0: "x", 1: "y", 2: "z", 4: "w"}
            for i in range(arr.size):
                key = prefix + index_names[i] if i in index_names else str(i)
                if key not in data:
                    data[key] = []
                data[key].append(arr[i])

            if extra_recorder is not None:
                extra_recorder(data, arr, context)

        if isinstance(data_point, Motion):
            record_array(data_point.r)
        else:
            record_array(data_point)  # type: ignore

    record_data(initial, 0)

    working = initial
    for step_index in range(math.floor(duration / step_duration)):
        working = solver(working, step_duration, context)
        record_data(working, step_index * step_duration)

    # duration_remainder = duration - step_count * step_duration

    return data


# -- Solvers -- #


def euler(old: Motion, time_step: float, context) -> Motion:
    new = old.deep_copy()
    new.r = old.r + old.v * time_step
    new.v = old.v + old.a * time_step
    return new


def semi_implicit_euler(old: Motion, time_step: float, context) -> Motion:
    new = old.deep_copy()
    new.v = old.v + old.a * time_step
    new.r = old.r + new.v * time_step
    return new


class RungeKutta:
    _ArrT = NDArray[np.number]
    _CallableT = Callable[[_ArrT, float, Any], _ArrT]

    def __init__(self, function: _CallableT):
        self.function = function

    def __call__(self, old: _ArrT, time_step: float, context) -> _ArrT:
        raise NotImplementedError


class RungeKuttaSecondOrder(RungeKutta):
    def __call__(
        self, old: RungeKutta._ArrT, time_step: float, context
    ) -> RungeKutta._ArrT:
        k1 = self.function(old, 0, context)
        k2 = self.function(old + k1 * time_step / 2, time_step / 2, context)
        return old + k2 * time_step


class RungeKuttaFourthOrder(RungeKutta):
    def __call__(
        self, old: RungeKutta._ArrT, time_step: float, context
    ) -> RungeKutta._ArrT:
        k1 = self.function(old, 0, context)
        k2 = self.function(old + k1 * time_step / 2, time_step / 2, context)
        k3 = self.function(old + k2 * time_step / 2, time_step / 2, context)
        k4 = self.function(old + k3 * time_step, time_step, context)
        return old + (k1 + 2 * k2 + 2 * k3 + k4) * time_step / 6
