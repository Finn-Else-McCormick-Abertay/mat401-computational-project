import math
from typing import Any, Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from graph import DataSet
from src.numeric_types import Motion
from type_checking import type_matches


class MagnitudeRecorder:
    def __init__(self, key="magnitude"):
        self.key = key

    def __call__(self, data_point: DataSet.DataPoint, arr: NDArray, ctx: Any):
        # Don't ask me why the function you use to get the magnitude is called 'norm'
        data_point[self.key] = np.linalg.norm(arr)

type SolverCallable = Callable[[NDArray[np.number] | Motion, float, Any],NDArray[np.number] | Motion]
type RecorderCallable = Callable[[DataSet.DataPoint, NDArray, Any]]

def simulate(
    solver: SolverCallable,
    initial: NDArray[np.number] | Motion,
    duration: float,
    step_duration: float = 0.1,
    name: Optional[str] = None,
    context: Any = None,
    recorder_cb: List[RecorderCallable] | RecorderCallable | None = None,
):
    if not type_matches(recorder_cb, List) and recorder_cb is not None:
        recorder_cb = [recorder_cb] # type: ignore
    elif recorder_cb is None:
        recorder_cb = []
             
    data: DataSet[float] = DataSet(x_key="time")

    def record_data(motion: NDArray[np.number] | Motion, time: float):
        nonlocal data
        nonlocal name
        nonlocal context
        nonlocal recorder_cb

        final_array: NDArray = motion.r if type_matches(motion, Motion) else motion # type: ignore

        data_point: DataSet[float].DataPoint = {
            "time": time,
            name if name else "__unnamed__": final_array
        }
        for cb in recorder_cb: # type: ignore
            cb(data_point, final_array, context)

        data.add_data(data_point)

    record_data(initial, 0)

    working = initial
    for step_index in range(math.floor(duration / step_duration)):
        working = solver(working, step_duration, context)
        record_data(working, step_index * step_duration)

    # duration_remainder = duration - step_count * step_duration

    return data