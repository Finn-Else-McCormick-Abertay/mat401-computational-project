import math
from typing import Any, Callable, Optional

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

def simulate(
    solver: Callable[[NDArray[np.number] | Motion, float, Any],NDArray[np.number] | Motion],
    initial: NDArray[np.number] | Motion,
    context: Any,
    duration: float,
    step_duration: float = 0.1,
    array_name: Optional[str] = None,
    additional_recorder_callback: Optional[Callable[[DataSet.DataPoint, NDArray, Any]]] = None,
):
    data: DataSet[float] = DataSet(x_key="time")

    def record_data(motion: NDArray[np.number] | Motion, time: float):
        nonlocal data
        nonlocal array_name
        nonlocal context
        nonlocal additional_recorder_callback

        final_array: NDArray = motion.r if type_matches(motion, Motion) else motion # type: ignore

        data_point: DataSet[float].DataPoint = {
            "time": time,
            array_name if array_name else "__unnamed__": final_array
        }
        if additional_recorder_callback:
            additional_recorder_callback(data_point, final_array, context) # type: ignore

        data.add_data(data_point)

    record_data(initial, 0)

    working = initial
    for step_index in range(math.floor(duration / step_duration)):
        working = solver(working, step_duration, context)
        record_data(working, step_index * step_duration)

    # duration_remainder = duration - step_count * step_duration

    return data