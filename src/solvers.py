
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.numeric_types import Motion


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
