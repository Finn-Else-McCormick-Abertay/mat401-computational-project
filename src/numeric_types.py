import numpy as np
from numpy.typing import NDArray


class Motion:
    _ArrT = NDArray[np.number]

    def __init__(self, r: _ArrT, v: _ArrT, a: _ArrT):
        self.set_position(r)
        self.set_velocity(v)
        self.set_acceleration(a)

    def get_position(self) -> _ArrT:
        return self._position

    def set_position(self, r: _ArrT):
        self._position = r

    def get_velocity(self) -> _ArrT:
        return self._velocity

    def set_velocity(self, v: _ArrT):
        self._velocity = v

    def get_acceleration(self) -> _ArrT:
        return self._acceleration

    def set_acceleration(self, a: _ArrT):
        self._acceleration = a

    position = property(get_position, set_position)
    r = property(get_position, set_position)

    velocity = property(get_velocity, set_velocity)
    v = property(get_velocity, set_velocity)

    acceleration = property(get_acceleration, set_acceleration)
    a = property(get_acceleration, set_acceleration)

    def deep_copy(self) -> Motion:
        return Motion(self.r, self.v, self.a)
