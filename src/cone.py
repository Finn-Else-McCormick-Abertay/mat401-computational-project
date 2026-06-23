import numpy as np


class Cone:
    def __init__(self, mass: float, radius: float, height: float):
        self._mass = mass
        self._radius = radius
        self._height = height
        self._recalculate_inertia_tensor()

    def _set_and_recalc(self, prop, val):
        self.__setattr__(prop, val)
        self._recalculate_inertia_tensor()

    def _recalculate_inertia_tensor(self):
        self._principal_moments_of_inertia = (3 / 20 * self.M) * np.array(
            [self.r**2 + (self.h**2) / 2, self.r**2 + (self.h**2) / 2, self.r**2]
        )

    M = property(
        lambda self: self._mass, lambda self, val: self._set_and_recalc("_mass", val)
    )
    r = property(
        lambda self: self._radius,
        lambda self, val: self._set_and_recalc("_radius", val),
    )
    h = property(
        lambda self: self._height,
        lambda self, val: self._set_and_recalc("_height", val),
    )
    principal_moments_of_inertia = property(
        lambda self: self._principal_moments_of_inertia
    )


if __name__ == "__main__":
    cone = Cone(8, 3, 8)
    print(cone.principal_moments_of_inertia)
