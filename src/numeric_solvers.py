import math


class Motion:
    def __init__(self, r, v, a):
        self.r = r
        self.v = v
        self.a = a


class NumericSolver:
    def __init__(self, step_time=0.1):
        self.step_time = step_time

    def perform_step(self, old: Motion) -> Motion:
        return old

    def simulate(self, initial_motion: Motion, duration: float, start_time: float = 0):
        data = {
            "time": [],
            "x": [],
            "y": [],
            "z": [],
        }
        step_count = math.floor(duration / self.step_time)

        working = initial_motion

        for i in range(step_count):
            data["time"].append(start_time + self.step_time * i)

            data["x"].append(working.r[0])
            data["y"].append(working.r[1])
            data["z"].append(working.r[2])

            working = self.perform_step(working)

        return data


class Euler(NumericSolver):
    def perform_step(self, old: Motion) -> Motion:
        v_new = old.v + old.a * self.step_time
        r_new = old.r + old.v * self.step_time
        return Motion(r_new, v_new, old.a)


class SemiImplicitEuler(NumericSolver):
    def perform_step(self, old: Motion) -> Motion:
        v_new = old.v + old.a * self.step_time
        r_new = old.r + v_new * self.step_time
        return Motion(r_new, v_new, old.a)
