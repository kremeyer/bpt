import numpy as np
from numba import njit


@njit
def solve_trajectory(steps, dt, e_field, b_field, q, m, r, v):
    # allocate memory
    trajectory = np.empty(
        (steps, 6)
    )  # 6 states (r_x, r_y, r_z, v_x, v_y, v_z) for position and velocity
    trajectory[0, :3] = r
    trajectory[0, 3:] = v

    for i in range(1, steps):
        t = steps * dt
        e = e_field(t, trajectory[i - 1][:3])
        b = b_field(t, trajectory[i - 1][:3])

        q_prime = dt * q / (2 * m)
        h = q_prime * b
        s = 2 * h / (1 + np.dot(h, h))
        u = trajectory[i - 1][3:] + q_prime * e
        u_prime = u + np.cross(u + np.cross(u, h), s)

        trajectory[i][3:] = u_prime + (q_prime * e)  # new velocity
        trajectory[i][:3] = trajectory[i - 1][:3] + (dt * trajectory[i][3:])

    return trajectory


class Solver:
    trajectories = None

    def __init__(self, e_field, b_field, particles, steps=1e4, dt=1e-4):
        self.e_field = e_field
        self.b_field = b_field
        self.particles = particles
        self.steps = int(steps)
        self.dt = dt

    def solve(self):
        self.trajectories = np.empty((len(self.particles), self.steps, 6))
        for i, particle in enumerate(self.particles):
            print(f"solving trajectory for {particle}")
            self.trajectories[i] = solve_trajectory(
                self.steps,
                self.dt,
                (njit)(self.e_field),
                (njit)(self.b_field),
                particle.q,
                particle.m,
                particle.r,
                particle.v,
            )


class Electron:
    q = -1
    m = 1

    def __init__(self, r, v, name=""):
        if not isinstance(r, np.ndarray):
            self.r = np.array(r)
        else:
            self.r = r

        if not isinstance(v, np.ndarray):
            self.v = np.array(v)
        else:
            self.v = v

        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
