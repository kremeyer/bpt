import re
import numpy as np
from numba import njit
from matplotlib.pyplot import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


@njit
def solve_trajectory(steps, dt, e_field, b_field, q, m, r, v):
    # allocate memory
    trajectory = np.empty(
        (steps, 6)
    )  # 6 states (r_x, r_y, r_z, v_x, v_y, v_z) for position and velocity
    trajectory[0, :3] = r
    trajectory[0, 3:] = v

    for i in range(1, steps):
        # boris algorithm
        # https://en.wikipedia.org/wiki/Particle-in-cell#The_particle_mover
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


def shift_trajectories_to_origin(trajectories, origin=(0, 0, 0)):
    for i in range(3):
        trajectories[..., i] -= np.min(trajectories[..., i]) + origin[i]
    return trajectories


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
        for i, particle in tqdm(enumerate(self.particles), total=len(self.particles)):
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

    def shift_trajectories(self):
        self.trajectories = shift_trajectories_to_origin(self.trajectories)


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


def color_enumerate(iterable, start=0, cmap=get_cmap("viridis")):
    """same functionality as enumerate, but additionally yields sequential colors from
    a given cmap
    """
    n = start
    length = len(iterable)
    for item in iterable:
        yield n, cmap(n / (length - 1)), item
        n += 1


def reverse_sizes(sizes):
    old_sizes_max = np.max(sizes)
    dist = old_sizes_max - np.min(sizes)
    new_sizes = dist - sizes
    shift = old_sizes_max - np.max(new_sizes)
    new_sizes += shift
    return new_sizes

def cmap_from_plotly(plotly_cmap, name='plotly_generated'):
    rgb_arr = np.array([rgb_str_to_tuple(rgb) for rgb in plotly_cmap]) / 255
    return LinearSegmentedColormap.from_list(name, rgb_arr)

def rgb_str_to_tuple(rgb_str):
    return tuple((int(match) for match in re.findall(r'\d+', rgb_str)))
