import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from beautiful_particle_tracer import Electron, Solver


def e_field(t, r):
    return np.array([0, 0, 0])


def b_field(t, r):
    return np.array([0, 1, 0])


electrons = [
    Electron([x_start, 0, 0], [1, 0, 0], str(x_start))
    for x_start in np.linspace(-3, 3, 5)
]
s = Solver(e_field, b_field, electrons, steps=1e4, dt=1e-3)
s.solve()

f = plt.figure()
ax = f.add_subplot(111, projection="3d")

for i in range(len(electrons)):
    ax.scatter(
        s.trajectories[i, ::10, 0],
        s.trajectories[i, ::10, 1],
        s.trajectories[i, ::10, 2],
    )

plt.show()
