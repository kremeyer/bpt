import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import Electron, Solver, color_enumerate


def e_field(t, r):
    return np.array([0, 0, 0])


def b_field(t, r):
    return np.array([0, 1, 0])


electrons = [
    Electron([x_start, 0, 0], [1, 1, vz_start], str(x_start))
    for x_start, vz_start in zip(np.linspace(-3, 3, 5), np.linspace(-2, 2, 5))
]
s = Solver(e_field, b_field, electrons, steps=1e4, dt=1e-3)
s.solve()

f = plt.figure()
ax = f.add_subplot(111, projection="3d")

for i, c, _ in color_enumerate(electrons):
    ax.scatter(
        s.trajectories[i, ::10, 0],
        s.trajectories[i, ::10, 1],
        s.trajectories[i, ::10, 2],
        color=c,
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
