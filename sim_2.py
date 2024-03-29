import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import Electron, Solver, color_enumerate, reverse_sizes


################
# INIT PHYSICS #
################


def e_field(t, r):
    return np.array([0, 0, 0])


def b_field(t, r):
    return np.array([0, 0, 1])


N = 5
x_starts = np.linspace(-0.2, 0.2, N)
y_starts = np.zeros(N)
z_starts = np.linspace(0.8, 1.2, N)
vx_starts = np.zeros(N)
vy_starts = np.ones(N)
vz_starts = np.linspace(2, -2, N)
# vz_starts = np.zeros(N)

electrons = [
    Electron([x, y, z], [vx, vy, vz])
    for x, y, z, vx, vy, vz in zip(
        x_starts, y_starts, z_starts, vx_starts, vy_starts, vz_starts
    )
]


#################
# SOLVE PHYSICS #
#################

s = Solver(e_field, b_field, electrons, steps=1.5e3, dt=1e-3)
s.solve()
s.shift_trajectories()


###########
# 3D plot #
###########

f_3d = plt.figure(figsize=(6, 6))
ax_3d = f_3d.add_subplot(111, projection="3d")

for i, c, _ in color_enumerate(electrons):
    ax_3d.scatter(
        xs=s.trajectories[i, ::10, 0],
        ys=s.trajectories[i, ::10, 1],
        zs=s.trajectories[i, ::10, 2],
        color=c,
    )

ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")
ax_3d.set_zlabel("z")
ax_3d.set_xlim(0, np.max(s.trajectories[..., 0]))
ax_3d.set_ylim(0, np.max(s.trajectories[..., 1]))
ax_3d.set_zlim(0, np.max(s.trajectories[..., 2]))

f_3d.tight_layout()


###################
# projection plot #
###################

f_proj, axs_proj = plt.subplots(2, 3, figsize=(6, 4))

for i, c, _ in color_enumerate(electrons):
    for idxs in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        sizes = (s.trajectories[i, ::10, idxs[2]] + 0.1) * 50
        axs_proj[0, idxs[0]].scatter(
            x=s.trajectories[i, ::10, idxs[0]],
            y=s.trajectories[i, ::10, idxs[1]],
            s=sizes,
            color=c,
        )
        axs_proj[1, idxs[0]].scatter(
            x=-s.trajectories[i, ::10, idxs[0]],
            y=s.trajectories[i, ::10, idxs[1]],
            s=reverse_sizes(sizes),
            color=c,
        )


for ax in axs_proj.flatten():
    ax.axis("off")

f_proj.tight_layout()
f_proj.savefig(__file__.replace(".py", ".png"), dpi=1200)

# plt.show()
