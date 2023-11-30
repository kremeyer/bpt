import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import Electron, Solver, color_enumerate, reverse_sizes, cmap_from_plotly
import plotly.express.colors as pcolors


################
# INIT PHYSICS #
################


def e_field(t, r):
    return np.array([0, 0, 0])


def b_field(t, r):
    return np.array([0.4, 0, 1])


N = 9
x_starts = np.linspace(-0.4, 0.2, N)
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

s = Solver(e_field, b_field, electrons, steps=3e3, dt=0.5e-3)
s.solve()
s.shift_trajectories()


###########
# 3D plot #
###########

# f_3d = plt.figure(figsize=(6, 6))
# ax_3d = f_3d.add_subplot(111, projection="3d")

# for i, c, _ in color_enumerate(electrons):
#     ax_3d.scatter(
#         xs=s.trajectories[i, ::10, 0],
#         ys=s.trajectories[i, ::10, 1],
#         zs=s.trajectories[i, ::10, 2],
#         color=c,
#     )

# ax_3d.set_xlabel("x")
# ax_3d.set_ylabel("y")
# ax_3d.set_zlabel("z")
# ax_3d.set_xlim(0, np.max(s.trajectories[..., 0]))
# ax_3d.set_ylim(0, np.max(s.trajectories[..., 1]))
# ax_3d.set_zlim(0, np.max(s.trajectories[..., 2]))

# f_3d.tight_layout()


###################
# projection plot #
###################

CMAP = pcolors.sequential.Sunsetdark

f_proj, axs_proj = plt.subplots(3, 2, figsize=(6, 9))
axs_proj = axs_proj.T

plt_cmap = cmap_from_plotly(CMAP)

for i, c, _ in color_enumerate(electrons, cmap=plt_cmap):
    for idxs in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        sizes = (s.trajectories[i, ::10, idxs[2]] + 1) * 500
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
    mi, ma = ax.get_xlim()
    ax.set_xlim(mi - 0.35, ma + 0.35)
    mi, ma = ax.get_ylim()
    ax.set_ylim(mi - 0.35, ma + 0.35)

f_proj.tight_layout()
f_proj.subplots_adjust(wspace=0, hspace=0)
f_proj.savefig(__file__.replace(".py", ".png"), dpi=1200)

# plt.show()
