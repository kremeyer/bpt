import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib import Electron, Solver, color_enumerate, reverse_sizes, cmap_from_plotly
import plotly.express.colors as pcolors


################
# INIT PHYSICS #
################


def e_field(t, r):
    return np.array([1 / (r[2] + 0.1), 0, 0])


def b_field(t, r):
    return np.array([0, r[1] + 1, 0])


N = 9
x_starts = np.linspace(-0.4, 0.2, N)
y_starts = np.linspace(-0.4, 0.2, N)
z_starts = np.zeros(N)
vx_starts = np.zeros(N)
vy_starts = np.zeros(N) + 0.1
vz_starts = np.zeros(N)

electrons = [
    Electron([x, y, z], [vx, vy, vz])
    for x, y, z, vx, vy, vz in zip(
        x_starts, y_starts, z_starts, vx_starts, vy_starts, vz_starts
    )
]


#################
# SOLVE PHYSICS #
#################

s = Solver(e_field, b_field, electrons, steps=3e3, dt=5e-3)
s.solve()
s.shift_trajectories()


###########
# 3D plot #
###########

# f_3d = plt.figure(figsize=(6, 6))
# ax_3d = f_3d.add_subplot(111, projection="3d")

# for i, c, _ in color_enumerate(electrons):
#     ax_3d.scatter(
#         xs=s.trajectories[i, :, 0],
#         ys=s.trajectories[i, :, 1],
#         zs=s.trajectories[i, :, 2],
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

CMAP = pcolors.sequential.deep

f_proj, axs_proj = plt.subplots(3, 2, figsize=(6, 9))
axs_proj = axs_proj.T

plt_cmap = cmap_from_plotly(CMAP)

step = 237
for i, c, _ in color_enumerate(electrons, cmap=plt_cmap):
    for idxs in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        sizes = (s.trajectories[i, ::step, idxs[2]] + 1) * 50
        sizes_r = reverse_sizes(sizes)
        for j, _ in enumerate(s.trajectories[0, ::step]):
            axs_proj[0, idxs[0]].scatter(
                x=s.trajectories[i, j * step, idxs[0]],
                y=s.trajectories[i, j * step, idxs[1]],
                s=sizes[i],
                zorder=s.trajectories[i, j * step, idxs[2]],
                color=c,
            )
            axs_proj[0, idxs[0]].margins(x=0.2, y=0.2)
            axs_proj[1, idxs[0]].scatter(
                x=-s.trajectories[i, j * step, idxs[0]],
                y=s.trajectories[i, j * step, idxs[1]],
                s=sizes_r[i],
                zorder=-s.trajectories[i, j * step, idxs[2]],
                color=c,
            )
            axs_proj[1, idxs[0]].margins(x=0.2, y=0.2)


for ax in axs_proj.flatten():
    ax.axis("off")

f_proj.tight_layout()
f_proj.subplots_adjust(wspace=0, hspace=0)
f_proj.savefig(__file__.replace(".py", ".png"), dpi=1200)

# plt.show()
