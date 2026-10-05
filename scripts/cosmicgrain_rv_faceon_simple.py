#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("cosmicgrain.mplstyle")

rv = np.loadtxt(
    "halo441_current_maps_for_FT/250pc_RV_unmasked.txt"
)

pixel = 0.25
radius = 2.5

ny, nx = rv.shape

x = (np.arange(nx) - 0.5*(nx - 1)) * pixel
y = (np.arange(ny) - 0.5*(ny - 1)) * pixel
xx, yy = np.meshgrid(x, y)

# Same R <= 2.5 kpc aperture as Z&G
inside = np.hypot(xx, yy) <= radius

show = rv.copy()
show[~inside] = np.nan

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    show,
    origin="lower",
    extent=[
        x[0] - pixel/2, x[-1] + pixel/2,
        y[0] - pixel/2, y[-1] + pixel/2
    ],
    cmap="coolwarm_r",
    interpolation="none",
)

ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.set_aspect("equal")

ax.set_xlabel("x (kpc)")
ax.set_ylabel("y (kpc)")
ax.set_title("CosmicGrain CG10 face-on R(V)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("R(V)")

fig.tight_layout()

fig.savefig(
    "halo441_current_RV_250pc_faceon_R2p5.png",
    dpi=220
)
fig.savefig(
    "halo441_current_RV_250pc_faceon_R2p5.pdf"
)

plt.close(fig)
