#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cmasher as cmr

plt.style.use("cosmicgrain.mplstyle")

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

ZG_FILE = "zhang_green_rv_250pc_products.npz"
CG_FILE = "halo441_current_maps_for_FT/250pc_RV_unmasked.txt"

PIXEL = 0.25       # kpc
RADIUS = 2.5       # kpc


# ------------------------------------------------------------
# Load the two 250-pc R(V) maps
# ------------------------------------------------------------

zg = np.load(ZG_FILE)
zg_rv = np.asarray(zg["rv"], dtype=float)

cg_full = np.loadtxt(CG_FILE)

# CG exported map is 20 x 20 kpc at 250-pc pixels (80 x 80).
# Extract the central 5 x 5 kpc region to match the Z&G map.
nmatch = zg_rv.shape[0]       # 20 pixels
ny, nx = cg_full.shape

y0 = (ny - nmatch) // 2
x0 = (nx - nmatch) // 2

cg_rv = cg_full[
    y0:y0+nmatch,
    x0:x0+nmatch
]

print(f"Z&G map:     {zg_rv.shape}")
print(f"CG full map: {cg_full.shape}")
print(f"CG crop:     {cg_rv.shape}")

if zg_rv.shape != cg_rv.shape:
    raise ValueError(
        f"Final map shapes do not match: "
        f"Z&G={zg_rv.shape}, CG10={cg_rv.shape}"
    )


# ------------------------------------------------------------
# Same R <= 2.5 kpc aperture for both maps
# ------------------------------------------------------------

ny, nx = zg_rv.shape

x = (np.arange(nx) - 0.5 * (nx - 1)) * PIXEL
y = (np.arange(ny) - 0.5 * (ny - 1)) * PIXEL

xx, yy = np.meshgrid(x, y)
inside = np.hypot(xx, yy) <= RADIUS

zg_show = zg_rv.copy()
cg_show = cg_rv.copy()

zg_show[~inside] = np.nan
cg_show[~inside] = np.nan


# ------------------------------------------------------------
# ONE color scale for BOTH maps
# ------------------------------------------------------------

all_values = np.concatenate([
    zg_show[np.isfinite(zg_show)],
    cg_show[np.isfinite(cg_show)]
])

norm = Normalize(
    vmin=np.min(all_values),
    vmax=np.max(all_values)
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

extent = [
    x[0] - PIXEL/2,
    x[-1] + PIXEL/2,
    y[0] - PIXEL/2,
    y[-1] + PIXEL/2
]

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5.5),
    sharex=True,
    sharey=True
)

im = axes[0].imshow(
    zg_show,
    origin="lower",
    extent=extent,
    cmap=cmr.cosmic,
    norm=norm,
    interpolation="none"
)

axes[1].imshow(
    cg_show,
    origin="lower",
    extent=extent,
    cmap=cmr.cosmic,
    norm=norm,
    interpolation="none"
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

axes[0].set_title("Zhang & Green")
axes[1].set_title("CosmicGrain CG10")

for ax in axes:
    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    ax.set_aspect("equal")
    ax.set_xlabel("x (kpc)")

axes[0].set_ylabel("y (kpc)")

fig.suptitle(
    r"Face-on $R(V)$ at 250 pc",
    fontsize=16
)

# One shared colorbar
cbar = fig.colorbar(
    im,
    ax=axes,
    fraction=0.035,
    pad=0.03
)
cbar.set_label(r"$R(V)$")

fig.savefig(
    "zhang_green_vs_cg10_RV_250pc_faceon.png",
    dpi=220,
    bbox_inches="tight"
)

fig.savefig(
    "zhang_green_vs_cg10_RV_250pc_faceon.pdf",
    bbox_inches="tight"
)

plt.close(fig)
