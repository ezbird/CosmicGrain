#!/usr/bin/env python3

"""
COSMICGRAIN GAS VISUALIZER

Presentation-oriented face-on and edge-on rendering of gas particles.

VISUALIZATION CONVENTION
------------------------
Each PartType0 gas particle is represented by a 2-D Gaussian whose

    FWHM = physical PartType0/SmoothingLength

therefore

    sigma = h / (2 sqrt(2 ln 2)) = h / 2.35482.

Each Gaussian is normalized so that its integral over an infinite plane
equals the particle gas mass.

IMPORTANT:
This is a visualization prescription. It is NOT the native GADGET-4
Wendland C4 SPH kernel reconstruction.

Coordinates and smoothing lengths are converted from comoving kpc/h
to physical kpc:

    x_phys = a/h * x_code
    hsml_phys = a/h * hsml_code

Masses are converted from 1e10 Msun/h to Msun.
"""

import argparse
import glob

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# Visualization scale:
# Gaussian FWHM = 0.25 * physical SmoothingLength
FWHM_TO_SIGMA = 0.25 / 2.3548200450309493


# ============================================================================
# Utilities
# ============================================================================

def scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def get_attribute(header, parameters, names):
    for name in names:
        if parameters is not None and name in parameters.attrs:
            return scalar(parameters.attrs[name])

        if name in header.attrs:
            return scalar(header.attrs[name])

    raise RuntimeError(
        "Could not find attribute: " + ", ".join(names)
    )


def snapshot_info(filename):
    with h5py.File(filename, "r") as f:
        header = f["Header"]
        parameters = f.get("Parameters")

        a = scalar(header.attrs["Time"])
        box = scalar(header.attrs["BoxSize"])

        hubble = get_attribute(
            header,
            parameters,
            ("HubbleParam",),
        )

    return a, box, hubble


def load_particles(files, ptype, fields):
    output = {
        field: []
        for field in fields
    }

    for filename in files:

        with h5py.File(filename, "r") as f:

            name = f"PartType{ptype}"

            if name not in f:
                continue

            group = f[name]

            n = len(group["Coordinates"])

            for field in fields:

                if field == "Masses" and field not in group:

                    mass = f["Header"].attrs[
                        "MassTable"
                    ][ptype]

                    if mass <= 0:
                        raise RuntimeError(
                            f"No mass information for {name}"
                        )

                    output[field].append(
                        np.full(
                            n,
                            mass,
                            dtype=float,
                        )
                    )

                else:

                    if field not in group:
                        raise RuntimeError(
                            f"{name}/{field} missing"
                        )

                    output[field].append(
                        group[field][:]
                    )

    if not output[fields[0]]:
        return None

    return {
        key: np.concatenate(value)
        for key, value in output.items()
    }


def periodic_delta(position, center, box):
    delta = position.astype(float) - center

    delta -= (
        box
        * np.rint(
            delta / box
        )
    )

    return delta


# ============================================================================
# Orientation
# ============================================================================

def make_rotation_matrix(j):
    zhat = j / np.linalg.norm(j)

    reference = np.array(
        [0.0, 0.0, 1.0]
    )

    if abs(reference @ zhat) > 0.9:
        reference = np.array(
            [0.0, 1.0, 0.0]
        )

    xhat = np.cross(
        reference,
        zhat,
    )

    xhat /= np.linalg.norm(xhat)

    yhat = np.cross(
        zhat,
        xhat,
    )

    return np.vstack(
        (xhat, yhat, zhat)
    )


# ============================================================================
# Gaussian deposition
# ============================================================================

def gaussian_map(
    x,
    y,
    mass,
    hsml,
    xlim,
    ylim,
    nx,
    ny,
    gaussian_extent=4.0,
    label="Gaussian",
):
    """
    Deposit particles as mass-normalized 2-D Gaussians.

    FWHM = hsml

    sigma = hsml / 2.35482

    Surface density is returned in Msun/pc^2.
    """

    xmin, xmax = xlim
    ymin, ymax = ylim

    dx = (
        xmax - xmin
    ) / nx

    dy = (
        ymax - ymin
    ) / ny

    xc = (
        xmin
        + (
            np.arange(nx)
            + 0.5
        ) * dx
    )

    yc = (
        ymin
        + (
            np.arange(ny)
            + 0.5
        ) * dy
    )

    sigma_map = np.zeros(
        (ny, nx),
        dtype=np.float64,
    )

    represented_mass = 0.0

    n = len(mass)

    report_every = max(
        1,
        n // 20,
    )

    for i in range(n):

        if i % report_every == 0:
            print(
                f"  {label}: "
                f"{i:,}/{n:,} "
                f"({100.0*i/max(n,1):5.1f}%)"
            )

        xi = x[i]
        yi = y[i]
        mi = mass[i]
        hi = hsml[i]

        if (
            not np.isfinite(xi)
            or not np.isfinite(yi)
            or not np.isfinite(mi)
            or not np.isfinite(hi)
            or mi <= 0.0
            or hi <= 0.0
        ):
            continue

        sigma = (
            hi
            * FWHM_TO_SIGMA
        )

        radius = (
            gaussian_extent
            * sigma
        )

        if (
            xi + radius < xmin
            or xi - radius > xmax
            or yi + radius < ymin
            or yi - radius > ymax
        ):
            continue

        ix0 = max(
            0,
            int(
                np.floor(
                    (
                        xi
                        - radius
                        - xmin
                    ) / dx
                )
            ),
        )

        ix1 = min(
            nx - 1,
            int(
                np.floor(
                    (
                        xi
                        + radius
                        - xmin
                    ) / dx
                )
            ),
        )

        iy0 = max(
            0,
            int(
                np.floor(
                    (
                        yi
                        - radius
                        - ymin
                    ) / dy
                )
            ),
        )

        iy1 = min(
            ny - 1,
            int(
                np.floor(
                    (
                        yi
                        + radius
                        - ymin
                    ) / dy
                )
            ),
        )

        if ix1 < ix0 or iy1 < iy0:
            continue

        xx, yy = np.meshgrid(
            xc[ix0:ix1 + 1],
            yc[iy0:iy1 + 1],
        )

        r2 = (
            (xx - xi)**2
            + (yy - yi)**2
        )

        kernel = np.exp(
            -0.5
            * r2
            / sigma**2
        )

        # Infinite 2-D Gaussian normalization.
        kernel *= (
            1.0
            / (
                2.0
                * np.pi
                * sigma**2
            )
        )

        # Msun / kpc^2
        contribution = (
            mi
            * kernel
        )

        sigma_map[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += contribution

    print(
        f"  {label}: "
        f"{n:,}/{n:,} "
        f"(100.0%)"
    )

    # Numerical integral represented inside displayed image.
    represented_mass = (
        np.sum(sigma_map)
        * dx
        * dy
    )

    # Msun/kpc^2 -> Msun/pc^2
    sigma_map /= 1.0e6

    return sigma_map, represented_mass


# ============================================================================
# Display range
# ============================================================================

def display_limits(
    arrays,
    requested_vmin=None,
    requested_vmax=None,
    percentile_low=2.0,
    percentile_high=99.7,
    display_floor=0.03,
):
    positive = []

    for array in arrays:

        mask = (
            np.isfinite(array)
            & (array > 0.0)
        )

        positive.append(
            array[mask]
        )

    positive = np.concatenate(
        positive
    )

    if positive.size == 0:
        raise RuntimeError(
            "No positive map pixels"
        )

    if requested_vmin is None:

        vmin = np.percentile(
            positive,
            percentile_low,
        )

        vmin = max(
            vmin,
            display_floor,
        )

    else:

        vmin = requested_vmin

    if requested_vmax is None:

        vmax = np.percentile(
            positive,
            percentile_high,
        )

    else:

        vmax = requested_vmax

    if vmax <= vmin:
        raise RuntimeError(
            f"Bad display limits: "
            f"{vmin}, {vmax}"
        )

    return vmin, vmax


# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "snapshot",
)

parser.add_argument(
    "--center",
    nargs=3,
    type=float,
    required=True,
)

parser.add_argument(
    "--orientation-radius",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--xy-half-size",
    type=float,
    default=15.0,
)

parser.add_argument(
    "--edge-z-half-size",
    type=float,
    default=6.0,
)

parser.add_argument(
    "--faceon-zmax",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--edgeon-ymax",
    type=float,
    default=5.0,
)

parser.add_argument(
    "--pixels",
    type=int,
    default=400,
)

parser.add_argument(
    "--gaussian-extent",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--display-floor",
    type=float,
    default=0.03,
    help=(
        "Minimum automatic display vmin "
        "in Msun/pc^2"
    ),
)

parser.add_argument(
    "--vmin",
    type=float,
    default=None,
)

parser.add_argument(
    "--vmax",
    type=float,
    default=None,
)

parser.add_argument(
    "--cmap",
    default="magma",
)

parser.add_argument(
    "--prefix",
    default="cosmicgrain_gas_visual",
)

args = parser.parse_args()


# ============================================================================
# Snapshot
# ============================================================================

files = sorted(
    glob.glob(
        args.snapshot
    )
)

if not files:
    raise FileNotFoundError(
        args.snapshot
    )


print("=" * 78)
print("COSMICGRAIN GAS VISUALIZER")
print("=" * 78)

print(
    "Visualization prescription:"
)

print(
    "  Gaussian FWHM = physical PartType0/SmoothingLength"
)

print(
    "  Gaussian integral = particle gas mass"
)

print(
    "  This is a visualization convention, not the native SPH kernel."
)

print()

a, box, hubble = snapshot_info(
    files[0]
)

z = (
    1.0 / a
    - 1.0
)

print(
    f"Snapshot pieces: {len(files)}"
)

print(
    f"a = {a:.8f}"
)

print(
    f"z = {z:.8f}"
)

print(
    f"h = {hubble:.8f}"
)

center = np.asarray(
    args.center,
    dtype=float,
)


# ============================================================================
# Stellar orientation
# ============================================================================

print(
    "\nLoading stars..."
)

stars = load_particles(
    files,
    4,
    (
        "Coordinates",
        "Masses",
        "Velocities",
    ),
)

if stars is None:
    raise RuntimeError(
        "No PartType4"
    )


star_position = (
    periodic_delta(
        stars["Coordinates"],
        center,
        box,
    )
    * a
    / hubble
)

star_mass = (
    stars["Masses"].astype(float)
    * 1.0e10
    / hubble
)

star_velocity = (
    stars["Velocities"].astype(float)
)


radius = np.linalg.norm(
    star_position,
    axis=1,
)

inside = (
    radius
    < args.orientation_radius
)

bulk_velocity = np.average(
    star_velocity[inside],
    axis=0,
    weights=star_mass[inside],
)

angular_momentum = np.sum(
    np.cross(
        star_position[inside],
        star_velocity[inside]
        - bulk_velocity,
    )
    * star_mass[inside, None],
    axis=0,
)

j_hat = (
    angular_momentum
    / np.linalg.norm(
        angular_momentum
    )
)

rotation = make_rotation_matrix(
    angular_momentum
)

print(
    f"Stars inside {args.orientation_radius:g} kpc: "
    f"{inside.sum():,}"
)

print(
    "j_hat =",
    j_hat,
)


# ============================================================================
# Gas
# ============================================================================

print(
    "\nLoading gas..."
)

gas = load_particles(
    files,
    0,
    (
        "Coordinates",
        "Masses",
        "SmoothingLength",
    ),
)


gas_position = (
    periodic_delta(
        gas["Coordinates"],
        center,
        box,
    )
    * a
    / hubble
)

gas_position = (
    gas_position
    @ rotation.T
)

gas_mass = (
    gas["Masses"].astype(float)
    * 1.0e10
    / hubble
)

gas_h = (
    gas["SmoothingLength"].astype(float)
    * a
    / hubble
)


finite = (
    np.all(
        np.isfinite(
            gas_position
        ),
        axis=1,
    )
    & np.isfinite(gas_mass)
    & np.isfinite(gas_h)
    & (gas_mass > 0)
    & (gas_h > 0)
)


print(
    f"Total gas particles: "
    f"{len(gas_mass):,}"
)


# ============================================================================
# Selections
# ============================================================================

x = gas_position[:, 0]
y = gas_position[:, 1]
zz = gas_position[:, 2]


face = (
    finite
    & (
        np.abs(zz)
        <= args.faceon_zmax
    )
)


edge = (
    finite
    & (
        np.abs(y)
        <= args.edgeon_ymax
    )
)


print(
    "\nProjection selections"
)

print(
    "-" * 78
)

print(
    f"Face-on: |z'| <= "
    f"{args.faceon_zmax:g} kpc "
    f"-> {face.sum():,} particles"
)

print(
    f"Edge-on: |y'| <= "
    f"{args.edgeon_ymax:g} kpc "
    f"-> {edge.sum():,} particles"
)


# h statistics for selected particles

for name, selection in (
    ("Face-on", face),
    ("Edge-on", edge),
):

    hsel = gas_h[
        selection
    ]

    q = np.percentile(
        hsel,
        [
            0,
            25,
            50,
            75,
            90,
            95,
            99,
            100,
        ],
    )

    print(
        f"\n{name} physical h:"
    )

    for percentile, value in zip(
        [
            0,
            25,
            50,
            75,
            90,
            95,
            99,
            100,
        ],
        q,
    ):

        print(
            f"  p{percentile:>3}: "
            f"{value:10.4f} kpc"
        )


# ============================================================================
# Maps
# ============================================================================

xlim = (
    -args.xy_half_size,
    args.xy_half_size,
)

face_ylim = xlim

edge_ylim = (
    -args.edge_z_half_size,
    args.edge_z_half_size,
)


ny_edge = max(
    50,
    int(
        round(
            args.pixels
            * args.edge_z_half_size
            / args.xy_half_size
        )
    ),
)


print(
    "\nBuilding face-on Gaussian rendering..."
)

face_map, face_mass_rendered = (
    gaussian_map(
        x[face],
        y[face],
        gas_mass[face],
        gas_h[face],
        xlim,
        face_ylim,
        args.pixels,
        args.pixels,
        gaussian_extent=
            args.gaussian_extent,
        label="Face-on",
    )
)


print(
    "\nBuilding edge-on Gaussian rendering..."
)

edge_map, edge_mass_rendered = (
    gaussian_map(
        x[edge],
        zz[edge],
        gas_mass[edge],
        gas_h[edge],
        xlim,
        edge_ylim,
        args.pixels,
        ny_edge,
        gaussian_extent=
            args.gaussian_extent,
        label="Edge-on",
    )
)


vmin, vmax = display_limits(
    (
        face_map,
        edge_map,
    ),
    requested_vmin=args.vmin,
    requested_vmax=args.vmax,
    display_floor=args.display_floor,
)


print(
    "\nDisplay"
)

print(
    "-" * 78
)

print(
    f"vmin = {vmin:.6e} Msun/pc^2"
)

print(
    f"vmax = {vmax:.6e} Msun/pc^2"
)


norm = LogNorm(
    vmin=vmin,
    vmax=vmax,
)


# ============================================================================
# Plot helper
# ============================================================================

def draw(
    data,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
    filename,
    figsize,
):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    image = ax.imshow(
        data,
        origin="lower",
        extent=(
            xlim[0],
            xlim[1],
            ylim[0],
            ylim[1],
        ),
        cmap=args.cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    cb = fig.colorbar(
        image,
        ax=ax,
    )

    cb.set_label(
        r"$\Sigma_{\rm gas}$ "
        r"($M_\odot\,{\rm pc}^{-2}$)"
    )

    fig.tight_layout()

    fig.savefig(
        filename + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        filename + ".pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# Individual plots
# ============================================================================

draw(
    face_map,
    xlim,
    face_ylim,
    "x (kpc)",
    "y (kpc)",
    "Halo 441 — Gas, face-on",
    args.prefix + "_faceon",
    (7.0, 6.2),
)


draw(
    edge_map,
    xlim,
    edge_ylim,
    "x (kpc)",
    "z (kpc)",
    "Halo 441 — Gas, edge-on",
    args.prefix + "_edgeon",
    (8.0, 4.5),
)


# ============================================================================
# Combined plot
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.0, 5.5),
)


image = axes[0].imshow(
    face_map,
    origin="lower",
    extent=(
        *xlim,
        *face_ylim,
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
    aspect="equal",
)


axes[1].imshow(
    edge_map,
    origin="lower",
    extent=(
        *xlim,
        *edge_ylim,
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
    aspect="equal",
)


axes[0].set_title(
    "Face-on"
)

axes[1].set_title(
    "Edge-on"
)


axes[0].set_xlabel(
    "x (kpc)"
)

axes[0].set_ylabel(
    "y (kpc)"
)


axes[1].set_xlabel(
    "x (kpc)"
)

axes[1].set_ylabel(
    "z (kpc)"
)


fig.suptitle(
    "Halo 441 — Gas morphology",
    fontsize=16,
)


cb = fig.colorbar(
    image,
    ax=axes,
    fraction=0.035,
    pad=0.04,
)

cb.set_label(
    r"$\Sigma_{\rm gas}$ "
    r"($M_\odot\,{\rm pc}^{-2}$)"
)


fig.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.12,
    top=0.87,
    wspace=0.25,
)


fig.savefig(
    args.prefix
    + "_faceon_edgeon.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    args.prefix
    + "_faceon_edgeon.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================================
# Save numerical maps
# ============================================================================

np.savez_compressed(
    args.prefix
    + "_maps.npz",

    faceon_sigma_gas_msun_pc2=
        face_map,

    edgeon_sigma_gas_msun_pc2=
        edge_map,

    xlim_kpc=
        np.asarray(xlim),

    faceon_ylim_kpc=
        np.asarray(face_ylim),

    edgeon_zlim_kpc=
        np.asarray(edge_ylim),

    faceon_zmax_kpc=
        args.faceon_zmax,

    edgeon_ymax_kpc=
        args.edgeon_ymax,

    j_hat=
        j_hat,

    rotation_matrix=
        rotation,

    gaussian_fwhm_definition=
        "FWHM = physical PartType0/SmoothingLength",

    gaussian_extent_sigma=
        args.gaussian_extent,

    display_vmin=
        vmin,

    display_vmax=
        vmax,

    faceon_selected_particles=
        face.sum(),

    edgeon_selected_particles=
        edge.sum(),

    faceon_selected_mass_msun=
        np.sum(
            gas_mass[face]
        ),

    edgeon_selected_mass_msun=
        np.sum(
            gas_mass[edge]
        ),

    faceon_rendered_mass_msun=
        face_mass_rendered,

    edgeon_rendered_mass_msun=
        edge_mass_rendered,

    scale_factor=
        a,

    redshift=
        z,

    hubble=
        hubble,
)


# ============================================================================
# Finish
# ============================================================================

print(
    "\n" + "=" * 78
)

print(
    "FINISHED"
)

print(
    "=" * 78
)

print(
    "Visualization convention:"
)

print(
    "  Gaussian FWHM = physical gas smoothing length"
)

print(
    f"  Gaussian rendered to "
    f"{args.gaussian_extent:g} sigma"
)

print(
    f"\nFace-on selected mass = "
    f"{np.sum(gas_mass[face]):.6e} Msun"
)

print(
    f"Face-on mass represented inside image = "
    f"{face_mass_rendered:.6e} Msun"
)

print(
    f"\nEdge-on selected mass = "
    f"{np.sum(gas_mass[edge]):.6e} Msun"
)

print(
    f"Edge-on mass represented inside image = "
    f"{edge_mass_rendered:.6e} Msun"
)

print(
    "\nSaved:"
)

for suffix in (
    "_faceon.png",
    "_faceon.pdf",
    "_edgeon.png",
    "_edgeon.pdf",
    "_faceon_edgeon.png",
    "_faceon_edgeon.pdf",
    "_maps.npz",
):

    print(
        " ",
        args.prefix + suffix,
    )
