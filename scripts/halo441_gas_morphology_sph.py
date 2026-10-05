#!/usr/bin/env python3
"""
Make adaptively smoothed face-on and edge-on gas surface-density maps.

Gas particles are deposited as 2D Gaussians whose FWHM is proportional to
PartType0/SmoothingLength:

    FWHM_i = hsml_fwhm_factor * h_i

The Gaussian is normalized in physical space, so integrating Sigma over the
infinite plane returns the particle mass. Gaussian wings falling outside the
map are NOT renormalized back into the displayed region.

The disk orientation is determined from the stellar angular momentum inside
--orientation-radius, matching the convention used by the existing Halo 441
stellar morphology scripts.

Outputs:
    <prefix>_faceon.png/pdf
    <prefix>_edgeon.png/pdf
    <prefix>_faceon_edgeon.png/pdf
    <prefix>_maps.npz
"""

import argparse
import glob

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# ---------------------------------------------------------------------------
# Basic snapshot helpers
# ---------------------------------------------------------------------------

def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def attribute(header, parameters, names, default=None):
    for name in names:
        if parameters is not None and name in parameters.attrs:
            return scalar(parameters.attrs[name])
        if name in header.attrs:
            return scalar(header.attrs[name])

    if default is not None:
        return default

    raise RuntimeError(
        "Missing snapshot attribute: " + ", ".join(names)
    )


def snapshot_info(filename):
    with h5py.File(filename, "r") as handle:
        header = handle["Header"]
        parameters = handle.get("Parameters")

        scale_factor = scalar(header.attrs["Time"])
        box = scalar(header.attrs["BoxSize"])
        hubble = attribute(header, parameters, ("HubbleParam",))

    return scale_factor, box, hubble


def load_particles(files, particle_type, fields):
    """
    Load requested fields from one particle type across all snapshot pieces.
    """

    result = {field: [] for field in fields}

    for filename in files:
        with h5py.File(filename, "r") as handle:
            group_name = f"PartType{particle_type}"

            if group_name not in handle:
                continue

            group = handle[group_name]
            count = len(group["Coordinates"])

            for field in fields:
                if field == "Masses" and field not in group:
                    mass = handle["Header"].attrs["MassTable"][particle_type]

                    if mass <= 0:
                        raise RuntimeError(
                            f"No Masses field or MassTable entry for "
                            f"PartType{particle_type}"
                        )

                    result[field].append(
                        np.full(count, mass, dtype=float)
                    )

                else:
                    if field not in group:
                        raise RuntimeError(
                            f"{group_name}/{field} missing in {filename}"
                        )

                    result[field].append(group[field][:])

    if not result[fields[0]]:
        return None

    return {
        field: np.concatenate(values)
        for field, values in result.items()
    }


def periodic_delta(coordinates, center, box):
    delta = coordinates.astype(float) - center
    delta -= box * np.rint(delta / box)
    return delta


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def rotation_matrix(angular_momentum):
    """
    Construct rotation matrix whose z-axis is parallel to angular momentum.
    """

    zhat = angular_momentum / np.linalg.norm(angular_momentum)

    reference = np.array([0.0, 0.0, 1.0])

    if abs(reference @ zhat) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    xhat = np.cross(reference, zhat)
    xhat /= np.linalg.norm(xhat)

    yhat = np.cross(zhat, xhat)

    return np.vstack((xhat, yhat, zhat))


# ---------------------------------------------------------------------------
# Adaptive Gaussian deposition
# ---------------------------------------------------------------------------

def gaussian_surface_density(
    x,
    y,
    mass,
    smoothing_length,
    xlim,
    ylim,
    nx,
    ny,
    hsml_fwhm_factor=1.0,
    kernel_sigma_extent=4.0,
    minimum_sigma_pixels=0.35,
):
    """
    Deposit particles onto a 2D surface-density grid using adaptive Gaussians.

    Parameters
    ----------
    x, y : arrays
        Particle projected coordinates in physical kpc.

    mass : array
        Particle masses in Msun.

    smoothing_length : array
        Physical gas smoothing lengths in kpc.

    hsml_fwhm_factor : float
        Defines

            Gaussian FWHM = hsml_fwhm_factor * SmoothingLength

    kernel_sigma_extent : float
        Evaluate each Gaussian out to this many sigma.

    minimum_sigma_pixels : float
        Numerical floor on sigma relative to pixel size. This avoids
        undersampling extremely narrow Gaussians. It is NOT a physical
        smoothing floor.

    Returns
    -------
    sigma_msun_pc2 : ndarray
        Surface density in Msun / pc^2.

    deposited_mass : float
        Mass represented inside the finite image after Gaussian truncation
        at both the map boundaries and kernel_sigma_extent.
    """

    xmin, xmax = xlim
    ymin, ymax = ylim

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    pixel_area_kpc2 = dx * dy

    x_centers = xmin + (np.arange(nx) + 0.5) * dx
    y_centers = ymin + (np.arange(ny) + 0.5) * dy

    surface_density = np.zeros((ny, nx), dtype=np.float64)

    # FWHM = 2 sqrt(2 ln 2) sigma = 2.354820045... sigma
    gaussian_sigma = (
        hsml_fwhm_factor
        * smoothing_length
        / 2.3548200450309493
    )

    # Only a numerical sampling floor.
    pixel_scale = np.sqrt(dx * dy)
    sigma_floor = minimum_sigma_pixels * pixel_scale

    gaussian_sigma = np.maximum(
        gaussian_sigma,
        sigma_floor,
    )

    n_particles = len(x)

    report_every = max(1, n_particles // 20)

    for i in range(n_particles):

        if i % report_every == 0:
            percentage = 100.0 * i / max(n_particles, 1)
            print(
                f"  Gaussian deposition: "
                f"{i:,}/{n_particles:,} "
                f"({percentage:5.1f}%)"
            )

        xi = x[i]
        yi = y[i]
        mi = mass[i]
        sigma = gaussian_sigma[i]

        if (
            not np.isfinite(xi)
            or not np.isfinite(yi)
            or not np.isfinite(mi)
            or not np.isfinite(sigma)
            or mi <= 0
            or sigma <= 0
        ):
            continue

        radius = kernel_sigma_extent * sigma

        # Quick rejection if the entire evaluated kernel lies off-image.
        if (
            xi + radius < xmin
            or xi - radius > xmax
            or yi + radius < ymin
            or yi - radius > ymax
        ):
            continue

        ix0 = max(
            0,
            int(np.floor((xi - radius - xmin) / dx))
        )

        ix1 = min(
            nx - 1,
            int(np.floor((xi + radius - xmin) / dx))
        )

        iy0 = max(
            0,
            int(np.floor((yi - radius - ymin) / dy))
        )

        iy1 = min(
            ny - 1,
            int(np.floor((yi + radius - ymin) / dy))
        )

        if ix1 < ix0 or iy1 < iy0:
            continue

        local_x = x_centers[ix0:ix1 + 1]
        local_y = y_centers[iy0:iy1 + 1]

        xx, yy = np.meshgrid(local_x, local_y)

        r2 = (
            (xx - xi) ** 2
            + (yy - yi) ** 2
        )

        # Proper normalized 2D Gaussian:
        #
        # W(R) = 1/(2 pi sigma^2) exp[-R^2/(2 sigma^2)]
        #
        # Units: kpc^-2
        kernel = (
            np.exp(-0.5 * r2 / sigma**2)
            / (2.0 * np.pi * sigma**2)
        )

        surface_density[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += mi * kernel

    print(
        f"  Gaussian deposition: "
        f"{n_particles:,}/{n_particles:,} (100.0%)"
    )

    # Surface density is currently Msun/kpc^2.
    #
    # 1 kpc^2 = 10^6 pc^2
    surface_density_pc2 = surface_density / 1.0e6

    # Estimate mass represented by the finite sampled image.
    deposited_mass = (
        np.sum(surface_density)
        * pixel_area_kpc2
    )

    return surface_density_pc2, deposited_mass


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def positive_limits(values, low_percentile=1.0, high_percentile=99.7):
    positive = values[
        np.isfinite(values) & (values > 0)
    ]

    if positive.size == 0:
        raise RuntimeError("Surface-density map contains no positive pixels")

    vmin = np.percentile(positive, low_percentile)
    vmax = np.percentile(positive, high_percentile)

    if vmax <= vmin:
        vmax = positive.max()

    return vmin, vmax


def style_axis(axis, xlim, ylim, xlabel, ylabel):
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_aspect("equal")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=(
        "Adaptive Gaussian gas morphology maps for a CosmicGrain halo."
    )
)

parser.add_argument(
    "snapshot",
    help="Snapshot glob, e.g. '../run/snapdir_029/snapshot_029.*.hdf5'"
)

parser.add_argument(
    "--center",
    nargs=3,
    type=float,
    required=True,
    metavar=("X", "Y", "Z"),
    help="Halo center in snapshot coordinate units"
)

parser.add_argument(
    "--orientation-radius",
    type=float,
    default=20.0,
    help="Physical radius in kpc used for stellar angular momentum"
)

parser.add_argument(
    "--xy-half-size",
    type=float,
    default=20.0,
    help="Face-on and edge-on horizontal half-width in physical kpc"
)

parser.add_argument(
    "--edge-z-half-size",
    type=float,
    default=10.0,
    help="Edge-on vertical half-height in physical kpc"
)

parser.add_argument(
    "--faceon-zmax",
    type=float,
    default=5.0,
    help="Face-on gas projection uses |z'| < this value in kpc"
)

parser.add_argument(
    "--edgeon-ymax",
    type=float,
    default=5.0,
    help="Edge-on gas projection uses |y'| < this value in kpc"
)

parser.add_argument(
    "--pixels",
    type=int,
    default=400,
    help="Number of horizontal pixels"
)

parser.add_argument(
    "--hsml-fwhm-factor",
    type=float,
    default=1.0,
    help="Gaussian FWHM / PartType0 SmoothingLength"
)

parser.add_argument(
    "--kernel-sigma-extent",
    type=float,
    default=4.0,
    help="Evaluate Gaussian kernels to this many sigma"
)

parser.add_argument(
    "--minimum-sigma-pixels",
    type=float,
    default=0.35,
    help="Numerical minimum Gaussian sigma in units of pixel size"
)

parser.add_argument(
    "--cmap",
    default="magma",
    help="Matplotlib colormap"
)

parser.add_argument(
    "--vmin",
    type=float,
    default=None,
    help="Minimum Sigma_gas in Msun/pc^2"
)

parser.add_argument(
    "--vmax",
    type=float,
    default=None,
    help="Maximum Sigma_gas in Msun/pc^2"
)

parser.add_argument(
    "--prefix",
    default="halo441_gas_morphology_sph"
)

args = parser.parse_args()


if args.hsml_fwhm_factor <= 0:
    raise ValueError("--hsml-fwhm-factor must be > 0")

if args.pixels < 10:
    raise ValueError("--pixels must be >= 10")


# ---------------------------------------------------------------------------
# Load snapshot
# ---------------------------------------------------------------------------

files = sorted(glob.glob(args.snapshot))

if not files:
    raise FileNotFoundError(args.snapshot)

print("=" * 78)
print("COSMICGRAIN ADAPTIVE GAS MORPHOLOGY")
print("=" * 78)
print(f"Snapshot pieces: {len(files)}")

scale_factor, box, hubble = snapshot_info(files[0])

redshift = 1.0 / scale_factor - 1.0

print(f"a = {scale_factor:.8f}")
print(f"z = {redshift:.8f}")
print(f"h = {hubble:.8f}")
print(f"BoxSize = {box:.8f}")

center = np.asarray(args.center, dtype=float)


# ---------------------------------------------------------------------------
# Stars: orientation only
# ---------------------------------------------------------------------------

print("\nLoading stars for disk orientation...")

stars = load_particles(
    files,
    4,
    ("Coordinates", "Masses", "Velocities")
)

if stars is None:
    raise RuntimeError("PartType4 stars are required for orientation")

star_position = (
    periodic_delta(
        stars["Coordinates"],
        center,
        box
    )
    * scale_factor
    / hubble
)

star_mass = (
    stars["Masses"].astype(float)
    * 1.0e10
    / hubble
)

star_velocity = stars["Velocities"].astype(float)

radius3d = np.linalg.norm(
    star_position,
    axis=1
)

orient = radius3d < args.orientation_radius

if orient.sum() < 3:
    raise RuntimeError(
        "Too few stars inside orientation radius"
    )

bulk_velocity = np.average(
    star_velocity[orient],
    axis=0,
    weights=star_mass[orient]
)

angular_momentum = np.sum(
    np.cross(
        star_position[orient],
        star_velocity[orient] - bulk_velocity
    )
    * star_mass[orient, None],
    axis=0
)

j_hat = (
    angular_momentum
    / np.linalg.norm(angular_momentum)
)

rotation = rotation_matrix(
    angular_momentum
)

print(f"Stars inside orientation radius: {orient.sum():,}")
print("j_hat =", j_hat)
print("rotation matrix:")
print(rotation)


# ---------------------------------------------------------------------------
# Gas
# ---------------------------------------------------------------------------

print("\nLoading gas...")

gas = load_particles(
    files,
    0,
    ("Coordinates", "Masses", "SmoothingLength")
)

if gas is None:
    raise RuntimeError("PartType0 gas is required")

gas_position = (
    periodic_delta(
        gas["Coordinates"],
        center,
        box
    )
    * scale_factor
    / hubble
)

gas_mass = (
    gas["Masses"].astype(float)
    * 1.0e10
    / hubble
)

# IMPORTANT:
# This assumes SmoothingLength is stored in the same comoving coordinate
# convention as Coordinates. We print diagnostics below so this can be
# checked directly against the snapshot and expected hydro resolution.
gas_h = (
    gas["SmoothingLength"].astype(float)
    * scale_factor
    / hubble
)

gas_rotated = gas_position @ rotation.T

print(f"Total gas particles: {len(gas_mass):,}")


# ---------------------------------------------------------------------------
# Smoothing-length diagnostics
# ---------------------------------------------------------------------------

finite_h = (
    np.isfinite(gas_h)
    & (gas_h > 0)
)

if not np.any(finite_h):
    raise RuntimeError(
        "No finite positive PartType0/SmoothingLength values"
    )

percentiles = [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    100,
]

h_values = np.percentile(
    gas_h[finite_h],
    percentiles
)

print("\nPhysical PartType0/SmoothingLength diagnostics")
print("-" * 78)

for percentile, value in zip(
    percentiles,
    h_values
):
    print(
        f"p{percentile:>3}: "
        f"{value:12.6f} kpc"
    )


# Also show smoothing lengths near the galaxy itself.
local = (
    finite_h
    & (np.linalg.norm(gas_rotated, axis=1) < 30.0)
)

if np.any(local):

    local_values = np.percentile(
        gas_h[local],
        percentiles
    )

    print("\nSmoothing lengths for gas within 30 physical kpc")
    print("-" * 78)

    for percentile, value in zip(
        percentiles,
        local_values
    ):
        print(
            f"p{percentile:>3}: "
            f"{value:12.6f} kpc"
        )


# ---------------------------------------------------------------------------
# Select gas for projections
# ---------------------------------------------------------------------------

xlim = (
    -args.xy_half_size,
    args.xy_half_size
)

ylim_face = xlim

zlim_edge = (
    -args.edge_z_half_size,
    args.edge_z_half_size
)

face_keep = (
    finite_h
    & (np.abs(gas_rotated[:, 2]) < args.faceon_zmax)
)

edge_keep = (
    finite_h
    & (np.abs(gas_rotated[:, 1]) < args.edgeon_ymax)
)

ny_edge = max(
    50,
    int(
        round(
            args.pixels
            * args.edge_z_half_size
            / args.xy_half_size
        )
    )
)

print("\nProjection selections")
print("-" * 78)
print(
    f"Face-on: |z'| < {args.faceon_zmax:g} kpc "
    f"-> {face_keep.sum():,} gas particles"
)
print(
    f"Edge-on: |y'| < {args.edgeon_ymax:g} kpc "
    f"-> {edge_keep.sum():,} gas particles"
)

print(
    f"Face-on grid: {args.pixels} x {args.pixels}"
)

print(
    f"Edge-on grid: {args.pixels} x {ny_edge}"
)

face_pixel_kpc = (
    2.0 * args.xy_half_size
    / args.pixels
)

edge_pixel_z_kpc = (
    2.0 * args.edge_z_half_size
    / ny_edge
)

print(
    f"Face-on pixel size: "
    f"{face_pixel_kpc:.4f} kpc"
)

print(
    f"Edge-on pixel size: "
    f"{face_pixel_kpc:.4f} x "
    f"{edge_pixel_z_kpc:.4f} kpc"
)


# ---------------------------------------------------------------------------
# Adaptive Gaussian projections
# ---------------------------------------------------------------------------

print("\nBuilding FACE-ON adaptive Gaussian map...")

face_sigma, face_deposited_mass = gaussian_surface_density(
    gas_rotated[face_keep, 0],
    gas_rotated[face_keep, 1],
    gas_mass[face_keep],
    gas_h[face_keep],
    xlim,
    ylim_face,
    args.pixels,
    args.pixels,
    hsml_fwhm_factor=args.hsml_fwhm_factor,
    kernel_sigma_extent=args.kernel_sigma_extent,
    minimum_sigma_pixels=args.minimum_sigma_pixels,
)


print("\nBuilding EDGE-ON adaptive Gaussian map...")

edge_sigma, edge_deposited_mass = gaussian_surface_density(
    gas_rotated[edge_keep, 0],
    gas_rotated[edge_keep, 2],
    gas_mass[edge_keep],
    gas_h[edge_keep],
    xlim,
    zlim_edge,
    args.pixels,
    ny_edge,
    hsml_fwhm_factor=args.hsml_fwhm_factor,
    kernel_sigma_extent=args.kernel_sigma_extent,
    minimum_sigma_pixels=args.minimum_sigma_pixels,
)


# ---------------------------------------------------------------------------
# Color scale
# ---------------------------------------------------------------------------

if args.vmin is None or args.vmax is None:

    combined_positive = np.concatenate([
        face_sigma[
            np.isfinite(face_sigma)
            & (face_sigma > 0)
        ],
        edge_sigma[
            np.isfinite(edge_sigma)
            & (edge_sigma > 0)
        ],
    ])

    auto_vmin = np.percentile(
        combined_positive,
        1.0
    )

    auto_vmax = np.percentile(
        combined_positive,
        99.7
    )

else:
    auto_vmin = None
    auto_vmax = None


vmin = (
    args.vmin
    if args.vmin is not None
    else auto_vmin
)

vmax = (
    args.vmax
    if args.vmax is not None
    else auto_vmax
)

if vmin <= 0:
    positive = combined_positive[
        combined_positive > 0
    ]
    vmin = positive.min()

if vmax <= vmin:
    raise RuntimeError(
        f"Invalid display limits: "
        f"vmin={vmin}, vmax={vmax}"
    )


print("\nDisplay")
print("-" * 78)
print(
    f"Gaussian FWHM = "
    f"{args.hsml_fwhm_factor:g} x SmoothingLength"
)
print(
    f"Gaussian extent = "
    f"{args.kernel_sigma_extent:g} sigma"
)
print(
    f"Sigma_gas display range = "
    f"{vmin:.6e} -- {vmax:.6e} Msun/pc^2"
)


norm = LogNorm(
    vmin=vmin,
    vmax=vmax
)


# ---------------------------------------------------------------------------
# Individual face-on plot
# ---------------------------------------------------------------------------

fig, axis = plt.subplots(
    figsize=(7.0, 6.2)
)

image = axis.imshow(
    face_sigma,
    origin="lower",
    extent=(
        *xlim,
        *ylim_face
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axis,
    xlim,
    ylim_face,
    "x (kpc)",
    "y (kpc)"
)

axis.set_title(
    "Halo 441 — Gas surface density, face-on"
)

colorbar = fig.colorbar(
    image,
    ax=axis
)

colorbar.set_label(
    r"$\Sigma_{\rm gas}$ ($M_\odot\,{\rm pc}^{-2}$)"
)

fig.tight_layout()

fig.savefig(
    args.prefix + "_faceon.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    args.prefix + "_faceon.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Individual edge-on plot
# ---------------------------------------------------------------------------

fig, axis = plt.subplots(
    figsize=(8.0, 4.6)
)

image = axis.imshow(
    edge_sigma,
    origin="lower",
    extent=(
        *xlim,
        *zlim_edge
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axis,
    xlim,
    zlim_edge,
    "x (kpc)",
    "z (kpc)"
)

axis.set_title(
    "Halo 441 — Gas surface density, edge-on"
)

colorbar = fig.colorbar(
    image,
    ax=axis
)

colorbar.set_label(
    r"$\Sigma_{\rm gas}$ ($M_\odot\,{\rm pc}^{-2}$)"
)

fig.tight_layout()

fig.savefig(
    args.prefix + "_edgeon.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    args.prefix + "_edgeon.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Combined face-on / edge-on plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.8),
    gridspec_kw={
        "width_ratios": [1.0, 1.0]
    }
)

face_image = axes[0].imshow(
    face_sigma,
    origin="lower",
    extent=(
        *xlim,
        *ylim_face
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

axes[1].imshow(
    edge_sigma,
    origin="lower",
    extent=(
        *xlim,
        *zlim_edge
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axes[0],
    xlim,
    ylim_face,
    "x (kpc)",
    "y (kpc)"
)

style_axis(
    axes[1],
    xlim,
    zlim_edge,
    "x (kpc)",
    "z (kpc)"
)

axes[0].set_title("Face-on")
axes[1].set_title("Edge-on")

fig.suptitle(
    "Halo 441 — Adaptively smoothed gas surface density",
    fontsize=15
)

colorbar = fig.colorbar(
    face_image,
    ax=axes,
    fraction=0.035,
    pad=0.04
)

colorbar.set_label(
    r"$\Sigma_{\rm gas}$ ($M_\odot\,{\rm pc}^{-2}$)"
)

fig.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.12,
    top=0.88,
    wspace=0.25
)

fig.savefig(
    args.prefix + "_faceon_edgeon.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    args.prefix + "_faceon_edgeon.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Save actual numerical maps
# ---------------------------------------------------------------------------

np.savez_compressed(
    args.prefix + "_maps.npz",

    faceon_sigma_gas_msun_pc2=face_sigma,
    edgeon_sigma_gas_msun_pc2=edge_sigma,

    xlim_kpc=np.asarray(xlim),
    faceon_ylim_kpc=np.asarray(ylim_face),
    edgeon_zlim_kpc=np.asarray(zlim_edge),

    faceon_zmax_kpc=args.faceon_zmax,
    edgeon_ymax_kpc=args.edgeon_ymax,

    hsml_fwhm_factor=args.hsml_fwhm_factor,
    kernel_sigma_extent=args.kernel_sigma_extent,
    minimum_sigma_pixels=args.minimum_sigma_pixels,

    j_hat=j_hat,
    rotation_matrix=rotation,

    smoothing_length_percentiles=np.asarray(percentiles),
    smoothing_length_values_kpc=h_values,

    faceon_deposited_mass_msun=face_deposited_mass,
    edgeon_deposited_mass_msun=edge_deposited_mass,

    scale_factor=scale_factor,
    redshift=redshift,
    hubble=hubble,
)


# ---------------------------------------------------------------------------
# Final diagnostics
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("FINISHED")
print("=" * 78)

print("j_hat =", j_hat)

print(
    f"Face-on selected gas mass = "
    f"{gas_mass[face_keep].sum():.6e} Msun"
)

print(
    f"Face-on mass represented inside map = "
    f"{face_deposited_mass:.6e} Msun"
)

print(
    f"Edge-on selected gas mass = "
    f"{gas_mass[edge_keep].sum():.6e} Msun"
)

print(
    f"Edge-on mass represented inside map = "
    f"{edge_deposited_mass:.6e} Msun"
)

print("\nSaved:")

for suffix in (
    "_faceon.png",
    "_faceon.pdf",
    "_edgeon.png",
    "_edgeon.pdf",
    "_faceon_edgeon.png",
    "_faceon_edgeon.pdf",
    "_maps.npz",
):
    print(" ", args.prefix + suffix)
