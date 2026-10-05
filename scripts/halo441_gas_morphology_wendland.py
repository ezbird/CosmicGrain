#!/usr/bin/env python3
"""
Reconstruct face-on and edge-on gas surface-density maps using the exact
3-D Wendland C4 SPH kernel implemented by GADGET-4.

GADGET-4 definition (src/sph/kernel.h):

    u = r / h
    W_3D(r,h) = [495 / (32*pi*h^3)]
                (1-u)^6 [1 + 6u + (35/3)u^2],  0 <= u <= 1
              = 0,                               u > 1

Thus PartType0/SmoothingLength is the compact-support radius h.

For an infinite line-of-sight projection,

    W_proj(R,h) = 1/h^2 * F(R/h),

where F(q) is obtained by numerically integrating the normalized 3-D
Wendland C4 kernel along the line of sight.

For finite face-on / edge-on slabs, the actual 3-D kernel is integrated
only over the part of the particle kernel intersecting the requested
line-of-sight slab. This means particles whose centers lie outside the
slab can still contribute when their SPH kernels overlap it.

Surface density is returned in Msun / pc^2.

The disk orientation is determined from stellar angular momentum inside
--orientation-radius, matching the existing Halo 441 morphology scripts.
"""

import argparse
import glob
import math

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# ============================================================================
# Snapshot helpers
# ============================================================================

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
        hubble = attribute(
            header,
            parameters,
            ("HubbleParam",),
        )

    return scale_factor, box, hubble


def load_particles(files, particle_type, fields):
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

                    mass = handle["Header"].attrs[
                        "MassTable"
                    ][particle_type]

                    if mass <= 0:
                        raise RuntimeError(
                            f"No Masses field or MassTable entry "
                            f"for PartType{particle_type}"
                        )

                    result[field].append(
                        np.full(
                            count,
                            mass,
                            dtype=float,
                        )
                    )

                else:

                    if field not in group:
                        raise RuntimeError(
                            f"{group_name}/{field} missing "
                            f"in {filename}"
                        )

                    result[field].append(
                        group[field][:]
                    )

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


# ============================================================================
# Orientation
# ============================================================================

def rotation_matrix(angular_momentum):
    zhat = (
        angular_momentum
        / np.linalg.norm(angular_momentum)
    )

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
# Exact GADGET-4 Wendland C4 kernel
# ============================================================================

WENDLAND_C4_NORM_3D = (
    495.0
    / (32.0 * np.pi)
)


def wendland_c4_dimensionless(u):
    """
    Dimensionless part of GADGET-4's 3-D Wendland C4 kernel.

    Returns

        w(u) = NORM * (1-u)^6 *
               [1 + 6u + (35/3)u^2]

    such that

        W(r,h) = w(r/h) / h^3.

    The kernel is exactly zero for u >= 1.
    """

    u = np.asarray(
        u,
        dtype=float,
    )

    result = np.zeros_like(
        u,
        dtype=float,
    )

    valid = (
        (u >= 0.0)
        & (u < 1.0)
    )

    uv = u[valid]

    t = 1.0 - uv

    result[valid] = (
        WENDLAND_C4_NORM_3D
        * t**6
        * (
            1.0
            + 6.0 * uv
            + (35.0 / 3.0) * uv**2
        )
    )

    return result


# ============================================================================
# Infinite projected-kernel lookup table
# ============================================================================

def make_projected_kernel_table(
    nq=10001,
    nz=2000,
):
    """
    Precompute the infinite-LOS projected dimensionless kernel

        F(q) = integral w(sqrt(q^2 + t^2)) dt

    with q = R/h.

    Then

        W_proj(R,h) = F(R/h) / h^2.

    This table is primarily useful as a validation reference. The finite-slab
    renderer below integrates the actual 3-D kernel over the requested slab.
    """

    q_values = np.linspace(
        0.0,
        1.0,
        nq,
    )

    projected = np.zeros_like(
        q_values
    )

    # Work in chunks so we do not allocate an enormous nq x nz array.
    chunk = 250

    for start in range(
        0,
        nq,
        chunk,
    ):

        stop = min(
            nq,
            start + chunk,
        )

        q = q_values[
            start:stop
        ]

        # Map a fixed integration variable s in [0,1] onto
        # t in [0, sqrt(1-q^2)] for every q.
        s = np.linspace(
            0.0,
            1.0,
            nz,
        )

        tmax = np.sqrt(
            np.maximum(
                0.0,
                1.0 - q**2,
            )
        )

        t = (
            tmax[:, None]
            * s[None, :]
        )

        radius = np.sqrt(
            q[:, None]**2
            + t**2
        )

        integrand = (
            wendland_c4_dimensionless(
                radius
            )
        )

        # Symmetric +/- LOS integral.
        projected[
            start:stop
        ] = (
            2.0
            * tmax
            * np.trapezoid(
                integrand,
                s,
                axis=1,
            )
        )

    return (
        q_values,
        projected,
    )


# ============================================================================
# Finite-slab projected Wendland renderer
# ============================================================================

def deposit_wendland_finite_slab(
    plane_x,
    plane_y,
    los_position,
    mass,
    hsml,
    xlim,
    ylim,
    nx,
    ny,
    slab_half_depth,
    los_samples=64,
    progress_label="Projection",
):
    """
    Project the exact 3-D GADGET-4 Wendland C4 kernel through a finite slab.

    The displayed plane coordinates are plane_x, plane_y.
    los_position is the particle coordinate along the line of sight.

    The slab is

        -slab_half_depth <= LOS <= +slab_half_depth.

    A particle contributes whenever its compact support intersects both
    the image and the LOS slab.

    The LOS integral is evaluated numerically at every affected map pixel.

    Returns surface density in Msun/pc^2 plus useful diagnostics.
    """

    xmin, xmax = xlim
    ymin, ymax = ylim

    dx = (
        xmax - xmin
    ) / nx

    dy = (
        ymax - ymin
    ) / ny

    pixel_area_kpc2 = (
        dx * dy
    )

    x_centers = (
        xmin
        + (
            np.arange(nx)
            + 0.5
        ) * dx
    )

    y_centers = (
        ymin
        + (
            np.arange(ny)
            + 0.5
        ) * dy
    )

    sigma_kpc2 = np.zeros(
        (ny, nx),
        dtype=np.float64,
    )

    # Gauss-Legendre quadrature on [-1,1].
    nodes, weights = (
        np.polynomial.legendre.leggauss(
            los_samples
        )
    )

    n_particles = len(mass)

    intersect_count = 0
    center_inside_slab_count = 0
    center_inside_map_count = 0

    center_inside_slab_mass = 0.0
    center_inside_volume_mass = 0.0

    report_every = max(
        1,
        n_particles // 20,
    )

    for i in range(n_particles):

        if i % report_every == 0:
            pct = (
                100.0
                * i
                / max(
                    n_particles,
                    1,
                )
            )

            print(
                f"  {progress_label}: "
                f"{i:,}/{n_particles:,} "
                f"({pct:5.1f}%)"
            )

        xi = plane_x[i]
        yi = plane_y[i]
        zi = los_position[i]
        hi = hsml[i]
        mi = mass[i]

        if (
            not np.isfinite(xi)
            or not np.isfinite(yi)
            or not np.isfinite(zi)
            or not np.isfinite(hi)
            or not np.isfinite(mi)
            or hi <= 0.0
            or mi <= 0.0
        ):
            continue

        center_in_slab = (
            abs(zi)
            <= slab_half_depth
        )

        center_in_map = (
            xmin <= xi <= xmax
            and ymin <= yi <= ymax
        )

        if center_in_slab:
            center_inside_slab_count += 1
            center_inside_slab_mass += mi

        if center_in_slab and center_in_map:
            center_inside_map_count += 1
            center_inside_volume_mass += mi

        # Does the kernel overlap the LOS slab at all?
        los_low = max(
            -slab_half_depth,
            zi - hi,
        )

        los_high = min(
            slab_half_depth,
            zi + hi,
        )

        if los_high <= los_low:
            continue

        # Does the kernel overlap the image?
        if (
            xi + hi < xmin
            or xi - hi > xmax
            or yi + hi < ymin
            or yi - hi > ymax
        ):
            continue

        intersect_count += 1

        ix0 = max(
            0,
            int(
                np.floor(
                    (
                        xi - hi - xmin
                    ) / dx
                )
            ),
        )

        ix1 = min(
            nx - 1,
            int(
                np.floor(
                    (
                        xi + hi - xmin
                    ) / dx
                )
            ),
        )

        iy0 = max(
            0,
            int(
                np.floor(
                    (
                        yi - hi - ymin
                    ) / dy
                )
            ),
        )

        iy1 = min(
            ny - 1,
            int(
                np.floor(
                    (
                        yi + hi - ymin
                    ) / dy
                )
            ),
        )

        if (
            ix1 < ix0
            or iy1 < iy0
        ):
            continue

        local_x = x_centers[
            ix0:ix1 + 1
        ]

        local_y = y_centers[
            iy0:iy1 + 1
        ]

        xx, yy = np.meshgrid(
            local_x,
            local_y,
        )

        projected_r2 = (
            (xx - xi)**2
            + (yy - yi)**2
        )

        # Pixels whose centers are already outside support can be skipped.
        support = (
            projected_r2
            < hi**2
        )

        if not np.any(support):
            continue

        # For each projected pixel, the kernel itself restricts LOS to
        #
        #   |l - zi| <= sqrt(h^2 - R^2).
        #
        # Intersect this with the requested finite slab.
        support_depth = np.zeros_like(
            projected_r2
        )

        support_depth[support] = np.sqrt(
            np.maximum(
                0.0,
                hi**2
                - projected_r2[support],
            )
        )

        local_low = np.maximum(
            -slab_half_depth,
            zi - support_depth,
        )

        local_high = np.minimum(
            slab_half_depth,
            zi + support_depth,
        )

        valid_pixel = (
            support
            & (
                local_high
                > local_low
            )
        )

        if not np.any(valid_pixel):
            continue

        # Flatten only the affected pixels so quadrature arrays remain modest.
        valid_indices = np.flatnonzero(
            valid_pixel.ravel()
        )

        r2_flat = (
            projected_r2.ravel()[
                valid_indices
            ]
        )

        low_flat = (
            local_low.ravel()[
                valid_indices
            ]
        )

        high_flat = (
            local_high.ravel()[
                valid_indices
            ]
        )

        midpoint = (
            0.5
            * (
                high_flat
                + low_flat
            )
        )

        half_width = (
            0.5
            * (
                high_flat
                - low_flat
            )
        )

        # Shape:
        #   Naffected_pixels x Nquadrature_nodes
        los = (
            midpoint[:, None]
            + half_width[:, None]
            * nodes[None, :]
        )

        r = np.sqrt(
            r2_flat[:, None]
            + (
                los
                - zi
            )**2
        )

        u = (
            r
            / hi
        )

        kernel_3d = (
            wendland_c4_dimensionless(
                u
            )
            / hi**3
        )

        # Integrate W_3D dl -> W_projected in kpc^-2.
        projected_kernel = (
            half_width
            * np.sum(
                kernel_3d
                * weights[None, :],
                axis=1,
            )
        )

        contribution = (
            mi
            * projected_kernel
        )

        target = sigma_kpc2[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ]

        target_flat = target.ravel()

        target_flat[
            valid_indices
        ] += contribution

        sigma_kpc2[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] = target_flat.reshape(
            target.shape
        )

    print(
        f"  {progress_label}: "
        f"{n_particles:,}/{n_particles:,} "
        f"(100.0%)"
    )

    represented_mass = (
        np.sum(
            sigma_kpc2
        )
        * pixel_area_kpc2
    )

    # Msun/kpc^2 -> Msun/pc^2.
    sigma_pc2 = (
        sigma_kpc2
        / 1.0e6
    )

    diagnostics = {
        "kernel_intersect_count":
            intersect_count,

        "center_inside_slab_count":
            center_inside_slab_count,

        "center_inside_map_and_slab_count":
            center_inside_map_count,

        "center_inside_slab_mass":
            center_inside_slab_mass,

        "center_inside_volume_mass":
            center_inside_volume_mass,

        "represented_mass":
            represented_mass,
    }

    return (
        sigma_pc2,
        diagnostics,
    )


# ============================================================================
# Plot helpers
# ============================================================================

def choose_display_limits(
    arrays,
    requested_vmin=None,
    requested_vmax=None,
):
    positive = np.concatenate(
        [
            array[
                np.isfinite(array)
                & (array > 0.0)
            ]
            for array in arrays
        ]
    )

    if positive.size == 0:
        raise RuntimeError(
            "No positive surface-density pixels"
        )

    if requested_vmin is None:
        # Do not let numerically tiny kernel-edge pixels dominate the
        # morphology stretch.
        vmin = np.percentile(
            positive,
            5.0,
        )

        # A modest physical floor for display only.
        vmin = max(
            vmin,
            1.0e-2,
        )

    else:
        vmin = requested_vmin

    if requested_vmax is None:
        vmax = np.percentile(
            positive,
            99.7,
        )

    else:
        vmax = requested_vmax

    if (
        vmin <= 0.0
        or vmax <= vmin
    ):
        raise RuntimeError(
            f"Invalid display limits: "
            f"vmin={vmin}, vmax={vmax}"
        )

    return vmin, vmax


def style_axis(
    axis,
    xlim,
    ylim,
    xlabel,
    ylabel,
):
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    axis.set_aspect("equal")


# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description=(
        "Projected GADGET-4 Wendland C4 gas morphology maps."
    )
)

parser.add_argument(
    "snapshot",
)

parser.add_argument(
    "--center",
    nargs=3,
    type=float,
    required=True,
    metavar=("X", "Y", "Z"),
)

parser.add_argument(
    "--orientation-radius",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--xy-half-size",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--edge-z-half-size",
    type=float,
    default=8.0,
)

parser.add_argument(
    "--faceon-zmax",
    type=float,
    default=5.0,
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
    "--los-samples",
    type=int,
    default=64,
    help=(
        "Gauss-Legendre nodes used for finite LOS integration "
        "of each affected projected pixel"
    ),
)

parser.add_argument(
    "--cmap",
    default="magma",
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
    "--prefix",
    default="halo441_gas_morphology_wendland",
)

args = parser.parse_args()


if args.pixels < 10:
    raise ValueError(
        "--pixels must be >= 10"
    )

if args.los_samples < 8:
    raise ValueError(
        "--los-samples should be >= 8"
    )


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
print("COSMICGRAIN GAS MORPHOLOGY — GADGET-4 WENDLAND C4")
print("=" * 78)

print(
    f"Snapshot pieces: {len(files)}"
)

scale_factor, box, hubble = (
    snapshot_info(
        files[0]
    )
)

redshift = (
    1.0
    / scale_factor
    - 1.0
)

print(
    f"a = {scale_factor:.8f}"
)

print(
    f"z = {redshift:.8f}"
)

print(
    f"h = {hubble:.8f}"
)

print(
    f"BoxSize = {box:.8f}"
)

print(
    "\nKernel: GADGET-4 Wendland C4"
)

print(
    "3-D normalization: 495/(32*pi)"
)

print(
    "Compact support: r/h < 1"
)

print(
    "Projection: finite-slab LOS integral of the 3-D kernel"
)


center = np.asarray(
    args.center,
    dtype=float,
)


# ============================================================================
# Stellar orientation
# ============================================================================

print(
    "\nLoading stars for disk orientation..."
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
        "PartType4 stars required"
    )


star_position = (
    periodic_delta(
        stars["Coordinates"],
        center,
        box,
    )
    * scale_factor
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

star_radius = np.linalg.norm(
    star_position,
    axis=1,
)

orient = (
    star_radius
    < args.orientation_radius
)

if orient.sum() < 3:
    raise RuntimeError(
        "Too few stars inside orientation radius"
    )

bulk_velocity = np.average(
    star_velocity[orient],
    axis=0,
    weights=star_mass[orient],
)

angular_momentum = np.sum(
    np.cross(
        star_position[orient],
        (
            star_velocity[orient]
            - bulk_velocity
        ),
    )
    * star_mass[orient, None],
    axis=0,
)

j_hat = (
    angular_momentum
    / np.linalg.norm(
        angular_momentum
    )
)

rotation = rotation_matrix(
    angular_momentum
)

print(
    f"Stars inside orientation radius: "
    f"{orient.sum():,}"
)

print(
    "j_hat =",
    j_hat,
)

print(
    "rotation matrix:"
)

print(
    rotation
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

if gas is None:
    raise RuntimeError(
        "PartType0 gas required"
    )


gas_position = (
    periodic_delta(
        gas["Coordinates"],
        center,
        box,
    )
    * scale_factor
    / hubble
)

gas_mass = (
    gas["Masses"].astype(float)
    * 1.0e10
    / hubble
)

gas_h = (
    gas["SmoothingLength"].astype(float)
    * scale_factor
    / hubble
)

gas_rotated = (
    gas_position
    @ rotation.T
)


finite = (
    np.all(
        np.isfinite(
            gas_rotated
        ),
        axis=1,
    )
    & np.isfinite(
        gas_mass
    )
    & np.isfinite(
        gas_h
    )
    & (gas_mass > 0.0)
    & (gas_h > 0.0)
)

print(
    f"Total gas particles: "
    f"{len(gas_mass):,}"
)

print(
    f"Finite positive gas particles: "
    f"{finite.sum():,}"
)


# ============================================================================
# Local h diagnostics
# ============================================================================

radius = np.linalg.norm(
    gas_rotated,
    axis=1,
)

local30 = (
    finite
    & (radius < 30.0)
)

if not np.any(local30):
    raise RuntimeError(
        "No gas within 30 kpc"
    )


h_local = gas_h[
    local30
]

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

values = np.percentile(
    h_local,
    percentiles,
)


print(
    "\nPartType0/SmoothingLength within 30 physical kpc"
)

print(
    "-" * 78
)

for p, value in zip(
    percentiles,
    values,
):

    print(
        f"p{p:>3}: "
        f"{value:12.6f} kpc"
    )


h_min = np.min(
    h_local
)

# Use a tight numerical tolerance because the floor values should be
# essentially identical.
floor_tolerance = max(
    1.0e-8,
    abs(h_min) * 1.0e-6,
)

at_floor = (
    np.abs(
        h_local
        - h_min
    )
    <= floor_tolerance
)

floor_fraction = (
    at_floor.sum()
    / len(h_local)
)


print(
    f"\nMinimum local h = "
    f"{h_min:.6f} kpc"
)

print(
    f"Gas within 30 kpc at minimum h: "
    f"{at_floor.sum():,}/{len(h_local):,} "
    f"({100.0 * floor_fraction:.2f}%)"
)


# ============================================================================
# Geometry
# ============================================================================

xlim = (
    -args.xy_half_size,
    args.xy_half_size,
)

face_ylim = xlim

edge_zlim = (
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
    "\nMap geometry"
)

print(
    "-" * 78
)

print(
    f"Face-on: x,y = +/-"
    f"{args.xy_half_size:g} kpc; "
    f"LOS slab |z'| <= "
    f"{args.faceon_zmax:g} kpc"
)

print(
    f"Edge-on: x = +/-"
    f"{args.xy_half_size:g} kpc; "
    f"z = +/-"
    f"{args.edge_z_half_size:g} kpc; "
    f"LOS slab |y'| <= "
    f"{args.edgeon_ymax:g} kpc"
)

print(
    f"Face-on grid: "
    f"{args.pixels} x "
    f"{args.pixels}"
)

print(
    f"Edge-on grid: "
    f"{args.pixels} x "
    f"{ny_edge}"
)

print(
    f"Horizontal pixel size: "
    f"{2.0 * args.xy_half_size / args.pixels:.4f} kpc"
)


# ============================================================================
# Important performance preselection
# ============================================================================

# There is no need to pass all ~924,000 gas particles to the expensive
# renderer. A particle can only contribute if its compact support intersects
# the image AND the finite LOS slab.

def possible_intersection(
    px,
    py,
    plos,
    hsml,
    xlim,
    ylim,
    slab,
):
    return (
        finite
        & (px + hsml >= xlim[0])
        & (px - hsml <= xlim[1])
        & (py + hsml >= ylim[0])
        & (py - hsml <= ylim[1])
        & (plos + hsml >= -slab)
        & (plos - hsml <= slab)
    )


face_candidates = possible_intersection(
    gas_rotated[:, 0],
    gas_rotated[:, 1],
    gas_rotated[:, 2],
    gas_h,
    xlim,
    face_ylim,
    args.faceon_zmax,
)

edge_candidates = possible_intersection(
    gas_rotated[:, 0],
    gas_rotated[:, 2],
    gas_rotated[:, 1],
    gas_h,
    xlim,
    edge_zlim,
    args.edgeon_ymax,
)


print(
    f"\nFace-on candidate kernels: "
    f"{face_candidates.sum():,}"
)

print(
    f"Edge-on candidate kernels: "
    f"{edge_candidates.sum():,}"
)


# ============================================================================
# Render
# ============================================================================

print(
    "\nBuilding FACE-ON Wendland C4 projection..."
)

face_sigma, face_diag = (
    deposit_wendland_finite_slab(
        gas_rotated[
            face_candidates,
            0,
        ],
        gas_rotated[
            face_candidates,
            1,
        ],
        gas_rotated[
            face_candidates,
            2,
        ],
        gas_mass[
            face_candidates
        ],
        gas_h[
            face_candidates
        ],
        xlim,
        face_ylim,
        args.pixels,
        args.pixels,
        args.faceon_zmax,
        los_samples=args.los_samples,
        progress_label="Face-on",
    )
)


print(
    "\nBuilding EDGE-ON Wendland C4 projection..."
)

edge_sigma, edge_diag = (
    deposit_wendland_finite_slab(
        gas_rotated[
            edge_candidates,
            0,
        ],
        gas_rotated[
            edge_candidates,
            2,
        ],
        gas_rotated[
            edge_candidates,
            1,
        ],
        gas_mass[
            edge_candidates
        ],
        gas_h[
            edge_candidates
        ],
        xlim,
        edge_zlim,
        args.pixels,
        ny_edge,
        args.edgeon_ymax,
        los_samples=args.los_samples,
        progress_label="Edge-on",
    )
)


# ============================================================================
# Display scale
# ============================================================================

vmin, vmax = choose_display_limits(
    (
        face_sigma,
        edge_sigma,
    ),
    requested_vmin=args.vmin,
    requested_vmax=args.vmax,
)


print(
    "\nDisplay"
)

print(
    "-" * 78
)

print(
    f"Sigma_gas range: "
    f"{vmin:.6e} -- "
    f"{vmax:.6e} "
    f"Msun/pc^2"
)


norm = LogNorm(
    vmin=vmin,
    vmax=vmax,
)


# ============================================================================
# Individual face-on
# ============================================================================

fig, axis = plt.subplots(
    figsize=(7.0, 6.2)
)

image = axis.imshow(
    face_sigma,
    origin="lower",
    extent=(
        *xlim,
        *face_ylim,
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axis,
    xlim,
    face_ylim,
    "x (kpc)",
    "y (kpc)",
)

axis.set_title(
    "Halo 441 — Gas surface density, face-on"
)

cb = fig.colorbar(
    image,
    ax=axis,
)

cb.set_label(
    r"$\Sigma_{\rm gas}$ "
    r"($M_\odot\,{\rm pc}^{-2}$)"
)

fig.tight_layout()

fig.savefig(
    args.prefix
    + "_faceon.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    args.prefix
    + "_faceon.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================================
# Individual edge-on
# ============================================================================

fig, axis = plt.subplots(
    figsize=(8.0, 4.6)
)

image = axis.imshow(
    edge_sigma,
    origin="lower",
    extent=(
        *xlim,
        *edge_zlim,
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axis,
    xlim,
    edge_zlim,
    "x (kpc)",
    "z (kpc)",
)

axis.set_title(
    "Halo 441 — Gas surface density, edge-on"
)

cb = fig.colorbar(
    image,
    ax=axis,
)

cb.set_label(
    r"$\Sigma_{\rm gas}$ "
    r"($M_\odot\,{\rm pc}^{-2}$)"
)

fig.tight_layout()

fig.savefig(
    args.prefix
    + "_edgeon.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    args.prefix
    + "_edgeon.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================================
# Combined
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.8),
)

image = axes[0].imshow(
    face_sigma,
    origin="lower",
    extent=(
        *xlim,
        *face_ylim,
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
        *edge_zlim,
    ),
    cmap=args.cmap,
    norm=norm,
    interpolation="nearest",
)

style_axis(
    axes[0],
    xlim,
    face_ylim,
    "x (kpc)",
    "y (kpc)",
)

style_axis(
    axes[1],
    xlim,
    edge_zlim,
    "x (kpc)",
    "z (kpc)",
)

axes[0].set_title(
    "Face-on"
)

axes[1].set_title(
    "Edge-on"
)

fig.suptitle(
    "Halo 441 — Wendland C4 gas surface density",
    fontsize=15,
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
    top=0.88,
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
# Save maps
# ============================================================================

np.savez_compressed(
    args.prefix
    + "_maps.npz",

    faceon_sigma_gas_msun_pc2=
        face_sigma,

    edgeon_sigma_gas_msun_pc2=
        edge_sigma,

    xlim_kpc=
        np.asarray(xlim),

    faceon_ylim_kpc=
        np.asarray(face_ylim),

    edgeon_zlim_kpc=
        np.asarray(edge_zlim),

    faceon_zmax_kpc=
        args.faceon_zmax,

    edgeon_ymax_kpc=
        args.edgeon_ymax,

    j_hat=
        j_hat,

    rotation_matrix=
        rotation,

    local_h_percentiles=
        np.asarray(percentiles),

    local_h_values_kpc=
        values,

    local_h_min_kpc=
        h_min,

    local_h_floor_fraction=
        floor_fraction,

    faceon_candidate_kernels=
        face_candidates.sum(),

    edgeon_candidate_kernels=
        edge_candidates.sum(),

    faceon_represented_mass_msun=
        face_diag["represented_mass"],

    edgeon_represented_mass_msun=
        edge_diag["represented_mass"],

    scale_factor=
        scale_factor,

    redshift=
        redshift,

    hubble=
        hubble,

    kernel_name=
        "GADGET-4 Wendland C4",

    kernel_support_definition=
        "u=r/h; compact support u<1",
)


# ============================================================================
# Final diagnostics
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
    "j_hat =",
    j_hat,
)

print(
    "\nFACE-ON"
)

print(
    f"Kernel candidates: "
    f"{face_candidates.sum():,}"
)

print(
    f"Kernel intersections: "
    f"{face_diag['kernel_intersect_count']:,}"
)

print(
    f"Particle centers inside slab: "
    f"{face_diag['center_inside_slab_count']:,}"
)

print(
    f"Particle centers inside displayed volume: "
    f"{face_diag['center_inside_map_and_slab_count']:,}"
)

print(
    f"Mass with centers inside displayed volume: "
    f"{face_diag['center_inside_volume_mass']:.6e} Msun"
)

print(
    f"SPH mass represented in displayed map: "
    f"{face_diag['represented_mass']:.6e} Msun"
)


print(
    "\nEDGE-ON"
)

print(
    f"Kernel candidates: "
    f"{edge_candidates.sum():,}"
)

print(
    f"Kernel intersections: "
    f"{edge_diag['kernel_intersect_count']:,}"
)

print(
    f"Particle centers inside slab: "
    f"{edge_diag['center_inside_slab_count']:,}"
)

print(
    f"Particle centers inside displayed volume: "
    f"{edge_diag['center_inside_map_and_slab_count']:,}"
)

print(
    f"Mass with centers inside displayed volume: "
    f"{edge_diag['center_inside_volume_mass']:.6e} Msun"
)

print(
    f"SPH mass represented in displayed map: "
    f"{edge_diag['represented_mass']:.6e} Msun"
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
