#!/usr/bin/env python3

"""
Halo 441 morphology tracer comparison.

Face-on panels:
  1. All stars: stellar mass surface density
  2. Young stars (<100 Myr): stellar mass surface density
  3. All gas: gas surface density
  4. Star formation: SFR surface density

Common center and orientation.

Visualization conventions:
  Stars:
      fixed Gaussian FWHM = --star-fwhm (default 0.2 kpc)

  Gas / SFR:
      adaptive Gaussian FWHM = --gas-h-factor * SmoothingLength
      default gas-h-factor = 0.25

This is a morphology visualization, not an exact SPH reconstruction.
"""

import argparse
import glob

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("snapshot")

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
    "--half-size",
    type=float,
    default=15.0,
)

parser.add_argument(
    "--zmax",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--pixels",
    type=int,
    default=600,
)

parser.add_argument(
    "--young-age-myr",
    type=float,
    default=100.0,
)

parser.add_argument(
    "--star-fwhm",
    type=float,
    default=0.20,
    help="Fixed stellar visualization FWHM in physical kpc.",
)

parser.add_argument(
    "--gas-h-factor",
    type=float,
    default=0.25,
    help="Gas Gaussian FWHM = factor * physical SmoothingLength.",
)

parser.add_argument(
    "--gaussian-extent",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--cmap",
    default="magma",
)

parser.add_argument(
    "--prefix",
    default="halo441_morphology_tracers",
)

args = parser.parse_args()


# ============================================================================
# Basic utilities
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


def load_particles(files, ptype, fields):

    out = {field: [] for field in fields}

    for filename in files:

        with h5py.File(filename, "r") as f:

            name = f"PartType{ptype}"

            if name not in f:
                continue

            group = f[name]
            n = len(group["Coordinates"])

            for field in fields:

                if field == "Masses" and field not in group:

                    m = f["Header"].attrs["MassTable"][ptype]

                    if m <= 0:
                        raise RuntimeError(
                            f"No masses for {name}"
                        )

                    out[field].append(
                        np.full(n, m, dtype=float)
                    )

                else:

                    if field not in group:
                        raise RuntimeError(
                            f"Missing {name}/{field}"
                        )

                    out[field].append(
                        group[field][:]
                    )

    return {
        k: np.concatenate(v)
        for k, v in out.items()
    }


def periodic_delta(position, center, box):

    delta = position.astype(float) - center

    delta -= box * np.rint(
        delta / box
    )

    return delta


# ============================================================================
# Rotation
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
# Cosmology: stellar ages
# ============================================================================

def cosmic_age_flat_lcdm(a, H0, omega_m, omega_l):

    """
    Cosmic age for flat matter + Lambda cosmology.

    Returns Gyr.

    t(a) =
      2 / (3 H0 sqrt(Omega_L))
      asinh[sqrt(Omega_L/Omega_m) a^(3/2)]
    """

    a = np.asarray(a, dtype=float)

    # Hubble time in Gyr:
    # 1/H0 = 9.778/h Gyr
    h = H0 / 100.0

    hubble_time_gyr = 9.778 / h

    prefactor = (
        2.0
        / (
            3.0
            * np.sqrt(omega_l)
        )
    )

    value = (
        np.sqrt(
            omega_l / omega_m
        )
        * a**1.5
    )

    return (
        hubble_time_gyr
        * prefactor
        * np.arcsinh(value)
    )


# ============================================================================
# Gaussian renderer
# ============================================================================

def render_gaussians(
    x,
    y,
    weight,
    fwhm,
    half_size,
    pixels,
    extent_sigma=4.0,
    label="map",
):

    """
    Render normalized 2-D Gaussians.

    weight may be:
      Msun        -> result Msun/kpc^2
      Msun/yr     -> result Msun/yr/kpc^2

    fwhm may be scalar or one value per particle.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    weight = np.asarray(weight)

    if np.ndim(fwhm) == 0:
        fwhm = np.full(
            len(x),
            float(fwhm),
        )
    else:
        fwhm = np.asarray(fwhm)

    xmin = -half_size
    xmax = +half_size

    ymin = -half_size
    ymax = +half_size

    dx = (xmax - xmin) / pixels
    dy = (ymax - ymin) / pixels

    xc = (
        xmin
        + (
            np.arange(pixels)
            + 0.5
        ) * dx
    )

    yc = (
        ymin
        + (
            np.arange(pixels)
            + 0.5
        ) * dy
    )

    image = np.zeros(
        (pixels, pixels),
        dtype=np.float64,
    )

    n = len(x)

    print(
        f"\nRendering {label}: "
        f"{n:,} particles"
    )

    report = max(
        1,
        n // 10,
    )

    for i in range(n):

        if i % report == 0:
            print(
                f"  {100*i/max(n,1):5.1f}%"
            )

        if (
            not np.isfinite(weight[i])
            or weight[i] <= 0
            or not np.isfinite(fwhm[i])
            or fwhm[i] <= 0
        ):
            continue

        sigma = (
            fwhm[i]
            * FWHM_TO_SIGMA
        )

        radius = (
            extent_sigma
            * sigma
        )

        xi = x[i]
        yi = y[i]

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
            pixels - 1,
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
            pixels - 1,
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

        kernel /= (
            2.0
            * np.pi
            * sigma**2
        )

        image[
            iy0:iy1 + 1,
            ix0:ix1 + 1
        ] += (
            weight[i]
            * kernel
        )

    print("  100.0%")

    return image


# ============================================================================
# Display utility
# ============================================================================

def log_limits(
    image,
    low=2.0,
    high=99.7,
):

    good = (
        np.isfinite(image)
        & (image > 0)
    )

    values = image[good]

    if values.size == 0:
        return 1.0, 10.0

    vmin = np.percentile(
        values,
        low,
    )

    vmax = np.percentile(
        values,
        high,
    )

    if vmax <= vmin:
        vmax = 10.0 * vmin

    return vmin, vmax


# ============================================================================
# Read snapshot
# ============================================================================

files = sorted(
    glob.glob(args.snapshot)
)

if not files:
    raise FileNotFoundError(
        args.snapshot
    )


with h5py.File(files[0], "r") as f:

    header = f["Header"]
    parameters = f.get("Parameters")

    a = scalar(
        header.attrs["Time"]
    )

    box = scalar(
        header.attrs["BoxSize"]
    )

    hubble = get_attribute(
        header,
        parameters,
        ("HubbleParam",),
    )


# CosmicGrain cosmology
omega_m = 0.3158
omega_l = 0.6842

H0 = 100.0 * hubble

z = 1.0 / a - 1.0

print("=" * 78)
print("HALO 441 MORPHOLOGY TRACER TEST")
print("=" * 78)

print(
    f"a={a:.8f}  "
    f"z={z:.8f}  "
    f"h={hubble:.6f}"
)


center = np.asarray(
    args.center,
    dtype=float,
)


# ============================================================================
# Stars
# ============================================================================

stars = load_particles(
    files,
    4,
    (
        "Coordinates",
        "Masses",
        "Velocities",
        "StellarFormationTime",
    ),
)


star_pos = (
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

star_vel = (
    stars["Velocities"].astype(float)
)

aform = (
    stars["StellarFormationTime"].astype(float)
)


# ============================================================================
# Orientation
# ============================================================================

r = np.linalg.norm(
    star_pos,
    axis=1,
)

orient = (
    r < args.orientation_radius
)

bulk_velocity = np.average(
    star_vel[orient],
    axis=0,
    weights=star_mass[orient],
)


J = np.sum(
    np.cross(
        star_pos[orient],
        star_vel[orient]
        - bulk_velocity,
    )
    * star_mass[orient, None],
    axis=0,
)


j_hat = (
    J
    / np.linalg.norm(J)
)

rotation = make_rotation_matrix(
    J
)


star_pos = (
    star_pos
    @ rotation.T
)


print(
    f"\nOrientation stars: "
    f"{orient.sum():,}"
)

print(
    "j_hat =",
    j_hat,
)


# ============================================================================
# Stellar ages
# ============================================================================

valid_birth = (
    np.isfinite(aform)
    & (aform > 0)
    & (aform <= a)
)


age_gyr = np.full(
    len(aform),
    np.nan,
)


t_now = cosmic_age_flat_lcdm(
    a,
    H0,
    omega_m,
    omega_l,
)


age_gyr[valid_birth] = (
    t_now
    - cosmic_age_flat_lcdm(
        aform[valid_birth],
        H0,
        omega_m,
        omega_l,
    )
)


print(
    f"Cosmic age at snapshot: "
    f"{t_now:.3f} Gyr"
)


# ============================================================================
# Gas
# ============================================================================

gas = load_particles(
    files,
    0,
    (
        "Coordinates",
        "Masses",
        "SmoothingLength",
        "StarFormationRate",
    ),
)


gas_pos = (
    periodic_delta(
        gas["Coordinates"],
        center,
        box,
    )
    * a
    / hubble
)


gas_pos = (
    gas_pos
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


gas_sfr = (
    gas["StarFormationRate"].astype(float)
)


# ============================================================================
# Face-on selections
# ============================================================================

sx = star_pos[:, 0]
sy = star_pos[:, 1]
sz = star_pos[:, 2]

gx = gas_pos[:, 0]
gy = gas_pos[:, 1]
gz = gas_pos[:, 2]


star_disk = (
    np.abs(sz) <= args.zmax
)


young = (
    star_disk
    & np.isfinite(age_gyr)
    & (
        age_gyr
        <= args.young_age_myr / 1000.0
    )
)


gas_disk = (
    np.abs(gz) <= args.zmax
)


sf_disk = (
    gas_disk
    & np.isfinite(gas_sfr)
    & (gas_sfr > 0)
)


print("\nSelections")
print("-" * 78)

print(
    f"Stars |z'|<{args.zmax:g}: "
    f"{star_disk.sum():,}"
)

print(
    f"Young stars "
    f"(<{args.young_age_myr:g} Myr): "
    f"{young.sum():,}"
)

print(
    f"Gas |z'|<{args.zmax:g}: "
    f"{gas_disk.sum():,}"
)

print(
    f"Star-forming gas: "
    f"{sf_disk.sum():,}"
)

print(
    f"SFR in selected slab: "
    f"{gas_sfr[sf_disk].sum():.6f} "
    f"Msun/yr"
)


# ============================================================================
# Render maps
# ============================================================================

star_map = render_gaussians(
    sx[star_disk],
    sy[star_disk],
    star_mass[star_disk],
    args.star_fwhm,
    args.half_size,
    args.pixels,
    args.gaussian_extent,
    "all stars",
)


young_map = render_gaussians(
    sx[young],
    sy[young],
    star_mass[young],
    args.star_fwhm,
    args.half_size,
    args.pixels,
    args.gaussian_extent,
    "young stars",
)


gas_fwhm = (
    args.gas_h_factor
    * gas_h[gas_disk]
)


gas_map = render_gaussians(
    gx[gas_disk],
    gy[gas_disk],
    gas_mass[gas_disk],
    gas_fwhm,
    args.half_size,
    args.pixels,
    args.gaussian_extent,
    "all gas",
)


sf_fwhm = (
    args.gas_h_factor
    * gas_h[sf_disk]
)


sfr_map = render_gaussians(
    gx[sf_disk],
    gy[sf_disk],
    gas_sfr[sf_disk],
    sf_fwhm,
    args.half_size,
    args.pixels,
    args.gaussian_extent,
    "star formation",
)


# stellar/gas mass maps:
# Msun/kpc^2 -> Msun/pc^2

star_map /= 1.0e6
young_map /= 1.0e6
gas_map /= 1.0e6

# sfr_map remains:
# Msun yr^-1 kpc^-2


# ============================================================================
# Plot
# ============================================================================

maps = [
    star_map,
    young_map,
    gas_map,
    sfr_map,
]


titles = [
    "All stars",
    f"Young stars (<{args.young_age_myr:g} Myr)",
    "All gas",
    "Star formation",
]


labels = [
    r"$\Sigma_\star$ ($M_\odot\,{\rm pc}^{-2}$)",
    r"$\Sigma_{\star,\rm young}$ ($M_\odot\,{\rm pc}^{-2}$)",
    r"$\Sigma_{\rm gas}$ ($M_\odot\,{\rm pc}^{-2}$)",
    r"$\Sigma_{\rm SFR}$ ($M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2}$)",
]


fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 11),
)


extent = (
    -args.half_size,
    args.half_size,
    -args.half_size,
    args.half_size,
)


for ax, image, title, label in zip(
    axes.flat,
    maps,
    titles,
    labels,
):

    vmin, vmax = log_limits(
        image
    )

    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap=args.cmap,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(
        title,
        fontsize=15,
    )

    ax.set_xlabel(
        "x (kpc)"
    )

    ax.set_ylabel(
        "y (kpc)"
    )

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cb.set_label(
        label
    )


fig.suptitle(
    "Halo 441 — Face-on morphology tracers",
    fontsize=18,
)


fig.tight_layout(
    rect=[0, 0, 1, 0.97]
)


fig.savefig(
    args.prefix + ".png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    args.prefix + ".pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================================
# Save data
# ============================================================================

np.savez_compressed(
    args.prefix + ".npz",

    star_map=star_map,
    young_star_map=young_map,
    gas_map=gas_map,
    sfr_map=sfr_map,

    half_size_kpc=args.half_size,
    zmax_kpc=args.zmax,

    young_age_myr=args.young_age_myr,
    star_fwhm_kpc=args.star_fwhm,
    gas_h_factor=args.gas_h_factor,

    j_hat=j_hat,
    rotation_matrix=rotation,

    young_star_count=young.sum(),
    sf_gas_count=sf_disk.sum(),
)


print("\nSaved:")
print(" ", args.prefix + ".png")
print(" ", args.prefix + ".pdf")
print(" ", args.prefix + ".npz")
