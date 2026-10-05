#!/usr/bin/env python3

"""
halo441_rv_grain_physics_beta.py

Halo 441 face-on R_V heatmap diagnostic for modern CosmicGrain.

This script:
  * reads CosmicGrain PartType6 dust particles from split GADGET-4 HDF5 snapshots
  * reads the refined galaxy center and face-on rotation matrix
  * reads Draine graphite and silicate Q_ext tables
  * computes per-particle B- and V-band extinction weights
  * projects the dust face-on
  * builds an R_V heatmap on a user-defined pixel grid
  * builds a raw dust-superparticle count map
  * builds a dust-mass map
  * produces R_V maps masked at N_dust >= 1, 3, 5, and 10
  * prints sampling statistics for each particle-count threshold
  * saves all core map arrays to a compressed NPZ file

No other Python scripts are imported or required.

Example for 500 pc pixels across +/-20 pkpc:

python3 halo569_rv_heatmap.py \
    '../S10_output_1024/snapdir_047/snapshot_047.*.hdf5' \
    --center-file halo569_refined_center.txt \
    --rotation-file halo569_rotation_matrix.txt \
    --carbon-table dust_optics/callqcomp.out_D16MGemt \
    --silicate-table dust_optics/callqcomp.out_DH16asil \
    --plot-radius 20 \
    --zmax 20 \
    --bins 80 \
    --smooth 0 \
    --prefix halo569_faceon_500pc
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Constants
# ============================================================

RHO_C = 2.24
RHO_SIL = 3.50

B_MICRON = 0.44
V_MICRON = 0.55

# SI constants for synthetic modified-blackbody photometry.
PLANCK_H = 6.62607015e-34
BOLTZMANN_K = 1.380649e-23
LIGHT_C = 299792458.0


# ============================================================
# Draine optical tables
# ============================================================

_float_re = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)


def numbers_in_line(line):
    vals = []
    for x in _float_re.findall(line):
        try:
            vals.append(float(x.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return vals


def read_draine_qext_table(filename):
    blocks = {}
    current_radius = None
    wave = []
    qext = []

    def finish_block():
        nonlocal wave, qext, current_radius

        if current_radius is not None and len(wave) >= 10:
            w = np.asarray(wave, dtype=float)
            q = np.asarray(qext, dtype=float)

            good = (
                np.isfinite(w)
                & np.isfinite(q)
                & (w > 0)
                & (q >= 0)
            )

            w = w[good]
            q = q[good]

            if len(w) > 0:
                order = np.argsort(w)
                blocks[current_radius] = (
                    w[order],
                    q[order]
                )

        wave = []
        qext = []

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            low = line.lower()

            if (
                ("radius" in low or "rad" in low)
                and "micron" in low
            ):
                nums = numbers_in_line(line)

                if nums:
                    finish_block()
                    current_radius = nums[0]

                continue

            if current_radius is None:
                continue

            vals = numbers_in_line(line)

            if len(vals) >= 4:
                lam = vals[0]
                qe = vals[1]

                if lam > 0 and qe >= 0:
                    wave.append(lam)
                    qext.append(qe)

    finish_block()

    if len(blocks) == 0:
        raise RuntimeError(
            f"Could not parse any radius blocks from:\n{filename}"
        )

    radii = np.array(sorted(blocks.keys()))

    print()
    print(f"Loaded optical table: {filename}")
    print(f"  radius blocks = {len(radii)}")
    print(
        f"  radius range  = "
        f"{radii.min():.6g} - {radii.max():.6g} micron"
    )

    return blocks


def read_draine_qabs_table(filename):
    """Read the Q_abs (third numeric column) from a Draine callqcomp table."""
    blocks = {}
    current_radius = None
    wave, qabs = [], []

    def finish_block():
        nonlocal wave, qabs, current_radius
        if current_radius is not None and len(wave) >= 10:
            w = np.asarray(wave, dtype=float)
            q = np.asarray(qabs, dtype=float)
            good = np.isfinite(w) & np.isfinite(q) & (w > 0) & (q >= 0)
            w, q = w[good], q[good]
            if len(w):
                order = np.argsort(w)
                blocks[current_radius] = (w[order], q[order])
        wave, qabs = [], []

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if (("radius" in low or "rad" in low) and "micron" in low):
                nums = numbers_in_line(line)
                if nums:
                    finish_block()
                    current_radius = nums[0]
                continue
            if current_radius is None:
                continue
            vals = numbers_in_line(line)
            # callqcomp columns: wavelength, Q_ext, Q_abs, Q_sca, ...
            if len(vals) >= 4:
                lam, qa = vals[0], vals[2]
                if lam > 0 and qa >= 0:
                    wave.append(lam)
                    qabs.append(qa)
    finish_block()
    if not blocks:
        raise RuntimeError(f"Could not parse Q_abs radius blocks from: {filename}")
    radii = np.array(sorted(blocks.keys()))
    print(f"Loaded Q_abs table: {filename}")
    print(f"  radius blocks = {len(radii)}")
    print(f"  radius range  = {radii.min():.6g} - {radii.max():.6g} micron")
    return blocks


def qext_at_wavelength(
    blocks,
    particle_radius_um,
    wavelength_um
):
    table_radii = np.array(
        sorted(blocks.keys()),
        dtype=float
    )

    q_at_lambda = np.empty(
        len(table_radii),
        dtype=float
    )

    for i, a in enumerate(table_radii):
        wave, q = blocks[a]

        if wavelength_um < wave.min() or wavelength_um > wave.max():
            raise RuntimeError(
                f"Wavelength {wavelength_um} micron "
                f"outside optical table range."
            )

        q_at_lambda[i] = np.interp(
            np.log(wavelength_um),
            np.log(wave),
            q
        )

    pr = np.asarray(
        particle_radius_um,
        dtype=float
    )

    if np.any(pr <= 0):
        raise RuntimeError(
            "Found non-positive grain radius."
        )

    if (
        pr.min() < table_radii.min()
        or pr.max() > table_radii.max()
    ):
        raise RuntimeError(
            "Particle grain radius falls outside optical "
            "table radius range.\n"
            f"Particle range: "
            f"{pr.min():.6g} - {pr.max():.6g} micron\n"
            f"Table range: "
            f"{table_radii.min():.6g} - "
            f"{table_radii.max():.6g} micron"
        )

    return np.interp(
        np.log(pr),
        np.log(table_radii),
        q_at_lambda
    )


# ============================================================
# Snapshot
# ============================================================

def expand_files(pattern):
    files = sorted(glob.glob(pattern))

    if not files and os.path.exists(pattern):
        files = [pattern]

    if not files:
        raise FileNotFoundError(
            f"No files matched:\n{pattern}"
        )

    return files


def read_dust_snapshot(pattern):
    files = expand_files(pattern)

    coords = []
    mass = []
    radius_nm = []
    carbon = []
    temperature_k = []

    a = None
    h = None
    box = None

    print()
    print("Reading PartType6")
    print("-----------------")

    for fn in files:
        print(f"  {fn}")

        with h5py.File(fn, "r") as f:
            if a is None:
                a = float(
                    f["Header"].attrs["Time"]
                )

                box = float(
                    f["Header"].attrs["BoxSize"]
                )

                h = float(
                    f["Parameters"].attrs["HubbleParam"]
                )

            if "PartType6" not in f:
                continue

            d = f["PartType6"]

            required = [
                "Coordinates", "Masses", "GrainRadius", "DustTemperature"
            ]
            for field in required:
                if field not in d:
                    raise KeyError(f"{fn}: PartType6/{field} missing")

            if "CarbonMassFraction" in d:
                carbon_field = "CarbonMassFraction"
            elif "CarbonFraction" in d:
                carbon_field = "CarbonFraction"
            else:
                raise KeyError(
                    f"{fn}: neither PartType6/CarbonMassFraction nor "
                    "PartType6/CarbonFraction exists"
                )

            coords.append(
                np.asarray(
                    d["Coordinates"][:],
                    dtype=np.float64
                )
            )

            mass.append(
                np.asarray(
                    d["Masses"][:],
                    dtype=np.float64
                )
            )

            radius_nm.append(
                np.asarray(
                    d["GrainRadius"][:],
                    dtype=np.float64
                )
            )

            carbon.append(
                np.asarray(
                    d[carbon_field][:],
                    dtype=np.float64
                )
            )

            temperature_k.append(
                np.asarray(
                    d["DustTemperature"][:],
                    dtype=np.float64
                )
            )

    if len(mass) == 0:
        raise RuntimeError(
            "No PartType6 particles found."
        )

    coords = np.concatenate(coords)
    mass = np.concatenate(mass)
    radius_nm = np.concatenate(radius_nm)
    carbon = np.concatenate(carbon)
    temperature_k = np.concatenate(temperature_k)

    print()
    print(f"Dust particles = {len(mass):,}")
    print(
        f"radius range   = "
        f"{radius_nm.min():.4f} - "
        f"{radius_nm.max():.4f} nm"
    )
    print(
        f"CF range       = "
        f"{carbon.min():.4f} - "
        f"{carbon.max():.4f}"
    )
    print(
        f"dust T range   = "
        f"{temperature_k.min():.4f} - "
        f"{temperature_k.max():.4f} K"
    )

    return {
        "coords": coords,
        "mass": mass,
        "radius_nm": radius_nm,
        "carbon": carbon,
        "temperature_k": temperature_k,
        "a": a,
        "h": h,
        "box": box
    }



def read_starforming_gas_snapshot(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files match: {pattern}")

    coords_all = []
    sfr_all = []
    a = h = box = None

    print()
    print("Reading star-forming gas")
    print("------------------------")

    for fn in files:
        with h5py.File(fn, "r") as f:
            if a is None:
                hdr = f["Header"].attrs
                a = float(hdr.get("Time", 1.0))
                box = float(hdr["BoxSize"])
                h = float(hdr.get("HubbleParam",
                                  f.get("Parameters", {}).attrs.get("HubbleParam", 1.0)
                                  if "Parameters" in f else 1.0))

            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            if "Coordinates" not in g or "StarFormationRate" not in g:
                continue

            sfr = np.asarray(g["StarFormationRate"][:], dtype=float)
            sf = np.isfinite(sfr) & (sfr > 0.0)
            if np.any(sf):
                coords_all.append(np.asarray(g["Coordinates"][:], dtype=float)[sf])
                sfr_all.append(sfr[sf])
            print(f"  {fn}: SF gas = {np.count_nonzero(sf):,}")

    if not coords_all:
        return {"coords": np.empty((0,3)), "sfr": np.empty(0),
                "a": a, "h": h, "box": box}

    return {"coords": np.concatenate(coords_all),
            "sfr": np.concatenate(sfr_all),
            "a": a, "h": h, "box": box}



def cosmic_time_gyr(a, h, omega_m, omega_l):
    """Flat matter+Lambda cosmic age; sufficient for z=0 stellar ages."""
    a = np.asarray(a, dtype=float)
    H0_inv_gyr = 9.778 / h
    if omega_l > 0 and omega_m > 0:
        return (2.0 / (3.0 * np.sqrt(omega_l))) * H0_inv_gyr * np.arcsinh(
            np.sqrt(omega_l / omega_m) * np.power(a, 1.5)
        )
    return (2.0 / 3.0) * H0_inv_gyr * np.power(a, 1.5)


def read_young_stars_snapshot(pattern, max_age_myr=100.0):
    files = sorted(glob.glob(pattern))
    coords_all, age_all = [], []
    a_now = h = box = omega_m = omega_l = None

    print()
    print("Reading young stars")
    print("-------------------")

    for fn in files:
        with h5py.File(fn, "r") as f:
            hdr = f["Header"].attrs
            if a_now is None:
                a_now = float(hdr.get("Time", 1.0))
                box = float(hdr["BoxSize"])
                if "Parameters" in f:
                    par = f["Parameters"].attrs
                    h = float(par.get("HubbleParam", hdr.get("HubbleParam", 1.0)))
                    omega_m = float(par.get("Omega0", 0.3158))
                    omega_l = float(par.get("OmegaLambda", 0.6842))
                else:
                    h = float(hdr.get("HubbleParam", 1.0))
                    omega_m, omega_l = 0.3158, 0.6842

            if "PartType4" not in f:
                continue
            st = f["PartType4"]
            form_name = None
            for name in ("StellarFormationTime", "GFM_StellarFormationTime"):
                if name in st:
                    form_name = name
                    break
            if form_name is None or "Coordinates" not in st:
                continue

            af = np.asarray(st[form_name][:], dtype=float)
            valid = np.isfinite(af) & (af > 0.0) & (af <= a_now)
            age_myr = np.full_like(af, np.nan, dtype=float)
            age_myr[valid] = 1000.0 * (
                cosmic_time_gyr(a_now, h, omega_m, omega_l)
                - cosmic_time_gyr(af[valid], h, omega_m, omega_l)
            )
            young = valid & (age_myr <= max_age_myr)
            if np.any(young):
                coords_all.append(np.asarray(st["Coordinates"][:], dtype=float)[young])
                age_all.append(age_myr[young])
            print(f"  {fn}: age < {max_age_myr:g} Myr = {np.count_nonzero(young):,}")

    if not coords_all:
        return {"coords": np.empty((0,3)), "age_myr": np.empty(0),
                "a": a_now, "h": h, "box": box}

    return {"coords": np.concatenate(coords_all),
            "age_myr": np.concatenate(age_all),
            "a": a_now, "h": h, "box": box}


def snap_xy_to_pixel_centers(xy, plot_radius, bins):
    """Return unique occupied heatmap pixel centers for particle positions."""
    if len(xy) == 0:
        return np.empty((0, 2))
    edges = np.linspace(-plot_radius, plot_radius, bins + 1)
    ix = np.searchsorted(edges, xy[:, 0], side="right") - 1
    iy = np.searchsorted(edges, xy[:, 1], side="right") - 1
    good = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    pix = np.unique(np.column_stack((ix[good], iy[good])), axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.column_stack((centers[pix[:,0]], centers[pix[:,1]]))


# ============================================================
# Geometry
# ============================================================

def periodic_delta(coords, center, box):
    dr = coords - center[None, :]
    dr -= box * np.round(dr / box)
    return dr


def rotate_positions(
    coords,
    center,
    box,
    a,
    h,
    rotation
):
    dr_code = periodic_delta(
        coords,
        center,
        box
    )

    dr_pkpc = dr_code * a / h

    return dr_pkpc @ rotation.T


# ============================================================
# Extinction
# ============================================================

def extinction_weights(
    mass,
    radius_nm,
    carbon_fraction,
    carbon_blocks,
    silicate_blocks
):
    radius_um = radius_nm * 1.0e-3

    fC = np.clip(
        carbon_fraction,
        0.0,
        1.0
    )

    qCB = qext_at_wavelength(
        carbon_blocks,
        radius_um,
        B_MICRON
    )

    qCV = qext_at_wavelength(
        carbon_blocks,
        radius_um,
        V_MICRON
    )

    qSB = qext_at_wavelength(
        silicate_blocks,
        radius_um,
        B_MICRON
    )

    qSV = qext_at_wavelength(
        silicate_blocks,
        radius_um,
        V_MICRON
    )

    prefactor = mass / radius_um

    WB = prefactor * (
        fC / RHO_C * qCB
        + (1.0 - fC) / RHO_SIL * qSB
    )

    WV = prefactor * (
        fC / RHO_C * qCV
        + (1.0 - fC) / RHO_SIL * qSV
    )

    return WB, WV


# ============================================================
# Map construction
# ============================================================

def histogram_map(
    x,
    y,
    weights,
    radius,
    bins
):
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[
            [-radius, radius],
            [-radius, radius]
        ],
        weights=weights
    )

    return H.T, xedges, yedges


def histogram_count_map(
    x,
    y,
    radius,
    bins
):
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[
            [-radius, radius],
            [-radius, radius]
        ]
    )

    return H.T, xedges, yedges


def build_maps(
    x,
    y,
    mass,
    WB,
    WV,
    plot_radius,
    bins,
    smooth_pkpc,
    min_signal_fraction,
    min_color_excess_fraction
):
    tauB_raw, xedges, yedges = histogram_map(
        x,
        y,
        WB,
        plot_radius,
        bins
    )

    tauV_raw, _, _ = histogram_map(
        x,
        y,
        WV,
        plot_radius,
        bins
    )

    mass_raw, _, _ = histogram_map(
        x,
        y,
        mass,
        plot_radius,
        bins
    )

    count_raw, _, _ = histogram_count_map(
        x,
        y,
        plot_radius,
        bins
    )

    pixel_size = (
        2.0 * plot_radius / bins
    )

    sigma_pix = (
        smooth_pkpc / pixel_size
        if smooth_pkpc > 0
        else 0.0
    )

    print()
    print("Map sampling")
    print("------------")
    print(f"grid             = {bins} x {bins}")
    print(f"pixel size       = {pixel_size:.4f} pkpc")
    print(f"smoothing sigma  = {smooth_pkpc:.4f} pkpc")
    print(f"smoothing pixels = {sigma_pix:.3f}")

    if sigma_pix > 0:
        tauB = gaussian_filter(
            tauB_raw,
            sigma=sigma_pix,
            mode="constant"
        )

        tauV = gaussian_filter(
            tauV_raw,
            sigma=sigma_pix,
            mode="constant"
        )

        massmap = gaussian_filter(
            mass_raw,
            sigma=sigma_pix,
            mode="constant"
        )
    else:
        tauB = tauB_raw.copy()
        tauV = tauV_raw.copy()
        massmap = mass_raw.copy()

    denom = tauB - tauV

    rv = np.full_like(
        tauV,
        np.nan
    )

    if np.nanmax(tauV) > 0:
        signal_cut = (
            min_signal_fraction
            * np.nanmax(tauV)
        )
    else:
        signal_cut = np.inf

    positive_denom = np.isfinite(denom) & (denom > 0)
    if np.any(positive_denom):
        color_excess_cut = (
            min_color_excess_fraction
            * np.nanmax(denom[positive_denom])
        )
    else:
        color_excess_cut = np.inf

    print(f"tauV signal cut   = {signal_cut:.6e}")
    print(f"E(B-V) proxy cut  = {color_excess_cut:.6e}")

    good = (
        (tauV > signal_cut)
        & (denom > color_excess_cut)
    )

    rv[good] = (
        tauV[good] / denom[good]
    )

    extent = [
        -plot_radius,
        plot_radius,
        -plot_radius,
        plot_radius
    ]

    return {
        "rv": rv,
        "tauB": tauB,
        "tauV": tauV,
        "color_excess": denom,
        "tauB_raw": tauB_raw,
        "tauV_raw": tauV_raw,
        "mass": massmap,
        "mass_raw": mass_raw,
        "count_raw": count_raw,
        "pixel_size": pixel_size,
        "tauV_signal_cut": signal_cut,
        "color_excess_cut": color_excess_cut,
        "extent": extent,
        "xedges": xedges,
        "yedges": yedges
    }


# ============================================================
# Plotting helpers
# ============================================================

def plot_map(
    image,
    extent,
    title,
    cbar_label,
    filename,
    vmin=None,
    vmax=None,
    sf_xy=None,
    young_xy=None
):
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    if sf_xy is not None and len(sf_xy) > 0:
        ax.scatter(
            sf_xy[:, 0], sf_xy[:, 1],
            s=18, facecolors="white", edgecolors="none",
            marker="o", alpha=0.95,
            label=r"SFR $>0$ gas", zorder=6
        )
    if young_xy is not None and len(young_xy) > 0:
        ax.scatter(
            young_xy[:, 0], young_xy[:, 1],
            s=42, facecolors="none", edgecolors="white",
            linewidths=1.3, marker="o", alpha=0.95,
            label="stars < 100 Myr", zorder=7
        )
    if ((sf_xy is not None and len(sf_xy) > 0) or
        (young_xy is not None and len(young_xy) > 0)):
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax.set_xlabel("x' [pkpc]")
    ax.set_ylabel("y' [pkpc]")
    ax.set_title(title)

    cb = fig.colorbar(
        im,
        ax=ax
    )

    cb.set_label(
        cbar_label
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved {filename}")


def plot_count_thresholds(
    rv_map,
    count_map,
    extent,
    thresholds,
    filename
):
    finite = np.isfinite(rv_map)

    if np.any(finite):
        vmin = np.percentile(
            rv_map[finite],
            2
        )

        vmax = np.percentile(
            rv_map[finite],
            98
        )
    else:
        vmin = None
        vmax = None

    fig, axes = plt.subplots(
        1,
        len(thresholds),
        figsize=(18, 4.8),
        sharex=True,
        sharey=True
    )

    last_im = None

    for ax, nmin in zip(
        axes,
        thresholds
    ):
        good = (
            np.isfinite(rv_map)
            & (count_map >= nmin)
        )

        shown = np.full_like(
            rv_map,
            np.nan
        )

        shown[good] = rv_map[good]

        last_im = ax.imshow(
            shown,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax
        )

        nvalid = np.count_nonzero(good)

        ax.set_title(
            rf"$N_{{dust}}\geq {nmin}$"
            + "\n"
            + f"{nvalid:,} valid pixels"
        )

        ax.set_xlabel(
            "x' [pkpc]"
        )

    axes[0].set_ylabel(
        "y' [pkpc]"
    )

    if last_im is not None:
        cb = fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.025,
            pad=0.02
        )

        cb.set_label(
            r"$R_V$"
        )

    fig.suptitle(
        "Halo 441 — face-on $R_V$ sampling test"
    )

    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved {filename}")



# ============================================================
# R_V / grain-physics diagnostics
# ============================================================

def assign_pixels(x, y, plot_radius, bins):
    """Return (ix, iy, good) for the same grid used by histogram2d."""
    edges = np.linspace(-plot_radius, plot_radius, bins + 1)
    ix = np.searchsorted(edges, x, side="right") - 1
    iy = np.searchsorted(edges, y, side="right") - 1
    good = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    return ix, iy, good


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    v = values[good]
    w = weights[good]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    return v[np.searchsorted(np.cumsum(w), 0.5 * np.sum(w))]


def build_grain_property_maps(x, y, mass, radius_nm, carbon, plot_radius, bins,
                              small_grain_nm=50.0):
    """Mass-weighted grain properties on the exact R_V pixel grid."""
    ix, iy, good = assign_pixels(x, y, plot_radius, bins)
    ix, iy = ix[good], iy[good]
    m = np.asarray(mass[good], dtype=float)
    a = np.asarray(radius_nm[good], dtype=float)
    fc = np.clip(np.asarray(carbon[good], dtype=float), 0.0, 1.0)

    shape = (bins, bins)
    massmap = np.zeros(shape)
    carbon_mass = np.zeros(shape)
    radius_mass = np.zeros(shape)
    small_mass = np.zeros(shape)

    # map arrays are [y, x], matching histogram2d(...).T in build_maps().
    np.add.at(massmap, (iy, ix), m)
    np.add.at(carbon_mass, (iy, ix), m * fc)
    np.add.at(radius_mass, (iy, ix), m * a)
    np.add.at(small_mass, (iy, ix), m * (a < small_grain_nm))

    carbon_fraction = np.full(shape, np.nan)
    mean_radius_nm = np.full(shape, np.nan)
    small_fraction = np.full(shape, np.nan)
    occupied = massmap > 0
    carbon_fraction[occupied] = carbon_mass[occupied] / massmap[occupied]
    mean_radius_nm[occupied] = radius_mass[occupied] / massmap[occupied]
    small_fraction[occupied] = small_mass[occupied] / massmap[occupied]

    median_radius_nm = np.full(shape, np.nan)
    flatpix = iy * bins + ix
    order = np.argsort(flatpix)
    flat_sorted = flatpix[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    stops = np.r_[starts[1:], len(order)]
    for lo, hi in zip(starts, stops):
        inds = order[lo:hi]
        fp = flatpix[inds[0]]
        py, px = divmod(int(fp), bins)
        median_radius_nm[py, px] = weighted_median(a[inds], m[inds])

    return {
        "carbon_fraction": carbon_fraction,
        "mean_radius_nm": mean_radius_nm,
        "median_radius_nm": median_radius_nm,
        "small_fraction": small_fraction,
        "mass": massmap,
    }


def diagnostic_pixel_mask(rv, count_raw, min_dust_count):
    return np.isfinite(rv) & (count_raw >= min_dust_count)


def plot_rv_binned_size_distributions(rv_map, x, y, mass, radius_nm,
                                      plot_radius, bins, min_dust_count,
                                      count_raw, filename):
    """Actual dust mass-size distributions for particles in R_V-selected pixels."""
    ix, iy, inside = assign_pixels(x, y, plot_radius, bins)
    rv_for_particle = np.full(len(x), np.nan)
    rv_for_particle[inside] = rv_map[iy[inside], ix[inside]]
    count_for_particle = np.zeros(len(x))
    count_for_particle[inside] = count_raw[iy[inside], ix[inside]]

    valid = (inside & np.isfinite(rv_for_particle)
             & (count_for_particle >= min_dust_count)
             & np.isfinite(radius_nm) & (radius_nm > 0)
             & np.isfinite(mass) & (mass > 0))
    if not np.any(valid):
        print("WARNING: no particles available for R_V-binned size distributions")
        return

    amin = np.min(radius_nm[valid])
    amax = np.max(radius_nm[valid])
    size_edges = np.logspace(np.log10(amin), np.log10(amax), 36)
    centers = np.sqrt(size_edges[:-1] * size_edges[1:])
    dlog = np.diff(np.log10(size_edges))
    rv_edges = np.array([1., 2., 3., 4., 5., 6., 8., np.inf])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    print()
    print("R_V-BINNED GRAIN-SIZE DISTRIBUTIONS")
    print("-----------------------------------")
    for lo, hi in zip(rv_edges[:-1], rv_edges[1:]):
        sel = valid & (rv_for_particle >= lo) & (rv_for_particle < hi)
        if not np.any(sel):
            continue
        hist, _ = np.histogram(radius_nm[sel], bins=size_edges, weights=mass[sel])
        total = np.sum(hist)
        if total <= 0:
            continue
        curve = hist / total / dlog
        label = (rf"${lo:g}\leq R_V<{hi:g}$" if np.isfinite(hi)
                 else rf"$R_V\geq {lo:g}$")
        ax.plot(centers, curve, lw=2.0, label=label)
        pix = diagnostic_pixel_mask(rv_map, count_raw, min_dust_count) & (rv_map >= lo)
        pix &= (rv_map < hi) if np.isfinite(hi) else True
        print(f"  {label:24s} pixels={np.count_nonzero(pix):5d}  dust={np.count_nonzero(sel):7d}")

    ax.set_xscale("log")
    ax.set_xlabel(r"Grain radius $a$ (nm)")
    ax.set_ylabel(r"Normalized dust mass distribution $d(M/M_{\rm dust})/d\log_{10}a$")
    ax.set_title(r"Halo 441 — grain-size distributions by $R_V$")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_scatter(xv, yv, cv, xlabel, ylabel, clabel, title, filename):
    good = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(cv)
    if not np.any(good):
        return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(xv[good], yv[good], c=cv[good], s=20, alpha=0.65,
                    linewidths=0, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(clabel)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_grain_property_map(image, extent, title, cbar_label, filename,
                            valid_mask=None, cmap="viridis"):
    shown = np.asarray(image, dtype=float).copy()
    if valid_mask is not None:
        shown[~valid_mask] = np.nan
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(shown, origin="lower", extent=extent, aspect="equal",
                   interpolation="nearest", cmap=cmap)
    ax.set_xlabel("x (pkpc)")
    ax.set_ylabel("y (pkpc)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def build_beta_kappa_map(x, y, mass, radius_nm, carbon, plot_radius, bins,
                         carbon_qabs, silicate_qabs, wavelengths_um):
    """Intrinsic opacity-slope beta_kappa from the actual grains in each pixel.

    We sum M/a * [fC/rhoC Qabs_C + (1-fC)/rhoSil Qabs_Sil] at each
    wavelength.  Constants independent of wavelength cancel from beta.
    """
    radius_um = np.asarray(radius_nm, dtype=float) * 1.0e-3
    m = np.asarray(mass, dtype=float)
    fc = np.clip(np.asarray(carbon, dtype=float), 0.0, 1.0)
    spectra = []
    for lam in wavelengths_um:
        qc = qext_at_wavelength(carbon_qabs, radius_um, float(lam))
        qs = qext_at_wavelength(silicate_qabs, radius_um, float(lam))
        wabs = (m / radius_um) * (fc / RHO_C * qc + (1.0 - fc) / RHO_SIL * qs)
        H, _, _ = histogram_map(x, y, wabs, plot_radius, bins)
        spectra.append(H)
    spectra = np.asarray(spectra)

    beta = np.full((bins, bins), np.nan)
    loglam = np.log(np.asarray(wavelengths_um, dtype=float))
    for iy in range(bins):
        for ix in range(bins):
            kap = spectra[:, iy, ix]
            good = np.isfinite(kap) & (kap > 0)
            if np.count_nonzero(good) >= 3:
                slope, _ = np.polyfit(loglam[good], np.log(kap[good]), 1)
                beta[iy, ix] = -slope
    return beta, spectra


def planck_bnu(frequency_hz, temperature_k):
    """Planck function B_nu in SI units (the absolute units cancel in fitting)."""
    nu = np.asarray(frequency_hz, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    x = PLANCK_H * nu / (BOLTZMANN_K * temperature)
    return (2.0 * PLANCK_H * nu**3 / LIGHT_C**2) / np.expm1(x)


def build_temperature_mixed_intensity_maps(
        x, y, mass, radius_nm, carbon, temperature_k, plot_radius, bins,
        carbon_qabs, silicate_qabs, frequencies_hz):
    """Project optically thin emission using every particle's dust temperature."""
    radius_um = np.asarray(radius_nm, dtype=float) * 1.0e-3
    m = np.asarray(mass, dtype=float)
    fc = np.clip(np.asarray(carbon, dtype=float), 0.0, 1.0)
    temperature = np.asarray(temperature_k, dtype=float)
    wavelengths_um = (LIGHT_C / np.asarray(frequencies_hz, dtype=float)) * 1.0e6

    intensity_maps = []
    for nu, wavelength_um in zip(frequencies_hz, wavelengths_um):
        qc = qext_at_wavelength(carbon_qabs, radius_um, float(wavelength_um))
        qs = qext_at_wavelength(silicate_qabs, radius_um, float(wavelength_um))
        opacity_weight = (
            (m / radius_um)
            * (fc / RHO_C * qc + (1.0 - fc) / RHO_SIL * qs)
        )
        emission_weight = opacity_weight * planck_bnu(float(nu), temperature)
        intensity_map, _, _ = histogram_map(
            x, y, emission_weight, plot_radius, bins
        )
        intensity_maps.append(intensity_map)

    return np.asarray(intensity_maps), wavelengths_um


def fit_fixed_temperature_mbb(kappa_spectra, frequencies_hz, temperature_k,
                              reference_frequency_hz=353.0e9):
    """Fit I_nu=A (nu/nu0)^beta B_nu(T) in every map pixel.

    Synthetic optically thin intensities are first made from the actual grain
    opacity spectra, I_nu = kappa_nu B_nu(T).  The same fixed temperature is
    then used in the MBB fit.  Consequently B_nu(T) cancels algebraically;
    this isolates the beta implied by the selected photometric bands without
    introducing temperature-beta covariance.
    """
    nu = np.asarray(frequencies_hz, dtype=float)
    bnu = planck_bnu(nu, temperature_k)
    intensity = np.asarray(kappa_spectra, dtype=float) * bnu[:, None, None]
    beta = np.full(intensity.shape[1:], np.nan)
    amplitude = np.full(intensity.shape[1:], np.nan)
    log_frequency_ratio = np.log(nu / float(reference_frequency_hz))

    for iy in range(intensity.shape[1]):
        for ix in range(intensity.shape[2]):
            flux = intensity[:, iy, ix]
            good = np.isfinite(flux) & (flux > 0) & np.isfinite(bnu) & (bnu > 0)
            if np.count_nonzero(good) >= 3:
                reduced_flux = flux[good] / bnu[good]
                slope, intercept = np.polyfit(
                    log_frequency_ratio[good], np.log(reduced_flux), 1
                )
                beta[iy, ix] = slope
                amplitude[iy, ix] = np.exp(intercept)
    return beta, amplitude, intensity


def correlation_summary(x, y, nboot=2000, seed=441):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 3:
        return None
    pr = pearsonr(x, y).statistic
    sr = spearmanr(x, y).statistic
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    scatter = np.sqrt(np.mean(resid**2))
    rng = np.random.default_rng(seed)
    bslope, bpr, bsr = [], [], []
    n = len(x)
    for _ in range(nboot):
        ind = rng.integers(0, n, n)
        xb, yb = x[ind], y[ind]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            continue
        bslope.append(np.polyfit(xb, yb, 1)[0])
        bpr.append(pearsonr(xb, yb).statistic)
        bsr.append(spearmanr(xb, yb).statistic)
    ci = lambda a: np.percentile(a, [16, 84]) if len(a) else [np.nan, np.nan]
    return dict(n=n, pearson=pr, spearman=sr, slope=slope, intercept=intercept,
                scatter=scatter, pearson_ci=ci(bpr), spearman_ci=ci(bsr), slope_ci=ci(bslope))


def equal_population_trend(x, y, nbins=10):
    """Median relation and 16--84% spread in nearly equal-population bins."""
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[good]
    y = np.asarray(y, dtype=float)[good]
    if len(x) < nbins:
        return None

    order = np.argsort(x)
    groups = np.array_split(order, nbins)
    rows = []
    for ind in groups:
        if len(ind) == 0:
            continue
        rows.append((
            np.median(x[ind]),
            np.median(y[ind]),
            np.percentile(y[ind], 16),
            np.percentile(y[ind], 84),
            len(ind),
        ))
    return np.asarray(rows, dtype=float)


def plot_rv_beta(rv, beta, filename, summary, pixel_size_pkpc,
                 trend_filename=None, ylabel=r"$\beta$ (250--850 $\mu$m)",
                 trend_header=("Rv_median beta_kappa_median beta_kappa_p16 "
                               "beta_kappa_p84 N_pixels")):
    """Simple collaborator-facing estimate of the intrinsic correlation."""
    good = np.isfinite(rv) & np.isfinite(beta)
    x = np.asarray(rv, dtype=float)[good]
    y = np.asarray(beta, dtype=float)[good]
    if len(x) == 0:
        print("WARNING: no valid pixels for the R_V-beta_kappa plot")
        return

    trend = equal_population_trend(x, y, nbins=10)

    fig, ax = plt.subplots(figsize=(8.2, 6.7))
    ax.scatter(
        x, y, s=11, alpha=0.20, linewidths=0,
        color="#2b6f9f", rasterized=True,
        label="Valid map pixels"
    )

    if trend is not None:
        yerr = np.vstack((trend[:, 1] - trend[:, 2],
                          trend[:, 3] - trend[:, 1]))
        ax.errorbar(
            trend[:, 0], trend[:, 1], yerr=yerr,
            fmt="o-", ms=6.5, lw=2.2, capsize=3.5,
            color="#b43c2f", ecolor="#b43c2f",
            markerfacecolor="white", markeredgewidth=1.8,
            label="Median and 16--84% range"
        )

    ax.set_xlabel(r"$R(V)$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(
        rf"Halo 441 (face-on {1000.0 * pixel_size_pkpc:g} pc pixels)",
        fontsize=14
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=11)

    # Keep rare surviving extremes from compressing the visible relation.
    # They remain in the correlation statistics and saved data.
    if len(x) >= 20:
        xlo, xhi = np.percentile(x, [0.5, 99.5])
        if np.isfinite(xlo) and np.isfinite(xhi) and xhi > xlo:
            pad = 0.03 * (xhi - xlo)
            plot_lo = max(0.0, xlo - pad)
            plot_hi = xhi + pad
            ax.set_xlim(plot_lo, plot_hi)
            n_outside = np.count_nonzero((x < plot_lo) | (x > plot_hi))
            print(
                f"R_V display range = {plot_lo:.4f} to {plot_hi:.4f}; "
                f"{n_outside} valid pixels outside display range"
            )

    fig.tight_layout()
    fig.savefig(filename, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

    if trend_filename is not None and trend is not None:
        np.savetxt(
            trend_filename, trend,
            header=trend_header,
            fmt=["%.8e", "%.8e", "%.8e", "%.8e", "%.0f"]
        )
        print(f"Saved {trend_filename}")


def run_grain_rv_diagnostics(args, maps, x, y, dust, mapmask, carbon_qabs, silicate_qabs):
    """Generate the six-figure R_V / grain-physics analysis package."""
    xm, ym = x[mapmask], y[mapmask]
    mm = dust["mass"][mapmask]
    am = dust["radius_nm"][mapmask]
    cm = dust["carbon"][mapmask]
    tm = dust["temperature_k"][mapmask]

    props = build_grain_property_maps(
        xm, ym, mm, am, cm, args.plot_radius, args.bins, args.small_grain_nm
    )
    valid = diagnostic_pixel_mask(maps["rv"], maps["count_raw"], args.diagnostic_min_count)

    print()
    print("=" * 72)
    print("R_V / GRAIN-PHYSICS DIAGNOSTICS")
    print("=" * 72)
    print(f"diagnostic N_dust cut = {args.diagnostic_min_count}")
    print(f"small-grain threshold = {args.small_grain_nm:g} nm")
    print(f"valid diagnostic pixels = {np.count_nonzero(valid):,}")
    print("=" * 72)

    # 1: existing R_V map is already produced by the parent pipeline.
    # 2: size distributions grouped by the R_V of the exact contributing pixels.
    plot_rv_binned_size_distributions(
        maps["rv"], xm, ym, mm, am, args.plot_radius, args.bins,
        args.diagnostic_min_count, maps["count_raw"],
        args.prefix + "_diag02_size_distribution_by_Rv.png"
    )

    rv = maps["rv"][valid]
    fc = props["carbon_fraction"][valid]
    meda = props["median_radius_nm"][valid]
    fsmall = props["small_fraction"][valid]

    # 3: R_V vs small-grain mass fraction, colored by live carbon fraction.
    plot_scatter(
        rv, fsmall, fc,
        r"$R_V$", rf"Small-grain mass fraction ($a<{args.small_grain_nm:g}$ nm)",
        "Carbon mass fraction",
        r"Halo 441 — $R_V$ vs small-grain fraction",
        args.prefix + "_diag03_Rv_vs_smallgrain_CF.png"
    )

    # 4: R_V vs composition, colored by mass-weighted median grain radius.
    plot_scatter(
        rv, fc, meda,
        r"$R_V$", "Carbon mass fraction", "Median grain radius (nm)",
        r"Halo 441 — $R_V$ vs composition",
        args.prefix + "_diag04_Rv_vs_CF_medianradius.png"
    )

    # 5: size-composition plane, colored by R_V.
    plot_scatter(
        fsmall, fc, rv,
        rf"Small-grain mass fraction ($a<{args.small_grain_nm:g}$ nm)",
        "Carbon mass fraction", r"$R_V$",
        r"Halo 441 — grain size/composition plane",
        args.prefix + "_diag05_smallgrain_vs_CF_Rv.png"
    )

    # 6: matched individual maps using precisely the same validity mask.
    plot_grain_property_map(
        maps["rv"], maps["extent"], r"Halo 441 — face-on $R_V$",
        r"$R_V$", args.prefix + "_diag06a_map_Rv.png", valid
    )
    plot_grain_property_map(
        props["small_fraction"], maps["extent"],
        "Halo 441 — face-on small-grain mass fraction",
        rf"$M(a<{args.small_grain_nm:g}\,{{\rm nm}})/M_{{\rm dust}}$",
        args.prefix + "_diag06b_map_smallgrain.png", valid
    )
    plot_grain_property_map(
        props["median_radius_nm"], maps["extent"],
        "Halo 441 — face-on median grain radius",
        "Mass-weighted median grain radius (nm)",
        args.prefix + "_diag06c_map_medianradius.png", valid
    )
    plot_grain_property_map(
        props["carbon_fraction"], maps["extent"],
        "Halo 441 — face-on carbon mass fraction",
        "Carbon mass fraction",
        args.prefix + "_diag06d_map_CF.png", valid
    )

    # 7: intrinsic FIR opacity slope from Q_abs of the same grains.
    beta_wavelengths_um = np.array([250.0, 350.0, 500.0, 550.0, 850.0])
    beta_map, kappa_spectra = build_beta_kappa_map(
        xm, ym, mm, am, cm, args.plot_radius, args.bins,
        carbon_qabs, silicate_qabs, beta_wavelengths_um
    )
    beta_valid = valid & np.isfinite(beta_map)
    summary = correlation_summary(maps["rv"][beta_valid], beta_map[beta_valid])
    print()
    print("INTRINSIC R_V - BETA_KAPPA CORRELATION (250-850 micron)")
    print("------------------------------------------------------")
    if summary is not None:
        print(f"pixels       = {summary['n']:,}")
        print(f"Pearson r    = {summary['pearson']:.4f}  (bootstrap 16-84%: {summary['pearson_ci'][0]:.4f} to {summary['pearson_ci'][1]:.4f})")
        print(f"Spearman rho = {summary['spearman']:.4f}  (bootstrap 16-84%: {summary['spearman_ci'][0]:.4f} to {summary['spearman_ci'][1]:.4f})")
        print(f"slope        = {summary['slope']:.4f}  (bootstrap 16-84%: {summary['slope_ci'][0]:.4f} to {summary['slope_ci'][1]:.4f})")
        print(f"RMS scatter  = {summary['scatter']:.4f}")
    plot_rv_beta(
        maps["rv"][beta_valid], beta_map[beta_valid],
        args.prefix + "_diag07_Rv_vs_beta_kappa.png", summary,
        maps["pixel_size"],
        args.prefix + "_diag07_Rv_vs_beta_kappa_binned.txt"
    )
    plot_grain_property_map(
        beta_map, maps["extent"], r"Halo 441 --- intrinsic $\beta_\kappa$",
        r"$\beta_\kappa$ (250--850 $\mu$m)",
        args.prefix + "_diag08_map_beta_kappa.png", beta_valid
    )

    # 9: fixed-temperature MBB beta in the IRAS/Planck bands used by
    # Zelko & Finkbeiner (2020): 3000, 857, 545, and 353 GHz.
    mbb_frequencies_ghz = np.array([3000.0, 857.0, 545.0, 353.0])
    mbb_frequencies_hz = mbb_frequencies_ghz * 1.0e9
    mbb_wavelengths_um = (LIGHT_C / mbb_frequencies_hz) * 1.0e6
    _, mbb_kappa_spectra = build_beta_kappa_map(
        xm, ym, mm, am, cm, args.plot_radius, args.bins,
        carbon_qabs, silicate_qabs, mbb_wavelengths_um
    )
    beta_mbb, mbb_amplitude, mbb_intensity = fit_fixed_temperature_mbb(
        mbb_kappa_spectra, mbb_frequencies_hz, args.dust_temperature_k
    )

    temperature_mixed_intensity, temperature_mixed_wavelengths_um = (
        build_temperature_mixed_intensity_maps(
            xm, ym, mm, am, cm, tm, args.plot_radius, args.bins,
            carbon_qabs, silicate_qabs, mbb_frequencies_hz
        )
    )
    print(
        "Saved ingredients for free-T MBB fitting from particle-by-particle "
        "DustTemperature values"
    )
    mbb_valid = valid & np.isfinite(beta_mbb)
    mbb_summary = correlation_summary(
        maps["rv"][mbb_valid], beta_mbb[mbb_valid]
    )
    print()
    print(f"FIXED-T MBB R_V - BETA CORRELATION (T_d={args.dust_temperature_k:g} K)")
    print("IRAS/Planck nominal bands: 3000, 857, 545, 353 GHz")
    print("------------------------------------------------------------")
    print("NOTE: because the synthetic and fitted temperatures are identical and")
    print("fixed, B_nu(T) cancels; this test has no T-beta degeneracy.")
    if mbb_summary is not None:
        print(f"pixels       = {mbb_summary['n']:,}")
        print(f"Pearson r    = {mbb_summary['pearson']:.4f}  (bootstrap 16-84%: {mbb_summary['pearson_ci'][0]:.4f} to {mbb_summary['pearson_ci'][1]:.4f})")
        print(f"Spearman rho = {mbb_summary['spearman']:.4f}  (bootstrap 16-84%: {mbb_summary['spearman_ci'][0]:.4f} to {mbb_summary['spearman_ci'][1]:.4f})")
        print(f"slope        = {mbb_summary['slope']:.4f}  (bootstrap 16-84%: {mbb_summary['slope_ci'][0]:.4f} to {mbb_summary['slope_ci'][1]:.4f})")
        print(f"RMS scatter  = {mbb_summary['scatter']:.4f}")
    plot_rv_beta(
        maps["rv"][mbb_valid], beta_mbb[mbb_valid],
        args.prefix + "_diag09_Rv_vs_beta_mbb_Tfixed.png", mbb_summary,
        maps["pixel_size"],
        args.prefix + "_diag09_Rv_vs_beta_mbb_Tfixed_binned.txt",
        ylabel=rf"$\beta_{{\rm MBB}}$ ($T_d={args.dust_temperature_k:g}$ K fixed)",
        trend_header=("Rv_median beta_MBB_median beta_MBB_p16 "
                      "beta_MBB_p84 N_pixels")
    )
    plot_grain_property_map(
        beta_mbb, maps["extent"],
        rf"Halo 441 --- fixed-$T$ $\beta_{{\rm MBB}}$ ({args.dust_temperature_k:g} K)",
        r"$\beta_{\rm MBB}$", args.prefix + "_diag10_map_beta_mbb_Tfixed.png",
        mbb_valid
    )

    np.savez_compressed(
        args.prefix + "_grain_rv_diagnostics.npz",
        Rv=maps["rv"],
        valid_mask=valid,
        carbon_mass_fraction=props["carbon_fraction"],
        mean_grain_radius_nm=props["mean_radius_nm"],
        median_grain_radius_nm=props["median_radius_nm"],
        small_grain_mass_fraction=props["small_fraction"],
        small_grain_threshold_nm=args.small_grain_nm,
        diagnostic_min_dust_count=args.diagnostic_min_count,
        pixel_size_pkpc=maps["pixel_size"],
        extent=np.asarray(maps["extent"]),
        beta_kappa_250_850=beta_map,
        beta_kappa_wavelengths_um=beta_wavelengths_um,
        kappa_relative_spectra=kappa_spectra,
        beta_valid_mask=beta_valid,
        beta_mbb_fixed_temperature=beta_mbb,
        beta_mbb_valid_mask=mbb_valid,
        mbb_temperature_k=args.dust_temperature_k,
        mbb_frequencies_ghz=mbb_frequencies_ghz,
        mbb_wavelengths_um=mbb_wavelengths_um,
        mbb_relative_kappa_spectra=mbb_kappa_spectra,
        mbb_synthetic_intensity=mbb_intensity,
        mbb_fitted_amplitude=mbb_amplitude,
        mbb_temperature_mixed_intensity=temperature_mixed_intensity,
        mbb_temperature_mixed_wavelengths_um=temperature_mixed_wavelengths_um,
    )
    print(f"Saved {args.prefix}_grain_rv_diagnostics.npz")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot"
    )

    parser.add_argument(
        "--center-file",
        default="halo441_center.txt"
    )

    parser.add_argument(
        "--rotation-file",
        default="halo441_rotation_matrix.txt"
    )

    parser.add_argument(
        "--carbon-table",
        default=(
            "dust_optics/"
            "callqcomp.out_D16MGemt"
        )
    )

    parser.add_argument(
        "--silicate-table",
        default=(
            "dust_optics/"
            "callqcomp.out_DH16asil"
        )
    )

    parser.add_argument(
        "--plot-radius",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=80
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--min-signal-fraction",
        type=float,
        default=1.0e-3
    )

    parser.add_argument(
        "--min-color-excess-fraction",
        type=float,
        default=1.0e-3,
        help=(
            "Minimum (tauB-tauV) as a fraction of the map maximum "
            "for a valid R_V pixel (default: 1e-3)"
        )
    )

    parser.add_argument(
        "--young-star-age-myr",
        type=float,
        default=100.0,
        help="Maximum stellar age for open-circle markers (default: 100 Myr)"
    )

    parser.add_argument(
        "--prefix",
        default="halo441_faceon_500pc"
    )

    parser.add_argument(
        "--diagnostic-min-count",
        type=int,
        default=5,
        help="Minimum raw dust particles per pixel for grain/R_V diagnostics (default: 5)"
    )

    parser.add_argument(
        "--small-grain-nm",
        type=float,
        default=50.0,
        help="Radius threshold defining small grains in nm (default: 50)"
    )

    parser.add_argument(
        "--dust-temperature-k",
        type=float,
        default=17.0,
        help=(
            "Fixed dust temperature used both to synthesize and fit the "
            "IRAS/Planck modified blackbody (default: 17 K)"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Center and rotation
    # --------------------------------------------------------

    center = np.asarray(
        np.loadtxt(args.center_file),
        dtype=float
    ).reshape(3)

    rotation = np.asarray(
        np.loadtxt(args.rotation_file),
        dtype=float
    ).reshape(3, 3)

    print()
    print("Galaxy center")
    print("-------------")
    print(
        f"({center[0]:.8f}, "
        f"{center[1]:.8f}, "
        f"{center[2]:.8f})"
    )

    print()
    print("Face-on rotation matrix")
    print("-----------------------")

    for row in rotation:
        print(
            "  "
            + " ".join(
                f"{q: .8f}"
                for q in row
            )
        )

    # --------------------------------------------------------
    # Optical tables and dust snapshot
    # --------------------------------------------------------

    carbon_blocks = read_draine_qext_table(
        args.carbon_table
    )

    silicate_blocks = read_draine_qext_table(
        args.silicate_table
    )

    carbon_qabs = read_draine_qabs_table(args.carbon_table)
    silicate_qabs = read_draine_qabs_table(args.silicate_table)

    dust = read_dust_snapshot(
        args.snapshot
    )

    pos = rotate_positions(
        dust["coords"],
        center,
        dust["box"],
        dust["a"],
        dust["h"],
        rotation
    )

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    gas = read_starforming_gas_snapshot(args.snapshot)
    if len(gas["coords"]) > 0:
        gpos = rotate_positions(
            gas["coords"], center, gas["box"], gas["a"], gas["h"], rotation
        )
        sfmap = (
            (np.abs(gpos[:, 0]) < args.plot_radius)
            & (np.abs(gpos[:, 1]) < args.plot_radius)
            & (np.abs(gpos[:, 2]) < args.zmax)
        )
        sf_xy = gpos[sfmap, :2]
        sf_sfr = gas["sfr"][sfmap]
    else:
        sf_xy = np.empty((0, 2))
        sf_sfr = np.empty(0)

    print(f"Star-forming gas particles entering face-on map = {len(sf_xy):,}")
    if len(sf_sfr):
        print(f"Total instantaneous SFR in map = {np.sum(sf_sfr):.6g} Msun/yr")

    young = read_young_stars_snapshot(args.snapshot, args.young_star_age_myr)
    if len(young["coords"]) > 0:
        ypos = rotate_positions(
            young["coords"], center, young["box"], young["a"], young["h"], rotation
        )
        ymap = (
            (np.abs(ypos[:, 0]) < args.plot_radius)
            & (np.abs(ypos[:, 1]) < args.plot_radius)
            & (np.abs(ypos[:, 2]) < args.zmax)
        )
        young_xy = ypos[ymap, :2]
    else:
        young_xy = np.empty((0, 2))

    # Put both tracers at the centers of the SAME 500-pc pixels as the R_V map.
    # Multiple particles in one pixel become one marker.
    sf_pix_xy = snap_xy_to_pixel_centers(sf_xy, args.plot_radius, args.bins)
    young_pix_xy = snap_xy_to_pixel_centers(young_xy, args.plot_radius, args.bins)

    print(f"Young stars (<{args.young_star_age_myr:g} Myr) entering map = {len(young_xy):,}")
    print(f"R_V pixels containing SFR>0 gas = {len(sf_pix_xy):,}")
    print(f"R_V pixels containing young stars = {len(young_pix_xy):,}")

    WB, WV = extinction_weights(
        dust["mass"],
        dust["radius_nm"],
        dust["carbon"],
        carbon_blocks,
        silicate_blocks
    )

    # --------------------------------------------------------
    # Select LOS volume for map
    # --------------------------------------------------------

    mapmask = (
        (np.abs(x) < args.plot_radius)
        & (np.abs(y) < args.plot_radius)
        & (np.abs(z) < args.zmax)
    )

    ndust_map = np.count_nonzero(
        mapmask
    )

    print()
    print(
        f"Dust particles entering face-on map = "
        f"{ndust_map:,}"
    )

    print(
        f"Map dust mass [snapshot units] = "
        f"{np.sum(dust['mass'][mapmask]):.8e}"
    )

    if ndust_map < 20:
        raise RuntimeError(
            "Too few dust particles for face-on map."
        )

    maps = build_maps(
        x[mapmask],
        y[mapmask],
        dust["mass"][mapmask],
        WB[mapmask],
        WV[mapmask],
        args.plot_radius,
        args.bins,
        args.smooth,
        args.min_signal_fraction,
        args.min_color_excess_fraction
    )

    rvvals = maps["rv"][
        np.isfinite(maps["rv"])
    ]

    print()
    print("=" * 72)
    print("FACE-ON R_V MAP STATISTICS")
    print("=" * 72)

    if len(rvvals) > 0:
        print(f"valid pixels = {len(rvvals):,}")
        print(
            f"median R_V   = "
            f"{np.median(rvvals):.4f}"
        )
        print(
            f"16-84%       = "
            f"{np.percentile(rvvals,16):.4f} - "
            f"{np.percentile(rvvals,84):.4f}"
        )
        print(
            f"min / max    = "
            f"{np.min(rvvals):.4f} / "
            f"{np.max(rvvals):.4f}"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # Main R_V heatmap
    # --------------------------------------------------------

    if len(rvvals) > 5:
        rv_vmin = np.percentile(
            rvvals,
            2
        )
        rv_vmax = np.percentile(
            rvvals,
            98
        )
    else:
        rv_vmin = None
        rv_vmax = None

    plot_map(
        maps["rv"],
        maps["extent"],
        "Halo 441 — face-on $R_V$",
        r"$R_V$",
        args.prefix
        + "_Rv.png",
        vmin=rv_vmin,
        vmax=rv_vmax,
        sf_xy=sf_pix_xy,
        young_xy=young_pix_xy
    )

    # --------------------------------------------------------
    # Dust-particle count map
    # --------------------------------------------------------

    count_display = np.asarray(
        maps["count_raw"],
        dtype=float
    ).copy()

    count_display[
        count_display <= 0
    ] = np.nan

    plot_map(
        count_display,
        maps["extent"],
        "Halo 441 — dust superparticles per map pixel",
        r"$N_{\rm dust}$",
        args.prefix
        + "_dustcount.png"
    )

    # --------------------------------------------------------
    # Dust mass map
    # --------------------------------------------------------

    logmass = np.full_like(
        maps["mass_raw"],
        np.nan
    )

    good_mass = (
        maps["mass_raw"] > 0
    )

    logmass[good_mass] = np.log10(
        maps["mass_raw"][good_mass]
    )

    plot_map(
        logmass,
        maps["extent"],
        "Halo 441 — face-on dust mass per pixel",
        "log dust mass [snapshot units]",
        args.prefix
        + "_dustmass.png"
    )

    # --------------------------------------------------------
    # Sampling cuts
    # --------------------------------------------------------

    thresholds = [
        1,
        3,
        5,
        10
    ]

    print()
    print("=" * 80)
    print("R_V PIXEL SAMPLING BY RAW DUST-PARTICLE COUNT")
    print("=" * 80)

    for nmin in thresholds:
        good = (
            np.isfinite(maps["rv"])
            & (
                maps["count_raw"] >= nmin
            )
        )

        vals = maps["rv"][
            good
        ]

        print(
            f"N_dust >= {nmin:2d}: "
            f"valid pixels = {len(vals):5d}",
            end=""
        )

        if len(vals) > 0:
            print(
                f"   median R_V = {np.median(vals):7.3f}"
                f"   16-84% = "
                f"{np.percentile(vals,16):7.3f}"
                f" - {np.percentile(vals,84):7.3f}"
            )
        else:
            print()

    print("=" * 80)

    plot_count_thresholds(
        maps["rv"],
        maps["count_raw"],
        maps["extent"],
        thresholds,
        args.prefix
        + "_Rv_count_thresholds.png"
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.savez_compressed(
        args.prefix
        + "_heatmap_data.npz",
        Rv=maps["rv"],
        tauB=maps["tauB"],
        tauV=maps["tauV"],
        color_excess=maps["color_excess"],
        tauB_raw=maps["tauB_raw"],
        tauV_raw=maps["tauV_raw"],
        dustmass_raw=maps["mass_raw"],
        dust_particle_count_raw=maps["count_raw"],
        pixel_size_pkpc=maps["pixel_size"],
        plot_radius_pkpc=args.plot_radius,
        zmax_pkpc=args.zmax,
        bins=args.bins,
        smoothing_sigma_pkpc=args.smooth,
        min_signal_fraction=args.min_signal_fraction,
        min_color_excess_fraction=args.min_color_excess_fraction,
        tauV_signal_cut=maps["tauV_signal_cut"],
        color_excess_cut=maps["color_excess_cut"],
        center=center,
        rotation_matrix=rotation
    )

    print(
        f"Saved "
        f"{args.prefix}_heatmap_data.npz"
    )

    # --------------------------------------------------------
    # R_V / grain-physics analysis package
    # --------------------------------------------------------
    run_grain_rv_diagnostics(args, maps, x, y, dust, mapmask, carbon_qabs, silicate_qabs)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
