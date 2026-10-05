#!/usr/bin/env python3

"""
halo441_rv_heatmap.py

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


# ============================================================
# Constants
# ============================================================

RHO_C = 2.24
RHO_SIL = 3.50

B_MICRON = 0.44
V_MICRON = 0.55


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

            required = ["Coordinates", "Masses", "GrainRadius"]
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

    if len(mass) == 0:
        raise RuntimeError(
            "No PartType6 particles found."
        )

    coords = np.concatenate(coords)
    mass = np.concatenate(mass)
    radius_nm = np.concatenate(radius_nm)
    carbon = np.concatenate(carbon)

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

    return {
        "coords": coords,
        "mass": mass,
        "radius_nm": radius_nm,
        "carbon": carbon,
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
    min_signal_fraction
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

    good = (
        (tauV > signal_cut)
        & (denom > 0)
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
        "tauB_raw": tauB_raw,
        "tauV_raw": tauV_raw,
        "mass": massmap,
        "mass_raw": mass_raw,
        "count_raw": count_raw,
        "pixel_size": pixel_size,
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
        "Halo 569 — face-on $R_V$ sampling test"
    )

    fig.savefig(
        filename,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved {filename}")


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
        "--young-star-age-myr",
        type=float,
        default=100.0,
        help="Maximum stellar age for open-circle markers (default: 100 Myr)"
    )

    parser.add_argument(
        "--prefix",
        default="halo441_faceon_500pc"
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
        args.min_signal_fraction
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
        tauB_raw=maps["tauB_raw"],
        tauV_raw=maps["tauV_raw"],
        dustmass_raw=maps["mass_raw"],
        dust_particle_count_raw=maps["count_raw"],
        pixel_size_pkpc=maps["pixel_size"],
        plot_radius_pkpc=args.plot_radius,
        zmax_pkpc=args.zmax,
        bins=args.bins,
        smoothing_sigma_pkpc=args.smooth,
        center=center,
        rotation_matrix=rotation
    )

    print(
        f"Saved "
        f"{args.prefix}_heatmap_data.npz"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
