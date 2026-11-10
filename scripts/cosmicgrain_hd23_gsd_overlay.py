
#!/usr/bin/env python3
"""
CosmicGrain vs Hensley & Draine (2023)
======================================

Compare mass-weighted grain-size distributions (GSDs).

Rows:
    1. Entire simulation
    2. Central galaxy (spherical aperture)

Columns:
    1. Dust mass fraction per dex
    2. Cumulative dust mass fraction

Curves:
    Black: CosmicGrain
    Blue: Hensley & Draine (2023), astrodust + PAHs

HD23 data:
    astrodust+PAH_MW_RV3.1.fits

The HD23 FITS size-distribution columns contain the number of
grains per H atom IN EACH SIZE BIN, not dn/da.

The conversion to dust mass is therefore proportional to

    M_bin = N_bin * rho * a**3

without multiplying by another bin width.

The HD23 conversion uses effective spherical radii and assumed
material densities. It should be checked against published
component mass abundances before quantitative calibration.
"""

from pathlib import Path
import argparse
import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# HD23
# ============================================================

def read_hd23(path, rho_ast=2.74, rho_pah=2.20):
    """
    Return HD23 grain radii and mass weights.

    Grain radius: nm
    Mass weights: proportional to mass per H atom
    """

    with fits.open(path) as hdus:
        d = np.asarray(
            hdus["SIZE DISTRIBUTION"].data,
            dtype=float
        )

    if d.ndim != 2 or d.shape[1] != 5:
        raise ValueError(
            f"Unexpected HD23 size distribution shape: {d.shape}"
        )

    a = d[:, 0] * 1000.0

    n_ast = d[:, 1]
    n_pah = d[:, 2]

    if np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError("HD23 radii must be positive and increasing")

    # Common geometric factor 4*pi/3 cancels in normalization.
    # Units of radius also cancel between the components.
    m_ast = n_ast * rho_ast * a**3
    m_pah = n_pah * rho_pah * a**3

    m_total = m_ast + m_pah

    if not np.all(np.isfinite(m_total)) or m_total.sum() <= 0:
        raise ValueError("Invalid HD23 mass distribution")

    return a, m_ast, m_pah, m_total


# ============================================================
# HALO CENTER
# ============================================================

def read_group_center(run, snap, group_index):
    """
    Read GroupPos from a FoF group catalog.

    Group indices are zero-based and should be verified
    to identify the intended halo.
    """

    patterns = [
        f"{run}/groups_{snap:03d}/"
        f"fof_subhalo_tab_{snap:03d}.*.hdf5",

        f"{run}/groups_{snap:03d}/"
        f"fof_subhalo_tab_{snap:03d}.hdf5",
    ]

    files = []

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"No group catalog found for {run}, snap {snap}"
        )

    # Read GroupPos from the first catalog chunk.
    # Suitable for Group 1 when present in that chunk.
    with h5py.File(files[0], "r") as f:
        positions = f["Group"]["GroupPos"][:]

    if group_index >= len(positions):
        raise ValueError(
            f"Group {group_index} not present in first catalog chunk"
        )

    center = np.asarray(positions[group_index], dtype=float)

    print(
        f"Group {group_index} center: "
        f"{center[0]:.6f} "
        f"{center[1]:.6f} "
        f"{center[2]:.6f}"
    )

    return center


# ============================================================
# COSMICGRAIN SNAPSHOT
# ============================================================

def snapshot_files(run, snap):
    """
    Locate chunked or single-file GADGET snapshots.
    """

    patterns = [
        f"{run}/snapdir_{snap:03d}/"
        f"snapshot_{snap:03d}.*.hdf5",

        f"{run}/snapdir_{snap:03d}/"
        f"snapshot_{snap:03d}.hdf5",

        f"{run}/snapshot_{snap:03d}.hdf5",
    ]

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return files

    raise FileNotFoundError(
        f"No snapshot files found for {run}, snap {snap}"
    )


def read_cg(run, snap, center, aperture, hubble_override=None):
    """
    Read PartType6 dust and select the central galaxy.

    Assumes snapshot coordinates are in comoving kpc/h,
    with physical distances calculated using scale/h.
    """

    files = snapshot_files(run, snap)

    radii = []
    masses = []
    positions = []

    scale = None
    box = None
    hubble = None

    for fn in files:
        with h5py.File(fn, "r") as f:

            if scale is None:
                header = f["Header"].attrs

                scale = float(header["Time"])
                box = float(header["BoxSize"])

                if hubble_override is not None:
                    hubble = float(hubble_override)
                elif "HubbleParam" in header:
                    hubble = float(header["HubbleParam"])
                elif "Parameters" in f and \
                        "HubbleParam" in f["Parameters"].attrs:
                    hubble = float(
                        f["Parameters"].attrs["HubbleParam"]
                    )
                elif "Config" in f and \
                        "HubbleParam" in f["Config"].attrs:
                    hubble = float(
                        f["Config"].attrs["HubbleParam"]
                    )
                else:
                    raise ValueError(
                        "HubbleParam not found in snapshot metadata. "
                        "Pass --hubble explicitly."
                    )

            if "PartType6" not in f:
                continue

            dust = f["PartType6"]

            radii.append(
                np.asarray(dust["GrainRadius"][:], dtype=float)
            )

            masses.append(
                np.asarray(dust["Masses"][:], dtype=float)
            )

            positions.append(
                np.asarray(dust["Coordinates"][:], dtype=float)
            )

    if not radii:
        raise ValueError("No PartType6 dust found")

    a = np.concatenate(radii)
    m = np.concatenate(masses)
    pos = np.concatenate(positions)

    valid = (
        np.isfinite(a) &
        np.isfinite(m) &
        (a > 0) &
        (m > 0) &
        np.all(np.isfinite(pos), axis=1)
    )

    if not np.all(valid):
        print(
            f"Warning: excluding {(~valid).sum():,} "
            "invalid dust particles"
        )

        a = a[valid]
        m = m[valid]
        pos = pos[valid]

    center = np.asarray(center, dtype=float)

    delta = pos - center

    if box > 0:
        delta -= box * np.rint(delta / box)

    distance = np.linalg.norm(delta, axis=1) * scale / hubble

    central = distance < aperture

    print("\nCosmicGrain snapshot:")
    print(f"  Run: {run}")
    print(f"  Snapshot: {snap:03d}")
    print(f"  Scale factor: {scale:.8f}")
    print(f"  Hubble parameter: {hubble:.6f}")
    print(f"  Physical conversion: {scale/hubble:.6f} pkpc/unit")
    print(f"  Entire simulation: {len(a):,} dust particles")
    print(f"  Central galaxy: {central.sum():,} dust particles")
    print(f"  Entire dust mass: {m.sum():.8e} code units")
    print(f"  Central dust mass: {m[central].sum():.8e} code units")

    return (a, m), (a[central], m[central])


# ============================================================
# DISTRIBUTION UTILITIES
# ============================================================

def log_edges(a):
    """
    Logarithmic bin edges around HD23 bin centers.
    """

    loga = np.log10(a)

    mid = 0.5 * (loga[1:] + loga[:-1])

    return 10**np.r_[
        loga[0] - (mid[0] - loga[0]),
        mid,
        loga[-1] + (loga[-1] - mid[-1])
    ]


def binned_gsd(a, m, edges):
    """
    Normalized dust mass per dex.

    Integral over log10(radius) equals unity if all
    grain radii fall inside the bin range.
    """

    total = m.sum()

    if total <= 0:
        raise ValueError("Zero dust mass")

    hist = np.histogram(
        a,
        bins=edges,
        weights=m
    )[0]

    widths = np.diff(np.log10(edges))

    return hist / total / widths


def cumulative(a, m):
    """
    Cumulative fraction of dust mass below grain radius a.
    """

    order = np.argsort(a)

    x = a[order]
    y = np.cumsum(m[order]) / m.sum()

    return x, y


def mass_fraction(a, m, cutoff):
    return m[a <= cutoff].sum() / m.sum()


def weighted_median(a, m):
    order = np.argsort(a)

    aa = a[order]
    mm = m[order]

    cumulative_mass = np.cumsum(mm)

    index = np.searchsorted(
        cumulative_mass,
        0.5 * cumulative_mass[-1]
    )

    return aa[min(index, len(aa) - 1)]


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--run",
        required=True
    )

    p.add_argument(
        "--snap",
        type=int,
        default=29
    )

    p.add_argument(
        "--fits",
        default=(
            "data/dust_observations/"
            "astrodust+PAH_MW_RV3.1.fits"
        )
    )

    p.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None
    )

    p.add_argument(
        "--center-from-group",
        type=int,
        default=None,
        metavar="INDEX"
    )

    p.add_argument(
        "--aperture",
        type=float,
        default=30.0
    )

    p.add_argument(
        "--hubble",
        type=float,
        default=None
    )

    p.add_argument(
        "--out",
        default="analysis_products/cg10_hd23_gsd"
    )

    args = p.parse_args()

    if args.center is not None and \
            args.center_from_group is not None:
        p.error(
            "Use --center OR --center-from-group, not both"
        )

    if args.center is None and \
            args.center_from_group is None:
        p.error(
            "Specify --center or --center-from-group"
        )

    # --------------------------------------------------------
    # Matplotlib style
    # --------------------------------------------------------

    style = Path("scripts/cosmicgrain.mplstyle")

    if style.exists():
        plt.style.use(str(style))
        print(f"Matplotlib style: {style}")

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    if args.center_from_group is not None:
        center = read_group_center(
            args.run,
            args.snap,
            args.center_from_group
        )
    else:
        center = np.asarray(args.center, dtype=float)

    # --------------------------------------------------------
    # HD23
    # --------------------------------------------------------

    a_hd, m_ast, m_pah, m_hd = read_hd23(
        args.fits
    )

    hd_total = m_hd.sum()

    edges_hd = log_edges(a_hd)

    widths_hd = np.diff(
        np.log10(edges_hd)
    )

    y_hd = m_hd / hd_total / widths_hd

    cumulative_hd = np.cumsum(m_hd) / hd_total

    # --------------------------------------------------------
    # CosmicGrain
    # --------------------------------------------------------

    full, central = read_cg(
        args.run,
        args.snap,
        center,
        args.aperture,
        args.hubble
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True
    )

    cg_color = "black"
    hd_color = "tab:blue"

    edges_cg = np.logspace(
        np.log10(0.3),
        np.log10(6000),
        110
    )

    rows = [
        ("Entire simulation", full),
        (
            f"Central galaxy (r < {args.aperture:g} pkpc)",
            central
        )
    ]

    for row, (row_title, data) in enumerate(rows):

        a_cg, m_cg = data

        ax, axc = axs[row]

        # ----------------------------------------------------
        # Differential GSD
        # ----------------------------------------------------

        y_cg = binned_gsd(
            a_cg,
            m_cg,
            edges_cg
        )

        ax.stairs(
            y_cg,
            edges_cg,
            color=cg_color,
            linewidth=2.0,
            label="CosmicGrain"
        )

        ax.plot(
            a_hd,
            y_hd,
            color=hd_color,
            linewidth=2.5,
            label="Hensley & Draine (2023)"
        )

        ax.set_xscale("log")

        ax.set_ylabel(
            "Dust mass fraction per dex"
        )

        ax.set_title(row_title)

        ax.legend(
            fontsize=8,
            loc="upper right"
        )

        # ----------------------------------------------------
        # Cumulative GSD
        # ----------------------------------------------------

        x_cg, c_cg = cumulative(
            a_cg,
            m_cg
        )

        axc.plot(
            x_cg,
            c_cg,
            color=cg_color,
            linewidth=2.0,
            label="CosmicGrain"
        )

        axc.plot(
            a_hd,
            cumulative_hd,
            color=hd_color,
            linewidth=2.5,
            label="Hensley & Draine (2023)"
        )

        axc.set_xscale("log")

        axc.set_ylim(0, 1.03)

        axc.set_ylabel(
            "Cumulative dust mass fraction"
        )

        axc.set_title(row_title)

        axc.legend(
            fontsize=8,
            loc="lower right"
        )

    # --------------------------------------------------------
    # Common axis formatting
    # --------------------------------------------------------

    for ax in axs[-1]:
        ax.set_xlabel(
            "Grain radius (nm)"
        )

    for ax in axs.flat:
        ax.set_xlim(0.3, 6000)
        ax.grid(alpha=0.15)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = Path(args.out)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")

    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Numerical diagnostics
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("MASS-WEIGHTED GRAIN-SIZE COMPARISON")
    print("=" * 72)

    print(
        f"\n{'Cutoff':>10s} "
        f"{'HD23':>12s} "
        f"{'CG entire':>12s} "
        f"{'CG central':>12s}"
    )

    for cutoff in [1.5, 10, 50, 100, 200]:

        hd = mass_fraction(
            a_hd, m_hd, cutoff
        )

        cf = mass_fraction(
            full[0], full[1], cutoff
        )

        cc = mass_fraction(
            central[0], central[1], cutoff
        )

        print(
            f"{cutoff:8.1f} nm "
            f"{hd:12.3%} "
            f"{cf:12.3%} "
            f"{cc:12.3%}"
        )

    print("\nMass-weighted median grain radius:")

    print(
        f"  HD23: "
        f"{weighted_median(a_hd, m_hd):.3f} nm"
    )

    print(
        f"  CG entire: "
        f"{weighted_median(*full):.3f} nm"
    )

    print(
        f"  CG central: "
        f"{weighted_median(*central):.3f} nm"
    )

    print(
        f"\nHD23 PAH mass fraction: "
        f"{m_pah.sum()/hd_total:.3%}"
    )

    print("\nSaved:")
    print(f"  {png}")
    print(f"  {pdf}")


if __name__ == "__main__":
    main()
