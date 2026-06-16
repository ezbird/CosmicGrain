#!/usr/bin/env python3
"""
plot_dust_histograms.py
-----------------------
Grid of histograms showing dust particle properties for Halo 569.

Uses halo_utils.get_halo569_reference / get_halo569 for correct halo
identification — consistent with all other CosmicGrain analysis scripts.

Unit conventions:
  Coordinates : comoving kpc/h  →  physical kpc via * a / h
  Masses      : 1e10 Msun/h     →  Msun via * 1e10 / h
  GrainRadius : already in nm in HDF5 (snap_io applies cm->nm on write)
  h           : from f["Parameters"].attrs["HubbleParam"]  (NOT Header)

Usage:
    python plot_dust_histograms.py ../S10_output_1024/
    python plot_dust_histograms.py ../S10_output_2048/ --snap 049
    python plot_dust_histograms.py ../S10_output_1024/ --out dust_hist_1024.png
    python plot_dust_histograms.py ../S10_output_1024/ --rmax 50
"""

import argparse
import glob
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path

from halo_utils import get_halo569_reference, get_halo569

# Load shared paper style
_STYLE = Path(__file__).parent / "cosmicgrain.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))

MSUN_PER_CODE = 1e10   # Gadget default: 1 code unit = 1e10 Msun/h

# ─────────────────────────────────────────────────────────────────────────────
# Header / cosmology helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_snap_meta(snap_files):
    """Read h (from Parameters), a, z, box, cosmology from first chunk."""
    with h5py.File(snap_files[0], "r") as f:
        h   = float(f["Parameters"].attrs["HubbleParam"])
        a   = float(f["Header"].attrs["Time"])
        z   = float(f["Header"].attrs["Redshift"])
        box = float(f["Header"].attrs["BoxSize"])
        Om  = float(f["Header"].attrs.get("Omega0",      0.3158))
        OL  = float(f["Header"].attrs.get("OmegaLambda", 0.6842))
    return dict(h=h, a=a, z=z, box=box, Om=Om, OL=OL)


def age_of_universe_gyr(a, h, Om, OL):
    """Age of universe at scale factor a in Gyr (flat ΛCDM)."""
    H0    = (100.0 * h) / 3.085678e19   # s^-1
    N     = 2000
    la0   = math.log(1e-8)
    la1   = math.log(a)
    acc   = sum(
        1.0 / math.sqrt(Om / math.exp(la0 + (i+0.5)*(la1-la0)/N)**3 + OL)
        for i in range(N)
    )
    t_s = acc * (la1 - la0) / N / H0
    return t_s / (3600 * 24 * 365.25 * 1e9)


# ─────────────────────────────────────────────────────────────────────────────
# Halo identification
# ─────────────────────────────────────────────────────────────────────────────

def get_center_r200(output_dir, snap_num_str):
    """
    Return (center_ckpch, r200_pkpc) for Halo 569 via halo_utils.
    Consistent with snap_overview, plot_gsd_comparison, etc.
    """
    snap_num   = int(snap_num_str)
    groups_dir = Path(output_dir) / f"groups_{snap_num:03d}"

    ref  = get_halo569_reference(output_dir)
    halo = get_halo569(groups_dir, snap_num, ref, verbose=True)
    if halo is None or halo["r200_ckpch"] <= 0:
        raise RuntimeError(f"No valid halo for snap {snap_num}")

    print(f"  Center (ckpc/h)  : [{halo['center'][0]:.1f}, "
          f"{halo['center'][1]:.1f}, {halo['center'][2]:.1f}]")
    print(f"  R_Crit200        : {halo['r200_ckpch']:.1f} ckpc/h  "
          f"({halo['r200_pkpc']:.1f} pkpc)")
    print(f"  M_Crit200        : {halo['m200_code']*MSUN_PER_CODE/ref['h']:.3e} Msun")

    return halo["center"], halo["r200_pkpc"]


# ─────────────────────────────────────────────────────────────────────────────
# Dust particle loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dust(snap_files, ctr_ckpch, rmax_ckpch, box_ckpch, h, a):
    """
    Load PartType6 within rmax_ckpch of ctr_ckpch.
    Returns dict of arrays with masses in Msun, positions in pkpc.
    """
    fields_wanted = ["Masses", "GrainRadius", "CarbonFraction",
                     "Velocities", "DustTemperature", "DustFormationTime"]

    buffers = {f: [] for f in fields_wanted}
    buffers["pos_pkpc"] = []
    n_total = len(snap_files)

    for idx, fpath in enumerate(snap_files):
        with h5py.File(fpath, "r") as f:
            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]
            if len(pt6["Masses"]) == 0:
                continue

            coords = pt6["Coordinates"][:]
            dx     = coords - ctr_ckpch[None, :]
            dx    -= box_ckpch * np.round(dx / box_ckpch)
            r      = np.sqrt((dx**2).sum(axis=1))
            mask   = r < rmax_ckpch
            if not mask.any():
                continue

            buffers["pos_pkpc"].append(coords[mask] * a / h)
            for field in fields_wanted:
                if field in pt6:
                    arr = pt6[field][:]
                    buffers[field].append(
                        arr[mask] if arr.ndim == 1 else arr[mask])

        if (idx + 1) % 20 == 0 or (idx + 1) == n_total:
            print(f"    chunk {idx+1}/{n_total}", end="\r", flush=True)
    print()

    if not buffers["pos_pkpc"]:
        return None

    result = {"pos_pkpc": np.vstack(buffers["pos_pkpc"])}
    for field in fields_wanted:
        if buffers[field]:
            arr = (np.concatenate(buffers[field])
                   if buffers[field][0].ndim == 1
                   else np.vstack(buffers[field]))
            result[field] = arr

    # Convert masses to Msun
    if "Masses" in result:
        result["Masses"] = result["Masses"] * MSUN_PER_CODE / h

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Histogram helper
# ─────────────────────────────────────────────────────────────────────────────

def make_histogram(ax, data, xlabel, title, bins=50,
                   log_x=False, color="steelblue",
                   xlim=None):
    data = data[np.isfinite(data)]
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return

    if log_x:
        data = data[data > 0]
        if len(data) == 0:
            ax.text(0.5, 0.5, "No positive data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        # Span the full xlim range so all bins are visible even if
        # data clusters tightly (e.g. narrow mass range at injection)
        lo = xlim[0] if xlim is not None else data.min()
        hi = xlim[1] if xlim is not None else data.max()
        bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins)
    else:
        bin_edges = bins

    ax.hist(data, bins=bin_edges, color=color, alpha=0.75,
            edgecolor="white", linewidth=0.3)

    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.5)

    stats = (
             f"Median = {np.median(data):.2e}\n"
             f"Mean = {np.mean(data):.2e}")
    ax.text(0.04, 0.97, stats, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dust particle property histograms for Halo 569")
    parser.add_argument("output_dir",
                        help="Gadget-4 output directory (e.g. ../S10_output_1024/)")
    parser.add_argument("--snap",    default="049",
                        help="Snapshot number string (default: 049)")
    parser.add_argument("--rmax",    type=float, default=None,
                        help="Extraction radius in pkpc (default: R200)")
    parser.add_argument("--out",     default=None,
                        help="Output PNG (default: auto-named)")
    parser.add_argument("--bins",    type=int,   default=50)
    parser.add_argument("--dpi",     type=int,   default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    snap_num   = args.snap.zfill(3)
    snap_dir   = output_dir / f"snapdir_{snap_num}"
    run_label  = output_dir.name

    out_path = Path(args.out) if args.out else \
               Path(f"dust_histograms_{run_label}_snap{snap_num}.png")

    print("=" * 60)
    print(f"Dust Histograms  |  {run_label}  |  snap {snap_num}")
    print("=" * 60)

    snap_base  = str(snap_dir / f"snapshot_{snap_num}")
    snap_files = sorted(glob.glob(snap_base + "*.hdf5"))
    if not snap_files:
        sys.exit(f"ERROR: no snapshot files at {snap_base}*.hdf5")
    print(f"Snapshot chunks: {len(snap_files)}")

    meta = read_snap_meta(snap_files)
    h, a, z = meta["h"], meta["a"], meta["z"]
    box     = meta["box"]
    print(f"h={h:.4f}  a={a:.6f}  z={z:.4f}  box={box:.1f} ckpc/h")

    age_now = age_of_universe_gyr(a, h, meta["Om"], meta["OL"])
    print(f"Age of universe at z={z:.3f}: {age_now:.3f} Gyr")

    print("\nLocating Halo 569 ...")
    ctr_ckpch, r200_pkpc = get_center_r200(str(output_dir), snap_num)

    rmax_pkpc  = args.rmax if args.rmax is not None else r200_pkpc
    rmax_ckpch = rmax_pkpc * h / a
    print(f"\nExtraction radius: {rmax_pkpc:.1f} pkpc  "
          f"({rmax_ckpch:.1f} ckpc/h)")

    print("\nLoading PartType6 ...")
    dust = load_dust(snap_files, ctr_ckpch, rmax_ckpch, box, h, a)
    if dust is None:
        sys.exit("ERROR: no dust particles found within aperture")

    n_dust = len(dust["Masses"])
    print(f"  Loaded {n_dust:,} dust particles")
    print(f"  GrainRadius range: "
          f"{dust['GrainRadius'].min():.2f} -- "
          f"{dust['GrainRadius'].max():.2f} nm")

    # Dust age from DustFormationTime (scale factor at injection)
    dust_age_gyr = np.zeros(n_dust)
    if "DustFormationTime" in dust:
        a_form = dust["DustFormationTime"]
        valid  = (a_form > 0) & (a_form <= a)
        print(f"  DustFormationTime: {valid.sum():,} valid "
              f"/ {n_dust:,} total")
        for i in np.where(valid)[0]:
            age_at_form        = age_of_universe_gyr(
                float(a_form[i]), h, meta["Om"], meta["OL"])
            dust_age_gyr[i]    = age_now - age_at_form
    else:
        print("  DustFormationTime not found — age set to zero")

    # Velocity magnitude
    vel_mag = np.linalg.norm(dust["Velocities"], axis=1) \
              if "Velocities" in dust else np.zeros(n_dust)

    # ── Figure: 2x2 grid (grain radius, mass, velocity, age) ────────────
    fig = plt.figure(figsize=(10, 7))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30, top=0.86)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    make_histogram(ax1, dust["GrainRadius"],
                   "Radius (nm)", "Grain Radius",
                   bins=args.bins, color="steelblue")

    make_histogram(ax2, dust["Masses"],
                   r"Mass (M$_\odot$)", "Mass",
                   bins=args.bins, log_x=True, color="coral",
                   xlim=(1e1, 1e4))

    make_histogram(ax3, vel_mag,
                   "Velocity Magnitude (km/s)", "Velocity",
                   bins=args.bins, color="purple",
                   xlim=(0, 400))

    make_histogram(ax4, dust_age_gyr,
                   "Age (Gyr)", "Age",
                   bins=args.bins, color="darkorange")

    # Titles
    import re as _re
    _m = _re.search(r"(S\d+)_output_(\d+)", run_label)
    run_fmt = (f"{_m.group(1)} ${_m.group(2)}^3$" if _m else run_label)
    fig.text(0.5, 0.975, "Dust Properties",
             fontsize=18, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.935,
             (f"{run_fmt}  |  "
              r"$R_{200}$" + f" < {rmax_pkpc:.1f} kpc  |  "
              r"$N_{\mathrm{dust}}$" + f" = {n_dust:,}  |  "
              f"$z = {z:.3f}$"),
             fontsize=11, ha="center", va="top")

    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dust particles : {n_dust:,}")
    print(f"GrainRadius    : {dust['GrainRadius'].min():.2f} -- "
          f"{dust['GrainRadius'].max():.2f} nm  "
          f"(median {np.median(dust['GrainRadius']):.2f})")
    print(f"CarbonFraction : {dust['CarbonFraction'].min():.3f} -- "
          f"{dust['CarbonFraction'].max():.3f}  "
          f"(median {np.median(dust['CarbonFraction']):.3f})")
    print(f"Mass           : {dust['Masses'].min():.2e} -- "
          f"{dust['Masses'].max():.2e} Msun  "
          f"(median {np.median(dust['Masses']):.2e})")
    print(f"Velocity       : {vel_mag.min():.1f} -- "
          f"{vel_mag.max():.1f} km/s  "
          f"(median {np.median(vel_mag):.1f})")
    if "DustTemperature" in dust:
        T = dust["DustTemperature"]
        print(f"Temperature    : {T.min():.1f} -- {T.max():.1f} K  "
              f"(median {np.median(T):.1f})")
    print(f"Dust age       : {dust_age_gyr.min():.3f} -- "
          f"{dust_age_gyr.max():.3f} Gyr  "
          f"(median {np.median(dust_age_gyr):.3f})")


if __name__ == "__main__":
    main()
