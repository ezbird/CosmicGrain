#!/usr/bin/env python3
"""
plot_radial_dgr.py
------------------
Publication-quality radial dust-to-gas (D/G) and dust-to-metals (D/Z)
profiles for the CosmicGrain simulation ladder at z=0.

Two panels stacked vertically (D/G top, D/Z bottom), sharing the x-axis.
Profiles are computed in 5 kpc spherical shells in physical kpc.

Halo identification uses halo_utils.get_halo569_reference / get_halo569,
consistent with all other CosmicGrain analysis scripts.

Unit conventions:
  Positions  : comoving kpc/h  →  physical kpc  via  * a / h
  h          : from f["Parameters"].attrs["HubbleParam"]  (NOT Header)
  R200       : Group_R_Crit200 in ckpc/h, converted to pkpc via * a / h

Usage:
    python plot_radial_dgr.py --res 1024
    python plot_radial_dgr.py --res 1024 --runs S0 S4 S10
    python plot_radial_dgr.py --res 1024 --r-max 50
    python plot_radial_dgr.py --res 1024 --output myplot.png
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from datetime import datetime

from halo_utils import (
    get_halo569_reference,
    get_halo569,
    read_snap_header,
    glob_snap_chunks,
)
plt.style.use('sleek.mplstyle')

# ─────────────────────────────────────────────────────────────────────────────
# Run styling
# ─────────────────────────────────────────────────────────────────────────────
RUN_CONFIGS = {
    "S0":  {"label": "S0: Creation only",           "color": "#888888", "marker": "o"},
    "S1":  {"label": "S1: + Cooling",               "color": "#1f77b4", "marker": "s"},
    "S2":  {"label": "S2: + Drag",                  "color": "#ff7f0e", "marker": "^"},
    "S3":  {"label": "S3: + Astration",             "color": "#2ca02c", "marker": "D"},
    "S4":  {"label": "S4: + Thermal sputtering",    "color": "#d62728", "marker": "v"},
    "S5":  {"label": "S5: + Grain growth",          "color": "#9467bd", "marker": "P"},
    "S6":  {"label": "S6: + Clumping factor",       "color": "#8c564b", "marker": "X"},
    "S7":  {"label": "S7: + SN shock destruction",  "color": "#e377c2", "marker": "<"},
    "S8":  {"label": "S8: + Coagulation",           "color": "#17becf", "marker": ">"},
    "S9":  {"label": "S9: + Shattering",            "color": "#bcbd22", "marker": "h"},
    "S10": {"label": "S10: + Rad. pressure (full)", "color": "#1f9e89", "marker": "*"},
}

FIGDIR        = "dust_figures"
RESOLUTION    = 512
R_MAX_DEFAULT = 50.0   # physical kpc
BIN_WIDTH     = 5.0    # physical kpc

# Radial region guides for the top of the figure.
# These are deliberately simple visual guides, not hard physical cuts.
REGION_BOUNDS_PKPC = [5.0, 15.0, 25.0]
REGION_LABELS = [
    "Inner galaxy",
    "Stellar disk",
    "Outer disk",
    "Disk-Halo interface",
]

os.makedirs(FIGDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reference curves
# ─────────────────────────────────────────────────────────────────────────────

def z_relative(r_kpc, r0=8.0, grad=-0.04):
    return 10**(grad * (r_kpc - r0))

def remy_ruyer_dgr(r_kpc, r0=8.0, dgr0=0.01, grad=-0.04, alpha=1.5):
    return dgr0 * z_relative(r_kpc, r0=r0, grad=grad)**alpha

def remy_ruyer_dtz(r_kpc, r0=8.0, dtz0=0.5, grad=-0.04, alpha=1.5):
    return dtz0 * z_relative(r_kpc, r0=r0, grad=grad)**(alpha - 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run):
    """Return sorted list of snapshot base paths for this run/resolution."""
    output_dir = f"../{run}_output_{RESOLUTION}"
    if not os.path.isdir(output_dir):
        return []
    seen, bases = set(), []
    for snapdir in sorted(glob.glob(os.path.join(output_dir, "snapdir_*"))):
        for f in sorted(glob.glob(os.path.join(snapdir, "snapshot_*.0.hdf5"))):
            base = re.sub(r"\.0\.hdf5$", "", f)
            if base not in seen:
                seen.add(base); bases.append(base)
        for f in sorted(glob.glob(os.path.join(snapdir, "snapshot_*.hdf5"))):
            if ".0.hdf5" in f: continue
            base = re.sub(r"\.hdf5$", "", f)
            if base not in seen:
                seen.add(base); bases.append(base)
    return sorted(bases)


def snap_redshift(snap_base):
    import h5py
    for suffix in [".0.hdf5", ".hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
                    z = hf["Header"].attrs.get("Redshift", None)
                    if z is not None:
                        return float(z)
            except Exception:
                pass
    return None


def find_snap_near_z(snap_bases, target_z):
    best, best_dz = None, 1e30
    for sb in snap_bases:
        z = snap_redshift(sb)
        if z is not None and abs(z - target_z) < best_dz:
            best_dz = abs(z - target_z)
            best    = sb
    return best, best_dz


# ─────────────────────────────────────────────────────────────────────────────
# Halo center and R200 via halo_utils  (corrected)
# ─────────────────────────────────────────────────────────────────────────────

def get_halo_center_r200(run, snap_base):
    """
    Return (center_ckpch, r200_pkpc) for Halo 569 at the given snapshot.

    Uses halo_utils.get_halo569_reference (stellar-mass argmax across ALL
    catalog chunks) and get_halo569 (position-tracked) — consistent with
    snap_overview, run_radial_evolution, and plot_radial_dust_analysis.

    Returns (None, None) if the catalog is missing or empty.
    """
    # Derive snap_num from snap_base path
    m = re.search(r"snapshot_(\d+)$", snap_base)
    if not m:
        print(f"  [get_halo_center_r200] cannot parse snap_num from {snap_base}")
        return None, None
    snap_num   = int(m.group(1))
    output_dir = f"../{run}_output_{RESOLUTION}"
    groups_dir = os.path.join(output_dir, f"groups_{snap_num:03d}")

    # Establish z=0 reference once (cached per run via module-level dict)
    ref = _get_ref(run, output_dir, refine_center=False)
    if ref is None:
        return None, None

    halo = get_halo569(
        groups_dir, snap_num, ref,
        verbose=False,
        refine_center=False,
    )
    if halo is None or halo["r200_ckpch"] <= 0:
        return None, None

    return halo["center"], halo["r200_pkpc"]


# Module-level cache so get_halo569_reference is called once per run
_ref_cache = {}

def _get_ref(run, output_dir, refine_center=True):
    key = (run, refine_center)
    if key not in _ref_cache:
        try:
            _ref_cache[key] = get_halo569_reference(
                output_dir,
                refine_center=refine_center,
            )
        except Exception as e:
            print(f"  [{run}] get_halo569_reference failed: {e}")
            _ref_cache[key] = None
    return _ref_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Particle loader
# ─────────────────────────────────────────────────────────────────────────────

def load_particles(snap_base, ctr_ckpch, rmax_ckpch, a, h, ptype, fields):
    """
    Load PartType{ptype} within rmax_ckpch (comoving kpc/h) of ctr_ckpch.

    Coordinates are returned in PHYSICAL kpc (converted via * a / h).
    Masses are returned in code units (1e10 Msun/h) — caller multiplies
    by 1e10/h if needed.
    """
    import h5py
    key    = f"PartType{ptype}"
    result = {f: [] for f in fields}
    result["pos_pkpc"] = []

    files = sorted(glob.glob(snap_base + ".*.hdf5"))
    if not files:
        single = snap_base + ".hdf5"
        files  = [single] if os.path.exists(single) else []

    box_ckpch = None

    for fname in files:
        try:
            with h5py.File(fname, "r") as hf:
                if box_ckpch is None:
                    box_ckpch = float(hf["Header"].attrs["BoxSize"])
                if key not in hf:
                    continue
                pt     = hf[key]
                coords = pt["Coordinates"][:]   # ckpc/h

                # Periodic distance in ckpc/h
                dx   = coords - ctr_ckpch[None, :]
                dx  -= box_ckpch * np.round(dx / box_ckpch)
                r    = np.sqrt((dx**2).sum(axis=1))
                mask = r < rmax_ckpch
                if not mask.any():
                    continue

                # Store positions in physical kpc
                result["pos_pkpc"].append(coords[mask] * a / h)
                for f in fields:
                    if f in pt:
                        result[f].append(pt[f][:][mask])
        except Exception as e:
            print(f"  load_particles(type={ptype}): {e}")

    if not result["pos_pkpc"]:
        return None

    out = {"pos_pkpc": np.vstack(result["pos_pkpc"])}
    for f in fields:
        if result[f]:
            out[f] = (np.concatenate(result[f])
                      if result[f][0].ndim == 1
                      else np.vstack(result[f]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Radial profile computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_radial_profiles(snap_base, run, r_max_pkpc, r_bins_pkpc):
    """
    Bin-averaged D/G and D/Z in physical kpc shells.

    Returns (r_cen, dgr, dtz, r200_pkpc) or (None, None, None, None).
    """
    # Read h and a from Parameters (not Header) per Gadget-4 convention
    files = sorted(glob.glob(snap_base + ".*.hdf5"))
    if not files:
        files = [snap_base + ".hdf5"]
    import h5py
    h = a = None
    for fname in files:
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, "r") as hf:
            if h is None and "Parameters" in hf:
                h = float(hf["Parameters"].attrs["HubbleParam"])
            if a is None:
                a = float(hf["Header"].attrs["Time"])
        if h is not None and a is not None:
            break
    if h is None:
        h = 0.6774
        print(f"  [{run}] WARNING: HubbleParam not found, using {h}")

    ctr_ckpch, r200_pkpc = get_halo_center_r200(run, snap_base)
    if ctr_ckpch is None:
        return None, None, None, None

    # Search radius in ckpc/h = physical kpc * h / a
    rmax_ckpch = r_max_pkpc * h / a

    print(f"  [{run}] R200={r200_pkpc:.1f} pkpc  h={h:.4f}  a={a:.4f}  loading...")

    gas  = load_particles(snap_base, ctr_ckpch, rmax_ckpch, a, h,
                          0, ["Masses", "Metallicity"])
    dust = load_particles(snap_base, ctr_ckpch, rmax_ckpch, a, h,
                          6, ["Masses"])

    if gas is None or dust is None:
        print(f"  [{run}] missing gas or dust")
        return None, None, None, None

    # Radial distances in physical kpc from halo center
    ctr_pkpc = ctr_ckpch * a / h
    r_gas    = np.sqrt(((gas["pos_pkpc"]  - ctr_pkpc)**2).sum(axis=1))
    r_dust   = np.sqrt(((dust["pos_pkpc"] - ctr_pkpc)**2).sum(axis=1))

    # Bin masses into shells
    gas_m,   _ = np.histogram(r_gas,  bins=r_bins_pkpc, weights=gas["Masses"])
    dust_m,  _ = np.histogram(r_dust, bins=r_bins_pkpc, weights=dust["Masses"])

    # Metal mass = gas mass * metallicity
    Z = gas.get("Metallicity", None)
    if Z is not None:
        if Z.ndim == 2:
            Z = Z[:, 0]   # total metallicity (first element)
        metal_m, _ = np.histogram(r_gas, bins=r_bins_pkpc,
                                   weights=gas["Masses"] * Z)
    else:
        metal_m = np.zeros_like(gas_m)

    r_cen  = 0.5 * (r_bins_pkpc[:-1] + r_bins_pkpc[1:])

    # Exclude bins with negligible gas mass (< 1% of median)
    med_gm = np.nanmedian(gas_m[gas_m > 0]) if np.any(gas_m > 0) else 1.0
    good   = gas_m > 0.01 * med_gm

    with np.errstate(invalid="ignore", divide="ignore"):
        dgr = np.where(good & (gas_m   > 0), dust_m / gas_m,   np.nan)
        dtz = np.where(good & (metal_m > 0), dust_m / metal_m, np.nan)

    return r_cen, dgr, dtz, r200_pkpc


def add_radial_region_guides(ax_hdr, axes, r_max_pkpc):
    """
    Add the same region-header style used by run_radial_evolution.py:
    a thin horizontal rule above the labels, broad pale vertical boundary bands
    that continue through the header and both panels, and no darker center line.
    """
    bounds = [b for b in REGION_BOUNDS_PKPC if 0.0 < b < r_max_pkpc]
    edges = [0.0] + bounds + [r_max_pkpc]
    centers = [0.5 * (edges[i] + edges[i + 1])
               for i in range(len(edges) - 1)]

    # Match the companion plot: broad translucent boundary bands, no thin line.
    band_half_width = 0.45  # pkpc
    for ax in list(axes) + [ax_hdr]:
        for b in bounds:
            ax.axvspan(
                b - band_half_width,
                b + band_half_width,
                color="0.70",
                alpha=0.35,
                linewidth=0,
                zorder=1,
            )

    # Header row with labels, styled like the radial-evolution figure.
    ax_hdr.set_xlim(0.0, r_max_pkpc)
    ax_hdr.set_ylim(0.0, 1.0)
    ax_hdr.axis("off")
    ax_hdr.axhline(1.0, color="black", lw=0.8, zorder=10)

    for x, label in zip(centers, REGION_LABELS):
        ax_hdr.text(
            x, 0.45, label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="0.3",
            fontweight="normal",
            transform=ax_hdr.transData,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_radial_profiles(runs, r_max_pkpc=None, output_path=None):
    r_max  = r_max_pkpc if r_max_pkpc is not None else R_MAX_DEFAULT
    r_bins = np.arange(0, r_max + BIN_WIDTH, BIN_WIDTH)

    _style = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cosmicgrain.mplstyle")
    if os.path.exists(_style):
        plt.style.use(_style)

    fig = plt.figure(figsize=(9, 9))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.08, 2.0], hspace=0.0
    )
    ax_hdr = fig.add_subplot(outer[0])
    inner = outer[1].subgridspec(2, 1, hspace=0.06)
    ax_dgr = fig.add_subplot(inner[0])
    ax_dtz = fig.add_subplot(inner[1], sharex=ax_dgr)

    handles_runs = []

    for run in runs:
        cfg       = RUN_CONFIGS.get(run, {})
        color     = cfg.get("color", "black")
        label     = cfg.get("label", run)
        linestyle = "-" if run == "S10" else ":"
        lw        = 3.8 if run == "S10" else 1.5
        alpha     = 1.0 if run == "S10" else 0.72

        snaps = find_snapshots(run)
        if not snaps:
            print(f"  [{run}] no snapshots"); continue

        snap_base, dz = find_snap_near_z(snaps, 0.0)
        if dz > 0.2:
            print(f"  [{run}] no z~0 snap (dz={dz:.2f})"); continue

        r_cen, dgr, dtz, r200_p = compute_radial_profiles(
            snap_base, run, r_max, r_bins)
        if r_cen is None:
            continue

        marker = cfg.get("marker", "o")

        kw = dict(
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            marker=marker,
            markersize=9.0 if run != "S10" else 15.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9 if run != "S10" else 1.4,
            zorder=20 if run == "S10" else 5,
        )

        good_dgr = np.isfinite(dgr) & (dgr > 0)
        good_dtz = np.isfinite(dtz) & (dtz > 0)

        if good_dgr.any():
            ax_dgr.plot(r_cen[good_dgr], dgr[good_dgr], **kw, label=label)
        if good_dtz.any():
            ax_dtz.plot(r_cen[good_dtz], dtz[good_dtz], **kw)

        handles_runs.append(
            plt.Line2D(
                [0], [0],
                color=color,
                lw=3.0 if run == "S10" else 1.8,
                marker=marker,
                markersize=9.0 if run != "S10" else 14.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9 if run != "S10" else 1.4,
                label=label,
            )
        )

    # Reference curves
    r_ref = np.linspace(0.5, r_max, 300)

    dgr_mw_line, = ax_dgr.plot(r_ref, remy_ruyer_dgr(r_ref),
                                color="black", lw=2.0, ls="--", zorder=5)
    #handles_runs.append(
    #    plt.Line2D([0], [0], color="black", lw=2.0, ls="--",
    #               label="MW/M31 gradient (Rémy-Ruyer et al. 2014)"))

    dtz_ref_line, = ax_dtz.plot(r_ref, remy_ruyer_dtz(r_ref),
                                 color="black", lw=2.0, ls="--", zorder=4)
    ax_dtz.legend(handles=[dtz_ref_line],
                  labels=["MW/M31 gradient (Rémy-Ruyer et al. 2014)"],
                  fontsize=8, loc="upper right",
                  framealpha=0.9, edgecolor="0.8")

    # Axes: D/G
    ax_dgr.set_yscale("log")
    ax_dgr.set_ylim(3e-5, 5e-1)
    ax_dgr.set_ylabel("Dust-to-Gas Ratio (D/G)", fontsize=12)
    ax_dgr.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax_dgr.yaxis.set_major_formatter(
        matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax_dgr.grid(True, which="major", axis="y", color="0.88", lw=0.5, alpha=0.75)
    ax_dgr.grid(False, which="minor", axis="both")

    # Axes: D/Z
    ax_dtz.set_yscale("log")
    ax_dtz.set_ylim(3e-3, 20)
    ax_dtz.set_ylabel("Dust-to-Metals Ratio (D/Z)", fontsize=12)
    ax_dtz.set_xlabel("Galactocentric Radius (kpc)", fontsize=12)
    ax_dtz.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax_dtz.yaxis.set_major_formatter(
        matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax_dtz.grid(True, which="major", axis="y", color="0.88", lw=0.5, alpha=0.75)
    ax_dtz.grid(False, which="minor", axis="both")

    ax_dtz.set_xlim(0, r_max)
    ax_dtz.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax_dtz.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    plt.setp(ax_dgr.get_xticklabels(), visible=False)

    # Radial region guides: header plus boundary bands through both panels.
    add_radial_region_guides(ax_hdr, (ax_dgr, ax_dtz), r_max)

    ax_dgr.legend(handles=handles_runs, fontsize=7.5, loc="upper right",
                  framealpha=0.9, edgecolor="0.8", ncol=1,
                  handlelength=2.2, labelspacing=0.35, borderpad=0.6)

    # Manual spacing avoids tight_layout warnings and keeps the header aligned.
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.09, top=0.94)

    today = datetime.now().strftime("%-m-%-d-%y")
    out = output_path or os.path.join(
        FIGDIR,
        f"radial_dg_dz_{RESOLUTION}_{today}.pdf"
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+",
                        default=["S0","S1","S2","S3","S4","S5",
                                 "S6","S7","S8","S9","S10"])
    parser.add_argument("--res",    type=int,   default=512)
    parser.add_argument("--r-max",  type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    global RESOLUTION
    RESOLUTION = args.res

    print(f"\nRuns:       {args.runs}")
    print(f"Resolution: {RESOLUTION}^3")
    print(f"r_max:      {args.r_max or R_MAX_DEFAULT} pkpc")
    print(f"Bin width:  {BIN_WIDTH} pkpc\n")

    plot_radial_profiles(args.runs, r_max_pkpc=args.r_max,
                         output_path=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
