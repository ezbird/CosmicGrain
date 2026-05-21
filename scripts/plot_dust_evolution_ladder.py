#!/usr/bin/env python3
"""
plot_dust_evolution_ladder.py
------------------------------
Same three-panel dust evolution figure as plot_dust_evolution.py, but
tuned for overlaying all 11 physics-ladder rungs (S0–S10) on one plot.

Changes from the original:
  - 11-color sequential palette (viridis-like, perceptually uniform)
  - M_star plotted only for S0 as a single grey reference line (not 11
    dashed lines)
  - Compact two-column legend in Panel 1
  - --no-mstar flag to suppress even the reference line
  - --linthresh flag to tune the symlog linear region

Usage:
    python plot_dust_evolution_ladder.py \
        ../S0_output_1024/ ../S1_output_1024/ ../S2_output_1024/ ../S3_output_1024/ ../S4_output_1024/ ../S5_output_1024/ ../S6_output_1024/ ../S7_output_1024/ ../S8_output_1024/ ../S9_output_1024/ ../S10_output_1024/ \
        --labels S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
        --output dust_ladder_1024.png
"""

import sys
import os
import re
import glob
import argparse
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.integrate import quad

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SOLAR_MASS = 1.989e33   # g
YR_IN_SEC  = 3.156e7    # s
GYR_IN_SEC = 3.156e16   # s

# 11 perceptually distinct colours for S0–S10.
# Sampled from matplotlib's tab20 + a few manual tweaks so adjacent rungs
# are distinguishable even in greyscale (alternating light/dark).
LADDER_COLORS = [
    "#1f77b4",   # S0  blue
    "#aec7e8",   # S1  light blue
    "#2ca02c",   # S2  green
    "#98df8a",   # S3  light green
    "#d62728",   # S4  red
    "#ff9896",   # S5  light red/salmon
    "#9467bd",   # S6  purple
    "#8c564b",   # S7  brown
    "#e377c2",   # S8  pink
    "#bcbd22",   # S9  yellow-green
    "#17becf",   # S10 cyan
]

# ─────────────────────────────────────────────────────────────────────────────
# Cosmology
# ─────────────────────────────────────────────────────────────────────────────

def z_to_age_gyr(z, h=0.6774, Om=0.3089):
    H0 = h * 100.0 * 1e3 / 3.0857e22
    age, _ = quad(lambda zp: 1.0 / ((1+zp)*np.sqrt(Om*(1+zp)**3+(1-Om))),
                  z, np.inf)
    return age / H0 / GYR_IN_SEC


def z_arr_to_age(z_arr):
    return np.array([z_to_age_gyr(z) for z in z_arr])


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(output_dir):
    output_dir = Path(output_dir)
    entries = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        catalog_files = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) \
                        if groups_dir.exists() else []
        if not catalog_files:
            continue
        snap_files = sorted(snapdir.glob("snap_*.hdf5")) + \
                     sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snap_files:
            continue
        entries.append((snap_num, snapdir, str(snap_files[0]), str(catalog_files[0])))
    return entries


def get_header(snap_file):
    with h5py.File(snap_file, "r") as f:
        hdr    = f["Header"].attrs
        params = f["Parameters"].attrs
        z   = float(hdr["Redshift"])
        h   = float(params["HubbleParam"])
        box = float(hdr["BoxSize"])
    return z, h, box


def get_unit_mass(snap_file):
    with h5py.File(snap_file, "r") as f:
        params = f.get("Parameters") or f.get("Config") or {}
        um = params.attrs.get("UnitMass_in_g", None) if params else None
        if um is None:
            um = 1.989e43
    return float(um)


def get_unit_time(snap_file):
    with h5py.File(snap_file, "r") as f:
        params = f.get("Parameters") or f.get("Config") or {}
        ut = params.attrs.get("UnitTime_in_s", None) if params else None
        if ut is None:
            ut = 3.0857e16
    return float(ut)


# ─────────────────────────────────────────────────────────────────────────────
# SubFind catalog reader
# ─────────────────────────────────────────────────────────────────────────────

def read_primary_group(catalog_file):
    p = Path(catalog_file)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    pos_list, r200_list, m200_list, sfr_list = [], [], [], []
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_list.append(grp["GroupPos"][:])
            r200_list.append(grp["Group_R_Crit200"][:])
            m200_list.append(grp["Group_M_Crit200"][:])
            sfr_list.append(grp["GroupSFR"][:] if "GroupSFR" in grp
                            else np.zeros(len(grp["GroupPos"])))

    if not pos_list:
        return None

    pos  = np.concatenate(pos_list,  axis=0)
    r200 = np.concatenate(r200_list, axis=0)
    m200 = np.concatenate(m200_list, axis=0)
    sfr  = np.concatenate(sfr_list,  axis=0)

    return dict(center=pos[0], r200=float(r200[0]),
                m200=float(m200[0]), sfr_code=float(sfr[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Particle mass reader
# ─────────────────────────────────────────────────────────────────────────────

def masses_within_r200(snap_file_first, center_kph, r200_kph, part_types=(4, 6)):
    p = Path(snap_file_first)
    stem = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem}*.hdf5"))
    if not chunks:
        chunks = [p]

    result = {pt: [] for pt in part_types}
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            box = float(f["Header"].attrs["BoxSize"])
            mt  = f["Header"].attrs.get("MassTable", None)
            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                coords = f[key]["Coordinates"][:]
                if "Masses" in f[key]:
                    masses = f[key]["Masses"][:]
                elif mt is not None and mt[pt] > 0:
                    masses = np.full(len(coords), mt[pt])
                else:
                    continue
                dx = coords - center_kph
                dx -= box * np.round(dx / box)
                r  = np.sqrt((dx**2).sum(axis=1))
                result[pt].append(masses[r <= r200_kph])

    return {pt: np.concatenate(v) if v else np.array([])
            for pt, v in result.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot
# ─────────────────────────────────────────────────────────────────────────────

def process_snapshot(snap_num, snap_file, catalog_file, unit_mass_g, unit_time_s):
    z, h, box = get_header(snap_file)
    grp = read_primary_group(catalog_file)
    if grp is None or grp["r200"] <= 0:
        return None

    parts = masses_within_r200(snap_file, grp["center"], grp["r200"],
                               part_types=(4, 6))
    code_to_msun = unit_mass_g / SOLAR_MASS
    m_star_msun  = parts[4].sum() * code_to_msun if len(parts[4]) else 0.0
    m_dust_msun  = parts[6].sum() * code_to_msun if len(parts[6]) else 0.0
    sfr_msun_yr  = grp["sfr_code"] * code_to_msun / (unit_time_s / YR_IN_SEC)

    if m_star_msun <= 0:
        return None

    print(f"  snap {snap_num:03d}  z={z:.3f}  "
          f"logMs={np.log10(m_star_msun):.2f}  "
          f"logMd={np.log10(max(m_dust_msun, 1)):.2f}  "
          f"SFR={sfr_msun_yr:.2f} M☉/yr")

    return dict(z=z, m_star=m_star_msun, m_dust=m_dust_msun, sfr=sfr_msun_yr)


def run_simulation(output_dir, skip_every=1):
    snaps = find_snapshots(output_dir)
    if not snaps:
        raise RuntimeError(f"No snapshots with SubFind catalogs in {output_dir}")
    print(f"  Found {len(snaps)} snapshots with catalogs")

    unit_mass_g = get_unit_mass(snaps[0][2])
    unit_time_s = get_unit_time(snaps[0][2])

    rows = []
    for i, (snap_num, _, snap_file, catalog_file) in enumerate(snaps):
        if i % skip_every != 0:
            continue
        r = process_snapshot(snap_num, snap_file, catalog_file,
                             unit_mass_g, unit_time_s)
        if r is not None:
            rows.append(r)

    if not rows:
        raise RuntimeError("No valid snapshots")

    rows.sort(key=lambda r: -r["z"])
    z      = np.array([r["z"]      for r in rows])
    m_star = np.array([r["m_star"] for r in rows])
    m_dust = np.array([r["m_dust"] for r in rows])
    sfr    = np.array([r["sfr"]    for r in rows])
    return z, m_star, m_dust, sfr


# ─────────────────────────────────────────────────────────────────────────────
# Net dust rate
# ─────────────────────────────────────────────────────────────────────────────

def net_dust_rate(z_arr, m_dust_arr):
    age_yr = z_arr_to_age(z_arr) * 1e9
    rate   = np.empty_like(m_dust_arr)
    rate[1:-1] = (m_dust_arr[2:] - m_dust_arr[:-2]) / (age_yr[2:] - age_yr[:-2])
    rate[0]    = (m_dust_arr[1]  - m_dust_arr[0])   / (age_yr[1]  - age_yr[0])
    rate[-1]   = (m_dust_arr[-1] - m_dust_arr[-2])  / (age_yr[-1] - age_yr[-2])
    return rate


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(runs, output_path, linthresh=0.05, show_mstar=True):
    """
    runs: list of (label, z, m_star, m_dust, sfr)
    show_mstar: if True, draws M_star from the FIRST run as a grey reference.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10),
                             sharex=True,
                             gridspec_kw={"hspace": 0.08})
    ax_dust, ax_sfr, ax_rate = axes

    z_ticks = np.array([0, 0.5, 1, 2, 3, 4, 5, 6])
    z_max   = max(r[1].max() for r in runs)
    x_max   = min(z_max + 0.3, 7.0)

    # Grey M_star reference (S0 only, so the panel isn't 11 dashed lines)
    if show_mstar and runs:
        _, z0, m_star0, _, _ = runs[0]
        ax_dust.semilogy(z0, m_star0, color="0.65", lw=1.4, ls=":",
                         zorder=1, label=r"$M_\star$ (S0 ref.)")

    legend_handles = []
    for i, (label, z, m_star, m_dust, sfr) in enumerate(runs):
        c    = LADDER_COLORS[i % len(LADDER_COLORS)]
        rate = net_dust_rate(z, m_dust)
        lw   = 1.8

        # Panel 1: M_dust
        line, = ax_dust.semilogy(z, m_dust, color=c, lw=lw, zorder=2+i)
        legend_handles.append(Line2D([0], [0], color=c, lw=lw, label=label))

        # Panel 2: SFR
        sfr_plot = np.where(sfr > 0, sfr, np.nan)
        ax_sfr.semilogy(z, sfr_plot, color=c, lw=lw)

        # Panel 3: net dust rate
        ax_rate.plot(z, rate, color=c, lw=lw)
        ax_rate.fill_between(z, rate, 0,
                             where=rate >= 0, color=c, alpha=0.10, zorder=1)

    ax_rate.axhline(0, color="0.45", lw=0.8, ls="--", zorder=5)

    # Symlog y-scale for rate panel
    ax_rate.set_yscale("symlog", linthresh=linthresh, linscale=0.3)
    pos_ticks = [0.05, 0.1, 0.5, 1, 10]
    neg_ticks = [-t for t in pos_ticks]
    ax_rate.set_yticks(sorted(neg_ticks + [0] + pos_ticks))
    ax_rate.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: (f"${x:.0f}$"      if abs(x) >= 1
                      else f"${x:.2f}$" if abs(x) >= 0.01
                      else "") if x != 0 else "$0$"))

    # ── Shared x-axis formatting ──────────────────────────────────────────────
    for ax in axes:
        ax.set_xlim(x_max, 0)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, which="major", color="0.88", lw=0.5)
        ax.grid(True, which="minor", color="0.94", lw=0.3)
        ax.tick_params(labelsize=9)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax_dust.set_ylabel(r"Mass ($\mathrm{M}_\odot$)", fontsize=10)
    ax_dust.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"$10^{{{int(np.log10(x))}}}$" if x > 0 else ""))

    ax_sfr.set_ylabel(r"SFR ($\mathrm{M}_\odot\,\mathrm{yr}^{-1}$)", fontsize=10)

    ax_rate.set_ylabel(
        r"$\dot{M}_\mathrm{dust}$  ($\mathrm{M}_\odot\,\mathrm{yr}^{-1}$)",
        fontsize=10)
    ax_rate.set_xlabel("Redshift $z$", fontsize=10)
    ax_rate.text(0.97, 0.90, "net growth",       transform=ax_rate.transAxes,
                 fontsize=8, color="0.35", ha="right", style="italic")
    ax_rate.text(0.97, 0.06, "net destruction",  transform=ax_rate.transAxes,
                 fontsize=8, color="0.35", ha="right", style="italic")

    # ── Lookback-time twin axis on top ────────────────────────────────────────
    ax2 = ax_dust.twiny()
    ax2.set_xlim(ax_dust.get_xlim())
    valid     = z_ticks[z_ticks <= x_max]
    age_ticks = [f"{z_to_age_gyr(z):.1f}" for z in valid]
    ax2.set_xticks(valid)
    ax2.set_xticklabels(age_ticks, fontsize=8)
    ax2.set_xlabel("Lookback time (Gyr)", fontsize=9)
    ax2.invert_xaxis()

    # ── Legend: two columns in Panel 1, compact ───────────────────────────────
    if show_mstar:
        mstar_handle = Line2D([0], [0], color="0.65", lw=1.4, ls=":",
                              label=r"$M_\star$ (S0 ref.)")
        legend_handles = [mstar_handle] + legend_handles

    ax_dust.legend(handles=legend_handles,
                   fontsize=7.5, ncol=2,
                   loc="lower right",
                   framealpha=0.90,
                   labelspacing=0.25,
                   handlelength=1.5,
                   columnspacing=0.8)

    # Panel annotations
    for ax, lbl in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.93, lbl, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-panel dust-evolution ladder figure (S0–S10 overlaid)")
    parser.add_argument("output_dirs", nargs="+",
                        help="Gadget-4 output directories, one per rung")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels (default: S0, S1, …)")
    parser.add_argument("--skip-every", type=int, default=1,
                        help="Use every N-th snapshot (speeds up loading)")
    parser.add_argument("--output", default="dust_ladder_evolution.png",
                        help="Output PNG filename")
    parser.add_argument("--linthresh", type=float, default=0.05,
                        help="Symlog linear threshold for rate panel (M☉/yr)")
    parser.add_argument("--no-mstar", action="store_true",
                        help="Suppress the M_star reference line")
    args = parser.parse_args()

    n = len(args.output_dirs)

    # Default labels: S0, S1, …, S{n-1}
    if args.labels:
        labels = args.labels
        if len(labels) != n:
            parser.error("--labels count must match number of output_dirs")
    else:
        labels = [f"S{i}" for i in range(n)]

    runs = []
    for d, lbl in zip(args.output_dirs, labels):
        print(f"\nProcessing: {lbl}  ({d})")
        z, m_star, m_dust, sfr = run_simulation(d, skip_every=args.skip_every)
        runs.append((lbl, z, m_star, m_dust, sfr))

    make_plot(runs, args.output,
              linthresh=args.linthresh,
              show_mstar=not args.no_mstar)


if __name__ == "__main__":
    main()
