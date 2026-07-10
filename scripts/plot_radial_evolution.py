#!/usr/bin/env python3
"""
run_radial_evolution_combined.py
--------------------------------
Single-script radial dust evolution figure for CosmicGrain / Halo 569.

This replaces the older two-step workflow:
  1. run_radial_evolution.py
  2. plot_radial_dust_analysis.py

It directly:
  - finds snapshots near requested redshifts,
  - identifies Halo 569 using halo_utils with the frozen FOF/catalog center,
  - uses halo_utils' spherical-overdensity/catalog-fallback R200,
  - reads dust particles from all snapshot chunks,
  - computes enclosed dust fraction as a function of r/R200,
  - saves per-epoch NPZ data,
  - makes the summary plot.

------------------------------------------------------------------------------
CHANGES vs. the previous version
------------------------------------------------------------------------------
1. All local halo-finding/catalog-discovery code has been removed. This
   script previously carried its own copies of catalog globbing, primary-halo
   selection, periodic deltas, and an R200 fallback estimator
   (_catalog_halo_fallback / _r200_from_m200_catalog / _select_primary_halo_idx
   / _periodic_delta). Those duplicated -- and could silently drift out of
   sync with -- the logic in halo_utils.py. In particular, the local catalog
   glob only ever checked "fof_subhalo_tab_*" and would find nothing for
   FOF-only runs (SUBFIND disabled), and the local R200 fallback used a
   simple M200-from-catalog analytic estimate rather than halo_utils'
   spherical-overdensity calculation.

   Both the halo identification (get_halo569 / get_halo569_reference) and the
   snapshot/catalog discovery (find_snapshots / find_last_snap_num) are now
   imported directly from halo_utils.py. halo_utils.py already tries
   "fof_subhalo_tab_*" first and falls back to "fof_tab_*" (FOF-only mode)
   everywhere it looks for a catalog, and its spherical-overdensity crossing
   finder is robust to small-N density noise near the halo center. Fixing
   the catalog-naming/SO issues once in halo_utils.py means every script
   that imports from it -- this one included -- benefits automatically,
   rather than needing the same fix reapplied in N different places.

2. Added --compare-dir, an optional second Gadget output directory (e.g. a
   different resolution of the same run) whose radial profiles are overlaid
   on the same summary figure as dashed lines, using the same color per
   redshift as the primary run's solid lines. Resolution labels for the
   legend are auto-detected from each directory name (looking for a
   "<digits>^3"-style resolution token such as "1024" or "2048"); if none is
   found the literal directory name is used instead.

Usage:
  python plot_radial_evolution.py ../S10_output_1024/ \
      --redshifts 0 0.5 1 2 3 \
      --rmax-factor 1.0 \
      --outdir radial_evolution \
      --summary dust_radial_evolution.pdf

  # Overlay a second resolution as dashed lines on the same plot:
  python plot_radial_evolution.py ../S10_output_1024/ \
      --compare-dir ../S10_output_2048/ \
      --redshifts 0 0.5 1 2 3 \
      --summary dust_radial_evolution_1024_vs_2048.pdf

  # Rebuild only from cached NPZ files:
  python run_radial_evolution.py ../S10_output_1024/ --summary-only
"""

import argparse
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use('cosmicgrain.mplstyle')

try:
    from halo_utils import (
        get_halo569_reference,
        get_halo569,
        find_last_snap_num,
    )
except ImportError as e:
    print("ERROR: This script requires halo_utils.py in the same directory or PYTHONPATH")
    raise e

MSUN_PER_CODE = 1e10  # Gadget default: 1 code mass unit = 1e10 Msun/h

SCRIPT_DIR = Path(__file__).resolve().parent

# Distinguish redshift curves by both color and marker for print/colorblind safety.
LINE_MARKERS = ["o", "s", "^", "D", "", "P", "X"]
MARK_EVERY = 5

# Per-output-dir cache of get_halo569_reference() results, so repeated calls
# across many redshifts for the same run only build the (relatively
# expensive) z=0 reference once.
_ref_cache = {}


# -----------------------------------------------------------------------------
# Snapshot discovery
# -----------------------------------------------------------------------------
# Snapshot/catalog discovery (find_snapshots-equivalent) and halo
# identification both now come straight from halo_utils.py -- see module
# docstring. The only thing kept local is matching a snapshot's redshift,
# since halo_utils doesn't need to read snapshot Redshift attributes itself
# for its own API (it's given a snap_num and reads what it needs internally).

def find_all_snapshots(output_dir):
    """
    Return sorted list of dicts (one per snapshot) with snap_num, snap_file_0,
    groups_dir, snapdir, and z, for every snapshot that has both snapshot
    chunks and a usable FOF/FOF+Subfind catalog.

    Catalog presence is delegated to halo_utils.find_last_snap_num's sibling
    check inline below (same "fof_subhalo_tab_* then fof_tab_*" fallback
    halo_utils uses elsewhere), so FOF-only runs are picked up the same way
    they are throughout the rest of the pipeline.
    """
    output_dir = Path(output_dir)
    entries = []

    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))

        snap_files = sorted(snapdir.glob("snapshot_*.0.hdf5"))
        if not snap_files:
            snap_files = sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snap_files:
            snap_files = sorted(snapdir.glob("snap_*.0.hdf5"))
        if not snap_files:
            snap_files = sorted(snapdir.glob("snap_*.hdf5"))
        if not snap_files:
            continue

        groups_dir = output_dir / f"groups_{snap_num:03d}"
        if not groups_dir.exists():
            continue
        has_catalog = (
            list(groups_dir.glob("fof_subhalo_tab_*.hdf5")) or
            list(groups_dir.glob("fof_tab_*.hdf5"))
        )
        if not has_catalog:
            continue

        try:
            with h5py.File(str(snap_files[0]), "r") as f:
                z = float(f["Header"].attrs["Redshift"])
        except Exception:
            continue

        entries.append({
            "snap_num": snap_num,
            "snap_file_0": snap_files[0],
            "groups_dir": groups_dir,
            "snapdir": snapdir,
            "z": z,
        })

    return sorted(entries, key=lambda d: d["snap_num"])


def find_nearest_snapshot(entries, z_target):
    return min(entries, key=lambda d: abs(d["z"] - z_target))


def snapshot_chunks_from_file0(snap_file_0):
    """Return all chunks for a representative snapshot chunk path."""
    p = Path(snap_file_0)
    # Handles snapshot_047.0.hdf5 -> snapshot_047*.hdf5
    stem = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem}*.hdf5"))
    if not chunks and p.exists():
        chunks = [p]
    return chunks


# -----------------------------------------------------------------------------
# Halo identification (delegated entirely to halo_utils.py)
# -----------------------------------------------------------------------------

def get_halo569_frozen(output_dir, snap_num, verbose=False):
    """
    Return the halo_utils Halo 569 dictionary using the frozen FOF/catalog
    center plus particle-SO/catalog-fallback R200, for one snapshot.

    This is now a thin wrapper around halo_utils.get_halo569_reference /
    get_halo569 -- no local fallback logic. halo_utils.py's own
    spherical-overdensity crossing finder and Group_R_Crit200/Group_M_Crit200
    catalog fallback already handle the cases the old local
    _catalog_halo_fallback() existed for; keeping a second, less rigorous
    fallback here risked the two implementations disagreeing on R200 for the
    same snapshot.
    """
    output_dir = Path(output_dir)
    groups_dir = output_dir / f"groups_{snap_num:03d}"
    snapdir = output_dir / f"snapdir_{snap_num:03d}"

    if not groups_dir.exists() or not snapdir.exists():
        raise FileNotFoundError(f"Missing groups/snapdir for snap {snap_num:03d}")

    key = str(output_dir.resolve())
    if key not in _ref_cache:
        last_snap = find_last_snap_num(output_dir)
        if last_snap is None:
            raise RuntimeError(f"No valid z=0/reference snapshot found in {output_dir}")
        _ref_cache[key] = get_halo569_reference(
            output_dir,
            snap_num_z0=last_snap,
            refine_center=False,
            verbose=verbose,
        )

    ref = _ref_cache[key]

    halo = get_halo569(
        groups_dir,
        snap_num,
        ref,
        refine_center=False,
        verbose=verbose,
    )

    if halo is None or not np.isfinite(halo.get("r200_pkpc", 0.0)) or halo.get("r200_pkpc", 0.0) <= 1e-6:
        if halo is None:
            detail = "None"
        else:
            detail = f"r200_pkpc={halo.get('r200_pkpc')}"
        raise RuntimeError(
            f"Could not identify a valid Halo 569 R200 at snap {snap_num:03d} "
            f"(halo_utils.get_halo569 returned {detail})"
        )
    return halo


def read_haz_box(snap_file_0):
    with h5py.File(str(snap_file_0), "r") as f:
        h = float(f["Parameters"].attrs["HubbleParam"])
        a = float(f["Header"].attrs["Time"])
        z = float(f["Header"].attrs["Redshift"])
        box_ckpch = float(f["Header"].attrs["BoxSize"])
    return h, a, z, box_ckpch


# -----------------------------------------------------------------------------
# Resolution label detection (for legend, when --compare-dir is used)
# -----------------------------------------------------------------------------

def detect_resolution_label(output_dir):
    """
    Best-effort resolution label from a directory name, e.g.
    'S10_output_2048' -> '2048^3', 'S10_output_1024_cygnus_foo' -> '1024^3'.
    Falls back to the literal directory name if no resolution-looking token
    (a run of 3+ digits) is found.
    """
    name = Path(output_dir).resolve().name
    m = re.search(r"(\d{3,})", name)
    if m:
        return f"{m.group(1)}" + r"$^3$"
    return name


# -----------------------------------------------------------------------------
# Radial dust extraction
# -----------------------------------------------------------------------------

def extract_dust_radial_profile(entry, output_dir, rmax_factor=1.0, n_bins=40,
                                verbose=True):
    """
    Compute dust mass radial bins for one snapshot.

    Returns a dict with r_edges_pkpc, r_mid_pkpc, r_mid_over_r200,
    dust_mass_bins, enclosed_fraction, total_dust, z, r200_kpc, etc.
    """
    snap_num = entry["snap_num"]
    snap_file_0 = entry["snap_file_0"]
    h, a, z, box_ckpch = read_haz_box(snap_file_0)

    halo = get_halo569_frozen(output_dir, snap_num, verbose=verbose)
    center_ckpch = np.asarray(halo["center"], dtype=float)
    center_pkpc = center_ckpch * a / h
    r200_pkpc = float(halo["r200_pkpc"])
    rmax_pkpc = r200_pkpc * float(rmax_factor)

    if verbose:
        fallback = "  [catalog fallback]" if halo.get("used_catalog_fallback", False) else ""
        print(
            f"  Halo 569: snap={snap_num:03d} z={z:.3f} "
            f"R200={r200_pkpc:.1f} pkpc M200={halo['m200_msun']:.3e} Msun{fallback}"
        )
        print(
            f"  Center: ({center_pkpc[0]:.1f}, {center_pkpc[1]:.1f}, "
            f"{center_pkpc[2]:.1f}) pkpc"
        )

    chunks = snapshot_chunks_from_file0(snap_file_0)
    box_pkpc = box_ckpch * a / h

    dust_r_all = []
    dust_m_all = []

    for chunk in chunks:
        with h5py.File(str(chunk), "r") as f:
            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]
            if "Coordinates" not in pt6 or "Masses" not in pt6:
                continue
            if len(pt6["Masses"]) == 0:
                continue

            pos_pkpc = pt6["Coordinates"][:] * a / h
            mass_msun = pt6["Masses"][:] * MSUN_PER_CODE / h

            dx = pos_pkpc - center_pkpc[None, :]
            dx -= box_pkpc * np.round(dx / box_pkpc)
            r = np.sqrt((dx * dx).sum(axis=1))

            mask = r <= rmax_pkpc
            if np.any(mask):
                dust_r_all.append(r[mask])
                dust_m_all.append(mass_msun[mask])

    r_edges = np.linspace(0.0, rmax_pkpc, n_bins + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    if not dust_r_all:
        dust_mass_bins = np.zeros(n_bins)
        total_dust = 0.0
    else:
        dust_r = np.concatenate(dust_r_all)
        dust_m = np.concatenate(dust_m_all)
        dust_mass_bins, _ = np.histogram(dust_r, bins=r_edges, weights=dust_m)
        total_dust = float(dust_m.sum())

    enclosed = np.cumsum(dust_mass_bins) / total_dust if total_dust > 0 else np.zeros(n_bins)

    if verbose:
        print(f"  Dust within {rmax_factor:.2f} R200: {total_dust:.3e} Msun")

    return {
        "snap_num": snap_num,
        "z": z,
        "z_label": float(entry.get("z_target", z)),
        "z_actual": z,
        "h": h,
        "a": a,
        "center_pkpc": center_pkpc,
        "center_ckpch": center_ckpch,
        "r200_kpc": r200_pkpc,
        "m200_msun": float(halo["m200_msun"]),
        "used_catalog_fallback": bool(halo.get("used_catalog_fallback", False)),
        "r_edges": r_edges,
        "r_mid": r_mid,
        "r_mid_over_r200": r_mid / r200_pkpc,
        "dust_mass_bins": dust_mass_bins,
        "total_dust": total_dust,
        "enclosed_fraction": enclosed,
    }


def save_epoch_npz(epoch, outpath):
    np.savez(outpath, **{k: np.asarray(v) for k, v in epoch.items()})


def load_epoch_npz(path):
    d = dict(np.load(path, allow_pickle=True))
    # Convert scalar arrays back to Python floats/ints where convenient.
    for key in ("snap_num",):
        if key in d:
            d[key] = int(np.asarray(d[key]))
    for key in ("z", "z_label", "z_actual", "h", "a", "r200_kpc", "m200_msun", "total_dust"):
        if key in d:
            d[key] = float(np.asarray(d[key]))
    if "used_catalog_fallback" in d:
        d["used_catalog_fallback"] = bool(np.asarray(d["used_catalog_fallback"]))
    return d


# -----------------------------------------------------------------------------
# Per-run epoch collection (shared by primary and --compare-dir runs)
# -----------------------------------------------------------------------------

def collect_epochs(output_dir, redshifts, rmax_factor, n_bins, outdir,
                   summary_only, tag_prefix=""):
    """
    Build (or load from cache) the list of per-redshift radial-profile epoch
    dicts for one Gadget output directory. tag_prefix distinguishes cached
    NPZ files between multiple runs (e.g. different resolutions) sharing the
    same --outdir, so a 1024^3 and 2048^3 run can be cached side by side
    without overwriting each other's files.
    """
    output_dir = Path(output_dir).resolve()
    print(f"Scanning {output_dir} for snapshots...")
    all_snaps = find_all_snapshots(output_dir)
    if not all_snaps:
        print(f"  No snapshots with usable FOF/FOF+Subfind catalogs found in {output_dir}.")
        return []
    print(f"  Found {len(all_snaps)} snapshots (z={all_snaps[0]['z']:.2f} -> z={all_snaps[-1]['z']:.2f})")

    epoch_data = []

    for z_target in sorted(redshifts, reverse=True):
        entry = dict(find_nearest_snapshot(all_snaps, z_target))
        entry["z_target"] = float(z_target)

        snap_num = entry["snap_num"]
        z_actual = entry["z"]

        z_label_str = f"{z_target:g}".replace(".", "p")
        tag = f"{tag_prefix}snap{snap_num:03d}_ztarget{z_label_str}_zactual{z_actual:.2f}"
        npz_out = outdir / f"radial_{tag}.npz"

        print(f"\n[{output_dir.name}] Target z={z_target:g} -> snap {snap_num:03d} z={z_actual:.3f}")

        if summary_only and npz_out.exists():
            print(f"  Loading cached data from {npz_out}")
            epoch = load_epoch_npz(npz_out)
            epoch["z_label"] = float(epoch.get("z_label", z_target))
            epoch["z_actual"] = float(epoch.get("z_actual", z_actual))
        else:
            try:
                epoch = extract_dust_radial_profile(
                    entry,
                    output_dir=output_dir,
                    rmax_factor=rmax_factor,
                    n_bins=n_bins,
                    verbose=True,
                )

                if (not np.isfinite(epoch["r200_kpc"])) or epoch["r200_kpc"] <= 1.0:
                    print(
                        f"  WARNING: skipping target z={z_target:g}; "
                        f"invalid R200={epoch['r200_kpc']:.3g} pkpc"
                    )
                    continue

                save_epoch_npz(epoch, npz_out)
                print(f"  Radial data saved: {npz_out}")
            except Exception as e:
                print(f"  WARNING: failed snap {snap_num:03d}: {e}")
                continue

        if (not np.isfinite(epoch["r200_kpc"])) or epoch["r200_kpc"] <= 1.0:
            print(
                f"  WARNING: skipping target z={z_target:g}; "
                f"invalid R200={epoch['r200_kpc']:.3g} pkpc"
            )
            continue

        epoch_data.append(epoch)

    return epoch_data


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_summary_figure(epoch_data, output_path, rmax_factor,
                        compare_epoch_data=None,
                        primary_label=None, compare_label=None):
    """
    Make enclosed dust fraction vs r/R200 figure with region header row.

    epoch_data         : list of epoch dicts for the primary run (solid lines)
    compare_epoch_data : optional list of epoch dicts for a second run
                         (e.g. a different resolution), drawn as dashed lines
                         in the SAME color as the primary run's matching
                         redshift. Matching is by z_label, falling back to
                         positional order if a given z_label isn't present
                         in both lists.
    primary_label / compare_label : resolution labels used in a small
                         linestyle legend when compare_epoch_data is given.
    """
    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.08, 1.0], hspace=0.0)
    ax_hdr = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    line_colors = ["#85d1d9", "#46b5c4", "#2196a8", "#1d6fa4", "#0d0d0d"]
    epochs = sorted(epoch_data, key=lambda x: -float(x.get("z_label", x["z"])))

    # Map z_label -> color/marker index so the comparison run can reuse the
    # exact same color per redshift, rather than re-deriving its own order
    # (which could disagree if the two runs don't have identical redshift
    # lists).
    color_by_zlabel = {}

    for i, d in enumerate(epochs):
        z_label = float(d.get("z_label", d["z"]))
        z_actual = float(d.get("z_actual", d["z"]))
        r200 = float(d["r200_kpc"])
        x = np.asarray(d["r_mid_over_r200"])
        y = np.asarray(d["enclosed_fraction"])
        color = line_colors[i % len(line_colors)]
        marker = LINE_MARKERS[i % len(LINE_MARKERS)]
        color_by_zlabel[z_label] = (color, marker)

        label = f"z = {z_label:g}   (R$_{{200}}$ = {r200:.0f} kpc)"

        ax.plot(
            x, y,
            color=color,
            lw=2.3,
            marker=marker,
            markersize=4.8 if marker != "*" else 7.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markevery=MARK_EVERY,
            label=label,
            zorder=5 if z_label > 0 else 7,
        )

    # Overlay the comparison run (e.g. a different resolution) as dashed
    # lines in the same color per redshift. These are intentionally left
    # out of the main z-legend (which already lists R200 for the primary
    # run) and distinguished instead via a small linestyle legend below.
    if compare_epoch_data:
        compare_epochs = sorted(
            compare_epoch_data, key=lambda x: -float(x.get("z_label", x["z"]))
        )
        for j, d in enumerate(compare_epochs):
            z_label = float(d.get("z_label", d["z"]))
            x = np.asarray(d["r_mid_over_r200"])
            y = np.asarray(d["enclosed_fraction"])
            color, marker = color_by_zlabel.get(
                z_label, (line_colors[j % len(line_colors)], LINE_MARKERS[j % len(LINE_MARKERS)])
            )
            ax.plot(
                x, y,
                color=color,
                lw=2.0,
                ls="--",
                marker=marker,
                markersize=4.0 if marker != "*" else 6.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markevery=MARK_EVERY,
                alpha=0.9,
                zorder=4 if z_label > 0 else 6,
            )

    # Region boundaries in r/R200. Keep the background white; only the broad
    # transition bands mark the disk/CGM and inner/outer-CGM boundaries.
    trans_w = 0.04
    for x_center in [0.05, 0.30]:
        ax.axvspan(
            x_center * (1.0 - trans_w),
            x_center * (1.0 + trans_w),
            color="0.70", alpha=0.35, zorder=1, linewidth=0,
        )

    ax.set_xlabel(r"$r\,/\,R_{200}$", fontsize=11)
    ax.set_ylabel("Enclosed dust fraction", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.01, rmax_factor)
    ax.set_ylim(0.0, 1.05)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.88", lw=0.5, zorder=0)
    ax.minorticks_off()
    ax.tick_params(labelsize=9)

    z_legend = ax.legend(
        fontsize=8.5, loc="upper left", framealpha=0.92,
        labelspacing=0.35, borderpad=0.7, bbox_to_anchor=(0.01, 0.99),
    )

    # Small second legend distinguishing line style by resolution, only
    # shown when a comparison run was actually plotted.
    if compare_epoch_data:
        ax.add_artist(z_legend)
        style_handles = [
            plt.Line2D([0], [0], color="0.25", lw=2.3, ls="-",
                      label=primary_label or "primary"),
            plt.Line2D([0], [0], color="0.25", lw=2.0, ls="--",
                      label=compare_label or "comparison"),
        ]
        ax.legend(
            handles=style_handles, fontsize=8.0, loc="lower right",
            framealpha=0.92, labelspacing=0.3, borderpad=0.6,
            title="Resolution", title_fontsize=8.0,
        )

    # Header row with region labels.
    ax_hdr.set_xscale("log")
    ax_hdr.set_xlim(0.01, rmax_factor)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.axhline(1.0, color="black", lw=0.8, zorder=10)
    for x_center in [0.05, 0.30]:
        ax_hdr.axvspan(
            x_center * (1.0 - trans_w),
            x_center * (1.0 + trans_w),
            color="0.70", alpha=0.35, linewidth=0,
        )

    disk_mid = math.exp(0.5 * (math.log(0.01) + math.log(0.05)))
    inner_mid = math.exp(0.5 * (math.log(0.05) + math.log(0.30)))
    outer_mid = math.exp(0.5 * (math.log(0.30) + math.log(max(rmax_factor, 0.31))))

    for x, lbl in [
        (disk_mid, "Disk"),
        (inner_mid, "Inner CGM"),
        (outer_mid, "Outer CGM / halo"),
    ]:
        ax_hdr.text(
            x, 0.45, lbl, ha="center", va="center",
            fontsize=8.5, color="0.3", fontweight="normal",
            transform=ax_hdr.transData,
        )

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Summary figure saved: {output_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-script radial enclosed dust evolution for Halo 569"
    )
    parser.add_argument("output_dir", help="Primary Gadget output dir, e.g. ../S10_output_1024")
    parser.add_argument("--compare-dir", default=None,
                        help="Optional second Gadget output dir (e.g. a different "
                             "resolution) to overlay as dashed lines on the same plot")
    parser.add_argument("--redshifts", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--rmax-factor", type=float, default=1.0,
                        help="Maximum radius as a multiple of R200")
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--outdir", default="radial_evolution")
    parser.add_argument("--summary", default="dust_radial_evolution.pdf")
    parser.add_argument("--summary-only", action="store_true",
                        help="Load cached NPZ files and remake only the summary plot")
    parser.add_argument("--primary-label", default=None,
                        help="Override auto-detected resolution label for the primary run")
    parser.add_argument("--compare-label", default=None,
                        help="Override auto-detected resolution label for --compare-dir")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    primary_label = args.primary_label or detect_resolution_label(output_dir)

    epoch_data = collect_epochs(
        output_dir, args.redshifts, args.rmax_factor, args.n_bins,
        outdir, args.summary_only,
        tag_prefix="",
    )

    if not epoch_data:
        print("\nNo valid epochs for primary run -- cannot make summary figure.")
        sys.exit(1)

    compare_epoch_data = None
    compare_label = None
    if args.compare_dir:
        compare_dir = Path(args.compare_dir).resolve()
        compare_label = args.compare_label or detect_resolution_label(compare_dir)
        # tag_prefix keeps cached NPZ files from the two runs from colliding
        # when they share --outdir.
        compare_tag_prefix = f"{compare_dir.name}_"
        compare_epoch_data = collect_epochs(
            compare_dir, args.redshifts, args.rmax_factor, args.n_bins,
            outdir, args.summary_only,
            tag_prefix=compare_tag_prefix,
        )
        if not compare_epoch_data:
            print(
                f"\nWARNING: --compare-dir {compare_dir} produced no valid epochs; "
                f"continuing with the primary run only."
            )
            compare_epoch_data = None

    print(f"\nMaking summary figure ({len(epoch_data)} primary epochs"
          + (f", {len(compare_epoch_data)} comparison epochs" if compare_epoch_data else "")
          + ")...")
    make_summary_figure(
        epoch_data, args.summary, args.rmax_factor,
        compare_epoch_data=compare_epoch_data,
        primary_label=primary_label,
        compare_label=compare_label,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
