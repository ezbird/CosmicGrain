#!/usr/bin/env python3
"""
run_radial_evolution.py
-----------------------
Loops over a set of target redshifts, finds the nearest snapshot, reads
R_200 from the SubFind catalog for Halo 569, and produces a summary figure
overlaying the enclosed dust fraction profiles at all epochs.

Halo 569 identification
-----------------------
At each epoch the PRIMARY halo is selected as the most massive FOF group
(argmax Group_M_Crit200) across ALL catalog chunks — NOT just index 0 of
chunk .0, which only contains a fraction of the groups and can silently
return the wrong object.

Unit conventions (Gadget-4 defaults):
  Positions  : comoving kpc/h  →  physical kpc  via  x * a / h
  Masses     : 1e10 M_sun/h   →  M_sun          via  m * 1e10 / h
  R_200      : comoving kpc/h  →  physical kpc  via  r * a / h
  HubbleParam: from f["Parameters"].attrs["HubbleParam"]  (NOT Header)

Usage:
    python run_radial_evolution.py /path/to/output/ [options]

    python run_radial_evolution.py ../S10_output_1024/ \\
        --redshifts 0 0.5 1 2 3 \\
        --rmax-factor 1.0 \\
        --outdir radial_evolution/ \\
        --summary dust_radial_evolution.png

    # Skip re-running the analysis script (just remake the summary):
    python run_radial_evolution.py ../S10_output_1024/ --summary-only
"""

import os
import sys
import re
import argparse
import subprocess
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from halo_utils import (get_halo569_reference, get_halo569,
                        glob_catalog_chunks, read_snap_header,
                        find_shrinking_sphere_center)

plt.style.use('cosmicgrain.mplstyle')

MSUN_PER_CODE = 1e10   # Gadget default: 1 code mass unit = 1e10 M_sun/h

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / catalog discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_last_snap_num(output_dir):
    """Return the highest snapshot number that has both a snapdir and a catalog."""
    output_dir = Path(output_dir)
    last = None
    for groups_dir in sorted(output_dir.glob("groups_*")):
        m = re.search(r'groups_(\d+)', groups_dir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        snapdir  = output_dir / f'snapdir_{snap_num:03d}'
        if snapdir.exists():
            last = snap_num
    return last

def find_all_snapshots(output_dir):
    """
    Return sorted list of (snap_num, snap_file_0, catalog_file_0, redshift).
    Only snapshots that have both a snapshot file and a SubFind catalog.
    """
    output_dir = Path(output_dir)
    entries = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))

        snap_files = sorted(snapdir.glob("snap_*.hdf5")) + \
                     sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snap_files:
            continue

        groups_dir = output_dir / f"groups_{snap_num:03d}"
        cat_files  = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) \
                     if groups_dir.exists() else []
        if not cat_files:
            continue

        try:
            with h5py.File(str(snap_files[0]), "r") as f:
                z = float(f["Header"].attrs["Redshift"])
        except Exception:
            continue

        entries.append((snap_num, str(snap_files[0]), str(cat_files[0]), z))

    return entries


def find_nearest_snapshot(entries, z_target):
    return min(entries, key=lambda e: abs(e[3] - z_target))




_ref_cache = {}

def get_primary_halo(catalog_file):
    p          = Path(catalog_file)
    groups_dir = p.parent
    output_dir = groups_dir.parent
    m          = re.search(r'groups_(\d+)', groups_dir.name)
    if not m:
        return None, None, None, None, None
    snap_num = int(m.group(1))

    snapdir = output_dir / f'snapdir_{snap_num:03d}'
    if not snapdir.exists():
        return None, None, None, None, None
    try:
        hdr = read_snap_header(snapdir)
    except Exception:
        return None, None, None, None, None
    h, a, z = hdr['h'], hdr['a'], hdr['z']

    # Cache reference per output_dir
    key = str(output_dir)
    if key not in _ref_cache:
        try:
            last_snap = find_last_snap_num(output_dir)
            if last_snap is None:
                print(f'  [get_primary_halo] no snapshots found in {output_dir}')
                return None, None, None, None, None
            _ref_cache[key] = get_halo569_reference(output_dir,
                                                     snap_num_z0=last_snap)
        except Exception as e:
            print(f'  [get_primary_halo] reference failed: {e}')
            return None, None, None, None, None
    ref = _ref_cache[key]

    halo = get_halo569(groups_dir, snap_num, ref, verbose=False)
    if halo is None:
        return None, None, None, None, None

    print(f'  DEBUG halo dict: r200_ckpch={halo["r200_ckpch"]:.2f} '
          f'r200_pkpc={halo["r200_pkpc"]:.2f} '
          f'm200={halo["m200_code"]:.3f}')

    center_phys = halo['center'] * a / h
    r200_phys   = halo['r200_pkpc']

    return center_phys, r200_phys, z, h, a


# ─────────────────────────────────────────────────────────────────────────────
# Radial binning
# ─────────────────────────────────────────────────────────────────────────────

def _extract_radial_bins(snap_file, catalog_file, rmax_kpc, n_bins=40):
    """
    Compute dust mass in radial bins around Halo 569's center.
    Uses the same primary-halo identification as get_primary_halo().
    Returns dict {r_edges, dust_mass_bins, total_dust}.
    """
    center_phys, r200_phys, z, h, a = get_primary_halo(catalog_file)
    if center_phys is None:
        return dict(r_edges=np.linspace(0, rmax_kpc, n_bins + 1),
                    dust_mass_bins=np.zeros(n_bins),
                    total_dust=0.0)

    # Read dust particles from all snapshot chunks
    sf        = Path(snap_file)
    stem      = re.sub(r"\.\d+$", "", sf.stem)
    snap_chunks = sorted(sf.parent.glob(f"{stem}*.hdf5"))
    if not snap_chunks:
        snap_chunks = [sf]

    box_phys_kpc = None
    dust_r_list, dust_m_list = [], []

    for chunk in snap_chunks:
        with h5py.File(str(chunk), "r") as f:
            if box_phys_kpc is None:
                box_code     = float(f["Header"].attrs["BoxSize"])
                box_phys_kpc = box_code * a / h

            if "PartType6" not in f:
                continue

            # Coordinates: comoving kpc/h → physical kpc
            pos  = f["PartType6"]["Coordinates"][:] * a / h
            # Masses: code units (1e10 M_sun/h) → M_sun
            mass = f["PartType6"]["Masses"][:] * MSUN_PER_CODE / h

            # Periodic boundary wrap
            dx = pos - center_phys
            dx -= box_phys_kpc * np.round(dx / box_phys_kpc)
            r  = np.sqrt((dx**2).sum(axis=1))

            mask = r <= rmax_kpc
            dust_r_list.append(r[mask])
            dust_m_list.append(mass[mask])

    if not dust_r_list or sum(len(x) for x in dust_r_list) == 0:
        return dict(r_edges=np.linspace(0, rmax_kpc, n_bins + 1),
                    dust_mass_bins=np.zeros(n_bins),
                    total_dust=0.0)

    dust_r = np.concatenate(dust_r_list)
    dust_m = np.concatenate(dust_m_list)

    r_edges        = np.linspace(0, rmax_kpc, n_bins + 1)
    dust_mass_bins, _ = np.histogram(dust_r, bins=r_edges, weights=dust_m)

    return dict(r_edges=r_edges,
                dust_mass_bins=dust_mass_bins,
                total_dust=float(dust_m.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_figure(epoch_data, output_path, rmax_factor):
    """
    epoch_data: list of dicts with keys:
        z, r200_kpc, r_edges, dust_mass_bins, total_dust
    """
    import math

    fig = plt.figure(figsize=(6.5, 5.4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.08, 1.0], hspace=0.0)
    ax_hdr = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1])

    line_colors = ["#85d1d9", "#46b5c4", "#2196a8", "#1d6fa4", "#0d0d0d"]
    epochs = sorted(epoch_data, key=lambda x: -x["z"])

    for i, d in enumerate(epochs):
        z       = d["z"]
        r200    = d["r200_kpc"]
        r_edges = d["r_edges"]
        m_bins  = d["dust_mass_bins"]
        total   = d["total_dust"]
        color   = line_colors[i % len(line_colors)]
        label   = f"z = {z:.1f}   (R$_{{200}}$ = {r200:.0f} kpc)"

        r_mid  = 0.5 * (r_edges[:-1] + r_edges[1:])
        r_r200 = r_mid / r200

        enclosed_frac = np.cumsum(m_bins) / total if total > 0 \
                        else np.zeros_like(m_bins)
        ax.plot(r_r200, enclosed_frac, color=color, lw=2.2, label=label)

    # Region shading
    ax.axvspan(0.05, 0.30, color="0.96", alpha=1.0, zorder=0)
    ax.axvspan(0.30, 1.00, color="0.99", alpha=1.0, zorder=0)
    trans_w = 0.04
    for x_center in [0.05, 0.30]:
        ax.axvspan(x_center * (1 - trans_w), x_center * (1 + trans_w),
                   color="0.70", alpha=0.35, zorder=1, linewidth=0)

    ax.set_xlabel(r"$r\,/\,R_{200}$", fontsize=11)
    ax.set_ylabel("Enclosed dust fraction", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.01, rmax_factor)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.88", lw=0.5, zorder=0)
    ax.minorticks_off()
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92,
              labelspacing=0.35, borderpad=0.7,
              bbox_to_anchor=(0.01, 0.99))
    ax.set_title("")

    # Header row with region labels
    ax_hdr.set_xscale("log")
    ax_hdr.set_xlim(0.01, rmax_factor)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.axhline(1.0, color="black", lw=0.8, zorder=10)
    ax_hdr.axvspan(0.05, 0.30, color="0.96", alpha=1.0)
    ax_hdr.axvspan(0.30, 1.00, color="0.99", alpha=1.0)
    for x_center in [0.05, 0.30]:
        ax_hdr.axvspan(x_center * (1 - trans_w), x_center * (1 + trans_w),
                       color="0.70", alpha=0.35, linewidth=0)

    disk_mid  = math.exp(0.5 * (math.log(0.01) + math.log(0.05)))
    inner_mid = math.exp(0.5 * (math.log(0.05) + math.log(0.30)))
    outer_mid = math.exp(0.5 * (math.log(0.30) + math.log(1.00)))

    for x, lbl in [(disk_mid,  "Disk"),
                   (inner_mid, "Inner CGM"),
                   (outer_mid, "Outer CGM / halo")]:
        ax_hdr.text(x, 0.45, lbl, ha="center", va="center",
                    fontsize=8.5, color="0.3", fontweight="normal",
                    transform=ax_hdr.transData)

    #fig.suptitle("Radial Dust Distribution by Redshift", fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Summary figure saved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_base_path(snap_file_0):
    p    = Path(snap_file_0)
    stem = re.sub(r"\.\d+$", "", p.stem)
    return str(p.parent / stem)


def run_epoch(snap_num, snap_file, catalog_file, z_actual, r200_phys,
              rmax_factor, outdir, script_path, show_plot):
    rmax     = r200_phys * rmax_factor
    tag      = f"snap{snap_num:03d}_z{z_actual:.2f}"
    plot_out = os.path.join(outdir, f"radial_{tag}.png")
    npz_out  = os.path.join(outdir, f"radial_{tag}.npz")

    snap_base = snapshot_base_path(snap_file)
    cmd = [
        sys.executable, script_path,
        "--catalog",   catalog_file,
        "--snapshot",  snap_base,
        "--out",       plot_out,
        "--rmax",      f"{rmax:.1f}",
        "--show_plot", "0",
    ]

    print(f"\n{'─'*60}")
    print(f"  z={z_actual:.3f}  R_200={r200_phys:.1f} pkpc  rmax={rmax:.1f} pkpc")
    print(f"  Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: analysis script failed for snap {snap_num}: {e}")
        return None

    try:
        epoch = _extract_radial_bins(snap_file, catalog_file, rmax, n_bins=40)
        epoch["z"]        = z_actual
        epoch["r200_kpc"] = r200_phys
        np.savez(npz_out, **{k: np.array(v) for k, v in epoch.items()})
        print(f"  Radial data saved: {npz_out}")
        return epoch
    except Exception as e:
        print(f"  WARNING: could not extract radial bins: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Radial dust analysis at multiple epochs for Halo 569")
    parser.add_argument("output_dir")
    parser.add_argument("--redshifts", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--rmax-factor", type=float, default=1.0,
                        help="rmax as multiple of R_200 (default: 1.0)")
    parser.add_argument("--outdir",      default="radial_evolution")
    parser.add_argument("--summary",     default="dust_radial_evolution.png")
    parser.add_argument("--script",      default="plot_radial_dust_analysis.py")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--show-plot",   type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve analysis script path
    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(os.path.dirname(__file__), script_path)
    if not os.path.exists(script_path):
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.basename(args.script))
    if not os.path.exists(script_path):
        print(f"ERROR: cannot find {args.script}")
        sys.exit(1)

    print(f"Scanning {args.output_dir} for snapshots...")
    all_snaps = find_all_snapshots(args.output_dir)
    if not all_snaps:
        print("No snapshots with SubFind catalogs found.")
        sys.exit(1)
    print(f"Found {len(all_snaps)} snapshots  "
          f"(z={all_snaps[0][3]:.1f} → z={all_snaps[-1][3]:.1f})")

    epoch_data = []

    for z_target in sorted(args.redshifts, reverse=True):
        snap_num, snap_file, catalog_file, z_actual = \
            find_nearest_snapshot(all_snaps, z_target)
        print(f"\nTarget z={z_target:.1f}  →  snap {snap_num:03d}  z={z_actual:.3f}")

        center_phys, r200_phys, _, _, _ = get_primary_halo(catalog_file)
        if r200_phys is None:
            print("  No groups in catalog — skipping")
            continue

        print(f"  Halo 569: R_200 = {r200_phys:.1f} pkpc  "
              f"center = ({center_phys[0]:.0f}, {center_phys[1]:.0f}, "
              f"{center_phys[2]:.0f}) pkpc")
        print(f"  rmax = {r200_phys * args.rmax_factor:.1f} pkpc")

        tag     = f"snap{snap_num:03d}_z{z_actual:.2f}"
        npz_out = os.path.join(args.outdir, f"radial_{tag}.npz")

        if args.summary_only and os.path.exists(npz_out):
            print(f"  Loading cached data from {npz_out}")
            d = dict(np.load(npz_out, allow_pickle=True))
            d["z"]        = float(d.get("z",        z_actual))
            d["r200_kpc"] = float(d.get("r200_kpc", r200_phys))
            epoch_data.append(d)
        else:
            d = run_epoch(snap_num, snap_file, catalog_file, z_actual,
                          r200_phys, args.rmax_factor, args.outdir,
                          script_path, args.show_plot)
            if d is not None:
                epoch_data.append(d)

    if not epoch_data:
        print("\nNo valid epochs — cannot make summary figure.")
        sys.exit(1)

    print(f"\nMaking summary figure ({len(epoch_data)} epochs)...")
    make_summary_figure(epoch_data, args.summary, args.rmax_factor)
    print("\nDone.")


if __name__ == "__main__":
    main()
