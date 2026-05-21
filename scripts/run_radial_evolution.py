#!/usr/bin/env python3
"""
run_radial_evolution.py
-----------------------
Loops over a set of target redshifts, finds the nearest snapshot, reads
R_200 from the SubFind catalog, and calls plot_radial_dust_analysis.py
with rmax = R_200 (physical kpc) at each epoch.

Then produces a summary figure overlaying the enclosed dust fraction
profiles at all epochs.

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
import glob
import argparse
import subprocess
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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

        # Read redshift from snapshot header
        try:
            with h5py.File(str(snap_files[0]), "r") as f:
                z = float(f["Header"].attrs["Redshift"])
        except Exception:
            continue

        entries.append((snap_num, str(snap_files[0]), str(cat_files[0]), z))

    return entries   # sorted by snap_num (ascending)


def find_nearest_snapshot(entries, z_target):
    """Return the entry whose redshift is closest to z_target."""
    return min(entries, key=lambda e: abs(e[3] - z_target))


def get_r200_physical(catalog_file):
    """
    Read Group_R_Crit200 [comoving kpc/h] and convert to physical kpc.
    Returns (r200_physical_kpc, z, h).
    """
    p = Path(catalog_file)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    for chunk in chunks:
        with h5py.File(str(chunk), "r") as f:
            if "Group" not in f or "GroupPos" not in f["Group"]:
                continue
            if len(f["Group"]["GroupPos"]) == 0:
                continue
            r200_comov = float(f["Group"]["Group_R_Crit200"][0])  # comoving kpc/h
            h   = float(f["Parameters"].attrs["HubbleParam"])
            a   = float(f["Header"].attrs["Time"])          # scale factor
            z   = float(f["Header"].attrs["Redshift"])
            # physical kpc = comoving kpc/h  × a / h
            r200_phys = r200_comov * a / h
            return r200_phys, z, h

    return None, None, None


def snapshot_base_path(snap_file_0):
    """
    Convert snap_file_0 (e.g. snapdir_049/snap_049.0.hdf5) to the base
    path expected by plot_radial_dust_analysis.py (no chunk suffix, no .hdf5).
    e.g. snapdir_049/snap_049
    """
    p = Path(snap_file_0)
    # Strip trailing .N.hdf5 or .hdf5
    stem = re.sub(r"\.\d+$", "", p.stem)   # snap_049.0 → snap_049
    return str(p.parent / stem)


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def load_radial_data(npz_path):
    """Load the per-epoch radial data saved by each analysis run."""
    if not os.path.exists(npz_path):
        return None
    return dict(np.load(npz_path, allow_pickle=True))


def make_summary_figure(epoch_data, output_path, rmax_factor):
    """
    epoch_data: list of dicts, each with keys:
        z, r200_kpc, r_edges, dust_mass_bins, total_dust
    """
    # ── Layout: small header row above main axes for region labels ────────────
    fig = plt.figure(figsize=(6.5, 5.4))
    # Two rows: thin header (10%) + main plot (90%)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.08, 1.0], hspace=0.0)
    ax_hdr = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1])

    # ── Colors: light at high-z → black at z=0 ───────────────────────────────
    line_colors = ["#85d1d9", "#46b5c4", "#2196a8", "#1d6fa4", "#0d0d0d"]

    zvals  = [d["z"] for d in epoch_data]
    epochs = sorted(epoch_data, key=lambda x: -x["z"])   # high-z first

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

    # ── Region background shading ─────────────────────────────────────────────
    # Only inner and outer CGM get subtle shading; disk stays white
    ax.axvspan(0.05,  0.30, color="0.96", alpha=1.0, zorder=0)
    ax.axvspan(0.30,  1.00, color="0.99", alpha=1.0, zorder=0)

    # Fuzzy transition bands — same width for both boundaries
    trans_w = 0.04   # fractional half-width in log space (matches right divider)
    for x_center in [0.05, 0.30]:
        ax.axvspan(x_center * (1 - trans_w), x_center * (1 + trans_w),
                   color="0.70", alpha=0.35, zorder=1, linewidth=0)

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$r\,/\,R_{200}$", fontsize=11)
    ax.set_ylabel("Enclosed dust fraction", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.01, rmax_factor)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    # Major grid only, behind everything
    ax.grid(True, which="major", color="0.88", lw=0.5, zorder=0)
    ax.minorticks_off()
    ax.tick_params(labelsize=9)

    # Legend in upper left — curves are low there at early times,
    # and upper right gets crowded
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92,
              labelspacing=0.35, borderpad=0.7,
              bbox_to_anchor=(0.01, 0.99))

    ax.set_title("")   # title goes in the header row instead

    # ── Header row: region labels centered on their bands ────────────────────
    # In log space: disk 0.01–0.05, inner CGM 0.05–0.30, outer CGM 0.30–1.0
    # Compute geometric midpoints for centering
    import math
    ax_hdr.set_xscale("log")
    ax_hdr.set_xlim(0.01, rmax_factor)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")

    # Thin black top border
    ax_hdr.axhline(1.0, color="black", lw=0.8, zorder=10)

    # Region fill in header — disk stays white, others match main plot
    ax_hdr.axvspan(0.05,  0.30, color="0.96", alpha=1.0)
    ax_hdr.axvspan(0.30,  1.00, color="0.99", alpha=1.0)
    # Fuzzy transitions in header — same width as main plot
    for x_center in [0.05, 0.30]:
        ax_hdr.axvspan(x_center * (1 - trans_w), x_center * (1 + trans_w),
                       color="0.70", alpha=0.35, linewidth=0)

    # Geometric center of each band in log space
    disk_mid     = math.exp(0.5 * (math.log(0.01)  + math.log(0.05)))
    inner_mid    = math.exp(0.5 * (math.log(0.05)  + math.log(0.30)))
    outer_mid    = math.exp(0.5 * (math.log(0.30)  + math.log(1.00)))

    for x, label in [(disk_mid,  "Disk"),
                     (inner_mid, "Inner CGM"),
                     (outer_mid, "Outer CGM / halo")]:
        ax_hdr.text(x, 0.45, label, ha="center", va="center",
                    fontsize=8.5, color="0.3", fontweight="normal",
                    transform=ax_hdr.transData)

    # Title above the header
    fig.suptitle("Radial Dust Distribution by Redshift",
                 fontsize=10, y=1.01)

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Summary figure saved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(snap_num, snap_file, catalog_file, z_actual, r200_phys,
              rmax_factor, outdir, script_path, show_plot):
    """
    Call plot_radial_dust_analysis.py for one snapshot and save the radial
    data as a .npz alongside the plot.
    Returns a dict suitable for make_summary_figure, or None on failure.
    """
    rmax = r200_phys * rmax_factor
    tag  = f"snap{snap_num:03d}_z{z_actual:.2f}"
    plot_out = os.path.join(outdir, f"radial_{tag}.png")
    npz_out  = os.path.join(outdir, f"radial_{tag}.npz")

    snap_base = snapshot_base_path(snap_file)

    cmd = [
        sys.executable, script_path,
        "--catalog",  catalog_file,
        "--snapshot", snap_base,
        "--out",      plot_out,
        "--rmax",     f"{rmax:.1f}",
        "--show_plot", "0",
    ]

    print(f"\n{'─'*60}")
    print(f"  z={z_actual:.3f}  R_200={r200_phys:.1f} kpc  rmax={rmax:.1f} kpc")
    print(f"  Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: analysis script failed for snap {snap_num}: {e}")
        return None

    # ── Extract radial data from the analysis output ──────────────────────
    # plot_radial_dust_analysis.py doesn't natively save .npz, so we
    # re-read the dust particles ourselves here for the summary figure.
    # This is a lightweight read — only dust positions and masses.
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


def _extract_radial_bins(snap_file, catalog_file, rmax_kpc, n_bins=40):
    """
    Lightweight extraction: dust particle positions/masses → radial bins.
    Returns dict {r_edges, dust_mass_bins, total_dust}.
    """
    # Get halo center and R_200
    r200_phys, z, h = get_r200_physical(catalog_file)
    a = 1.0 / (1.0 + z)

    # Read halo position from catalog (comoving kpc/h → physical kpc)
    p = Path(catalog_file)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    center_phys = None
    with h5py.File(str(chunks[0]), "r") as f:
        pos_comov = f["Group"]["GroupPos"][0]   # comoving kpc/h
        center_phys = pos_comov * a / h          # physical kpc

    # Read dust particles from all snapshot chunks
    sf = Path(snap_file)
    stem = re.sub(r"\.\d+$", "", sf.stem)
    snap_chunks = sorted(sf.parent.glob(f"{stem}*.hdf5"))
    if not snap_chunks:
        snap_chunks = [sf]

    unit_mass_g = None
    dust_r_list, dust_m_list = [], []

    for chunk in snap_chunks:
        with h5py.File(str(chunk), "r") as f:
            if unit_mass_g is None:
                params = f.get("Parameters") or {}
                unit_mass_g = float(params.attrs.get("UnitMass_in_g", 1.989e43))
            if "PartType6" not in f:
                continue
            box = float(f["Header"].attrs["BoxSize"]) * a / h  # physical kpc
            pos = f["PartType6"]["Coordinates"][:] * a / h
            mass = f["PartType6"]["Masses"][:]  * unit_mass_g / 1.989e33  # M_sun
            dx = pos - center_phys
            dx -= box * np.round(dx / box)
            r  = np.sqrt((dx**2).sum(axis=1))
            mask = r <= rmax_kpc
            dust_r_list.append(r[mask])
            dust_m_list.append(mass[mask])

    if not dust_r_list:
        return dict(r_edges=np.linspace(0, rmax_kpc, n_bins + 1),
                    dust_mass_bins=np.zeros(n_bins),
                    total_dust=0.0)

    dust_r = np.concatenate(dust_r_list)
    dust_m = np.concatenate(dust_m_list)

    r_edges = np.linspace(0, rmax_kpc, n_bins + 1)
    dust_mass_bins, _ = np.histogram(dust_r, bins=r_edges, weights=dust_m)

    return dict(r_edges=r_edges, dust_mass_bins=dust_mass_bins,
                total_dust=float(dust_m.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run radial dust analysis at multiple epochs")
    parser.add_argument("output_dir",
                        help="Gadget-4 output directory")
    parser.add_argument("--redshifts", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 3.0],
                        help="Target redshifts to analyse (default: 0 0.5 1 2 3)")
    parser.add_argument("--rmax-factor", type=float, default=1.0,
                        help="rmax as a multiple of R_200 (default: 1.0)")
    parser.add_argument("--outdir", default="radial_evolution",
                        help="Directory for per-snapshot plots (default: radial_evolution/)")
    parser.add_argument("--summary", default="dust_radial_evolution.png",
                        help="Filename for summary figure")
    parser.add_argument("--script", default="plot_radial_dust_analysis.py",
                        help="Path to plot_radial_dust_analysis.py")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip re-running analysis; regenerate summary from saved .npz files")
    parser.add_argument("--show-plot", type=int, default=0,
                        help="Pass to plot_radial_dust_analysis.py (0=no display)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve script path
    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(os.path.dirname(__file__), script_path)
    if not os.path.exists(script_path):
        # Try same directory as this script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   os.path.basename(args.script))
    if not os.path.exists(script_path):
        print(f"ERROR: cannot find {args.script}")
        sys.exit(1)

    # ── Find all snapshots ────────────────────────────────────────────────
    print(f"Scanning {args.output_dir} for snapshots...")
    all_snaps = find_all_snapshots(args.output_dir)
    if not all_snaps:
        print("No snapshots with SubFind catalogs found.")
        sys.exit(1)
    print(f"Found {len(all_snaps)} snapshots  "
          f"(z={all_snaps[0][3]:.1f} → z={all_snaps[-1][3]:.1f})")

    # ── Process each target redshift ──────────────────────────────────────
    epoch_data = []

    for z_target in sorted(args.redshifts, reverse=True):
        snap_num, snap_file, catalog_file, z_actual = \
            find_nearest_snapshot(all_snaps, z_target)
        print(f"\nTarget z={z_target:.1f}  →  snap {snap_num:03d}  z={z_actual:.3f}")

        r200_phys, _, _ = get_r200_physical(catalog_file)
        if r200_phys is None:
            print("  No groups in catalog — skipping")
            continue
        print(f"  R_200 = {r200_phys:.1f} kpc physical"
              f"  →  rmax = {r200_phys * args.rmax_factor:.1f} kpc")

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

    # ── Summary figure ────────────────────────────────────────────────────
    print(f"\nMaking summary figure ({len(epoch_data)} epochs)...")
    make_summary_figure(epoch_data, args.summary, args.rmax_factor)
    print("\nDone.")


if __name__ == "__main__":
    main()
