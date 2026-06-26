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

Usage:
  python run_radial_evolution_combined.py ../S10_output_1024/ \
      --redshifts 0 0.5 1 2 3 \
      --rmax-factor 1.0 \
      --outdir radial_evolution \
      --summary dust_radial_evolution.pdf

  # Rebuild only from cached NPZ files:
  python run_radial_evolution_combined.py ../S10_output_1024/ --summary-only
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use('sleek.mplstyle')

try:
    from halo_utils import (
        get_halo569_reference,
        get_halo569,
        read_snap_header,
        read_fof_catalog,
    )
except ImportError as e:
    print("ERROR: This script requires halo_utils.py in the same directory or PYTHONPATH")
    raise e

MSUN_PER_CODE = 1e10  # Gadget default: 1 code mass unit = 1e10 Msun/h

# For catalog-mass fallback R200 estimates.
# Units chosen so R comes out in physical kpc when M is in Msun and H is km/s/Mpc.
G_KPC_KMS2_MSUN = 4.30091e-6
H0_100_KMS_MPC = 100.0

SCRIPT_DIR = Path(__file__).resolve().parent

# Distinguish redshift curves by both color and marker for print/colorblind safety.
LINE_MARKERS = ["o", "s", "^", "D", "", "P", "X"]
MARK_EVERY = 5


# -----------------------------------------------------------------------------
# Snapshot/catalog discovery
# -----------------------------------------------------------------------------

def find_last_snap_num(output_dir):
    """Return the highest snapshot number with both a snapdir and a groups dir."""
    output_dir = Path(output_dir)
    last = None
    for groups_dir in sorted(output_dir.glob("groups_*")):
        m = re.search(r"groups_(\d+)", groups_dir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        snapdir = output_dir / f"snapdir_{snap_num:03d}"
        if snapdir.exists():
            last = snap_num
    return last


def find_all_snapshots(output_dir):
    """
    Return sorted list of dictionaries for snapshots with both snapshot chunks
    and SubFind/FOF catalog chunks.
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
        cat_files = sorted(groups_dir.glob("fof_subhalo_tab_*.0.hdf5")) if groups_dir.exists() else []
        if not cat_files:
            cat_files = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) if groups_dir.exists() else []
        if not cat_files:
            continue

        try:
            with h5py.File(str(snap_files[0]), "r") as f:
                z = float(f["Header"].attrs["Redshift"])
        except Exception:
            continue

        entries.append({
            "snap_num": snap_num,
            "snap_file_0": snap_files[0],
            "catalog_file_0": cat_files[0],
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
# Halo identification through updated halo_utils
# -----------------------------------------------------------------------------

_ref_cache = {}



def _periodic_delta(pos, center, box):
    """Minimum-image displacement in ckpc/h."""
    dx = np.asarray(pos, dtype=float) - np.asarray(center, dtype=float)[None, :]
    dx -= box * np.round(dx / box)
    return dx


def _select_primary_halo_idx(cat):
    """Use stellar mass if available, otherwise GroupMass."""
    if "mstar" in cat and np.nanmax(cat["mstar"]) > 0:
        return int(np.nanargmax(cat["mstar"])), "Mstar"
    return int(np.nanargmax(cat["group_mass"])), "GroupMass"


def _r200_from_m200_catalog(m200_code, hdr):
    """
    Estimate R200c from catalog M200 when Group_R_Crit200 is zero/missing.

    Catalog masses are in 1e10 Msun/h.  We convert to Msun and use
        M = 4/3 pi R^3 200 rho_crit(z)
    with rho_crit(z) = 3 H(z)^2 / (8 pi G).
    Returns (r200_ckpch, r200_pkpc, m200_msun).
    """
    m200_code = float(m200_code)
    if not np.isfinite(m200_code) or m200_code <= 0:
        return None

    h = float(hdr["h"])
    a = float(hdr["a"])
    Om0 = float(hdr.get("Omega0", 0.3158))
    Ol0 = float(hdr.get("OmegaLambda", 0.6842))
    z = float(hdr.get("z", 1.0 / a - 1.0))

    m_msun = m200_code * MSUN_PER_CODE / h
    Hz = H0_100_KMS_MPC * h * np.sqrt(Om0 * (1.0 + z)**3 + Ol0)
    rho_crit = 3.0 * (Hz / 1000.0)**2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)  # Msun/kpc^3
    r_pkpc = (3.0 * m_msun / (4.0 * np.pi * 200.0 * rho_crit))**(1.0 / 3.0)
    r_ckpch = r_pkpc * h / a
    return r_ckpch, r_pkpc, m_msun


def _catalog_halo_fallback(output_dir, snap_num, ref, verbose=False):
    """
    Local safety fallback if halo_utils returns an invalid/zero R200.

    This mirrors halo_utils' target selection, then uses catalog
    Group_R_Crit200/Group_M_Crit200 if available. If Group_R_Crit200 is zero
    but M200 is positive, it estimates R200c from M200 and rho_crit(z).
    """
    output_dir = Path(output_dir)
    groups_dir = output_dir / f"groups_{snap_num:03d}"
    snapdir = output_dir / f"snapdir_{snap_num:03d}"
    cat = read_fof_catalog(groups_dir, snap_num)
    if cat is None:
        raise RuntimeError(f"No catalog available for fallback at snap {snap_num:03d}")

    hdr = read_snap_header(snapdir)
    ref_pos = np.asarray(ref.get("center_ckpch", ref.get("center")), dtype=float)
    box = float(ref.get("box_ckpch", hdr["box"]))
    dx = _periodic_delta(cat["pos"], ref_pos, box)
    dist = np.sqrt((dx * dx).sum(axis=1))
    within = dist <= 5000.0

    if within.any():
        valid = within & (cat["group_mass"] > 1.0)
        if valid.any():
            idx = int(np.argmin(np.where(valid, dist, np.inf)))
            sel_by = "nearest to reference"
        else:
            idx, sel_by = _select_primary_halo_idx(cat)
    else:
        idx, sel_by = _select_primary_halo_idx(cat)

    center = np.asarray(cat["pos"][idx], dtype=float)
    rcat = float(cat["r200_catalog"][idx])
    mcat = float(cat["m200_catalog"][idx])

    if np.isfinite(rcat) and rcat > 0 and np.isfinite(mcat) and mcat > 0:
        r200_ckpch = rcat
        r200_pkpc = rcat * hdr["a"] / hdr["h"]
        m200_msun = mcat * MSUN_PER_CODE / hdr["h"]
        fallback_mode = "catalog R200/M200"
    else:
        est = _r200_from_m200_catalog(mcat, hdr)
        if est is None:
            # Last-ditch estimate from GroupMass if M200 is missing. This is less
            # formal than M200, but avoids plotting an unphysical R200=0 curve.
            est = _r200_from_m200_catalog(cat["group_mass"][idx], hdr)
            fallback_mode = "estimated from GroupMass"
        else:
            fallback_mode = "estimated from catalog M200"
        if est is None:
            raise RuntimeError(
                f"Catalog fallback failed at snap {snap_num:03d}: "
                f"Group_R_Crit200={rcat}, Group_M_Crit200={mcat}"
            )
        r200_ckpch, r200_pkpc, m200_msun = est

    if verbose:
        print(
            f"  [fallback] snap={snap_num:03d}: group {idx}, {sel_by}, "
            f"R200={r200_pkpc:.1f} pkpc ({fallback_mode})"
        )

    return {
        "center": center,
        "center_fof": center,
        "r200_ckpch": float(r200_ckpch),
        "r200_pkpc": float(r200_pkpc),
        "m200_code": float(mcat),
        "m200_msun": float(m200_msun),
        "group_idx": int(idx),
        "catalog_r200_ckpch": float(rcat),
        "catalog_r200_pkpc": float(rcat * hdr["a"] / hdr["h"]) if rcat > 0 else 0.0,
        "catalog_m200_code": float(mcat),
        "catalog_m200_msun": float(mcat * MSUN_PER_CODE / hdr["h"]) if mcat > 0 else 0.0,
        "used_catalog_fallback": True,
        "fallback_mode": fallback_mode,
        "dist_ckpch": float(dist[idx]),
        "selection": sel_by,
        "h": hdr["h"],
        "a": hdr["a"],
    }


def get_halo569_frozen(output_dir, snap_num, verbose=False):
    """
    Return halo_utils Halo 569 dictionary using the new stable center definition:
    frozen FOF/catalog center plus particle-SO/catalog fallback R200.
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

    try:
        halo = get_halo569(
            groups_dir,
            snap_num,
            ref,
            refine_center=False,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  [halo_utils] get_halo569 failed at snap {snap_num:03d}: {e}")
        halo = None

    # Some early catalogs can produce particle-SO failures or zero catalog radii.
    # Do not allow an R200=0 curve onto an r/R200 plot.
    if halo is None or not np.isfinite(halo.get("r200_pkpc", 0.0)) or halo.get("r200_pkpc", 0.0) <= 1e-6:
        halo = _catalog_halo_fallback(output_dir, snap_num, ref, verbose=verbose)

    if halo is None or halo.get("r200_pkpc", 0.0) <= 1e-6:
        raise RuntimeError(f"Could not identify Halo 569 at snap {snap_num:03d}")
    return halo


def read_haz_box(snap_file_0):
    with h5py.File(str(snap_file_0), "r") as f:
        h = float(f["Parameters"].attrs["HubbleParam"])
        a = float(f["Header"].attrs["Time"])
        z = float(f["Header"].attrs["Redshift"])
        box_ckpch = float(f["Header"].attrs["BoxSize"])
    return h, a, z, box_ckpch


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
# Plotting
# -----------------------------------------------------------------------------

def make_summary_figure(epoch_data, output_path, rmax_factor):
    """Make enclosed dust fraction vs r/R200 figure with region header row."""
    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.08, 1.0], hspace=0.0)
    ax_hdr = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    line_colors = ["#85d1d9", "#46b5c4", "#2196a8", "#1d6fa4", "#0d0d0d"]
    epochs = sorted(epoch_data, key=lambda x: -float(x.get("z_label", x["z"])))

    for i, d in enumerate(epochs):
        z_label = float(d.get("z_label", d["z"]))
        z_actual = float(d.get("z_actual", d["z"]))
        r200 = float(d["r200_kpc"])
        x = np.asarray(d["r_mid_over_r200"])
        y = np.asarray(d["enclosed_fraction"])
        color = line_colors[i % len(line_colors)]
        marker = LINE_MARKERS[i % len(LINE_MARKERS)]

        # Legend reports the requested redshift, while the actual nearest
        # snapshot redshift is preserved in the NPZ as z_actual.
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
    ax.legend(
        fontsize=8.5, loc="upper left", framealpha=0.92,
        labelspacing=0.35, borderpad=0.7, bbox_to_anchor=(0.01, 0.99),
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
    parser.add_argument("output_dir", help="Gadget output dir, e.g. ../S10_output_1024")
    parser.add_argument("--redshifts", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--rmax-factor", type=float, default=1.0,
                        help="Maximum radius as a multiple of R200")
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--outdir", default="radial_evolution")
    parser.add_argument("--summary", default="dust_radial_evolution.pdf")
    parser.add_argument("--summary-only", action="store_true",
                        help="Load cached NPZ files and remake only the summary plot")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {output_dir} for snapshots...")
    all_snaps = find_all_snapshots(output_dir)
    if not all_snaps:
        print("No snapshots with SubFind catalogs found.")
        sys.exit(1)
    print(f"Found {len(all_snaps)} snapshots (z={all_snaps[0]['z']:.2f} → z={all_snaps[-1]['z']:.2f})")

    epoch_data = []

    for z_target in sorted(args.redshifts, reverse=True):
        entry = dict(find_nearest_snapshot(all_snaps, z_target))
        entry["z_target"] = float(z_target)

        snap_num = entry["snap_num"]
        z_actual = entry["z"]

        # Cache by requested z as well as snapshot, so labels remain stable
        # even when the nearest available snapshot is z=1.9 for target z=2.
        z_label_str = f"{z_target:g}".replace(".", "p")
        tag = f"snap{snap_num:03d}_ztarget{z_label_str}_zactual{z_actual:.2f}"
        npz_out = outdir / f"radial_{tag}.npz"

        print(f"\nTarget z={z_target:g} → snap {snap_num:03d} z={z_actual:.3f}")

        if args.summary_only and npz_out.exists():
            print(f"  Loading cached data from {npz_out}")
            epoch = load_epoch_npz(npz_out)
            epoch["z_label"] = float(epoch.get("z_label", z_target))
            epoch["z_actual"] = float(epoch.get("z_actual", z_actual))
        else:
            try:
                epoch = extract_dust_radial_profile(
                    entry,
                    output_dir=output_dir,
                    rmax_factor=args.rmax_factor,
                    n_bins=args.n_bins,
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

    if not epoch_data:
        print("\nNo valid epochs — cannot make summary figure.")
        sys.exit(1)

    print(f"\nMaking summary figure ({len(epoch_data)} epochs)...")
    make_summary_figure(epoch_data, args.summary, args.rmax_factor)
    print("\nDone.")


if __name__ == "__main__":
    main()
