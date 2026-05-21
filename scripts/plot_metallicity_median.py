#!/usr/bin/env python3
"""
plot_metallicity_median.py
---------------------------
2-panel running-median plot of gas-phase metallicity vs. hydrogen number
density, with 16th–84th percentile shaded envelope.

Panel layout
------------
  Top    : All gas in the high-resolution zoom region (all PartType0)
  Bottom : Gas within R_crit200 of the primary halo only

X-axis : log10(n_H  / cm^-3)
Y-axis : log10(Z)   [mass fraction, log scale]

Usage
-----
  python plot_metallicity_median.py \\
      /path/to/snapdir_049/snapshot_049 \\
      /path/to/groups_049/fof_subhalo_tab_049.0.hdf5 \\
      [--output plot.png] [--bins 80]
"""

import argparse
import glob
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Physical constants ───────────────────────────────────────────────────────
X_H      = 0.76
M_PROTON = 1.67262192e-24        # g
Z_SOLAR  = 0.0127                # Asplund+2009 proto-solar

# ─── Defaults ─────────────────────────────────────────────────────────────────
NH_LIM         = (-6.0,  4.0)    # log10(n_H / cm^-3)
Z_LIM          = (-6.0, -0.5)    # log10(Z)
N_BINS         = 80              # bins along the x-axis for the running median
N_MIN_PER_BIN  = 10              # minimum particles to draw a median point
N_CRIT_DEFAULT = 0.13            # cm^-3


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers  (identical to plot_metallicity_vs_density.py)
# ═══════════════════════════════════════════════════════════════════════════════

def iter_snap_files(snap_base):
    chunks = sorted(glob.glob(f"{snap_base}.*.hdf5"))
    if chunks:
        yield from chunks
    elif os.path.isfile(f"{snap_base}.hdf5"):
        yield f"{snap_base}.hdf5"
    else:
        raise FileNotFoundError(f"No snapshot files found for base: {snap_base}")


def read_header(snap_base):
    for fpath in iter_snap_files(snap_base):
        with h5py.File(fpath, "r") as f:
            return dict(f["Header"].attrs)


def get_units(header):
    um = float(header.get("UnitMass_in_g",           1.989e43))
    ul = float(header.get("UnitLength_in_cm",        3.085678e21))
    uv = float(header.get("UnitVelocity_in_cm_per_s",1.0e5))
    h  = float(header.get("HubbleParam",             0.6774))
    a  = float(header.get("Time",                    1.0))
    return um, ul, uv, h, a


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_catalog(snap_base, catalog_hint=None):
    if catalog_hint and os.path.isfile(catalog_hint):
        return catalog_hint
    snap_dir = os.path.dirname(os.path.abspath(snap_base))
    parent   = os.path.dirname(snap_dir)
    snap_tag = os.path.basename(snap_dir).replace("snapdir_", "")
    for pattern in [
        os.path.join(parent, f"groups_{snap_tag}", "fof_subhalo_tab_*.hdf5"),
        os.path.join(parent, f"groups_{snap_tag}", "*.hdf5"),
    ]:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def get_halo_center_r200(snap_base, catalog_hint=None):
    cat = find_catalog(snap_base, catalog_hint)
    if cat is None:
        print("  [warn] No group catalog found — bottom panel will be skipped.")
        return None, None
    try:
        with h5py.File(cat, "r") as f:
            if "Group" not in f or f["Group"]["GroupPos"].shape[0] == 0:
                return None, None
            grp    = f["Group"]
            center = grp["GroupPos"][0].astype(float)
            if "Group_R_Crit200" in grp:
                r200 = float(grp["Group_R_Crit200"][0])
            elif "Group_R_Mean200" in grp:
                r200 = float(grp["Group_R_Mean200"][0])
                print("  [info] Using Group_R_Mean200")
            else:
                return center, None
            return center, r200
    except Exception as e:
        print(f"  [warn] Catalog read error: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Gas loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_gas(snap_base, center_ckpch=None, r200_ckpch=None,
             um=None, ul=None, h=None, a=None, boxsize_ckpch=None):
    """Return (log_nH, log_Z, mask_r200) as flat arrays."""

    rho_to_cgs = (um / ul**3) / a**3

    log_nH_list = []
    log_Z_list  = []

    for fpath in iter_snap_files(snap_base):
        with h5py.File(fpath, "r") as f:
            if "PartType0" not in f:
                continue
            pt0 = f["PartType0"]

            # n_H
            rho_cgs = pt0["Density"][:] * rho_to_cgs
            nH      = rho_cgs * X_H / M_PROTON
            good_nH = nH > 0

            # Z
            if "Metallicity" in pt0:
                Z = pt0["Metallicity"][:]
            elif "GFM_Metallicity" in pt0:
                Z = pt0["GFM_Metallicity"][:]
            elif "GFM_Metals" in pt0:
                arr = pt0["GFM_Metals"][:]
                Z   = np.sum(arr[:, 1:], axis=1) if arr.ndim == 2 else arr
            else:
                Z = np.zeros(len(nH), dtype=np.float32)

            valid = good_nH & (Z > 0)
            log_nH_list.append(np.log10(nH[valid]))
            log_Z_list.append(np.log10(Z[valid]))

    if not log_nH_list:
        raise RuntimeError("No PartType0 gas found in snapshot.")

    log_nH = np.concatenate(log_nH_list)
    log_Z  = np.concatenate(log_Z_list)

    # ── Spatial mask ──────────────────────────────────────────────────────────
    mask_r200 = None
    if center_ckpch is not None and r200_ckpch is not None:
        pos_list = []
        # We need positions for ALL particles (including those with Z=0 filtered
        # above), so we rebuild a full valid index.  Simpler: reload coords only.
        n_valid_per_file = []
        valid_flags = []
        for fpath in iter_snap_files(snap_base):
            with h5py.File(fpath, "r") as f:
                if "PartType0" not in f:
                    continue
                pt0    = f["PartType0"]
                rho    = pt0["Density"][:]
                nH_    = rho * rho_to_cgs * X_H / M_PROTON
                if "Metallicity" in pt0:
                    Z_ = pt0["Metallicity"][:]
                elif "GFM_Metallicity" in pt0:
                    Z_ = pt0["GFM_Metallicity"][:]
                elif "GFM_Metals" in pt0:
                    arr = pt0["GFM_Metals"][:]
                    Z_  = np.sum(arr[:, 1:], axis=1) if arr.ndim == 2 else arr
                else:
                    Z_  = np.zeros(len(rho))
                v = (nH_ > 0) & (Z_ > 0)
                valid_flags.append(v)
                pos_list.append(pt0["Coordinates"][:][v])

        pos = np.concatenate(pos_list, axis=0)
        d   = pos - center_ckpch[None, :]
        if boxsize_ckpch:
            d -= np.round(d / boxsize_ckpch) * boxsize_ckpch
        r         = np.sqrt((d * d).sum(axis=1))
        mask_r200 = r <= r200_ckpch

    return log_nH, log_Z, mask_r200


# ═══════════════════════════════════════════════════════════════════════════════
# Running median / percentiles
# ═══════════════════════════════════════════════════════════════════════════════

def running_median(log_nH, log_Z, nH_lim, n_bins, n_min):
    """
    Bin log_nH into n_bins equal-width bins over nH_lim.
    Return (bin_centres, median, p16, p84) — NaN where fewer than n_min particles.
    """
    edges   = np.linspace(nH_lim[0], nH_lim[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.searchsorted(edges, log_nH, side="right") - 1
    idx     = np.clip(idx, 0, n_bins - 1)

    med = np.full(n_bins, np.nan)
    p16 = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)

    for i in range(n_bins):
        vals = log_Z[idx == i]
        if len(vals) >= n_min:
            med[i] = np.median(vals)
            p16[i] = np.percentile(vals, 25)
            p84[i] = np.percentile(vals, 75)

    return centres, med, p16, p84


# ═══════════════════════════════════════════════════════════════════════════════
# Panel drawing
# ═══════════════════════════════════════════════════════════════════════════════

def draw_panel(ax, log_nH, log_Z, nH_lim, Z_lim, n_bins, n_min,
               n_crit, title, color="green"):
    centres, med, p16, p84 = running_median(log_nH, log_Z, nH_lim, n_bins, n_min)

    valid = np.isfinite(med)
    ax.fill_between(centres[valid], p16[valid], p84[valid],
                    color=color, alpha=0.25, label="25/75 pct")
    ax.plot(centres[valid], med[valid],
            color=color, lw=2.0, label="Median")

    # Reference lines
    ax.axvline(np.log10(n_crit), color="gray", ls="--", lw=1.0,
               label=rf"$n_H={n_crit:.2f}\ {{\rm cm}}^{{-3}}$")
    #ax.axhline(np.log10(Z_SOLAR), color="tomato", ls=":", lw=1.2,
    #           label=r"$Z_\odot$")

    ax.set_xlim(nH_lim)
    ax.set_ylim(Z_lim)
    ax.set_ylabel(r"$\log Z$", fontsize=11)
    ax.set_title(title, fontsize=11, pad=4)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="upper left", framealpha=0.7)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def make_plot(snap_base, catalog_hint, args):
    header            = read_header(snap_base)
    um, ul, uv, h, a  = get_units(header)
    z                 = 1.0 / a - 1.0
    boxsize           = float(header.get("BoxSize", 0.0))

    print(f"Snapshot z = {z:.3f}  (a = {a:.4f})")

    center, r200 = get_halo_center_r200(snap_base, catalog_hint)
    if center is not None:
        print(f"Halo centre (ckpc/h): {center}")
        print(f"R_crit200 = {r200:.1f} ckpc/h  = {r200/h*a:.1f} pkpc")

    print("Loading PartType0 gas...")
    log_nH, log_Z, mask_r200 = load_gas(
        snap_base,
        center_ckpch=center, r200_ckpch=r200,
        um=um, ul=ul, h=h, a=a,
        boxsize_ckpch=boxsize,
    )
    print(f"  Valid gas particles : {len(log_nH):,}")
    if mask_r200 is not None:
        print(f"  Within R200        : {mask_r200.sum():,}")

    # ── Figure ────────────────────────────────────────────────────────────────
    n_panels = 2 if mask_r200 is not None else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(7, 4.0 * n_panels),
                             sharex=True,
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # Top — full zoom region
    draw_panel(axes[0], log_nH, log_Z,
               args.nH_lim, args.Z_lim, args.bins, args.n_min,
               args.n_crit,
               title=f"2048^3 -- Full high-res region — $z={z:.2f}$",
               color="steelblue")

    # Bottom — within R200
    if n_panels == 2:
        r200_pkpc = r200 / h * a
        r200_label = rf"R_{{\rm 200c}} = {r200_pkpc:.0f}\,{{\rm pkpc}}"
        draw_panel(axes[1], log_nH[mask_r200], log_Z[mask_r200],
                   args.nH_lim, args.Z_lim, args.bins, args.n_min,
                   args.n_crit,
                   title=f"Within ${r200_label}$ — $z={z:.2f}$",
                   color="darkorange")
        axes[1].set_xlabel(r"$\log (n_{\rm H}\ /\ {\rm cm}^{3})$", fontsize=11)
    else:
        axes[0].set_xlabel(r"$\log (n_{\rm H}\ /\ {\rm cm}^{3})$", fontsize=11)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Running-median gas metallicity vs n_H (2-panel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("snap_base",
                   help="Snapshot base path, e.g. snapdir_049/snapshot_049")
    p.add_argument("catalog", nargs="?", default=None,
                   help="SubFind catalog file (auto-detected if omitted)")
    p.add_argument("--output", "-o", default="metallicity_median.png")
    p.add_argument("--bins",   type=int,   default=N_BINS,
                   help="Number of x-axis bins for the running median")
    p.add_argument("--n-min",  type=int,   default=N_MIN_PER_BIN,
                   help="Min particles per bin to draw a point")
    p.add_argument("--nH-min", type=float, default=NH_LIM[0], dest="nH_min")
    p.add_argument("--nH-max", type=float, default=NH_LIM[1], dest="nH_max")
    p.add_argument("--Z-min",  type=float, default=Z_LIM[0],  dest="Z_min")
    p.add_argument("--Z-max",  type=float, default=Z_LIM[1],  dest="Z_max")
    p.add_argument("--n-crit", type=float, default=N_CRIT_DEFAULT,
                   help="Star-formation density threshold [cm^-3]")
    args = p.parse_args()
    args.nH_lim = (args.nH_min, args.nH_max)
    args.Z_lim  = (args.Z_min,  args.Z_max)
    return args


def main():
    args = parse_args()
    make_plot(args.snap_base, args.catalog, args)


if __name__ == "__main__":
    main()
