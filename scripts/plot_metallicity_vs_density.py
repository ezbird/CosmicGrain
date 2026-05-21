#!/usr/bin/env python3
"""
plot_metallicity_vs_density.py
-------------------------------
2-panel mass-weighted 2D histogram of gas-phase metallicity vs.
hydrogen number density for a Gadget-4 zoom simulation.

Panel layout
------------
  Top    : All gas in the high-resolution zoom region (all PartType0)
  Bottom : Gas within R_crit200 of the primary halo only

X-axis : n_H  [cm^-3]  (hydrogen number density, log scale)
Y-axis : Z/Z_sun       (gas-phase metallicity, log scale)

Color  : Mass-weighted particle count per bin (log scale)

Unit conversion
---------------
  Density (code) → n_H [cm^-3]:
      rho_cgs = rho_code × (UnitMass_in_g) / (UnitLength_in_cm)^3
      n_H     = rho_cgs × X_H / m_p

  Metallicity : PartType0/Metallicity is a dimensionless mass fraction Z.
      Z/Z_sun = Z / 0.0127   (Asplund+2009 proto-solar)

Usage
-----
python plot_metallicity_vs_density.py ../S10_output_1024/snapdir_049/snapshot_049 ../S10_output_1024/groups_049/fof_subhalo_tab_049.0.hdf5
      [--output plot.png] [--snap-num 49] [--log]

  The catalog argument is optional; if omitted the script searches for a
  groups_* directory adjacent to the snapshot directory.

References for phase boundaries drawn on the plot
--------------------------------------------------
  Cold neutral medium   : T < 10^3.9 K  (not shown here — pure density plot)
  Star-forming ISM      : n_H > n_crit (shown as vertical dashed line)
  Solar metallicity     : Z/Z_sun = 1   (shown as horizontal dashed line)
"""

import argparse
import glob
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Physical constants ───────────────────────────────────────────────────────
X_H       = 0.76                  # hydrogen mass fraction
M_PROTON  = 1.67262192e-24        # g
BOLTZMANN = 1.38065e-16           # erg/K
GAMMA     = 5.0 / 3.0
Z_SOLAR   = 0.0127                # Asplund+2009 proto-solar

# ─── Defaults ─────────────────────────────────────────────────────────────────
N_BINS     = 200
NH_LIM     = (-6.0,  4.0)         # log10(n_H / cm^-3)
Z_LIM      = (-6.0, -0.5)         # log10(Z)  [mass fraction]
T_LIM      = (2.0,   8.0)         # log10(T / K)
N_CRIT_DEFAULT = 0.13             # cm^-3  (CritPhysDensity proxy)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def iter_snap_files(snap_base):
    """Yield all HDF5 chunk files for a snapshot base path."""
    chunks = sorted(glob.glob(f"{snap_base}.*.hdf5"))
    if chunks:
        yield from chunks
    elif os.path.isfile(f"{snap_base}.hdf5"):
        yield f"{snap_base}.hdf5"
    else:
        raise FileNotFoundError(f"No snapshot files found for base: {snap_base}")


def read_header(snap_base):
    """Return the Header attrs dict from the first chunk."""
    for fpath in iter_snap_files(snap_base):
        with h5py.File(fpath, "r") as f:
            return dict(f["Header"].attrs)


def get_units(header):
    """Return (UnitMass_g, UnitLength_cm, UnitVelocity_cms, h, a)."""
    um  = float(header.get("UnitMass_in_g",      1.989e43))   # 10^10 M_sun
    ul  = float(header.get("UnitLength_in_cm",   3.085678e21)) # kpc
    uv  = float(header.get("UnitVelocity_in_cm_per_s", 1.0e5)) # km/s
    h   = float(header.get("HubbleParam", 0.6774))
    a   = float(header.get("Time",        1.0))
    return um, ul, uv, h, a


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog / halo-centre helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_catalog(snap_base, catalog_hint=None):
    """
    Locate the SubFind group catalog.
    Priority: explicit hint → groups_* dir next to snapdir → None.
    """
    if catalog_hint and os.path.isfile(catalog_hint):
        return catalog_hint

    snap_dir = os.path.dirname(os.path.abspath(snap_base))
    parent   = os.path.dirname(snap_dir)

    # e.g. snapdir_049 → groups_049
    snap_tag = os.path.basename(snap_dir).replace("snapdir_", "")
    for pattern in [
        os.path.join(parent, f"groups_{snap_tag}", "fof_subhalo_tab_*.hdf5"),
        os.path.join(parent, f"groups_{snap_tag}", "*.hdf5"),
        os.path.join(snap_dir, "..", f"groups_{snap_tag}", "*.hdf5"),
    ]:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]

    return None


def get_halo_center_r200(snap_base, catalog_hint=None):
    """
    Return (center_ckpch, r200_ckpch) in comoving kpc/h from the
    SubFind Group catalog (Group_R_Crit200 preferred, Mean200 fallback).
    Returns (None, None) if catalog is unavailable.
    """
    cat = find_catalog(snap_base, catalog_hint)
    if cat is None:
        print("  [warn] No group catalog found — bottom panel will be skipped.")
        return None, None

    try:
        with h5py.File(cat, "r") as f:
            if "Group" not in f:
                return None, None
            grp = f["Group"]
            if grp["GroupPos"].shape[0] == 0:
                return None, None

            center = grp["GroupPos"][0].astype(float)         # comoving kpc/h

            if "Group_R_Crit200" in grp:
                r200 = float(grp["Group_R_Crit200"][0])
            elif "Group_R_Mean200" in grp:
                r200 = float(grp["Group_R_Mean200"][0])
                print("  [info] Using Group_R_Mean200 (Crit200 not found)")
            else:
                print("  [warn] No R200 field in catalog.")
                return center, None

            return center, r200
    except Exception as e:
        print(f"  [warn] Catalog read error: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Gas data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_gas(snap_base, center_ckpch=None, r200_ckpch=None,
             um=None, ul=None, h=None, a=None, boxsize_ckpch=None, header=None):
    """
    Load PartType0 gas and return arrays:
        nH   : hydrogen number density [cm^-3]
        Zsun : metallicity in solar units
        mass : gas mass [code units]
        mask : boolean — within r200 sphere (only if center/r200 given)

    Applies a periodic boundary correction when computing distances.
    """
    nH_list   = []
    Zsun_list = []
    mass_list = []
    temp_list = []

    # Precompute density conversion factor:
    #   rho [code] × (um/ul^3) / (a^3)  →  rho_physical [g/cm^3]
    # Gadget-4 internal density is in comoving units: rho_code = rho_phys * a^3
    # so physical rho_cgs = rho_code * (um/ul^3) / a^3
    rho_to_cgs = (um / ul**3) / a**3

    # InternalEnergy is in code velocity^2 = (UnitVelocity_in_cm_per_s)^2
    uv  = float(header.get("UnitVelocity_in_cm_per_s", 1.0e5)) if header else 1.0e5
    u_to_cgs = uv**2

    for fpath in iter_snap_files(snap_base):
        with h5py.File(fpath, "r") as f:
            if "PartType0" not in f:
                continue
            pt0 = f["PartType0"]

            # ── Density → n_H ────────────────────────────────────────────────
            rho_code = pt0["Density"][:]
            rho_cgs  = rho_code * rho_to_cgs
            nH       = rho_cgs * X_H / M_PROTON       # cm^-3

            # ── Metallicity ───────────────────────────────────────────────────
            if "Metallicity" in pt0:
                Z = pt0["Metallicity"][:]
            elif "GFM_Metallicity" in pt0:
                Z = pt0["GFM_Metallicity"][:]
            elif "GFM_Metals" in pt0:
                arr = pt0["GFM_Metals"][:]
                Z   = np.sum(arr[:, 1:], axis=1) if arr.ndim == 2 else arr
            else:
                Z = np.zeros(len(rho_code), dtype=np.float32)

            Zsun = Z

            # ── Masses ───────────────────────────────────────────────────────
            mass = pt0["Masses"][:]

            # ── Temperature from InternalEnergy ───────────────────────────────
            if "Temperature" in pt0:
                temp = pt0["Temperature"][:]
            elif "InternalEnergy" in pt0:
                u_cgs = pt0["InternalEnergy"][:] * u_to_cgs
                # Mean molecular weight from ElectronAbundance (ne per H atom)
                if "ElectronAbundance" in pt0:
                    ne   = pt0["ElectronAbundance"][:]
                    mu   = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * ne)
                else:
                    mu   = np.full(len(u_cgs), 0.6)   # assume fully ionized
                temp = (GAMMA - 1.0) * u_cgs * mu * M_PROTON / BOLTZMANN
            else:
                temp = np.full(len(mass), 1e4, dtype=np.float32)

            nH_list.append(nH)
            Zsun_list.append(Zsun)
            mass_list.append(mass)
            temp_list.append(temp)

    if not nH_list:
        raise RuntimeError("No PartType0 gas found in snapshot.")

    nH   = np.concatenate(nH_list)
    Zsun = np.concatenate(Zsun_list)
    mass = np.concatenate(mass_list)
    temp = np.concatenate(temp_list)

    # ── Spatial mask for R200 panel ───────────────────────────────────────────
    mask_r200 = None
    if center_ckpch is not None and r200_ckpch is not None:
        pos_list = []
        for fpath in iter_snap_files(snap_base):
            with h5py.File(fpath, "r") as f:
                if "PartType0" in f:
                    pos_list.append(f["PartType0/Coordinates"][:])
        pos = np.concatenate(pos_list, axis=0)           # comoving kpc/h

        d = pos - center_ckpch[None, :]
        if boxsize_ckpch is not None:
            d -= np.round(d / boxsize_ckpch) * boxsize_ckpch
        r = np.sqrt((d * d).sum(axis=1))
        mask_r200 = (r <= r200_ckpch)

    return nH, Zsun, mass, temp, mask_r200


# ═══════════════════════════════════════════════════════════════════════════════
# 2D histogram helper
# ═══════════════════════════════════════════════════════════════════════════════

def make_hist2d(nH, Z, mass, temp, nH_lim, Z_lim, n_bins):
    """
    Build a mass-weighted mean temperature 2D map in log(nH) vs log(Z) space.
    Returns (T_mean, xedges, yedges) where T_mean is in Kelvin (NaN for empty bins).
    """
    log_nH = np.log10(np.clip(nH, 10**nH_lim[0], 10**nH_lim[1]))
    log_Z  = np.log10(np.clip(Z,  10**Z_lim[0],  10**Z_lim[1]))

    bins   = [n_bins, n_bins]
    ranges = [nH_lim, Z_lim]

    # mass-weighted temperature sum and mass sum per bin
    H_mT, xe, ye = np.histogram2d(log_nH, log_Z, bins=bins, range=ranges,
                                   weights=mass * temp)
    H_m,  _,  _  = np.histogram2d(log_nH, log_Z, bins=bins, range=ranges,
                                   weights=mass)

    T_mean = np.full_like(H_m, np.nan)
    ok = H_m > 0
    T_mean[ok] = H_mT[ok] / H_m[ok]

    return T_mean, xe, ye


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_panel(ax, H, xe, ye, cmap, norm, title,
               n_crit=N_CRIT_DEFAULT, z_r=0.0):
    """Draw one histogram panel with annotations."""
    # pcolormesh expects (x, y, C) where C = H^T because x is cols, y is rows
    im = ax.pcolormesh(xe, ye, H.T, cmap=cmap, norm=norm, rasterized=True)

    # ── Reference lines ───────────────────────────────────────────────────────
    #ax.axvline(np.log10(n_crit), color="black", ls="--", lw=1.2,
    #           alpha=0.8, label=rf"$n_{{\rm crit}}={n_crit:.2f}\ {{\rm cm}}^{{-3}}$")
    #ax.axhline(np.log10(Z_SOLAR), color="steelblue", ls="--", lw=1.2,
    #           alpha=0.8, label=r"$Z_\odot$")
    #if z_r > 0:
    #    ax.axhline(np.log10(z_r), color="green", ls=":", lw=1.2,
    #               alpha=0.7, label=rf"$z={z_r:.1f}$")

    ax.set_title(title, fontsize=11, pad=4)
    ax.set_ylabel(r"$\log Z$", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.6)

    return im


def make_plot(snap_base, catalog_hint, args):
    # ── Header & units ────────────────────────────────────────────────────────
    header   = read_header(snap_base)
    um, ul, uv, h, a = get_units(header)
    z        = 1.0 / a - 1.0
    boxsize  = float(header.get("BoxSize", 0.0))       # comoving kpc/h

    print(f"Snapshot z = {z:.3f}  (a = {a:.4f})")
    print(f"Units: UnitMass = {um:.3e} g,  UnitLength = {ul:.3e} cm,  h = {h:.4f}")

    # ── Halo centre ───────────────────────────────────────────────────────────
    center, r200 = get_halo_center_r200(snap_base, catalog_hint)
    if center is not None:
        print(f"Halo centre (ckpc/h): {center}")
        print(f"R_crit200 = {r200:.1f} ckpc/h  = {r200/h*a:.1f} pkpc")

    # ── Load gas ──────────────────────────────────────────────────────────────
    print("Loading PartType0 gas...")
    nH, Zsun, mass, temp, mask_r200 = load_gas(
        snap_base,
        center_ckpch=center,
        r200_ckpch=r200,
        um=um, ul=ul, h=h, a=a,
        boxsize_ckpch=boxsize,
        header=header,
    )
    print(f"  Total gas particles: {len(nH):,}")
    if mask_r200 is not None:
        print(f"  Within R200:        {mask_r200.sum():,}")

    # Guard against non-positive values before logging
    valid_all  = (nH > 0) & (Zsun > 0) & (temp > 0)
    nH_all     = nH[valid_all]
    Zsun_all   = Zsun[valid_all]
    mass_all   = mass[valid_all]
    temp_all   = temp[valid_all]

    if mask_r200 is not None:
        valid_r200  = valid_all & mask_r200
        nH_r200     = nH[valid_r200]
        Zsun_r200   = Zsun[valid_r200]
        mass_r200   = mass[valid_r200]
        temp_r200   = temp[valid_r200]
    else:
        nH_r200 = Zsun_r200 = mass_r200 = temp_r200 = None

    # ── Histograms ────────────────────────────────────────────────────────────
    T_all, xe, ye = make_hist2d(nH_all, Zsun_all, mass_all, temp_all,
                                args.nH_lim, args.Z_lim, args.bins)

    if nH_r200 is not None:
        T_r200, _, _ = make_hist2d(nH_r200, Zsun_r200, mass_r200, temp_r200,
                                   args.nH_lim, args.Z_lim, args.bins)

    # ── Colour normalisation: log10(T/K) ──────────────────────────────────────
    log_T_all  = np.log10(np.where(np.isfinite(T_all)  & (T_all  > 0), T_all,  np.nan))
    log_T_r200 = np.log10(np.where(np.isfinite(T_r200) & (T_r200 > 0), T_r200, np.nan)) \
                 if T_r200 is not None else None

    norm = mcolors.Normalize(vmin=args.T_lim[0], vmax=args.T_lim[1])

    # ── Figure ────────────────────────────────────────────────────────────────
    n_panels = 2 if nH_r200 is not None else 1
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(7, 4.5 * n_panels),
                             sharex=True,
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.RdYlBu_r   # cold=blue, hot=red — intuitive for temperature

    r200_pkpc = r200 / h * a if (r200 is not None and h > 0) else None
    r200_label = (f"R_{{\\rm 200c}} = {r200_pkpc:.0f}\\,{{\\rm pkpc}}"
                  if r200_pkpc else "R_{{\\rm 200c}}")

    # Top panel — full zoom region
    ax = axes[0]
    im = plot_panel(ax, log_T_all, xe, ye, cmap, norm,
                    title=f"Full high-res region — $z={z:.2f}$",
                    n_crit=args.n_crit)

    # Bottom panel — within R200
    if n_panels == 2:
        ax2 = axes[1]
        plot_panel(ax2, log_T_r200, xe, ye, cmap, norm,
                   title=f"Within ${r200_label}$ — $z={z:.2f}$",
                   n_crit=args.n_crit)
        ax2.set_xlabel(r"$\log (n_{\rm H}\ /\ {\rm cm}^{-3})$", fontsize=10)
    else:
        axes[0].set_xlabel(r"$\log (n_{\rm H}\ /\ {\rm cm}^{-3})$", fontsize=10)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log T (K)$", fontsize=10)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Gas-phase metallicity vs n_H phase plot (2-panel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("snap_base",
                   help="Snapshot base path, e.g. snapdir_049/snapshot_049")
    p.add_argument("catalog", nargs="?", default=None,
                   help="Path to SubFind catalog file (auto-detected if omitted)")
    p.add_argument("--output", "-o", default="metallicity_vs_density.png",
                   help="Output figure path")
    p.add_argument("--bins", type=int, default=N_BINS,
                   help="Number of bins along each axis")
    p.add_argument("--nH-min", type=float, default=NH_LIM[0], dest="nH_min",
                   help="log10(n_H) axis lower limit")
    p.add_argument("--nH-max", type=float, default=NH_LIM[1], dest="nH_max",
                   help="log10(n_H) axis upper limit")
    p.add_argument("--Z-min", type=float, default=Z_LIM[0], dest="Z_min",
                   help="log10(Z) axis lower limit")
    p.add_argument("--Z-max", type=float, default=Z_LIM[1], dest="Z_max",
                   help="log10(Z) axis upper limit")
    p.add_argument("--T-min", type=float, default=T_LIM[0], dest="T_min",
                   help="log10(T/K) colorbar lower limit")
    p.add_argument("--T-max", type=float, default=T_LIM[1], dest="T_max",
                   help="log10(T/K) colorbar upper limit")
    p.add_argument("--n-crit", type=float, default=N_CRIT_DEFAULT,
                   help="Star-formation density threshold [cm^-3] to mark")
    args = p.parse_args()

    args.nH_lim = (args.nH_min, args.nH_max)
    args.Z_lim  = (args.Z_min,  args.Z_max)
    args.T_lim  = (args.T_min,  args.T_max)
    return args


def main():
    args = parse_args()
    make_plot(args.snap_base, args.catalog, args)


if __name__ == "__main__":
    main()
