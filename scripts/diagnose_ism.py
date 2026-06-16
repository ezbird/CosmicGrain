#!/usr/bin/env python3
"""
diagnose_ism.py  —  CosmicGrain post-fix ISM diagnostic
========================================================
Produces a 3-panel figure:
  Panel 1: M_ISM (gas, nH>0.1 cm⁻³, r<20 pkpc) + M_★ vs time
  Panel 2: SFR vs time (from star birth times, binned)
  Panel 3: Phase diagram (T vs nH) at z=0 (and optionally z~1, z~2)

Usage:
  python diagnose_ism.py --snapdir /path/to/S10_output_1024 [--outdir ./figs]

Relies on halo_utils for centering (falls back to hardcoded 1024³ center if
halo_utils is not importable).
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROTON_MASS_CGS = 1.6726e-24   # g
BOLTZMANN_CGS   = 1.3806e-23   # J/K  (1.3806e-16 erg/K)
GAMMA           = 5.0 / 3.0
HYDROGEN_FRAC   = 0.76         # X_H, primordial-ish
UnitMass_in_g   = 1.989e43     # 1e10 M_sun in grams
UnitLength_in_cm= 3.085678e24  # 1 Mpc in cm
UnitVelocity_in_cms = 1e5      # 1 km/s in cm/s
UnitTime_in_s   = UnitLength_in_cm / UnitVelocity_in_cms

# ISM aperture
R_ISM_PKPC      = 20.0         # physical kpc
NH_ISM_THRESH   = 0.1          # cm⁻³

# Hardcoded 1024³ center (ckpc/h) from shrinking-sphere analysis
HALO569_CENTER_1024 = np.array([23048.920, 23163.650, 23699.611])  # ckpc/h
R200_CKPCH          = 85.95    # ckpc/h

# Phase diagram redshift targets (pick nearest available snapshot)
PHASE_REDSHIFTS = [2.0, 1.0, 0.5, 0.0]

# ---------------------------------------------------------------------------
# halo_utils import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from halo_utils import get_halo569_reference, get_halo569
    HAS_HALO_UTILS = True
    print("halo_utils imported successfully.")
except ImportError:
    HAS_HALO_UTILS = False
    print("halo_utils not found — using hardcoded 1024³ center.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def snap_files(snapdir, snapnum):
    """
    Return list of HDF5 files for a given snapshot number.
    Handles two layouts:
      - flat:      <snapdir>/snapshot_NNN.hdf5  or  snapshot_NNN.K.hdf5
      - subdir:    <snapdir>/snapdir_NNN/snapshot_NNN.K.hdf5
    """
    subdir = os.path.join(snapdir, f"snapdir_{snapnum:03d}")
    if os.path.isdir(subdir):
        files = sorted(glob.glob(os.path.join(subdir, f"snapshot_{snapnum:03d}.*.hdf5")))
        if files:
            return files
        # single-file variant inside subdir
        single = os.path.join(subdir, f"snapshot_{snapnum:03d}.hdf5")
        if os.path.exists(single):
            return [single]

    # flat layout fallback
    single = os.path.join(snapdir, f"snapshot_{snapnum:03d}.hdf5")
    if os.path.exists(single):
        return [single]
    files = sorted(glob.glob(os.path.join(snapdir, f"snapshot_{snapnum:03d}.*.hdf5")))
    return files


def list_snapshots(snapdir):
    """
    Return sorted list of snapshot numbers present in snapdir.
    Handles both flat and snapdir_NNN/ subdirectory layouts.
    """
    nums = set()
    # subdir layout: snapdir_NNN/
    for d in glob.glob(os.path.join(snapdir, "snapdir_???")):
        base = os.path.basename(d)
        try:
            nums.add(int(base.split("_")[1]))
        except (ValueError, IndexError):
            pass
    # flat layout fallback
    for f in (glob.glob(os.path.join(snapdir, "snapshot_???.hdf5")) +
              glob.glob(os.path.join(snapdir, "snapshot_???.0.hdf5"))):
        base = os.path.basename(f)
        try:
            nums.add(int(base.split("_")[1][:3]))
        except (ValueError, IndexError):
            pass
    return sorted(nums)


def read_header(snapdir, snapnum):
    """Read header from first file of snapshot; return dict."""
    files = snap_files(snapdir, snapnum)
    if not files:
        return None
    with h5py.File(files[0], "r") as f:
        h = dict(f["Header"].attrs)
        h["HubbleParam"] = float(f["Parameters"].attrs["HubbleParam"])
    return h


def read_gas(snapdir, snapnum):
    """
    Read PartType0 fields needed for ISM diagnostics.
    Returns dict of concatenated arrays (all chunks).
    Fields: Coordinates, Masses, InternalEnergy, Density, ElectronAbundance,
            NeutralHydrogenAbundance (if present), StarFormationRate (if present)
    """
    files = snap_files(snapdir, snapnum)
    if not files:
        return None

    arrays = {k: [] for k in [
        "Coordinates", "Masses", "InternalEnergy",
        "Density", "ElectronAbundance"
    ]}
    optional = ["NeutralHydrogenAbundance", "StarFormationRate"]
    opt_present = {k: None for k in optional}

    for fname in files:
        with h5py.File(fname, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            for key in arrays:
                if key in g:
                    arrays[key].append(g[key][:])
            for key in optional:
                if key in g:
                    if opt_present[key] is None:
                        opt_present[key] = []
                    opt_present[key].append(g[key][:])

    if not arrays["Coordinates"]:
        return None

    out = {k: np.concatenate(v) for k, v in arrays.items() if v}
    for key, val in opt_present.items():
        if val is not None:
            out[key] = np.concatenate(val)
    return out


def read_stars(snapdir, snapnum):
    """
    Read PartType4 fields: Coordinates, Masses, StellarFormationTime.
    Returns dict or None.
    """
    files = snap_files(snapdir, snapnum)
    if not files:
        return None
    coords, masses, sft = [], [], []
    for fname in files:
        with h5py.File(fname, "r") as f:
            if "PartType4" not in f:
                continue
            g = f["PartType4"]
            if "Coordinates" in g:
                coords.append(g["Coordinates"][:])
            if "Masses" in g:
                masses.append(g["Masses"][:])
            if "StellarFormationTime" in g:
                sft.append(g["StellarFormationTime"][:])
    if not coords:
        return None
    return {
        "Coordinates":           np.concatenate(coords),
        "Masses":                np.concatenate(masses),
        "StellarFormationTime":  np.concatenate(sft) if sft else None,
    }


def gas_temperature(u, xe):
    """
    Convert internal energy u [code units: (km/s)²] and electron abundance xe
    to temperature [K].
    """
    mu = 4.0 / (1.0 + 3.0 * HYDROGEN_FRAC + 4.0 * HYDROGEN_FRAC * xe)
    u_cgs = u * (UnitVelocity_in_cms ** 2)
    T = (GAMMA - 1.0) * mu * PROTON_MASS_CGS / BOLTZMANN_CGS * u_cgs
    return T


def gas_number_density(rho, h):
    """
    Convert comoving Gadget density [code units: 1e10 M_sun h² / Mpc³]
    to physical hydrogen number density [cm⁻³].
    rho: array in code units
    h:   HubbleParam
    Returns nH in cm⁻³ (physical).
    """
    # code density → g/cm³ (physical)
    rho_cgs = rho * (UnitMass_in_g * h**2) / (UnitLength_in_cm**3)
    nH = HYDROGEN_FRAC * rho_cgs / PROTON_MASS_CGS
    return nH


def periodic_distance(pos, center, boxsize):
    """Periodic distance from center; all in same units."""
    d = pos - center
    d = d - boxsize * np.round(d / boxsize)
    return np.sqrt((d**2).sum(axis=1))


def scale_to_time_gyr(a, h):
    """
    Approximate conversion of scale factor to cosmic time [Gyr].
    Assumes flat ΛCDM with Ω_m=0.3, Ω_Λ=0.7.
    """
    from scipy.integrate import quad
    H0 = h * 100.0 * 1e3 / (3.085678e22)  # s⁻¹
    Om, OL = 0.3, 0.7
    def integrand(ap):
        return 1.0 / (ap * np.sqrt(Om / ap**3 + OL))
    t, _ = quad(integrand, 0, a)
    return t / H0 / (3.15576e16)  # Gyr


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CosmicGrain ISM diagnostic")
    parser.add_argument("--snapdir", required=True, help="Snapshot directory")
    parser.add_argument("--outdir",  default=".",   help="Output figure directory")
    parser.add_argument("--every",   type=int, default=1,
                        help="Process every Nth snapshot (default: all)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    snapnums = list_snapshots(args.snapdir)
    if not snapnums:
        raise RuntimeError(f"No snapshots found in {args.snapdir}")
    print(f"Found {len(snapnums)} snapshots: {snapnums[0]} … {snapnums[-1]}")

    snapnums_use = snapnums[::args.every]

    # ------------------------------------------------------------------
    # Pass 1: Time series — M_ISM, M_★, SFR
    # ------------------------------------------------------------------
    times_gyr  = []
    redshifts  = []
    M_ISM_list = []
    M_star_list= []
    SFR_list   = []

    for sn in snapnums_use:
        hdr = read_header(args.snapdir, sn)
        if hdr is None:
            continue
        a   = float(hdr["Time"])
        z   = float(hdr["Redshift"])
        hub = float(hdr["HubbleParam"])
        box = float(hdr["BoxSize"])  # ckpc/h

        # Center
        if HAS_HALO_UTILS:
            try:
                ref = get_halo569_reference()
                cx, cy, cz = ref["center_ckpch"]
                center = np.array([cx, cy, cz])
            except Exception:
                center = HALO569_CENTER_1024
        else:
            center = HALO569_CENTER_1024

        # Physical R_ISM threshold in ckpc/h
        r_ism_ckpch = R_ISM_PKPC / a * hub  # pkpc → ckpc/h

        # --- Gas ---
        gas = read_gas(args.snapdir, sn)
        M_ism = 0.0
        sfr_snap = 0.0
        if gas is not None:
            r = periodic_distance(gas["Coordinates"], center, box)
            nH = gas_number_density(gas["Density"], hub)
            mask_ism = (r < r_ism_ckpch) & (nH > NH_ISM_THRESH)
            M_ism = gas["Masses"][mask_ism].sum() * 1e10 / hub  # M_sun
            if "StarFormationRate" in gas:
                sfr_snap = gas["StarFormationRate"].sum()  # M_sun/yr (Gadget native)

        # --- Stars ---
        stars = read_stars(args.snapdir, sn)
        M_star = 0.0
        if stars is not None:
            r_s = periodic_distance(stars["Coordinates"], center, box)
            r200_ckpch = R200_CKPCH
            mask_r200 = r_s < r200_ckpch
            M_star = stars["Masses"][mask_r200].sum() * 1e10 / hub

        try:
            t_gyr = scale_to_time_gyr(a, hub)
        except Exception:
            t_gyr = np.nan

        times_gyr.append(t_gyr)
        redshifts.append(z)
        M_ISM_list.append(M_ism)
        M_star_list.append(M_star)
        SFR_list.append(sfr_snap)

        print(f"  snap {sn:03d}  z={z:.3f}  t={t_gyr:.2f} Gyr  "
              f"M_ISM={M_ism:.3e}  M_*={M_star:.3e}  SFR={sfr_snap:.3f}")

    times_gyr  = np.array(times_gyr)
    M_ISM_arr  = np.array(M_ISM_list)
    M_star_arr = np.array(M_star_list)
    SFR_arr    = np.array(SFR_list)

    # ----------------------------------------------------------------
    # SFR history from sfr.txt (Gadget writes this natively; much
    # faster than re-reading star particles across all snapshots).
    # Columns: scale_factor  SFR[M_sun/yr]  ...
    # ----------------------------------------------------------------
    sfr_birth_times = None
    sfr_birth_sfr   = None
    sfr_txt = os.path.join(args.snapdir, "sfr.txt")
    if os.path.exists(sfr_txt):
        print(f"Reading SFR from {sfr_txt}")
        try:
            sfr_data = np.loadtxt(sfr_txt, comments="#")
            if sfr_data.ndim == 2 and sfr_data.shape[1] >= 2:
                a_sfr   = sfr_data[:, 0]
                sfr_raw = sfr_data[:, 1]
                hdr0 = read_header(args.snapdir, snapnums[0])
                hub0 = float(hdr0["HubbleParam"]) if hdr0 else 0.6774
                sfr_birth_times = np.array([scale_to_time_gyr(float(ai), hub0)
                                            for ai in a_sfr])
                sfr_birth_sfr   = sfr_raw
                print(f"  sfr.txt: {len(sfr_birth_times)} entries, "
                      f"SFR range {sfr_raw.min():.2f}-{sfr_raw.max():.2f} M_sun/yr")
        except Exception as e:
            print(f"Warning: sfr.txt parse failed: {e}")
    else:
        print("sfr.txt not found — falling back to instantaneous SFR from snapshots.")

    # ------------------------------------------------------------------
    # Pass 2: Phase diagrams at target redshifts
    # ------------------------------------------------------------------
    phase_snaps = {}
    for z_target in PHASE_REDSHIFTS:
        # find nearest snapshot
        best_sn, best_dz = None, 1e9
        for sn in snapnums:
            hdr = read_header(args.snapdir, sn)
            if hdr is None:
                continue
            dz = abs(float(hdr["Redshift"]) - z_target)
            if dz < best_dz:
                best_dz = dz
                best_sn = sn
        if best_sn is not None and best_dz < 0.3:
            phase_snaps[z_target] = best_sn

    phase_data = {}
    for z_target, sn in phase_snaps.items():
        hdr = read_header(args.snapdir, sn)
        a   = float(hdr["Time"])
        hub = float(hdr["HubbleParam"])
        box = float(hdr["BoxSize"])
        center = HALO569_CENTER_1024
        r200_ckpch = R200_CKPCH

        gas = read_gas(args.snapdir, sn)
        if gas is None:
            continue
        r   = periodic_distance(gas["Coordinates"], center, box)
        mask = r < r200_ckpch
        nH  = gas_number_density(gas["Density"][mask], hub)
        T   = gas_temperature(gas["InternalEnergy"][mask],
                              gas["ElectronAbundance"][mask])
        # ISM flag for coloring
        r_ism_ckpch = R_ISM_PKPC / a * hub
        is_ism = (r[mask] < r_ism_ckpch) & (nH > NH_ISM_THRESH)
        phase_data[z_target] = dict(nH=nH, T=T, is_ism=is_ism,
                                    z_actual=float(hdr["Redshift"]))
        print(f"Phase z≈{z_target}: snap {sn}, {mask.sum()} gas particles within R200")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor("#0d1117")

    # Color palette
    C_ISM  = "#4fc3f7"   # light blue  — ISM gas / M_ISM
    C_STAR = "#ffb74d"   # amber       — stellar mass
    C_SFR  = "#81c995"   # green       — SFR
    C_BG   = "#0d1117"
    C_AX   = "#1c2433"
    C_TEXT = "#cdd9e5"
    C_GRID = "#2d3748"

    plt.rcParams.update({
        "text.color":         C_TEXT,
        "axes.labelcolor":    C_TEXT,
        "xtick.color":        C_TEXT,
        "ytick.color":        C_TEXT,
        "axes.edgecolor":     C_GRID,
        "figure.facecolor":   C_BG,
        "axes.facecolor":     C_AX,
    })

    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.1, 1],
        hspace=0.38, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.08
    )

    ax_mass = fig.add_subplot(gs[0, :2])
    ax_sfr  = fig.add_subplot(gs[0, 2:])

    # Phase panels — up to 4, using available snapshots
    phase_keys = sorted(phase_data.keys(), reverse=True)[:4]
    phase_axes = [fig.add_subplot(gs[1, i]) for i in range(len(phase_keys))]

    # --- Panel 1: M_ISM + M_★ ---
    valid = np.isfinite(times_gyr)
    ax_mass.plot(times_gyr[valid], M_ISM_arr[valid] / 1e9,
                 color=C_ISM,  lw=2.5, label=r"$M_\mathrm{ISM}$ (gas, $r<20\,\mathrm{pkpc}$, $n_H>0.1$)")
    ax_mass.plot(times_gyr[valid], M_star_arr[valid] / 1e9,
                 color=C_STAR, lw=2.5, linestyle="--", label=r"$M_\star$ ($r < R_{200}$)")

    # MW reference bands
    ax_mass.axhspan(5, 8,  alpha=0.08, color=C_ISM,  label="MW ISM range")
    ax_mass.axhspan(30, 60, alpha=0.08, color=C_STAR, label="MW $M_\\star$ range")

    ax_mass.set_xlabel("Cosmic time [Gyr]", fontsize=11)
    ax_mass.set_ylabel(r"Mass [$10^9\,M_\odot$]", fontsize=11)
    ax_mass.set_title("ISM Gas + Stellar Mass History", fontsize=12, pad=6)
    ax_mass.legend(fontsize=8.5, loc="upper left", framealpha=0.3)
    ax_mass.set_ylim(bottom=0)

    # Add z axis on top
    ax2 = ax_mass.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5]
    t_ticks = []
    hub_ref = 0.6774
    for zt in z_ticks:
        try:
            t_ticks.append(scale_to_time_gyr(1.0/(1+zt), hub_ref))
        except Exception:
            t_ticks.append(np.nan)
    ax2.set_xlim(ax_mass.get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels([str(z) for z in z_ticks], fontsize=8)
    ax2.set_xlabel("Redshift", fontsize=9, labelpad=4)
    ax2.tick_params(colors=C_TEXT)

    # --- Panel 2: SFR ---
    if sfr_birth_times is not None:
        ax_sfr.fill_between(sfr_birth_times, sfr_birth_sfr,
                            alpha=0.25, color=C_SFR)
        ax_sfr.plot(sfr_birth_times, sfr_birth_sfr,
                    color=C_SFR, lw=2.0, label="SFR (birth times)")
    else:
        # Fall back to on-the-fly SFR field
        ax_sfr.plot(times_gyr[valid], SFR_arr[valid],
                    color=C_SFR, lw=2.0, label="SFR (instantaneous)")

    ax_sfr.axhspan(1, 3, alpha=0.1, color=C_SFR, label="MW SFR range")
    ax_sfr.set_xlabel("Cosmic time [Gyr]", fontsize=11)
    ax_sfr.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]", fontsize=11)
    ax_sfr.set_title("Star Formation Rate History", fontsize=12, pad=6)
    ax_sfr.legend(fontsize=8.5, loc="upper right", framealpha=0.3)
    ax_sfr.set_ylim(bottom=0)

    ax3 = ax_sfr.twiny()
    ax3.set_xlim(ax_sfr.get_xlim())
    ax3.set_xticks(t_ticks)
    ax3.set_xticklabels([str(z) for z in z_ticks], fontsize=8)
    ax3.set_xlabel("Redshift", fontsize=9, labelpad=4)
    ax3.tick_params(colors=C_TEXT)

    # --- Phase diagram panels ---
    phase_cmap_ism = matplotlib.colormaps.get_cmap("plasma")
    phase_cmap_cgm = matplotlib.colormaps.get_cmap("Blues")

    for ax, z_key in zip(phase_axes, phase_keys):
        pd = phase_data[z_key]
        nH, T, is_ism = pd["nH"], pd["T"], pd["is_ism"]
        z_actual = pd["z_actual"]

        log_nH = np.log10(np.clip(nH, 1e-8, None))
        log_T  = np.log10(np.clip(T,  10,   None))

        # CGM first (background), ISM on top
        ax.hexbin(log_nH[~is_ism], log_T[~is_ism],
                  gridsize=60, mincnt=1, bins="log",
                  cmap="Blues", alpha=0.6,
                  extent=[-7, 4, 2, 8])
        ax.hexbin(log_nH[is_ism], log_T[is_ism],
                  gridsize=60, mincnt=1, bins="log",
                  cmap="YlOrRd", alpha=0.9,
                  extent=[-7, 4, 2, 8])

        # ISM threshold lines
        ax.axvline(np.log10(NH_ISM_THRESH), color="#4fc3f7",
                   lw=1.0, ls="--", alpha=0.6)
        ax.axhline(np.log10(1e4), color="#aaa", lw=0.8, ls=":", alpha=0.5)
        ax.axhline(np.log10(1e5), color="#aaa", lw=0.8, ls=":", alpha=0.5)

        ax.set_xlim(-7, 4)
        ax.set_ylim(2, 8)
        ax.set_xlabel(r"$\log_{10}(n_H\,[\mathrm{cm}^{-3}])$", fontsize=9)
        ax.set_ylabel(r"$\log_{10}(T\,[\mathrm{K}])$",         fontsize=9)
        ax.set_title(f"Phase diagram  z = {z_actual:.2f}", fontsize=10, pad=5)

        # Annotation: ISM particle count
        n_ism = is_ism.sum()
        ax.text(0.97, 0.96, f"N_ISM = {n_ism:,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=C_ISM)

    fig.suptitle(
        "CosmicGrain  |  1024³ S10 post-fix  |  Halo 569  —  ISM Diagnostic",
        fontsize=13, color=C_TEXT, y=0.97
    )

    outpath = os.path.join(args.outdir, "diagnose_ism_1024_S10.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=C_BG)
    print(f"\nFigure saved → {outpath}")

    # Also dump time series to CSV for easy inspection
    csv_path = os.path.join(args.outdir, "diagnose_ism_timeseries.csv")
    with open(csv_path, "w") as fcsv:
        fcsv.write("snap,redshift,t_gyr,M_ISM_Msun,M_star_Msun,SFR_Msunyr\n")
        for i, sn in enumerate(snapnums_use):
            fcsv.write(f"{sn},{redshifts[i]:.4f},{times_gyr[i]:.4f},"
                       f"{M_ISM_list[i]:.4e},{M_star_list[i]:.4e},"
                       f"{SFR_list[i]:.4f}\n")
    print(f"Time series CSV → {csv_path}")


if __name__ == "__main__":
    main()
