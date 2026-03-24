#!/usr/bin/env python3
"""
plot_dz_vs_metallicity.py
--------------------------
Publication-quality D/Z vs. gas-phase metallicity for a Gadget-4 zoom
simulation with N-body dust (PartType6).

METHOD
------
For each gas particle, D/Z is estimated by summing dust (PartType6) mass
within that particle's SPH smoothing length, then dividing by the local
metal mass.  This is the physically correct approach for a subgrid-ISM
simulation: we cannot compare D/Z vs n_H (the EOS prevents high densities)
but D/Z vs Z is independent of the ISM density structure and is the
standard diagnostic used by McKinnon+2017, Aoyama+2018, Davé+2019, etc.

LITERATURE COMPARISONS
----------------------
Observations (galaxy-integrated):
  Rémy-Ruyer+2014 (A&A 563, A31)  — 126 DGS+KINGFISH galaxies, BPL fit
  De Vis+2019 (A&A 623, A5)       — extended DustPedia+RR14 compilation

Simulations (subgrid-ISM class, comparable to this work):
  McKinnon+2017 (MNRAS 468, 1505) — AREPO, moving-mesh, sputtering
  Aoyama+2018  (MNRAS 478, 4905) — Gadget, two-size grain distribution
  Davé+2019    (MNRAS 486, 2827) — SIMBA/GIZMO

NOTE: All three simulation comparisons use a subgrid EOS (no explicit cold
ISM), so their D/Z vs Z trends are directly comparable to this work.

Usage:
    python plot_dz_vs_metallicity.py snapshot_base catalog [--r-max kpc]
                                     [--output file.png]
"""

import argparse
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    raise ImportError("halo_utils not found — ensure halo_utils.py is in PATH")

# ============================================================
# Physical constants
# ============================================================
Z_SOLAR  = 0.0134          # Asplund+2009 solar metallicity (mass fraction)
OH_SOLAR = 8.69            # 12+log(O/H) solar (Asplund+2009)

# ============================================================
# Literature: Rémy-Ruyer+2014
# BPL parameters from their Table 1, XCO,MW scenario (most widely cited)
#   log10(G/D) = log10(G/D)_0  + α_high * Δ(O/H)   for 12+log(O/H) ≥ 8.0
#              = ...           + α_low  * Δ(O/H)   for 12+log(O/H) < 8.0
#   with 12+log(O/H) expressed relative to solar
# From Table 1: log(G/D)_0 = 2.21, α_high = 1.0, α_low = 3.15, break = 8.0
# ============================================================
RR14_LOGDG0  = -2.21      # log10(D/G) at solar metallicity
RR14_ALPHA_H =  1.0       # slope above break (linear with metallicity)
RR14_ALPHA_L =  3.15      # slope below break (steep at low Z)
RR14_OH_BREAK = 8.0       # 12+log(O/H) at the break

def rr14_dz(Z_mass_frac):
    """
    Rémy-Ruyer+2014 broken-power-law D/Z as function of metal mass fraction.
    Uses XCO,MW parameters (Table 1).
    """
    Z   = np.atleast_1d(np.asarray(Z_mass_frac, float))
    oh  = OH_SOLAR + np.log10(np.clip(Z / Z_SOLAR, 1e-8, None))
    Δoh = oh - OH_SOLAR
    # Above break: slope α_high; below break: pivot off break point then α_low
    log_dg = np.where(
        oh >= RR14_OH_BREAK,
        RR14_LOGDG0 + RR14_ALPHA_H * Δoh,
        RR14_LOGDG0 + RR14_ALPHA_H * (RR14_OH_BREAK - OH_SOLAR)
                    + RR14_ALPHA_L * (oh - RR14_OH_BREAK))
    dg = 10.0 ** log_dg
    return dg / Z   # D/G → D/Z

# Approximate 1-sigma scatter around RR14 BPL (read from their Fig. 1)
# expressed as ±dex in D/Z, approximately constant at ~0.4 dex
RR14_SCATTER_DEX = 0.45

# ============================================================
# Literature: De Vis+2019 (A&A 623, A5) — DustPedia compilation
# Representative data digitized from their Fig. 9 (DTM vs metallicity)
# Converted from 12+log(O/H) to Z mass fraction using O/H → Z proxy
# ============================================================
_dv19_oh  = np.array([7.3, 7.5, 7.7, 7.9, 8.1, 8.2, 8.35, 8.5,
                       8.6, 8.69, 8.75, 8.85])
_dv19_dtm = np.array([0.010, 0.018, 0.030, 0.055, 0.095, 0.130, 0.180,
                       0.250, 0.320, 0.380, 0.400, 0.420])
DEVIS19_Z  = Z_SOLAR * 10.0**(_dv19_oh - OH_SOLAR)
DEVIS19_DZ = _dv19_dtm

# ============================================================
# Literature simulation median trends
# All digitized from published figures; subgrid-ISM class only.
# McKinnon+2017 Fig. 7 (AREPO, z=0 galaxies):
# ============================================================
_mk17_z  = np.array([0.001, 0.002, 0.004, 0.007, 0.012, 0.020, 0.030, 0.040])
_mk17_dz = np.array([0.010, 0.020, 0.045, 0.100, 0.190, 0.290, 0.360, 0.400])
MK17_Z   = _mk17_z
MK17_DZ  = _mk17_dz

# Aoyama+2018 Fig. 4 (Gadget two-size, z=0):
_ao18_z  = np.array([0.0005, 0.001, 0.003, 0.006, 0.010, 0.018, 0.030, 0.040])
_ao18_dz = np.array([0.005,  0.012, 0.035, 0.090, 0.170, 0.280, 0.370, 0.410])
AO18_Z   = _ao18_z
AO18_DZ  = _ao18_dz

# Davé+2019 SIMBA Fig. 6 (GIZMO, subgrid ISM, z=0 median):
_dv19s_z  = np.array([0.0003, 0.001, 0.003, 0.007, 0.013, 0.020, 0.030, 0.040])
_dv19s_dz = np.array([0.003,  0.010, 0.030, 0.090, 0.180, 0.280, 0.370, 0.420])
DAVE19_Z  = _dv19s_z
DAVE19_DZ = _dv19s_dz


# ============================================================
# Snapshot utilities
# ============================================================

def get_units(snapshot_base):
    files = sorted(glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = [snapshot_base + ".hdf5"]
    with h5py.File(files[0], "r") as f:
        h = f["Header"].attrs
        return dict(
            a            = float(h.get("Time", 1.0)),
            hubble       = float(h.get("HubbleParam", 0.6774)),
            unit_mass_g  = float(h.get("UnitMass_in_g",        1.989e43)),
            unit_len_cm  = float(h.get("UnitLength_in_cm",     3.085678e21)),
            unit_vel_cms = float(h.get("UnitVelocity_in_cm_per_s", 1e5)),
        )


def read_gas_spatially(snap_base, center_phys, r_max_kpc, units):
    """
    Read PartType0 within r_max_kpc of center_phys (physical kpc).
    center_phys must already be in physical kpc.
    Returns coords [kpc], mass [Msun], metallicity [fraction], hsml [kpc].
    """
    a, h  = units["a"], units["hubble"]
    kpc   = units["unit_len_cm"] / 3.085678e21 * a / h
    msun  = units["unit_mass_g"] / 1.989e33 / h

    files = sorted(glob.glob(f"{snap_base}.*.hdf5"))
    if not files:
        files = [snap_base + ".hdf5"]

    c_l, m_l, Z_l, hsml_l = [], [], [], []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "PartType0" not in f:
                continue
            pt0  = f["PartType0"]
            cord = pt0["Coordinates"][:] * kpc
            r    = np.sqrt(np.sum((cord - center_phys)**2, axis=1))
            mask = r < r_max_kpc
            if not mask.any():
                continue
            c_l.append(cord[mask])
            m_l.append(pt0["Masses"][:][mask] * msun)
            Z_l.append(pt0["Metallicity"][:][mask]
                        if "Metallicity" in pt0 else np.zeros(mask.sum()))
            hsml_l.append(pt0["SmoothingLength"][:][mask] * kpc
                           if "SmoothingLength" in pt0 else None)

    if not c_l:
        raise RuntimeError(
            f"No PartType0 within {r_max_kpc} kpc of {center_phys}.\n"
            "Check that halo_pos is returned in code units by load_target_halo.")

    coords = np.vstack(c_l)
    mass   = np.concatenate(m_l)
    Z      = np.concatenate(Z_l)
    hsml   = np.concatenate(hsml_l) if hsml_l[0] is not None else None
    print(f"  Gas particles: {len(mass):,}  "
          f"(Z range: {Z.min():.2e}–{Z.max():.2e})")
    return coords, mass, Z, hsml


# ============================================================
# Per-gas-particle D/Z via KD-tree on dust particles
# ============================================================

def assign_dz_kdtree(gas_coords, gas_mass, gas_Z, gas_hsml,
                     dust_coords, dust_mass):
    """
    For each gas particle, sum dust mass within its SPH smoothing length.
    Returns per-particle DZ array (same length as gas_mass).
    Particles with no metallicity, no smoothing length, or zero kernel mass
    get DZ=0 and are excluded by the caller.
    """
    if gas_hsml is None or np.all(gas_hsml == 0):
        # Fallback: geometric mean nearest-neighbour spacing
        n       = max(len(gas_mass), 1)
        span    = np.ptp(gas_coords, axis=0)
        vol     = max(span[0] * span[1] * span[2], 1.0)
        fallback = (vol / n) ** (1.0 / 3.0)
        gas_hsml = np.full(len(gas_mass), fallback)
        print(f"  No smoothing lengths — using fallback r = {fallback:.3f} kpc")

    DZ = np.zeros(len(gas_mass))
    if len(dust_mass) == 0:
        print("  WARNING: no dust particles found — D/Z = 0 everywhere")
        return DZ

    tree = cKDTree(dust_coords)
    n_dust_assigned = 0
    n_capped = 0

    for i in range(len(gas_mass)):
        if gas_Z[i] <= 0 or gas_mass[i] <= 0:
            continue
        nbrs = tree.query_ball_point(gas_coords[i], r=gas_hsml[i])
        local_dust = dust_mass[nbrs].sum() if nbrs else 0.0
        if local_dust > 0:
            n_dust_assigned += 1
        raw = local_dust / (gas_mass[i] * gas_Z[i])
        if raw > 1.0:
            n_capped += 1
        DZ[i] = min(raw, 1.0)   # D/Z = 1 is the hard physical ceiling

    frac_capped = 100 * n_capped / max(n_dust_assigned, 1)
    print(f"  Gas particles with dust in kernel: {n_dust_assigned:,} "
          f"/ {len(gas_mass):,} ({100*n_dust_assigned/max(len(gas_mass),1):.1f}%)")
    print(f"  Particles capped at D/Z = 1.0: {n_capped:,} ({frac_capped:.1f}% of assigned)")
    return DZ


# ============================================================
# Plotting
# ============================================================

def make_plot(gas_Z, DZ, gas_mass, scale_factor, r_max_kpc,
              n_dust, output_path):

    z = 1.0 / scale_factor - 1.0

    ok = (gas_Z > 0) & np.isfinite(DZ) & (DZ > 0) & (gas_mass > 0)
    Z_p  = gas_Z[ok]
    DZ_p = DZ[ok]
    m_p  = gas_mass[ok]

    print(f"\n  Plot sample: {ok.sum():,} gas particles with DZ > 0")
    if ok.sum() > 0:
        mw_DZ = np.average(DZ_p, weights=m_p)
        mw_Z  = np.average(Z_p,  weights=m_p)
        print(f"  Mass-weighted D/Z = {mw_DZ:.3f}")
        print(f"  Mass-weighted Z   = {mw_Z:.4f}  ({mw_Z/Z_SOLAR:.2f} Z_sun)")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    # ----------------------------------------------------------
    # 2D hex-density of simulation particles
    # ----------------------------------------------------------
    if ok.sum() > 1:
        logZ  = np.log10(np.clip(Z_p,  1e-6, None))
        logDZ = np.log10(np.clip(DZ_p, 1e-5, None))
        # Mass-weighted: each bin shows total gas mass, not particle count
        # This suppresses the many low-mass CGM particles at low Z
        h2d, xe, ye = np.histogram2d(
            logZ, logDZ, bins=70,
            range=[[-4.0, np.log10(0.08)], [-4.5, 0.2]],
            weights=m_p)
        h2d[h2d == 0] = np.nan
        pcm = ax.pcolormesh(10**xe, 10**ye, h2d.T,
                            cmap="Blues",
                            norm=mcolors.LogNorm(vmin=h2d[np.isfinite(h2d)].min()),
                            zorder=1, rasterized=True)
        cbar = fig.colorbar(pcm, ax=ax, pad=0.01, aspect=30)
        cbar.set_label(r"Gas mass per bin  ($M_\odot$)", fontsize=9)

    # ----------------------------------------------------------
    # Binned median ± 1σ (16th–84th) of simulation
    # ----------------------------------------------------------
    if ok.sum() > 20:
        edges   = np.logspace(-4.0, np.log10(0.08), 20)
        idx     = np.digitize(Z_p, edges) - 1
        mZ, mDZ, loDZ, hiDZ = [], [], [], []
        for b in range(len(edges) - 1):
            sel = (idx == b) & (DZ_p > 0)
            if sel.sum() < 5:
                continue
            mZ.append(np.sqrt(edges[b] * edges[b+1]))
            mDZ.append(np.median(DZ_p[sel]))
            loDZ.append(np.percentile(DZ_p[sel], 16))
            hiDZ.append(np.percentile(DZ_p[sel], 84))

        if mZ:
            mZ = np.array(mZ); mDZ = np.array(mDZ)
            loDZ = np.array(loDZ); hiDZ = np.array(hiDZ)
            ax.fill_between(mZ, loDZ, hiDZ,
                            color="steelblue", alpha=0.35, zorder=3,
                            label="_nolegend_")
            ax.plot(mZ, mDZ, color="steelblue", lw=2.5, zorder=4,
                    label="This work (median ± 1σ)")

    # ----------------------------------------------------------
    # Rémy-Ruyer+2014 BPL fit + scatter band
    # ----------------------------------------------------------
    Z_fit  = np.logspace(-4.0, np.log10(0.08), 300)
    DZ_fit = rr14_dz(Z_fit)
    ax.fill_between(Z_fit,
                    DZ_fit * 10**(-RR14_SCATTER_DEX),
                    DZ_fit * 10**( RR14_SCATTER_DEX),
                    color="black", alpha=0.10, zorder=2, label="_nolegend_")
    ax.plot(Z_fit, DZ_fit, color="black", lw=2.2, ls="--", zorder=6,
            label="Rémy-Ruyer+2014 (BPL fit, ±0.45 dex)")

    # ----------------------------------------------------------
    # De Vis+2019 DustPedia data points
    # ----------------------------------------------------------
    ax.plot(DEVIS19_Z, DEVIS19_DZ, "o", color="black",
            ms=5, mew=0, alpha=0.65, zorder=5,
            label="De Vis+2019 (DustPedia obs.)")

    # ----------------------------------------------------------
    # Simulation model trends
    # ----------------------------------------------------------
    ax.plot(MK17_Z, MK17_DZ, color="firebrick", lw=1.8, ls="-.", zorder=5,
            label="McKinnon+2017 (AREPO, subgrid ISM)")

    ax.plot(AO18_Z, AO18_DZ, color="darkorange", lw=1.8, ls=":", zorder=5,
            label="Aoyama+2018 (Gadget, two-size)")

    ax.plot(DAVE19_Z, DAVE19_DZ, color="darkorchid", lw=1.8, ls=(0,(4,2,1,2)),
            zorder=5, label="Davé+2019 / SIMBA")

    # ----------------------------------------------------------
    # Reference lines
    # ----------------------------------------------------------
    ax.axvline(Z_SOLAR, color="goldenrod", lw=0.9, ls=":", alpha=0.8, zorder=4)
    ax.text(Z_SOLAR * 1.07, 0.55, r"$Z_\odot$",
            color="goldenrod", fontsize=9, va="top", zorder=7)

    ax.axhline(0.4, color="gray", lw=0.9, ls=":", alpha=0.7, zorder=3)
    ax.text(1.1e-4, 0.42, "D/Z = 0.40 (MW canonical)",
            color="gray", fontsize=8, va="bottom", zorder=7)

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e-5, 0.06)
    ax.set_ylim(3e-4, 1.5)

    ax.set_xlabel(r"Gas Metallicity  $Z$  (metal mass fraction)", fontsize=13)
    ax.set_ylabel(r"Dust-to-Metal Ratio  $D/Z$", fontsize=13)
    ax.set_title(
        f"D/Z vs. Metallicity  —  $z = {z:.2f}$"
        f"    |    $R < {r_max_kpc:.0f}$ kpc"
        f"    |    {n_dust:,} dust particles",
        fontsize=11)

    # Secondary top x-axis in Z/Z_sun
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(np.array(ax.get_xlim()) / Z_SOLAR)
    ax2.set_xlabel(r"$Z / Z_\odot$", fontsize=11)
    ax2.tick_params(labelsize=9)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.92,
              ncol=1, handlelength=2.4, borderpad=0.7)
    ax.grid(True, which="both", ls=":", alpha=0.25, color="gray")
    ax.tick_params(which="both", direction="in", top=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot",
                        help="Snapshot base (e.g. ../snapdir_049/snapshot_049)")
    parser.add_argument("catalog",
                        help="Subfind catalog (e.g. ../groups_049/fof_subhalo_tab_049.0.hdf5)")
    parser.add_argument("--r-max", type=float, default=200.0,
                        help="Aperture radius in physical kpc (default: 200, "
                             "0 = use 2 × halfmass_rad)")
    parser.add_argument("--output", default="dz_vs_metallicity.png")
    args = parser.parse_args()

    units = get_units(args.snapshot)
    a, h  = units["a"], units["hubble"]
    print(f"  a = {a:.4f}  z = {1/a-1:.2f}")

    kpc_per_code  = units["unit_len_cm"] / 3.085678e21 * a / h
    msun_per_code = units["unit_mass_g"] / 1.989e33 / h

    # --- Halo centre (code units from subfind) ---
    print("\nLoading target halo...")
    halo     = load_target_halo(args.catalog, args.snapshot, verbose=True)
    halo_pos = halo["halo_info"]["position"]           # code units

    r_max = (args.r_max if args.r_max > 0
             else halo["halo_info"]["halfmass_rad"] * kpc_per_code * 2.0)
    center_phys = np.array(halo_pos) * kpc_per_code
    print(f"  Centre (phys kpc): {center_phys}")
    print(f"  Aperture: {r_max:.1f} kpc")

    # --- Gas ---
    print("\nExtracting gas (PartType0)...")
    gas_coords, gas_mass, gas_Z, gas_hsml = \
        read_gas_spatially(args.snapshot, center_phys, r_max, units)

    # --- Dust (PartType6 — not in Subfind, extract spatially) ---
    print("\nExtracting dust (PartType6)...")
    dust_raw = extract_dust_spatially(args.snapshot, halo_pos, radius_kpc=r_max)
    if dust_raw is not None:
        dust_coords = dust_raw["Coordinates"] * kpc_per_code
        dust_mass   = dust_raw["Masses"]      * msun_per_code
        print(f"  Dust particles: {len(dust_mass):,}")
    else:
        print("  WARNING: No PartType6 found — will produce empty plot")
        dust_coords = np.zeros((0, 3))
        dust_mass   = np.zeros(0)

    # --- Per-gas-particle D/Z assignment ---
    print("\nAssigning D/Z per gas particle via KD-tree...")
    DZ = assign_dz_kdtree(gas_coords, gas_mass, gas_Z, gas_hsml,
                           dust_coords, dust_mass)

    # --- Plot ---
    make_plot(gas_Z, DZ, gas_mass,
              scale_factor=a, r_max_kpc=r_max,
              n_dust=len(dust_mass),
              output_path=args.output)


if __name__ == "__main__":
    main()
