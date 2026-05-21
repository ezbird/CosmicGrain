#!/usr/bin/env python3
"""
plot_drift_length.py

Measures local aerodynamic drift of dust grains by computing:

  Δv       = |v_dust - v_gas_nearest|        (physical km/s)
  τ_stop   = ρ_grain × a_grain / (ρ_gas × v_th)   (Epstein drag, Myr)
  ℓ_drift  = Δv × τ_stop                     (physical kpc)

For each dust particle, the nearest gas neighbour is found via KDTree.
This gives the local aerodynamic drift length — how far a grain travels
relative to its surrounding gas before drag re-couples it — without any
dependence on merger-tree geometry or birth position.

Two figures:
  1. CDF of Δv and ℓ_drift at z≈0, split ISM / CGM
  2. ℓ_drift vs galactocentric radius (hexbin)

Usage:
  python plot_drift_length.py --snap-dir ../S10_output_1024/ --snap 49
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ──────────────────────────────────────────────────────────────

SNAP_DIR        = Path("../S10_output_1024")
GROUP_DIR       = None          # defaults to SNAP_DIR
OUT_DIR         = Path(".")
SNAP_N          = 49

ISM_RADIUS_PKPC = 20.0          # ISM/CGM boundary (pkpc)
MAX_RADIUS_PKPC = 600.0         # outer cut ~2×R_200; excludes background

# Grain material density (g/cm³)
RHO_GRAIN_SIL   = 2.2           # silicate
RHO_GRAIN_CARB  = 2.0           # carbonaceous

# ── Style ──────────────────────────────────────────────────────────────────────

C_ISM = "#2a9d8f"
C_CGM = "#e76f51"
C_ALL = "#264653"

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.facecolor"   : "white",
    "figure.facecolor" : "white",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── Unit constants ─────────────────────────────────────────────────────────────

KPC_CM    = 3.085677581e21     # cm per kpc
KM_CM     = 1e5                # cm per km
MYR_S     = 3.15576e13        # s per Myr
K_B       = 1.380649e-16      # erg/K
M_PROTON  = 1.6726219e-24     # g
GAMMA     = 5.0 / 3.0
MU        = 0.588             # mean molecular weight (fully ionised primordial)

# ── Path helpers ───────────────────────────────────────────────────────────────

def snap_chunks(n, snap_dir):
    d = snap_dir / f"snapdir_{n:03d}"
    return sorted(d.glob(f"snapshot_{n:03d}.*.hdf5"))

def group_chunk0(n, group_dir):
    candidates = [
        group_dir / f"groups_{n:03d}" / f"fof_subhalo_tab_{n:03d}.0.hdf5",
        group_dir / f"fof_subhalo_tab_{n:03d}.0.hdf5",
        group_dir / f"fof_subhalo_tab_{n:03d}.hdf5",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ── Data loading ───────────────────────────────────────────────────────────────

def load_header(chunk):
    with h5py.File(chunk, "r") as f:
        h       = float(f["Parameters"].attrs["HubbleParam"])
        a       = float(f["Header"].attrs["Time"])
        z       = float(f["Header"].attrs["Redshift"])
        boxsize = float(f["Header"].attrs["BoxSize"])
    return dict(h=h, a=a, z=z, boxsize=boxsize)


def load_halo_center(n, group_dir):
    gp = group_chunk0(n, group_dir)
    if gp is None:
        return None
    with h5py.File(gp, "r") as f:
        if "Subhalo/SubhaloPos" in f and len(f["Subhalo/SubhaloPos"]) > 0:
            return f["Subhalo/SubhaloPos"][0].copy()
        if "Group/GroupPos" in f and len(f["Group/GroupPos"]) > 0:
            return f["Group/GroupPos"][0].copy()
    return None


def load_parttype(chunks, ptype, fields):
    """
    Concatenate requested fields across all chunks for a given particle type.
    Returns dict of field → array, or None if no particles found.
    """
    ptype_key = f"PartType{ptype}"
    arrays    = {f: [] for f in fields}
    found_any = False

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            npart = int(f["Header"].attrs["NumPart_ThisFile"][ptype])
            if npart == 0 or ptype_key not in f:
                continue
            found_any = True
            for field in fields:
                if field in f[ptype_key]:
                    arrays[field].append(f[ptype_key][field][:])
                else:
                    print(f"  Warning: {ptype_key}/{field} not found in {chunk.name}")
                    arrays[field].append(np.zeros((npart,) if field != "Coordinates"
                                                  else (npart, 3)))

    if not found_any:
        return None
    return {f: np.concatenate(arrays[f], axis=0) for f in fields}


def physical_velocity(v_stored, a):
    """
    Convert Gadget-4 stored velocity to physical peculiar velocity (km/s).
    Gadget stores v_stored = v_peculiar / sqrt(a), so:
      v_physical = v_stored * sqrt(a)
    """
    return v_stored * np.sqrt(a)


def gas_temperature(u_internal):
    """
    Convert specific internal energy (code units = (km/s)²) to temperature (K).
    T = (γ-1) × μ × m_p × u / k_B
    u in (km/s)² → convert to erg/g first.
    """
    u_cgs = u_internal * KM_CM**2   # (km/s)² → cm²/s² = erg/g
    T = (GAMMA - 1.0) * MU * M_PROTON * u_cgs / K_B
    return T


def stopping_time_myr(a_grain_nm, rho_gas_code, T_gas, h_cosmo, a_scale,
                       rho_grain=RHO_GRAIN_SIL):
    """
    Epstein drag stopping time in Myr.

    Parameters
    ----------
    a_grain_nm : grain radius in nm
    rho_gas_code : gas density in Gadget code units (10^10 M_sun/h / (kpc/h)^3)
    T_gas : gas temperature in K
    h_cosmo : HubbleParam
    a_scale : scale factor
    rho_grain : grain material density in g/cm³

    Returns τ_stop in Myr.
    """
    # Convert grain radius nm → cm
    a_grain_cm = a_grain_nm * 1e-7

    # Convert gas density: code units → g/cm³
    # code density = 10^10 M_sun/h per (kpc/h)^3 in comoving coords
    # physical density = code_density * h^2 / a^3  (in 10^10 M_sun/kpc^3)
    # then × (1.989e43 g / M_sun) / (3.086e21 cm/kpc)^3
    M_sun_g    = 1.989e33           # g
    kpc_cm     = KPC_CM
    rho_phys_code = rho_gas_code * h_cosmo**2 / a_scale**3   # 10^10 M_sun/kpc^3 physical
    rho_phys_cgs  = (rho_phys_code * 1e10 * M_sun_g
                     / kpc_cm**3)                              # g/cm³

    # Thermal velocity of gas
    v_th = np.sqrt(8.0 * K_B * T_gas / (np.pi * M_PROTON * MU))  # cm/s

    # Clip density to avoid float overflow in extreme SF gas
    rho_phys_cgs = np.clip(rho_phys_cgs, 1e-35, 1e-13)  # g/cm^3

    # Stopping time (s)
    tau_s = rho_grain * a_grain_cm / (rho_phys_cgs * v_th)

    # Replace non-finite (overflow / degenerate) with 0
    tau_s = np.where(np.isfinite(tau_s), tau_s, 0.0)

    return tau_s / MYR_S


# ── Main analysis ───────────────────────────────────────────────────────────────

def run(snap_dir, group_dir, snap_n, out_dir,
        ism_radius, max_radius):

    chunks = snap_chunks(snap_n, snap_dir)
    if not chunks:
        sys.exit(f"No snapshot chunks found for snap {snap_n:03d} in {snap_dir}")

    hdr    = load_header(chunks[0])
    a      = hdr["a"]
    h      = hdr["h"]
    z      = hdr["z"]
    boxsize = hdr["boxsize"]
    print(f"Snapshot {snap_n:03d}  z={z:.3f}  a={a:.4f}  h={h}")

    center = load_halo_center(snap_n, group_dir)
    if center is None:
        sys.exit("Halo center not found — cannot compute galactocentric radii.")
    print(f"Halo center (comoving kpc/h): {center}")

    # ── Load dust ──────────────────────────────────────────────────────────────
    print("Loading dust (PartType6)...")
    dust_fields = ["Coordinates", "Velocities", "GrainRadius"]
    dust = load_parttype(chunks, 6, dust_fields)
    if dust is None:
        sys.exit("No PartType6 particles found.")
    print(f"  {len(dust['Coordinates']):,} dust particles")

    # Physical dust positions and galactocentric radii
    dp_com = dust["Coordinates"] - center
    dp_com -= boxsize * np.round(dp_com / boxsize)
    dp_phys = dp_com * a / h                   # physical kpc
    r_dust  = np.linalg.norm(dp_phys, axis=1)  # physical kpc

    # Filter dust to halo region
    halo_mask = r_dust < max_radius
    print(f"  {halo_mask.sum():,} within {max_radius:.0f} pkpc")

    # ── Load gas ───────────────────────────────────────────────────────────────
    print("Loading gas (PartType0)...")
    gas_fields = ["Coordinates", "Velocities", "Density", "InternalEnergy"]
    gas = load_parttype(chunks, 0, gas_fields)
    if gas is None:
        sys.exit("No PartType0 particles found.")
    print(f"  {len(gas['Coordinates']):,} gas particles")

    # Physical gas positions relative to halo center
    gp_com = gas["Coordinates"] - center
    gp_com -= boxsize * np.round(gp_com / boxsize)
    gp_phys = gp_com * a / h

    # Filter gas to slightly larger region to ensure boundary coverage
    r_gas = np.linalg.norm(gp_phys, axis=1)
    gas_mask_radial = r_gas < max_radius * 1.5

    # Compute temperature for all gas in radial cut — used for phase filter
    T_gas_full   = gas_temperature(gas["InternalEnergy"])  # full array
    T_gas_all    = T_gas_full[gas_mask_radial]              # radial subset

    # ── Phase diagnostic: what gas is the dust being matched to? ─────────────
    print(f"\n  Gas phase breakdown within {max_radius*1.5:.0f} pkpc:")
    for label, lo, hi in [
        ("Cold    T < 1e4 K",   0,    1e4),
        ("Warm  1e4–1e5 K",     1e4,  1e5),
        ("Hot   1e5–1e7 K",     1e5,  1e7),
        ("V.Hot   > 1e7 K",     1e7,  np.inf),
    ]:
        frac = ((T_gas_all >= lo) & (T_gas_all < hi)).sum()
        print(f"    {label}: {frac:,} ({100*frac/len(T_gas_all):.1f}%)")

    # ── Two KDTrees: all gas and cool gas only (T < T_COOL_MAX) ──────────────
    T_COOL_MAX = 1e5    # K — below this, gas is in a phase relevant for drag

    cool_mask  = gas_mask_radial & (T_gas_full < T_COOL_MAX)  # both shape (N_gas,)
    print(f"\n  Building KDTree: all gas ({gas_mask_radial.sum():,} particles)...")
    tree_all  = KDTree(gp_phys[gas_mask_radial])

    print(f"  Building KDTree: cool gas T<{T_COOL_MAX:.0e} K "
          f"({cool_mask.sum():,} particles)...")
    if cool_mask.sum() == 0:
        print("  WARNING: no cool gas found — skipping cool-only match")
        tree_cool = None
    else:
        tree_cool = KDTree(gp_phys[cool_mask])

    dust_pos_halo = dp_phys[halo_mask]

    print("  Querying nearest neighbours...")
    _, idx_all  = tree_all.query(dust_pos_halo, workers=-1)
    idx_cool    = None
    if tree_cool is not None:
        _, idx_cool = tree_cool.query(dust_pos_halo, workers=-1)

    # ── Velocity offset (uses all-gas match — Δv doesn't need phase filter) ──
    v_dust = physical_velocity(dust["Velocities"][halo_mask], a)
    v_gas  = physical_velocity(gas["Velocities"][gas_mask_radial][idx_all], a)
    dv_vec = v_dust - v_gas
    dv     = np.linalg.norm(dv_vec, axis=1)   # |Δv| physical km/s

    # ── Stopping time: compute for both all-gas and cool-gas matches ──────────
    print("  Computing stopping times...")
    KM_S_TO_KPC_MYR = KM_CM / (KPC_CM / MYR_S)
    a_grain_nm  = dust["GrainRadius"][halo_mask]
    rho_grain   = np.full(len(a_grain_nm), RHO_GRAIN_SIL)

    def _tau_and_drift(idx_match, gas_mask_match):
        rho_g = gas["Density"][gas_mask_match][idx_match]
        u_g   = gas["InternalEnergy"][gas_mask_match][idx_match]
        T_g   = gas_temperature(u_g)
        tau   = stopping_time_myr(a_grain_nm, rho_g, T_g, h, a, rho_grain)
        ell   = dv * tau * KM_S_TO_KPC_MYR
        return tau, ell, T_g

    tau_all,  l_all,  T_matched_all  = _tau_and_drift(idx_all,  gas_mask_radial)
    if idx_cool is not None:
        tau_cool, l_cool, T_matched_cool = _tau_and_drift(idx_cool, cool_mask)
    else:
        tau_cool = l_cool = T_matched_cool = None

    # ISM / CGM split
    ism_mask = r_dust[halo_mask] < ism_radius
    cgm_mask = ~ism_mask

    # ── Print matched-gas diagnostic ──────────────────────────────────────────
    print()
    print("  Matched gas temperature (what phase dust is matched to):")
    print(f"  {'Match':<12} {'T_gas p16':>12}  {'T_gas med':>12}  {'T_gas p84':>12}")
    print(f"  {'-'*52}")
    print(f"  {'All gas':<12} "
          f"{np.percentile(T_matched_all,16):>12.2e}  "
          f"{np.median(T_matched_all):>12.2e}  "
          f"{np.percentile(T_matched_all,84):>12.2e}")
    if T_matched_cool is not None:
        print(f"  {'Cool gas':<12} "
              f"{np.percentile(T_matched_cool,16):>12.2e}  "
              f"{np.median(T_matched_cool):>12.2e}  "
              f"{np.percentile(T_matched_cool,84):>12.2e}")

    # ── Summary tables ────────────────────────────────────────────────────────
    print()
    for match_label, tau_myr, l_drift in [
        ("ALL GAS match",  tau_all,  l_all),
        ("COOL GAS match", tau_cool, l_cool),
    ]:
        if tau_myr is None:
            continue
        print(f"  ── {match_label} ──────────────────────────────────────────")
        print(f"  {'Pop':<6} {'N':>8}  {'Δv med':>9}  {'τ_stop med':>11}  "
              f"{'ℓ_drift med':>12}  {'ℓ_drift p84':>12}")
        print(f"  {'':6} {'':8}  {'(km/s)':>9}  {'(Myr)':>11}  "
              f"{'(pkpc)':>12}  {'(pkpc)':>12}")
        print(f"  {'-'*67}")
        for label, mask in [("All", np.ones(len(dv), dtype=bool)),
                             ("ISM", ism_mask), ("CGM", cgm_mask)]:
            if mask.sum() == 0:
                continue
            print(f"  {label:<6} {mask.sum():>8,}  "
                  f"{np.median(dv[mask]):>9.2f}  "
                  f"{np.median(tau_myr[mask]):>11.3e}  "
                  f"{np.median(l_drift[mask]):>12.3e}  "
                  f"{np.percentile(l_drift[mask],84):>12.3e}")
        print()

    # Return both matches for plotting
    return dict(
        dv=dv, r_gal=r_dust[halo_mask],
        ism_mask=ism_mask, cgm_mask=cgm_mask,
        a_grain_nm=a_grain_nm, z=z,
        # All-gas match
        tau_all=tau_all,   l_all=l_all,   T_matched_all=T_matched_all,
        # Cool-gas match
        tau_cool=tau_cool, l_cool=l_cool, T_matched_cool=T_matched_cool,
    )


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_cdf(res, out_dir):
    """Three-panel CDF: Δv, ℓ_drift (all gas), ℓ_drift (cool gas)."""
    z    = res["z"]
    has_cool = res["l_cool"] is not None

    ncols = 2 + int(has_cool)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2))

    panels = [
        (axes[0], res["dv"],    r"$|\Delta v|$ (km s$^{-1}$)",
         rf"Dust–gas $|\Delta v|$,  $z={z:.2f}$"),
        (axes[1], res["l_all"], r"$\ell_{\rm drift}$ (pkpc)",
         rf"$\ell_{{\rm drift}}$ all gas,  $z={z:.2f}$"),
    ]
    if has_cool:
        panels.append(
            (axes[2], res["l_cool"], r"$\ell_{\rm drift}$ (pkpc)",
             rf"$\ell_{{\rm drift}}$ cool gas ($T<10^5$ K),  $z={z:.2f}$")
        )

    for ax, data, xlabel, title in panels:
        for arr, label, color, ls in [
            (data,                    "All", C_ALL, "-"),
            (data[res["ism_mask"]],   "ISM", C_ISM, "--"),
            (data[res["cgm_mask"]],   "CGM", C_CGM, ":"),
        ]:
            if len(arr) == 0:
                continue
            x = np.sort(np.clip(arr, 1e-9, None))
            y = np.linspace(0, 1, len(x))
            ax.plot(x, y, color=color, ls=ls, lw=2,
                    label=rf"{label} ($N={len(arr):,}$)")

        # Annotate medians
        y_pos = {"ISM": 0.55, "CGM": 0.42}
        for arr, label, color in [
            (data[res["ism_mask"]], "ISM", C_ISM),
            (data[res["cgm_mask"]], "CGM", C_CGM),
        ]:
            if len(arr) == 0:
                continue
            med = np.median(arr)
            ax.axvline(med, color=color, lw=1.0, alpha=0.5, ls="--")
            ax.text(med * 1.15, y_pos[label],
                    f"{med:.2g}", color=color, fontsize=9, va="center")

        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(title, fontsize=10)
        ax.legend(framealpha=0.9, fontsize=9, loc="upper left")

    fig.tight_layout()
    out = out_dir / "drift_length_cdf.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)

def fig_drift_vs_radius(res, out_dir):
    """Hexbin of drift length vs galactocentric radius."""
    r        = np.clip(res["r_gal"],   0.1,  None)
    ell      = np.clip(res["l_all"],   1e-6, None)
    ell_cool = np.clip(res["l_cool"],  1e-6, None) if res["l_cool"] is not None else None

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hb = ax.hexbin(r, ell, gridsize=50, cmap="YlOrBr",
                   xscale="log", yscale="log", mincnt=1, linewidths=0.2)

    ax.axvline(20, color="gray", lw=1.2, ls="--", alpha=0.7)
    ylims = ax.get_ylim()
    ax.text(18,  ylims[0] * 2, "ISM", color="gray", fontsize=9, ha="right")
    ax.text(22,  ylims[0] * 2, "CGM", color="gray", fontsize=9, ha="left")

    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Particle count", fontsize=9)

    ax.set_xlabel(r"Galactocentric radius (pkpc)")
    ax.set_ylabel(r"$\ell_{\rm drift}$ (pkpc)")
    ax.set_title(rf"Aerodynamic drift length vs location, $z\approx{res['z']:.2f}$")

    fig.tight_layout()
    out = out_dir / "drift_length_vs_radius.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def fig_dv_vs_grain_size(res, out_dir):
    """Hexbin of Δv vs grain radius — shows size-dependent decoupling."""
    a_nm = np.clip(res["a_grain_nm"], 0.1,  None)
    dv   = np.clip(res["dv"],          0.01, None)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.hexbin(a_nm, dv, gridsize=50, cmap="YlOrBr",
              xscale="log", yscale="log", mincnt=1, linewidths=0.2)

    ax.set_xlabel("Grain radius (nm)")
    ax.set_ylabel(r"$|\Delta v|$ (km s$^{-1}$)")
    ax.set_title(rf"Velocity offset vs grain size, $z\approx{res['z']:.2f}$")

    fig.tight_layout()
    out = out_dir / "dv_vs_grain_size.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap-dir",    default=str(SNAP_DIR))
    p.add_argument("--group-dir",   default=None,
                   help="Group catalog root (default: same as --snap-dir)")
    p.add_argument("--out-dir",     default=str(OUT_DIR))
    p.add_argument("--snap",        type=int, default=SNAP_N,
                   help="Snapshot number (default: 49)")
    p.add_argument("--ism-radius",  type=float, default=ISM_RADIUS_PKPC,
                   help="ISM/CGM boundary in pkpc (default: 20)")
    p.add_argument("--max-radius",  type=float, default=MAX_RADIUS_PKPC,
                   help="Outer radius cut in pkpc (default: 600 = ~2×R200)")
    return p.parse_args()


def main():
    args     = parse_args()
    snap_dir = Path(args.snap_dir)
    group_dir = Path(args.group_dir) if args.group_dir else snap_dir
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run(snap_dir, group_dir, args.snap, out_dir,
              args.ism_radius, args.max_radius)

    print("\nGenerating figures...")
    fig_cdf(res, out_dir)
    fig_drift_vs_radius(res, out_dir)
    fig_dv_vs_grain_size(res, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
