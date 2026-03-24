#!/usr/bin/env python3
"""
plot_phase_diagram.py
----------------------
Gas temperature vs. hydrogen number density phase diagram for the main
halo in a Gadget-4 zoom simulation, out to 1 × R_vir (R_crit200).

The 2D histogram is mass-weighted so that the ISM and CGM are shown
with physical weight rather than being dominated by particle count.

Phase regions annotated:
  - Cold/warm neutral medium  (T < 2e4 K,  n_H > 0.01)
  - Warm ionized medium       (T ~ 1e4 K,  diffuse)
  - Hot CGM / halo gas        (T > 1e6 K,  n_H < 0.01)
  - Star-forming EOS          (n_H > CritPhysDensity, pressurized EOS track)

Reference lines:
  - T = 1e4 K  (hydrogen recombination / photoionization equilibrium)
  - Effective EOS track (Springel & Hernquist 2003)
  - Virial temperature of the halo (T_vir ~ μ m_p V_c² / 2 k_B)

Usage:
    python plot_phase_diagram.py snapshot_base catalog [options]

    python plot_phase_diagram.py ../7_output_512/snapdir_049/snapshot_049 ../7_output_512/groups_049/fof_subhalo_tab_049.0.hdf5 --r-frac 1.0 --color-by mass --output phase_diagram_z0.png
"""

import argparse
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from halo_utils import load_target_halo
except ImportError:
    raise ImportError("halo_utils not found — ensure halo_utils.py is in PATH")

# ============================================================
# Physical constants (cgs)
# ============================================================
BOLTZMANN   = 1.38065e-16   # erg/K
PROTONMASS  = 1.67262e-24   # g
XH          = 0.76          # hydrogen mass fraction
GAMMA       = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0


# ============================================================
# Unit helpers
# ============================================================

def get_header(snapshot_base):
    files = sorted(glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = [snapshot_base + ".hdf5"]
    with h5py.File(files[0], "r") as f:
        h = f["Header"].attrs
        return dict(
            a            = float(h.get("Time",                        1.0)),
            hubble       = float(h.get("HubbleParam",                 0.6774)),
            unit_mass_g  = float(h.get("UnitMass_in_g",              1.989e43)),
            unit_len_cm  = float(h.get("UnitLength_in_cm",           3.085678e21)),
            unit_vel_cms = float(h.get("UnitVelocity_in_cm_per_s",   1e5)),
        )


# ============================================================
# Gas extraction
# ============================================================

def read_gas_in_sphere(snap_base, center_phys_kpc, r_max_kpc, units):
    """
    Read PartType0 within r_max_kpc of center_phys_kpc.
    Returns:
        coords  [kpc, physical]
        mass    [Msun]
        nH      [cm^-3]
        T       [K]
        Z       [mass fraction]
        sfr     [Msun/yr, zeros if not present]
    """
    a, h  = units["a"], units["hubble"]
    kpc   = units["unit_len_cm"] / 3.085678e21 * a / h
    msun  = units["unit_mass_g"] / 1.989e33 / h
    # density: code → physical cgs
    dens_factor = (units["unit_mass_g"] / units["unit_len_cm"]**3) \
                  * h**2 / a**3

    files = sorted(glob.glob(f"{snap_base}.*.hdf5"))
    if not files:
        files = [snap_base + ".hdf5"]

    c_l, m_l, rho_l, u_l, ne_l, Z_l, sfr_l = [], [], [], [], [], [], []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "PartType0" not in f:
                continue
            pt0  = f["PartType0"]
            cord = pt0["Coordinates"][:] * kpc
            r    = np.sqrt(np.sum((cord - center_phys_kpc)**2, axis=1))
            mask = r < r_max_kpc
            if not mask.any():
                continue

            c_l.append(cord[mask])
            m_l.append(pt0["Masses"][:][mask]         * msun)
            rho_l.append(pt0["Density"][:][mask]      * dens_factor)
            u_l.append(pt0["InternalEnergy"][:][mask] * units["unit_vel_cms"]**2)

            ne_l.append(pt0["ElectronAbundance"][:][mask]
                         if "ElectronAbundance" in pt0
                         else np.zeros(mask.sum()))
            Z_l.append(pt0["Metallicity"][:][mask]
                        if "Metallicity" in pt0
                        else np.zeros(mask.sum()))
            sfr_l.append(pt0["StarFormationRate"][:][mask]
                          if "StarFormationRate" in pt0
                          else np.zeros(mask.sum()))

    if not c_l:
        raise RuntimeError(
            f"No PartType0 within {r_max_kpc:.1f} kpc — check halo_pos units")

    coords = np.vstack(c_l)
    mass   = np.concatenate(m_l)
    rho    = np.concatenate(rho_l)   # g/cm³
    u_cgs  = np.concatenate(u_l)     # erg/g
    ne     = np.concatenate(ne_l)    # electrons per H atom
    Z      = np.concatenate(Z_l)
    sfr    = np.concatenate(sfr_l)

    # n_H
    nH = XH * rho / PROTONMASS

    # Temperature: T = (γ-1) × u × μ × m_p / k_B
    # μ ≈ 4 / (1 + 3×XH + 4×XH×ne)  — mean molecular weight
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * ne)
    T  = GAMMA_MINUS1 * u_cgs * mu * PROTONMASS / BOLTZMANN

    print(f"  Gas particles : {len(mass):,}")
    print(f"  n_H range     : {nH.min():.2e} – {nH.max():.2e} cm⁻³")
    print(f"  T   range     : {T.min():.2e} – {T.max():.2e} K")

    return coords, mass, nH, T, Z, sfr


# ============================================================
# Virial temperature estimate
# ============================================================

def virial_temperature(M200_msun, R200_kpc):
    """
    T_vir = μ m_p G M200 / (2 k_B R200)
    Uses μ = 0.59 (fully ionized primordial gas)
    """
    G    = 6.674e-8          # cgs
    MU   = 0.59
    M_g  = M200_msun * 1.989e33
    R_cm = R200_kpc  * 3.086e21
    return MU * PROTONMASS * G * M_g / (2.0 * BOLTZMANN * R_cm)


# ============================================================
# Springel & Hernquist 2003 effective EOS track
# (approximate, for orientation only)
# ============================================================

def sh03_eos_track(nH_arr, T_floor=1e3):
    """
    Very approximate EOS track for the multiphase pressurized ISM.
    Above n_crit the gas is placed on a stiff EOS; below it cools freely.
    We just sketch the slope T ∝ n_H^(γ_eff - 1) with γ_eff ~ 4/3
    starting near the SF threshold (~0.1 cm⁻³, ~10⁴ K).
    """
    nH_thresh = 0.13   # typical CritPhysDensity
    T_thresh  = 1.5e4
    gamma_eff = 4.0 / 3.0
    mask = nH_arr >= nH_thresh
    T = np.full_like(nH_arr, np.nan)
    T[mask] = T_thresh * (nH_arr[mask] / nH_thresh) ** (gamma_eff - 1.0)
    return T


# ============================================================
# Plot
# ============================================================

def make_plot(mass, nH, T, sfr, halo_info, r_max_kpc, r_frac,
              color_by, output_path):

    a     = halo_info["a"]
    z     = 1.0 / a - 1.0
    M200  = halo_info.get("M200_msun", None)
    R200  = halo_info.get("R200_kpc",  None)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ----------------------------------------------------------
    # 2D mass-weighted histogram
    # ----------------------------------------------------------
    log_nH = np.log10(np.clip(nH, 1e-7, None))
    log_T  = np.log10(np.clip(T,  10,   None))

    h2d, xe, ye = np.histogram2d(
        log_nH, log_T, bins=200,
        range=[[-7, 4], [2, 9]],
        weights=mass)
    h2d[h2d == 0] = np.nan

    pcm = ax.pcolormesh(
        10**xe, 10**ye, h2d.T,
        norm=mcolors.LogNorm(
            vmin=np.nanpercentile(h2d, 10),
            vmax=np.nanpercentile(h2d, 99.5)),
        cmap="inferno",
        rasterized=True, zorder=1)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label(r"Gas mass per bin  ($M_\odot$)", fontsize=9)

    # ----------------------------------------------------------
    # Star-forming gas overlay (SFR > 0)
    # ----------------------------------------------------------
    sf = sfr > 0
    if sf.sum() > 0:
        ax.scatter(nH[sf], T[sf], s=1, c="cyan", alpha=0.3,
                   zorder=3, label=f"Star-forming ({sf.sum():,} particles)",
                   rasterized=True)

    # ----------------------------------------------------------
    # Reference lines
    # ----------------------------------------------------------
    # T = 1e4 K (hydrogen recombination)
    ax.axhline(1e4, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=4)
    ax.text(2e-7, 1.15e4, r"$T = 10^4$ K", color="gray",
            fontsize=8, alpha=0.8, zorder=5)

    # T = 1e5 K (OVI cooling peak / CGM boundary)
    ax.axhline(1e5, color="gray", lw=0.8, ls=":", alpha=0.5, zorder=4)
    ax.text(2e-7, 1.15e5, r"$T = 10^5$ K", color="gray",
            fontsize=8, alpha=0.7, zorder=5)

    # Virial temperature
    if M200 is not None and R200 is not None:
        T_vir = virial_temperature(M200, R200)
        ax.axhline(T_vir, color="goldenrod", lw=1.2, ls="-.", alpha=0.8, zorder=4)
        ax.text(2e-7, T_vir * 1.15, fr"$T_{{vir}}={T_vir:.1e}$ K",
                color="goldenrod", fontsize=8, alpha=0.9, zorder=5)

    # Effective EOS track (subgrid SF threshold)
    nH_eos = np.logspace(-1, 3.5, 200)
    T_eos  = sh03_eos_track(nH_eos)
    valid  = np.isfinite(T_eos)
    ax.plot(nH_eos[valid], T_eos[valid],
            color="lime", lw=1.5, ls="--", alpha=0.6, zorder=4,
            label="S&H 2003 EOS track (approx.)")
    ax.axvline(0.13, color="lime", lw=0.8, ls=":", alpha=0.4, zorder=3)
    ax.text(0.15, 2e2, r"$n_{\rm SF}$", color="lime",
            fontsize=8, alpha=0.6, zorder=5)

    # ----------------------------------------------------------
    # Phase region labels
    # ----------------------------------------------------------
    label_kw = dict(fontsize=8.5, alpha=0.45, ha="center", zorder=5,
                    style="italic", color="#444444")
    ax.text(1e-4,  2e7,  "Hot CGM / Shock-heated",    **label_kw)
    ax.text(1e-4,  3e4,  "Warm Ionized Medium",        **label_kw)
    ax.text(3e-3,  4e3,  "Warm Neutral\nMedium",       **label_kw)
    ax.text(3e1,   2e3,  "ISM / Star-forming",         **label_kw)
    ax.text(1e-5,  5e6,  "Cooling\nflows",             **label_kw)

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 1e4)
    ax.set_ylim(1e2,  1e9)

    ax.set_xlabel(r"Hydrogen Number Density  $n_H$  (cm$^{-3}$)", fontsize=13)
    ax.set_ylabel(r"Temperature  $T$  (K)", fontsize=13)
    ax.set_title(
        fr"Gas Phase Diagram  —  $z = {z:.2f}$"
        fr"    |    $R < {r_frac:.1f}\,R_{{200}}$  ({r_max_kpc:.0f} kpc)",
        fontsize=11)

    ax.tick_params(which="both", direction="in")
    ax.grid(True, which="major", ls=":", alpha=0.2, color="gray")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="upper left",
                  framealpha=0.7)

    # Stats annotation
    mw_T  = np.average(T,  weights=mass)
    mw_nH = np.average(nH, weights=mass)
    stats = (f"$N_{{gas}}$ = {len(mass):,}\n"
             f"$\\langle T \\rangle_m$ = {mw_T:.2e} K\n"
             f"$\\langle n_H \\rangle_m$ = {mw_nH:.2e} cm$^{{-3}}$")
    ax.text(0.98, 0.97, stats, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="gray"), zorder=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot",
                        help="Snapshot base path")
    parser.add_argument("catalog",
                        help="Subfind catalog path")
    parser.add_argument("--r-frac", type=float, default=1.0,
                        help="Aperture as fraction of R_vir/R_200 "
                             "(default: 1.0)")
    parser.add_argument("--color-by", choices=["mass"], default="mass",
                        help="Colour the histogram by (default: mass)")
    parser.add_argument("--output", default="phase_diagram.png")
    args = parser.parse_args()

    units = get_header(args.snapshot)
    a, h  = units["a"], units["hubble"]
    z     = 1.0 / a - 1.0
    print(f"  a = {a:.4f}  z = {z:.2f}")

    kpc_per_code  = units["unit_len_cm"] / 3.085678e21 * a / h
    msun_per_code = units["unit_mass_g"] / 1.989e33 / h

    # --- Halo ---
    print("\nLoading target halo...")
    halo      = load_target_halo(args.catalog, args.snapshot, verbose=True)
    halo_pos  = halo["halo_info"]["position"]     # code units
    center    = np.array(halo_pos) * kpc_per_code # physical kpc

    # Prefer R_crit200; fall back to halfmass_rad × 1.5
    R200_code = halo["halo_info"].get("R200", None)
    if R200_code is not None:
        R200_kpc = float(R200_code) * kpc_per_code
    else:
        R200_kpc = halo["halo_info"]["halfmass_rad"] * kpc_per_code * 1.5
        print(f"  R_200 not in halo_info — using 1.5 × halfmass_rad = {R200_kpc:.1f} kpc")

    r_max = args.r_frac * R200_kpc
    print(f"  R_200      = {R200_kpc:.1f} kpc")
    print(f"  Aperture   = {args.r_frac:.1f} × R_200 = {r_max:.1f} kpc")

    # M200 for virial temperature
    M200_code = halo["halo_info"].get("M200", None)
    M200_msun = float(M200_code) * msun_per_code if M200_code is not None else None

    # --- Gas ---
    print("\nExtracting gas (PartType0)...")
    coords, mass, nH, T, Z, sfr = read_gas_in_sphere(
        args.snapshot, center, r_max, units)

    # --- Plot ---
    halo_info = dict(a=a, M200_msun=M200_msun, R200_kpc=R200_kpc)
    make_plot(mass, nH, T, sfr, halo_info,
              r_max_kpc=r_max, r_frac=args.r_frac,
              color_by=args.color_by,
              output_path=args.output)


if __name__ == "__main__":
    main()
