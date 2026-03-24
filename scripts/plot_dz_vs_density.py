#!/usr/bin/env python3
"""
plot_dz_vs_density.py
---------------------
Plot dust-to-metal ratio (D/Z) vs. hydrogen number density (n_H) for a
Gadget-4 zoom simulation with N-body dust (PartType6).

Uses halo_utils for halo identification and spatial extraction — no need
to manually specify a halo center.

Since dust is not stored per gas particle, gas and dust are co-binned onto
a 3D spatial grid. Each bin's n_H is the mass-weighted mean hydrogen number
density of its gas, and D/Z = M_dust / M_metal per bin.

Usage:
    python plot_dz_vs_density.py snapshot_base catalog [options]

    python plot_dz_vs_density.py \
        ../snapdir_049/snapshot_049 \
        ../groups_049/fof_subhalo_tab_049.0.hdf5 \
        --r-max 200 \
        --color-by temperature \
        --output dz_vs_nH_z0.png
"""

import argparse
import numpy as np
import h5py
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    raise ImportError("halo_utils not found — make sure halo_utils.py is in your PATH")

# Physical constants (cgs)
PROTONMASS   = 1.6726e-24
BOLTZMANN    = 1.3807e-16
GAMMA_MINUS1 = 2.0 / 3.0
XH           = 0.76
Z_SOLAR      = 0.014


def get_unit_conversions(snapshot_base):
    files = sorted(glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = [snapshot_base + ".hdf5"]
    with h5py.File(files[0], "r") as f:
        hdr = f["Header"].attrs
        return dict(
            scale_factor = float(hdr.get("Time", 1.0)),
            hubble       = float(hdr.get("HubbleParam", 0.6774)),
            unit_mass_g  = float(hdr.get("UnitMass_in_g",            1.989e43)),
            unit_len_cm  = float(hdr.get("UnitLength_in_cm",         3.085678e21)),
            unit_vel_cms = float(hdr.get("UnitVelocity_in_cm_per_s", 1e5)),
        )


def read_gas_spatially(snapshot_base, halo_pos, r_max_kpc, units):
    """
    Spatially extract PartType0 gas fields from a possibly multi-file snapshot.
    halo_pos is in code units (as returned by load_target_halo); we convert
    both it and particle coordinates to physical kpc before the distance cut.
    """
    a, h = units["scale_factor"], units["hubble"]
    kpc_per_code  = units["unit_len_cm"] / 3.085678e21 * a / h
    msun_per_code = units["unit_mass_g"] / 1.989e33 / h
    dens_conv     = (units["unit_mass_g"] / units["unit_len_cm"]**3) * h**2 / a**3

    # Convert halo_pos from code units → physical kpc
    center_phys = np.array(halo_pos) * kpc_per_code

    files = sorted(glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = [snapshot_base + ".hdf5"]

    all_coords, all_mass, all_dens, all_Z, all_u, all_hsml = [], [], [], [], [], []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "PartType0" not in f:
                continue
            pt0    = f["PartType0"]
            coords = pt0["Coordinates"][:] * kpc_per_code   # physical kpc
            r      = np.sqrt(np.sum((coords - center_phys)**2, axis=1))
            mask   = r < r_max_kpc
            if not mask.any():
                continue
            all_coords.append(coords[mask])
            all_mass.append(  pt0["Masses"][:][mask]   * msun_per_code)
            all_dens.append(  pt0["Density"][:][mask]  * dens_conv)
            Z = pt0["Metallicity"][:][mask] if "Metallicity" in pt0 \
                else np.zeros(mask.sum())
            all_Z.append(Z)
            u = pt0["InternalEnergy"][:][mask] if "InternalEnergy" in pt0 \
                else None
            all_u.append(u)
            hsml = pt0["SmoothingLength"][:][mask] * kpc_per_code \
                   if "SmoothingLength" in pt0 else None
            all_hsml.append(hsml)

    if not all_coords:
        raise RuntimeError(
            f"No PartType0 gas found within {r_max_kpc} kpc of center "
            f"{center_phys} (code-unit center was {halo_pos})")

    coords = np.vstack(all_coords)
    mass   = np.concatenate(all_mass)
    dens   = np.concatenate(all_dens)
    Z      = np.concatenate(all_Z)
    nH     = XH * dens / PROTONMASS

    hsml = np.concatenate(all_hsml) if all_hsml[0] is not None else None

    T = None
    if all_u[0] is not None:
        u_code = np.concatenate(all_u)
        u_cgs  = u_code * units["unit_vel_cms"]**2
        T      = GAMMA_MINUS1 * u_cgs * 1.22 * PROTONMASS / BOLTZMANN

    print(f"  Gas particles extracted: {len(mass):,}")
    return coords, mass, nH, Z, T, hsml, center_phys


def compute_dz_particle_based(gas_coords, gas_mass, gas_nH, gas_Z, gas_T,
                               gas_hsml, dust_coords, dust_mass, center,
                               r_max_kpc, min_gas_mass_msun=0.0):
    """
    Particle-based D/Z estimation: for each gas particle, sum dust mass
    within its SPH smoothing length using a KD-tree.  This is physically
    correct and avoids grid resolution issues entirely.

    If smoothing lengths are unavailable, falls back to the mean inter-
    particle spacing as a fixed kernel radius.
    """
    from scipy.spatial import cKDTree

    gp = gas_coords  - center
    dp = dust_coords - center

    gsel = np.sqrt(np.sum(gp**2, axis=1)) < r_max_kpc
    dsel = np.sqrt(np.sum(dp**2, axis=1)) < r_max_kpc

    gp   = gp[gsel];   gmas = gas_mass[gsel]
    gnH  = gas_nH[gsel]; gZ  = gas_Z[gsel]
    gT   = gas_T[gsel]   if gas_T   is not None else None
    ghsml = gas_hsml[gsel] if gas_hsml is not None else None
    dp   = dp[dsel];   dmas = dust_mass[dsel]

    print(f"  Gas after r-cut:  {len(gmas):,}")
    print(f"  Dust after r-cut: {len(dmas):,}")

    if len(dmas) == 0:
        print("  WARNING: no dust in aperture — D/Z will be zero everywhere")

    # Fallback kernel radius: mean gas inter-particle spacing
    if ghsml is None or np.all(ghsml == 0):
        vol_per_part = (4/3 * np.pi * r_max_kpc**3) / max(len(gmas), 1)
        fallback_r   = (3 * vol_per_part / (4 * np.pi))**(1/3)
        ghsml = np.full(len(gmas), fallback_r)
        print(f"  No smoothing lengths — using fallback kernel r = {fallback_r:.2f} kpc")

    # Build KD-tree on dust positions
    DZ_arr = np.zeros(len(gmas))
    if len(dmas) > 0:
        tree = cKDTree(dp)
        for idx in range(len(gmas)):
            metal_mass = gmas[idx] * gZ[idx]
            if metal_mass <= 0:
                continue
            h = ghsml[idx]
            neighbors = tree.query_ball_point(gp[idx], r=h)
            local_dust = dmas[neighbors].sum() if neighbors else 0.0
            DZ_arr[idx] = local_dust / metal_mass

    # Filter: require gas mass above floor and positive metallicity
    ok = (gmas > min_gas_mass_msun) & (gZ > 0) & np.isfinite(DZ_arr)
    return gnH[ok], DZ_arr[ok], gZ[ok], (gT[ok] if gT is not None else None), gmas[ok]
    gp = gas_coords  - center
    dp = dust_coords - center

    gsel = np.sqrt(np.sum(gp**2, axis=1)) < r_max_kpc
    dsel = np.sqrt(np.sum(dp**2, axis=1)) < r_max_kpc
    gp, gmas, gnH, gZ = gp[gsel], gas_mass[gsel], gas_nH[gsel], gas_Z[gsel]
    gT   = gas_T[gsel]  if gas_T    is not None else None
    dp, dmas = dp[dsel], dust_mass[dsel]

    print(f"  Gas after r-cut:  {gsel.sum():,}")
    print(f"  Dust after r-cut: {dsel.sum():,}")

    edge = np.linspace(-r_max_kpc, r_max_kpc, n_bins + 1)

    def digitize(pos):
        i = np.searchsorted(edge, pos[:, 0]) - 1
        j = np.searchsorted(edge, pos[:, 1]) - 1
        k = np.searchsorted(edge, pos[:, 2]) - 1
        v = (i >= 0) & (i < n_bins) & (j >= 0) & (j < n_bins) & (k >= 0) & (k < n_bins)
        return i[v], j[v], k[v], v

    gi, gj, gk, gv = digitize(gp)
    gmas_v, gnH_v, gZ_v = gmas[gv], gnH[gv], gZ[gv]
    gT_v = gT[gv] if gT is not None else None

    sh = (n_bins, n_bins, n_bins)
    bin_gmas   = np.zeros(sh); bin_gmetal = np.zeros(sh)
    bin_nH_wt  = np.zeros(sh); bin_T_wt   = np.zeros(sh)

    np.add.at(bin_gmas,   (gi, gj, gk), gmas_v)
    np.add.at(bin_gmetal, (gi, gj, gk), gmas_v * gZ_v)
    np.add.at(bin_nH_wt,  (gi, gj, gk), gmas_v * gnH_v)
    if gT_v is not None:
        np.add.at(bin_T_wt, (gi, gj, gk), gmas_v * gT_v)

    bin_dust = np.zeros(sh)
    if len(dmas) > 0:
        di, dj, dk, dv = digitize(dp)
        np.add.at(bin_dust, (di, dj, dk), dmas[dv])

    bg  = bin_gmas.ravel();   bm  = bin_gmetal.ravel()
    bd  = bin_dust.ravel();   bnH = bin_nH_wt.ravel()
    bT  = bin_T_wt.ravel()

    ok = (bg > min_gas_mass) & (bm > 0)
    bg, bm, bd, bnH, bT = bg[ok], bm[ok], bd[ok], bnH[ok], bT[ok]

    nH_mean = bnH / bg;  T_mean = bT / bg
    Z_mean  = bm  / bg;  DZ     = bd / bm

    finite = np.isfinite(DZ) & np.isfinite(nH_mean) & (DZ >= 0) & (nH_mean > 0)
    return nH_mean[finite], DZ[finite], Z_mean[finite], T_mean[finite], bg[finite]


def make_plot(nH, DZ, Z_arr, T_arr, gas_mass_bins,
              scale_factor, r_max_kpc, n_dust_total,
              color_by, output_path):

    redshift = 1.0 / scale_factor - 1.0
    fig, ax  = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")

    log_nH = np.log10(np.clip(nH, 1e-6, None))
    log_DZ = np.log10(np.clip(DZ, 1e-6, None))

    h2d, xe, ye = np.histogram2d(log_nH, log_DZ, bins=80,
                                  range=[[-5, 4], [-4, 0.7]])
    h2d = np.where(h2d > 0, h2d, np.nan)
    ax.pcolormesh(xe, ye, h2d.T, cmap="Greys",
                  norm=mcolors.LogNorm(), zorder=1, rasterized=True)

    if color_by == "temperature" and T_arr is not None:
        c_vals = np.log10(np.clip(T_arr, 10, None))
        cmap, clabel = "RdYlBu_r", r"$\log_{10}(T\ /\ \mathrm{K})$"
        vmin, vmax = 2.0, 7.0
    elif color_by == "metallicity":
        c_vals = np.log10(np.clip(Z_arr / Z_SOLAR, 1e-3, None))
        cmap, clabel = "viridis", r"$\log_{10}(Z/Z_\odot)$"
        vmin, vmax = -3.0, 1.0
    else:
        c_vals = np.log10(gas_mass_bins)
        cmap, clabel = "plasma", r"$\log_{10}(M_\mathrm{gas}\ /\ M_\odot)$"
        vmin, vmax = None, None

    idx = np.arange(len(nH))
    if len(idx) > 30000:
        idx = np.random.choice(idx, 30000, replace=False)

    sc = ax.scatter(log_nH[idx], log_DZ[idx], c=c_vals[idx],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=3, alpha=0.5, zorder=2, rasterized=True)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label(clabel, fontsize=11)

    # Jenkins 2009 / MW depletion observational band
    obs_nH    = np.array([-3.0, -2.0, -1.0,  0.0,  1.0,  2.0,  3.0])
    obs_DZ_lo = np.array([-2.5, -1.8, -1.1, -0.55,-0.40,-0.38,-0.35])
    obs_DZ_hi = np.array([-1.5, -0.9, -0.55,-0.25,-0.20,-0.18,-0.15])
    ax.fill_between(obs_nH, obs_DZ_lo, obs_DZ_hi,
                    color="cornflowerblue", alpha=0.25, zorder=3,
                    label="Jenkins 2009 / MW depletion (approx.)")
    ax.plot(obs_nH, (obs_DZ_lo + obs_DZ_hi) / 2,
            color="cornflowerblue", lw=1.5, ls="--", zorder=4)

    ax.axhline(np.log10(0.4), color="goldenrod", lw=1.5, ls=":",
               zorder=5, label="D/Z = 0.4 (MW canonical)")

    for nH_val, label in [(-1, "WNM/CNM"), (1, "CNM/dense")]:
        ax.axvline(nH_val, color="white", lw=0.8, ls=":", alpha=0.6, zorder=3)
        ax.text(nH_val + 0.05, -3.8, label, color="white",
                fontsize=7, rotation=90, va="bottom", alpha=0.7)

    ax.set_xlabel(r"$\log_{10}(n_\mathrm{H}\ /\ \mathrm{cm}^{-3})$", fontsize=13)
    ax.set_ylabel(r"$\log_{10}(D/Z)$", fontsize=13)
    ax.set_title(
        f"D/Z vs. Hydrogen Density  —  z = {redshift:.2f}\n"
        f"R < {r_max_kpc:.0f} kpc   |   {n_dust_total:,} dust particles",
        fontsize=12)
    ax.set_xlim(-5, 4); ax.set_ylim(-4, 0.7)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([-4, -2, 0, 2, 4])
    ax2.set_xticklabels([r"$10^{-4}$",r"$10^{-2}$",r"$10^{0}$",
                         r"$10^{2}$",r"$10^{4}$"], fontsize=8)
    ax2.set_xlabel(r"$n_\mathrm{H}\ (\mathrm{cm}^{-3})$", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot",
                        help="Snapshot base (e.g. ../snapdir_049/snapshot_049)")
    parser.add_argument("catalog",
                        help="Subfind catalog (e.g. ../groups_049/fof_subhalo_tab_049.0.hdf5)")
    parser.add_argument("--r-max",        type=float, default=200.0,
                        help="Aperture radius in physical kpc (default: 200, 0=use 2×halfmass_rad)")
    parser.add_argument("--color-by",     choices=["temperature","metallicity","mass"],
                        default="temperature")
    parser.add_argument("--output",       default="dz_vs_nH.png")
    args = parser.parse_args()

    units = get_unit_conversions(args.snapshot)
    a = units["scale_factor"]
    print(f"  a = {a:.4f}  z = {1/a-1:.2f}")

    print("\nLoading target halo...")
    halo     = load_target_halo(args.catalog, args.snapshot, verbose=True)
    halo_pos = halo["halo_info"]["position"]
    r_max    = args.r_max if args.r_max > 0 else halo["halo_info"]["halfmass_rad"] * 2.0
    print(f"  Center: {halo_pos} kpc  |  Aperture: {r_max:.1f} kpc")

    print("\nExtracting gas (PartType0)...")
    gas_coords, gas_mass, gas_nH, gas_Z, gas_T, gas_hsml, center_phys = \
        read_gas_spatially(args.snapshot, halo_pos, r_max, units)

    print("\nExtracting dust (PartType6)...")
    a, h = units["scale_factor"], units["hubble"]
    kpc_per_code  = units["unit_len_cm"] / 3.085678e21 * a / h
    msun_per_code = units["unit_mass_g"] / 1.989e33 / h

    dust_raw = extract_dust_spatially(args.snapshot, halo_pos,
                                      radius_kpc=r_max)
    if dust_raw is not None:
        dust_coords = dust_raw["Coordinates"] * kpc_per_code
        dust_mass   = dust_raw["Masses"]      * msun_per_code
    else:
        print("  WARNING: No PartType6 found")
        dust_coords = np.zeros((0, 3))
        dust_mass   = np.zeros(0)

    nH, DZ, Z_arr, T_arr, gmass = compute_dz_particle_based(
        gas_coords, gas_mass, gas_nH, gas_Z, gas_T, gas_hsml,
        dust_coords, dust_mass,
        center=center_phys, r_max_kpc=r_max)

    print(f"\n  Populated bins: {len(nH):,}")
    if len(nH) > 0:
        ok = DZ > 0
        if ok.any():
            print(f"  D/Z range: {DZ[ok].min():.3e} – {DZ[ok].max():.3e}")
        print(f"  Mass-weighted D/Z: {np.average(DZ, weights=gmass):.3f}")

    make_plot(nH, DZ, Z_arr, T_arr, gmass,
              scale_factor=a, r_max_kpc=r_max,
              n_dust_total=len(dust_mass),
              color_by=args.color_by,
              output_path=args.output)


if __name__ == "__main__":
    main()
