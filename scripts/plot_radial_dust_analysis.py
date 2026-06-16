#!/usr/bin/env python3
"""
plot_radial_dust_analysis.py

Radial dust-to-metal and dust-to-gas ratio profiles with TEMPERATURE FILTERING.

CRITICAL: Measures D/M in WARM ISM (T<5e5 K) where dust survives thermal sputtering!

Halo identification:
  Reads ALL catalog chunks and selects the FOF group with the highest
  STELLAR mass (argmax GroupMassType[:,4]) — same logic as snap_overview.py
  and run_radial_evolution.py. Falls back to argmax(M200) before stars form.
  Never uses SubhaloPos[0] or GroupPos[0] of chunk .0 only.

Unit conventions (Gadget-4 defaults):
  Positions  : comoving kpc/h  →  physical kpc  via  x * a / h
  h          : from f["Parameters"].attrs["HubbleParam"]  (NOT Header)
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path

try:
    from halo_utils import compute_radial_distance
except ImportError:
    print("ERROR: This script requires halo_utils.py in the same directory")
    exit(1)

MSUN_PER_CODE = 1e10   # Gadget default: 1 code mass unit = 1e10 M_sun/h

# ─────────────────────────────────────────────────────────────────────────────
# Primary halo identification — stellar mass argmax across ALL catalog chunks
# ─────────────────────────────────────────────────────────────────────────────

def get_primary_halo(catalog_path):
    """
    Identify the primary halo (Halo 569) as the FOF group with the highest
    STELLAR mass (argmax GroupMassType[:,4]) across ALL catalog chunks.

    Using stellar mass correctly targets the zoom galaxy rather than
    potentially more massive but star-poor neighbouring dark matter halos.
    Falls back to argmax(M200) at early epochs before stars form.

    Returns (center_phys_kpc, halo_mass_msun, a, h)
    """
    p         = Path(catalog_path)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks    = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    all_pos   = []
    all_m200  = []
    all_mstar = []
    a = h = None

    for chunk in chunks:
        with h5py.File(str(chunk), "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or "Group_M_Crit200" not in grp:
                continue
            if len(grp["GroupPos"]) == 0:
                continue
            if a is None:
                a = float(f["Header"].attrs["Time"])
                h = float(f["Parameters"].attrs["HubbleParam"])
            all_pos.append(grp["GroupPos"][:])
            all_m200.append(grp["Group_M_Crit200"][:])
            if "GroupMassType" in grp:
                all_mstar.append(grp["GroupMassType"][:, 4])
            else:
                all_mstar.append(np.zeros(len(grp["GroupPos"])))

    if not all_m200 or a is None:
        raise RuntimeError(f"No usable groups found in catalog: {catalog_path}")

    pos_all   = np.concatenate(all_pos,   axis=0)
    m200_all  = np.concatenate(all_m200,  axis=0)
    mstar_all = np.concatenate(all_mstar, axis=0)

    # Select by stellar mass; fall back to M200 if no stars yet
    if mstar_all.max() > 0:
        idx          = int(np.argmax(mstar_all))
        selection_by = "M*"
    else:
        idx          = int(np.argmax(m200_all))
        selection_by = "M200 (no stars yet)"

    center_phys = pos_all[idx]         * a / h   # physical kpc
    m200_msun   = float(m200_all[idx]) * MSUN_PER_CODE / h

    print(f"  Primary halo selected by: {selection_by}")
    print(f"  FOF group index: {idx}  (across {sum(len(x) for x in all_pos)} total groups)")
    print(f"  Position: [{center_phys[0]:.2f}, {center_phys[1]:.2f}, {center_phys[2]:.2f}] pkpc")
    print(f"  M200: {m200_msun:.2e} M☉")

    return center_phys, m200_msun, a, h


# ─────────────────────────────────────────────────────────────────────────────
# Temperature calculation
# ─────────────────────────────────────────────────────────────────────────────

def get_gas_temperature(internal_energy, electron_abundance):
    GAMMA         = 5.0 / 3.0
    BOLTZMANN     = 1.38064852e-16   # erg/K
    PROTONMASS    = 1.6726219e-24    # g
    HYDROGEN_MASSFRAC = 0.76

    XH  = HYDROGEN_MASSFRAC
    Y   = (1.0 - XH) / (4.0 * XH)
    mu  = (1.0 + 4.0 * Y) / (1.0 + Y + electron_abundance)
    u_cgs = internal_energy * 1e10   # (km/s)^2 → (cm/s)^2 = erg/g
    return (GAMMA - 1.0) * u_cgs * PROTONMASS * mu / BOLTZMANN


# ─────────────────────────────────────────────────────────────────────────────
# Particle extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_particles_in_sphere(snapshot_base, center_pos, radius_kpc,
                                particle_type, a, h):
    """
    Extract all particles of given type within radius_kpc of center_pos.
    Coordinates are converted from comoving kpc/h to physical kpc using a, h.
    """
    import glob
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        print(f"ERROR: No snapshot files found matching {snapshot_base}.*.hdf5")
        return None

    print(f"  Found {len(files)} snapshot files")

    ptype_name = f"PartType{particle_type}"

    all_coords            = []
    all_masses            = []
    all_metallicity       = []
    all_internal_energy   = []
    all_electron_abundance= []

    with h5py.File(files[0], "r") as f:
        box_code     = float(f["Header"].attrs["BoxSize"])
        box_phys_kpc = box_code * a / h

    print(f"  Extracting {ptype_name} particles within {radius_kpc:.1f} pkpc...")

    for file in files:
        with h5py.File(file, "r") as f:
            if ptype_name not in f:
                continue

            # Coordinates: comoving kpc/h → physical kpc
            coords = f[ptype_name]["Coordinates"][:] * a / h
            # Masses: code units → M_sun
            masses = f[ptype_name]["Masses"][:] * MSUN_PER_CODE / h

            dx = coords - center_pos
            dx -= box_phys_kpc * np.round(dx / box_phys_kpc)
            r  = np.sqrt((dx**2).sum(axis=1))
            mask = r < radius_kpc

            if mask.sum() == 0:
                continue

            all_coords.append(coords[mask])
            all_masses.append(masses[mask])

            if particle_type == 0:
                if "Metallicity" in f[ptype_name]:
                    all_metallicity.append(f[ptype_name]["Metallicity"][:][mask])
                if "InternalEnergy" in f[ptype_name]:
                    all_internal_energy.append(
                        f[ptype_name]["InternalEnergy"][:][mask])
                if "ElectronAbundance" in f[ptype_name]:
                    all_electron_abundance.append(
                        f[ptype_name]["ElectronAbundance"][:][mask])

    if not all_coords:
        print(f"  WARNING: No {ptype_name} particles found!")
        return None

    result = {
        "Coordinates": np.vstack(all_coords),
        "Masses":      np.concatenate(all_masses),
    }

    if all_metallicity:
        result["Metallicity"] = np.concatenate(all_metallicity)
    if all_internal_energy:
        result["InternalEnergy"] = np.concatenate(all_internal_energy)
    if all_electron_abundance:
        result["ElectronAbundance"] = np.concatenate(all_electron_abundance)

    if particle_type == 0 and "InternalEnergy" in result:
        ne = result.get("ElectronAbundance",
                        np.ones(len(result["Masses"])))
        result["Temperature"] = get_gas_temperature(
            result["InternalEnergy"], ne)
        T = result["Temperature"]
        print(f"  Temperature range: {T.min():.1e} - {T.max():.1e} K")
        warm = T < 5e5
        print(f"  Gas below sputtering (T<5e5 K): {warm.sum():,} "
              f"({100*warm.sum()/len(T):.1f}%)")
        print(f"  Gas above sputtering (T>5e5 K): {(~warm).sum():,} "
              f"({100*(~warm).sum()/len(T):.1f}%)")

    print(f"  Extracted {len(result['Masses']):,} particles, "
          f"total mass = {result['Masses'].sum():.2e} M☉")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Radial binning — monotonic bin guard
# ─────────────────────────────────────────────────────────────────────────────

def make_radial_bins(r_max):
    """
    Build radial bin edges appropriate for r_max.
    Guarantees strictly monotonically increasing edges with no duplicates,
    which is required by np.digitize.
    """
    if r_max > 150:
        inner  = np.linspace(0,  10, 11)
        middle = np.linspace(10, 50, 21)[1:]
        outer  = np.linspace(50, r_max, 16)[1:]
    elif r_max > 30:
        inner  = np.linspace(0,  10, 11)
        middle = np.linspace(10, 30, 11)[1:]
        outer  = np.linspace(30, r_max, 11)[1:]
    else:
        # Small r_max (early epochs with tiny halos)
        # Use at least 10 bins; minimum bin width = r_max/20
        n      = max(10, int(r_max / 0.5))
        inner  = np.linspace(0, r_max, n + 1)
        return np.unique(inner)

    r_bins = np.unique(np.concatenate([inner, middle, outer]))

    # Final safety: ensure strictly increasing
    if not np.all(np.diff(r_bins) > 0):
        r_bins = np.linspace(0, r_max, 21)

    return r_bins


def compute_radial_profile(halo, halo_pos, r_max=200):
    """
    Compute radial profiles with temperature filtering.
    Uses make_radial_bins() to guarantee monotonic bin edges.
    """
    r_bins   = make_radial_bins(r_max)
    n_bins   = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    profile = {
        "r_bins":          r_bins,
        "r_centers":       r_centers,
        "dust_mass":       np.zeros(n_bins),
        "metal_mass_all":  np.zeros(n_bins),
        "metal_mass_warm": np.zeros(n_bins),
        "gas_mass_all":    np.zeros(n_bins),
        "gas_mass_warm":   np.zeros(n_bins),
        "dust_count":      np.zeros(n_bins, dtype=int),
        "gas_count_all":   np.zeros(n_bins, dtype=int),
        "gas_count_warm":  np.zeros(n_bins, dtype=int),
        "DM_ratio_all":    np.zeros(n_bins),
        "DM_ratio_warm":   np.zeros(n_bins),
        "DG_ratio_all":    np.zeros(n_bins),
        "DG_ratio_warm":   np.zeros(n_bins),
    }

    if "dust" in halo and halo["dust"] is not None:
        dust_r       = compute_radial_distance(
            halo["dust"]["Coordinates"], halo_pos)
        dust_bin_idx = np.digitize(dust_r, r_bins) - 1
        for i in range(n_bins):
            mask = dust_bin_idx == i
            if mask.sum() > 0:
                profile["dust_mass"][i]  = halo["dust"]["Masses"][mask].sum()
                profile["dust_count"][i] = int(mask.sum())

    if "gas" in halo and halo["gas"] is not None:
        gas_r       = compute_radial_distance(
            halo["gas"]["Coordinates"], halo_pos)
        gas_bin_idx = np.digitize(gas_r, r_bins) - 1
        T_gas       = halo["gas"].get(
            "Temperature",
            np.ones(len(halo["gas"]["Masses"])) * 1e4)
        warm_mask   = T_gas < 5e5

        for i in range(n_bins):
            mask_all  = gas_bin_idx == i
            mask_warm = mask_all & warm_mask

            if mask_all.sum() > 0:
                profile["gas_mass_all"][i]  = \
                    halo["gas"]["Masses"][mask_all].sum()
                profile["gas_count_all"][i] = int(mask_all.sum())
                if "Metallicity" in halo["gas"]:
                    profile["metal_mass_all"][i] = (
                        halo["gas"]["Masses"][mask_all]
                        * halo["gas"]["Metallicity"][mask_all]).sum()

            if mask_warm.sum() > 0:
                profile["gas_mass_warm"][i]  = \
                    halo["gas"]["Masses"][mask_warm].sum()
                profile["gas_count_warm"][i] = int(mask_warm.sum())
                if "Metallicity" in halo["gas"]:
                    profile["metal_mass_warm"][i] = (
                        halo["gas"]["Masses"][mask_warm]
                        * halo["gas"]["Metallicity"][mask_warm]).sum()

    # Ratios
    m = profile["metal_mass_all"] > 0
    profile["DM_ratio_all"][m] = (
        profile["dust_mass"][m] / profile["metal_mass_all"][m])
    m = profile["gas_mass_all"] > 0
    profile["DG_ratio_all"][m] = (
        profile["dust_mass"][m] / profile["gas_mass_all"][m])
    m = profile["metal_mass_warm"] > 0
    profile["DM_ratio_warm"][m] = (
        profile["dust_mass"][m] / profile["metal_mass_warm"][m])
    m = profile["gas_mass_warm"] > 0
    profile["DG_ratio_warm"][m] = (
        profile["dust_mass"][m] / profile["gas_mass_warm"][m])

    profile["DM_ratio_all"]  = np.clip(profile["DM_ratio_all"],  0, 1.0)
    profile["DM_ratio_warm"] = np.clip(profile["DM_ratio_warm"], 0, 1.0)
    profile["DG_ratio_all"]  = np.clip(profile["DG_ratio_all"],  0, 0.1)
    profile["DG_ratio_warm"] = np.clip(profile["DG_ratio_warm"], 0, 0.1)

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Dust distribution diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dust_distribution(halo, halo_pos):
    if "dust" not in halo or halo["dust"] is None:
        print("No dust particles to analyze!")
        return None

    dust_r      = compute_radial_distance(
        halo["dust"]["Coordinates"], halo_pos)
    dust_masses = halo["dust"]["Masses"]

    print("\n" + "="*60)
    print("DUST DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"\nTotal dust particles: {len(dust_r):,}")
    print(f"Total dust mass: {dust_masses.sum():.2e} M☉")

    print("\nRadial Distribution:")
    radial_bins = [0, 5, 10, 20, 30, 50, 100, 150, 200, 500]
    for i in range(len(radial_bins) - 1):
        mask  = (dust_r >= radial_bins[i]) & (dust_r < radial_bins[i+1])
        count = int(mask.sum())
        if count > 0:
            mass = float(dust_masses[mask].sum())
            print(f"  {radial_bins[i]:4.0f}-{radial_bins[i+1]:4.0f} kpc: "
                  f"{count:6d} particles, {mass:10.3e} M☉")

    print("\nMass Distribution:")
    print(f"  Min mass:    {dust_masses.min():.2e} M☉")
    print(f"  Max mass:    {dust_masses.max():.2e} M☉")
    print(f"  Median mass: {np.median(dust_masses):.2e} M☉")

    far_mask  = dust_r > 100
    far_count = int(far_mask.sum())
    if far_count > 0:
        print(f"\nWARNING: {far_count} dust particles beyond 100 kpc "
              f"({100*far_count/len(dust_r):.1f}%)")
    print("="*60 + "\n")

    return {"dust_r": dust_r, "far_dust_count": far_count,
            "dust_masses": dust_masses}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def plot_radial_profiles(profile, halo_mass, redshift,
                         dust_analysis=None, output_file=None, show_plot=0):
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, top=0.92)
    fig.suptitle(
        f"Radial Dust Analysis (Thermal Sputtering Threshold) — "
        f"M={halo_mass:.2e} M☉ — z={redshift:.3f}",
        fontsize=14, fontweight="bold")

    r         = profile["r_centers"]
    r_max_plot= min(profile["r_bins"][-1], 200)

    ax1 = fig.add_subplot(gs[0, 0])
    m   = (profile["DM_ratio_all"] > 0) & (profile["metal_mass_all"] > 0)
    ax1.plot(r[m], profile["DM_ratio_all"][m], "s--",
             color="gray", lw=1.5, markersize=4, alpha=0.5, label="D/M (all gas)")
    m   = (profile["DM_ratio_warm"] > 0) & (profile["metal_mass_warm"] > 0)
    ax1.plot(r[m], profile["DM_ratio_warm"][m], "o-",
             color="darkblue", lw=2, markersize=5, label="D/M (warm ISM, T<5e5 K)")
    ax1.axhline(0.4, color="red", ls="--", alpha=0.7, lw=2, label="MW ISM (0.4)")
    ax1.set(xlabel="Radius (kpc)", ylabel="Dust-to-Metal Ratio",
            ylim=(0, 1.0), xlim=(0, r_max_plot))
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_title("Dust-to-Metal Ratio", fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    m   = (profile["DG_ratio_all"] > 0) & (profile["gas_mass_all"] > 0)
    ax2.plot(r[m], profile["DG_ratio_all"][m] * 100, "s--",
             color="gray", lw=1.5, markersize=4, alpha=0.5, label="D/G (all gas)")
    m   = (profile["DG_ratio_warm"] > 0) & (profile["gas_mass_warm"] > 0)
    ax2.plot(r[m], profile["DG_ratio_warm"][m] * 100, "o-",
             color="darkgreen", lw=2, markersize=5,
             label="D/G (warm ISM, T<5e5 K)")
    ax2.axhline(0.7, color="red", ls="--", alpha=0.7, lw=2, label="MW (~0.7%)")
    ax2.set(xlabel="Radius (kpc)", ylabel="Dust-to-Gas Ratio (%)",
            ylim=(0, 5.0), xlim=(0, r_max_plot))
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("Dust-to-Gas Ratio", fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    m   = profile["dust_mass"] > 0
    ax3.semilogy(r[m], profile["dust_mass"][m], "o-", color="brown", lw=2, markersize=5)
    ax3.set(xlabel="Radius (kpc)", ylabel="Dust Mass per bin (M$_\\odot$)",
            xlim=(0, r_max_plot))
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Dust Mass Profile", fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 0])
    m   = profile["gas_mass_all"] > 0
    ax4.semilogy(r[m], profile["gas_mass_all"][m], "s--",
                 color="lightblue", lw=1.5, markersize=4, alpha=0.7, label="All gas")
    m   = profile["gas_mass_warm"] > 0
    ax4.semilogy(r[m], profile["gas_mass_warm"][m], "o-",
                 color="steelblue", lw=2, markersize=5, label="Warm ISM")
    ax4.set(xlabel="Radius (kpc)", ylabel="Gas Mass per bin (M$_\\odot$)",
            xlim=(0, r_max_plot))
    ax4.grid(True, alpha=0.3); ax4.legend(fontsize=8)
    ax4.set_title("Gas Mass Profile", fontweight="bold")

    ax5 = fig.add_subplot(gs[1, 1])
    m   = profile["metal_mass_all"] > 0
    ax5.semilogy(r[m], profile["metal_mass_all"][m], "s--",
                 color="orange", lw=1.5, markersize=4, alpha=0.5, label="All gas")
    m   = profile["metal_mass_warm"] > 0
    ax5.semilogy(r[m], profile["metal_mass_warm"][m], "o-",
                 color="darkorange", lw=2, markersize=5, label="Warm ISM")
    ax5.set(xlabel="Radius (kpc)", ylabel="Metal Mass per bin (M$_\\odot$)",
            xlim=(0, r_max_plot))
    ax5.grid(True, alpha=0.3); ax5.legend(fontsize=8)
    ax5.set_title("Metal Mass Profile", fontweight="bold")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(r, np.maximum(profile["dust_count"], 1), "o-",
                 color="brown", lw=2, markersize=5, label="Dust")
    ax6.semilogy(r, np.maximum(profile["gas_count_all"], 1), "s--",
                 color="lightblue", lw=1.5, markersize=4, alpha=0.7, label="Gas (all)")
    ax6.semilogy(r, np.maximum(profile["gas_count_warm"], 1), "o-",
                 color="steelblue", lw=2, markersize=5, label="Gas (warm)")
    ax6.set(xlabel="Radius (kpc)", ylabel="Particle Count",
            xlim=(0, r_max_plot))
    ax6.grid(True, alpha=0.3); ax6.legend(fontsize=8)
    ax6.set_title("Particle Counts", fontweight="bold")

    if dust_analysis:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(np.log10(np.maximum(dust_analysis["dust_masses"], 1e-30)),
                 bins=50, color="brown", alpha=0.7, edgecolor="black")
        ax7.set(xlabel="log$_{10}$(Dust Mass / M$_\\odot$)", ylabel="Count")
        ax7.grid(True, alpha=0.3)
        ax7.set_title("Dust Mass Distribution", fontweight="bold")

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(dust_analysis["dust_r"], bins=50,
                 range=(0, r_max_plot), color="brown", alpha=0.7, edgecolor="black")
        ax8.set(xlabel="Radius (kpc)", ylabel="Dust Particle Count")
        ax8.grid(True, alpha=0.3)
        ax8.set_title("Dust Radial Distribution", fontweight="bold")

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    td   = profile["dust_mass"].sum()
    tga  = profile["gas_mass_all"].sum()
    tgw  = profile["gas_mass_warm"].sum()
    tma  = profile["metal_mass_all"].sum()
    tmw  = profile["metal_mass_warm"].sum()
    wf   = 100 * tgw / tga if tga > 0 else 0
    dm_a = td / tma if tma > 0 else 0
    dm_w = td / tmw if tmw > 0 else 0
    dg_a = td / tga if tga > 0 else 0
    dg_w = td / tgw if tgw > 0 else 0
    m30  = profile["r_centers"] < 30
    dm_i = (profile["dust_mass"][m30].sum() /
            profile["metal_mass_warm"][m30].sum()
            if profile["metal_mass_warm"][m30].sum() > 0 else 0)

    txt = (f"Summary Statistics:\n\n"
           f"Total Dust:          {td:.2e} M☉\n"
           f"Total Gas (all):     {tga:.2e} M☉\n"
           f"Total Gas (warm):    {tgw:.2e} M☉ ({wf:.1f}%)\n"
           f"Total Metals (all):  {tma:.2e} M☉\n"
           f"Total Metals (warm): {tmw:.2e} M☉\n\n"
           f"D/M (all gas):       {dm_a:.3f}\n"
           f"D/M (warm ISM):      {dm_w:.3f}  ← PHYSICAL!\n"
           f"D/M (r<30 kpc):      {dm_i:.3f}  ← DISK ONLY!\n\n"
           f"D/G (all gas):       {dg_a*100:.3f}%\n"
           f"D/G (warm ISM):      {dg_w*100:.3f}%\n\n"
           f"Dust particles:  {profile['dust_count'].sum():,}\n"
           f"Gas (all):       {profile['gas_count_all'].sum():,}\n"
           f"Gas (warm ISM):  {profile['gas_count_warm'].sum():,}")
    if dust_analysis and dust_analysis["far_dust_count"] > 0:
        txt += f"\n\n⚠ {dust_analysis['far_dust_count']} dust beyond 100 kpc"

    ax9.text(0.05, 0.95, txt, transform=ax9.transAxes,
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {output_file}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Radial dust analysis with temperature filtering")
    parser.add_argument("--catalog",   required=True)
    parser.add_argument("--snapshot",  required=True)
    parser.add_argument("--out",       default="radial_dust_analysis.png")
    parser.add_argument("--rmax",      type=float, default=200)
    parser.add_argument("--show_plot", type=int,   default=0)
    args = parser.parse_args()

    print("="*60)
    print("RADIAL DUST ANALYSIS - THERMAL SPUTTERING THRESHOLD")
    print("="*60)

    print("\nReading target halo from catalog...")
    halo_pos, halo_mass, a, h = get_primary_halo(args.catalog)

    with h5py.File(f"{args.snapshot}.0.hdf5", "r") as f:
        redshift = float(f["Header"].attrs["Redshift"])
    print(f"Redshift: z = {redshift:.3f}")

    print(f"\n{'='*60}")
    print(f"SPATIAL EXTRACTION (r < {args.rmax} kpc)")
    print(f"{'='*60}")

    halo = {}
    print("\n1. Extracting GAS particles (with temperature):")
    halo["gas"]  = extract_particles_in_sphere(
        args.snapshot, halo_pos, args.rmax, 0, a, h)

    print("\n2. Extracting DUST particles:")
    halo["dust"] = extract_particles_in_sphere(
        args.snapshot, halo_pos, args.rmax, 6, a, h)

    if halo["gas"] is None:
        print("ERROR: No gas particles found!")
        return

    dust_analysis = None
    if halo["dust"] is not None:
        dust_analysis = analyze_dust_distribution(halo, halo_pos)

    print("\nComputing radial profiles with temperature filtering...")
    profile = compute_radial_profile(halo, halo_pos, r_max=args.rmax)

    print("\nCreating plots...")
    plot_radial_profiles(profile, halo_mass, redshift,
                         dust_analysis=dust_analysis,
                         output_file=args.out,
                         show_plot=args.show_plot)

    npz_file = args.out.replace(".png", ".npz")
    np.savez(npz_file,
             r_centers        = profile["r_centers"],
             r_bins           = profile["r_bins"],
             DM_ratio_all     = profile["DM_ratio_all"],
             DM_ratio_warm    = profile["DM_ratio_warm"],
             DG_ratio_all     = profile["DG_ratio_all"],
             DG_ratio_warm    = profile["DG_ratio_warm"],
             dust_mass        = profile["dust_mass"],
             metal_mass_all   = profile["metal_mass_all"],
             metal_mass_warm  = profile["metal_mass_warm"],
             gas_mass_all     = profile["gas_mass_all"],
             gas_mass_warm    = profile["gas_mass_warm"],
             dust_count       = profile["dust_count"],
             gas_count_all    = profile["gas_count_all"],
             gas_count_warm   = profile["gas_count_warm"])
    print(f"Saved data: {npz_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
