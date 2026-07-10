#!/usr/bin/env python3
"""
dust_snapshot_summary.py — Complete, non-sampled dust statistics from a
CosmicGrain snapshot (default: the last complete snapshot in output_dir).

Unlike dust_log_summary.py, which parses the Gadget-4 stdout log (where most
per-event detail is intentionally SPARSE-SAMPLED, and cumulative log totals
can mix contributions from different code versions across job restarts),
this script reads the actual dust particle data (PartType6) directly from
the snapshot HDF5 files. Every statistic here is computed from EVERY dust
particle present at that snapshot -- nothing sampled, nothing historical.

Use dust_log_summary.py to understand the PROCESS (which physics fired, how
often, any bugs). Use this script to understand the STATE (what the dust
population actually looks like right now).

Usage:
    python3 dust_snapshot_summary.py ../S10_output_1024/
    python3 dust_snapshot_summary.py ../S10_output_1024/ --snap 49
    python3 dust_snapshot_summary.py ../S10_output_1024/ --ism-radius 20
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from halo_utils import (
    get_halo569_reference,
    get_halo569,
    read_snap_header,
    load_particles_within_radius,
    convert_code_mass_to_msun,
)


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


BOX_WIDTH = 68


def box_top(title):
    label = f" {title} "
    dashes = "─" * max(0, BOX_WIDTH - len(label) - 1)
    print(f"┌{label}{dashes}┐")


def box_bottom():
    print(f"└{'─' * BOX_WIDTH}┘")
    print()


def ok(text):
    return f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}"


def warn(text):
    return f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}"


def fail(text):
    return f"{Colors.FAIL}❌ {text}{Colors.ENDC}"


# Milky Way literature reference values (Konstantopoulou+2024; Draine+2007)
MW_DTG = 0.0067
MW_DTZ = 0.45


DUST_FIELDS_6 = ["Coordinates", "Masses", "Velocities", "GrainRadius",
                 "GrainType", "CarbonFraction", "DustTemperature",
                 "DustFormationTime", "ParticleIDs"]
GAS_FIELDS_0 = ["Coordinates", "Masses", "Metallicity"]


def age_gyr_from_scale_factor(a_birth, a_now, Om0, Ol0, h):
    """
    Age of the universe elapsed between scale factors a_birth and a_now for
    flat LCDM, via direct numerical integration of dt/da = 1/(a H(a)):
        t(a) = (1/H0) * integral_0^a da' / (a' sqrt(Om0/a'^3 + Ol0))
    Returns (age_of_universe_at_a_now - age_of_universe_at_a_birth) in Gyr.
    No astropy dependency -- plain numpy trapz on a fine grid per particle
    would be slow, so this is vectorized: build one shared fine grid from
    ~1e-3 to 1, integrate cumulatively once, then interpolate for each a.
    """
    H0_km_s_Mpc = 100.0 * h
    # 1 / H0 in Gyr: H0 [km/s/Mpc] -> invert -> Gyr
    Mpc_km = 3.0856775814913673e19
    Gyr_s = 3.15576e16
    inv_H0_Gyr = (Mpc_km / H0_km_s_Mpc) / Gyr_s

    a_grid = np.linspace(1e-4, 1.0, 200000)
    E = np.sqrt(Om0 / a_grid**3 + Ol0)
    integrand = 1.0 / (a_grid * E)
    # cumulative trapezoidal integral -> t(a)/inv_H0_Gyr at each grid point
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a_grid))])
    t_of_a = cum * inv_H0_Gyr

    t_birth = np.interp(np.clip(a_birth, a_grid[0], a_grid[-1]), a_grid, t_of_a)
    t_now = np.interp(np.clip(a_now, a_grid[0], a_grid[-1]), a_grid, t_of_a)
    return t_now - t_birth


def fmt_stats(vals, label, unit=""):
    if vals is None or len(vals) == 0:
        return f"  {label}: (no particles)"
    n = len(vals)
    vals = np.asarray(vals, dtype=float)
    return (f"  {label}: n={n:,}  mean={np.mean(vals):.4e}{unit}  "
            f"median={np.median(vals):.4e}{unit}  "
            f"min={np.min(vals):.4e}{unit}  max={np.max(vals):.4e}{unit}")


def text_histogram(values, bins, labels, title, width=30):
    values = np.asarray(values, dtype=float)
    hist, _ = np.histogram(values, bins=bins)
    total = len(values)
    pct = (hist / total * 100.0) if total > 0 else np.zeros_like(hist, dtype=float)
    max_count = hist.max() if hist.max() > 0 else 1
    lines = [f"  {title}:"]
    for count, p, label in zip(hist, pct, labels):
        bar_len = int(width * count / max_count)
        bar = "█" * bar_len
        lines.append(f"    {label:20s} │{bar:<{width}s}│ {count:>8,d} ({p:5.1f}%)")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="Gadget-4 output directory, e.g. ../S10_output_1024/")
    p.add_argument("--snap", type=int, default=None,
                    help="Snapshot number to analyze (default: last complete snapshot)")
    p.add_argument("--ism-radius", type=float, default=20.0,
                    help="ISM/CGM boundary in physical kpc (default: 20, matching the "
                         "convention used elsewhere in this pipeline)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("╔" + "═" * BOX_WIDTH + "╗")
    title = "COSMICGRAIN DUST SNAPSHOT SUMMARY (complete, not sampled)"
    print("║" + title.center(BOX_WIDTH) + "║")
    print("╚" + "═" * BOX_WIDTH + "╝")
    print(f"  Output dir: {output_dir}")
    print()

    ref = get_halo569_reference(output_dir, snap_num_z0=args.snap, verbose=True)
    snap_num = ref["snap_num_z0"]
    groups_dir = output_dir / f"groups_{snap_num:03d}"
    snapdir = output_dir / f"snapdir_{snap_num:03d}"
    hdr = read_snap_header(snapdir)
    halo = get_halo569(groups_dir, snap_num, ref, verbose=True)
    if halo is None:
        print(fail(f"Could not identify Halo 569 at snap {snap_num:03d} -- aborting."))
        return

    h, a, z = hdr["h"], hdr["a"], hdr["z"]
    r200_pkpc = halo["r200_pkpc"]
    r200_ckpch = halo["r200_ckpch"]
    center_ckpch = halo["center"]

    print()
    print(f"  Snapshot {snap_num:03d}  z={z:.4f}  a={a:.6f}")
    print(f"  Halo center: {center_ckpch} ckpc/h")
    print(f"  R200c: {r200_pkpc:.1f} pkpc ({r200_ckpch:.1f} ckpc/h)")
    print()

    # -------------------------------------------------------------------
    # Load ALL dust + gas within R200 (complete, not sampled)
    # -------------------------------------------------------------------
    data = load_particles_within_radius(
        snapdir, center_ckpch, r200_ckpch,
        part_types=(0, 6),
        fields_by_type={0: GAS_FIELDS_0, 6: DUST_FIELDS_6},
    )
    dust = data.get(6, {})
    gas = data.get(0, {})

    if not dust or "Coordinates" not in dust:
        box_top("RESULT")
        print(fail("No dust particles (PartType6) found within R200 at this snapshot."))
        print("  Check that dust creation is enabled and has had time to fire.")
        box_bottom()
        return

    # Physical-unit conversions
    dust_pos_ckpch = dust["Coordinates"]
    dust_r_ckpch = np.sqrt(((dust_pos_ckpch - center_ckpch[None, :]) ** 2).sum(axis=1))
    dust_r_pkpc = dust_r_ckpch * a / h

    dust_mass_msun = convert_code_mass_to_msun(dust["Masses"], h)
    grain_radius_nm = dust["GrainRadius"]
    grain_type = dust["GrainType"].astype(int) if "GrainType" in dust else None
    carbon_frac = dust.get("CarbonFraction")
    dust_temp = dust.get("DustTemperature")

    n_dust_total = len(dust_mass_msun)
    m_dust_total = float(np.sum(dust_mass_msun))

    ism_mask = dust_r_pkpc < args.ism_radius
    cgm_mask = (dust_r_pkpc >= args.ism_radius) & (dust_r_pkpc <= r200_pkpc)

    if gas and "Masses" in gas:
        gas_mass_msun = convert_code_mass_to_msun(gas["Masses"], h)
        gas_metallicity = gas.get("Metallicity")
        if gas_metallicity is not None and gas_metallicity.ndim > 1:
            gas_metallicity = gas_metallicity.sum(axis=1)
        m_gas_total = float(np.sum(gas_mass_msun))
        m_metal_total = float(np.sum(gas_mass_msun * gas_metallicity)) if gas_metallicity is not None else None
    else:
        m_gas_total = None
        m_metal_total = None

    # -------------------------------------------------------------------
    # POPULATION OVERVIEW
    # -------------------------------------------------------------------
    box_top("POPULATION OVERVIEW (within R200, complete)")
    print(f"  Total dust particles: {n_dust_total:,}")
    print(f"  Total dust mass:      {m_dust_total:.4e} Msun")
    print(f"  ISM (r < {args.ism_radius:.0f} pkpc):  {ism_mask.sum():,} particles, "
          f"{np.sum(dust_mass_msun[ism_mask]):.4e} Msun")
    print(f"  CGM ({args.ism_radius:.0f}-{r200_pkpc:.0f} pkpc): {cgm_mask.sum():,} particles, "
          f"{np.sum(dust_mass_msun[cgm_mask]):.4e} Msun")
    box_bottom()

    # -------------------------------------------------------------------
    # DUST-TO-GAS / DUST-TO-METAL RATIOS
    # -------------------------------------------------------------------
    box_top("DUST-TO-GAS / DUST-TO-METAL RATIOS (within R200)")
    if m_gas_total:
        dtg = m_dust_total / m_gas_total
        print(f"  Dust-to-gas ratio:   {dtg:.4e}  (MW: ~{MW_DTG:.4e}, "
              f"ratio to MW: {dtg / MW_DTG:.2f}x)")
        if m_metal_total and m_metal_total > 0:
            dtz = m_dust_total / m_metal_total
            flag = ok(f"{dtz:.4f} ({dtz / MW_DTZ:.2f}x solar)") if 0.05 < dtz < 0.7 else \
                   warn(f"{dtz:.4f} ({dtz / MW_DTZ:.2f}x solar) -- outside typical 0.05-0.7 range")
            print(f"  Dust-to-metal ratio: {flag}")
        else:
            print("  Dust-to-metal ratio: (no metallicity data found on gas particles)")
    else:
        print("  (no gas particles found within R200 -- cannot compute D/G, D/Z)")
    box_bottom()

    # -------------------------------------------------------------------
    # GRAIN SIZE DISTRIBUTION (complete histogram)
    # -------------------------------------------------------------------
    box_top("GRAIN SIZE DISTRIBUTION (complete, all particles)")
    print(fmt_stats(grain_radius_nm, "Grain radius", " nm"))
    size_bins = np.array([0, 10, 50, 100, 150, 200, 500, 2000])
    size_labels = ["0-10 nm", "10-50 nm", "50-100 nm", "100-150 nm",
                   "150-200 nm", "200-500 nm", "500-2000 nm"]
    print(text_histogram(grain_radius_nm, size_bins, size_labels, "Full population"))
    box_bottom()

    # -------------------------------------------------------------------
    # CARBON / SILICATE SPLIT
    # -------------------------------------------------------------------
    box_top("CARBON / SILICATE COMPOSITION")
    if grain_type is not None:
        n_sil = int(np.sum(grain_type == 0))
        n_carb = int(np.sum(grain_type == 1))
        n_other = int(n_dust_total - n_sil - n_carb)
        m_sil = float(np.sum(dust_mass_msun[grain_type == 0]))
        m_carb = float(np.sum(dust_mass_msun[grain_type == 1]))
        print(f"  Silicate (type 0): {n_sil:,} particles ({100*n_sil/n_dust_total:.1f}%), "
              f"{m_sil:.4e} Msun ({100*m_sil/m_dust_total:.1f}% of mass)")
        print(f"  Carbon   (type 1): {n_carb:,} particles ({100*n_carb/n_dust_total:.1f}%), "
              f"{m_carb:.4e} Msun ({100*m_carb/m_dust_total:.1f}% of mass)")
        if n_other > 0:
            print(warn(f"  {n_other:,} particle(s) with GrainType not in {{0,1}} "
                        f"(placeholder value 2 never overwritten? worth checking)"))

        # ISM vs CGM carbon/silicate split
        if ism_mask.any() and cgm_mask.any():
            m_carb_ism = float(np.sum(dust_mass_msun[ism_mask & (grain_type == 1)]))
            m_carb_cgm = float(np.sum(dust_mass_msun[cgm_mask & (grain_type == 1)]))
            m_sil_ism = float(np.sum(dust_mass_msun[ism_mask & (grain_type == 0)]))
            m_sil_cgm = float(np.sum(dust_mass_msun[cgm_mask & (grain_type == 0)]))
            print()
            print("  Environment split (mass):")
            if (m_carb_ism + m_carb_cgm) > 0:
                print(f"    Carbon:   ISM {100*m_carb_ism/(m_carb_ism+m_carb_cgm):.1f}%  "
                      f"CGM {100*m_carb_cgm/(m_carb_ism+m_carb_cgm):.1f}%")
            if (m_sil_ism + m_sil_cgm) > 0:
                print(f"    Silicate: ISM {100*m_sil_ism/(m_sil_ism+m_sil_cgm):.1f}%  "
                      f"CGM {100*m_sil_cgm/(m_sil_ism+m_sil_cgm):.1f}%")
    else:
        print("  (GrainType field not found)")
    if carbon_frac is not None:
        print()
        print(fmt_stats(carbon_frac, "CarbonFraction (per-particle)"))
    box_bottom()

    # -------------------------------------------------------------------
    # DUST TEMPERATURE
    # -------------------------------------------------------------------
    box_top("DUST TEMPERATURE (complete, all particles)")
    if dust_temp is not None:
        print(fmt_stats(dust_temp, "T_dust", " K"))
        if np.mean(dust_temp) > 10000:
            print(warn("Mean T_dust > 10,000 K -- suspiciously hot for dust (check units/coupling)"))
        temp_bins = [0, 5, 10, 20, 30, 50, 100, 200, 1e5]
        temp_labels = ["<5K (CMB floor)", "5-10K", "10-20K", "20-30K",
                       "30-50K", "50-100K", "100-200K", ">200K"]
        print(text_histogram(dust_temp, temp_bins, temp_labels, "Temperature"))
    else:
        print("  (DustTemperature field not found)")
    box_bottom()

    # -------------------------------------------------------------------
    # GRAIN AGE (from DustFormationTime)
    # -------------------------------------------------------------------
    box_top("GRAIN AGE (complete, all particles)")
    a_birth = dust.get("DustFormationTime")
    if a_birth is not None and len(a_birth) == n_dust_total:
        age_gyr = age_gyr_from_scale_factor(a_birth, a, hdr["Omega0"], hdr["OmegaLambda"], h)
        age_gyr = np.clip(age_gyr, 0, None)
        print(fmt_stats(age_gyr, "Grain age", " Gyr"))
        age_bins = [0, 1, 3, 5, 8, 10, 14]
        age_labels = ["<1 Gyr", "1-3 Gyr", "3-5 Gyr", "5-8 Gyr", "8-10 Gyr", "10-14 Gyr"]
        print(text_histogram(age_gyr, age_bins, age_labels, "Age distribution"))
    else:
        print("  (DustFormationTime field not found on this snapshot -- older runs may "
              "predate this field being added to snap_io.cc)")
    box_bottom()

    # -------------------------------------------------------------------
    # VELOCITY
    # -------------------------------------------------------------------
    box_top("GRAIN VELOCITY (complete, all particles)")
    if "Velocities" in dust:
        vel = dust["Velocities"]
        speed_kms = np.sqrt((vel ** 2).sum(axis=1))  # already km/s in Gadget-4 velocity convention
        print(fmt_stats(speed_kms, "Speed", " km/s"))
    else:
        print("  (Velocities field not found)")
    box_bottom()

    # -------------------------------------------------------------------
    # OVERALL ASSESSMENT
    # -------------------------------------------------------------------
    print("╔" + "═" * BOX_WIDTH + "╗")
    print("║" + "OVERALL ASSESSMENT".center(BOX_WIDTH) + "║")
    print("╚" + "═" * BOX_WIDTH + "╝")
    print()
    successes, issues = [], []

    successes.append(f"{n_dust_total:,} dust particles within R200 at z={z:.3f}, "
                      f"{m_dust_total:.3e} Msun total")

    if m_gas_total and m_metal_total:
        dtz = m_dust_total / m_metal_total
        if 0.05 <= dtz <= 0.7:
            successes.append(f"Dust-to-metal ratio ({dtz:.3f}) within typical observed range")
        else:
            issues.append(f"Dust-to-metal ratio ({dtz:.3f}) outside typical 0.05-0.7 range")

    if grain_type is not None:
        n_other = int(n_dust_total - int(np.sum(grain_type == 0)) - int(np.sum(grain_type == 1)))
        if n_other > 0:
            issues.append(f"{n_other:,} particles have an unexpected GrainType value")
        else:
            successes.append("All particles have a valid GrainType (0=silicate, 1=carbon)")

    if dust_temp is not None and np.mean(dust_temp) > 10000:
        issues.append("Mean dust temperature is implausibly high")

    if successes:
        print(f"{Colors.OKGREEN}SUCCESSES:{Colors.ENDC}")
        for s in successes:
            print(f"  • {s}")
        print()
    if issues:
        print(f"{Colors.WARNING}ISSUES TO REVIEW:{Colors.ENDC}")
        for i in issues:
            print(f"  • {i}")
        print()

    print("─" * (BOX_WIDTH + 2))
    print(f"Rerun with: python3 {sys.argv[0]} {args.output_dir}"
          + (f" --snap {args.snap}" if args.snap else ""))
    print()


if __name__ == "__main__":
    main()
