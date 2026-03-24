#!/usr/bin/env python3
"""
compare_dust_grid.py
--------------------
Reads the z=0 (or latest available) snapshot from each simulation in the
S0–S7 (or S0–S10) physics ladder and computes the key statistics needed for
the deluxetable in the CosmicGrain paper.

Metrics (all measured in the ISM: n_H > ISM_DENSITY_THRESH cm⁻³, within
the main halo virial radius):
  - M_star / M_halo             stellar-to-halo mass ratio
  - M_dust  [M_sun]             total dust mass in ISM
  - <a>     [nm]                mass-weighted mean grain radius
  - f_surv                      surviving dust fraction  = N_dust / N_spawned
                                 (approximated as M_dust / M_dust_created_cumul
                                  if creation log is available; else we
                                  estimate from the dust age distribution)
  - D/G                         dust-to-gas mass ratio  (ISM gas)
  - D/Z                         dust-to-metals mass ratio (ISM gas metals)

Usage
-----
python compare_dust_grid.py  <sim_root_dir>  [--sims S0 S1 S2 ...]
                                              [--res  512]
                                              [--snap LAST|<number>]
                                              [--halo-id 569]
                                              [--latex-out table_512.tex]

Example
-------
python compare_dust_grid.py  /scratch/gadget4/runs/  \\
    --sims S0 S1 S2 S3 S4 S5 S6 S7  \\
    --res  512  \\
    --halo-id 569  \\
    --latex-out dust_sim_ladder_512.tex

The script expects simulation output directories named like:
  <sim_root_dir>/<sim_name>_<res>/   e.g.  .../S3_512/
  inside which there are snapdir_NNN/ subdirectories.

It uses SubFind group catalogs (if present) to extract halo 569 properties;
if catalogs are absent it falls back to a sphere of radius R_200 around the
centre of mass of the stellar component.
"""

import argparse
import glob
import os
import sys
import numpy as np
import h5py

# ── physical constants / unit conversions ────────────────────────────────────
MSUN_IN_G      = 1.989e33
H_MASS_G       = 1.6726e-24
PROTON_MASS_G  = 1.6726e-24
KPC_IN_CM      = 3.086e21
NM_PER_CM      = 1e7

# ── defaults ──────────────────────────────────────────────────────────────────
ISM_DENSITY_THRESH = 0.1    # cm^-3   (n_H threshold for "ISM")
DEFAULT_HALO_RAD   = 300.0  # kpc/h   fallback aperture if no subfind


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(sim_dir):
    """Return sorted list of snapdir paths inside sim_dir."""
    snaps = sorted(glob.glob(os.path.join(sim_dir, "snapdir_*")))
    return snaps


def latest_snapshot(sim_dir):
    snaps = find_snapshots(sim_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapdir_* found in {sim_dir}")
    return snaps[-1]


def open_snap_files(snapdir):
    """Return list of all .hdf5 files in snapdir, sorted."""
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {snapdir}")
    return files


def read_header(snapdir):
    """Read header attrs from first file of snapshot."""
    f0 = open_snap_files(snapdir)[0]
    with h5py.File(f0, "r") as f:
        h = dict(f["Header"].attrs)
    return h


def read_part_concat(snapdir, part_type, fields):
    """
    Read and concatenate `fields` from PartType<part_type> across all
    snapshot files in snapdir.  Returns dict {field: array}.
    Missing fields are silently skipped (entry not in returned dict).
    """
    data = {fi: [] for fi in fields}
    snap_files = open_snap_files(snapdir)

    for fname in snap_files:
        with h5py.File(fname, "r") as f:
            key = f"PartType{part_type}"
            if key not in f:
                continue
            grp = f[key]
            for fi in fields:
                if fi in grp:
                    data[fi].append(grp[fi][:])

    out = {}
    for fi in fields:
        if data[fi]:
            out[fi] = np.concatenate(data[fi], axis=0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Halo identification
# ─────────────────────────────────────────────────────────────────────────────

def load_subfind_halo(snapdir, halo_id=0):
    """
    Try to load halo properties (M200, R200, centre) from SubFind group
    catalogs living alongside the snapshot.  Returns dict or None.
    """
    snap_root = os.path.dirname(snapdir)      # parent of snapdir_NNN
    # Typical subfind catalog pattern
    cat_patterns = [
        os.path.join(snap_root, "groups_*", "fof_subhalo_tab_*.0.hdf5"),
        os.path.join(snap_root, "groups_*", "*.0.hdf5"),
    ]
    cat_files = []
    for pat in cat_patterns:
        cat_files.extend(sorted(glob.glob(pat)))

    if not cat_files:
        return None

    # Use the catalog that matches the snapshot number
    snap_num = int(os.path.basename(snapdir).split("_")[-1])
    matched = [c for c in cat_files if f"{snap_num:03d}" in c]
    if not matched:
        matched = cat_files  # fall back to latest

    cat0 = matched[-1]
    try:
        with h5py.File(cat0, "r") as f:
            if "Group" not in f:
                return None
            grp  = f["Group"]
            m200 = grp["Group_M_Mean200"][:] if "Group_M_Mean200" in grp else grp["GroupMass"][:]
            r200 = grp["Group_R_Mean200"][:] if "Group_R_Mean200" in grp else None
            pos  = grp["GroupPos"][:]
            idx  = min(halo_id, len(m200) - 1)
            result = {
                "M200":   float(m200[idx]),
                "centre": pos[idx].copy(),
            }
            if r200 is not None:
                result["R200"] = float(r200[idx])
            return result
    except Exception as e:
        print(f"  [warn] subfind load failed ({e}), using fallback")
        return None


def estimate_halo_centre(snapdir, header):
    """
    Fallback: use stellar centre of mass as halo centre.
    Returns (centre_kpc_h, R200_kpc_h_estimate).
    """
    stars = read_part_concat(snapdir, 4, ["Coordinates", "Masses"])
    if "Coordinates" not in stars or len(stars["Coordinates"]) == 0:
        # No stars – use gas centre
        gas = read_part_concat(snapdir, 0, ["Coordinates", "Masses"])
        pos = gas.get("Coordinates", np.zeros((1,3)))
        mass = gas.get("Masses", np.ones(len(pos)))
    else:
        pos  = stars["Coordinates"]
        mass = stars.get("Masses", np.ones(len(pos)))

    # mass-weighted centre
    cen = np.average(pos, weights=mass, axis=0)
    # Rough R200 estimate from stellar mass
    M_star = np.sum(mass) * header["UnitMass_in_g"] / MSUN_IN_G
    # Abundance matching: log(M_halo) ~ log(M_star) + 1.5  (very rough)
    M_halo_est = M_star * 30.0
    rho_200 = 200 * 2.775e11 * header.get("Omega0", 0.3) * (header["HubbleParam"] ** 2)
    R200_Mpc = (3 * M_halo_est / (4 * np.pi * rho_200)) ** (1/3)
    R200_kpc_h = R200_Mpc * 1000 * header["HubbleParam"]
    return cen, R200_kpc_h


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversions
# ─────────────────────────────────────────────────────────────────────────────

def get_units(header):
    """Return (UL_cm, UM_g, scale_factor, h) from header."""
    UL = header.get("UnitLength_in_cm", 3.086e21)   # default: kpc/h
    UM = header.get("UnitMass_in_g",    1.989e43)   # default: 1e10 Msun/h
    a  = header["Time"]
    h  = header["HubbleParam"]
    return UL, UM, a, h


def density_to_nH(rho_code, UL, UM, a, h, XH=0.76):
    """Convert Gadget comoving density to physical n_H [cm^-3]."""
    rho_phys_cgs = rho_code * UM / (UL * a / h) ** 3   # g/cm^3
    nH = rho_phys_cgs * XH / PROTON_MASS_G
    return nH


# ─────────────────────────────────────────────────────────────────────────────
# Per-simulation statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sim_stats(sim_dir, snap_spec, halo_id, verbose=True):
    """
    Main workhorse.  Returns a dict of statistics for one simulation.
    """
    stats = {}

    # ── find snapshot ──────────────────────────────────────────────────────
    if snap_spec == "LAST":
        snapdir = latest_snapshot(sim_dir)
    else:
        snapdir = os.path.join(sim_dir, f"snapdir_{int(snap_spec):03d}")
        if not os.path.isdir(snapdir):
            raise FileNotFoundError(f"Snapshot dir not found: {snapdir}")

    if verbose:
        print(f"\n  snapshot: {snapdir}")

    header = read_header(snapdir)
    UL, UM, a, h  = get_units(header)
    redshift = header["Redshift"]
    stats["redshift"] = redshift

    if verbose:
        print(f"  z = {redshift:.4f},  a = {a:.4f}")

    # Msun per code-mass unit
    MSUN_per_code = UM / MSUN_IN_G

    # ── halo centre / R200 ─────────────────────────────────────────────────
    halo_info = load_subfind_halo(snapdir, halo_id)
    if halo_info:
        centre = halo_info["centre"]           # comoving kpc/h
        M200   = halo_info["M200"] * MSUN_per_code
        R200   = halo_info.get("R200", DEFAULT_HALO_RAD)  # comoving kpc/h
        if verbose:
            print(f"  SubFind: M200={M200:.3e} Msun, R200={R200:.1f} ckpc/h")
    else:
        centre, R200 = estimate_halo_centre(snapdir, header)
        M200 = None
        if verbose:
            print(f"  Fallback centre: {centre},  R200_est={R200:.1f} ckpc/h")

    stats["M_halo"] = M200   # may be None for S0 (no subfind dust)

    def in_halo(pos_arr):
        """Boolean mask: within 1.5×R200 of centre."""
        dr = pos_arr - centre[np.newaxis, :]
        r  = np.sqrt(np.sum(dr**2, axis=1))
        return r < 1.5 * R200

    # ── stellar mass ───────────────────────────────────────────────────────
    stars = read_part_concat(snapdir, 4, ["Coordinates", "Masses"])
    if "Coordinates" in stars and len(stars["Coordinates"]) > 0:
        mask_s  = in_halo(stars["Coordinates"])
        M_star  = np.sum(stars["Masses"][mask_s]) * MSUN_per_code
    else:
        M_star = 0.0
    stats["M_star"] = M_star

    if M200 is not None and M200 > 0:
        stats["Mstar_over_Mhalo"] = M_star / M200
    else:
        stats["Mstar_over_Mhalo"] = None

    # ── gas (ISM subset) ───────────────────────────────────────────────────
    gas_fields = ["Coordinates", "Masses", "Density", "Metallicity"]
    gas = read_part_concat(snapdir, 0, gas_fields)

    has_gas = "Coordinates" in gas and len(gas["Coordinates"]) > 0

    if has_gas:
        mask_g_halo = in_halo(gas["Coordinates"])
        rho_code    = gas["Density"][mask_g_halo]
        nH          = density_to_nH(rho_code, UL, UM, a, h)
        mask_ism    = nH > ISM_DENSITY_THRESH        # ISM cut
        # ISM gas
        M_gas_ism   = np.sum(gas["Masses"][mask_g_halo][mask_ism]) * MSUN_per_code
        # ISM metals
        if "Metallicity" in gas:
            met = gas["Metallicity"][mask_g_halo]
            if met.ndim == 2:
                met = met[:, 0]    # total metals
            M_metal_ism = np.sum(met[mask_ism] * gas["Masses"][mask_g_halo][mask_ism]) * MSUN_per_code
        else:
            M_metal_ism = 0.0
    else:
        M_gas_ism   = 0.0
        M_metal_ism = 0.0

    stats["M_gas_ism"]   = M_gas_ism
    stats["M_metal_ism"] = M_metal_ism

    # ── dust (PartType6) ───────────────────────────────────────────────────
    dust_fields = ["Coordinates", "Masses", "DustGrainRadius"]
    dust = read_part_concat(snapdir, 6, dust_fields)

    has_dust = "Coordinates" in dust and len(dust["Coordinates"]) > 0

    if has_dust:
        mask_d_halo = in_halo(dust["Coordinates"])
        M_dust_all  = dust["Masses"][mask_d_halo]
        M_dust      = np.sum(M_dust_all) * MSUN_per_code

        if "DustGrainRadius" in dust:
            # stored in nm (as per our refactor)
            radii_nm  = dust["DustGrainRadius"][mask_d_halo]
            mean_a_nm = np.average(radii_nm, weights=M_dust_all)
        else:
            mean_a_nm = np.nan

        N_dust_halo = int(np.sum(mask_d_halo))
    else:
        M_dust      = 0.0
        mean_a_nm   = np.nan
        N_dust_halo = 0

    stats["M_dust"]     = M_dust
    stats["mean_a_nm"]  = mean_a_nm
    stats["N_dust"]     = N_dust_halo

    # ── derived ratios ─────────────────────────────────────────────────────
    stats["DtoG"] = M_dust / M_gas_ism  if M_gas_ism   > 0 else 0.0
    stats["DtoZ"] = M_dust / M_metal_ism if M_metal_ism > 0 else 0.0

    # ── survival fraction (rough) ──────────────────────────────────────────
    # We don't have cumulative creation logged in the snapshot, so we
    # compute it from the age distribution:
    #   f_surv ~ N_alive / N_ever_created
    # N_ever_created is estimated from stellar particle count and yields,
    # OR we just report NaN and let the user fill from the sim logs.
    stats["f_surv"] = np.nan   # fill from log parser below

    if verbose:
        print(f"  M_star      = {M_star:.3e} Msun")
        print(f"  M_dust      = {M_dust:.3e} Msun")
        print(f"  M_gas(ISM)  = {M_gas_ism:.3e} Msun")
        print(f"  D/G         = {stats['DtoG']:.3e}")
        print(f"  D/Z         = {stats['DtoZ']:.4f}")
        print(f"  <a>         = {mean_a_nm:.2f} nm")
        print(f"  N_dust(halo)= {N_dust_halo}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Parse dust statistics from Gadget log files (for f_surv + event counts)
# ─────────────────────────────────────────────────────────────────────────────

def parse_dust_log(log_path):
    """
    Scan the Gadget output log for DUST STATISTICS lines and return the
    values from the LAST occurrence (i.e. z~0).

    Looks for lines like:
      [DUST|...|z=0.001] STATISTICS Particles: 95423  Mass: ...
      [DUST|...|z=0.001] STATISTICS Destroyed by thermal: 12345
      ...
    """
    result = {}
    if not log_path or not os.path.isfile(log_path):
        return result

    # We'll accumulate blocks keyed by the last-seen redshift header
    current_z   = None
    best_block  = {}   # block at smallest z

    with open(log_path, "r") as fh:
        for line in fh:
            # Extract redshift tag
            if "STATISTICS" not in line and "DUST STATISTICS" not in line:
                continue
            # Parse z from tag like [DUST|T=0|a=0.999|z=0.001]
            import re
            zm = re.search(r'z=([\d.]+)', line)
            if zm:
                z_now = float(zm.group(1))
                if current_z is None or z_now < current_z:
                    current_z  = z_now
                    best_block = {}

            line = line.lower()
            def _val(key):
                m = re.search(key + r'[:\s]+([\d.e+\-]+)', line)
                return float(m.group(1)) if m else None

            for key, tag in [("particles",        "n_particles"),
                              ("destroyed by thermal", "destroyed_thermal"),
                              ("destroyed by shocks",  "destroyed_shock"),
                              ("destroyed by astration","destroyed_astration"),
                              ("growth events",         "growth_events"),
                              ("avg grain size",        "mean_a_nm_log")]:
                v = _val(key)
                if v is not None:
                    best_block[tag] = v

    return best_block


def estimate_f_surv_from_log(log_stats, stats):
    """
    Rough survival fraction:
      f_surv = N_alive / (N_alive + N_destroyed_total)
    """
    if not log_stats:
        return np.nan
    n_alive = log_stats.get("n_particles", stats.get("N_dust", np.nan))
    n_th    = log_stats.get("destroyed_thermal",    0)
    n_sh    = log_stats.get("destroyed_shock",      0)
    n_ast   = log_stats.get("destroyed_astration",  0)
    n_total = n_alive + n_th + n_sh + n_ast
    if n_total == 0:
        return np.nan
    return n_alive / n_total


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX output
# ─────────────────────────────────────────────────────────────────────────────

# Physics flags per sim (S0–S7), adjust if your ladder is different
PHYSICS_FLAGS = {
    "S0":  dict(form=0, drag=0, growth=0, clump=0, sput=0, shock=0, astrat=0),
    "S1":  dict(form=1, drag=1, growth=0, clump=0, sput=0, shock=0, astrat=0),
    "S2":  dict(form=1, drag=1, growth=1, clump=0, sput=0, shock=0, astrat=0),
    "S3":  dict(form=1, drag=1, growth=1, clump=1, sput=0, shock=0, astrat=0),
    "S4":  dict(form=1, drag=1, growth=1, clump=1, sput=0, shock=0, astrat=0),  # coag/shatt
    "S5":  dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=0, astrat=0),
    "S6":  dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=1, astrat=0),
    "S7":  dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=1, astrat=1),
    "S8":  dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=1, astrat=1),
    "S9":  dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=1, astrat=1),
    "S10": dict(form=1, drag=1, growth=1, clump=1, sput=1, shock=1, astrat=1),
}

NOTES = {
    "S0":  "Baseline (no dust)",
    "S1":  "Add stellar sources + drag",
    "S2":  "Add growth (uniform $n$)",
    "S3":  "Add ISM density clumping",
    "S4":  "Add coagulation/shattering",
    "S5":  "Add thermal sputtering",
    "S6":  "Add SN shock destruction",
    "S7":  "Add astration (full model)",
    "S8":  "Add rad.~pressure",
    "S9":  "Add dust cooling",
    "S10": "Full model + rad.~pressure + cool.",
}

CM  = r"\cmark"
XM  = r"\xmark"

def flag(v):
    return CM if v else XM

def fmt_ratio(val, digits=3):
    if val is None or np.isnan(val):
        return r"\ldots"
    if val == 0.0:
        return "0.000"
    exp  = int(np.floor(np.log10(abs(val))))
    mant = val / 10**exp
    if -2 <= exp <= -1:
        return f"{val:.4f}"
    return f"${mant:.2f}\\times10^{{{exp}}}$"

def fmt_float(val, fmt=".3f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return r"\ldots"
    return f"{val:{fmt}}"

def fmt_dz(val):
    if val is None or np.isnan(val):
        return r"\ldots"
    return f"{val:.2f}"


def write_latex_table(sim_names, all_stats, resolution, out_path):
    lines = []
    lines.append(r"% Auto-generated by compare_dust_grid.py")
    lines.append(r"% Resolution: " + f"{resolution}^3")
    lines.append(r"\begin{deluxetable*}{lcccccccrccccc}")
    lines.append(r"\tablecaption{CosmicGrain simulation runs at $" + str(resolution) +
                 r"^3$ resolution, incrementing in dust-physics complexity"
                 r"\label{tab:dust_sim_ladder_" + str(resolution) + r"}}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\tablewidth{0pt}")
    lines.append(r"\tablehead{")
    lines.append(r"  \colhead{Sim} &")
    lines.append(r"  \colhead{\shortstack{Dust\\form.}} &")
    lines.append(r"  \colhead{\shortstack{Dust\\drag}} &")
    lines.append(r"  \colhead{\shortstack{Dust\\growth}} &")
    lines.append(r"  \colhead{\shortstack{ISM\\clump.}} &")
    lines.append(r"  \colhead{\shortstack{Therm.\\sput.}} &")
    lines.append(r"  \colhead{\shortstack{SN\\shocks}} &")
    lines.append(r"  \colhead{\shortstack{Astr-\\ation}} &")
    lines.append(r"  \colhead{$M_\star/M_{\rm halo}$} &")
    lines.append(r"  \colhead{$\langle a\rangle$\,(nm)} &")
    lines.append(r"  \colhead{$f_{\rm surv}$} &")
    lines.append(r"  \colhead{$D/G$} &")
    lines.append(r"  \colhead{$D/Z$} &")
    lines.append(r"  \colhead{Notes}")
    lines.append(r"}")
    lines.append(r"\startdata")

    for sim in sim_names:
        s = all_stats.get(sim, {})
        pf = PHYSICS_FLAGS.get(sim, {})

        msmh   = fmt_float(s.get("Mstar_over_Mhalo"), ".3f")
        mean_a = fmt_float(s.get("mean_a_nm"),        ".1f")
        fsurv  = fmt_dz(s.get("f_surv"))
        dtog   = fmt_ratio(s.get("DtoG"))
        dtoz   = fmt_dz(s.get("DtoZ"))
        note   = NOTES.get(sim, "")

        row = (f"  {sim} & "
               f"{flag(pf.get('form',0))} & "
               f"{flag(pf.get('drag',0))} & "
               f"{flag(pf.get('growth',0))} & "
               f"{flag(pf.get('clump',0))} & "
               f"{flag(pf.get('sput',0))} & "
               f"{flag(pf.get('shock',0))} & "
               f"{flag(pf.get('astrat',0))} & "
               f"{msmh} & "
               f"{mean_a} & "
               f"{fsurv} & "
               f"{dtog} & "
               f"{dtoz} & "
               r"\tiny{" + note + r"} \\")
        lines.append(row)

    lines.append(r"\enddata")
    lines.append(r"\tablecomments{%")
    lines.append(r"  Each simulation builds on the previous by activating one additional")
    lines.append(r"  dust-physics process; all runs use the same $" + str(resolution) +
                 r"^3$ resolution and identical initial conditions (50~Mpc box, halo 569,")
    lines.append(r"  $M_{200}\approx2\times10^{12}\,h^{-1}\,M_\odot$).")
    lines.append(r"  All simulations include metal-line cooling.")
    lines.append(r"  S0 establishes baseline galaxy properties with no dust physics.")
    lines.append(r"  S1 introduces stellar dust sources (SNe~II + AGB) and gas--dust drag.")
    lines.append(r"  S2 adds grain growth via ISM accretion assuming uniform gas density.")
    lines.append(r"  S3 includes subgrid density clumping ($C=1$--30) for unresolved")
    lines.append(r"  molecular clouds. S4 adds coagulation and shattering.")
    lines.append(r"  S5--S7 add destruction mechanisms sequentially:")
    lines.append(r"  thermal sputtering (S5), SN shock destruction (S6), and astration (S7).")
    lines.append(r"  $\langle a\rangle$ is the mass-weighted mean grain radius;")
    lines.append(r"  $f_{\rm surv}$ is the fraction of ever-created dust particles")
    lines.append(r"  still alive at $z=0$; $D/G$ and $D/Z$ are measured in the ISM")
    lines.append(r"  ($n_{\rm H}>0.1\,{\rm cm}^{-3}$) within $1.5\,R_{200}$ at $z=0$.}")
    lines.append(r"\end{deluxetable*}")
    lines.append("")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))

    print(f"\nLaTeX table written to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_root",   help="Root directory containing sim sub-dirs")
    parser.add_argument("--sims",     nargs="+",
                        default=["S0","S1","S2","S3","S4","S5","S6","S7"],
                        help="Simulation labels to process")
    parser.add_argument("--res",      type=int, default=512,
                        help="Resolution (e.g. 512, 1024)")
    parser.add_argument("--snap",     default="LAST",
                        help="Snapshot number or LAST")
    parser.add_argument("--halo-id",  type=int, default=0,
                        help="SubFind group index (0 = most massive)")
    parser.add_argument("--log-pattern", default=None,
                        help="Glob pattern for Gadget stdout logs, "
                             "e.g. '../runs/S{sim}_{res}/output.log'")
    parser.add_argument("--latex-out", default="dust_sim_ladder_{res}.tex",
                        help="Output LaTeX file path")
    args = parser.parse_args()

    res      = args.res
    out_path = args.latex_out.replace("{res}", str(res))

    print(f"CosmicGrain grid stats  |  resolution: {res}^3")
    print(f"Root: {args.sim_root}")
    print(f"Sims: {args.sims}")
    print(f"Output: {out_path}\n")

    all_stats = {}

    for sim in args.sims:
        # Convention: <root>/<sim>_<res>/  e.g.  .../S3_512/
        # Try a few naming patterns
        candidates = [
            os.path.join(args.sim_root, f"{sim}_{res}"),
            os.path.join(args.sim_root, f"{sim.lower()}_{res}"),
            os.path.join(args.sim_root, sim),
            os.path.join(args.sim_root, f"{sim}_{res}_halo569"),
        ]
        sim_dir = None
        for c in candidates:
            if os.path.isdir(c):
                sim_dir = c
                break

        if sim_dir is None:
            print(f"\n[SKIP] {sim}: no directory found (tried {candidates[0]} etc.)")
            all_stats[sim] = {}
            continue

        print(f"\n{'='*60}")
        print(f"  Processing {sim}  →  {sim_dir}")

        try:
            stats = compute_sim_stats(sim_dir, args.snap, args.halo_id, verbose=True)

            # Optionally parse log for f_surv and grain-size cross-check
            if args.log_pattern:
                log_path = args.log_pattern.format(sim=sim, res=res)
                log_stats = parse_dust_log(log_path)
                stats["f_surv"] = estimate_f_surv_from_log(log_stats, stats)
                if log_stats.get("mean_a_nm_log") and np.isnan(stats.get("mean_a_nm", np.nan)):
                    stats["mean_a_nm"] = log_stats["mean_a_nm_log"]

            all_stats[sim] = stats

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            all_stats[sim] = {}

    # ── print summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Sim':<5} {'M*/Mh':>8} {'<a>/nm':>8} {'f_surv':>8} {'D/G':>12} {'D/Z':>8}")
    print(f"{'-'*70}")
    for sim in args.sims:
        s = all_stats.get(sim, {})
        print(f"{sim:<5} "
              f"{s.get('Mstar_over_Mhalo', float('nan')):>8.4f} "
              f"{s.get('mean_a_nm',         float('nan')):>8.2f} "
              f"{s.get('f_surv',            float('nan')):>8.3f} "
              f"{s.get('DtoG',              float('nan')):>12.3e} "
              f"{s.get('DtoZ',              float('nan')):>8.4f}")

    # ── write LaTeX ────────────────────────────────────────────────────────
    write_latex_table(args.sims, all_stats, res, out_path)


if __name__ == "__main__":
    main()
