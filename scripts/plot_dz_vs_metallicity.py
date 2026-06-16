#!/usr/bin/env python3
"""
plot_dz_vs_metallicity.py
--------------------------
Galaxy-integrated D/Z vs. gas-phase metallicity for the CosmicGrain
simulation ladder, compared to observations and other simulations.

Run from anywhere — paths are anchored to the parent of this script file:

    cd ~/gadget4/scripts
    python plot_dz_vs_metallicity.py --res 1024
    python plot_dz_vs_metallicity.py --res 512 1024 2048

TWO MODES (auto-detected from --res):

  Physics ladder  (--res 1024)
      One resolution, multiple runs (S0–S10). Each run is one point,
      colored and labelled by physics step from RUN_CONFIGS.

  Convergence  (--res 512 1024 2048)
      Multiple resolutions, one or more runs. Each (run, resolution)
      pair is one point, styled by resolution from RES_CONFIGS.
      Default run for this mode: S10.

NOTE ON METALLICITY AXES
------------------------
The simulation Z_gas is gas-phase metallicity only (dust tracked separately
as PartType6 — it does not contribute to the Metallicity field of PartType0).
Observational references (RR14, DustPedia) use strong-line oxygen abundances
which approximate TOTAL metallicity (gas + dust).

The Milky Way reference point is therefore placed at the GAS-PHASE metallicity:
  Z_gas_MW = Z_total * (1 - D/Z) ≈ 0.0134 * 0.60 = 0.008
  D/Z_MW   = 0.40  (Jenkins 2009; Draine et al. 2007)
This is more self-consistent with the simulation x-axis than using Z_solar.
The observational trend lines are plotted as-is (strong-line ~ total Z) and
the small systematic offset (~0.4 dex in x) is noted in the caption.

CONVERGENCE MODE — FLAGGING BAD RUNS
-------------------------------------
Use --unphysical res:run to mark specific (resolution, run) pairs as
numerically compromised (e.g. gas-expelled ISM) without dropping them
from the plot entirely. They appear as open markers with an × overlay
and are labelled "(unphysical)" in the legend.

Example:
    python plot_dz_vs_metallicity.py --res 512 1024 2048 \\
        --unphysical 2048:S10

REFERENCES
----------
Observations:
  Rémy-Ruyer+2014 (A&A 563, A31)  — 126 DGS+KINGFISH galaxies, BPL fit
  De Vis+2019 (A&A 623, A5)       — DustPedia+RR14 compilation
  Jenkins 2009 (ApJ 700, 1299)    — MW ISM depletions
  Draine et al. 2007 (ApJ 663, 866) — MW dust budget

Simulations:
  McKinnon+2017 (MNRAS 468, 1505) — AREPO subgrid ISM
  Aoyama+2018  (MNRAS 478, 4905)  — Gadget two-size
  Li+2019      (MNRAS 490, 1425)  — SIMBA/GIZMO
"""

import os
import re
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path anchoring — works regardless of cwd
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

FIGDIR = BASE_DIR / "dust_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

_mplstyle = SCRIPT_DIR / "sleek.mplstyle"
if _mplstyle.exists():
    plt.style.use(str(_mplstyle))

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
RUN_CONFIGS = {
    "S0":  {"label": "S0: Creation only",           "color": "#888888", "marker": "o", "ms": 80},
    "S1":  {"label": "S1: + Cooling",               "color": "#1f77b4", "marker": "o", "ms": 80},
    "S2":  {"label": "S2: + Drag",                  "color": "#ff7f0e", "marker": "o", "ms": 80},
    "S3":  {"label": "S3: + Astration",             "color": "#2ca02c", "marker": "o", "ms": 80},
    "S4":  {"label": "S4: + Thermal sputtering",    "color": "#d62728", "marker": "o", "ms": 80},
    "S5":  {"label": "S5: + Grain growth",          "color": "#9467bd", "marker": "o", "ms": 80},
    "S6":  {"label": "S6: + Clumping factor",       "color": "#8c564b", "marker": "o", "ms": 80},
    "S7":  {"label": "S7: + SN shock destruction",  "color": "#e377c2", "marker": "o", "ms": 80},
    "S8":  {"label": "S8: + Coagulation",           "color": "#17becf", "marker": "o", "ms": 80},
    "S9":  {"label": "S9: + Shattering",            "color": "#bcbd22", "marker": "o", "ms": 80},
    "S10": {"label": "S10: + Rad. pressure (full)", "color": "#2a9d8f", "marker": "*", "ms": 320},
}

RES_CONFIGS = {
    512:  {"label": r"$512^3$",  "color": "#7ec8c0", "marker": "o", "ms": 100},
    1024: {"label": r"$1024^3$", "color": "#2a9d8f", "marker": "*", "ms": 320},
    2048: {"label": r"$2048^3$", "color": "#1a6b62", "marker": "D", "ms": 100},
    4096: {"label": r"$4096^3$", "color": "#0d3d38", "marker": "s", "ms": 100},
}

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
Z_SOLAR  = 0.0134      # Asplund+2009
OH_SOLAR = 8.69
X_H      = 0.76
M_H_G    = 1.6736e-24  # proton mass [g]

# MW reference: gas-phase Z = Z_total * (1 - D/Z) ≈ 0.0134 * 0.60 = 0.008
# D/Z from Jenkins 2009 (ISM depletions) + Draine et al. 2007 (dust budget)
MW_Z_GAS = 0.0134 * (1.0 - 0.40)   # ≈ 0.008, i.e. ~0.6 Z_sun gas-phase
MW_DZ    = 0.40

# Depletion fraction: fraction of metals locked in dust at MW conditions.
# Used only to position the MW reference point on the gas-phase x-axis.
# The observational trend lines (RR14, DustPedia) are based on strong-line
# oxygen abundances ≈ total Z, so they are shifted ~0.2 dex right of where
# a self-consistent gas-phase axis would place them.  The MW point accounts
# for this by using Z_gas rather than Z_total.

# ─────────────────────────────────────────────────────────────────────────────
# Literature reference data
# ─────────────────────────────────────────────────────────────────────────────
RR14_LOGDG0   = -2.21
RR14_ALPHA_H  =  1.0
RR14_ALPHA_L  =  3.15
RR14_OH_BREAK =  8.0

def rr14_dz(Z_mass_frac):
    Z   = np.atleast_1d(np.asarray(Z_mass_frac, float))
    oh  = OH_SOLAR + np.log10(np.clip(Z / Z_SOLAR, 1e-8, None))
    doh = oh - OH_SOLAR
    log_dg = np.where(
        oh >= RR14_OH_BREAK,
        RR14_LOGDG0 + RR14_ALPHA_H * doh,
        RR14_LOGDG0 + RR14_ALPHA_H * (RR14_OH_BREAK - OH_SOLAR)
                    + RR14_ALPHA_L * (oh - RR14_OH_BREAK))
    return 10.0**log_dg / Z

_dv19_oh  = np.array([7.3, 7.5, 7.7, 7.9, 8.1, 8.2, 8.35, 8.5,
                       8.6, 8.69, 8.75, 8.85])
_dv19_dtm = np.array([0.010, 0.018, 0.030, 0.055, 0.095, 0.130, 0.180,
                       0.250, 0.320, 0.380, 0.400, 0.420])
DEVIS19_Z  = Z_SOLAR * 10.0**(_dv19_oh - OH_SOLAR)
DEVIS19_DZ = _dv19_dtm

_mk17_z  = np.array([0.001, 0.002, 0.004, 0.007, 0.012, 0.020, 0.030, 0.040])
_mk17_dz = np.array([0.010, 0.020, 0.045, 0.100, 0.190, 0.290, 0.360, 0.400])

_ao18_z  = np.array([0.0005, 0.001, 0.003, 0.006, 0.010, 0.018, 0.030, 0.040])
_ao18_dz = np.array([0.005,  0.012, 0.035, 0.090, 0.170, 0.280, 0.370, 0.410])

_li19_z  = np.array([0.0003, 0.001, 0.003, 0.007, 0.013, 0.020, 0.030, 0.040])
_li19_dz = np.array([0.003,  0.010, 0.030, 0.090, 0.180, 0.280, 0.370, 0.420])

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / catalog helpers
# ─────────────────────────────────────────────────────────────────────────────

def output_dir(run, resolution):
    return BASE_DIR / f"{run}_output_{resolution}"


def find_snapshots(run, resolution):
    odir = output_dir(run, resolution)
    if not odir.is_dir():
        return []
    seen, bases = set(), []
    for snapdir in sorted(glob.glob(str(odir / "snapdir_*"))):
        for f in sorted(glob.glob(os.path.join(snapdir, "snapshot_*.0.hdf5"))):
            base = re.sub(r"\.0\.hdf5$", "", f)
            if base not in seen:
                seen.add(base); bases.append(base)
        for f in sorted(glob.glob(os.path.join(snapdir, "snapshot_*.hdf5"))):
            if ".0.hdf5" in f: continue
            base = re.sub(r"\.hdf5$", "", f)
            if base not in seen:
                seen.add(base); bases.append(base)
    return sorted(bases)


def snap_redshift(snap_base):
    for suffix in [".hdf5", ".0.hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
                    z = hf["Header"].attrs.get("Redshift", None)
                    if z is not None: return float(z)
            except Exception:
                pass
    return None


def find_snap_near_z(snap_bases, target_z):
    best, best_dz = None, 1e30
    for sb in snap_bases:
        z = snap_redshift(sb)
        if z is not None and abs(z - target_z) < best_dz:
            best_dz = abs(z - target_z); best = sb
    return best, best_dz


def read_header(snap_base):
    """Read cosmological header; HubbleParam lives in Parameters, not Header."""
    defaults = dict(h=0.6774, a=1.0, um_cgs=1.989e43, ul_cm=3.085678e21)
    for suffix in [".0.hdf5", ".hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
                    # HubbleParam is in Parameters group, not Header
                    h = float(hf["Parameters"].attrs["HubbleParam"])
                    attrs = hf["Header"].attrs
                    return dict(
                        h      = h,
                        a      = float(attrs.get("Time",             defaults["a"])),
                        um_cgs = float(attrs.get("UnitMass_in_g",    defaults["um_cgs"])),
                        ul_cm  = float(attrs.get("UnitLength_in_cm", defaults["ul_cm"])),
                    )
            except Exception:
                pass
    print("  WARNING: could not read header — using defaults")
    return defaults


def subfiles(snap_base):
    files = sorted(glob.glob(snap_base + ".*.hdf5"))
    if not files:
        single = snap_base + ".hdf5"
        files = [single] if os.path.exists(single) else []
    return files


def get_halo_center(run, snap_base, resolution):
    """Return halo center in comoving kpc/h and R_crit200."""
    m = re.search(r"snapshot_(\d+)$", snap_base)
    if not m: return None, None
    snap_num   = m.group(1)
    groups_dir = output_dir(run, resolution) / f"groups_{snap_num}"
    cats = sorted(glob.glob(
        str(groups_dir / f"fof_subhalo_tab_{snap_num}.*.hdf5")))
    if not cats: return None, None
    try:
        with h5py.File(cats[0], "r") as hf:
            if "Group" not in hf: return None, None
            grp = hf["Group"]
            if "GroupPos" not in grp or grp["GroupPos"].shape[0] == 0:
                return None, None
            ctr  = grp["GroupPos"][0].astype(float)
            # Prefer Crit200 over Mean200 for consistency with rest of pipeline
            if "Group_R_Crit200" in grp:
                r200 = float(grp["Group_R_Crit200"][0])
            elif "Group_R_Mean200" in grp:
                r200 = float(grp["Group_R_Mean200"][0])
            else:
                r200 = None
    except Exception as e:
        print(f"  [{run}/{resolution}] catalog error: {e}")
        return None, None
    return ctr, r200


def density_to_nH(rho_code, um_cgs, ul_cm, h, a):
    """
    Convert Gadget-4 comoving code density to physical nH [cm^-3].

    Gadget-4 stores density in units of (UnitMass/h) / (UnitLength/h)^3,
    so converting to physical CGS requires a factor of h^2 / a^3:
        rho_phys = rho_code * (um_cgs / ul_cm^3) * h^2 / a^3
    This matches the conversion in cooling.cc:
        rho *= UnitDensity_in_cgs * HubbleParam^2
    """
    rho_cgs = rho_code * (um_cgs / ul_cm**3) * (h**2 / a**3)
    return rho_cgs * X_H / M_H_G


# ─────────────────────────────────────────────────────────────────────────────
# Galaxy-integrated D/Z and Z_gas
# ─────────────────────────────────────────────────────────────────────────────

# Threshold below which we flag a point as having a depleted/evacuated ISM.
# A run where max nH within the aperture barely reaches this value has no
# meaningful ISM and its D/Z measurement is unreliable.
_ISM_DEPLETED_NH_WARN = 1e-3   # cm^-3


def compute_integrated_dz(snap_base, run, resolution,
                           aperture_pkpc=20.0, nh_min=None):
    """
    Compute galaxy-integrated Z_gas (mass- and SFR-weighted) and D/Z
    within a fixed physical aperture.

    Parameters
    ----------
    aperture_pkpc : float
        Aperture radius in physical kpc (default 20 pkpc).
    nh_min : float or None
        If set, restrict gas to nH >= nh_min [cm^-3] (ISM cut).

    Returns
    -------
    Z_mass, Z_sfr, DZ, r200_pkpc, sfr_fallback, ism_depleted
        ism_depleted : bool
            True if no gas surpassed _ISM_DEPLETED_NH_WARN within the aperture,
            indicating that the ISM has been artificially evacuated.
    """
    hdr    = read_header(snap_base)
    h      = hdr["h"]
    a      = hdr["a"]
    um_cgs = hdr["um_cgs"]
    ul_cm  = hdr["ul_cm"]
    to_pkpc = a / h

    ctr, r200 = get_halo_center(run, snap_base, resolution)
    if ctr is None:
        return None, None, None, None, False, False

    r200_pkpc = r200 * to_pkpc if r200 is not None else float("nan")
    rmax_com  = aperture_pkpc / to_pkpc   # comoving kpc/h

    gas_mass_total  = 0.0
    gas_metal_total = 0.0
    sfr_total       = 0.0
    sfr_metal_total = 0.0
    n_gas_selected  = 0
    nH_max_seen     = 0.0   # track to detect evacuated ISM

    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, "r") as hf:
                if "PartType0" not in hf: continue
                pt0  = hf["PartType0"]
                pos  = pt0["Coordinates"][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < rmax_com

                if not mask.any(): continue

                # Always compute nH for all gas within aperture (for ISM
                # depletion detection), even when no nh_min cut is applied.
                if "Density" in pt0:
                    rho  = pt0["Density"][:][mask]
                    nH   = density_to_nH(rho, um_cgs, ul_cm, h, a)
                    nH_max_seen = max(nH_max_seen, float(nH.max()) if len(nH) else 0.0)

                    if nh_min is not None:
                        inner_mask = nH >= nh_min
                        idx = np.where(mask)[0]
                        full_mask = np.zeros(len(r), dtype=bool)
                        full_mask[idx[inner_mask]] = True
                        mask = full_mask
                        if not mask.any(): continue

                m = pt0["Masses"][:][mask]
                Z = pt0["Metallicity"][:][mask] \
                    if "Metallicity" in pt0 else np.zeros(mask.sum())
                if Z.ndim == 2:
                    Z = Z[:, 0]

                gas_mass_total  += m.sum()
                gas_metal_total += (m * Z).sum()
                n_gas_selected  += mask.sum()

                sfr = pt0["StarFormationRate"][:][mask] \
                      if "StarFormationRate" in pt0 else np.zeros(mask.sum())
                sfr_total       += sfr.sum()
                sfr_metal_total += (sfr * Z).sum()

        except Exception as e:
            print(f"  [{run}/{resolution}] gas read error: {e}")

    ism_depleted = nH_max_seen < _ISM_DEPLETED_NH_WARN

    if gas_mass_total <= 0:
        print(f"  [{run}/{resolution}] WARNING: no gas within aperture "
              f"(nH_min={nh_min}, max nH seen={nH_max_seen:.2e} cm^-3)")
        return None, None, None, r200_pkpc, False, ism_depleted

    dust_mass_total = 0.0
    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, "r") as hf:
                if "PartType6" not in hf: continue
                pt6  = hf["PartType6"]
                pos  = pt6["Coordinates"][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < rmax_com
                if not mask.any(): continue
                dust_mass_total += pt6["Masses"][:][mask].sum()
        except Exception as e:
            print(f"  [{run}/{resolution}] dust read error: {e}")

    Z_mass = gas_metal_total / gas_mass_total

    sfr_fallback = sfr_total <= 0.0
    Z_sfr = Z_mass if sfr_fallback else sfr_metal_total / sfr_total

    M_metals_total = gas_metal_total + dust_mass_total
    DZ = dust_mass_total / M_metals_total if M_metals_total > 0 else 0.0

    nh_str  = f"  nH≥{nh_min}" if nh_min is not None else ""
    fb_str  = " [SFR→mass fallback]" if sfr_fallback \
              else f"  Z_sfr={Z_sfr:.4f} ({Z_sfr/Z_SOLAR:.2f} Z☉)"
    dep_str = "  *** ISM DEPLETED ***" if ism_depleted else ""

    print(f"  [{run}/{resolution}]  R200={r200_pkpc:.0f} pkpc  "
          f"ap={aperture_pkpc:.0f} pkpc{nh_str}  N_gas={n_gas_selected}  "
          f"max_nH={nH_max_seen:.2e} cm^-3  "
          f"M_gas={gas_mass_total:.3e}  M_dust={dust_mass_total:.3e}  "
          f"Z_mass={Z_mass:.4f} ({Z_mass/Z_SOLAR:.2f} Z☉){fb_str}  "
          f"D/Z={DZ:.3f}{dep_str}")

    if ism_depleted:
        print(f"  [{run}/{resolution}] WARNING: max nH within {aperture_pkpc:.0f} pkpc "
              f"is only {nH_max_seen:.2e} cm^-3 — ISM appears evacuated. "
              f"D/Z measurement is unreliable. Consider --unphysical {resolution}:{run}")

    return Z_mass, Z_sfr, DZ, r200_pkpc, sfr_fallback, ism_depleted


# ─────────────────────────────────────────────────────────────────────────────
# Shared plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_reference_data(ax):
    """Plot observational and simulation reference curves."""
    # Extend fit slightly past the data to show saturation plateau clearly
    Z_fit = np.logspace(-4.0, np.log10(0.10), 400)
    ax.plot(Z_fit, rr14_dz(Z_fit), color="black", lw=1.6, ls="--", zorder=5,
            label="Rémy-Ruyer et al. 2014 (DGS+KINGFISH)")
    ax.scatter(DEVIS19_Z, DEVIS19_DZ,
               color="dimgray", marker="o", s=22, zorder=4,
               edgecolors="none", alpha=0.7,
               label="De Vis et al. 2019 (DustPedia)")

    # MW reference at GAS-PHASE metallicity (see module docstring).
    # Z_gas_MW = Z_total * (1 - D/Z) = 0.0134 * 0.60 ≈ 0.008
    ax.scatter(MW_Z_GAS, MW_DZ,
               marker="*", s=280, color="gray", zorder=8,
               label=r"Milky Way (Jenkins 2009; Draine et al. 2007)")

    ax.plot(_mk17_z, _mk17_dz, color="firebrick",  lw=1.6, ls="-.", zorder=3,
            label="McKinnon et al. 2017 (AREPO)")
    ax.plot(_ao18_z, _ao18_dz, color="darkorange", lw=1.6, ls=":",  zorder=3,
            label="Aoyama et al. 2018 (Gadget)")
    ax.plot(_li19_z, _li19_dz, color="darkorchid", lw=1.6,
            ls=(0,(4,2,1,2)), zorder=3, label="Li et al. 2019 (SIMBA)")


def _scatter_sim_point(ax, Z, DZ, color, marker, ms, label,
                        edge_color="white", alpha=1.0, unphysical=False):
    """
    Plot a single simulation data point.

    unphysical : bool
        If True, render as an open marker with an × overlay to signal
        that this measurement is numerically compromised (e.g. evacuated ISM).
    """
    if unphysical:
        # Open marker
        ax.scatter(Z, DZ, color="none", marker=marker, s=ms,
                   edgecolors=color, linewidths=1.8,
                   alpha=0.7, zorder=9, label=label)
        # × overlay
        ax.scatter(Z, DZ, color=color, marker="x",
                   s=ms * 0.6, linewidths=1.8,
                   alpha=0.7, zorder=10)
    else:
        ax.scatter(Z, DZ, color=color, marker=marker, s=ms,
                   edgecolors=edge_color, linewidths=0.6,
                   alpha=alpha, zorder=9, label=label)


def _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title_main):
    """Apply axis formatting, labels, twin x-axis, and split legends."""
    ax.set_xscale("log"); ax.set_yscale("log")

    # Extend right to 0.13 so the 10^-1 major tick is visible with margin
    ax.set_xlim(5e-4, 0.13)
    ax.set_ylim(3e-3, 1.5)

    ax.set_xlabel(r"Gas-phase Metallicity  $Z_{\rm gas}$", fontsize=12)
    ax.set_ylabel(r"Dust-to-Metal Ratio  $D/Z$",           fontsize=12)

    weight_label = {"mass": r"mass-weighted $Z$",
                    "sfr":  r"SFR-weighted $Z$",
                    "both": r"mass- and SFR-weighted $Z$"}[z_weight]
    nh_label = (f"    |    $n_{{\\rm H}} \\geq {nh_min}\\,{{\\rm cm}}^{{-3}}$"
                if nh_min is not None else "")
    # Title without "D/Z at" prefix — that information is on the y-axis label
    ax.set_title(
        f"{title_main}"
        f"    |    $z = 0$    |    $r < {aperture_pkpc:.0f}\\,{{\\rm pkpc}}$"
        f"{nh_label}    |    {weight_label}",
        fontsize=9)

    # Secondary x-axis in Z_sun units — set AFTER xlim is finalised
    ax2 = ax.twiny()
    ax2.grid(False)
    ax2.set_xscale("log")
    ax2.set_xlim(np.array(ax.get_xlim()) / Z_SOLAR)
    ax2.set_xlabel(r"$Z_{\rm gas} / Z_\odot$", fontsize=10)
    ax2.tick_params(labelsize=8)

    # ── Split legend ──────────────────────────────────────────────────────────
    # Separate handles into (1) observational / simulation-comparison references
    # and (2) CosmicGrain run points, so neither legend obscures the data.
    #
    # Reference items are identified by a fixed set of label prefixes.
    # Everything else (S0–S10 rungs, resolution labels) goes into the sim legend.
    REF_PREFIXES = (
        "Rémy-Ruyer", "De Vis", "Milky Way",
        "McKinnon",   "Aoyama", "Li et al",
    )

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))   # deduplicate by label text

    ref_h, ref_l, sim_h, sim_l = [], [], [], []
    for lbl, hdl in by_label.items():
        if any(lbl.startswith(p) for p in REF_PREFIXES):
            ref_h.append(hdl); ref_l.append(lbl)
        else:
            sim_h.append(hdl); sim_l.append(lbl)

    def _make_leg(handles, labels, loc, title=None):
        leg = ax.legend(handles, labels, fontsize=8, loc=loc,
                        ncol=1, handlelength=2.2, borderpad=0.6,
                        title=title, title_fontsize=8)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.7)
        leg.get_frame().set_edgecolor("0.8")
        leg.set_zorder(20)
        return leg

    # Comparison references → lower right (compact, below the sim data cluster)
    if ref_h:
        leg_ref = _make_leg(ref_h, ref_l, "lower right")
        ax.add_artist(leg_ref)

    # CosmicGrain rungs → lower left (empty triangle below the trend lines)
    if sim_h:
        _make_leg(sim_h, sim_l, "lower left")

    ax.grid(True, which="both", ls=":", alpha=0.25, color="gray", zorder=0)
    ax.tick_params(which="both", direction="in", top=False)


# ─────────────────────────────────────────────────────────────────────────────
# Plot — physics ladder (single resolution)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot_ladder(run_points, resolution, aperture_pkpc, nh_min,
                     z_weight, output_path):
    """
    run_points : list of (run, Z_mass, Z_sfr, DZ, sfr_fallback, ism_depleted)
    """
    show_mass = z_weight in ("mass", "both")
    show_sfr  = z_weight in ("sfr",  "both")
    show_both = z_weight == "both"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _draw_reference_data(ax)

    valid = [(r, Zm, Zs, DZ, fb, dep)
             for r, Zm, Zs, DZ, fb, dep in run_points
             if Zm is not None and DZ is not None and Zm > 0]

    if not valid:
        print("WARNING: no valid points for ladder plot")
        return

    for run, Z_mass, Z_sfr, DZ, sfr_fallback, ism_depleted in valid:
        cfg    = RUN_CONFIGS.get(run, {})
        color  = cfg.get("color",  "#2a9d8f")
        marker = cfg.get("marker", "o")
        ms     = cfg.get("ms",     80)
        label  = cfg.get("label",  run)
        if ism_depleted:
            label += " (ISM evacuated)"

        # Skip points with DZ=0 but warn
        if DZ <= 0:
            print(f"  WARNING: [{run}] D/Z = {DZ:.3e} — skipping point")
            continue

        if show_mass or not show_sfr:
            _scatter_sim_point(ax, Z_mass, DZ, color, marker, ms,
                               label if not show_both else "_nolegend_",
                               unphysical=ism_depleted)

        if show_sfr:
            sfr_marker = "*" if marker == "*" else "s"
            _scatter_sim_point(ax, Z_sfr, DZ, color, sfr_marker, ms,
                               "_nolegend_",
                               alpha=0.45 if sfr_fallback else 1.0,
                               unphysical=ism_depleted)

        # Arrow connecting mass-weighted → SFR-weighted only in 'both' mode
        if show_both and Z_mass != Z_sfr and not sfr_fallback:
            ax.annotate("", xy=(Z_sfr, DZ), xytext=(Z_mass, DZ),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=7), zorder=8)

    title = f"CosmicGrain ${resolution}^3$"
    _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot — convergence (multiple resolutions)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot_convergence(run_points, resolutions, aperture_pkpc, nh_min,
                           z_weight, output_path, unphysical_set=None):
    """
    run_points     : list of (run, resolution, Z_mass, Z_sfr, DZ,
                              sfr_fallback, ism_depleted)
    unphysical_set : set of (resolution, run) pairs to render as unphysical
                     regardless of auto-detection (from --unphysical CLI arg)
    """
    if unphysical_set is None:
        unphysical_set = set()

    show_mass = z_weight in ("mass", "both")
    show_sfr  = z_weight in ("sfr",  "both")
    show_both = z_weight == "both"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _draw_reference_data(ax)

    valid = [(r, res, Zm, Zs, DZ, fb, dep)
             for r, res, Zm, Zs, DZ, fb, dep in run_points
             if Zm is not None and Zm > 0]

    if not valid:
        print("WARNING: no valid points for convergence plot")
        return

    unique_runs = list(dict.fromkeys(r for r, *_ in valid))

    for run, resolution, Z_mass, Z_sfr, DZ, sfr_fallback, ism_depleted in valid:
        rcfg      = RES_CONFIGS.get(resolution, {})
        color     = rcfg.get("color",  "#2a9d8f")
        marker    = rcfg.get("marker", "o")
        ms        = rcfg.get("ms",     100)
        res_label = rcfg.get("label",  str(resolution))

        label = (f"CosmicGrain {res_label}" if len(unique_runs) == 1
                 else f"CosmicGrain {run} {res_label}")

        # A point is unphysical if auto-detected OR explicitly flagged
        bad = ism_depleted or (resolution, run) in unphysical_set
        if bad:
            label += " (unphysical — ISM evacuated)"

        if DZ <= 0:
            print(f"  WARNING: [{run}/{resolution}] D/Z = {DZ:.3e} — skipping point")
            # Still draw the x marker to show where it would have been
            ax.scatter(Z_mass, 3.5e-3, color=color, marker="x",
                       s=ms * 0.6, linewidths=1.8, alpha=0.7, zorder=9)
            continue

        if show_mass or not show_sfr:
            _scatter_sim_point(ax, Z_mass, DZ, color, marker, ms, label,
                               unphysical=bad)
        if show_sfr:
            _scatter_sim_point(ax, Z_sfr, DZ, color, "^", ms, "_nolegend_",
                               alpha=0.45 if sfr_fallback else 1.0,
                               unphysical=bad)
        if show_both and Z_mass != Z_sfr and not sfr_fallback:
            ax.annotate("", xy=(Z_sfr, DZ), xytext=(Z_mass, DZ),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=7), zorder=8)

    runs_label = ", ".join(unique_runs)
    title = f"CosmicGrain {runs_label} — resolution convergence"
    _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_unphysical(spec_list):
    """
    Parse --unphysical args of the form 'res:run' into a set of (int, str).
    E.g. ['2048:S10', '512:S0'] → {(2048, 'S10'), (512, 'S0')}
    """
    result = set()
    for spec in (spec_list or []):
        try:
            res_str, run = spec.split(":")
            result.add((int(res_str), run))
        except ValueError:
            print(f"WARNING: could not parse --unphysical '{spec}' "
                  f"(expected format: res:run, e.g. 2048:S10) — ignoring")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--res", nargs="+", type=int, default=[512])
    parser.add_argument("--aperture", type=float, default=20.0,
                        help="Aperture radius in physical kpc (default: 20 pkpc)")
    parser.add_argument("--nh-min", dest="nh_min", type=float, default=None,
                        help="Minimum nH [cm^-3] for ISM gas cut (e.g. 0.1)")
    parser.add_argument("--z-weight", dest="z_weight",
                        choices=["mass", "sfr", "both"], default="mass")
    parser.add_argument("--unphysical", nargs="+", default=None,
                        metavar="RES:RUN",
                        help="Mark specific (resolution, run) pairs as "
                             "numerically compromised, e.g. --unphysical 2048:S10. "
                             "Rendered as open markers with × overlay.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    resolutions      = args.res
    convergence_mode = len(resolutions) > 1
    unphysical_set   = parse_unphysical(args.unphysical)

    if args.runs is None:
        runs = ["S10"] if convergence_mode else \
               ["S0","S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]
    else:
        runs = args.runs

    if args.output:
        out = Path(args.output)
    elif convergence_mode:
        runs_str = "_".join(runs)
        res_str  = "_".join(str(r) for r in resolutions)
        out = FIGDIR / f"dz_convergence_{runs_str}_{res_str}_{args.z_weight}.png"
    else:
        out = FIGDIR / f"dz_integrated_{resolutions[0]}_{args.z_weight}.png"

    nh_str = f"{args.nh_min} cm^-3" if args.nh_min is not None else "none"
    print(f"\nMode:        {'convergence' if convergence_mode else 'physics ladder'}")
    print(f"Runs:        {runs}")
    print(f"Resolutions: {resolutions}")
    print(f"Aperture:    {args.aperture:.0f} pkpc (physical)")
    print(f"nH cut:      {nh_str}")
    print(f"Z-weight:    {args.z_weight}")
    print(f"Unphysical:  {unphysical_set if unphysical_set else 'none'}")
    print(f"Base dir:    {BASE_DIR}")
    print(f"Output:      {out}\n")
    print(f"MW reference: Z_gas = {MW_Z_GAS:.4f} ({MW_Z_GAS/Z_SOLAR:.2f} Z_sun), "
          f"D/Z = {MW_DZ:.2f}  [gas-phase Z, depleted]\n")

    if convergence_mode:
        run_points = []
        for resolution in resolutions:
            for run in runs:
                snaps = find_snapshots(run, resolution)
                if not snaps:
                    print(f"  [{run}/{resolution}] no snapshots — skipping")
                    continue
                snap_base, dz = find_snap_near_z(snaps, 0.0)
                if dz > 0.2:
                    print(f"  [{run}/{resolution}] no z~0 snapshot "
                          f"(dz={dz:.2f}) — skipping")
                    continue
                result = compute_integrated_dz(
                    snap_base, run, resolution,
                    aperture_pkpc=args.aperture, nh_min=args.nh_min)
                Z_mass, Z_sfr, DZ, _, sfr_fallback, ism_depleted = result
                run_points.append(
                    (run, resolution, Z_mass, Z_sfr, DZ,
                     sfr_fallback, ism_depleted))

        make_plot_convergence(run_points, resolutions,
                              args.aperture, args.nh_min,
                              args.z_weight, out,
                              unphysical_set=unphysical_set)
    else:
        resolution = resolutions[0]
        run_points = []
        for run in runs:
            snaps = find_snapshots(run, resolution)
            if not snaps:
                print(f"  [{run}] no snapshots — skipping")
                continue
            snap_base, dz = find_snap_near_z(snaps, 0.0)
            if dz > 0.2:
                print(f"  [{run}] no z~0 snapshot (dz={dz:.2f}) — skipping")
                continue
            result = compute_integrated_dz(
                snap_base, run, resolution,
                aperture_pkpc=args.aperture, nh_min=args.nh_min)
            Z_mass, Z_sfr, DZ, _, sfr_fallback, ism_depleted = result
            run_points.append((run, Z_mass, Z_sfr, DZ, sfr_fallback, ism_depleted))

        make_plot_ladder(run_points, resolution,
                         args.aperture, args.nh_min,
                         args.z_weight, out)

    print("Done.")


if __name__ == "__main__":
    main()
