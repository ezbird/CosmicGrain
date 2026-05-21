#!/usr/bin/env python3
"""
plot_dz_vs_metallicity.py
--------------------------
Galaxy-integrated D/Z vs. gas-phase metallicity for the CosmicGrain
simulation ladder, compared to observations and other simulations.

Run from anywhere — paths are anchored to the parent of this script file,
so the working directory does not matter:

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

Each point is computed as averages within a fixed physical aperture:

    Z_mass = Σ(m_gas × Z_gas) / Σ(m_gas)          [mass-weighted]
    Z_sfr  = Σ(SFR_gas × Z_gas) / Σ(SFR_gas)      [SFR-weighted, HII-analog]
    D/Z    = M_dust / (M_gas_metals + M_dust)        [total-metal definition]

Aperture is specified in physical kpc (--aperture, default 20 pkpc), motivated
by the stellar surface density break at ~20–25 pkpc in Halo 569. An optional
minimum hydrogen number density cut (--nh-min, default 0.1 cm^-3) restricts
the sample to ISM gas, matching the observational comparisons which all measure
disk/ISM properties rather than CGM.

R200 is still read from the FOF catalog to locate the halo center, but does
not enter the aperture calculation.

SFR-weighting more faithfully mimics strong-line observational metallicities
(O3N2, R23, etc.) which trace HII regions proportional to instantaneous SFR.
Falls back to mass-weighting for runs with no star-forming gas (e.g. S0).

REFERENCES
----------
Observations:
  Rémy-Ruyer+2014 (A&A 563, A31)  — 126 DGS+KINGFISH galaxies, BPL fit
  De Vis+2019 (A&A 623, A5)       — DustPedia+RR14 compilation

Simulations:
  McKinnon+2017 (MNRAS 468, 1505) — AREPO subgrid ISM
  Aoyama+2018  (MNRAS 478, 4905)  — Gadget two-size
  Li+2019      (MNRAS 490, 1425)  — SIMBA/GIZMO

Usage:
    # Physics ladder — one resolution, all runs
    python plot_dz_vs_metallicity.py --res 1024

    # Physics ladder — subset of runs
    python plot_dz_vs_metallicity.py --res 1024 --runs S0 S4 S10

    # Convergence — multiple resolutions, default S10
    python plot_dz_vs_metallicity.py --res 512 1024 2048

    # Convergence — multiple resolutions, specific run(s)
    python plot_dz_vs_metallicity.py --res 512 1024 2048 --runs S10

    # Common options
    python plot_dz_vs_metallicity.py --res 1024 --aperture 20 --nh-min 0.1
    python plot_dz_vs_metallicity.py --res 1024 --z-weight sfr
    python plot_dz_vs_metallicity.py --res 1024 --z-weight both
    python plot_dz_vs_metallicity.py --res 1024 --output my_dz.png
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
# This script lives in  ~/gadget4/scripts/
# Simulation data lives in ~/gadget4/  (one level up)
SCRIPT_DIR = Path(__file__).resolve().parent   # .../gadget4/scripts
BASE_DIR   = SCRIPT_DIR.parent                 # .../gadget4

FIGDIR = BASE_DIR / "dust_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.style.use(str(SCRIPT_DIR / "sleek.mplstyle"))

# ─────────────────────────────────────────────────────────────────────────────
# Styling — physics ladder (one res, multiple runs)
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
    "S10": {"label": "S10: + Rad. pressure (full)", "color": "#2a9d8f", "marker": "*", "ms": 350},
}

# ─────────────────────────────────────────────────────────────────────────────
# Styling — convergence (multiple res, one or more runs)
# ─────────────────────────────────────────────────────────────────────────────
RES_CONFIGS = {
    512:  {"label": r"$512^3$",  "color": "#7ec8c0", "marker": "o", "ms": 100},
    1024: {"label": r"$1024^3$", "color": "#2a9d8f", "marker": "*", "ms": 350},
    2048: {"label": r"$2048^3$", "color": "#1a6b62", "marker": "D", "ms": 100},
    4096: {"label": r"$4096^3$", "color": "#0d3d38", "marker": "s", "ms": 100},
}

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
Z_SOLAR  = 0.0134      # Asplund+2009
OH_SOLAR = 8.69
X_H      = 0.76        # hydrogen mass fraction
M_H_G    = 1.6736e-24  # proton mass [g]

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

# De Vis+2019 digitized points
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
    defaults = dict(h=0.7, a=1.0, um_cgs=1.989e43, ul_cm=3.085678e21)
    for suffix in [".0.hdf5", ".hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
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
    return defaults


def subfiles(snap_base):
    files = sorted(glob.glob(snap_base + ".*.hdf5"))
    if not files:
        single = snap_base + ".hdf5"
        files = [single] if os.path.exists(single) else []
    return files


def get_halo_center(run, snap_base, resolution):
    """Return halo center in comoving kpc/h (and R200 for logging only)."""
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
            r200 = float(grp["Group_R_Crit200"][0]) \
                   if "Group_R_Crit200" in grp else None
    except Exception as e:
        print(f"  [{run}/{resolution}] catalog error: {e}")
        return None, None
    return ctr, r200


def density_to_nH(rho_code, um_cgs, ul_cm, h, a):
    """Convert Gadget-4 comoving code density to physical nH [cm^-3]."""
    rho_cgs = rho_code * (um_cgs / ul_cm**3) * (h**2 / a**3)
    return rho_cgs * X_H / M_H_G


# ─────────────────────────────────────────────────────────────────────────────
# Galaxy-integrated D/Z and Z_gas
# ─────────────────────────────────────────────────────────────────────────────

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
        If set, restrict gas to nH >= nh_min [cm^-3] to select ISM gas.

    Returns
    -------
    Z_mass, Z_sfr, DZ, r200_pkpc, sfr_fallback
    """
    hdr    = read_header(snap_base)
    h      = hdr["h"]
    a      = hdr["a"]
    um_cgs = hdr["um_cgs"]
    ul_cm  = hdr["ul_cm"]
    to_pkpc = a / h

    ctr, r200 = get_halo_center(run, snap_base, resolution)
    if ctr is None:
        return None, None, None, None, False

    r200_pkpc = r200 * to_pkpc if r200 is not None else float("nan")
    rmax_com  = aperture_pkpc / to_pkpc   # comoving kpc/h

    # ── Gas ──────────────────────────────────────────────────────────────────
    gas_mass_total  = 0.0
    gas_metal_total = 0.0
    sfr_total       = 0.0
    sfr_metal_total = 0.0
    n_gas_selected  = 0

    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, "r") as hf:
                if "PartType0" not in hf: continue
                pt0  = hf["PartType0"]
                pos  = pt0["Coordinates"][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < rmax_com

                if nh_min is not None and "Density" in pt0:
                    rho  = pt0["Density"][:][mask]
                    nH   = density_to_nH(rho, um_cgs, ul_cm, h, a)
                    full_mask = np.zeros(len(r), dtype=bool)
                    idx = np.where(mask)[0]
                    full_mask[idx[nH >= nh_min]] = True
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

    if gas_mass_total <= 0:
        print(f"  [{run}/{resolution}] no gas within aperture (nh_min={nh_min})")
        return None, None, None, None, False

    # ── Dust ─────────────────────────────────────────────────────────────────
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

    # ── Integrated quantities ─────────────────────────────────────────────────
    Z_mass = gas_metal_total / gas_mass_total

    sfr_fallback = sfr_total <= 0.0
    Z_sfr = Z_mass if sfr_fallback else sfr_metal_total / sfr_total

    M_metals_total = gas_metal_total + dust_mass_total
    DZ = dust_mass_total / M_metals_total if M_metals_total > 0 else 0.0

    nh_str = f"  nH≥{nh_min}" if nh_min is not None else ""
    fb_str = " [SFR→mass fallback]" if sfr_fallback else \
             f" Z_sfr={Z_sfr:.4f} ({Z_sfr/Z_SOLAR:.2f} Z☉)"
    print(f"  [{run}/{resolution}]  R200={r200_pkpc:.0f} pkpc  "
          f"ap={aperture_pkpc:.0f} pkpc{nh_str}  N_gas={n_gas_selected}  "
          f"M_gas={gas_mass_total:.3e}  M_dust={dust_mass_total:.3e}  "
          f"Z_mass={Z_mass:.4f} ({Z_mass/Z_SOLAR:.2f} Z☉){fb_str}  D/Z={DZ:.3f}")

    return Z_mass, Z_sfr, DZ, r200_pkpc, sfr_fallback


# ─────────────────────────────────────────────────────────────────────────────
# Shared plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_reference_data(ax):
    Z_fit = np.logspace(-4.0, np.log10(0.08), 300)
    ax.plot(Z_fit, rr14_dz(Z_fit), color="black", lw=1.6, ls="--", zorder=5,
            label="Rémy-Ruyer et al. 2014 (DGS+KINGFISH)")
    ax.scatter(DEVIS19_Z, DEVIS19_DZ,
               color="dimgray", marker="o", s=22, zorder=4,
               edgecolors="none", alpha=0.7,
               label="De Vis et al. 2019 (DustPedia)")
    ax.plot(_mk17_z, _mk17_dz, color="firebrick",  lw=1.6, ls="-.", zorder=3,
            label="McKinnon et al. 2017 (AREPO)")
    ax.plot(_ao18_z, _ao18_dz, color="darkorange", lw=1.6, ls=":",  zorder=3,
            label="Aoyama et al. 2018 (Gadget)")
    ax.plot(_li19_z, _li19_dz, color="darkorchid", lw=1.6,
            ls=(0,(4,2,1,2)), zorder=3, label="Li et al. 2019 (SIMBA)")
    ax.axhline(0.4, color="gray", lw=0.9, ls=":", alpha=0.7, zorder=0)
    ax.text(7e-4, 0.42, "D/Z = 0.40 (MW estimate)",
            color="gray", fontsize=8, va="bottom")
    ax.text(Z_SOLAR * 1.07, 1.1, r"$Z_\odot$",
            color="goldenrod", fontsize=9, va="top")


def _scatter_point(ax, Z, DZ, color, marker, ms, label,
                   show_both=False, edge_color="white", alpha=1.0):
    ax.scatter(Z, DZ, color=color, marker=marker, s=ms,
               edgecolors=edge_color,
               linewidths=1.2 if show_both else 0.6,
               alpha=alpha, zorder=9, label=label)


def _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title_main):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e-4, 0.06); ax.set_ylim(3e-3, 1.5)
    ax.set_xlabel(r"Gas-phase Metallicity  $Z_{\rm gas}$", fontsize=12)
    ax.set_ylabel(r"Dust-to-Metal Ratio  $D/Z$",           fontsize=12)

    weight_label = {"mass": r"mass-weighted $Z$",
                    "sfr":  r"SFR-weighted $Z$",
                    "both": r"mass- and SFR-weighted $Z$"}[z_weight]
    nh_label = (f"    |    $n_{{\\rm H}} \\geq {nh_min}\\,{{\\rm cm}}^{{-3}}$"
                if nh_min is not None else "")
    ax.set_title(
        f"{title_main}"
        f"    |    $r < {aperture_pkpc:.0f}\\,{{\\rm pkpc}}${nh_label}"
        f"    |    {weight_label}",
        fontsize=9)

    ax2 = ax.twiny()
    ax2.grid(False); ax2.set_axisbelow(True); ax2.set_xscale("log")
    ax2.set_xlim(np.array(ax.get_xlim()) / Z_SOLAR)
    ax2.set_xlabel(r"$Z_{\rm gas} / Z_\odot$", fontsize=10)
    ax2.tick_params(labelsize=8)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),
                    fontsize=11, loc="lower right",
                    ncol=1, handlelength=2.2, borderpad=0.7)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.5)
    leg.get_frame().set_edgecolor("0.8")
    leg.set_zorder(20); leg.get_frame().set_zorder(20)

    ax.grid(True, which="both", ls=":", alpha=0.25, color="gray", zorder=0)
    ax.tick_params(which="both", direction="in", top=False)


# ─────────────────────────────────────────────────────────────────────────────
# Plot — physics ladder (single resolution)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot_ladder(run_points, resolution, aperture_pkpc, nh_min,
                     z_weight, output_path):
    """run_points : list of (run, Z_mass, Z_sfr, DZ, sfr_fallback)"""
    show_mass = z_weight in ("mass", "both")
    show_sfr  = z_weight in ("sfr",  "both")
    show_both = z_weight == "both"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _draw_reference_data(ax)

    if show_both:
        ax.scatter([], [], color="none", edgecolors="steelblue",
                   linewidths=1.2, s=55, marker="o",
                   label=r"CosmicGrain — mass-weighted $Z$")
        ax.scatter([], [], color="none", edgecolors="coral",
                   linewidths=1.2, s=55, marker="s",
                   label=r"CosmicGrain — SFR-weighted $Z$ (HII-analog)")

    valid = [(r, Zm, Zs, DZ, fb)
             for r, Zm, Zs, DZ, fb in run_points
             if Zm is not None and DZ is not None and DZ > 0 and Zm > 0]

    for run, Z_mass, Z_sfr, DZ, sfr_fallback in valid:
        cfg    = RUN_CONFIGS.get(run, {})
        color  = cfg.get("color",  "#2a9d8f")
        marker = cfg.get("marker", "o")
        ms     = cfg.get("ms",     80)
        label  = cfg.get("label",  run)

        point_label = label if not show_both else \
                      (label if run in ("S0", "S10") else "_nolegend_")

        if show_mass:
            _scatter_point(ax, Z_mass, DZ, color, marker, ms,
                           point_label if not show_sfr else "_nolegend_",
                           show_both,
                           edge_color="steelblue" if show_both else "white")
        if show_sfr:
            sfr_marker = "*" if marker == "*" else "s"
            _scatter_point(ax, Z_sfr, DZ, color, sfr_marker, ms,
                           point_label if not show_mass else "_nolegend_",
                           show_both,
                           edge_color="coral" if show_both else "white",
                           alpha=0.45 if sfr_fallback else 1.0)
        if not show_sfr and not show_both:
            _scatter_point(ax, Z_mass, DZ, color, marker, ms, label)

        if show_both and Z_mass != Z_sfr and not sfr_fallback:
            ax.annotate("", xy=(Z_sfr, DZ), xytext=(Z_mass, DZ),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=7), zorder=8)

    title = f"D/Z  at $z = 0$    |    CosmicGrain ${resolution}^3$"
    _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot — convergence (multiple resolutions)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot_convergence(run_points, resolutions, aperture_pkpc, nh_min,
                          z_weight, output_path):
    """run_points : list of (run, resolution, Z_mass, Z_sfr, DZ, sfr_fallback)"""
    show_mass = z_weight in ("mass", "both")
    show_sfr  = z_weight in ("sfr",  "both")
    show_both = z_weight == "both"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    _draw_reference_data(ax)

    if show_both:
        ax.scatter([], [], color="none", edgecolors="steelblue",
                   linewidths=1.2, s=55, marker="o",
                   label=r"CosmicGrain — mass-weighted $Z$")
        ax.scatter([], [], color="none", edgecolors="coral",
                   linewidths=1.2, s=55, marker="^",
                   label=r"CosmicGrain — SFR-weighted $Z$ (HII-analog)")

    valid = [(r, res, Zm, Zs, DZ, fb)
             for r, res, Zm, Zs, DZ, fb in run_points
             if Zm is not None and DZ is not None and DZ > 0 and Zm > 0]

    unique_runs = list(dict.fromkeys(r for r, *_ in valid))

    for run, resolution, Z_mass, Z_sfr, DZ, sfr_fallback in valid:
        rcfg      = RES_CONFIGS.get(resolution, {})
        color     = rcfg.get("color",  "#2a9d8f")
        marker    = rcfg.get("marker", "o")
        ms        = rcfg.get("ms",     100)
        res_label = rcfg.get("label",  str(resolution))
        label = (f"CosmicGrain {res_label}" if len(unique_runs) == 1
                 else f"CosmicGrain {run} {res_label}")

        if show_mass:
            _scatter_point(ax, Z_mass, DZ, color, marker, ms,
                           label if not show_sfr else "_nolegend_",
                           show_both,
                           edge_color="steelblue" if show_both else "white")
        if show_sfr:
            _scatter_point(ax, Z_sfr, DZ, color, "^", ms,
                           label if not show_mass else "_nolegend_",
                           show_both,
                           edge_color="coral" if show_both else "white",
                           alpha=0.45 if sfr_fallback else 1.0)
        if not show_sfr and not show_both:
            _scatter_point(ax, Z_mass, DZ, color, marker, ms, label)

        if show_both and Z_mass != Z_sfr and not sfr_fallback:
            ax.annotate("", xy=(Z_sfr, DZ), xytext=(Z_mass, DZ),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.8, mutation_scale=7), zorder=8)

    runs_label = ", ".join(unique_runs)
    title = f"D/Z  at $z = 0$    |    CosmicGrain {runs_label} — resolution convergence"
    _finalize_axes(ax, aperture_pkpc, nh_min, z_weight, title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Ladder runs to include. Default: S0–S10 for "
                             "single-res mode; S10 for convergence mode.")
    parser.add_argument("--res", nargs="+", type=int, default=[512],
                        help="Resolution(s). Single value → physics ladder. "
                             "Multiple values → convergence plot. (default: 512)")
    parser.add_argument("--aperture", type=float, default=20.0,
                        help="Aperture radius in physical kpc (default: 20 pkpc)")
    parser.add_argument("--nh-min", dest="nh_min", type=float, default=None,
                        help="Minimum nH [cm^-3] to restrict to ISM gas "
                             "(recommended: 0.1; default: None)")
    parser.add_argument("--z-weight", dest="z_weight",
                        choices=["mass", "sfr", "both"], default="mass",
                        help="Metallicity weighting: 'mass' (default), "
                             "'sfr', or 'both'")
    parser.add_argument("--output", default=None,
                        help="Output path (auto-named by default)")
    args = parser.parse_args()

    resolutions      = args.res
    convergence_mode = len(resolutions) > 1

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
    print(f"Base dir:    {BASE_DIR}")
    print(f"Output:      {out}\n")

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
                Z_mass, Z_sfr, DZ, _, sfr_fallback = compute_integrated_dz(
                    snap_base, run, resolution,
                    aperture_pkpc=args.aperture, nh_min=args.nh_min)
                run_points.append(
                    (run, resolution, Z_mass, Z_sfr, DZ, sfr_fallback))

        if not any(Zm is not None for _, _, Zm, _, _, _ in run_points):
            print("No valid run points — check snapshot paths.")
            return
        make_plot_convergence(run_points, resolutions,
                              args.aperture, args.nh_min,
                              args.z_weight, out)

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
            Z_mass, Z_sfr, DZ, _, sfr_fallback = compute_integrated_dz(
                snap_base, run, resolution,
                aperture_pkpc=args.aperture, nh_min=args.nh_min)
            run_points.append((run, Z_mass, Z_sfr, DZ, sfr_fallback))

        if not any(Zm is not None for _, Zm, _, _, _ in run_points):
            print("No valid run points — check snapshot paths.")
            return
        make_plot_ladder(run_points, resolution,
                         args.aperture, args.nh_min,
                         args.z_weight, out)

    print("Done.")


if __name__ == "__main__":
    main()
