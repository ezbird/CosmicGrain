#!/usr/bin/env python3
"""
plot_gsd_comparison.py
======================
Compare CosmicGrain grain-size distribution (GSD) against canonical
analytic models: MRN (1977), Weingartner & Draine (2001), and THEMIS
(Jones et al. 2013/2017).

Spatial regions:
  ISM : r < R_ISM_PKPC  physical kpc from Halo 569 center
  CGM : R_ISM_PKPC <= r < R200_pkpc

Halo identification uses halo_utils.get_halo569_reference / get_halo569,
consistent with all other CosmicGrain analysis scripts.

Unit conventions:
  GrainRadius   : values in HDF5 are already in nm (snap_io applies
                  cm->nm conversion on write). Do NOT apply extra factor.
  CarbonFraction: dimensionless [0, 1]
  Coordinates   : comoving kpc/h  -> physical kpc via * a / h
  Masses        : 1e10 Msun/h
  h             : from f["Parameters"].attrs["HubbleParam"]  (NOT Header)

Usage:
  python plot_gsd_comparison.py ../S10_output_1024/
  python plot_gsd_comparison.py ../S10_output_2048/ --snap 049
  python plot_gsd_comparison.py ../S10_output_1024/ --output gsd_1024.png
  python plot_gsd_comparison.py ../S10_output_1024/ --r-ism 20 --n-bins 45
"""

import argparse
import glob
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
from pathlib import Path
from datetime import datetime

from halo_utils import (
    get_halo569_reference,
    get_halo569,
)

# Load shared paper style
_STYLE = Path(__file__).parent / "cosmicgrain.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
else:
    print(f"Warning: style file not found at {_STYLE}")

# ── Physical constants ────────────────────────────────────────────────────────
MSUN_G   = 1.989e33
RHO_SIL  = 3.5    # g/cm^3  amorphous silicate
RHO_CARB = 2.2    # g/cm^3  amorphous carbon

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SNAP   = "047"
DEFAULT_R_ISM  = 20.0    # physical kpc
DEFAULT_N_BINS = 45
DEFAULT_OUTPUT = None    # auto-named from output_dir + snap

# ── I/O helpers ───────────────────────────────────────────────────────────────

def glob_chunks(directory, basename):
    pattern = str(Path(directory) / f"{basename}*.hdf5")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files


def read_h_a_z(snap_files):
    """Read h (from Parameters), a, z from first snapshot chunk."""
    with h5py.File(snap_files[0], "r") as f:
        h = float(f["Parameters"].attrs["HubbleParam"])
        a = float(f["Header"].attrs["Time"])
        z = float(f["Header"].attrs["Redshift"])
    return h, a, z


def read_box(snap_files):
    with h5py.File(snap_files[0], "r") as f:
        return float(f["Header"].attrs["BoxSize"])


# ── Halo center and R200 ──────────────────────────────────────────────────────

def get_center_r200(output_dir, snap_num_str):
    """
    Return (center_ckpch, r200_pkpc) for Halo 569.

    Uses halo_utils stellar-mass argmax with per-resolution overrides,
    frozen FOF/catalog centering, and catalog SO fallback if needed.
    """
    snap_num   = int(snap_num_str)
    groups_dir = Path(output_dir) / f"groups_{snap_num:03d}"

    ref = get_halo569_reference(output_dir, refine_center=False)
    if ref is None:
        raise RuntimeError("get_halo569_reference returned None")

    halo = get_halo569(
        groups_dir,
        snap_num,
        ref,
        verbose=True,
        refine_center=False,
    )
    if halo is None or halo["r200_ckpch"] <= 0:
        raise RuntimeError(
            f"get_halo569 returned no valid halo for snap {snap_num}")

    print(f"  Center (ckpc/h)  : [{halo['center'][0]:.1f}, "
          f"{halo['center'][1]:.1f}, {halo['center'][2]:.1f}]")
    print(f"  R_Crit200        : {halo['r200_ckpch']:.1f} ckpc/h  "
          f"({halo['r200_pkpc']:.1f} pkpc)")

    return halo["center"], halo["r200_pkpc"]


# ── Dust particle reader ───────────────────────────────────────────────────────

def read_dust(snap_files, box_ckpch, ctr_ckpch, r200_ckpch, h, a):
    """
    Read PartType6 dust particles within R200 of halo center.
    Returns dict: pos_ckpch, mass_code, a_nm, fc
    """
    pos_l, mass_l, arad_l, fc_l = [], [], [], []
    n_total = len(snap_files)

    for idx, fpath in enumerate(snap_files):
        with h5py.File(fpath, "r") as f:
            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]
            if len(pt6["Masses"]) == 0:
                continue

            coords = pt6["Coordinates"][:]   # ckpc/h
            dx   = coords - ctr_ckpch[None, :]
            dx  -= box_ckpch * np.round(dx / box_ckpch)
            r    = np.sqrt((dx**2).sum(axis=1))
            mask = r < r200_ckpch

            if not mask.any():
                continue

            pos_l.append(coords[mask])
            mass_l.append(pt6["Masses"][:][mask])
            arad_l.append(pt6["GrainRadius"][:][mask])
            fc_l.append(pt6["CarbonFraction"][:][mask])

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"    read chunk {idx+1}/{n_total}", end="\r", flush=True)
    print()

    if not pos_l:
        raise RuntimeError("No PartType6 dust found within R200")

    pos  = np.concatenate(pos_l,  axis=0)
    mass = np.concatenate(mass_l, axis=0)
    a_nm = np.concatenate(arad_l, axis=0)   # already nm in HDF5
    fc   = np.concatenate(fc_l,   axis=0)

    print(f"  Dust within R200 : {len(a_nm):,} particles")
    print(f"  GrainRadius range: {a_nm.min():.2f} -- {a_nm.max():.2f} nm")
    print(f"  CarbonFraction   : mean={fc.mean():.3f}  "
          f"min={fc.min():.3f}  max={fc.max():.3f}")

    return dict(pos_ckpch=pos, mass_code=mass, a_nm=a_nm, fc=fc)


# ── Spatial selection ─────────────────────────────────────────────────────────

def apply_radial_cut(dust, ctr_ckpch, box_ckpch, h, a,
                     r_min_pkpc, r_max_pkpc):
    """Select dust in [r_min_pkpc, r_max_pkpc) physical kpc."""
    dx      = dust["pos_ckpch"] - ctr_ckpch[None, :]
    dx     -= box_ckpch * np.round(dx / box_ckpch)
    r_pkpc  = np.sqrt((dx**2).sum(axis=1)) * a / h
    mask    = (r_pkpc >= r_min_pkpc) & (r_pkpc < r_max_pkpc)
    label   = f"r=[{r_min_pkpc:.0f}, {r_max_pkpc:.1f}) pkpc"
    print(f"  {label:45s}: {mask.sum():,} / {len(mask):,}"
          f"  ({100*mask.mean():.1f}%)")
    return {k: v[mask] for k, v in dust.items()}


# ── Number-weighted GSD histogram ─────────────────────────────────────────────

def build_histogram(dust, h, n_bins=DEFAULT_N_BINS,
                    a_min_nm=1.0, a_max_nm=200.0):
    """
    Number-weighted GSD split by composition.
    N_grains = m / (rho_mix * 4pi/3 * a_cm^3)
    """
    MASS_CGS = 1e10 * MSUN_G / h
    NM_TO_CM = 1e-7

    a_nm    = dust["a_nm"]
    fc      = dust["fc"]
    m_g     = dust["mass_code"] * MASS_CGS

    a_cm    = a_nm * NM_TO_CM
    rho_mix = fc * RHO_CARB + (1.0 - fc) * RHO_SIL
    vol     = (4.0 / 3.0) * np.pi * a_cm**3
    N_grains= m_g / (rho_mix * vol)
    N_carb  = fc         * N_grains
    N_sil   = (1.0 - fc) * N_grains

    bins = np.logspace(np.log10(a_min_nm), np.log10(a_max_nm), n_bins + 1)
    da   = np.diff(bins)
    bc   = np.sqrt(bins[:-1] * bins[1:])   # geometric bin centres

    h_c, _ = np.histogram(a_nm, bins=bins, weights=N_carb)
    h_s, _ = np.histogram(a_nm, bins=bins, weights=N_sil)

    with np.errstate(invalid="ignore", divide="ignore"):
        dn_c = np.where(h_c > 0, h_c / da, np.nan)
        dn_s = np.where(h_s > 0, h_s / da, np.nan)

    total = N_grains.sum()
    print(f"  Total grains: {total:.3e}  |  "
          f"Carbon: {N_carb.sum():.3e} ({100*N_carb.sum()/total:.1f}%)")

    return bc, dn_c, dn_s


def dust_masses_msun(dust, h):
    """Return total carbonaceous, silicate, and total dust masses in Msun."""
    mass_msun = dust["mass_code"] * 1e10 / h
    fc = dust["fc"]

    m_carb = np.sum(fc * mass_msun)
    m_sil  = np.sum((1.0 - fc) * mass_msun)
    return m_carb, m_sil, m_carb + m_sil


# ── Analytic reference models ─────────────────────────────────────────────────

def normalise(arr):
    finite = arr[np.isfinite(arr) & (arr > 0)]
    return arr / finite.max() if len(finite) else arr


def mrn_dn(a, a_min=5.0, a_max=200.0):
    return np.where((a >= a_min) & (a <= a_max), a**(-3.5), np.nan)


def wd01_silicate_dn(a):
    alpha, beta = -2.21, 0.3
    a_t, a_c    = 164.0, 100.0
    F           = 1.0 + beta * a / a_c
    cutoff      = np.where(a > a_t, np.exp(-((a - a_t) / a_c)**3), 1.0)
    return normalise((a / a_c)**alpha * F * cutoff)


def wd01_carbonaceous_dn(a):
    a0    = np.array([0.35, 3.0])
    sigma = np.array([0.4,  0.4])
    wt    = np.array([1.0,  0.3])
    dn_ln = np.zeros_like(a, dtype=float)
    for a0i, si, wi in zip(a0, sigma, wt):
        dn_ln += (wi / a) * np.exp(-0.5 * (np.log(a / a0i) / si)**2)
    dn_pow = np.where(a >= 10.7, (a / 10.7)**(-1.54), 0.0)
    return normalise(dn_ln + 0.4 * dn_pow)


def themis_silicate_dn(a):
    return normalise(np.where((a >= 2.0) & (a <= 200.0), a**(-3.4), np.nan))


def themis_carbon_dn(a):
    bumps = [(0.7, 0.4, 1.0), (2.0, 0.4, 0.3)]
    dn_sm = np.zeros_like(a, dtype=float)
    for a0, sig, w in bumps:
        dn_sm += w / a * np.exp(-0.5 * (np.log(a / a0) / sig)**2)
    dn_lg = np.where((a >= 4.0) & (a <= 200.0), a**(-3.5), 0.0)
    return normalise(dn_sm + 0.05 * dn_lg)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(bc, dn_c, dn_s, dn_c_cgm, dn_s_cgm,
                ism_masses, cgm_masses,
                r_ism, r200, output_path, run_label):
    """
    3-panel figure:
      left   = carbonaceous a^4 dn/da
      middle = silicate a^4 dn/da
      right  = integrated dust mass by region/composition

    The GSD panels are independently peak-normalized.
    The mass panel uses absolute dust masses.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]},
        constrained_layout=True
    )

    a_ref = np.logspace(np.log10(1.0), np.log10(200.0), 600)

    ISM_COLOR = "#7B2FBE"
    CGM_COLOR = "#009E73"

    ism_label = f"ISM  ($r < {r_ism:.0f}$ pkpc)"
    cgm_label = f"CGM  ({r_ism:.0f} pkpc < r < R$_{{200}}$)"

    ref_raw = {
        "MRN (1977)": dict(ls="--", color="0.2", lw=1.8,
                           sil=normalise(mrn_dn(a_ref)),
                           carb=normalise(mrn_dn(a_ref))),
        "W & D (2001)": dict(ls="-.", color="#E07B22", lw=1.8,
                             sil=wd01_silicate_dn(a_ref),
                             carb=wd01_carbonaceous_dn(a_ref)),
        "THEMIS": dict(ls=":", color="#3A7EC9", lw=2.0,
                       sil=themis_silicate_dn(a_ref),
                       carb=themis_carbon_dn(a_ref)),
    }

    ref_a4 = {
        name: dict(ls=kw["ls"], color=kw["color"], lw=kw["lw"],
                   sil=normalise(a_ref**4 * kw["sil"]),
                   carb=normalise(a_ref**4 * kw["carb"]))
        for name, kw in ref_raw.items()
    }

    sim_ism = {
        "carb": normalise(bc**4 * dn_c),
        "sil":  normalise(bc**4 * dn_s),
    }

    sim_cgm = {
        "carb": normalise(bc**4 * dn_c_cgm),
        "sil":  normalise(bc**4 * dn_s_cgm),
    }

    panels = [
        (axes[0], "carb", "Carbonaceous"),
        (axes[1], "sil",  "Silicate"),
    ]

    # Dedicated handles for separated legends
    ref_handles = []
    region_handles = [
        plt.Line2D([0], [0], color=ISM_COLOR, lw=2.6, label=ism_label),
        plt.Line2D([0], [0], color=CGM_COLOR, lw=2.6, label=cgm_label),
    ]

    for ax, comp, title in panels:
        ax.set_axisbelow(True)

        for name, kw in ref_a4.items():
            y = kw[comp]
            vis = np.isfinite(y) & (y > 0)
            line, = ax.plot(a_ref[vis], y[vis],
                            ls=kw["ls"], color=kw["color"], lw=kw["lw"],
                            label=name, zorder=3)
            if ax is axes[0]:
                ref_handles.append(line)

        y_ism = sim_ism[comp]
        v = np.isfinite(y_ism) & (y_ism > 0)
        ax.step(bc[v], y_ism[v], where="mid",
                color=ISM_COLOR, lw=2.2, zorder=6, label=ism_label)
        ax.fill_between(bc[v], 1e-10, y_ism[v],
                        step="mid", alpha=0.15, color=ISM_COLOR, zorder=2)

        y_cgm = sim_cgm[comp]
        v = np.isfinite(y_cgm) & (y_cgm > 0)
        ax.step(bc[v], y_cgm[v], where="mid",
                color=CGM_COLOR, lw=2.2, zorder=5, label=cgm_label)
        ax.fill_between(bc[v], 1e-10, y_cgm[v],
                        step="mid", alpha=0.15, color=CGM_COLOR, zorder=1)

        ax.set_title(title, fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 220.0)
        ax.set_ylim(5e-5, 3.0)

        # Keep the GSD-panel grids, but force them behind all data.
        ax.grid(True, which="major", color="0.78", lw=0.5, zorder=0)
        ax.grid(True, which="minor", color="0.92", lw=0.3, zorder=0)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:g}")
        )
        ax.set_xlabel(r"Grain radius (nm)")

    axes[0].set_ylabel(r"$a^4\,\mathrm{d}n/\mathrm{d}a$ (normalized to peak)")
    leg0 = axes[0].legend(
        handles=ref_handles,
        loc="lower left",
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="0.8",
    )
    leg0.set_zorder(50)

    # Mass panel
    ax = axes[2]
    ax.set_axisbelow(True)

    ism_carb, ism_sil, ism_tot = ism_masses
    cgm_carb, cgm_sil, cgm_tot = cgm_masses

    labels = ["Carbon", "Silicate", "Total"]
    x = np.arange(len(labels))
    width = 0.36

    ism_vals = np.array([ism_carb, ism_sil, ism_tot])
    cgm_vals = np.array([cgm_carb, cgm_sil, cgm_tot])

    # ISM is now left, CGM is right.
    ax.bar(x - width/2, ism_vals, width,
           color=ISM_COLOR, alpha=0.75, label=ism_label, zorder=3)
    ax.bar(x + width/2, cgm_vals, width,
           color=CGM_COLOR, alpha=0.75, label=cgm_label, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(1e5, 1e8)
    ax.set_ylabel(r"Dust mass ($M_\odot$)")
    ax.set_title(r"Total Dust Mass within $R_{200}$", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    # No grid/tick lines on the mass panel.
    ax.grid(False)
    ax.minorticks_off()

    ax.legend(
        handles=region_handles,
        fontsize=8.5,
        loc="upper left",
        framealpha=0.95,
        edgecolor="0.8",
    )

    for xpos, val in zip(x - width/2, ism_vals):
        ax.text(xpos, val * 1.12, f"{val:.2e}",
                ha="center", va="bottom", fontsize=8, color=ISM_COLOR)

    for xpos, val in zip(x + width/2, cgm_vals):
        ax.text(xpos, val * 1.12, f"{val:.2e}",
                ha="center", va="bottom", fontsize=8, color=CGM_COLOR)

    out = Path(output_path)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CosmicGrain GSD comparison vs MRN/WD01/THEMIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("output_dir",
                        help="Gadget-4 output directory (e.g. ../S10_output_1024/)")
    parser.add_argument("--snap",    default=DEFAULT_SNAP,
                        help=f"Snapshot number string (default: {DEFAULT_SNAP})")
    parser.add_argument("--r-ism",   type=float, default=DEFAULT_R_ISM,
                        help=f"ISM aperture in pkpc (default: {DEFAULT_R_ISM})")
    parser.add_argument("--n-bins",  type=int,   default=DEFAULT_N_BINS,
                        help=f"GSD histogram bins (default: {DEFAULT_N_BINS})")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT,
                        help="Output PNG path (default: auto)")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    snap_num    = args.snap.zfill(3)
    snap_dir    = output_dir / f"snapdir_{snap_num}"
    groups_dir  = output_dir / f"groups_{snap_num}"
    run_label   = output_dir.name

    # Auto output name
    today = datetime.now().strftime("%-m-%-d-%y")
    if args.output is None:
        out_path = Path(f"gsd_comparison_{run_label}_snap{snap_num}_{today}.pdf")
    else:
        out_path = Path(args.output)

    print("-" * 60)
    print(f"CosmicGrain GSD  |  {run_label}  |  snap {snap_num}")
    print("-" * 60)
    print(f"Output dir  : {output_dir}")
    print(f"ISM aperture: {args.r_ism} pkpc")
    print(f"Bins        : {args.n_bins}")

    snap_base  = str(snap_dir / f"snapshot_{snap_num}")
    snap_files = sorted(glob.glob(snap_base + "*.hdf5"))
    if not snap_files:
        sys.exit(f"ERROR: no snapshot files found at {snap_base}*.hdf5")
    print(f"Snapshot chunks: {len(snap_files)}")

    h, a, z = read_h_a_z(snap_files)
    box      = read_box(snap_files)
    print(f"h={h:.4f}  a={a:.6f}  z={z:.4f}  box={box:.1f} ckpc/h")

    print("\nLocating Halo 569 ...")
    ctr_ckpch, r200_pkpc = get_center_r200(str(output_dir), snap_num)

    r200_ckpch = r200_pkpc * h / a
    r_ism      = args.r_ism

    print(f"\nReading PartType6 (within R200={r200_pkpc:.1f} pkpc) ...")
    dust = read_dust(snap_files, box, ctr_ckpch, r200_ckpch, h, a)

    print("\nApplying spatial cuts ...")
    ism = apply_radial_cut(dust, ctr_ckpch, box, h, a,
                           r_min_pkpc=0.0, r_max_pkpc=r_ism)
    cgm = apply_radial_cut(dust, ctr_ckpch, box, h, a,
                           r_min_pkpc=r_ism, r_max_pkpc=r200_pkpc)

    if ism["a_nm"].size == 0:
        sys.exit("ERROR: no ISM dust — check centering")
    if cgm["a_nm"].size == 0:
        sys.exit("ERROR: no CGM dust — check R200")

    print("\nBuilding histograms ...")
    print("  ISM:")
    bc, dn_c, dn_s         = build_histogram(ism, h, n_bins=args.n_bins)
    print("  CGM:")
    _,  dn_c_cgm, dn_s_cgm = build_histogram(cgm, h, n_bins=args.n_bins)

    print("\nComputing dust masses ...")
    ism_masses = dust_masses_msun(ism, h)
    cgm_masses = dust_masses_msun(cgm, h)

    print(f"  ISM masses [carb, sil, total] Msun: "
          f"{ism_masses[0]:.3e}, {ism_masses[1]:.3e}, {ism_masses[2]:.3e}")
    print(f"  CGM masses [carb, sil, total] Msun: "
          f"{cgm_masses[0]:.3e}, {cgm_masses[1]:.3e}, {cgm_masses[2]:.3e}")

    print("\nRendering figure ...")
    make_figure(bc, dn_c, dn_s, dn_c_cgm, dn_s_cgm,
                ism_masses=ism_masses,
                cgm_masses=cgm_masses,
                r_ism=r_ism, r200=r200_pkpc,
                output_path=str(out_path),
                run_label=run_label)


if __name__ == "__main__":
    main()
