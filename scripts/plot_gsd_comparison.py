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
consistent with all other CosmicGrain analysis scripts. The primary halo
is selected by stellar mass argmax across ALL catalog chunks, with a
hardcoded override for 2048^3 where the most massive DM group is a
merger-inflated neighbour.

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
    consistent with snap_overview and all other analysis scripts.
    """
    snap_num   = int(snap_num_str)
    groups_dir = Path(output_dir) / f"groups_{snap_num:03d}"

    ref = get_halo569_reference(output_dir)
    if ref is None:
        raise RuntimeError("get_halo569_reference returned None")

    halo = get_halo569(groups_dir, snap_num, ref, verbose=True)
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
            # Periodic boundary wrap
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
                r_ism, r200, output_path, run_label):
    """
    4-panel 2x2 figure.
      Rows    : top = dn/da,  bottom = a^4 dn/da
      Columns : left = carbonaceous,  right = silicate
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5),
                             sharex=True, sharey="row",
                             constrained_layout=True)

    a_ref     = np.logspace(np.log10(1.0), np.log10(200.0), 600)
    ISM_COLOR = "#7B2FBE"
    CGM_COLOR = "#009E73"

    ism_label = f"ISM  ($r < {r_ism:.0f}$ pkpc)"
    cgm_label = f"CGM  ({r_ism:.0f}--{r200:.0f} pkpc)"

    ref_raw = {
        "MRN (1977)":   dict(ls="--", color="0.2",     lw=1.8,
                             sil=normalise(mrn_dn(a_ref)),
                             carb=normalise(mrn_dn(a_ref))),
        "W & D (2001)": dict(ls="-.", color="#E07B22",  lw=1.8,
                             sil=wd01_silicate_dn(a_ref),
                             carb=wd01_carbonaceous_dn(a_ref)),
        "THEMIS":       dict(ls=":",  color="#3A7EC9",  lw=2.0,
                             sil=themis_silicate_dn(a_ref),
                             carb=themis_carbon_dn(a_ref)),
    }
    ref_a4 = {
        name: dict(ls=kw["ls"], color=kw["color"], lw=kw["lw"],
                   sil=normalise(a_ref**4 * kw["sil"]),
                   carb=normalise(a_ref**4 * kw["carb"]))
        for name, kw in ref_raw.items()
    }

    sim = {
        ("dn", "carb"): normalise(dn_c),
        ("dn", "sil") : normalise(dn_s),
        ("a4", "carb"): normalise(bc**4 * dn_c),
        ("a4", "sil") : normalise(bc**4 * dn_s),
    }
    sim_cgm = {
        ("dn", "carb"): normalise(dn_c_cgm),
        ("dn", "sil") : normalise(dn_s_cgm),
        ("a4", "carb"): normalise(bc**4 * dn_c_cgm),
        ("a4", "sil") : normalise(bc**4 * dn_s_cgm),
    }

    panels = [
        (axes[0, 0], "dn", "carb", "Carbonaceous", False),
        (axes[0, 1], "dn", "sil",  "Silicate",     False),
        (axes[1, 0], "a4", "carb", "Carbonaceous", False),
        (axes[1, 1], "a4", "sil",  "Silicate",     True),
    ]

    for ax, rep, comp, col_label, show_legend in panels:
        refs     = ref_raw if rep == "dn" else ref_a4
        ism_data = sim[(rep, comp)]
        cgm_data = sim_cgm[(rep, comp)]

        for name, kw in refs.items():
            y   = kw[comp]
            vis = np.isfinite(y) & (y > 0)
            ax.plot(a_ref[vis], y[vis],
                    ls=kw["ls"], color=kw["color"], lw=kw["lw"],
                    label=name, zorder=2)

        v = np.isfinite(cgm_data) & (cgm_data > 0)
        ax.step(bc[v], cgm_data[v], where="mid",
                color=CGM_COLOR, lw=2.2, zorder=4, label=cgm_label)
        ax.fill_between(bc[v], 1e-10, cgm_data[v],
                        step="mid", alpha=0.15, color=CGM_COLOR, zorder=3)

        v = np.isfinite(ism_data) & (ism_data > 0)
        ax.step(bc[v], ism_data[v], where="mid",
                color=ISM_COLOR, lw=2.2, zorder=6, label=ism_label)
        ax.fill_between(bc[v], 1e-10, ism_data[v],
                        step="mid", alpha=0.15, color=ISM_COLOR, zorder=5)

        ax.text(0.97, 0.97, col_label,
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 220.0)
        ax.grid(True, which="minor", color="0.93", lw=0.3)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:g}"))

        if show_legend:
            ax.legend(loc="lower left", fontsize=8.5)

    axes[0, 0].set_ylim(1e-8, 3.0)
    axes[1, 0].set_ylim(5e-5, 3.0)
    axes[0, 0].set_ylabel(r"$\mathrm{dn/da}$")
    axes[1, 0].set_ylabel(r"$a^4\,\mathrm{dn/da}$")
    axes[1, 0].set_xlabel(r"Grain radius (nm)")
    axes[1, 1].set_xlabel(r"Grain radius (nm)")

    fig.suptitle(f"CosmicGrain GSD — {run_label}  $z=0$", fontsize=12)

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
    if args.output is None:
        out_path = Path(f"gsd_comparison_{run_label}_snap{snap_num}.png")
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

    print("\nRendering figure ...")
    make_figure(bc, dn_c, dn_s, dn_c_cgm, dn_s_cgm,
                r_ism=r_ism, r200=r200_pkpc,
                output_path=str(out_path),
                run_label=run_label)
    print("Done.")


if __name__ == "__main__":
    main()
