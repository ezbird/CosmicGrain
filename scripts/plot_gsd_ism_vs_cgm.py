#!/usr/bin/env python3
"""
plot_gsd_ism_vs_cgm.py
======================
Compare the grain-size distribution (GSD) of ISM dust vs. CGM dust
within Halo 569 at z=0, using the a^4 dn/da representation.

Spatial selections (physical kpc from halo centre):
  ISM :  r < R_ISM_MAX       (default 20 pkpc)
  CGM :  R_ISM_MAX < r < R200

MRN (dn/da proportional to a^{-3.5}) is shown as a neutral reference.

The comparison tests whether CosmicGrain reproduces the expected
physical result: ISM grains larger (coagulation/growth in dense gas),
CGM grains smaller (sputtering-dominated, no accretion growth).

Field conventions (from snap_io.cc):
  Dust        : PartType6
  GrainRadius : HDF5 values in nm  (raw range 1.0-200.0)
  CarbFrac    : CarbonFraction [0, 1]
  Coordinates : comoving kpc/h
  Masses      : 1e10 Msun/h
  HubbleParam : f["Parameters"].attrs["HubbleParam"]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import glob
import sys
from pathlib import Path

# -- CONFIGURATION ------------------------------------------------------------

SNAP_DIR   = Path("../S10_output_1024/snapdir_049")
GROUPS_DIR = Path("../S10_output_1024/groups_049")
SNAP_BASE  = "snapshot_049"
CAT_BASE   = "fof_subhalo_tab_049"

R_ISM_MAX  = 20.0   # ISM cut: r < R_ISM_MAX pkpc
# CGM extends from R_ISM_MAX to R200 (read from catalog at runtime)

N_BINS     = 40     # log-spaced bins across [1, 200] nm
OUTPUT_FIG = Path("gsd_ism_vs_cgm_S10_1024_z0.png")

# grain bulk densities (g/cm^3)
RHO_SIL  = 3.5
RHO_CARB = 2.2

MSUN_G   = 1.989e33

# -- I/O HELPERS --------------------------------------------------------------

def glob_files(directory, basename):
    pattern = str(directory / f"{basename}*.hdf5")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files


def read_header(snap_files):
    with h5py.File(snap_files[0], "r") as f:
        h = float(f["Parameters"].attrs["HubbleParam"])
        a = float(f["Header"].attrs["Time"])
        z = float(f["Header"].attrs["Redshift"])
    return h, a, z


def read_r200(cat_files, h, a):
    """Read R200 and halo centre from multi-chunk FOF catalog."""
    pos_l, r200_l = [], []
    for fpath in cat_files:
        with h5py.File(fpath, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_l.append(grp["GroupPos"][:])
            r200_l.append(grp["Group_R_Crit200"][:])
    if not pos_l:
        raise ValueError("No groups found in FOF catalog")
    all_pos  = np.concatenate(pos_l,  axis=0)
    all_r200 = np.concatenate(r200_l, axis=0)
    cen      = all_pos[0].astype(float)   # ckpc/h
    r200     = float(all_r200[0])         # ckpc/h
    r200_pk  = r200 / h * a               # physical kpc
    print(f"  Halo centre  : [{cen[0]:.1f}, {cen[1]:.1f}, {cen[2]:.1f}] ckpc/h")
    print(f"  R_Crit200    : {r200:.1f} ckpc/h  ({r200_pk:.1f} pkpc)")
    return cen, r200_pk


def read_dust(snap_files):
    """Read all PartType6 particles. Returns pos (ckpc/h), mass, a_nm, fc."""
    pos_l, mass_l, arad_l, fc_l = [], [], [], []
    for fpath in snap_files:
        with h5py.File(fpath, "r") as f:
            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]
            if len(pt6["Masses"]) == 0:
                continue
            pos_l.append(pt6["Coordinates"][:])
            mass_l.append(pt6["Masses"][:])
            arad_l.append(pt6["GrainRadius"][:])   # already in nm
            fc_l.append(pt6["CarbonFraction"][:])
    pos  = np.concatenate(pos_l,  axis=0)
    mass = np.concatenate(mass_l, axis=0)
    a_nm = np.concatenate(arad_l, axis=0)   # nm
    fc   = np.concatenate(fc_l,   axis=0)
    print(f"  Total PartType6 : {len(a_nm):,}")
    return dict(pos=pos, mass=mass, a_nm=a_nm, fc=fc)


def radial_cut(dust, cen_ckpch, h, a, r_min_pkpc, r_max_pkpc):
    """Select dust in r_min_pkpc < r < r_max_pkpc (physical kpc)."""
    pos_pk = dust["pos"] / h * a
    cen_pk = cen_ckpch   / h * a
    dr     = np.linalg.norm(pos_pk - cen_pk[None, :], axis=1)
    mask   = (dr >= r_min_pkpc) & (dr < r_max_pkpc)
    label  = f"r=[{r_min_pkpc:.0f},{r_max_pkpc:.0f}) pkpc"
    print(f"  {label:30s}: {mask.sum():,} / {len(mask):,} ({100*mask.mean():.1f}%)")
    return {k: v[mask] for k, v in dust.items()}


# -- HISTOGRAM ----------------------------------------------------------------

def build_histogram(dust, h, a_min_nm=1.0, a_max_nm=200.0, n_bins=N_BINS):
    """
    Number-weighted a^4 dn/da histograms for carbon and silicate.
    Each superparticle contributes N_grains = m / (rho_mix * 4pi/3 * a^3).
    Split: N_carb = f_C * N_grains,  N_sil = (1-f_C) * N_grains.
    Returns (bin_centres_nm, a4_carb, a4_sil).
    """
    MASS_CGS = 1e10 * MSUN_G / h
    NM_CM    = 1e-7

    m_g     = dust["mass"] * MASS_CGS
    a_nm    = dust["a_nm"]
    fc      = dust["fc"]
    a_cm    = a_nm * NM_CM
    rho_mix = fc * RHO_CARB + (1.0 - fc) * RHO_SIL
    N_grains = m_g / (rho_mix * (4.0/3.0) * np.pi * a_cm**3)
    N_carb   = fc       * N_grains
    N_sil    = (1.0-fc) * N_grains

    bins = np.logspace(np.log10(a_min_nm), np.log10(a_max_nm), n_bins + 1)
    da   = np.diff(bins)
    bc   = np.sqrt(bins[:-1] * bins[1:])

    h_c, _ = np.histogram(a_nm, bins=bins, weights=N_carb)
    h_s, _ = np.histogram(a_nm, bins=bins, weights=N_sil)

    with np.errstate(invalid="ignore", divide="ignore"):
        a4_c = np.where(h_c > 0, bc**4 * h_c / da, np.nan)
        a4_s = np.where(h_s > 0, bc**4 * h_s / da, np.nan)

    return bc, a4_c, a4_s


# -- REFERENCE MODEL ----------------------------------------------------------

def normalise(arr):
    finite = arr[np.isfinite(arr) & (arr > 0)]
    return arr / finite.max() if len(finite) else arr


def mrn_a4(a, a_min=5.0, a_max=200.0):
    """MRN: dn/da proportional to a^{-3.5}  =>  a^4 dn/da proportional to a^{0.5}"""
    return normalise(np.where((a >= a_min) & (a <= a_max), a**0.5, np.nan))


# -- FIGURE -------------------------------------------------------------------

def make_figure(bc, ism_c, ism_s, cgm_c, cgm_s, r200_pkpc):
    """
    Two-panel figure (carb top, sil bottom).
    ISM and CGM a^4 dn/da overlaid; MRN as reference.
    """
    plt.style.use("cosmicgrain.mplstyle")
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 8.0),
                             sharex=True, constrained_layout=True)
    ax_top, ax_bot = axes

    a_ref = np.logspace(np.log10(1.0), np.log10(200.0), 600)
    mrn   = mrn_a4(a_ref)

    ISM_COLOR = "#7B2FBE"   # purple  (consistent with GSD comparison figure)
    CGM_COLOR = "#E07B22"   # orange

    panels = [
        (ax_top, ism_c, cgm_c, "Carbonaceous", False),
        (ax_bot, ism_s, cgm_s, "Silicate",     True),
    ]

    for ax, ism_data, cgm_data, label_text, show_legend in panels:

        # -- MRN reference (lightest layer) -----------------------------------
        vis = np.isfinite(mrn) & (mrn > 0)
        ax.plot(a_ref[vis], mrn[vis],
                ls="--", color="0.55", lw=1.6, zorder=2, label="MRN (1977)")

        # -- ISM histogram ----------------------------------------------------
        ism_norm = normalise(ism_data)
        v_ism    = np.isfinite(ism_norm) & (ism_norm > 0)
        ax.step(bc[v_ism], ism_norm[v_ism], where="mid",
                color=ISM_COLOR, lw=2.2, zorder=5,
                label=f"ISM  ($r < {R_ISM_MAX:.0f}$ pkpc)")
        ax.fill_between(bc[v_ism], 1e-5, ism_norm[v_ism],
                        step="mid", alpha=0.15, color=ISM_COLOR, zorder=4)

        # -- CGM histogram ----------------------------------------------------
        cgm_norm = normalise(cgm_data)
        v_cgm    = np.isfinite(cgm_norm) & (cgm_norm > 0)
        ax.step(bc[v_cgm], cgm_norm[v_cgm], where="mid",
                color=CGM_COLOR, lw=2.2, zorder=5,
                label=f"CGM  ({R_ISM_MAX:.0f} pkpc $< r <$ {r200_pkpc:.0f} pkpc)")
        ax.fill_between(bc[v_cgm], 1e-5, cgm_norm[v_cgm],
                        step="mid", alpha=0.15, color=CGM_COLOR, zorder=4)

        # -- panel label (top right) ------------------------------------------
        ax.text(0.97, 0.97, label_text,
                transform=ax.transAxes, ha="right", va="top", fontsize=11)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 220.0)
        ax.set_ylim(5e-5, 3.0)
        ax.set_ylabel(r"$\mathrm{a}^4\,\mathrm{dn/da}$ (normalised)")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.grid(True, which="minor", color="0.93", lw=0.3)

        if show_legend:
            ax.legend(loc="lower left", fontsize=9)

    ax_bot.set_xlabel(r"Grain radius (nm)")

    r200_str = (f"$R_{{200c}} = {r200_pkpc:.0f}\\,\\mathrm{{pkpc}}$"
                if r200_pkpc else "")
    annot = (r"Halo 569  $|$  S10 $1024^3$  $|$  $z=0$  $|$  "
             + r200_str + "\n"
             r"Number-weighted $a^4\,dn/da$; each curve normalised to peak")
    fig.text(0.5, -0.02, annot, ha="center", va="top",
             fontsize=8.0, color="0.45",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.8))

    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {OUTPUT_FIG}")
    plt.show()


# -- MAIN ---------------------------------------------------------------------

def main():
    print("-" * 60)
    print("CosmicGrain GSD  ISM vs CGM  |  S10 1024^3  |  z=0")
    print("-" * 60)

    snap_files = glob_files(SNAP_DIR,   SNAP_BASE)
    cat_files  = glob_files(GROUPS_DIR, CAT_BASE)
    print(f"Snapshot chunks : {len(snap_files)}")
    print(f"Catalog chunks  : {len(cat_files)}")

    h, a, z = read_header(snap_files)
    print(f"h={h:.4f}  a={a:.6f}  z={z:.4f}")

    print("\nReading FOF catalog ...")
    cen_ckpch, r200_pkpc = read_r200(cat_files, h, a)

    print("\nReading PartType6 ...")
    dust = read_dust(snap_files)

    print("\nApplying spatial cuts ...")
    ism  = radial_cut(dust, cen_ckpch, h, a,
                      r_min_pkpc=0.0,
                      r_max_pkpc=R_ISM_MAX)
    cgm  = radial_cut(dust, cen_ckpch, h, a,
                      r_min_pkpc=R_ISM_MAX,
                      r_max_pkpc=r200_pkpc)

    if ism["a_nm"].size == 0:
        sys.exit("ERROR: no ISM dust particles found")
    if cgm["a_nm"].size == 0:
        sys.exit("ERROR: no CGM dust particles found")

    print("\nBuilding histograms ...")
    bc,      ism_c, ism_s = build_histogram(ism, h)
    bc_cgm,  cgm_c, cgm_s = build_histogram(cgm, h)
    # bin centres are identical (same grid); cgm bc not needed separately

    print(f"\n  ISM median grain size (carb) : "
          f"{bc[np.nanargmax(ism_c)]:.1f} nm  (peak of a^4 dn/da)")
    print(f"  CGM median grain size (carb) : "
          f"{bc[np.nanargmax(cgm_c)]:.1f} nm  (peak of a^4 dn/da)")
    print(f"  ISM median grain size (sil)  : "
          f"{bc[np.nanargmax(ism_s)]:.1f} nm")
    print(f"  CGM median grain size (sil)  : "
          f"{bc[np.nanargmax(cgm_s)]:.1f} nm")

    print("\nRendering figure ...")
    make_figure(bc, ism_c, ism_s, cgm_c, cgm_s, r200_pkpc)


if __name__ == "__main__":
    main()
