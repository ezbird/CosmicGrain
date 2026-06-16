#!/usr/bin/env python3
"""
plot_gsd.py — Grain Size Distribution for CosmicGrain
======================================================
Plots dn/da vs grain radius for PartType6 dust superparticles,
broken down by environment (ISM / Inner CGM / Outer CGM) and
grain type (silicate / carbon), with MRN reference slope.

Two panels:
  Left  — dM/d(log a) vs log(a)  [mass distribution; MRN ∝ a^+0.5]
  Right — dn/da × a^3.5 vs log(a) [compensated; MRN = flat line]

Usage
-----
  python plot_gsd.py --snapdir /path/to/snapshots --snapnum 13 \\
                     --groupdir /path/to/groups    --outdir ./figures

  # Compare two resolution runs side-by-side:
  python plot_gsd.py --snapdir /snap1 --snapnum 13 --groupdir /grp1 \\
                     --compare  /snap2 13 /grp2    --outdir ./figures

Author : CosmicGrain team
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# ── Style ─────────────────────────────────────────────────────────────────────
# Try the CosmicGrain style sheet; fall back to a clean default
_STYLE_CANDIDATES = [
    "cosmicgrain.mplstyle",
    os.path.expanduser("~/cosmicgrain/cosmicgrain.mplstyle"),
    os.path.join(os.path.dirname(__file__), "cosmicgrain.mplstyle"),
]
for _s in _STYLE_CANDIDATES:
    if os.path.isfile(_s):
        plt.style.use(_s)
        break

# ── Physical constants (CGS) ───────────────────────────────────────────────────
RHO_SIL_CGS  = 2.4          # silicate grain bulk density  [g cm⁻³]
RHO_CARB_CGS = 2.0          # carbon grain bulk density    [g cm⁻³]
NM_TO_CM     = 1.0e-7        # 1 nm in cm
NM_TO_UM     = 1.0e-3        # 1 nm in μm

# ── Environment thresholds ────────────────────────────────────────────────────
# DustTemperature is grain equilibrium temperature (~10-100 K), NOT gas temperature.
# Environment classification therefore uses purely radial apertures.
R_ISM_PKPC       = 25.0    # pkpc — ISM aperture (stellar surface density break)
R_INNER_CGM_PKPC = 100.0   # pkpc — inner CGM boundary

# ── GSD binning ───────────────────────────────────────────────────────────────
A_MIN_NM     = 0.3           # nm  — smallest expected grain after sputtering
A_MAX_NM     = 800.0         # nm  — largest expected grain after coagulation
N_BINS       = 40            # log-spaced bins across that range

# ── Colour palette ─────────────────────────────────────────────────────────────
ENV_COLORS = {
    "ISM":       "#4C9BE8",   # blue   r < 25 pkpc
    "Inner CGM": "#E8A84C",   # amber  25 – 100 pkpc
    "Outer CGM": "#A64CE8",   # purple 100 pkpc – R200c
    "All ISM":   "#2ECC71",   # green  (used when not splitting environments)
}
TYPE_LS = {
    "All":       "-",
    "Silicate":  "--",
    "Carbon":    ":",
}

# ══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def snap_files(snapdir: str, snapnum: int) -> list:
    """
    Return sorted list of all HDF5 files for this snapshot.
    Handles both single-file (snapshot_049.hdf5) and
    multi-file (snapshot_049.0.hdf5, snapshot_049.1.hdf5, ...) layouts.
    """
    import glob
    for base_fmt in [f"snapshot_{snapnum:03d}", f"snapshot_{snapnum:04d}",
                     f"snap_{snapnum:03d}"]:
        # Multi-file first (snapshot_049.0.hdf5 etc.)
        pattern = os.path.join(snapdir, f"{base_fmt}.*.hdf5")
        hits = sorted(glob.glob(pattern),
                      key=lambda p: int(p.rsplit(".", 2)[-2]))
        if hits:
            return hits
        # Single file
        p = os.path.join(snapdir, f"{base_fmt}.hdf5")
        if os.path.isfile(p):
            return [p]
    raise FileNotFoundError(
        f"No snapshot found in {snapdir} for snapnum={snapnum}"
    )


def snap_path(snapdir: str, snapnum: int) -> str:
    """Return path to the first (or only) snapshot file — used for header reads."""
    return snap_files(snapdir, snapnum)[0]


def group_path(groupdir: str, snapnum: int) -> str:
    """Return path to the .0 file of the SUBFIND group catalog (multi-file aware)."""
    import glob
    for base in [groupdir, os.path.join(groupdir, f"groups_{snapnum:03d}")]:
        for fmt in [f"fof_subhalo_tab_{snapnum:03d}", f"fof_subhalo_tab_{snapnum:04d}"]:
            # Multi-file: grab the .0 file
            p0 = os.path.join(base, f"{fmt}.0.hdf5")
            if os.path.isfile(p0):
                return p0
            # Single file
            p  = os.path.join(base, f"{fmt}.hdf5")
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"No group catalog found in {groupdir} for snapnum={snapnum}"
    )


def load_header(snapfile: str) -> dict:
    """Read snapshot header + Parameters block."""
    with h5py.File(snapfile, "r") as f:
        hdr = dict(f["Header"].attrs)
        # HubbleParam lives in Parameters, not Header
        h = float(f["Parameters"].attrs["HubbleParam"])
        omega_m = float(f["Parameters"].attrs.get("Omega0", 0.3))
    a_scale = float(hdr["Time"])
    z       = 1.0 / a_scale - 1.0
    return dict(h=h, a=a_scale, z=z, omega_m=omega_m,
                boxsize=float(hdr["BoxSize"]))


def load_main_halo(groupfile: str, h: float, a: float) -> dict:
    """
    Load main halo (index 0) from SUBFIND catalog.
    Returns center and R_Crit200 in physical kpc.

    If you prefer to use halo_utils, replace this function body with:
        import halo_utils
        halo = halo_utils.load_halo(snapdir, snapnum)
        return dict(center=halo['center'], r200c=halo['r200c'])
    """
    with h5py.File(groupfile, "r") as f:
        pos   = f["Group/GroupPos"][0]          # comoving kpc/h
        r200c = f["Group/Group_R_Crit200"][0]   # comoving kpc/h
    # Convert to physical kpc
    center_pkpc = pos   * a / h
    r200c_pkpc  = r200c * a / h
    return dict(center=center_pkpc, r200c=r200c_pkpc)


def periodic_delta(dx: np.ndarray, boxsize_pkpc: float) -> np.ndarray:
    """Wrap coordinate differences into [−L/2, L/2]."""
    dx = dx.copy()
    dx[dx >  0.5 * boxsize_pkpc] -= boxsize_pkpc
    dx[dx < -0.5 * boxsize_pkpc] += boxsize_pkpc
    return dx


def load_dust(snapdir: str, snapnum: int, h: float, a: float) -> dict:
    """
    Load PartType6 dust superparticles, concatenating across all split files.

    Field notes (from snapshot_049.N.hdf5 /PartType6):
        GrainRadius      — grain radius in NANOMETRES (physical)
        GrainType        — 0 = silicate, 1 = carbon   # VERIFY if different
        CarbonFraction   — continuous carbon mass fraction
        DustTemperature  — ambient gas temperature at dust position [K]
        Masses           — code units (10^10 Msun / h)
        Coordinates      — comoving kpc / h
    """
    files = snap_files(snapdir, snapnum)

    coords_list, masses_list, a_nm_list = [], [], []
    gtype_list, carb_list, temp_list    = [], [], []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "PartType6" not in f:
                continue          # this sub-file has no dust particles
            pt = f["PartType6"]
            if pt["Coordinates"].shape[0] == 0:
                continue
            coords_list.append(pt["Coordinates"][:])
            masses_list.append(pt["Masses"][:])
            a_nm_list.append(pt["GrainRadius"][:])
            gtype_list.append(pt["GrainType"][:])
            carb_list.append(pt["CarbonFraction"][:])
            temp_list.append(pt["DustTemperature"][:])

    if not coords_list:
        raise RuntimeError("No PartType6 dust particles found in any snapshot file.")

    coords    = np.concatenate(coords_list,  axis=0)
    masses    = np.concatenate(masses_list)
    a_nm      = np.concatenate(a_nm_list)
    gtype     = np.concatenate(gtype_list)
    carb_frac = np.concatenate(carb_list)
    dust_T    = np.concatenate(temp_list)

    # Coordinates → physical kpc
    coords_pkpc = coords * a / h
    # Masses → physical Msun  (×10^10, divide h)
    masses_msun = masses * 1.0e10 / h
    # GrainRadius: physical nm → μm for plotting
    a_um = a_nm * NM_TO_UM

    return dict(
        coords    = coords_pkpc,
        masses    = masses_msun,
        a_nm      = a_nm,
        a_um      = a_um,
        gtype     = gtype.astype(int),
        carb_frac = carb_frac,
        dust_T    = dust_T,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Environment masks
# ══════════════════════════════════════════════════════════════════════════════

def make_env_masks(dust: dict, halo: dict, boxsize_pkpc: float) -> dict:
    """
    Return boolean masks over dust particles for each environment.

    DustTemperature is grain equilibrium temperature (~10-100 K), not gas temperature,
    so environments are defined by radial apertures only.
      ISM       : r < R_ISM_PKPC (25 pkpc — stellar surface density break)
      Inner CGM : R_ISM_PKPC < r < R_INNER_CGM_PKPC (25–100 pkpc)
      Outer CGM : R_INNER_CGM_PKPC < r < R200c
    """
    dx = periodic_delta(dust["coords"] - halo["center"], boxsize_pkpc)
    r  = np.linalg.norm(dx, axis=1)           # physical kpc

    in_ism       = r < R_ISM_PKPC
    in_inner_cgm = (r >= R_ISM_PKPC) & (r < R_INNER_CGM_PKPC)
    in_outer_cgm = (r >= R_INNER_CGM_PKPC) & (r < halo["r200c"])

    return {
        "ISM":       in_ism,
        "Inner CGM": in_inner_cgm,
        "Outer CGM": in_outer_cgm,
        "All ISM":   in_ism,
        "All":       in_ism | in_inner_cgm | in_outer_cgm,
    }


def make_type_masks(dust: dict) -> dict:
    """Grain-type masks.  VERIFY: 0=silicate, 1=carbon."""
    g = dust["gtype"]
    return {
        "Silicate": g == 0,
        "Carbon":   g == 1,
        "All":      np.ones(len(g), dtype=bool),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GSD computation
# ══════════════════════════════════════════════════════════════════════════════

LOG_BINS = np.linspace(
    np.log10(A_MIN_NM * NM_TO_UM),
    np.log10(A_MAX_NM * NM_TO_UM),
    N_BINS + 1,
)   # log10(a/μm) bin edges

A_CENTERS_UM = 10 ** (0.5 * (LOG_BINS[:-1] + LOG_BINS[1:]))   # μm
D_LOG_A      = LOG_BINS[1] - LOG_BINS[0]                       # Δlog₁₀(a)


def compute_gsd(a_um: np.ndarray, masses: np.ndarray,
                gtype_int: np.ndarray,
                mask: np.ndarray) -> dict:
    """
    Compute GSD quantities for particles selected by `mask`.

    Returns dict with arrays over A_CENTERS_UM:
        dm_dloga     — dM/d(log₁₀ a)   [Msun]
        dn_da_sil    — dn/da for silicate component  [arb units]
        dn_da_carb   — dn/da for carbon component    [arb units]
        dn_da_total  — total dn/da  (mass-weighted, grain density corrected)
        slope_fit    — (slope, stderr) from log–log linear fit to dn/da
        n_particles  — number of superparticles in mask
    """
    a  = a_um[mask]
    m  = masses[mask]
    gt = gtype_int[mask]

    if len(a) < 5:
        return None

    log_a = np.log10(a)

    # ── dM/d(log a) ─────────────────────────────────────────────────────────
    dm_dloga, _ = np.histogram(log_a, bins=LOG_BINS, weights=m)
    dm_dloga   /= D_LOG_A    # normalize per unit log-interval

    # ── dn/da: account for grain type density ──────────────────────────────
    # Physical grains represented by one superparticle:
    #   N_grains = M_sp / ((4π/3) ρ_grain a³)
    # Contribution to dn/da in each bin = N_grains / (bin width in a-space)
    # In log-space, da = a × ln(10) × d(log a), so we weight by m / (ρ a³ a × ln10 × D_log_a)
    # We keep arbitrary normalization (relative shape only).

    rho = np.where(gt == 1, RHO_CARB_CGS, RHO_SIL_CGS)          # g/cm³
    a_cm = a * NM_TO_CM * 1e3   # μm → nm → cm
    weight_dn = m / (rho * a_cm**3)   # ∝ N_grains per superparticle

    dn_dloga, _  = np.histogram(log_a, bins=LOG_BINS, weights=weight_dn)

    # dn/da = dn/d(log a) / (a × ln 10)  — keep a_um scale; slope is what matters
    dn_da = dn_dloga / (A_CENTERS_UM * np.log(10) * D_LOG_A)

    # Per-type dn/da
    sil_mask  = gt == 0
    carb_mask = gt == 1

    def _dn_da_type(type_mask):
        if type_mask.sum() == 0:
            return np.zeros(N_BINS)
        w = weight_dn.copy()
        w[~type_mask] = 0.0
        h, _ = np.histogram(log_a, bins=LOG_BINS, weights=w)
        return h / (A_CENTERS_UM * np.log(10) * D_LOG_A)

    dn_da_sil  = _dn_da_type(sil_mask)
    dn_da_carb = _dn_da_type(carb_mask)

    # ── Power-law slope fit ─────────────────────────────────────────────────
    good = dn_da > 0
    slope, intercept, r, p, se = (np.nan,)*5
    if good.sum() >= 3:
        slope, intercept, r, p, se = linregress(
            np.log10(A_CENTERS_UM[good]), np.log10(dn_da[good])
        )

    return dict(
        dm_dloga    = dm_dloga,
        dn_da       = dn_da,
        dn_da_sil   = dn_da_sil,
        dn_da_carb  = dn_da_carb,
        slope       = slope,
        slope_se    = se,
        n_particles = int(mask.sum()),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def mrn_reference(a_um: np.ndarray, norm_at: float = 0.1) -> np.ndarray:
    """MRN power law dn/da ∝ a^{-3.5}, normalised at norm_at μm."""
    return (a_um / norm_at) ** (-3.5)


def plot_gsd(dust: dict, env_masks: dict, type_masks: dict,
             halo: dict, snap_label: str, outdir: str,
             show_composition: bool = True) -> None:
    """
    Produce the two-panel GSD figure.

    Left panel  : dM/d(log a) — where is the dust mass?
    Right panel : compensated dn/da × a^{3.5} — deviations from MRN visible
                  as bumps/deficits; MRN = horizontal line
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_mass, ax_comp = axes

    # ── Reference MRN lines ─────────────────────────────────────────────────
    mrn_line = mrn_reference(A_CENTERS_UM)

    for ax in axes:
        ax.set_xlabel(r"$\log_{10}(a\,/\,\mu\mathrm{m})$")
        ax.set_xlim(np.log10(A_MIN_NM * NM_TO_UM),
                    np.log10(A_MAX_NM * NM_TO_UM))
        # Minor ticks every 0.1 dex
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # Left panel: dM/d(log a) ────────────────────────────────────────────────
    ax_mass.set_ylabel(r"$\mathrm{d}M\,/\,\mathrm{d}\log_{10}a\;[M_\odot]$")
    ax_mass.set_yscale("log")
    ax_mass.set_title("Mass distribution")

    # MRN mass distribution: dM/d(log a) ∝ a^{+0.5}  (annotated, not shown as data)
    ax_mass.annotate(r"MRN: $\propto a^{+0.5}$", xy=(0.05, 0.92),
                     xycoords="axes fraction", fontsize=9,
                     color="0.5", style="italic")

    for env_name in ["ISM", "Inner CGM", "Outer CGM"]:
        mask = env_masks[env_name]
        gsd  = compute_gsd(dust["a_um"], dust["masses"],
                           dust["gtype"], mask)
        if gsd is None:
            continue

        color = ENV_COLORS[env_name]
        valid = gsd["dm_dloga"] > 0
        ax_mass.plot(np.log10(A_CENTERS_UM[valid]),
                     gsd["dm_dloga"][valid],
                     color=color, lw=2,
                     label=fr"{env_name}  (N={gsd['n_particles']:,})")

        if show_composition:
            # Silicate dashed, carbon dotted — same colour
            for comp_key, comp_arr in [("Silicate", gsd["dn_da_sil"]),
                                       ("Carbon",   gsd["dn_da_carb"])]:
                # For mass panel we need dm_dloga per type — recompute inline
                sub_mask = mask & (type_masks[comp_key])
                gsd_sub  = compute_gsd(dust["a_um"], dust["masses"],
                                       dust["gtype"], sub_mask)
                if gsd_sub is None:
                    continue
                v = gsd_sub["dm_dloga"] > 0
                ax_mass.plot(np.log10(A_CENTERS_UM[v]),
                             gsd_sub["dm_dloga"][v],
                             color=color,
                             ls=TYPE_LS[comp_key],
                             lw=1.2, alpha=0.6)

    ax_mass.legend(fontsize=8, framealpha=0.4)

    # Add MRN slope guide arrow on left panel
    # anchor at 0.05 μm, show rise corresponding to +0.5/dex
    x0, x1 = -1.5, -0.5   # log μm
    y0 = 1.0               # arbitrary — will be scaled to plot
    _ylim = ax_mass.get_ylim()
    if _ylim[0] > 0 and _ylim[1] > 0:
        y_guide = 10 ** (0.5 * (np.log10(_ylim[0]) + np.log10(_ylim[1])))
        ax_mass.annotate("",
            xy=(x1, y_guide * 10**( 0.5*(x1-x0))),
            xytext=(x0, y_guide),
            arrowprops=dict(arrowstyle="-", color="0.6",
                            linestyle="dashed", lw=1.2))

    # Right panel: compensated dn/da × a^{3.5} ──────────────────────────────
    ax_comp.set_ylabel(
        r"$(\mathrm{d}n/\mathrm{d}a)\times a^{3.5}$ (arbitrary units)"
    )
    ax_comp.set_yscale("log")
    ax_comp.set_title("Compensated (MRN = flat)")

    # MRN reference: flat line — normalise to ISM All result if available
    _norm_ref = None

    for env_name in ["ISM", "Inner CGM", "Outer CGM"]:
        mask = env_masks[env_name]
        gsd  = compute_gsd(dust["a_um"], dust["masses"],
                           dust["gtype"], mask)
        if gsd is None:
            continue

        color = ENV_COLORS[env_name]

        # Compensated: dn/da × a^{3.5}  (in μm units; slope is what matters)
        compensated = gsd["dn_da"] * A_CENTERS_UM**3.5
        valid = compensated > 0

        if _norm_ref is None and valid.sum() > 0:
            _norm_ref = np.nanmedian(compensated[valid])

        ax_comp.plot(np.log10(A_CENTERS_UM[valid]),
                     compensated[valid],
                     color=color, lw=2,
                     label=fr"{env_name}  $\alpha={gsd['slope']:.2f}\pm{gsd['slope_se']:.2f}$")

        if show_composition:
            for comp_key, arr in [("Silicate", gsd["dn_da_sil"]),
                                   ("Carbon",   gsd["dn_da_carb"])]:
                comp_c = arr * A_CENTERS_UM**3.5
                v = comp_c > 0
                ax_comp.plot(np.log10(A_CENTERS_UM[v]), comp_c[v],
                             color=color, ls=TYPE_LS[comp_key],
                             lw=1.2, alpha=0.6,
                             label=f"  {comp_key}" if env_name == "ISM" else None)

    # MRN flat reference
    if _norm_ref is not None:
        ax_comp.axhline(_norm_ref, color="0.55", lw=1.5,
                        linestyle="--", label="MRN reference")
        ax_comp.annotate(r"$\alpha = -3.5$ (MRN)", xy=(0.97, _norm_ref),
                         xycoords=("axes fraction", "data"),
                         xytext=(-5, 6), textcoords="offset points",
                         ha="right", va="bottom", fontsize=8, color="0.55")

    # Composition legend entries
    if show_composition:
        from matplotlib.lines import Line2D
        _extra = [
            Line2D([0], [0], color="0.4", ls="--", lw=1.2, label="Silicate"),
            Line2D([0], [0], color="0.4", ls=":",  lw=1.2, label="Carbon"),
        ]
        ax_comp.legend(handles=ax_comp.get_legend_handles_labels()[0] + _extra,
                       labels =ax_comp.get_legend_handles_labels()[1]  +
                               ["Silicate", "Carbon"],
                       fontsize=7.5, framealpha=0.4)
    else:
        ax_comp.legend(fontsize=8, framealpha=0.4)

    # ── Shared formatting ────────────────────────────────────────────────────
    z_label = snap_label  # e.g. "z = 0.0  |  1024³  |  S10"
    fig.suptitle(f"Grain Size Distribution — {z_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(outdir, exist_ok=True)
    label_safe = z_label.replace(" ", "_").replace("|", "").replace(".", "p")
    outfile = os.path.join(outdir, f"gsd_{label_safe}.pdf")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"[plot_gsd] Saved → {outfile}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Physics-ladder comparison variant
# ══════════════════════════════════════════════════════════════════════════════

def plot_gsd_ladder(run_list: list, outdir: str) -> None:
    """
    Single-panel compensated GSD comparing multiple runs / S-rungs.

    run_list : list of dicts, each with keys:
        snapfile, groupfile, label, color
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel(r"$\log_{10}(a\,/\,\mu\mathrm{m})$")
    ax.set_ylabel(r"$(\mathrm{d}n/\mathrm{d}a)\times a^{3.5}$ (arb. units)")
    ax.set_yscale("log")
    ax.set_xlim(np.log10(A_MIN_NM * NM_TO_UM),
                np.log10(A_MAX_NM * NM_TO_UM))
    ax.set_title("GSD Physics Ladder — ISM (r < 25 pkpc)")

    _norm_ref = None

    for run in run_list:
        hdr  = load_header(run["snapfile"])
        halo = load_main_halo(run["groupfile"], hdr["h"], hdr["a"])
        dust = load_dust(run["snapdir"], run["snapnum"], hdr["h"], hdr["a"])

        boxsize = hdr["boxsize"] * hdr["a"] / hdr["h"]   # physical kpc
        env_masks  = make_env_masks(dust, halo, boxsize)
        type_masks = make_type_masks(dust)

        ism_mask = env_masks["All ISM"]
        gsd = compute_gsd(dust["a_um"], dust["masses"],
                          dust["gtype"], ism_mask)
        if gsd is None:
            print(f"[plot_gsd] WARNING: no ISM dust for {run['label']}")
            continue

        compensated = gsd["dn_da"] * A_CENTERS_UM**3.5
        valid = compensated > 0
        if _norm_ref is None and valid.sum() > 0:
            _norm_ref = np.nanmedian(compensated[valid])

        ax.plot(np.log10(A_CENTERS_UM[valid]),
                compensated[valid],
                color=run.get("color", None), lw=2,
                label=fr"{run['label']}  $\alpha={gsd['slope']:.2f}$")

    if _norm_ref is not None:
        ax.axhline(_norm_ref, color="0.55", lw=1.4,
                   linestyle="--", label="MRN reference")

    ax.legend(fontsize=8, framealpha=0.4)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "gsd_ladder.pdf")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"[plot_gsd] Saved → {outfile}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CosmicGrain — grain size distribution plot"
    )
    p.add_argument("--snapdir",   required=True,
                   help="Directory containing snapshot_NNN.hdf5")
    p.add_argument("--snapnum",   required=True, type=int,
                   help="Snapshot number")
    p.add_argument("--groupdir",  required=True,
                   help="Directory containing SUBFIND group catalog")
    p.add_argument("--outdir",    default="./figures",
                   help="Output directory for PDFs")
    p.add_argument("--label",     default=None,
                   help="Run label for figure title (e.g. '1024³ S10')")
    p.add_argument("--no-composition", action="store_true",
                   help="Skip silicate/carbon sub-curves")
    p.add_argument("--ladder",    action="store_true",
                   help="If set, run the physics-ladder comparison "
                        "instead of environment breakdown (requires "
                        "--ladder-config)")
    p.add_argument("--ladder-config", default=None,
                   help="Path to a Python file defining 'RUNS' list "
                        "for ladder comparison")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ladder:
        if args.ladder_config is None:
            sys.exit("--ladder requires --ladder-config")
        import importlib.util
        spec = importlib.util.spec_from_file_location("lcfg", args.ladder_config)
        lcfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lcfg)
        plot_gsd_ladder(lcfg.RUNS, args.outdir)
        return

    # ── Single-snapshot run ──────────────────────────────────────────────────
    sfile = snap_path(args.snapdir, args.snapnum)
    gfile = group_path(args.groupdir, args.snapnum)

    print(f"[plot_gsd] Snapshot : {sfile}")
    print(f"[plot_gsd] Groups   : {gfile}")

    hdr  = load_header(sfile)
    halo = load_main_halo(gfile, hdr["h"], hdr["a"])
    dust = load_dust(args.snapdir, args.snapnum, hdr["h"], hdr["a"])

    print(f"[plot_gsd] z = {hdr['z']:.3f}  |  "
          f"N_dust = {len(dust['a_um']):,}  |  "
          f"R200c = {halo['r200c']:.1f} pkpc")

    boxsize_pkpc = hdr["boxsize"] * hdr["a"] / hdr["h"]
    env_masks  = make_env_masks(dust, halo, boxsize_pkpc)
    type_masks = make_type_masks(dust)

    # Report particle counts per environment
    for name, mask in env_masks.items():
        print(f"  {name:12s}: {mask.sum():>8,} particles")

    label = args.label or f"z={hdr['z']:.2f}  snap{args.snapnum:03d}"
    plot_gsd(dust, env_masks, type_masks, halo, label,
             args.outdir,
             show_composition=not args.no_composition)


if __name__ == "__main__":
    main()
