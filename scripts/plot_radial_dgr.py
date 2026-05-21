#!/usr/bin/env python3
"""
plot_radial_dgr.py
------------------
Publication-quality radial dust-to-gas (D/G) and dust-to-metals (D/Z)
profiles for the CosmicGrain simulation ladder at z=0.

Two panels stacked vertically (D/G top, D/Z bottom), sharing the x-axis.
Profiles are computed in 5 kpc spherical shells in physical kpc.

Usage:
    python plot_radial_dgr.py --res 1024
    python plot_radial_dgr.py --res 1024 --runs S0 S4 S10
    python plot_radial_dgr.py --res 1024 --r-max 50
    python plot_radial_dgr.py --res 1024 --output myplot.png
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# ─────────────────────────────────────────────────────────────────────────────
# Run styling
# ─────────────────────────────────────────────────────────────────────────────
RUN_CONFIGS = {
    "S0":  {"label": "S0: Creation only",            "color": "#888888"},
    "S1":  {"label": "S1: + Cooling",                "color": "#1f77b4"},
    "S2":  {"label": "S2: + Drag",                   "color": "#ff7f0e"},
    "S3":  {"label": "S3: + Astration",              "color": "#2ca02c"},
    "S4":  {"label": "S4: + Thermal sputtering",     "color": "#d62728"},
    "S5":  {"label": "S5: + Grain growth",           "color": "#9467bd"},
    "S6":  {"label": "S6: + Clumping factor",        "color": "#8c564b"},
    "S7":  {"label": "S7: + SN shock destruction",   "color": "#e377c2"},
    "S8":  {"label": "S8: + Coagulation",            "color": "#17becf"},
    "S9":  {"label": "S9: + Shattering",             "color": "#bcbd22"},
    "S10": {"label": "S10: + Rad. pressure (full)",  "color": "#000000"},
}

FIGDIR     = "dust_figures"
RESOLUTION = 512
R_MAX_DEFAULT = 50.0          # physical kpc
BIN_WIDTH     = 5.0           # physical kpc — matches the ~5 kpc spacing in target

os.makedirs(FIGDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────

def metallicity_gradient(r_kpc,
                         r0=8.0,
                         grad_dex_per_kpc=-0.04):
    """
    MW/M31-like radial metallicity gradient.

    Typical observed gradients:
        -0.03 to -0.05 dex/kpc
    """
    return 10**(grad_dex_per_kpc * (r_kpc - r0))

def z_relative(r_kpc, r0=8.0, grad=-0.04):
    return 10**(grad * (r_kpc - r0))

def remy_ruyer_dgr(r_kpc, r0=8.0, dgr0=0.01, grad=-0.04, alpha=1.5):
    zrel = z_relative(r_kpc, r0=r0, grad=grad)
    return dgr0 * zrel**alpha

def remy_ruyer_dtz(r_kpc, r0=8.0, dtz0=0.5, grad=-0.04, alpha=1.5):
    zrel = z_relative(r_kpc, r0=r0, grad=grad)
    return dtz0 * zrel**(alpha - 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot / catalog utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run):
    output_dir = f"{run}_output_{RESOLUTION}"
    if not os.path.isdir(output_dir):
        return []
    seen, bases = set(), []
    for snapdir in sorted(glob.glob(os.path.join(output_dir, "snapdir_*"))):
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
    import h5py
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
    import h5py
    defaults = dict(h=0.7, a=1.0)
    for suffix in [".0.hdf5", ".hdf5"]:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, "r") as hf:
                    attrs = hf["Header"].attrs
                    return dict(
                        h = float(attrs.get("HubbleParam", defaults["h"])),
                        a = float(attrs.get("Time",        defaults["a"])),
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


def get_halo_center_r200(run, snap_base):
    import h5py
    m = re.search(r"snapshot_(\d+)$", snap_base)
    if not m: return None, None
    snap_num   = m.group(1)
    groups_dir = os.path.join(f"{run}_output_{RESOLUTION}", f"groups_{snap_num}")
    cats = sorted(glob.glob(
        os.path.join(groups_dir, f"fof_subhalo_tab_{snap_num}.*.hdf5")))
    if not cats: return None, None
    try:
        with h5py.File(cats[0], "r") as hf:
            if "Group" not in hf: return None, None
            grp = hf["Group"]
            if "GroupPos" not in grp or grp["GroupPos"].shape[0] == 0:
                return None, None
            ctr  = grp["GroupPos"][0].astype(float)
            r200 = float(grp["Group_M_Crit200"][0]) \
                   if "Group_M_Crit200" in grp else None
    except Exception as e:
        print(f"  [{run}] catalog error: {e}")
        return None, None
    return ctr, r200

def mw_gradient_dtz(r_kpc, r_sun=8.0, dtz_sun=0.45,
                    grad_dex_per_kpc=-0.04):
    """
    Radial D/Z gradient for a MW/M31-mass galaxy.
    At near-solar Z the gradient is flat; at sub-solar Z (outer disk)
    D/G ∝ Z^~1.5 so D/Z ∝ Z^0.5 ~ follows metallicity gradient.
    Approximated here as a single exponential decline.
    Chiang+2021 (M31), Draine+2014, Rémy-Ruyer+2014.
    """
    return dtz_sun * 10**(grad_dex_per_kpc * (r_kpc - r_sun))

# ─────────────────────────────────────────────────────────────────────────────
# Particle loader
# ─────────────────────────────────────────────────────────────────────────────

def load_particles(snap_base, ctr, rmax_com, ptype, fields):
    """Load PartType{ptype} within rmax_com (comoving kpc/h) of ctr."""
    import h5py
    key    = f"PartType{ptype}"
    result = {f: [] for f in fields}
    result["pos"] = []
    for fname in subfiles(snap_base):
        try:
            with h5py.File(fname, "r") as hf:
                if key not in hf: continue
                pt   = hf[key]
                pos  = pt["Coordinates"][:]
                r    = np.linalg.norm(pos - ctr, axis=1)
                mask = r < rmax_com
                if not mask.any(): continue
                result["pos"].append(pos[mask])
                for f in fields:
                    if f in pt:
                        result[f].append(pt[f][:][mask])
        except Exception as e:
            print(f"  load_particles(type={ptype}): {e}")
    if not result["pos"]:
        return None
    out = {k: np.concatenate(v) for k, v in result.items() if v}
    if "pos" not in out:
        return None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Radial profile computation
# ─────────────────────────────────────────────────────────────────────────────

def running_median(r, y, r_bins):
    """
    Compute median of y in each r_bin.  Returns bin-centre array and
    median array; bins with fewer than MIN_PTS particles return NaN.
    """
    MIN_PTS = 5
    r_cen   = 0.5 * (r_bins[:-1] + r_bins[1:])
    y_med   = np.full(len(r_cen), np.nan)
    for i, (lo, hi) in enumerate(zip(r_bins[:-1], r_bins[1:])):
        mask = (r >= lo) & (r < hi) & np.isfinite(y) & (y > 0)
        if mask.sum() >= MIN_PTS:
            y_med[i] = np.median(y[mask])
    return r_cen, y_med


def compute_radial_profiles(snap_base, run, r_max_pkpc, r_bins_pkpc):
    """
    Bin-averaged D/G and D/Z in physical kpc shells.

    Returns (r_cen, dgr, dtz, r200_pkpc) or (None, None, None, None).
    """
    hdr     = read_header(snap_base)
    to_pkpc = hdr["a"] / hdr["h"]
    ctr, r200_com = get_halo_center_r200(run, snap_base)
    if ctr is None:
        return None, None, None, None

    r200_pkpc = r200_com * to_pkpc
    rmax_com  = r_max_pkpc / to_pkpc

    print(f"  [{run}] R200={r200_pkpc:.1f} pkpc  loading...")

    gas  = load_particles(snap_base, ctr, rmax_com, 0, ["Masses", "Metallicity"])
    dust = load_particles(snap_base, ctr, rmax_com, 6, ["Masses"])
    if gas is None or dust is None:
        print(f"  [{run}] missing gas or dust"); return None, None, None, None

    r_gas  = np.linalg.norm(gas["pos"]  - ctr, axis=1) * to_pkpc
    r_dust = np.linalg.norm(dust["pos"] - ctr, axis=1) * to_pkpc

    # Bin by shell
    gas_m,  _ = np.histogram(r_gas,  bins=r_bins_pkpc, weights=gas["Masses"])
    dust_m, _ = np.histogram(r_dust, bins=r_bins_pkpc, weights=dust["Masses"])

    Z = gas["Metallicity"]
    if Z.ndim == 2: Z = Z[:, 0]
    metal_m, _ = np.histogram(r_gas, bins=r_bins_pkpc,
                               weights=gas["Masses"] * Z)

    r_cen  = 0.5 * (r_bins_pkpc[:-1] + r_bins_pkpc[1:])
    med_gm = np.nanmedian(gas_m[gas_m > 0]) if np.any(gas_m > 0) else 1.0
    good   = gas_m > 0.01 * med_gm

    with np.errstate(invalid="ignore", divide="ignore"):
        dgr = np.where(good & (gas_m   > 0), dust_m / gas_m,   np.nan)
        dtz = np.where(good & (metal_m > 0), dust_m / metal_m, np.nan)

    return r_cen, dgr, dtz, r200_pkpc


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_radial_profiles(runs, r_max_pkpc=None, output_path=None):

    r_max  = r_max_pkpc if r_max_pkpc is not None else R_MAX_DEFAULT
    r_bins = np.arange(0, r_max + BIN_WIDTH, BIN_WIDTH)

    # ── Figure: 2 stacked panels sharing x-axis ──────────────────────────────
    fig, (ax_dgr, ax_dtz) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    fig.patch.set_facecolor("white")

    handles_runs = []   # accumulate for legend

    for run in runs:
        cfg   = RUN_CONFIGS.get(run, {})
        color = cfg.get("color", "black")
        label = cfg.get("label", run)

        if run == "S10":
            linestyle = "-"
            lw = 2.4
            alpha = 1.0
        else:
            linestyle = ":"
            lw = 1.5
            alpha = 0.75

        snaps = find_snapshots(run)
        if not snaps:
            print(f"  [{run}] no snapshots"); continue

        snap_base, dz = find_snap_near_z(snaps, 0.0)
        if dz > 0.2:
            print(f"  [{run}] no z~0 snap (dz={dz:.2f})"); continue

        r_cen, dgr, dtz, r200_p = compute_radial_profiles(
            snap_base, run, r_max, r_bins)
        if r_cen is None: continue

        kw = dict(
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            marker="o",
            markersize=4.0,
            markeredgewidth=0.0,
        )

        good_dgr = np.isfinite(dgr) & (dgr > 0)
        good_dtz = np.isfinite(dtz) & (dtz > 0)

        if good_dgr.any():
            ax_dgr.plot(r_cen[good_dgr], dgr[good_dgr], **kw, label=label)
        if good_dtz.any():
            ax_dtz.plot(r_cen[good_dtz], dtz[good_dtz], **kw)

        handles_runs.append(
            plt.Line2D([0], [0], color=color, lw=1.8,
                       marker="o", markersize=4.5, markeredgewidth=0.0,
                       label=label)
        )

    # ── D/G reference: exponential MW gradient ────────────────────────────────
    r_ref   = np.linspace(0.5, r_max, 300)
    dgr_mw  = remy_ruyer_dgr(r_ref)
    h_mw, = ax_dgr.plot(r_ref, dgr_mw, color="black", lw=2.0, ls="--",
                         zorder=5)
    handles_runs.append(h_mw)

    # ── D/Z reference: Mattsson+2012 / Zafar+2013 band ───────────────────────
    zeta_G  = 0.5
    r_ref  = np.linspace(0.5, r_max, 300)
    dtz_mw = remy_ruyer_dtz(r_ref)
    ax_dtz.plot(r_ref, dtz_mw, color='k', lw=2.0, ls='--', zorder=4,
                label='MW/M31-like (Rémy-Ruyer et al 2014)')
    # Plot the reference line and capture the handle
    dtz_ref_line, = ax_dtz.plot(r_ref, dtz_mw, color='k', lw=2.0, ls='--', zorder=4)

    # Standalone legend in D/Z panel for the reference only
    ax_dtz.legend(
        handles=[dtz_ref_line],
        labels=['MW/M31 gradient (Rémy-Ruyer et al 2014)'],
        fontsize=8, loc='upper right',
        framealpha=0.9, edgecolor='0.8',
    )

    # ── Axes: D/G ─────────────────────────────────────────────────────────────
    ax_dgr.set_yscale("log")
    ax_dgr.set_ylim(1e-5, 5e-1)
    ax_dgr.set_ylabel("Dust-to-Gas Ratio (D/G)", fontsize=12)
    #ax_dgr.set_title(f"Radial Dust Profiles at $z \\approx 0.0$  "
    #                 f"({RESOLUTION}$^3$)", fontsize=12, pad=8)
    ax_dgr.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax_dgr.yaxis.set_major_formatter(
        matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax_dgr.grid(True, alpha=0.25, which="both", lw=0.5)

    # ── Axes: D/Z ─────────────────────────────────────────────────────────────
    ax_dtz.set_yscale("log")
    ax_dtz.set_ylim(1e-3, 10)
    ax_dtz.set_ylabel("Dust-to-Metals Ratio (D/Z)", fontsize=12)
    ax_dtz.set_xlabel("Galactocentric Radius (kpc)", fontsize=12)
    ax_dtz.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax_dtz.yaxis.set_major_formatter(
        matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax_dtz.grid(True, alpha=0.25, which="both", lw=0.5)

    # ── Shared x-axis ─────────────────────────────────────────────────────────
    ax_dtz.set_xlim(0, r_max)
    ax_dtz.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax_dtz.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))

    # ── Legend: D/G panel, two columns ───────────────────────────────────────
    ax_dgr.legend(
        handles=handles_runs,
        fontsize=7.5, loc="lower left",
        framealpha=0.9, edgecolor="0.8",
        ncol=1, handlelength=2.2,
        labelspacing=0.35, borderpad=0.6,
    )

    plt.tight_layout()

    out = output_path or os.path.join(FIGDIR, f"radial_dg_dz_{RESOLUTION}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+",
                        default=["S0","S1","S2","S3","S4","S5",
                                 "S6","S7","S8","S9","S10"],
                        help="Ladder runs to plot (default: all)")
    parser.add_argument("--res", type=int, default=512,
                        help="Resolution label (default: 512)")
    parser.add_argument("--r-max", type=float, default=None,
                        help=f"Outer radius in physical kpc "
                             f"(default: {R_MAX_DEFAULT} pkpc)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    global RESOLUTION
    RESOLUTION = args.res

    print(f"\nRuns:       {args.runs}")
    print(f"Resolution: {RESOLUTION}^3")
    print(f"r_max:      {args.r_max or R_MAX_DEFAULT} pkpc")
    print(f"Bin width:  {BIN_WIDTH} pkpc\n")

    plot_radial_profiles(args.runs, r_max_pkpc=args.r_max,
                         output_path=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
