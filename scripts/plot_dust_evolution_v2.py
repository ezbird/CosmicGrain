#!/usr/bin/env python3
"""
plot_dust_evolution.py
----------------------
2×2 figure for Halo 569, all panels vs redshift (shared x-axis per column):

  Top-left:     M_dust and M_star vs redshift   [M_sun, log]
  Top-right:    SFR within R_200 vs redshift    [M_sun/yr, log]
  Bottom-left:  D/G within R_200 vs redshift    [log]
  Bottom-right: D/Z within R_200 vs redshift    [log]

Net dust rate panel available via --show-rate flag (replaces SFR).

Encoding:
  Color   = resolution / run
  Solid   = M_dust or main ratio
  Dashed  = M_star (mass panel only)

Usage:
    python plot_dust_evolution.py ../S10_output_1024/
    python plot_dust_evolution.py ../S10_output_512/ ../S10_output_1024/ \
        --labels '$512^3$' '$1024^3$' --output convergence.png
"""

import re
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "axes.linewidth":   0.9,
})

COLORS = ["#2a9d8f", "#2980b9", "#e67e22", "#8e44ad", "#c0392b"]

SOLAR_MASS = 1.989e33
YR_IN_SEC  = 3.156e7
GYR_IN_SEC = 3.156e16

# ─────────────────────────────────────────────────────────────────────────────
def _smooth(arr, sigma=1.5):
    ok = np.isfinite(arr) & (arr > 0)
    if ok.sum() < 3:
        return arr
    out = arr.copy()
    out[ok] = gaussian_filter1d(arr[ok].astype(float), sigma=sigma)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Cosmology
# ─────────────────────────────────────────────────────────────────────────────
def z_to_age_gyr(z, h=0.6774, Om=0.3089):
    H0 = h * 100.0 * 1e3 / 3.0857e22
    age, _ = quad(lambda zp: 1.0 / ((1+zp)*np.sqrt(Om*(1+zp)**3+(1-Om))),
                  z, np.inf)
    return age / H0 / GYR_IN_SEC

def z_arr_to_age(z_arr):
    return np.array([z_to_age_gyr(z) for z in z_arr])

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_snapshots(output_dir):
    output_dir = Path(output_dir)
    entries = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m: continue
        snap_num   = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        cats = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) \
               if groups_dir.exists() else []
        if not cats: continue
        snaps = sorted(snapdir.glob("snap_*.hdf5")) + \
                sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snaps: continue
        entries.append((snap_num, str(snaps[0]), str(cats[0])))
    return entries

def get_header(snap_file):
    with h5py.File(snap_file, "r") as f:
        hdr = f["Header"].attrs
        p   = f["Parameters"].attrs
        return float(hdr["Redshift"]), float(p["HubbleParam"]), float(hdr["BoxSize"])

def get_unit(snap_file, key, fallback):
    with h5py.File(snap_file, "r") as f:
        p = f.get("Parameters", {})
        v = p.attrs.get(key, None) if hasattr(p, "attrs") else None
    return float(v) if v is not None else fallback

def _chunks(snap_file):
    p = Path(snap_file)
    stem = re.sub(r"\.\d+$", "", p.stem)
    fs = sorted(p.parent.glob(f"{stem}*.hdf5"))
    return fs or [p]

def read_primary_group(catalog_file):
    p    = Path(catalog_file)
    base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{base}*.hdf5")) or [p]
    pos_l, r200_l, sfr_l = [], [], []
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f: continue
            g = f["Group"]
            if "GroupPos" not in g or len(g["GroupPos"]) == 0: continue
            pos_l.append(g["GroupPos"][:])
            r200_l.append(g["Group_R_Crit200"][:])
            sfr_l.append(g["GroupSFR"][:] if "GroupSFR" in g
                         else np.zeros(len(g["GroupPos"])))
    if not pos_l: return None
    return dict(center   = np.concatenate(pos_l)[0],
                r200     = float(np.concatenate(r200_l)[0]),
                sfr_code = float(np.concatenate(sfr_l)[0]))

def load_mass_sum(snap_file, center, r, ptype):
    total = 0.0
    for chunk in _chunks(snap_file):
        with h5py.File(chunk, "r") as f:
            key = f"PartType{ptype}"
            if key not in f: continue
            pt  = f[key]
            box = float(f["Header"].attrs["BoxSize"])
            mt  = f["Header"].attrs.get("MassTable", None)
            coords = pt["Coordinates"][:]
            if "Masses" in pt:
                masses = pt["Masses"][:]
            elif mt is not None and mt[ptype] > 0:
                masses = np.full(len(coords), float(mt[ptype]))
            else:
                continue
            dx = coords - center
            dx -= box * np.round(dx / box)
            total += masses[np.sqrt((dx**2).sum(1)) <= r].sum()
    return total

def load_gas_mass_and_metals(snap_file, center, r):
    mg, mm = 0.0, 0.0
    for chunk in _chunks(snap_file):
        with h5py.File(chunk, "r") as f:
            if "PartType0" not in f: continue
            pt  = f["PartType0"]
            box = float(f["Header"].attrs["BoxSize"])
            coords = pt["Coordinates"][:]
            masses = pt["Masses"][:]
            dx = coords - center
            dx -= box * np.round(dx / box)
            mask = np.sqrt((dx**2).sum(1)) <= r
            if not mask.any(): continue
            mg += masses[mask].sum()
            if "Metallicity" in pt:
                Z = pt["Metallicity"][:]
                if Z.ndim == 2: Z = Z[:, 0]
                mm += (masses[mask] * Z[mask]).sum()
    return mg, mm

# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot
# ─────────────────────────────────────────────────────────────────────────────
def process_snapshot(snap_num, snap_file, catalog_file, unit_mass_g, unit_time_s):
    z, h, box = get_header(snap_file)
    grp = read_primary_group(catalog_file)
    if grp is None or grp["r200"] <= 0: return None

    center, r200 = grp["center"], grp["r200"]
    m_dust = load_mass_sum(snap_file, center, r200, 6)
    m_star = load_mass_sum(snap_file, center, r200, 4)
    if m_star <= 0: return None

    m_gas, m_gasmet = load_gas_mass_and_metals(snap_file, center, r200)
    code_msun = unit_mass_g / SOLAR_MASS

    sfr = grp["sfr_code"] * code_msun / (unit_time_s / YR_IN_SEC)
    dgr = m_dust / m_gas if m_gas > 0 else np.nan
    dtz = m_dust / (m_gasmet + m_dust) if (m_gasmet + m_dust) > 0 else np.nan

    print(f"  snap {snap_num:03d}  z={z:.3f}  "
          f"logMs={np.log10(m_star*code_msun):.2f}  "
          f"logMd={np.log10(max(m_dust*code_msun, 1)):.2f}  "
          f"SFR={sfr:.1f}  D/G={dgr:.3e}  D/Z={dtz:.3f}")

    return dict(z=z, m_star=m_star*code_msun, m_dust=m_dust*code_msun,
                sfr=sfr, dgr=dgr, dtz=dtz)

def run_simulation(output_dir, skip_every=1):
    snaps = find_snapshots(output_dir)
    if not snaps:
        raise RuntimeError(f"No snapshots with catalogs in {output_dir}")
    print(f"  Found {len(snaps)} snapshots")
    um = get_unit(snaps[0][1], "UnitMass_in_g",  1.989e43)
    ut = get_unit(snaps[0][1], "UnitTime_in_s",  3.0857e16)
    rows = []
    for i, (sn, sf, cf) in enumerate(snaps):
        if i % skip_every != 0: continue
        r = process_snapshot(sn, sf, cf, um, ut)
        if r is not None: rows.append(r)
    if not rows: raise RuntimeError("No valid snapshots")
    rows.sort(key=lambda r: -r["z"])
    return (np.array([r["z"]      for r in rows]),
            np.array([r["m_star"] for r in rows]),
            np.array([r["m_dust"] for r in rows]),
            np.array([r["sfr"]    for r in rows]),
            np.array([r["dgr"]    for r in rows]),
            np.array([r["dtz"]    for r in rows]))

def net_dust_rate(z_arr, m_dust_arr):
    age_yr = z_arr_to_age(z_arr) * 1e9
    rate   = np.empty_like(m_dust_arr)
    rate[1:-1] = (m_dust_arr[2:] - m_dust_arr[:-2]) / (age_yr[2:] - age_yr[:-2])
    rate[0]  = (m_dust_arr[1]  - m_dust_arr[0])  / (age_yr[1]  - age_yr[0])
    rate[-1] = (m_dust_arr[-1] - m_dust_arr[-2]) / (age_yr[-1] - age_yr[-2])
    return rate

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def make_plot(runs, output_path, show_rate=False):
    """
    2x2 layout.  Panels share x-axis within each column (left col = col 0,
    right col = col 1).  All four panels get the same redshift range.

    Top-left:     M_dust + M_star
    Top-right:    SFR  (or net rate if show_rate)
    Bottom-left:  D/G
    Bottom-right: D/Z
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.07, "wspace": 0.32},
    )
    ax_mass, ax_sfr  = axes[0]
    ax_dgr,  ax_dtz  = axes[1]

    lw = 2.6

    for i, (label, z, m_star, m_dust, sfr, dgr, dtz) in enumerate(runs):
        c = COLORS[i % len(COLORS)]

        m_dust_s = _smooth(m_dust)
        m_star_s = _smooth(m_star)
        sfr_s    = _smooth(np.where(sfr > 0, sfr, np.nan))
        dgr_s    = _smooth(np.where(dgr > 0, dgr, np.nan))
        dtz_s    = _smooth(np.where(dtz > 0, dtz, np.nan))

        kw = dict(color=c, lw=lw)

        # ── Top-left: masses ─────────────────────────────────────────────────
        lbl = label if len(runs) > 1 else r"$M_\mathrm{dust}$"
        ax_mass.semilogy(z, m_dust_s, **kw, label=lbl)
        ax_mass.semilogy(z, m_star_s, color=c, lw=lw*0.6, ls="--", alpha=0.5)

        # ── Top-right: SFR or net rate ───────────────────────────────────────
        if show_rate:
            rate = net_dust_rate(z, m_dust_s)
            ax_sfr.plot(z, rate, **kw)
            ax_sfr.fill_between(z, rate, 0, where=rate>=0,
                                color=c, alpha=0.15)
            ax_sfr.fill_between(z, rate, 0, where=rate<0,
                                color=c, alpha=0.08, hatch="////",
                                linewidth=0)
        else:
            ax_sfr.semilogy(z, np.where(sfr_s>0, sfr_s, np.nan), **kw)

        # ── Bottom-left: D/G ─────────────────────────────────────────────────
        gd = np.isfinite(dgr_s) & (dgr_s > 0)
        if gd.any():
            ax_dgr.semilogy(z[gd], dgr_s[gd], **kw)

        # ── Bottom-right: D/Z ────────────────────────────────────────────────
        gz = np.isfinite(dtz_s) & (dtz_s > 0)
        if gz.any():
            ax_dtz.semilogy(z[gz], dtz_s[gz], **kw)

    # ── Axis formatting ───────────────────────────────────────────────────────
    z_max = max(r[1].max() for r in runs)
    x_max = min(z_max + 0.3, 7.0)
    for ax in axes.flat:
        ax.set_xlim(x_max, 0)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="0.88", lw=0.5)
        ax.grid(True, which="minor", color="0.93", lw=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    # x labels on bottom row only
    for ax in axes[1]:
        ax.set_xlabel("Redshift", fontsize=13)

    ax_mass.set_ylabel(r"Mass ($M_\odot$)")
    ax_mass.set_ylim(5e5, 5e10)

    if show_rate:
        ax_sfr.axhline(0, color="0.5", lw=0.8, ls="--")
        ax_sfr.set_yscale("linear")
        ax_sfr.set_ylim(-1, 1)
        ax_sfr.set_ylabel(r"$\dot{M}_\mathrm{dust}\ (M_\odot\,\mathrm{yr}^{-1})$",
                          fontsize=11)
        ax_sfr.text(0.96, 0.88, "net growth",      transform=ax_sfr.transAxes,
                    fontsize=10, color="0.4", ha="right", style="italic")
        ax_sfr.text(0.96, 0.05, "net destruction", transform=ax_sfr.transAxes,
                    fontsize=10, color="0.4", ha="right", style="italic")
    else:
        ax_sfr.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)")

    ax_dgr.set_ylabel(r"$D/G$")
    ax_dgr.set_ylim(1e-4, 5e-2)
    ax_dgr.axhline(0.01, color="0.5", lw=1.4, ls="--", alpha=0.7)
    ax_dgr.text(0.04, 0.08, r"MW $D/G \approx 0.01$",
                transform=ax_dgr.transAxes, fontsize=9.5, color="0.45")

    ax_dtz.set_ylabel(r"$D/Z$")
    ax_dtz.set_ylim(3e-2, 1.5)
    ax_dtz.axhline(0.5, color="0.5", lw=1.4, ls="--", alpha=0.7)
    ax_dtz.text(0.04, 0.88, r"MW $D/Z \approx 0.5$",
                transform=ax_dtz.transAxes, fontsize=9.5, color="0.45")

    # ── Lookback time on top of left column ───────────────────────────────────
    ax2 = ax_mass.twiny()
    ax2.set_xlim(ax_mass.get_xlim())
    z_ticks = np.array([0, 0.5, 1, 2, 3, 4, 5, 6])
    valid   = z_ticks[z_ticks <= x_max]
    ax2.set_xticks(valid)
    ax2.set_xticklabels([f"{z_to_age_gyr(z):.1f}" for z in valid], fontsize=10)
    ax2.set_xlabel("Lookback time (Gyr)", fontsize=11)
    ax2.invert_xaxis()

    # Matching lookback axis on top of right column
    ax3 = ax_sfr.twiny()
    ax3.set_xlim(ax_sfr.get_xlim())
    ax3.set_xticks(valid)
    ax3.set_xticklabels([f"{z_to_age_gyr(z):.1f}" for z in valid], fontsize=10)
    ax3.invert_xaxis()
    ax3.tick_params(axis="x", labeltop=False)   # hide labels, keep ticks

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    if len(runs) == 1:
        c0 = COLORS[0]
        ax_mass.legend(handles=[
            Line2D([0],[0], color=c0, lw=lw,
                   label=r"$M_\mathrm{dust}$"),
            Line2D([0],[0], color=c0, lw=lw*0.6, ls="--", alpha=0.5,
                   label=r"$M_*$"),
        ], fontsize=11, loc="lower right", framealpha=0.88)
    else:
        handles = [Line2D([0],[0], color=COLORS[j], lw=lw, label=runs[j][0])
                   for j in range(len(runs))]
        handles += [
            Line2D([0],[0], color="0.35", lw=lw,
                   label=r"$M_\mathrm{dust}$ (solid)"),
            Line2D([0],[0], color="0.35", lw=lw*0.6, ls="--", alpha=0.5,
                   label=r"$M_*$ (faint dashed)"),
        ]
        ax_mass.legend(handles=handles, fontsize=10.5, loc="lower right",
                       framealpha=0.88, ncol=1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dirs", nargs="+")
    parser.add_argument("--labels",     nargs="*", default=None)
    parser.add_argument("--skip-every", type=int,  default=1)
    parser.add_argument("--output",     default="dust_evolution.png")
    parser.add_argument("--show-rate",  action="store_true",
                        help="Replace SFR panel with net dust rate panel")
    args = parser.parse_args()

    n      = len(args.output_dirs)
    labels = args.labels if args.labels else [Path(d).name
                                              for d in args.output_dirs]
    if len(labels) != n:
        parser.error("--labels must match number of output_dirs")

    runs = []
    for d, lbl in zip(args.output_dirs, labels):
        print(f"\nProcessing: {lbl}")
        data = run_simulation(d, skip_every=args.skip_every)
        runs.append((lbl, *data))

    make_plot(runs, args.output, show_rate=args.show_rate)

if __name__ == "__main__":
    main()
