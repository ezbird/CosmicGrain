#!/usr/bin/env python3
"""
plot_dust_evolution.py
----------------------
Five-panel figure for Halo 569, all panels vs redshift (shared x-axis):

  Panel 1:  M_dust and M_star vs redshift   [M_sun, log]
  Panel 2:  SFR within R_200 vs redshift    [M_sun/yr, log]
  Panel 3:  Net dust growth rate dM_dust/dt  [M_sun/yr, linear ±1]
  Panel 4:  D/G within R_200 vs redshift    [log]
  Panel 5:  D/Z within R_200 vs redshift    [log]

All quantities within R_200 (SubFind Group_R_Crit200).
Same color per run across all panels.

Usage:
    python plot_dust_evolution.py ../S10_output_1024/
    python plot_dust_evolution.py ../S10_output_512/ ../S10_output_1024/ \\
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
from halo_utils import get_halo569_reference, get_halo569
plt.style.use('cosmicgrain.mplstyle')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SOLAR_MASS = 1.989e33
YR_IN_SEC  = 3.156e7
GYR_IN_SEC = 3.156e16
COLORS     = ["#2a9d8f", "#2980b9", "#e67e22", "#8e44ad", "#c0392b"]

from scipy.ndimage import gaussian_filter1d

def _smooth(arr, sigma=1.5):
    """Gaussian smooth, NaN-safe, in snapshot-index units."""
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
    from scipy.integrate import quad
    H0 = h * 100.0 * 1e3 / 3.0857e22
    age, _ = quad(lambda zp: 1.0 / ((1+zp) * np.sqrt(Om*(1+zp)**3 + (1-Om))),
                  z, np.inf)
    return age / H0 / GYR_IN_SEC

def z_arr_to_age(z_arr):
    return np.array([z_to_age_gyr(z) for z in z_arr])

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(output_dir):
    output_dir = Path(output_dir)
    entries = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m: continue
        snap_num   = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        cats       = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) \
                     if groups_dir.exists() else []
        if not cats: continue
        snaps = sorted(snapdir.glob("snap_*.hdf5")) + \
                sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snaps: continue
        entries.append((snap_num, str(snaps[0]), str(cats[0])))
    return entries


def get_header(snap_file):
    with h5py.File(snap_file, "r") as f:
        hdr    = f["Header"].attrs
        params = f["Parameters"].attrs
        return (float(hdr["Redshift"]),
                float(params["HubbleParam"]),
                float(hdr["BoxSize"]))


def get_unit_mass(snap_file):
    with h5py.File(snap_file, "r") as f:
        p  = f.get("Parameters", {})
        um = p.attrs.get("UnitMass_in_g", None) if hasattr(p, "attrs") else None
    return float(um) if um is not None else 1.989e43


def get_unit_time(snap_file):
    with h5py.File(snap_file, "r") as f:
        p  = f.get("Parameters", {})
        ut = p.attrs.get("UnitTime_in_s", None) if hasattr(p, "attrs") else None
    return float(ut) if ut is not None else 3.0857e16

# ─────────────────────────────────────────────────────────────────────────────
# SubFind catalog
# ─────────────────────────────────────────────────────────────────────────────

def read_primary_group(catalog_file):
    p    = Path(catalog_file)
    base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{base}*.hdf5")) or [p]
    pos_l, r200_l, m200_l, sfr_l = [], [], [], []
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f: continue
            g = f["Group"]
            if "GroupPos" not in g or len(g["GroupPos"]) == 0: continue
            pos_l.append(g["GroupPos"][:])
            r200_l.append(g["Group_R_Crit200"][:])
            m200_l.append(g["Group_M_Crit200"][:])
            sfr_l.append(g["GroupSFR"][:] if "GroupSFR" in g
                         else np.zeros(len(g["GroupPos"])))
    if not pos_l: return None
    return dict(
        center   = np.concatenate(pos_l)[0],
        r200     = float(np.concatenate(r200_l)[0]),
        m200     = float(np.concatenate(m200_l)[0]),
        sfr_code = float(np.concatenate(sfr_l)[0]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Particle loaders
# ─────────────────────────────────────────────────────────────────────────────

def _chunks(snap_file):
    p    = Path(snap_file)
    stem = re.sub(r"\.\d+$", "", p.stem)
    fs   = sorted(p.parent.glob(f"{stem}*.hdf5"))
    return fs or [p]


def load_mass_sum(snap_file, center_kph, r_kph, ptype):
    """Total mass (code units) for PartType{ptype} within r_kph."""
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
            dx = coords - center_kph
            dx -= box * np.round(dx / box)
            total += masses[np.sqrt((dx**2).sum(axis=1)) <= r_kph].sum()
    return total


def load_gas_mass_and_metals(snap_file, center_kph, r_kph):
    """
    Return (gas_mass, gas_metal_mass) in code units within r_kph.
    Metal mass is gas-phase only; caller adds dust mass for total metals.
    """
    m_gas = 0.0
    m_met = 0.0
    for chunk in _chunks(snap_file):
        with h5py.File(chunk, "r") as f:
            if "PartType0" not in f: continue
            pt  = f["PartType0"]
            box = float(f["Header"].attrs["BoxSize"])
            coords = pt["Coordinates"][:]
            masses = pt["Masses"][:]
            dx   = coords - center_kph
            dx  -= box * np.round(dx / box)
            mask = np.sqrt((dx**2).sum(axis=1)) <= r_kph
            if not mask.any(): continue
            m_gas += masses[mask].sum()
            if "Metallicity" in pt:
                Z = pt["Metallicity"][:]
                if Z.ndim == 2: Z = Z[:, 0]
                m_met += (masses[mask] * Z[mask]).sum()
    return m_gas, m_met

# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot extraction
# ─────────────────────────────────────────────────────────────────────────────

def process_snapshot(snap_num, snap_file, catalog_file, unit_mass_g, unit_time_s):
    z, h, box = get_header(snap_file)
    grp = read_primary_group(catalog_file)
    if grp is None or grp["r200"] <= 0: return None

    center = grp["center"]
    r200   = grp["r200"]

    m_dust_code = load_mass_sum(snap_file, center, r200, 6)
    m_star_code = load_mass_sum(snap_file, center, r200, 4)
    if m_star_code <= 0: return None

    m_gas_code, m_gasmet_code = load_gas_mass_and_metals(snap_file, center, r200)

    code_msun   = unit_mass_g / SOLAR_MASS
    sfr_msun_yr = grp["sfr_code"] * code_msun / (unit_time_s / YR_IN_SEC)

    # D/G and D/Z (dimensionless: code units cancel)
    dgr = m_dust_code / m_gas_code if m_gas_code > 0 else np.nan
    m_total_met = m_gasmet_code + m_dust_code   # gas-phase metals + dust
    dtz = m_dust_code / m_total_met if m_total_met > 0 else np.nan

    print(f"  snap {snap_num:03d}  z={z:.3f}  "
          f"logMs={np.log10(m_star_code * code_msun):.2f}  "
          f"logMd={np.log10(max(m_dust_code * code_msun, 1)):.2f}  "
          f"SFR={sfr_msun_yr:.2f}  D/G={dgr:.3e}  D/Z={dtz:.3f}")

    return dict(z=z,
                m_star=m_star_code * code_msun,
                m_dust=m_dust_code * code_msun,
                sfr=sfr_msun_yr, dgr=dgr, dtz=dtz)


def run_simulation(output_dir, skip_every=1):
    snaps = find_snapshots(output_dir)
    if not snaps:
        raise RuntimeError(f"No snapshots with catalogs in {output_dir}")
    print(f"  Found {len(snaps)} snapshots with catalogs")

    unit_mass_g = get_unit_mass(snaps[0][1])
    unit_time_s = get_unit_time(snaps[0][1])

    rows = []
    for i, (snap_num, snap_file, catalog_file) in enumerate(snaps):
        if i % skip_every != 0: continue
        r = process_snapshot(snap_num, snap_file, catalog_file,
                             unit_mass_g, unit_time_s)
        if r is not None:
            rows.append(r)

    if not rows:
        raise RuntimeError("No valid snapshots")

    rows.sort(key=lambda r: -r["z"])
    return (np.array([r["z"]      for r in rows]),
            np.array([r["m_star"] for r in rows]),
            np.array([r["m_dust"] for r in rows]),
            np.array([r["sfr"]    for r in rows]),
            np.array([r["dgr"]    for r in rows]),
            np.array([r["dtz"]    for r in rows]))

# ─────────────────────────────────────────────────────────────────────────────
# Net dust rate
# ─────────────────────────────────────────────────────────────────────────────

def net_dust_rate(z_arr, m_dust_arr):
    age_yr = z_arr_to_age(z_arr) * 1e9
    rate   = np.empty_like(m_dust_arr)
    rate[1:-1] = (m_dust_arr[2:]  - m_dust_arr[:-2]) / (age_yr[2:]  - age_yr[:-2])
    rate[0]    = (m_dust_arr[1]   - m_dust_arr[0])   / (age_yr[1]   - age_yr[0])
    rate[-1]   = (m_dust_arr[-1]  - m_dust_arr[-2])  / (age_yr[-1]  - age_yr[-2])
    return rate

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(runs, output_path):
    """
    runs: list of (label, z, m_star, m_dust, sfr, dgr, dtz)
    All 5 panels share the redshift x-axis.
    """
    fig, axes = plt.subplots(
        5, 1, figsize=(8, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    ax_dust, ax_sfr, ax_rate, ax_dgr, ax_dtz = axes

    z_ticks = np.array([0, 0.5, 1, 2, 3, 4, 5, 6])

    for i, (label, z, m_star, m_dust, sfr, dgr, dtz) in enumerate(runs):

        m_dust_s = _smooth(m_dust)
        m_star_s = _smooth(m_star)
        sfr_s    = _smooth(np.where(sfr > 0, sfr, np.nan))
        dgr_s    = _smooth(np.where(dgr > 0, dgr, np.nan))
        dtz_s    = _smooth(np.where(dtz > 0, dtz, np.nan))
        rate     = net_dust_rate(z, m_dust_s)   # ← from smoothed mass

        c    = COLORS[i % len(COLORS)]
        rate = net_dust_rate(z, m_dust)
        kw   = dict(color=c, lw=2.2)

        # Panel 1: M_dust (solid) and M_star (dashed)
        ax_dust.semilogy(z, m_dust_s, **kw,
                         label=label if len(runs) > 1 else r"$M_\mathrm{dust}$")
        ax_dust.semilogy(z, m_star_s, color=c, lw=1.5, ls="--", alpha=0.5)

        # Panel 2: SFR
        ax_sfr.semilogy(z, np.where(sfr_s > 0, sfr_s, np.nan), **kw)

        # Panel 3: Net dust rate — linear ±1
        ax_rate.plot(z, rate, **kw)
        #ax_rate.fill_between(z, rate, 0,
        #                     where=rate >= 0, color=c, alpha=0.15)
        #ax_rate.fill_between(z, rate, 0,
        #                     where=rate <  0, color=c, alpha=0.08,
        #                     hatch="////", linewidth=0)
        ax_rate.fill_between(
            z, rate, 0,
            where=(rate >= 0),
            color="#2ca02c",      # green
            alpha=0.20,
            interpolate=True,
        )

        ax_rate.fill_between(
            z, rate, 0,
            where=(rate < 0),
            color="#d62728",      # red
            alpha=0.20,
            interpolate=True,
        )

        # Panel 4: D/G vs redshift
        good_d = np.isfinite(dgr_s) & (dgr_s > 0)
        if good_d.any():
            ax_dgr.semilogy(z[good_d], dgr_s[good_d], **kw)

        # Panel 5: D/Z vs redshift
        good_z = np.isfinite(dtz_s) & (dtz_s > 0)
        if good_z.any():
            ax_dtz.semilogy(z[good_z], dtz_s[good_z], **kw)

    # ── Panel 3: linear y-axis, -1 to 1 ──────────────────────────────────────
    ax_rate.axhline(0, color="0.5", lw=0.8, ls="--")
    ax_rate.set_yscale("linear")
    ax_rate.set_ylim(-1, 1)
    ax_rate.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax_rate.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax_rate.text(0.97, 0.87, "net growth",      transform=ax_rate.transAxes,
                 fontsize=11, color="0.4", ha="right", style="italic")
    ax_rate.text(0.97, 0.06, "net destruction", transform=ax_rate.transAxes,
                 fontsize=11, color="0.4", ha="right", style="italic")

    # ── MW reference lines ────────────────────────────────────────────────────
    ax_dgr.axhline(0.01, color="0.5", lw=1.5, ls="--", alpha=0.7)
    ax_dgr.text(0.97, 0.06, r"MW $D/G \approx 0.01$",
                transform=ax_dgr.transAxes,
                fontsize=10, color="0.45", ha="right")

    ax_dtz.axhline(0.5, color="0.5", lw=1.5, ls="--", alpha=0.7)
    ax_dtz.text(0.97, 0.06, r"MW $D/Z \approx 0.5$",
                transform=ax_dtz.transAxes,
                fontsize=10, color="0.45", ha="right")

    ax_dgr.set_ylim(1e-4, 0.9e-1)

    # ── Shared x-axis ─────────────────────────────────────────────────────────
    z_max = max(r[1].max() for r in runs)
    x_max = min(z_max + 0.3, 7.0)
    ax_dtz.set_xlim(x_max, 0)
    ax_dtz.set_xlabel("Redshift", fontsize=14)
    ax_dtz.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax_dtz.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax_dtz.set_yscale("log")
    ax_dtz.set_ylim(3e-2, 1.2)
    ax_dtz.set_yticks([1e-1, 1e0])
    ax_dtz.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=True))

    # ── y-axis labels ─────────────────────────────────────────────────────────
    ax_dust.set_ylabel(r"Mass ($M_\odot$)")
    ax_sfr.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)")
    ax_rate.set_ylabel(r"$\dot{M}_\mathrm{dust}$"
                       r"  ($M_\odot\,\mathrm{yr}^{-1}$)")
    ax_dgr.set_ylabel(r"$D/G$")
    ax_dtz.set_ylabel(r"$D/Z$")

    # ── Grid ─────────────────────────────────────────────────────────────────
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="0.88", lw=0.5)
        #ax.grid(True, which="minor", color="0.93", lw=0.3)

    # ── Lookback-time axis on top ─────────────────────────────────────────────
    ax2 = ax_dust.twiny()
    ax2.set_xlim(ax_dust.get_xlim())
    valid    = z_ticks[z_ticks <= x_max]
    age_lbls = [f"{z_to_age_gyr(z):.1f}" for z in valid]
    ax2.set_xticks(valid)
    ax2.set_xticklabels(age_lbls, fontsize=11)
    ax2.set_xlabel("Lookback time (Gyr)", fontsize=12)
    ax2.invert_xaxis()
    ax2.grid(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    if len(runs) == 1:
        c0 = COLORS[0]
        ax_dust.legend(handles=[
            Line2D([0],[0], color=c0, lw=1.5, ls="--", alpha=0.5, label=r"$M_*$"),
            Line2D([0],[0], color=c0, lw=2.2, label=r"$M_\mathrm{dust}$"),            
        ], fontsize=10, loc="lower left", framealpha=0.85)
    else:
        handles = [Line2D([0],[0], color=COLORS[j], lw=2.2, label=runs[j][0])
                   for j in range(len(runs))]
        handles += [
            Line2D([0],[0], color="0.4", lw=1.5, ls="--", alpha=0.5,
                   label=r"$M_*$ (dashed)"),
            Line2D([0],[0], color="0.4", lw=2.2,
                   label=r"$M_\mathrm{dust}$ (solid)"),
        ]
        ax_dust.legend(handles=handles, fontsize=10, loc="lower left",
                       framealpha=0.85)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dirs", nargs="+")
    parser.add_argument("--labels",     nargs="*", default=None)
    parser.add_argument("--skip-every", type=int,  default=1)
    parser.add_argument("--output",     default="dust_evolution.pdf")
    args = parser.parse_args()

    n      = len(args.output_dirs)
    labels = args.labels if args.labels else [Path(d).name
                                              for d in args.output_dirs]
    if len(labels) != n:
        parser.error("--labels must match number of output_dirs")

    runs = []
    for d, lbl in zip(args.output_dirs, labels):
        print(f"\nProcessing: {lbl}")
        z, m_star, m_dust, sfr, dgr, dtz = run_simulation(
            d, skip_every=args.skip_every)
        runs.append((lbl, z, m_star, m_dust, sfr, dgr, dtz))

    make_plot(runs, args.output)


if __name__ == "__main__":
    main()
