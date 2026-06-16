#!/usr/bin/env python3
"""
plot_mdust_mstar_all_halos.py
------------------------------
Plot the total dust mass vs. stellar mass evolutionary track for Halo 569,
overlaid on observational data and optional SIMBA median comparison.

Both M_dust and M_star are measured within R_Crit200 at each snapshot.
Halo 569 is tracked by its z=0 comoving position across all epochs (via
halo_utils.get_halo569_reference / get_halo569) to avoid the FOF-rank
instability that caused sawtooth M*/R200 tracks in earlier versions.

Usage:
    python plot_mdust_mstar_all_halos.py ../S10_output_1024/ \\
        --simba-download --simba-dir ./simba/ \\
        --output mdust_mstar_S10_1024.png

    # Multiple resolutions:
    python plot_mdust_mstar_all_halos.py \\
        ../S10_output_512/ ../S10_output_1024/ \\
        --labels '$512^3$' '$1024^3$' \\
        --output mdust_mstar_convergence.png
"""

import sys
import os
import re
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from halo_utils import (
    get_halo569_reference,
    get_halo569,
    find_snapshots,
    get_unit_mass,
    load_particles_within_r200,
    read_snap_header,
)

SOLAR_MASS = 1.989e33

# ==============================================================================
# SIMBA caesar catalog support
# ==============================================================================

SIMBA_BASE_URL    = "http://simba.roe.ac.uk/simdata/m50n512/s50/catalogs"
SIMBA_OUTPUTS_URL = "http://simba.roe.ac.uk/outputs.txt"
SIMBA_DEFAULT_SNAPS = {151: 0.0}
SIMBA_CAESAR_FNAME  = "m50n512_{snap_num:03d}.hdf5"


def get_mancini2015():
    """
    Mancini et al. 2015, MNRAS, 451, L70, Table 1.
    Single detection: A1689-zD1 (z=7.5).
    Remaining 8 entries are upper limits, plotted as downward triangles.
    """
    det_ms = np.array([9.0])
    det_md = np.array([7.51])
    ul_ms  = np.array([9.2, 9.7, 9.5, 9.7, 9.9, 9.3, 9.3, 9.8])
    ul_md  = np.array([7.36, 8.28, 7.61, 7.43, 7.30, 7.02, 7.36, 7.38])
    return det_ms, det_md, ul_ms, ul_md


def fetch_simba_redshift_table(outputs_url=SIMBA_OUTPUTS_URL):
    import urllib.request
    print(f"Fetching SIMBA redshift table from {outputs_url} ...")
    try:
        with urllib.request.urlopen(outputs_url, timeout=15) as resp:
            lines = resp.read().decode().splitlines()
    except Exception as e:
        print(f"  WARNING: could not fetch outputs.txt ({e}). Using defaults.")
        return SIMBA_DEFAULT_SNAPS
    table = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                snap_num = int(parts[0])
                table[snap_num] = round(1.0 / float(parts[1]) - 1.0, 4)
            except ValueError:
                continue
    print(f"  Loaded {len(table)} entries from outputs.txt")
    return table


def download_simba_catalog(snap_num, dest_dir):
    import urllib.request
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = SIMBA_CAESAR_FNAME.format(snap_num=snap_num)
    dest  = dest_dir / fname
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  Already have {dest}  ({size_mb:.0f} MB)")
        if size_mb > 500:
            print(f"  WARNING: {fname} is {size_mb:.0f} MB -- likely a raw snapshot.")
        return str(dest)
    url = f"{SIMBA_BASE_URL}/{fname}"
    print(f"  Downloading {url} -> {dest} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1e6
        print(f"  Done ({size_mb:.0f} MB)")
        return str(dest)
    except Exception as e:
        print(f"  WARNING: download failed ({e})")
        if dest.exists():
            dest.unlink()
        return None


def hubble_time_yr(z, H0=68.0, Om0=0.3):
    KM_PER_MPC = 3.0857e19
    YR_IN_SEC  = 3.1557e7
    Hz = H0 * np.sqrt(Om0 * (1 + z)**3 + (1 - Om0))
    return 1.0 / (Hz / KM_PER_MPC * YR_IN_SEC)


def load_simba_catalog(filepath, mass_unit_msun=1.0, z=None, H0=68.0, Om0=0.3):
    with h5py.File(filepath, "r") as f:
        mstar = f["/galaxy_data/dicts/masses.stellar"][()] * mass_unit_msun
        mdust = f["/galaxy_data/dicts/masses.dust"][()]    * mass_unit_msun
        sfr   = f["/galaxy_data/sfr"][()]
        if z is None:
            sa = f.get("simulation_attributes")
            if sa is not None:
                z = sa.attrs.get("redshift", None)
                if z is not None:
                    z = float(z)
    if z is None:
        z = 0.0
    t_H      = hubble_time_yr(z, H0=H0, Om0=Om0)
    ssfr_min = 1.0 / t_H
    with np.errstate(divide="ignore", invalid="ignore"):
        ssfr = np.where(mstar > 0, sfr / mstar, 0.0)
    mask    = (mstar > 0) & (mdust > 0) & (ssfr > ssfr_min)
    n_total = int((mstar > 0).sum())
    n_sf    = int(mask.sum())
    print(f"    sSFR cut (z={z:.2f}): t_H={t_H/1e9:.2f} Gyr, "
          f"threshold={ssfr_min:.2e} yr^-1 -- "
          f"{n_sf}/{n_total} star-forming "
          f"(quenched fraction = {1 - n_sf/max(n_total,1):.2f})")
    if mask.sum() == 0:
        return np.array([]), np.array([])
    return np.log10(mstar[mask]), np.log10(mdust[mask])


def simba_running_median(log_mstar, log_mdust,
                         mstar_range=(7.5, 12.5), nbins=14, min_gal=15):
    edges   = np.linspace(mstar_range[0], mstar_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median  = np.full(nbins, np.nan)
    p16     = np.full(nbins, np.nan)
    p84     = np.full(nbins, np.nan)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (log_mstar >= lo) & (log_mstar < hi)
        if sel.sum() >= min_gal:
            vals      = log_mdust[sel]
            median[k] = np.median(vals)
            p16[k]    = np.percentile(vals, 16)
            p84[k]    = np.percentile(vals, 84)
    return centers, median, p16, p84


def load_simba_tracks(catalog_paths, redshift_table=None, mass_unit_msun=1.0):
    records = []
    for fpath in catalog_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"  WARNING: SIMBA catalog not found: {fpath}")
            continue
        z = None
        with h5py.File(fpath, "r") as f:
            sa = f.get("simulation_attributes")
            if sa is not None:
                z = sa.attrs.get("redshift", None)
                if z is not None:
                    z = float(z)
        if z is None and redshift_table is not None:
            m = re.search(r"_(\d{3})\.hdf5$", fpath.name)
            if m:
                z = redshift_table.get(int(m.group(1)), None)
        if z is None:
            print(f"  WARNING: could not determine redshift for {fpath.name}; skipping")
            continue
        log_ms, log_md = load_simba_catalog(fpath, mass_unit_msun=mass_unit_msun, z=z)
        if len(log_ms) == 0:
            print(f"  WARNING: no valid galaxies in {fpath.name}")
            continue
        centers, med, p16, p84 = simba_running_median(log_ms, log_md)
        print(f"  SIMBA {fpath.name}  z={z:.2f}  N_gal={len(log_ms):5d}  "
              f"median log(M_dust) range: [{np.nanmin(med):.1f}, {np.nanmax(med):.1f}]")
        records.append((z, centers, med, p16, p84))
    records.sort(key=lambda x: -x[0])
    return records


# ==============================================================================
# Observational data
# ==============================================================================

def load_obs_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"WARNING: obs data not found at {npz_path}")
        print("         Run:  python parse_obs_data.py obs_data/")
        empty = np.array([])
        return {k: empty for k in (
            "galliano2021_mstar", "galliano2021_mdust",
            "remyruyer2015_mstar", "remyruyer2015_mdust",
            "dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
        )}
    obs = np.load(npz_path)
    print(f"Loaded obs data from {npz_path}  keys: {list(obs.keys())}")
    return dict(obs)


# ==============================================================================
# Satellite halo catalogue at z=0
# ==============================================================================

def find_satellite_halos_z0(output_dir, ref, min_log_mstar=7.5):
    from halo_utils import (read_fof_catalog, load_particles_within_r200,
                            get_unit_mass, HALO569_SEARCH_RADIUS_CKPCH)

    snap_num   = ref.get("snap_num_z0", 49)
    output_dir = Path(output_dir)
    groups_dir = output_dir / f"groups_{snap_num:03d}"
    snapdir    = output_dir / f"snapdir_{snap_num:03d}"

    unit_mass_g  = get_unit_mass(snapdir)
    code_to_msun = unit_mass_g / SOLAR_MASS

    cat = read_fof_catalog(groups_dir, snap_num)
    if cat is None:
        print("WARNING: no z=0 catalog found -- skipping satellite halos")
        return []

    ref_pos = ref["center_ckpch"]
    box     = ref["box_ckpch"]

    dx     = cat["pos"] - ref_pos[None, :]
    dx    -= box * np.round(dx / box)
    dist   = np.sqrt((dx**2).sum(axis=1))
    within = dist <= HALO569_SEARCH_RADIUS_CKPCH

    print(f"\nSatellite halos: {int(within.sum())} FOF groups within "
          f"{HALO569_SEARCH_RADIUS_CKPCH:.0f} ckpc/h of Halo 569 at z=0")

    results = []
    for idx in np.where(within)[0]:
        r200 = float(cat["r200"][idx])
        if r200 <= 0:
            continue
        halo_dict = {"center": cat["pos"][idx].astype(float),
                     "r200_ckpch": r200}
        parts  = load_particles_within_r200(snapdir, halo_dict, part_types=(4, 6))
        m_star = parts[4].get("Masses", np.array([])).sum() * code_to_msun
        m_dust = parts[6].get("Masses", np.array([])).sum() * code_to_msun
        if m_star <= 0:
            continue
        log_ms = np.log10(m_star)
        if log_ms < min_log_mstar:
            continue
        log_md     = np.log10(m_dust) if m_dust > 0 else None
        is_primary = dist[idx] < 10.0
        results.append(dict(log_mstar=log_ms, log_mdust=log_md,
                            is_primary=is_primary))

    n_prim = sum(1 for r in results if r["is_primary"])
    n_sat  = len(results) - n_prim
    print(f"  Found {n_prim} primary + {n_sat} satellite halos "
          f"with log(M*) >= {min_log_mstar}")
    return results


# ==============================================================================
# Simulation track
# ==============================================================================

def run_simulation(output_dir, label, color, skip_every=1):
    snapshots = find_snapshots(output_dir)
    if not snapshots:
        raise RuntimeError(f"No snapshots with catalogs found in {output_dir}")

    print(f"\n[{label}]  {len(snapshots)} snapshots with catalogs")
    ref          = get_halo569_reference(output_dir)
    unit_mass_g  = get_unit_mass(snapshots[-1][1])
    code_to_msun = unit_mass_g / SOLAR_MASS

    results = []
    for i, (snap_num, snapdir, groups_dir) in enumerate(snapshots):
        if i % skip_every != 0:
            continue
        hdr  = read_snap_header(snapdir)
        halo = get_halo569(groups_dir, snap_num, ref, verbose=False)
        if halo is None or halo["r200_ckpch"] <= 0:
            continue
        particles = load_particles_within_r200(snapdir, halo, part_types=(4, 6))
        m_star = particles[4].get("Masses", np.array([])).sum() * code_to_msun
        m_dust = particles[6].get("Masses", np.array([])).sum() * code_to_msun
        if m_star <= 0 or m_dust <= 0:
            continue
        fallback = " [fallback]" if halo["used_fallback"] else ""
        print(f"  snap {snap_num:03d}  z={hdr['z']:.3f}  "
              f"log(M*/Msun)={np.log10(m_star):.2f}  "
              f"log(Md/Msun)={np.log10(m_dust):.2f}  "
              f"R200={halo['r200_pkpc']:.1f} pkpc"
              f"  d={halo['dist_ckpch']:.0f} ckpc/h{fallback}")
        results.append((hdr["z"], np.log10(m_star), np.log10(m_dust)))

    if not results:
        raise RuntimeError(f"No valid snapshots processed for {label}")

    results   = sorted(results, key=lambda x: -x[0])
    z_arr     = np.array([r[0] for r in results])
    mstar_arr = np.array([r[1] for r in results])
    mdust_arr = np.array([r[2] for r in results])
    return z_arr, mstar_arr, mdust_arr, ref


# ==============================================================================
# Figure
# ==============================================================================

def make_plot(sim_tracks, output_path, obs, simba_tracks=None,
              simba_show_band=True, satellite_halos=None,
              simba_max_epochs=1):

    _style = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cosmicgrain.mplstyle")
    plt.style.use(_style if os.path.exists(_style) else "default")

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)

    # ── Observational scatter ─────────────────────────────────────────────────
    def _scatter(key_ms, key_md, marker, color, label, size=12, alpha=0.6):
        ms = obs.get(key_ms, np.array([]))
        md = obs.get(key_md, np.array([]))
        if len(ms) > 0:
            ax.scatter(ms, md, s=size, marker=marker, color=color,
                       alpha=alpha, zorder=1, label=label, linewidths=0)

    _scatter("galliano2021_mstar", "galliano2021_mdust",
             "o", "0.55", "Galliano et al. 2021 (DustPedia, z~0)",
             size=10, alpha=0.5)
    _scatter("remyruyer2015_mstar", "remyruyer2015_mdust",
             "s", "0.30", u"R\u00e9my-Ruyer et al. 2015 (DGS+KINGFISH, z~0)",
             size=14, alpha=0.7)
    if len(obs.get("galliano2021_mstar", [])) == 0:
        _scatter("dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
                 "^", "0.55", "DustPedia CIGALE (z~0)", size=10, alpha=0.5)

    # Mancini+2015: detection (diamond) and upper limits (triangles), same size
    det_ms, det_md, ul_ms, ul_md = get_mancini2015()
    ax.scatter(det_ms, det_md, s=40, marker="D", color="#c0392b",
               alpha=0.85, zorder=2, label="Mancini et al. 2015 (z~6.5\u20137.5)")
    ax.scatter(ul_ms, ul_md, s=40, marker="v", color="#c0392b",
               alpha=0.5, zorder=2, label="_nolegend_")

    SIMBA_COLOR = "#393be0"
    # 1024^3 = teal (#009E73), 2048^3 = blue (#0072B2)
    SIM_COLORS  = ["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#E69F00"]

    # ── SIMBA median ──────────────────────────────────────────────────────────
    if simba_tracks:
        n_all = len(simba_tracks)
        shown = (list(range(n_all)) if n_all <= simba_max_epochs else
                 sorted(set(round(i*(n_all-1)/(simba_max_epochs-1))
                            for i in range(simba_max_epochs))))
        for plot_i, track_i in enumerate(shown):
            z_sim, centers, med, p16, p84 = simba_tracks[track_i]
            valid   = np.isfinite(med)
            leg_lbl = r"SIMBA m50n512 (median)" if plot_i == 0 else "_nolegend_"
            ax.plot(centers[valid], med[valid],
                    ls="--", lw=1.8, color=SIMBA_COLOR, zorder=3,
                    label=leg_lbl, alpha=0.85)
            if simba_show_band and track_i == shown[-1]:
                valid_b = np.isfinite(p16) & np.isfinite(p84)
                ax.fill_between(centers[valid_b], p16[valid_b], p84[valid_b],
                                color=SIMBA_COLOR, alpha=0.10, zorder=2, lw=0)

    # ── Satellite halos (open circles) ────────────────────────────────────────
    # Plotted before sim stars so satellites appear above SIMBA but below stars
    # in the legend.
    if satellite_halos:
        sat_ms = [h["log_mstar"] for h in satellite_halos
                  if not h["is_primary"] and h["log_mdust"] is not None]
        sat_md = [h["log_mdust"] for h in satellite_halos
                  if not h["is_primary"] and h["log_mdust"] is not None]
        if sat_ms:
            # Label includes N count and resolution of the run
            sat_label = (f"CosmicGrain satellites "
                         f"(z=0, N={len(sat_ms)}, $1024^3$)")
            ax.scatter(sat_ms, sat_md,
                       s=50, marker="o", facecolors="none",
                       edgecolors=SIM_COLORS[0], linewidths=1.4, zorder=8,
                       label=sat_label)

    # ── Sim resolution stars — plotted LAST so they sit at bottom of legend ──
    for si, (label, color, z_arr, mstar_arr, mdust_arr) in enumerate(sim_tracks):
        track_color = SIM_COLORS[si % len(SIM_COLORS)]
        idx_z0      = np.argmin(np.abs(z_arr))
        ax.scatter(mstar_arr[idx_z0], mdust_arr[idx_z0],
                   s=480, marker="*", color=track_color,
                   edgecolors="k", linewidths=0.7, zorder=10,
                   label=f"CosmicGrain Halo 569 {label} (z=0)")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\log\,M_\star\;(\mathrm{M}_\odot)$")
    ax.set_ylabel(r"$\log\,M_\mathrm{dust}\;(\mathrm{M}_\odot)$")
    ax.set_xlim(7.6, 11.4)
    ax.set_ylim(4.2, 8.2)
    ax.legend(loc="lower right", framealpha=0.85,
              handlelength=1.5, labelspacing=0.4, borderpad=0.7)
    ax.grid(True, which="minor", color="0.93", lw=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot M_dust vs M_star for CosmicGrain runs")
    parser.add_argument("output_dirs", nargs="+")
    parser.add_argument("--labels",         nargs="*", default=None)
    parser.add_argument("--skip-every",     type=int,  default=1)
    parser.add_argument("--output",         default="mdust_mstar.png")
    parser.add_argument("--obs-data",       default="obs_data/obs_dustmass.npz")
    parser.add_argument("--min-log-mstar",  type=float, default=7.5)
    parser.add_argument("--no-satellites",  action="store_true")
    parser.add_argument("--simba-max-epochs", type=int, default=3)

    simba = parser.add_argument_group("SIMBA comparison")
    simba.add_argument("--simba-catalogs",  nargs="*", default=None, metavar="FILE")
    simba.add_argument("--simba-download",  action="store_true")
    simba.add_argument("--simba-dir",       default="./simba/", metavar="DIR")
    simba.add_argument("--simba-snaps",     nargs="*", type=int, default=None)
    simba.add_argument("--simba-mass-unit", type=float, default=1.0)
    simba.add_argument("--simba-no-band",   action="store_true")

    args   = parser.parse_args()
    n      = len(args.output_dirs)
    labels = args.labels if args.labels else [Path(d).name for d in args.output_dirs]
    if len(labels) != n:
        parser.error("--labels must match the number of output_dirs")

    obs = load_obs_data(args.obs_data)

    SIM_COLORS = ["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#E69F00"]
    sim_tracks = []
    refs       = []
    for i, (d, lbl) in enumerate(zip(args.output_dirs, labels)):
        print(f"\nProcessing: {lbl}")
        z_arr, mstar_arr, mdust_arr, ref = run_simulation(
            d, lbl, SIM_COLORS[i % len(SIM_COLORS)],
            skip_every=args.skip_every)
        sim_tracks.append((lbl, SIM_COLORS[i % len(SIM_COLORS)],
                           z_arr, mstar_arr, mdust_arr))
        refs.append((d, ref))

    satellite_halos = None
    if not args.no_satellites:
        d0, ref0 = refs[0]
        satellite_halos = find_satellite_halos_z0(
            d0, ref0, min_log_mstar=args.min_log_mstar)

    simba_catalog_paths = []
    if args.simba_catalogs:
        simba_catalog_paths = args.simba_catalogs
    elif args.simba_download:
        redshift_table = fetch_simba_redshift_table()
        default_snaps  = args.simba_snaps or sorted(SIMBA_DEFAULT_SNAPS.keys())
        print(f"\nDownloading SIMBA snapshots: {default_snaps}")
        for sn in default_snaps:
            p = download_simba_catalog(sn, args.simba_dir)
            if p:
                simba_catalog_paths.append(p)

    simba_tracks = None
    if simba_catalog_paths:
        print(f"\nLoading {len(simba_catalog_paths)} SIMBA catalog(s)...")
        redshift_table = fetch_simba_redshift_table()
        simba_tracks   = load_simba_tracks(
            simba_catalog_paths,
            redshift_table=redshift_table,
            mass_unit_msun=args.simba_mass_unit)
        if not simba_tracks:
            print("WARNING: no valid SIMBA tracks loaded.")
            simba_tracks = None

    make_plot(sim_tracks, args.output, obs,
              simba_tracks=simba_tracks,
              simba_show_band=not args.simba_no_band,
              satellite_halos=satellite_halos,
              simba_max_epochs=args.simba_max_epochs)


if __name__ == "__main__":
    main()
