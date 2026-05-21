#!/usr/bin/env python3
"""
plot_mdust_mstar.py
-------------------
Plot the total dust mass vs. stellar mass evolutionary track for Halo 569,
overlaid on observational data compiled in Osman et al. 2025 (DUSTY-GAEA,
arXiv:2512.15902) Figure 2.

Both M_dust and M_star are measured within R_200 at each snapshot.

Usage:
    python plot_mdust_mstar.py /path/to/output/ [options]

    python plot_mdust_mstar.py ../5_output_zoom_1024_halo569_50Mpc_dust/ \\
        --output mdust_mstar_1024.png

    # Multiple resolutions on one plot:
    python plot_mdust_mstar.py \\
        ../5_output_zoom_512_halo569_50Mpc_dust/ \\
        ../5_output_zoom_1024_halo569_50Mpc_dust/ \\
        --labels "512^3" "1024^3" \\
        --output mdust_mstar_convergence.png

    # With SIMBA median comparison (caesar catalog files already downloaded):
    python plot_mdust_mstar_with_simba.py ../S10_output_1024 --simba-catalogs simba/m50n512_151.hdf5 --output mdust_mstar_simba.png

    # Download SIMBA catalogs automatically (requires internet):
    python plot_mdust_mstar.py ../5_output_zoom_1024_halo569_50Mpc_dust/ \\
        --simba-download --simba-dir ./simba/ \\
        --output mdust_mstar_simba.png
"""

import sys
import os
import glob
import re
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

try:
    from halo_utils import load_target_halo, extract_dust_spatially
    HALO_UTILS_AVAILABLE = True
except ImportError:
    HALO_UTILS_AVAILABLE = False
    print("Warning: halo_utils not found — will fall back to density centroid.")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_MASS = 1.989e33          # g
PC_IN_CM   = 3.0857e18         # cm per parsec

# ---------------------------------------------------------------------------
# SIMBA caesar catalog support
# ---------------------------------------------------------------------------
# SIMBA m50n512 box — 50 Mpc/h, 512^3 particles.
# These are caesar galaxy catalog files (not raw snapshots), available at:
#   http://simba.roe.ac.uk/simdata/m50n512/s50/snapshots/
# Redshift mapping is from http://simba.roe.ac.uk/outputs.txt
#
# Units in SIMBA caesar output: masses are in M_sun (not 10^10 M_sun).
# Verify with f["/simulation_attributes"].attrs if uncertain.

SIMBA_BASE_URL = "http://simba.roe.ac.uk/simdata/m50n512/s50/snapshots"
SIMBA_OUTPUTS_URL = "http://simba.roe.ac.uk/outputs.txt"

# Default set of snapshots to use for the median comparison:
# chosen to bracket z ~ 0, 0.5, 1, 2, 3, 4 where census is large enough.
# Map: snap_num -> approximate redshift (confirmed against outputs.txt).
# Set SIMBA_DEFAULT_SNAPS to None to auto-download outputs.txt.
SIMBA_DEFAULT_SNAPS = {
    151: 0.0,
    105: 0.5,
     78: 1.0,
     51: 2.0,
     36: 3.0,
     27: 4.0,
}


def fetch_simba_redshift_table(outputs_url=SIMBA_OUTPUTS_URL):
    """
    Download SIMBA outputs.txt and return a dict {snap_num: redshift}.
    Format expected (space-separated):  snap_num  scale_factor  ...
    """
    import urllib.request
    print(f"Fetching SIMBA redshift table from {outputs_url} ...")
    try:
        with urllib.request.urlopen(outputs_url, timeout=15) as resp:
            lines = resp.read().decode().splitlines()
    except Exception as e:
        print(f"  WARNING: could not fetch outputs.txt ({e}). "
              "Using hard-coded default snapshot list.")
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
                a = float(parts[1])
                table[snap_num] = round(1.0 / a - 1.0, 4)
            except ValueError:
                continue
    print(f"  Loaded {len(table)} entries from outputs.txt")
    return table


def download_simba_catalog(snap_num, dest_dir):
    """
    Download one SIMBA caesar catalog to dest_dir if not already present.
    Returns the local path, or None on failure.
    """
    import urllib.request
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"snap_m50n512_{snap_num:03d}.hdf5"
    dest  = dest_dir / fname
    if dest.exists():
        print(f"  Already have {dest}")
        return str(dest)
    url = f"{SIMBA_BASE_URL}/{fname}"
    print(f"  Downloading {url} → {dest} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Done ({dest.stat().st_size / 1e6:.0f} MB)")
        return str(dest)
    except Exception as e:
        print(f"  WARNING: download failed ({e})")
        if dest.exists():
            dest.unlink()
        return None


def hubble_time_yr(z, H0=68.0, Om0=0.3):
    """
    Compute the Hubble time t_H(z) = 1/H(z) in years for a flat LCDM cosmology.

    This is used as the sSFR threshold for star-forming galaxy selection:
    sSFR > 1/t_H(z) means the galaxy would double its stellar mass within
    the current age of the universe — a physically motivated, redshift-consistent
    definition of "actively star-forming" (vs. a fixed z=0 threshold that becomes
    far too permissive at high redshift).

    Parameters
    ----------
    z    : float  — redshift
    H0   : float  — Hubble constant in km/s/Mpc (SIMBA default: 68.0)
    Om0  : float  — matter density parameter (SIMBA default: 0.3)

    Returns
    -------
    t_H  : float  — Hubble time 1/H(z) in years
    """
    KM_PER_MPC = 3.0857e19          # km per Mpc
    YR_IN_SEC  = 3.1557e7           # seconds per year
    Hz = H0 * np.sqrt(Om0 * (1 + z)**3 + (1 - Om0))   # km/s/Mpc
    Hz_per_yr = Hz / KM_PER_MPC * YR_IN_SEC            # yr^-1
    return 1.0 / Hz_per_yr                              # years


def load_simba_catalog(filepath, mass_unit_msun=1.0, z=None,
                       H0=68.0, Om0=0.3):
    """
    Load M_star and M_dust from a SIMBA caesar catalog, retaining only
    star-forming galaxies with sSFR > 1/t_H(z).

    The Hubble-time threshold scales with redshift, so the criterion is
    consistent across epochs: a galaxy is "star-forming" if it would
    meaningfully grow its stellar mass within the current age of the universe.
    At z=0 this gives sSFR > ~7e-11 yr^-1; at z=2 it gives ~3e-10 yr^-1.

    Parameters
    ----------
    filepath : str | Path
    mass_unit_msun : float
        Multiply stored masses by this factor to get solar masses.
        Default 1.0 (SIMBA caesar outputs are already in M_sun).
    z : float or None
        Redshift of the snapshot. If None, tries to read from the catalog's
        simulation_attributes; falls back to z=0 with a warning.
    H0, Om0 : float
        Cosmological parameters matching the SIMBA run (h=0.68, Om=0.3).

    Returns
    -------
    log_mstar, log_mdust : 1-D arrays of log10(M / M_sun)
        Only star-forming galaxies with mstar > 0 and mdust > 0 are returned.
    """
    with h5py.File(filepath, "r") as f:
        mstar = f["/galaxy_data/dicts/masses.stellar"][()] * mass_unit_msun
        mdust = f["/galaxy_data/dicts/masses.dust"][()]    * mass_unit_msun
        sfr   = f["/galaxy_data/sfr"][()]   # M_sun / yr

        # Try to read redshift from catalog if not supplied
        if z is None:
            sa = f.get("simulation_attributes")
            if sa is not None:
                z = sa.attrs.get("redshift", None)
                if z is not None:
                    z = float(z)
        if z is None:
            print("    WARNING: redshift unknown, defaulting to z=0 for sSFR threshold")
            z = 0.0

    # Redshift-dependent sSFR threshold: 1 / t_H(z)
    t_H     = hubble_time_yr(z, H0=H0, Om0=Om0)
    ssfr_min = 1.0 / t_H

    with np.errstate(divide="ignore", invalid="ignore"):
        ssfr = np.where(mstar > 0, sfr / mstar, 0.0)

    mask    = (mstar > 0) & (mdust > 0) & (ssfr > ssfr_min)
    n_total = int((mstar > 0).sum())
    n_sf    = int(mask.sum())
    print(f"    sSFR cut (z={z:.2f}): t_H={t_H/1e9:.2f} Gyr, "
          f"threshold={ssfr_min:.2e} yr^-1 — "
          f"{n_sf}/{n_total} star-forming "
          f"(quenched fraction = {1 - n_sf/max(n_total,1):.2f})")

    if mask.sum() == 0:
        return np.array([]), np.array([])

    return np.log10(mstar[mask]), np.log10(mdust[mask])


def simba_running_median(log_mstar, log_mdust,
                         mstar_range=(7.5, 12.5), nbins=14, min_gal=15):
    """
    Compute median and 16/84th-percentile M_dust in equal-width M_star bins.

    nbins=14 and min_gal=15 avoids the noisy zigzag at the massive end
    that appears with finer binning and low count thresholds.

    Returns bin_centers, median, p16, p84 — NaN where fewer than min_gal
    galaxies fall in the bin.
    """
    edges   = np.linspace(mstar_range[0], mstar_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median  = np.full(nbins, np.nan)
    p16     = np.full(nbins, np.nan)
    p84     = np.full(nbins, np.nan)

    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # Quenched galaxies already removed upstream by the sSFR cut in
        # load_simba_catalog — no additional floor needed here.
        sel = (log_mstar >= lo) & (log_mstar < hi)
        if sel.sum() >= min_gal:
            vals      = log_mdust[sel]
            median[k] = np.median(vals)
            p16[k]    = np.percentile(vals, 16)
            p84[k]    = np.percentile(vals, 84)

    return centers, median, p16, p84


def load_simba_tracks(catalog_paths, redshift_table=None, mass_unit_msun=1.0):
    """
    Given a list of caesar catalog paths, load each one and return a list of
        (redshift, bin_centers, median_mdust, p16, p84)
    sorted by descending redshift.

    redshift_table : dict {snap_num: z} or None.
        If None, the redshift is read from the catalog's Header attributes;
        falls back to parsing the snap number via outputs.txt defaults.
    """
    records = []
    for fpath in catalog_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"  WARNING: SIMBA catalog not found: {fpath}")
            continue

        # --- determine redshift ---
        z = None
        with h5py.File(fpath, "r") as f:
            # Caesar stores current redshift in simulation_attributes
            sa = f.get("simulation_attributes")
            if sa is not None:
                z = sa.attrs.get("redshift", None)
                if z is not None:
                    z = float(z)
        if z is None and redshift_table is not None:
            m = re.search(r"_(\d{3})\.hdf5$", fpath.name)
            if m:
                snap_num = int(m.group(1))
                z = redshift_table.get(snap_num, None)
        if z is None:
            print(f"  WARNING: could not determine redshift for {fpath.name}; skipping")
            continue

        log_ms, log_md = load_simba_catalog(fpath, mass_unit_msun=mass_unit_msun, z=z)
        if len(log_ms) == 0:
            print(f"  WARNING: no valid galaxies in {fpath.name}")
            continue

        centers, med, p16, p84 = simba_running_median(log_ms, log_md)
        print(f"  SIMBA {fpath.name}  z={z:.2f}  N_gal={len(log_ms):5d}  "
              f"median log(M_dust) range: "
              f"[{np.nanmin(med):.1f}, {np.nanmax(med):.1f}]")
        records.append((z, centers, med, p16, p84))

    records.sort(key=lambda x: -x[0])   # high-z first
    return records


# ---------------------------------------------------------------------------
# Observational data loader
# ---------------------------------------------------------------------------

def load_obs_data(npz_path):
    """
    Load real observational data from the .npz produced by parse_obs_data.py.
    Falls back to empty arrays with a warning if the file is missing.

    Expected keys in the .npz:
        galliano2021_mstar / galliano2021_mdust   — 784 DustPedia galaxies, z~0
        remyruyer2015_mstar / remyruyer2015_mdust — 109 DGS+KINGFISH galaxies, z~0
        dustpedia_cigale_mstar / dustpedia_cigale_mdust — 815 galaxies, z~0
    """
    if not os.path.exists(npz_path):
        print(f"WARNING: obs data not found at {npz_path}")
        print("         Run:  python parse_obs_data.py obs_data/")
        print("         Using empty arrays — observational points will be absent.")
        empty = np.array([])
        return {k: empty for k in (
            "galliano2021_mstar", "galliano2021_mdust",
            "remyruyer2015_mstar", "remyruyer2015_mdust",
            "dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
        )}
    obs = np.load(npz_path)
    print(f"Loaded obs data from {npz_path}  keys: {list(obs.keys())}")
    return dict(obs)

# ---------------------------------------------------------------------------

def find_snapshots(output_dir):
    """
    Return sorted list of (snap_num, snapdir_path, groups_path) tuples.
    Skips entries where no groups catalog exists.
    """
    output_dir = Path(output_dir)
    entries = []

    snapdirs = sorted(output_dir.glob("snapdir_*"))
    for snapdir in snapdirs:
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))

        # Locate a groups catalog
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        catalog_files = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) if groups_dir.exists() else []
        if not catalog_files:
            continue   # no subfind catalog → skip

        # Locate first HDF5 chunk of the snapshot
        snap_files = sorted(snapdir.glob("snap_*.hdf5")) + sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snap_files:
            continue

        entries.append((snap_num, snapdir, str(snap_files[0]), str(catalog_files[0])))

    return entries


def get_header(snap_file):
    """Return (redshift, h, BoxSize_kpc) from an HDF5 snapshot header."""
    with h5py.File(snap_file, "r") as f:
        hdr = f["Header"].attrs
        params = f["Parameters"].attrs
        z    = float(hdr["Redshift"])
        h    = float(params["HubbleParam"])
        box  = float(hdr["BoxSize"])   # comoving kpc/h
    return z, h, box


def get_unit_mass(snap_file):
    """Return UnitMass_in_g from snapshot (fall back to Gadget default)."""
    with h5py.File(snap_file, "r") as f:
        params = f.get("Parameters") or f.get("Config") or {}
        um = None
        if params:
            um = params.attrs.get("UnitMass_in_g", None)
        if um is None:
            um = 1.989e43   # 10^10 M_sun in grams (Gadget default)
    return float(um)


def load_particles_within_r200(snap_file_first, halo_center_kph, r200_kph,
                                part_types=(4, 6)):
    """
    Load all particles of requested PartTypes within r200 (comoving kpc/h).
    Returns dict {ptype: {'mass': array_in_code_units}}.

    Handles multi-chunk snapshots by globbing sibling files.
    """
    # Collect all chunks
    p = Path(snap_file_first)
    chunks = sorted(p.parent.glob(p.name.split(".")[0].rstrip("0123456789") + "*.hdf5"))
    if not chunks:
        chunks = [p]

    result = {pt: [] for pt in part_types}

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                coords = f[key]["Coordinates"][:]   # comoving kpc/h
                mass_key = "Masses" if "Masses" in f[key] else None

                # Dust particles always have explicit Masses
                # Stars may use a mass table (check header)
                if mass_key is None:
                    hdr = f["Header"].attrs
                    mass_table = hdr.get("MassTable", None)
                    if mass_table is not None and mass_table[pt] > 0:
                        n = len(coords)
                        masses = np.full(n, mass_table[pt])
                    else:
                        continue
                else:
                    masses = f[key][mass_key][:]

                # Periodic distance check
                box = float(f["Header"].attrs["BoxSize"])
                dx = coords - halo_center_kph
                dx = dx - box * np.round(dx / box)
                r  = np.sqrt(np.sum(dx**2, axis=1))

                mask = r <= r200_kph
                result[pt].append(masses[mask])

    return {pt: np.concatenate(result[pt]) if result[pt] else np.array([])
            for pt in part_types}


def extract_r200_from_catalog(catalog_file):
    """
    Return (halo_center_kph, r200_kph, m200_code) from the primary SubFind group.

    Handles multi-chunk catalogs (fof_subhalo_tab_NNN.0.hdf5, .1.hdf5, …) by
    globbing siblings.  Returns None if no groups exist yet (early snapshots).
    """
    p = Path(catalog_file)
    # Glob all chunks: strip trailing digits from stem to get base name
    stem_base = re.sub(r"\.\d+$", "", p.stem)   # e.g. fof_subhalo_tab_000
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    pos_list  = []
    r200_list = []
    m200_list = []

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_list.append(grp["GroupPos"][:])
            r200_list.append(grp["Group_R_Crit200"][:])
            m200_list.append(grp["Group_M_Crit200"][:])

    if not pos_list:
        return None   # no groups yet (high-z snapshot)

    all_pos  = np.concatenate(pos_list,  axis=0)
    all_r200 = np.concatenate(r200_list, axis=0)
    all_m200 = np.concatenate(m200_list, axis=0)

    # Primary group = most massive (index 0 after Gadget sorts by mass desc)
    return all_pos[0], float(all_r200[0]), float(all_m200[0])


def process_snapshot(snap_num, snapdir, snap_file, catalog_file, unit_mass_g=None):
    """
    For one snapshot return (z, log_mstar_msun, log_mdust_msun) or None.
    """
    z, h, box = get_header(snap_file)
    if unit_mass_g is None:
        unit_mass_g = get_unit_mass(snap_file)

    # Code unit → solar masses conversion
    code_to_msun = unit_mass_g / SOLAR_MASS   # typically 1e10 / h

    # Halo center and R200 — read directly from subfind Group catalog.
    # halo_utils requires Subhalo which is absent at high-z or in FOF-only
    # catalogs, so we bypass it here and use the Group table directly.
    result = extract_r200_from_catalog(catalog_file)
    if result is None:
        return None   # no groups yet at this redshift — skip silently
    halo_center, r200, _ = result
    if r200 <= 0:
        return None

    # Load masses within R200 for stars (pt=4) and dust (pt=6)
    particles = load_particles_within_r200(snap_file, halo_center, r200,
                                           part_types=(4, 6))

    m_star_code = particles[4].sum() if len(particles[4]) else 0.0
    m_dust_code = particles[6].sum() if len(particles[6]) else 0.0

    m_star_msun = m_star_code * code_to_msun
    m_dust_msun = m_dust_code * code_to_msun

    if m_star_msun <= 0 or m_dust_msun <= 0:
        return None

    print(f"  snap {snap_num:03d}  z={z:.3f}  "
          f"log(M*/Msun)={np.log10(m_star_msun):.2f}  "
          f"log(Md/Msun)={np.log10(m_dust_msun):.2f}  "
          f"R200={r200:.1f} kpc/h")

    return z, np.log10(m_star_msun), np.log10(m_dust_msun)


def run_simulation(output_dir, label, color, skip_every=1):
    """
    Iterate over all snapshots with subfind catalogs and return arrays
    (z, log_mstar, log_mdust), sorted by descending z.
    """
    snapshots = find_snapshots(output_dir)
    if not snapshots:
        raise RuntimeError(f"No snapshots with subfind catalogs found in {output_dir}")

    print(f"\n[{label}] Found {len(snapshots)} snapshots with catalogs")

    # Read unit_mass once from the first available snapshot
    unit_mass_g = get_unit_mass(snapshots[0][2])

    results = []
    for i, (snap_num, snapdir, snap_file, catalog_file) in enumerate(snapshots):
        if i % skip_every != 0:
            continue
        r = process_snapshot(snap_num, snapdir, snap_file, catalog_file, unit_mass_g)
        if r is not None:
            results.append(r)

    if not results:
        raise RuntimeError(f"No valid snapshots processed for {label}")

    results = sorted(results, key=lambda x: -x[0])   # high-z first
    z_arr      = np.array([r[0] for r in results])
    mstar_arr  = np.array([r[1] for r in results])
    mdust_arr  = np.array([r[2] for r in results])
    return z_arr, mstar_arr, mdust_arr


def make_plot(sim_tracks, output_path, obs, simba_tracks=None,
              simba_show_band=True):
    """
    sim_tracks   : list of (label, color, z_arr, mstar_arr, mdust_arr)
    obs          : dict from load_obs_data()
    simba_tracks : list of (z, bin_centers, median, p16, p84) from
                   load_simba_tracks(), or None to omit
    simba_show_band : if True, shade the 16-84th percentile region for each
                   SIMBA snapshot
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # ------------------------------------------------------------------
    # Observational data  (real data from parse_obs_data.py)
    # ------------------------------------------------------------------
    def _scatter(key_ms, key_md, marker, color, label, size=12, alpha=0.6):
        ms = obs.get(key_ms, np.array([]))
        md = obs.get(key_md, np.array([]))
        if len(ms) > 0:
            ax.scatter(ms, md, s=size, marker=marker, color=color,
                       alpha=alpha, zorder=1, label=label, linewidths=0)

    # Galliano+2021: best single z~0 dataset (784 DustPedia galaxies,
    # hierarchical Bayesian SED, J/A+A/649/A18)
    _scatter("galliano2021_mstar", "galliano2021_mdust",
             "o", "0.55", "Galliano et al 2021 (DustPedia; z~0)", size=10, alpha=0.5)

    # Remy-Ruyer+2015: DGS+KINGFISH (109 galaxies, covers dwarf end,
    # J/A+A/582/A121)
    _scatter("remyruyer2015_mstar", "remyruyer2015_mdust",
             "s", "0.30", "Rémy-Ruyer et al 2015 (DGS+KINGFISH; z~0)", size=14, alpha=0.7)

    # DustPedia CIGALE (Nersesian+2019, 815 galaxies)
    # Only shown if Galliano+21 not available (same sample, older dust masses)
    if len(obs.get("galliano2021_mstar", [])) == 0:
        _scatter("dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
                 "^", "0.55", "DustPedia CIGALE (z~0)", size=10, alpha=0.5)

    # ------------------------------------------------------------------
    # SIMBA median tracks (one dashed line per snapshot/redshift)
    # ------------------------------------------------------------------
    # Offset vmin to -0.8 so z=0 maps to orange-gold rather than the
    # bright yellow tip of plasma_r, which is nearly invisible on white.
    # The colorbar ticks are set explicitly to 0–6 so the axis reads correctly.
    z_norm = plt.Normalize(vmin=-1.9, vmax=6)

    if simba_tracks:
        for i, (z_sim, centers, med, p16, p84) in enumerate(simba_tracks):
            simba_color = cm.plasma_r(z_norm(z_sim))
            valid = np.isfinite(med)

            # Only label the first entry in the legend (avoid clutter); tag
            # each line with its redshift via a text annotation instead.
            legend_label = r"SIMBA m50n512 (median, sSFR $> 1/t_\mathrm{H}(z)$)" if i == 0 else "_nolegend_"

            ax.plot(centers[valid], med[valid],
                    linestyle="--", linewidth=2,
                    color=simba_color, zorder=3,
                    label=legend_label, alpha=0.85)

            if simba_show_band:
                valid_band = np.isfinite(p16) & np.isfinite(p84)
                ax.fill_between(centers[valid_band],
                                p16[valid_band], p84[valid_band],
                                color=simba_color, alpha=0.10, zorder=2,
                                linewidth=0)

            # Annotate with redshift at the leftmost valid bin
            #left_idx = np.where(valid)[0]
            #if len(left_idx) > 0:
            #    li = left_idx[0]
            #    ax.annotate(
            #        f"SIMBA z={z_sim:.1f}",
            #        (centers[li], med[li]),
            #        textcoords="offset points", xytext=(-4, 4),
            #        fontsize=6.0, color=simba_color, zorder=7,
            #        ha="right", style="italic",
            #    )

    # ------------------------------------------------------------------
    # Simulation tracks
    # ------------------------------------------------------------------
    # Redshift label positions: annotate a few z milestones along each track
    Z_LABELS = [6.0, 4.5, 3.5, 2.5, 1.5, 0.5, 0.0]

    for (label, color, z_arr, mstar_arr, mdust_arr) in sim_tracks:
        # Color the track by redshift
        from matplotlib.collections import LineCollection
        points  = np.array([mstar_arr, mdust_arr]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        lc      = LineCollection(segs, cmap="plasma_r", norm=z_norm,
                                 linewidth=2.0, zorder=5, label=label)
        lc.set_array(0.5 * (z_arr[:-1] + z_arr[1:]))
        ax.add_collection(lc)

        # Mark z milestones — labels left of dot on rising section (z>2), right otherwise
        for z_target in Z_LABELS:
            idx = np.argmin(np.abs(z_arr - z_target))
            if np.abs(z_arr[idx] - z_target) < 0.3:
                ax.scatter(mstar_arr[idx], mdust_arr[idx],
                           s=35, color=cm.plasma_r(z_norm(z_arr[idx])),
                           zorder=6, edgecolors="k", linewidths=0.4)
                # Rising section (z >= 2): label to the left; late-time: to the right
                if z_arr[idx] >= 2.0:
                    ax.annotate(f"z={z_arr[idx]:.1f}",
                                (mstar_arr[idx], mdust_arr[idx]),
                                textcoords="offset points", xytext=(-38, 3),
                                fontsize=6.5, color="0.2", zorder=7,
                                ha="left")
                else:
                    ax.annotate(f"z={z_arr[idx]:.1f}",
                                (mstar_arr[idx], mdust_arr[idx]),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=6.5, color="0.2", zorder=7,
                                ha="left")

    # ------------------------------------------------------------------
    # Colorbar for redshift
    # ------------------------------------------------------------------
    sm = cm.ScalarMappable(cmap="plasma_r", norm=z_norm)
    sm.set_array([])
    #cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    #cbar.set_label("Redshift $z$", fontsize=9)
    #cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
    #cbar.ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$\log\,M_\star\,(\mathrm{M}_\odot)$", fontsize=11)
    ax.set_ylabel(r"$\log\,M_\mathrm{dust}\,(\mathrm{M}_\odot)$", fontsize=11)
    ax.set_xlim(7.0, 12.0)
    ax.set_ylim(3.5, 9.5)
    ax.tick_params(labelsize=9)

    ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
              handlelength=1.5, labelspacing=0.3, borderpad=0.6)

    ax.set_title("Evolution of Halo 569", fontsize=10)

    # Grid behind everything — set_axisbelow pushes grid under all artists
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="major", color="0.88", linewidth=0.5)
    ax.grid(True, which="minor", color="0.93", linewidth=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot M_dust vs M_star track for CosmicGrain runs, "
                    "with optional SIMBA median comparison")
    parser.add_argument("output_dirs", nargs="+",
                        help="One or more Gadget-4 output directories")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels for each run (default: directory names)")
    parser.add_argument("--skip-every", type=int, default=1,
                        help="Process every N-th snapshot (e.g. 2 = every other)")
    parser.add_argument("--output", default="mdust_mstar.png",
                        help="Output figure filename")
    parser.add_argument("--obs-data", default="obs_data/obs_dustmass.npz",
                        help="Path to obs_dustmass.npz from parse_obs_data.py")

    # SIMBA options
    simba_grp = parser.add_argument_group("SIMBA comparison")
    simba_grp.add_argument(
        "--simba-catalogs", nargs="*", default=None, metavar="FILE",
        help="Paths to SIMBA caesar catalog HDF5 files (one per redshift). "
             "E.g.  --simba-catalogs simba/snap_m50n512_051.hdf5 "
             "simba/snap_m50n512_105.hdf5")
    simba_grp.add_argument(
        "--simba-download", action="store_true",
        help="Auto-download default SIMBA snapshots from simba.roe.ac.uk "
             "(requires internet; ~200 MB each)")
    simba_grp.add_argument(
        "--simba-dir", default="./simba/", metavar="DIR",
        help="Directory to store/find downloaded SIMBA files "
             "(default: ./simba/)")
    simba_grp.add_argument(
        "--simba-snaps", nargs="*", type=int, default=None, metavar="N",
        help="Snapshot numbers to download/use when --simba-download is set "
             "(default: 027 036 051 078 105 151 → z~4,3,2,1,0.5,0)")
    simba_grp.add_argument(
        "--simba-mass-unit", type=float, default=1.0,
        help="Multiply SIMBA stored masses by this factor to get M_sun. "
             "Default 1.0 (caesar output already in M_sun). "
             "Use 1e10 for raw Gadget-style catalogs.")
    simba_grp.add_argument(
        "--simba-no-band", action="store_true",
        help="Suppress the 16-84th percentile shading on SIMBA lines")

    args = parser.parse_args()

    n = len(args.output_dirs)
    labels = args.labels if args.labels else [Path(d).name for d in args.output_dirs]
    if len(labels) != n:
        parser.error("--labels must match the number of output_dirs")

    # Load observational data
    obs = load_obs_data(args.obs_data)

    # Cycle through a few distinct colors (used for legend label; track is colored by z)
    track_colors = ["#e07b39", "#4682b4", "#55a868", "#c44e52", "#8172b2"]

    sim_tracks = []
    for i, (d, lbl) in enumerate(zip(args.output_dirs, labels)):
        color = track_colors[i % len(track_colors)]
        print(f"\nProcessing: {lbl}")
        z_arr, mstar_arr, mdust_arr = run_simulation(d, lbl, color,
                                                      skip_every=args.skip_every)
        sim_tracks.append((lbl, color, z_arr, mstar_arr, mdust_arr))

    # ------------------------------------------------------------------
    # SIMBA: resolve catalog file list
    # ------------------------------------------------------------------
    simba_catalog_paths = []

    if args.simba_catalogs:
        simba_catalog_paths = args.simba_catalogs

    elif args.simba_download:
        # Fetch redshift table first to know which snaps to grab
        redshift_table = fetch_simba_redshift_table()
        default_snaps = args.simba_snaps or sorted(SIMBA_DEFAULT_SNAPS.keys())
        print(f"\nDownloading SIMBA snapshots: {default_snaps}")
        for sn in default_snaps:
            p = download_simba_catalog(sn, args.simba_dir)
            if p:
                simba_catalog_paths.append(p)

    # Load SIMBA tracks if any catalogs available
    simba_tracks = None
    if simba_catalog_paths:
        print(f"\nLoading {len(simba_catalog_paths)} SIMBA catalog(s)...")
        # Fetch redshift table (needed if redshift not in catalog attributes)
        redshift_table = fetch_simba_redshift_table()
        simba_tracks = load_simba_tracks(
            simba_catalog_paths,
            redshift_table=redshift_table,
            mass_unit_msun=args.simba_mass_unit,
        )
        if not simba_tracks:
            print("WARNING: no valid SIMBA tracks loaded; SIMBA overlay will be absent.")
            simba_tracks = None

    make_plot(sim_tracks, args.output, obs,
              simba_tracks=simba_tracks,
              simba_show_band=not args.simba_no_band)


if __name__ == "__main__":
    main()
