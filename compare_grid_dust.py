"""
compare_grid_dust.py — Comparative dust physics analysis across grid runs
=========================================================================
Builds two families of plots:

  PART 1 — Log-based (fast, no snapshot I/O)
    - Dust mass vs redshift
    - Dust-to-Gas ratio vs redshift
    - Hash success rate vs redshift
    - Dust temperature distribution vs redshift
    - Destruction channel fractions vs redshift
    - Process rates (cumulative event counts) vs redshift
    - Summary panel (6 rows, N_dust on top)

  PART 2 — Snapshot-based (spatial/distribution, slower)
    - Grain size distribution at multiple redshifts
    - Carbon fraction PDF at z=0
    - Radial DGR profile at key redshifts
    - Gas phase diagram
    - Face-on dust surface density map

  PART 3 — LaTeX table output (--latex-out / --table-only)
    Writes a deluxetable* summarising z=0 statistics for all runs at the
    chosen resolution.  Columns: M*/Mhalo, <a> nm, f_surv, D/G, D/Z,
    Mdust/Mstar, f_carb.

Spatial scoping
---------------
ALL snapshot-based measurements and the LaTeX table use EXACTLY R200 as the
aperture radius, derived in this priority order:
  1. Group_R_Mean200 from the SubFind/FOF group catalog (most direct).
  2. First-principles calculation from Group_M_Mean200 + cosmological header.
  3. Hard fallback: 300 comoving kpc/h (logged as a warning).

Radii from the catalog are in comoving kpc/h (Gadget-4 code length units),
which is the same frame as particle Coordinates — so all distance cuts are
applied consistently in that frame.  Physical kpc = R_code * a / h.

Usage:
    python compare_grid_dust.py                          # all runs, 512^3, both parts
    python compare_grid_dust.py --res 1024               # use 1024^3 runs
    python compare_grid_dust.py --logs-only              # skip snapshot plots
    python compare_grid_dust.py --snaps-only             # skip log-based plots
    python compare_grid_dust.py --runs S0 S4 S10         # specific runs only
    python compare_grid_dust.py --latex-out table.tex    # also write LaTeX table
    python compare_grid_dust.py --table-only             # only write LaTeX table
                                                         # (auto-names dust_sim_ladder_{res}.tex)

Assumptions:
    - Log files:  ./{run}_output_{RESOLUTION}/output_{run}_{RESOLUTION}.log
    - Snapshots:  ./{run}_output_{RESOLUTION}/snapdir_NNN/snapshot_NNN.*.hdf5
    - Gadget-4 HDF5 snapshot format with PartType0 (gas), PartType4 (stars),
      PartType6 (dust)
"""

import re
import glob
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

UnitMass_in_g        = 1.989e43          # 1e10 M_sun in grams
UnitLength_in_cm     = 3.085678e21       # 1 comoving kpc/h in cm
UnitDensity_in_cgs   = UnitMass_in_g / UnitLength_in_cm**3
HYDROGEN_MASSFRAC    = 0.76
PROTON_MASS_CGS      = 1.6726219e-24    # g
BOLTZMANN_CGS        = 1.38064852e-16   # erg/K

RUN_CONFIGS = {
    'S0':  {'label': 'S0: Creation only',              'color': '#888888', 'ls': '-'},
    'S1':  {'label': 'S1: + Cooling',                  'color': '#1f77b4', 'ls': '-'},
    'S2':  {'label': 'S2: + Drag',                     'color': '#ff7f0e', 'ls': '-'},
    'S3':  {'label': 'S3: + Astration',                'color': '#2ca02c', 'ls': '-'},
    'S4':  {'label': 'S4: + Thermal sputtering',       'color': '#d62728', 'ls': '-'},
    'S5':  {'label': 'S5: + Grain growth',             'color': '#9467bd', 'ls': '-'},
    'S6':  {'label': 'S6: + Clumping factor',          'color': '#8c564b', 'ls': '-'},
    'S7':  {'label': 'S7: + SN shock destruction',     'color': '#e377c2', 'ls': '-'},
    'S8':  {'label': 'S8: + Coagulation',              'color': '#17becf', 'ls': '-'},
    'S9':  {'label': 'S9: + Shattering',               'color': '#bcbd22', 'ls': '-'},
    'S10': {'label': 'S10: + Rad. pressure (full)',    'color': '#000000', 'ls': '-'},
}

OBS_DGR_Z0         = 0.01
OBS_DGR_SIGMA      = 0.005
OBS_DUST_MASS_MSUN = 4.3e7
SNAP_REDSHIFTS     = [2.0, 1.0, 0.5, 0.0]

FIGDIR = 'dust_figures'
os.makedirs(FIGDIR, exist_ok=True)

RESOLUTION = 512   # updated by --res argument

# ─────────────────────────────────────────────────────────────────────────────
# R200 / halo utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_catalog(run, snap_base):
    """Return path to FOF/SubFind group catalog matching snap_base, or None."""
    m = re.search(r'snapshot_(\d+)$', snap_base)
    if not m:
        return None
    snap_num   = m.group(1)
    groups_dir = os.path.join(f'{run}_output_{RESOLUTION}', f'groups_{snap_num}')
    candidates = sorted(glob.glob(
        os.path.join(groups_dir, f'fof_subhalo_tab_{snap_num}.*.hdf5')))
    if candidates:
        return candidates[0]
    return None


def _read_header(snap_base):
    """Read cosmological/unit parameters from snapshot header."""
    import h5py
    defaults = dict(h=0.7, Omega0=0.3, OmegaL=0.7, a=1.0,
                    um_cgs=1.989e43, ul_cm=3.085678e21)
    for suffix in ['.0.hdf5', '.hdf5']:
        f = snap_base + suffix
        if not os.path.exists(f):
            continue
        try:
            with h5py.File(f, 'r') as hf:
                attrs = hf['Header'].attrs
                return dict(
                    h      = float(attrs.get('HubbleParam',    defaults['h'])),
                    Omega0 = float(attrs.get('Omega0',         defaults['Omega0'])),
                    OmegaL = float(attrs.get('OmegaLambda',    defaults['OmegaL'])),
                    a      = float(attrs.get('Time',           defaults['a'])),
                    um_cgs = float(attrs.get('UnitMass_in_g',     defaults['um_cgs'])),
                    ul_cm  = float(attrs.get('UnitLength_in_cm',  defaults['ul_cm'])),
                )
        except Exception:
            pass
    return defaults


def _compute_r200_from_m200(m200_code, snap_base):
    """
    Compute R200 in comoving kpc/h from M200 (code units) and snapshot header.
    Uses flat ΛCDM:  R200 = ( 3 M200 / (4π × 200 × ρ_crit(z)) )^(1/3)
    Returns R200 in comoving kpc/h (same frame as Gadget-4 Coordinates).
    """
    G_cgs      = 6.674e-8
    KM_IN_CM   = 1e5
    MPC_IN_CM  = 3.085678e24

    hdr = _read_header(snap_base)
    h, Omega0, OmegaL = hdr['h'], hdr['Omega0'], hdr['OmegaL']
    a, um_cgs, ul_cm  = hdr['a'], hdr['um_cgs'], hdr['ul_cm']

    H0_cgs      = 100.0 * h * KM_IN_CM / MPC_IN_CM
    Ez2         = Omega0 * a**-3 + OmegaL
    Hz_cgs      = H0_cgs * np.sqrt(Ez2)
    rho_crit    = 3.0 * Hz_cgs**2 / (8.0 * np.pi * G_cgs)

    m200_g      = m200_code * um_cgs
    r200_cm     = (3.0 * m200_g / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    r200_code   = r200_cm / ul_cm          # comoving kpc/h
    r200_phys   = r200_code * a / h        # physical kpc
    print(f'      [R200 computed]  M200={m200_code:.3f} code  '
          f'R200={r200_code:.1f} ckpc/h  ({r200_phys:.1f} pkpc, z={1/a-1:.3f})')
    return r200_code


def get_r200_and_center(run, snap_base, verbose=True, catalog_only=False):
    """
    Return (halo_center, R200_code) where both are in comoving kpc/h.

    Priority for R200:
      1. Group_R_Mean200 from SubFind catalog
      2. Derived from Group_M_Mean200 via _compute_r200_from_m200
      3. Hard fallback: 300 comoving kpc/h (warning printed)
         — suppressed when catalog_only=True (returns (None,None) instead)

    Priority for center:
      1. Group_Pos[0] from SubFind catalog
      2. Density-weighted centroid of all gas in the box
         — suppressed when catalog_only=True (returns (None,None) instead)

    catalog_only : bool
        If True, return (None, None) whenever the catalog is absent, has no
        Group key, has zero groups, or cannot provide R200.  Used by
        load_gas_mass_vs_z to avoid junk centroids from the full box.
    """
    import h5py

    catalog = find_catalog(run, snap_base)
    ctr     = None
    r200    = None

    # ── Try SubFind catalog ───────────────────────────────────────────────────
    if catalog is not None:
        try:
            with h5py.File(catalog, 'r') as hf:
                if 'Group' not in hf:
                    if verbose:
                        print(f'    [{run}] catalog has no "Group" key '
                              f'(top-level keys: {list(hf.keys())})')
                    if catalog_only:
                        return None, None
                else:
                    grp    = hf['Group']
                    n_grps = 0
                    if 'GroupPos' in grp:
                        n_grps = grp['GroupPos'].shape[0]
                    elif 'Group_M_Mean200' in grp:
                        n_grps = grp['Group_M_Mean200'].shape[0]

                    if n_grps == 0:
                        if verbose:
                            print(f'    [{run}] catalog has 0 groups — skipping snap')
                        return None, None   # always skip; no centroid for empty catalogs

                    if 'GroupPos' in grp:
                        ctr = grp['GroupPos'][0].astype(float)
                        if verbose:
                            print(f'    [{run}] center from GroupPos: {ctr}')

                    if 'Group_R_Mean200' in grp:
                        r200 = float(grp['Group_R_Mean200'][0])
                        if verbose:
                            hdr = _read_header(snap_base)
                            r200_phys = r200 * hdr['a'] / hdr['h']
                            print(f'    [{run}] R200 from catalog: '
                                  f'{r200:.1f} ckpc/h ({r200_phys:.1f} pkpc)')

                    if r200 is None and 'Group_M_Mean200' in grp:
                        m200 = float(grp['Group_M_Mean200'][0])
                        r200 = _compute_r200_from_m200(m200, snap_base)

                    if r200 is None:
                        if verbose:
                            print(f'    [{run}] could not get R200 from catalog. '
                                  f'Group keys: {list(grp.keys())}')
                        if catalog_only:
                            return None, None
        except Exception as e:
            print(f'    [{run}] catalog read error: {e}')
            if catalog_only:
                return None, None
    elif catalog_only:
        # No catalog file at all
        return None, None

    # ── Fallback center: density-weighted centroid ────────────────────────────
    # NOTE: only reached when catalog_only=False (interactive plot functions).
    if ctr is None:
        subfiles = sorted(glob.glob(snap_base + '.*.hdf5'))
        if not subfiles:
            single = snap_base + '.hdf5'
            subfiles = [single] if os.path.exists(single) else []
        all_pos, all_rho = [], []
        for fname in subfiles:
            try:
                with h5py.File(fname, 'r') as hf:
                    if 'PartType0' in hf:
                        all_pos.append(hf['PartType0']['Coordinates'][:])
                        all_rho.append(hf['PartType0']['Density'][:])
            except Exception:
                pass
        if all_pos:
            pos = np.concatenate(all_pos)
            rho = np.concatenate(all_rho)
            ctr = np.average(pos, weights=rho, axis=0)
            print(f'    [{run}] center fallback (density centroid): {ctr}')
        else:
            print(f'    [{run}] ERROR: cannot determine halo center')
            return None, None

    # ── Fallback R200 ─────────────────────────────────────────────────────────
    if r200 is None:
        r200 = 300.0
        print(f'    [{run}] WARNING: R200 fallback = {r200:.1f} ckpc/h')

    return ctr, r200


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot path helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_snapshots(run):
    output_dir = f'{run}_output_{RESOLUTION}'
    if not os.path.isdir(output_dir):
        return []
    seen_bases = set()
    bases = []
    for snapdir in sorted(glob.glob(os.path.join(output_dir, 'snapdir_*'))):
        if not os.path.isdir(snapdir):
            continue
        for f in sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.0.hdf5'))):
            base = re.sub(r'\.0\.hdf5$', '', f)
            if base not in seen_bases:
                seen_bases.add(base)
                bases.append(base)
        for f in sorted(glob.glob(os.path.join(snapdir, 'snapshot_*.hdf5'))):
            if '.0.hdf5' in f:
                continue
            base = re.sub(r'\.hdf5$', '', f)
            if base not in seen_bases:
                seen_bases.add(base)
                bases.append(base)
    return sorted(bases)


def snap_redshift(snap_base):
    import h5py
    for suffix in ['.hdf5', '.0.hdf5']:
        f = snap_base + suffix
        if os.path.exists(f):
            try:
                with h5py.File(f, 'r') as hf:
                    z = hf['Header'].attrs.get('Redshift',
                        hf['Header'].attrs.get('redshift', None))
                    if z is not None:
                        return float(z)
            except Exception:
                pass
    return None


def find_snap_near_z(snap_bases, target_z):
    best_base, best_dz = None, 1e30
    for sb in snap_bases:
        z = snap_redshift(sb)
        if z is not None and abs(z - target_z) < best_dz:
            best_dz = abs(z - target_z)
            best_base = sb
    return best_base, best_dz


def find_snap_near_z_with_catalog(run, snap_bases, target_z, dz_tol=0.3):
    """
    Like find_snap_near_z, but prefers snapshots that have a valid SubFind
    catalog with at least one group.  Falls back to any closest snapshot if
    no catalogued snapshot is within dz_tol.

    This prevents plot functions from landing on one of the many non-standard
    output snapshots (which lack catalogs) when a standard output snapshot
    with a catalog exists at nearly the same redshift.
    """
    import h5py

    # Collect (dz, snap_base, has_catalog) for all snaps within tolerance
    candidates = []
    for sb in snap_bases:
        z = snap_redshift(sb)
        if z is None:
            continue
        dz = abs(z - target_z)
        if dz > dz_tol:
            continue
        # Check for a usable catalog (exists + has Group with >0 entries)
        cat = find_catalog(run, sb)
        has_cat = False
        if cat is not None:
            try:
                with h5py.File(cat, 'r') as hf:
                    if 'Group' in hf:
                        grp = hf['Group']
                        n = grp['GroupPos'].shape[0] if 'GroupPos' in grp else 0
                        has_cat = (n > 0)
            except Exception:
                pass
        candidates.append((dz, sb, has_cat))

    if not candidates:
        return None, 1e30

    # Sort: catalog snapshots first, then by dz
    candidates.sort(key=lambda x: (not x[2], x[0]))
    best = candidates[0]
    return best[1], best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Particle loaders (all use rmax in comoving kpc/h = same frame as Coordinates)
# ─────────────────────────────────────────────────────────────────────────────

def _subfiles(snap_base):
    files = sorted(glob.glob(snap_base + '.*.hdf5'))
    if not files:
        single = snap_base + '.hdf5'
        files = [single] if os.path.exists(single) else []
    return files


def load_dust_for_snap(snap_base, halo_center, rmax):
    """Load PartType6 within rmax (comoving kpc/h) of halo_center."""
    import h5py
    pos_list, mass_list, radius_list, cfrac_list, temp_list = [], [], [], [], []
    for fname in _subfiles(snap_base):
        try:
            with h5py.File(fname, 'r') as hf:
                if 'PartType6' not in hf:
                    continue
                pt  = hf['PartType6']
                pos = pt['Coordinates'][:]
                r   = np.linalg.norm(pos - halo_center, axis=1)
                mask = r < rmax
                if mask.sum() == 0:
                    continue
                pos_list.append(pos[mask])
                mass_list.append(pt['Masses'][:][mask])
                radius_list.append(pt['GrainRadius'][:][mask])
                cfrac_list.append(pt['CarbonFraction'][:][mask])
                temp_list.append(pt['DustTemperature'][:][mask])
        except Exception as e:
            print(f'    load_dust: error reading {fname}: {e}')
    if not pos_list:
        return None
    return {
        'pos':          np.concatenate(pos_list),
        'mass':         np.concatenate(mass_list),
        'grain_radius': np.concatenate(radius_list),
        'carbon_frac':  np.concatenate(cfrac_list),
        'dust_temp':    np.concatenate(temp_list),
    }


def load_gas_for_snap(snap_base, halo_center, rmax):
    """Load PartType0 within rmax (comoving kpc/h) of halo_center."""
    import h5py
    pos_list, mass_list, dens_list, metal_list, ue_list = [], [], [], [], []
    has_metallicity = None
    for fname in _subfiles(snap_base):
        try:
            with h5py.File(fname, 'r') as hf:
                if 'PartType0' not in hf:
                    continue
                pt   = hf['PartType0']
                pos  = pt['Coordinates'][:]
                r    = np.linalg.norm(pos - halo_center, axis=1)
                mask = r < rmax
                if mask.sum() == 0:
                    continue
                pos_list.append(pos[mask])
                mass_list.append(pt['Masses'][:][mask])
                dens_list.append(pt['Density'][:][mask])
                if 'Metallicity' in pt:
                    metal_list.append(pt['Metallicity'][:][mask])
                    has_metallicity = True
                else:
                    has_metallicity = False
                ue_list.append(pt['InternalEnergy'][:][mask])
        except Exception as e:
            print(f'    load_gas: error reading {fname}: {e}')
    if not pos_list:
        return None
    return {
        'pos':             np.concatenate(pos_list),
        'mass':            np.concatenate(mass_list),
        'density':         np.concatenate(dens_list),
        'metallicity':     np.concatenate(metal_list) if has_metallicity else None,
        'internal_energy': np.concatenate(ue_list),
    }


def load_stars_for_snap(snap_base, halo_center, rmax):
    """Load PartType4 within rmax (comoving kpc/h) of halo_center."""
    import h5py
    pos_list, mass_list = [], []
    for fname in _subfiles(snap_base):
        try:
            with h5py.File(fname, 'r') as hf:
                if 'PartType4' not in hf:
                    continue
                pt   = hf['PartType4']
                pos  = pt['Coordinates'][:]
                r    = np.linalg.norm(pos - halo_center, axis=1)
                mask = r < rmax
                if mask.sum() == 0:
                    continue
                pos_list.append(pos[mask])
                mass_list.append(pt['Masses'][:][mask])
        except Exception as e:
            print(f'    load_stars: error reading {fname}: {e}')
    if not pos_list:
        return None
    return {
        'pos':  np.concatenate(pos_list),
        'mass': np.concatenate(mass_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gas mass for log-based D/G normalisation
# ─────────────────────────────────────────────────────────────────────────────

def load_gas_mass_vs_z(run, z_max=10.0):
    """
    Return (z_arr, gas_mass_arr) in code units by reading snapshots that have
    a valid SubFind group catalog (catalog_only=True), computing R200 and the
    halo center from that catalog, and summing total gas mass within R200.

    Snapshots without a catalog, with an empty Group table, or above z_max
    are silently skipped — the density-centroid fallback over a 50 Mpc box
    is unreliable and produces junk gas masses at high redshift.

    Returns two numpy arrays sorted by descending redshift (high-z first),
    or (None, None) if no valid snapshots are found.
    """
    snaps = find_snapshots(run)
    if not snaps:
        print(f'  [gas_mass_vs_z] {run}: no snapshots found')
        return None, None

    z_list, m_list = [], []
    n_no_catalog = 0
    n_high_z     = 0
    for snap_base in snaps:
        # Skip snapshots before halo formation
        z = snap_redshift(snap_base)
        if z is None:
            continue
        if z > z_max:
            n_high_z += 1
            continue

        # Require a valid catalog — no centroid fallback here
        ctr, r200 = get_r200_and_center(run, snap_base,
                                         verbose=False, catalog_only=True)
        if ctr is None:
            n_no_catalog += 1
            continue

        gas = load_gas_for_snap(snap_base, ctr, r200)
        if gas is None or len(gas['mass']) == 0:
            continue

        z_list.append(z)
        m_list.append(float(gas['mass'].sum()))

    if not z_list:
        print(f'  [gas_mass_vs_z] {run}: no valid snapshots found '
              f'(z>{z_max}: {n_high_z}, no/empty catalog: {n_no_catalog})')
        return None, None

    z_arr = np.array(z_list)
    m_arr = np.array(m_list)
    idx   = np.argsort(z_arr)[::-1]
    print(f'  [gas_mass_vs_z] {run}: {len(z_arr)} snapshots used, '
          f'z>[{z_max}] skipped: {n_high_z}, no/empty catalog: {n_no_catalog}, '
          f'z_range=[{z_arr.min():.2f}, {z_arr.max():.2f}], '
          f'M_gas(z~0)={m_arr[idx[-1]]:.4f} code')
    return z_arr[idx], m_arr[idx]


def get_gas_mass_curves(runs):
    """
    Build a dict  run -> (z_arr, gas_mass_arr)  for all runs that have
    accessible snapshots.  Used to give the summary D/G panel true evolution.
    """
    curves = {}
    print('\n  Loading gas mass vs redshift from snapshots (one scalar per snap)...')
    for run in runs:
        z_arr, m_arr = load_gas_mass_vs_z(run)
        if z_arr is not None:
            curves[run] = (z_arr, m_arr)
    if not curves:
        print('  WARNING: no gas mass curves loaded — D/G panel will be blank')
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def find_logfile(run):
    patterns = [
        f'{run}_output_{RESOLUTION}/output_{run}_{RESOLUTION}.log',
        f'{run}_output/output_{run}.log',
        f'output_{run}.log',
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    candidates = glob.glob(f'**/*{run}*{RESOLUTION}*.log', recursive=True)
    if candidates:
        print(f'  [find_logfile] fallback glob: {candidates[0]}')
        return candidates[0]
    return None


def parse_log(logfile):
    data = defaultdict(list)

    re_stats_header = re.compile(r'STATISTICS Particles:\s+(\d+)\s+Mass:\s+([\d.e+\-]+)')
    re_avgsize       = re.compile(r'Avg grain size:\s+([\d.]+) nm')
    re_avgtemp       = re.compile(r'Avg temperature:\s+([\d.]+) K')
    re_hashrate      = re.compile(r'Hash success rate:\s+([\d.]+)%')
    re_growth        = re.compile(r'Growth events:\s+(\d+)\s+\(([\d.e+\-]+) Msun grown\)')
    re_erosion       = re.compile(r'Partial erosion events:\s+(\d+)')
    re_coag          = re.compile(r'Coagulation events:\s+(\d+)')
    re_shatter       = re.compile(r'Shattering events:\s+(\d+)')
    re_flags         = re.compile(
        r'DUST_FLAGS.*Creation=(\d).*Drag=(\d).*Growth=(\d).*'
        r'Coagulation=(\d).*Sputtering=(\d).*ShockDestruction=(\d).*'
        r'Astration=(\d).*RadPressure=(\d).*Clumping=(\d)')

    with open(logfile) as f:
        content = f.read()

    blocks = re.split(r'=== STATISTICS \(global\) ===', content)

    for block in blocks[1:]:
        m = re.search(r'\|a=([\d.]+) z=([\d.]+)\]', block)
        if not m:
            continue
        a = float(m.group(1))
        z = float(m.group(2))

        m2 = re_stats_header.search(block)
        if not m2:
            continue
        n_part = int(m2.group(1))
        mass   = float(m2.group(2))

        m3 = re_avgsize.search(block);   avg_size     = float(m3.group(1)) if m3 else np.nan
        m4 = re_avgtemp.search(block);   avg_temp     = float(m4.group(1)) if m4 else np.nan
        m5 = re_hashrate.search(block);  hash_success = float(m5.group(1)) if m5 else np.nan
        m6 = re_growth.search(block)
        growth_events = int(m6.group(1))   if m6 else 0
        mass_grown    = float(m6.group(2)) if m6 else 0.0
        m7 = re_erosion.search(block);   erosion_events = int(m7.group(1)) if m7 else 0
        m8 = re_coag.search(block);      coag_events    = int(m8.group(1)) if m8 else 0
        m9 = re_shatter.search(block);   shatter_events = int(m9.group(1)) if m9 else 0

        m_tb = re.search(r'< 10 K.*?:\s+(\d+).*?10-50 K.*?:\s+(\d+)', block, re.DOTALL)
        cmb_frac  = int(m_tb.group(1)) / n_part if m_tb and n_part > 0 else np.nan
        cold_frac = int(m_tb.group(2)) / n_part if m_tb and n_part > 0 else np.nan

        m_th = re.search(r'Thermal sputtering:\s+(\d+)', block)
        m_sh = re.search(r'Shock destruction:\s+(\d+)', block)
        m_as = re.search(r'Astration:\s+(\d+)', block)
        n_thermal = int(m_th.group(1)) if m_th else 0
        n_shock   = int(m_sh.group(1)) if m_sh else 0
        n_astrat  = int(m_as.group(1)) if m_as else 0

        data['z'].append(z);               data['a'].append(a)
        data['n_part'].append(n_part);     data['mass'].append(mass)
        data['avg_size'].append(avg_size); data['avg_temp'].append(avg_temp)
        data['hash_success'].append(hash_success)
        data['growth_events'].append(growth_events)
        data['mass_grown'].append(mass_grown)
        data['erosion_events'].append(erosion_events)
        data['coag_events'].append(coag_events)
        data['shatter_events'].append(shatter_events)
        data['cmb_frac'].append(cmb_frac); data['cold_frac'].append(cold_frac)
        data['n_thermal'].append(n_thermal)
        data['n_shock'].append(n_shock);   data['n_astrat'].append(n_astrat)

    m_fl = re_flags.search(content)
    data['flags'] = {
        'creation': int(m_fl.group(1)), 'drag':      int(m_fl.group(2)),
        'growth':   int(m_fl.group(3)), 'coag':      int(m_fl.group(4)),
        'sputtering': int(m_fl.group(5)), 'shock':   int(m_fl.group(6)),
        'astration': int(m_fl.group(7)), 'radpressure': int(m_fl.group(8)),
        'clumping': int(m_fl.group(9)),
    } if m_fl else {}

    arr_keys = ['z','a','n_part','mass','avg_size','avg_temp','hash_success',
                'mass_grown','cmb_frac','cold_frac','growth_events',
                'erosion_events','coag_events','shatter_events',
                'n_thermal','n_shock','n_astrat']
    for key in arr_keys:
        data[key] = np.array(data[key], dtype=float)

    # Sort descending in redshift (high-z first, z~0 last)
    idx = np.argsort(data['z'])[::-1]
    for key in arr_keys:
        data[key] = data[key][idx]

    return dict(data)


def load_all_logs(runs):
    results = {}
    for run in runs:
        logfile = find_logfile(run)
        if logfile is None:
            print(f'  WARNING: no log found for {run}, skipping')
            continue
        print(f'  Parsing {logfile}...')
        try:
            results[run] = parse_log(logfile)
            n     = len(results[run]['z'])
            z_min = results[run]['z'].min() if n > 0 else np.nan
            print(f'    -> {n} blocks, z_min={z_min:.3f}')
        except Exception as e:
            print(f'    ERROR: {e}')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def style(run):
    cfg = RUN_CONFIGS.get(run, {})
    return {'color': cfg.get('color', 'black'), 'ls': cfg.get('ls', '-'),
            'lw': 2.0, 'label': cfg.get('label', run)}


def active_runs(log_data):
    return [run for run, d in log_data.items() if len(d.get('z', [])) > 0]


def legend_handles(log_data):
    runs    = active_runs(log_data)
    handles = [plt.Line2D([0], [0], **style(run)) for run in runs]
    labels  = [RUN_CONFIGS.get(run, {}).get('label', run) for run in runs]
    return handles, labels


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 plots — log-based
# ─────────────────────────────────────────────────────────────────────────────

def plot_dust_mass(log_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        ax.plot(d['z'], d['mass'] * 1e10, **style(run))
    ax.axhline(OBS_DUST_MASS_MSUN, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'Total Dust Mass ($M_\odot$)')
    ax.set_title(f'Dust Mass Evolution ({RESOLUTION}$^3$)')
    ax.set_xlim(4.5, 0); ax.set_yscale('log')
    handles, labels = legend_handles(log_data)
    handles.append(plt.Line2D([0], [0], color='k', ls='--', lw=1.5))
    labels.append('MW (Li & Draine 2001)')
    ax.legend(handles, labels, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    savefig(fig, 'dust_mass_vs_z.png')


def plot_dtg(log_data, gas_mass_curves=None):
    """
    Standalone dust-to-gas ratio vs redshift plot.
    gas_mass_curves : dict  run -> (z_arr, gas_mass_arr)  from get_gas_mass_curves()
    Each run's dust mass is divided by its own interpolated gas mass at each
    redshift — identical logic to the summary panel D/G row.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        if gas_mass_curves and run in gas_mass_curves:
            z_gas, m_gas = gas_mass_curves[run]
            m_gas_interp = np.interp(d['z'], z_gas[::-1], m_gas[::-1])
            good = m_gas_interp > 0
            dtg  = np.where(good, d['mass'] / m_gas_interp, np.nan)
            ax.plot(d['z'], dtg, **style(run))
        else:
            # No gas curve for this run — skip rather than show misleading data
            pass
    handles, labels = legend_handles(log_data)
    if gas_mass_curves:
        ax.axhspan(OBS_DGR_Z0 - OBS_DGR_SIGMA, OBS_DGR_Z0 + OBS_DGR_SIGMA,
                   alpha=0.15, color='gray')
        handles.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15))
        labels.append(r'MW (Draine & Li 2007)')
        ax.set_ylabel('Dust-to-Gas Ratio')
    else:
        ax.text(0.5, 0.5, 'Gas mass curves not available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='gray', style='italic')
        ax.set_ylabel('Dust-to-Gas Ratio')
    ax.set_xlabel('Redshift $z$')
    ax.set_title(f'Dust-to-Gas Ratio Evolution ({RESOLUTION}$^3$, per-run $M_{{\\rm gas}}(z)$)')
    ax.set_xlim(4.5, 0); ax.set_yscale('log')
    ax.legend(handles, labels, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    savefig(fig, 'dtg_vs_z.png')


def plot_hash_success(log_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        ax.plot(d['z'], d['hash_success'], **style(run))
    ax.axhline(80, color='gray', ls=':', lw=1)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Hash Search Success Rate (%)')
    ax.set_title(f'Dust-Gas Spatial Coupling ({RESOLUTION}$^3$)\n(Hash Success Rate)')
    ax.set_xlim(4.5, 0); ax.set_ylim(0, 105)
    handles, labels = legend_handles(log_data)
    handles.append(plt.Line2D([0], [0], color='gray', ls=':', lw=1))
    labels.append('80% reference')
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'hash_success_vs_z.png')


def plot_grain_size_log(log_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        ax.plot(d['z'], d['avg_size'], **style(run))
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Mean Grain Radius (nm)')
    ax.set_title(f'Mean Grain Size Evolution ({RESOLUTION}$^3$)')
    ax.set_xlim(4.5, 0)
    handles, labels = legend_handles(log_data)
    handles.append(plt.Line2D([0], [0], color='gray', ls=':', lw=1))
    labels.append('MRN mean (~100 nm)')
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'grain_size_vs_z.png')


def plot_dust_temp(log_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        ax.plot(d['z'], d['avg_temp'], **style(run))
    z_arr = np.linspace(0, 5, 200)
    ax.plot(z_arr, 2.73 * (1 + z_arr), 'k--', lw=1.5)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Mean Dust Temperature (K)')
    ax.set_title(f'Dust Temperature Evolution ({RESOLUTION}$^3$)')
    ax.set_xlim(4.5, 0); ax.set_yscale('log')
    handles, labels = legend_handles(log_data)
    handles.append(plt.Line2D([0], [0], color='k', ls='--', lw=1.5))
    labels.append('$T_{\\rm CMB}$')
    ax.legend(handles, labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'dust_temp_vs_z.png')


def plot_destruction_channels(log_data):
    runs_with_dest = {r: d for r, d in log_data.items()
                      if np.any(d['n_thermal'] > 0) or np.any(d['n_shock'] > 0)
                      or np.any(d['n_astrat'] > 0)}
    if not runs_with_dest:
        print('  No destruction events found — skipping')
        return
    n   = len(runs_with_dest)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1: axes = [axes]
    for ax, (run, d) in zip(axes, runs_with_dest.items()):
        total = np.where(d['n_thermal'] + d['n_shock'] + d['n_astrat'] == 0, 1,
                         d['n_thermal'] + d['n_shock'] + d['n_astrat'])
        ax.stackplot(d['z'],
                     d['n_thermal'] / total * 100,
                     d['n_shock']   / total * 100,
                     d['n_astrat']  / total * 100,
                     labels=['Thermal sputtering', 'Shock destruction', 'Astration'],
                     colors=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_xlabel('Redshift $z$'); ax.set_ylabel('Fraction (%)')
        ax.set_title(RUN_CONFIGS.get(run, {}).get('label', run))
        ax.set_xlim(4.5, 0); ax.set_ylim(0, 100)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Dust Destruction Channels', fontsize=12)
    plt.tight_layout()
    savefig(fig, 'destruction_channels.png')


def plot_process_rates(log_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_gr, ax_er, ax_co, ax_sh = axes.flat
    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        s = style(run)
        def rate(arr):
            r = np.diff(arr, prepend=0.0)
            return np.where(r < 0, 0, r)
        ax_gr.plot(d['z'], np.cumsum(rate(d['growth_events'])),  **s)
        ax_er.plot(d['z'], np.cumsum(rate(d['erosion_events'])), **s)
        ax_co.plot(d['z'], np.cumsum(rate(d['coag_events'])),    **s)
        ax_sh.plot(d['z'], np.cumsum(rate(d['shatter_events'])), **s)
    for ax, title in zip(axes.flat,
                         ['Grain Growth (cumulative)', 'Thermal Erosion (cumulative)',
                          'Coagulation (cumulative)', 'Shattering (cumulative)']):
        ax.set_xlabel('Redshift $z$'); ax.set_ylabel('Event count')
        ax.set_title(title); ax.set_xlim(4.5, 0)
        ax.set_yscale('symlog', linthresh=1)
        ax.grid(True, alpha=0.3)
    handles, labels = legend_handles(log_data)
    ax_gr.legend(handles, labels, fontsize=7, loc='upper right')
    fig.suptitle(f'Dust Physics Process Activity ({RESOLUTION}$^3$)', fontsize=13)
    plt.tight_layout()
    savefig(fig, 'process_rates.png')


def plot_summary_panel(log_data, gas_mass_curves=None):
    """
    Six-panel summary plot.  Panel order (top to bottom):
      1. N_dust (particle count)
      2. Dust-to-Gas Ratio  ← per-run gas mass interpolated from snapshots
      3. Dust Mass (code units)
      4. Mean Grain Radius (nm)
      5. Mean Dust Temperature (K)
      6. Hash Success Rate (%)

    gas_mass_curves : dict  run -> (z_arr, gas_mass_arr)
        Built by get_gas_mass_curves().  If None or empty the D/G panel
        shows a "not available" annotation instead of duplicating mass.
    """
    fig, axes = plt.subplots(6, 1, figsize=(6, 13), sharex=True,
                              gridspec_kw=dict(hspace=0.08))
    ax_np, ax_dtg, ax_mass, ax_size, ax_temp, ax_hash = axes

    for run, d in log_data.items():
        if len(d['z']) == 0: continue
        s = style(run)
        z = d['z']
        ax_np.plot(  z, d['n_part'],       **s)
        ax_mass.plot(z, d['mass'],         **s)
        ax_size.plot(z, d['avg_size'],     **s)
        ax_temp.plot(z, d['avg_temp'],     **s)
        ax_hash.plot(z, d['hash_success'], **s)

        # D/G: interpolate this run's gas mass curve onto the log redshift grid
        if gas_mass_curves and run in gas_mass_curves:
            z_gas, m_gas = gas_mass_curves[run]
            # np.interp needs x increasing; arrays are z-descending → flip
            m_gas_interp = np.interp(z, z_gas[::-1], m_gas[::-1])
            good = m_gas_interp > 0
            dtg  = np.where(good, d['mass'] / m_gas_interp, np.nan)
            ax_dtg.plot(z, dtg, **s)

    # CMB reference on temperature panel
    z_arr = np.linspace(0, 5, 300)
    ax_temp.plot(z_arr, 2.73 * (1 + z_arr), 'k--', lw=1.2, zorder=0)
    ax_temp.text(4.3, 2.73 * 5.3, '$T_{\\rm CMB}$', fontsize=7,
                 ha='left', va='bottom')

    # Annotate D/G panel if no curves available
    if not gas_mass_curves:
        ax_dtg.text(0.5, 0.5,
                    'Gas mass unavailable\n(run without snapshot access)',
                    transform=ax_dtg.transAxes, ha='center', va='center',
                    fontsize=8, color='gray', style='italic')

    panel_specs = [
        (ax_np,   '$N_{\\rm dust}$',                       'linear'),
        (ax_dtg,  'Dust-to-Gas Ratio',                     'log'),
        (ax_mass, r'Dust Mass ($10^{10}\,M_\odot$)',        'log'),
        (ax_size, 'Mean Grain Radius (nm)',                 'linear'),
        (ax_temp, 'Mean Dust Temp (K)',                     'log'),
        (ax_hash, 'Hash Success (%)',                       'linear'),
    ]
    for ax, ylabel, yscale in panel_specs:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
        ax.set_yscale(yscale)
        ax.set_xlim(4.5, 0)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7, pad=2)
        if yscale == 'log':
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10, numticks=10))
            ax.yaxis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=10,
                    subs=np.arange(2, 10) * 0.1, numticks=60))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=4, prune='both'))

    ax_hash.set_ylim(0, 105)
    ax_hash.set_xlabel('Redshift $z$', fontsize=9)

    handles, labels = legend_handles(log_data)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.suptitle(f'Dust Grid Comparison ({RESOLUTION}$^3$)', fontsize=10, y=0.987)
    fig.legend(handles, labels,
               fontsize=8,          # ← increased from 6
               ncol=2,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.962),
               framealpha=0.9,
               borderpad=0.4,
               handlelength=1.5,
               columnspacing=0.8,
               borderaxespad=0.0)
    savefig(fig, 'summary_panel.png')


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 plots — snapshot-based (all apertures = R200 from SubFind)
# ─────────────────────────────────────────────────────────────────────────────

def plot_grain_size_distribution(runs, target_redshifts=None):
    """Grain size distributions for multiple redshifts; aperture = R200."""
    if target_redshifts is None:
        target_redshifts = SNAP_REDSHIFTS

    z_linestyles = ['-', '--', ':', '-.']
    z_labels     = [f'$z \\approx {z}$' for z in target_redshifts]
    bins         = np.logspace(np.log10(1), np.log10(250), 40)

    fig, ax = plt.subplots(figsize=(9, 6))
    any_plotted  = False
    plotted_runs = []

    for run in runs:
        snaps = find_snapshots(run)
        if not snaps: continue
        color    = RUN_CONFIGS.get(run, {}).get('color', 'black')
        run_plotted = False

        for iz, (target_z, ls) in enumerate(zip(target_redshifts, z_linestyles)):
            snap_base, dz = find_snap_near_z_with_catalog(run, snaps, target_z)
            if snap_base is None or dz > 0.3: continue
            ctr, r200 = get_r200_and_center(run, snap_base, verbose=False)
            if ctr is None: continue
            d = load_dust_for_snap(snap_base, ctr, r200)
            if d is None or len(d.get('grain_radius', [])) == 0: continue

            counts, edges = np.histogram(d['grain_radius'], bins=bins,
                                         weights=d['mass'])
            centers = 0.5 * (edges[:-1] + edges[1:])
            dlog_a  = np.diff(np.log10(edges))
            norm    = counts / dlog_a
            if norm.sum() > 0: norm /= norm.sum()
            ax.plot(centers, norm, color=color, ls=ls, lw=1.8)
            any_plotted = True
            run_plotted = True
        if run_plotted:
            plotted_runs.append(run)

    if not any_plotted:
        print('  No grain size data found — skipping')
        plt.close(fig); return

    a_ref = np.logspace(np.log10(5), np.log10(200), 100)
    mrn   = a_ref ** (-2.5); mrn /= mrn.sum()
    ax.plot(a_ref, mrn, color='gray', ls='-', lw=2)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Grain Radius $a$ (nm)')
    ax.set_ylabel('Normalized mass-weighted $dN/d\\log a$')
    ax.set_title(f'Grain Size Distributions ({RESOLUTION}$^3$, within $R_{{200}}$)')
    ax.grid(True, alpha=0.3)

    run_handles = [plt.Line2D([0], [0],
                              color=RUN_CONFIGS.get(r, {}).get('color', 'k'), lw=1.8,
                              label=RUN_CONFIGS.get(r, {}).get('label', r))
                   for r in plotted_runs]
    run_handles.append(plt.Line2D([0], [0], color='gray', lw=2, label='MRN ($a^{-3.5}$)'))
    leg1 = ax.legend(handles=run_handles, fontsize=7, loc='upper left',
                     title='Run', title_fontsize=7)
    ax.add_artist(leg1)
    z_handles = [plt.Line2D([0], [0], color='gray', ls=ls, lw=1.8, label=zlbl)
                 for ls, zlbl in zip(z_linestyles[:len(target_redshifts)], z_labels)]
    ax.legend(handles=z_handles, fontsize=7, loc='upper right',
              title='Redshift', title_fontsize=7)

    savefig(fig, 'grain_size_dist_allz.png')


def plot_carbon_fraction_pdf(runs, target_z=0.0):
    """Carbon fraction PDF at target_z; aperture = R200."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 50)
    for run in runs:
        snaps = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z_with_catalog(run, snaps, target_z)
        if dz > 0.2: continue
        ctr, r200 = get_r200_and_center(run, snap_base, verbose=False)
        if ctr is None: continue
        d = load_dust_for_snap(snap_base, ctr, r200)
        if d is None or len(d.get('carbon_frac', [])) == 0: continue
        ax.hist(d['carbon_frac'], bins=bins, weights=d['mass'], density=True,
                histtype='step',
                **{k: v for k, v in style(run).items() if k != 'lw'},
                linewidth=2)
    ax.axvline(0.1, color='gray', ls=':', lw=1.2, label='SNII spawn (0.1)')
    ax.axvline(0.6, color='gray', ls='--', lw=1.2, label='AGB spawn (0.6)')
    ax.set_xlabel('Carbon Fraction')
    ax.set_ylabel('Mass-weighted PDF')
    ax.set_title(f'Carbon Fraction Distribution at $z \\approx {target_z}$'
                 f'  (within $R_{{200}}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, f'carbon_fraction_pdf_z{target_z:.1f}.png')


def plot_radial_dgr(runs, target_z=0.0):
    """
    Radial D/G and D/Z profiles in physical kpc.
    Particle coordinates (comoving kpc/h) are converted to physical kpc via
    r_phys = r_code * a / h before binning, so the x-axis is directly
    comparable to observational references like Mattsson+2012.
    """
    fig, (ax_dgr, ax_dtm) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                          gridspec_kw=dict(hspace=0.06))
    for run in runs:
        snaps = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z_with_catalog(run, snaps, target_z)
        if dz > 0.2: continue
        ctr, r200 = get_r200_and_center(run, snap_base, verbose=False)
        if ctr is None: continue

        # Conversion factor: comoving kpc/h → physical kpc
        hdr   = _read_header(snap_base)
        to_pkpc = hdr['a'] / hdr['h']

        r200_pkpc    = r200 * to_pkpc
        r_max_plot   = min(r200_pkpc, 50.0)
        r_bins       = np.arange(0, r_max_plot + 5, 5)        # physical kpc
        r_centers    = 0.5 * (r_bins[:-1] + r_bins[1:])       # physical kpc

        dust = load_dust_for_snap(snap_base, ctr, r200)
        gas  = load_gas_for_snap(snap_base,  ctr, r200)
        if dust is None or gas is None: continue

        # Convert particle radii to physical kpc for binning
        r_gas  = np.linalg.norm(gas['pos']  - ctr, axis=1) * to_pkpc
        r_dust = np.linalg.norm(dust['pos'] - ctr, axis=1) * to_pkpc

        gas_mass_r,  _ = np.histogram(r_gas,  bins=r_bins, weights=gas['mass'])
        dust_mass_r, _ = np.histogram(r_dust, bins=r_bins, weights=dust['mass'])

        if gas['metallicity'] is not None:
            Z = gas['metallicity']
            if Z.ndim == 2: Z = Z[:, 0]
            metal_mass_r, _ = np.histogram(r_gas, bins=r_bins,
                                            weights=gas['mass'] * Z)
        else:
            metal_mass_r = np.zeros_like(gas_mass_r)

        median_gas = np.nanmedian(gas_mass_r[gas_mass_r > 0]) \
                     if np.any(gas_mass_r > 0) else 1.0
        good = gas_mass_r > 0.01 * median_gas
        with np.errstate(invalid='ignore', divide='ignore'):
            dgr = np.where(good & (gas_mass_r  > 0), dust_mass_r / gas_mass_r,  np.nan)
            dtm = np.where(good & (metal_mass_r > 0), dust_mass_r / metal_mass_r, np.nan)

        s = style(run)
        ax_dgr.plot(r_centers, dgr, marker='o', ms=4, **s, alpha=0.9)
        ax_dtm.plot(r_centers, dtm, marker='o', ms=4, **s, alpha=0.9)

    # D/G reference: simple exponential disk approximation for MW
    # D/G(R) ~ D/G_0 * exp(-(R-R_sun)/h) with h~5 kpc, normalized to
    # local value 0.01 at R_sun=8 kpc (consistent with Draine et al. 2007,
    # Mattsson & Andersen 2012 SINGS profiles)
    r_ref = np.linspace(0.5, 50, 300)
    ax_dgr.plot(r_ref, OBS_DGR_Z0 * np.exp(-(r_ref - 8.0) / 5.0), 'k--', lw=2.0,
                label='MW exponential disk (Draine+2007 approx.)')

    # D/Z reference: Galactic solar-neighbourhood value ζ_G ≈ 0.5
    # with ±0.3 dex scatter (Mattsson & Andersen 2012, MNRAS 423, 38;
    # Zafar & Watson 2013, A&A 560, A26).
    # Negative D/Z slope = ISM grain growth dominates;
    # positive slope = SN destruction dominates (Mattsson et al. 2012 Paper I).
    zeta_G      = 0.5    # Galactic dust-to-metals ratio (solar neighbourhood)
    zeta_lo     = zeta_G * 10**(-0.3)
    zeta_hi     = zeta_G * 10**(+0.3)
    ax_dtm.axhspan(zeta_lo, zeta_hi, alpha=0.12, color='gray', zorder=0)
    ax_dtm.axhline(zeta_G,  color='gray', ls='--', lw=1.8, zorder=1,
                   label=r'MW \approx 0.5$ $\pm$0.3 dex (Mattsson et al 2012)')

    for ax, ylabel in [(ax_dgr, 'Dust-to-Gas Ratio'),
                       (ax_dtm, 'Dust-to-Metals Ratio')]:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_yscale('log')
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))

    handles, labels = legend_handles({r: log_data_global.get(r, {'z': []})
                                      for r in runs
                                      if r in log_data_global})
    ax_dgr.legend(handles + [plt.Line2D([0], [0], color='k', ls='--', lw=2)],
                  labels  + ['MW exponential disk (Draine+2007 approx.)'],
                  fontsize=7, loc='lower left')
    ax_dtm.legend(
        [plt.Line2D([0], [0], color='gray', ls='--', lw=1.8),
         plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25)] +
        [plt.Line2D([0], [0], **style(r))
         for r in runs if r in log_data_global],
        [r'MW \approx 0.5$ (Mattsson+2012)',
         r'$\pm$0.3 dex scatter'] +
        [RUN_CONFIGS.get(r, {}).get('label', r)
         for r in runs if r in log_data_global],
        fontsize=7, loc='upper right')
    ax_dtm.set_xlabel('Galactocentric Radius (physical kpc)', fontsize=12)
    ax_dgr.set_title(f'Radial Dust Profiles at $z \\approx {target_z}$', fontsize=13)
    plt.tight_layout()
    savefig(fig, f'radial_dgr_z{target_z:.1f}.png')


def plot_phase_diagram_dust(runs, target_z=0.0):
    """Gas T-nH phase diagram within R200."""
    unit_vel = 1e5
    mu, gamma = 0.62, 5.0 / 3.0

    run_data = {}
    for run in runs:
        snaps = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z_with_catalog(run, snaps, target_z)
        if dz > 0.2: continue
        ctr, r200 = get_r200_and_center(run, snap_base)
        if ctr is None: continue
        gas = load_gas_for_snap(snap_base, ctr, r200)
        if gas is not None:
            run_data[run] = gas

    if not run_data:
        print('  No snapshot data for phase diagram, skipping')
        return

    n   = len(run_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1: axes = [axes]

    for ax, (run, gas) in zip(axes, run_data.items()):
        u_cgs = gas['internal_energy'] * unit_vel ** 2
        T     = (gamma - 1.0) * u_cgs * mu * PROTON_MASS_CGS / BOLTZMANN_CGS
        nH    = gas['density'] * UnitDensity_in_cgs * HYDROGEN_MASSFRAC / PROTON_MASS_CGS
        c     = gas['metallicity'] if gas['metallicity'] is not None \
                else np.log10(nH + 1e-6)

        sc = ax.scatter(np.log10(nH + 1e-6), np.log10(T + 1), c=c,
                        s=0.5, alpha=0.4, cmap='plasma',
                        vmin=0, vmax=0.03, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Metallicity', pad=0.02)
        ax.set_xlabel('$\\log_{10}\\,n_H$ (cm$^{-3}$)')
        ax.set_title(RUN_CONFIGS.get(run, {}).get('label', run), fontsize=9)

        ymin, ymax = ax.get_ylim()
        for xval, label in [(-5.5, 'IGM'), (-3.5, 'CGM'), (-1.0, 'ISM')]:
            ax.axvline(xval, color='gray', lw=0.9, ls='--', alpha=0.7, zorder=3)
            ax.text(xval + 0.08, ymax - 0.05 * (ymax - ymin), label,
                    color='gray', fontsize=7, va='top', ha='left', alpha=0.95, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
        ax.set_yticks([4, 5, 6, 7])
        ax.set_yticklabels(['4', '5', '6', '7'], fontsize=8)
        ax.tick_params(labelleft=True)

    axes[0].set_ylabel('$\\log_{10}\\,T$ (K)')
    fig.suptitle(f'Gas Phase Diagram at $z \\approx {target_z}$'
                 f'  (within $R_{{200}}$)', fontsize=12)
    plt.tight_layout()
    savefig(fig, f'phase_diagram_z{target_z:.1f}.png')


def plot_dust_map(runs, target_z=0.0, size_kpc=50.0, npix=256):
    """Face-on dust surface density map, loaded within R200."""
    run_maps = {}
    for run in runs:
        snaps = find_snapshots(run)
        if not snaps: continue
        snap_base, dz = find_snap_near_z_with_catalog(run, snaps, target_z)
        if dz > 0.2: continue
        ctr, r200 = get_r200_and_center(run, snap_base, verbose=False)
        if ctr is None: continue
        dust = load_dust_for_snap(snap_base, ctr, r200)
        if dust is None: continue
        dx   = dust['pos'][:, 0] - ctr[0]
        dy   = dust['pos'][:, 1] - ctr[1]
        half = size_kpc / 2.0
        bins = np.linspace(-half, half, npix + 1)
        img, _, _ = np.histogram2d(dx, dy, bins=[bins, bins], weights=dust['mass'])
        run_maps[run] = img

    if not run_maps: return
    n   = len(run_maps)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1: axes = [axes]
    for ax, (run, img) in zip(axes, run_maps.items()):
        nonzero = img[img > 0]
        if nonzero.size == 0:
            ax.set_title(f'{RUN_CONFIGS.get(run,{}).get("label",run)}\n(no data)')
            continue
        img_log = np.log10(img + nonzero.min() * 0.01)
        im = ax.imshow(img_log.T, origin='lower',
                       extent=[-size_kpc/2, size_kpc/2, -size_kpc/2, size_kpc/2],
                       cmap='inferno', aspect='equal')
        plt.colorbar(im, ax=ax, label='$\\log_{10}\\Sigma_{\\rm dust}$', pad=0.02)
        ax.set_xlabel('$x$ (physical kpc)')
        ax.set_title(RUN_CONFIGS.get(run, {}).get('label', run), fontsize=9)
    axes[0].set_ylabel('$y$ (physical kpc)')
    fig.suptitle(f'Dust Surface Density at $z \\approx {target_z}$'
                 f'  (within $R_{{200}}$)', fontsize=12)
    plt.tight_layout()
    savefig(fig, f'dust_map_z{target_z:.1f}.png')


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Table statistics + LaTeX writer
# ─────────────────────────────────────────────────────────────────────────────

def compute_table_stats(runs, log_data):
    """
    Compute per-run z~0 statistics for the LaTeX table.
    Aperture = exactly R200 from SubFind (same as all snapshot plots).

    Returns dict  run -> {Mstar_over_Mhalo, mean_a_nm, f_surv, DtoG, DtoZ,
                           Mdust_over_Mstar, f_carb, z_snap}
    """
    table = {}

    for run in runs:
        print(f'\n  [{run}] computing table stats...')
        ts = {}

        snaps = find_snapshots(run)
        if not snaps:
            print(f'    no snapshots found')
            table[run] = ts; continue

        snap_base, dz = find_snap_near_z_with_catalog(run, snaps, 0.0)
        if dz > 0.5:
            print(f'    WARNING: closest snapshot is dz={dz:.2f} from z=0')

        ts['z_snap'] = snap_redshift(snap_base) or 0.0
        hdr          = _read_header(snap_base)
        msun_per_code = hdr['um_cgs'] / 1.989e33

        # Halo centre and R200
        ctr, r200 = get_r200_and_center(run, snap_base, verbose=True)
        if ctr is None:
            print(f'    cannot determine halo center — skipping')
            table[run] = ts; continue

        print(f'    aperture = R200 = {r200:.1f} ckpc/h '
              f'({r200 * hdr["a"] / hdr["h"]:.1f} pkpc)')

        # M200 from catalog (for M*/Mhalo)
        catalog = find_catalog(run, snap_base)
        m200_code = None
        if catalog is not None:
            try:
                import h5py
                with h5py.File(catalog, 'r') as hf:
                    if 'Group' in hf and 'Group_M_Mean200' in hf['Group']:
                        m200_code = float(hf['Group']['Group_M_Mean200'][0])
            except Exception:
                pass

        # Stellar mass
        stars = load_stars_for_snap(snap_base, ctr, r200)
        M_star_code = float(stars['mass'].sum()) \
                      if (stars and len(stars['mass']) > 0) else 0.0
        ts['M_star_msun']       = M_star_code * msun_per_code
        ts['Mstar_over_Mhalo']  = (M_star_code / m200_code) \
                                   if (m200_code and m200_code > 0) else None

        # Dust
        dust = load_dust_for_snap(snap_base, ctr, r200)
        if dust and len(dust.get('mass', [])) > 0:
            M_dust_code = float(dust['mass'].sum())
            ts['mean_a_nm'] = float(np.average(dust['grain_radius'],
                                               weights=dust['mass'])) \
                              if 'grain_radius' in dust else None
            ts['f_carb']    = float(np.average(dust['carbon_frac'],
                                               weights=dust['mass'])) \
                              if 'carbon_frac' in dust else None
        else:
            M_dust_code   = 0.0
            ts['mean_a_nm'] = None
            ts['f_carb']    = None

        ts['Mdust_over_Mstar'] = (M_dust_code / M_star_code) \
                                  if M_star_code > 0 else None

        # Gas (all halo gas within R200)
        gas = load_gas_for_snap(snap_base, ctr, r200)
        if gas and len(gas['mass']) > 0:
            M_gas_halo = float(gas['mass'].sum())
            if gas['metallicity'] is not None:
                Z = gas['metallicity']
                if Z.ndim == 2: Z = Z[:, 0]
                M_metal_halo = float((gas['mass'] * Z).sum())
            else:
                M_metal_halo = 0.0
            ts['DtoG'] = M_dust_code / M_gas_halo   if M_gas_halo   > 0 else 0.0
            ts['DtoZ'] = M_dust_code / M_metal_halo if M_metal_halo > 0 else 0.0
        else:
            ts['DtoG'] = 0.0
            ts['DtoZ'] = 0.0

        # f_surv from log
        if run in log_data and len(log_data[run]['z']) > 0:
            d = log_data[run]
            n_alive = d['n_part'][-1]
            n_dest  = d['n_thermal'][-1] + d['n_shock'][-1] + d['n_astrat'][-1]
            n_total = n_alive + n_dest
            ts['f_surv'] = float(n_alive / n_total) if n_total > 0 else np.nan
            if ts['mean_a_nm'] is None and not np.isnan(d['avg_size'][-1]):
                ts['mean_a_nm'] = float(d['avg_size'][-1])
        else:
            ts['f_surv'] = np.nan

        print(f'    z={ts["z_snap"]:.3f}  '
              f'M*/Mh={ts.get("Mstar_over_Mhalo")}  '
              f'<a>={ts.get("mean_a_nm")} nm  '
              f'D/G={ts.get("DtoG"):.3e}  '
              f'D/Z={ts.get("DtoZ"):.3f}  '
              f'f_surv={ts.get("f_surv")}')
        table[run] = ts

    return table


def _lx(v, sci_thresh=0.1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r'\ldots'
    if v == 0.0:
        return '0.000'
    if abs(v) < sci_thresh:
        exp  = int(np.floor(np.log10(abs(v))))
        mant = v / 10 ** exp
        return f'${mant:.2f}\\times10^{{{exp}}}$'
    return f'{v:.3f}'


def _lxf(v, fmt='.2f'):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r'\ldots'
    return f'{v:{fmt}}'


def write_latex_table(runs, table_stats, resolution, out_path):
    DESCRIPTIONS = {
        'S0':  'Creation only (baseline)',
        'S1':  r'$+$ Dust cooling',
        'S2':  r'$+$ Gas--dust drag',
        'S3':  r'$+$ Astration',
        'S4':  r'$+$ Thermal sputtering',
        'S5':  r'$+$ ISM grain growth',
        'S6':  r'$+$ Subgrid clumping',
        'S7':  r'$+$ SN shock destruction',
        'S8':  r'$+$ Coagulation',
        'S9':  r'$+$ Shattering',
        'S10': r'$+$ Radiation pressure (full)',
    }

    lines = []
    A = lines.append

    A(r'% Auto-generated by compare_grid_dust.py  (--latex-out)')
    A('')
    A(r'\begin{deluxetable*}{llrrrrrrr}')
    A(r'\tablecaption{%')
    A(f'  CosmicGrain simulation ladder at ${resolution}^3$ resolution:')
    A(r'  $z=0$ dust observables for each physics rung.')
    A(r'  All quantities measured within $R_{200}$ as determined from the')
    A(r'  \textsc{SubFind} group catalog.')
    A(r'  \label{tab:dust_sim_ladder_' + str(resolution) + r'}}')
    A(r'\setlength{\tabcolsep}{5pt}')
    A(r'\tablewidth{0pt}')
    A(r'\tablehead{')
    A(r'  \colhead{Sim} &')
    A(r'  \colhead{Description} &')
    A(r'  \colhead{$M_\star/M_{\rm halo}$} &')
    A(r'  \colhead{$\langle a\rangle$\,(nm)} &')
    A(r'  \colhead{$f_{\rm surv}$} &')
    A(r'  \colhead{$D/G$} &')
    A(r'  \colhead{$D/Z$} &')
    A(r'  \colhead{$M_{\rm dust}/M_\star$} &')
    A(r'  \colhead{$\bar{f}_{\rm C}$}')
    A(r'}')
    A(r'\startdata')

    for run in runs:
        ts   = table_stats.get(run, {})
        desc = DESCRIPTIONS.get(run, run)
        row  = (f'  {run} & '
                f'{desc} & '
                f'{_lx( ts.get("Mstar_over_Mhalo"),  sci_thresh=0.001)} & '
                f'{_lxf(ts.get("mean_a_nm"),          ".1f")} & '
                f'{_lxf(ts.get("f_surv"),             ".2f")} & '
                f'{_lx( ts.get("DtoG"))} & '
                f'{_lxf(ts.get("DtoZ"),               ".2f")} & '
                f'{_lx( ts.get("Mdust_over_Mstar"))} & '
                f'{_lxf(ts.get("f_carb"),             ".2f")} \\\\')
        A(row)

    A(r'\enddata')
    A(r'\tablecomments{%')
    A(f'  Each rung adds one dust-physics process to the previous configuration.')
    A(f'  All runs use ${resolution}^3$ resolution and identical initial conditions')
    A(r'  (50~Mpc comoving box, halo~569,')
    A(r'  $M_{{200}}\approx2\times10^{{12}}\,h^{{-1}}\,M_\odot$).')
    A(r'  Physics channel flags are listed in Table~\ref{tab:sim_grid}.')
    A(r'  $M_\star/M_{{\rm halo}}$: stellar-to-halo mass ratio.')
    A(r'  $\langle a\rangle$: ISM mass-weighted mean grain radius (nm).')
    A(r'  $f_{{\rm surv}}$: surviving dust fraction,')
    A(r'  $N_{{\rm alive}}/(N_{{\rm alive}}+N_{{\rm destroyed,cumul}})$.')
    A(r'  $D/G$, $D/Z$: dust-to-gas and dust-to-metals mass ratios,')
    A(r'  using total halo gas and metal mass within $R_{{200}}$ as denominators.')
    A(r'  $M_{{\rm dust}}/M_\star$: dust-to-stellar mass ratio.')
    A(r'  $\bar{{f}}_{{\rm C}}$: mass-weighted mean carbon fraction.}')
    A(r'\end{deluxetable*}')
    A('')

    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'\nLaTeX table written -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Global log_data reference (needed for legend in plot_radial_dgr)
# ─────────────────────────────────────────────────────────────────────────────
log_data_global = {}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dust grid analysis plots')
    parser.add_argument('--runs', nargs='+',
                        default=['S0','S1','S2','S3','S4','S5',
                                 'S6','S7','S8','S9','S10'],
                        help='Which runs to include (default: all)')
    parser.add_argument('--logs-only',  action='store_true',
                        help='Skip snapshot-based plots')
    parser.add_argument('--snaps-only', action='store_true',
                        help='Skip log-based plots')
    parser.add_argument('--table-only', action='store_true',
                        help='Only compute and write the LaTeX table')
    parser.add_argument('--latex-out',  default=None,
                        help='Write LaTeX deluxetable* to this path')
    parser.add_argument('--res', type=int, default=512,
                        help='Resolution label (default: 512)')
    args = parser.parse_args()

    global RESOLUTION, log_data_global
    RESOLUTION = args.res

    runs = args.runs
    print(f'\nAnalyzing runs: {runs}')
    print(f'Resolution:     {RESOLUTION}^3')
    print(f'Output dir:     {FIGDIR}/\n')
    print('NOTE: All spatial measurements use aperture = R200 from SubFind catalog.\n')

    if args.table_only and args.latex_out is None:
        args.latex_out = f'dust_sim_ladder_{args.res}.tex'

    # ── Part 1: Log-based plots ──────────────────────────────────────────────
    log_data = {}
    if not args.snaps_only:
        print('=== PART 1: Log-based plots ===')
        log_data = load_all_logs(runs)
        log_data_global = log_data
        gas_mass_curves = get_gas_mass_curves(runs)
        if log_data and not args.table_only:
            print('\nGenerating time-series plots...')
            plot_dust_mass(log_data)
            plot_dtg(log_data, gas_mass_curves)
            plot_hash_success(log_data)
            plot_grain_size_log(log_data)
            plot_dust_temp(log_data)
            plot_destruction_channels(log_data)
            plot_process_rates(log_data)
            plot_summary_panel(log_data, gas_mass_curves)
    else:
        log_data_global = {}

    # ── Part 2: Snapshot-based plots ────────────────────────────────────────
    if not args.logs_only and not args.table_only:
        print('\n=== PART 2: Snapshot-based plots ===\n')
        print('  Grain size distributions...')
        plot_grain_size_distribution(runs)
        for target_z in SNAP_REDSHIFTS:
            print(f'  z ~ {target_z}:')
            plot_carbon_fraction_pdf(runs, target_z)
            plot_radial_dgr(runs, target_z)
            plot_dust_map(runs, target_z)
        plot_phase_diagram_dust(runs, target_z=0.0)

    # ── Part 3: LaTeX table ──────────────────────────────────────────────────
    if args.latex_out:
        print('\n=== PART 3: LaTeX table ===')
        if not log_data:
            log_data = load_all_logs(runs)
        table_stats = compute_table_stats(runs, log_data)
        write_latex_table(runs, table_stats, args.res, args.latex_out)

    print(f'\nDone.  Figures -> {FIGDIR}/')


if __name__ == '__main__':
    main()
