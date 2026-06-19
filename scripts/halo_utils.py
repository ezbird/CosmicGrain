"""
halo_utils.py
=============
Shared utilities for all CosmicGrain analysis scripts.

=======================================================================
PRIMARY API  -- use these in all new and updated scripts
=======================================================================

Position-based Halo 569 tracking:

    from halo_utils import get_halo569_reference, get_halo569

    # Once at the top of a multi-snapshot loop:
    ref  = get_halo569_reference(output_dir)

    # Per snapshot:
    halo = get_halo569(groups_dir, snap_num, ref)
    halo['center']      # comoving kpc/h
    halo['r200_ckpch']  # R_Crit200 comoving kpc/h
    halo['r200_pkpc']   # R_Crit200 physical kpc
    halo['m200_code']   # M_Crit200 in 1e10 Msun/h

HALO 569 IDENTIFICATION
------------------------
Halo 569 is selected as the FOF group with the highest STELLAR mass
(argmax GroupMassType[:,4]) across ALL catalog chunks. This correctly
targets the zoom galaxy rather than potentially more massive but
star-poor neighbouring dark matter halos.

Falls back to argmax(M200) at early epochs before any stars form
(all GroupMassType[:,4] == 0).

Never uses Group[0] of chunk .0 only, which:
  a) reads only a fraction of the total groups, and
  b) ranks by M200 which can select a massive star-poor neighbour.

POSITION-BASED TRACKING
------------------------
Once the z=0 stellar-mass-selected centre is established, subsequent
snapshots find Halo 569 as the CLOSEST group to that reference position
within 3 Mpc/h. Comoving coordinates are approximately conserved along
the main progenitor; peculiar drift << search radius.

UNIT CONVENTIONS (enforced throughout)
---------------------------------------
  GroupPos / R200 / coordinates : comoving kpc/h  (ckpc/h)
  Masses                        : 1e10 Msun/h     (code units)
  HubbleParam h                 : from f["Parameters"].attrs["HubbleParam"]
  Physical kpc = ckpc/h / h * a

=======================================================================
BACKWARD-COMPATIBLE API  -- existing scripts unchanged
=======================================================================

All original signatures are preserved exactly.
"""

import re
import glob as _glob
import numpy as np
import h5py
from pathlib import Path

# Search radius for position-based tracking [ckpc/h].
# 3 Mpc/h spans any plausible progenitor drift while excluding
# unrelated halos (isolation radius >> 3 Mpc at all epochs).
HALO569_SEARCH_RADIUS_CKPCH = 3000.0

_SPATIAL_FIELDS = {
    0: ["Coordinates", "Masses", "Density", "Metallicity",
        "InternalEnergy", "StarFormationRate"],
    4: ["Coordinates", "Masses"],
    6: ["Coordinates", "Masses", "GrainRadius", "CarbonFraction",
        "GrainType", "DustTemperature", "Velocities"],
}

PARTICLE_TYPE_FIELDS = {
    0: ["Coordinates", "Masses", "Velocities", "Density", "Metallicity",
        "InternalEnergy", "StarFormationRate"],
    1: ["Coordinates", "Velocities", "ParticleIDs"],
    2: ["Coordinates", "Velocities", "ParticleIDs"],
    4: ["Coordinates", "Masses", "Velocities", "Metallicity",
        "StellarFormationTime", "ParticleIDs"],
    5: ["Coordinates", "Masses", "Velocities", "ParticleIDs"],
    6: ["Coordinates", "Masses", "GrainRadius", "GrainType",
        "DustTemperature", "CarbonFraction", "Velocities"],
}

UNIT_MASS     = 1e10
UNIT_LENGTH   = 1.0
UNIT_VELOCITY = 1.0
MSUN_PER_CODE = 1e10   # Gadget default: 1 code unit = 1e10 M_sun/h


# ==============================================================================
# Internal I/O helpers
# ==============================================================================

def find_last_snap_num(output_dir):
    """
    Return the highest snapshot number in output_dir that has both
    a snapdir_NNN directory and a groups_NNN catalog directory.
    Returns None if no complete snapshot is found.
    """
    output_dir = Path(output_dir)
    last = None
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num   = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        if groups_dir.exists() and list(groups_dir.glob("fof_subhalo_tab_*.hdf5")):
            last = snap_num
    return last

def _compute_r200_from_m200_simple(m200_code, h, a, Omega0=0.3158, OmegaL=0.6842):
    """Compute R200 in ckpc/h from M200 in code units."""
    G_cgs     = 6.674e-8
    KM_IN_CM  = 1e5
    MPC_IN_CM = 3.085678e24
    um_cgs    = 1.989e43
    ul_cm     = 3.085678e21
    H0_cgs    = 100.0 * h * KM_IN_CM / MPC_IN_CM
    Hz_cgs    = H0_cgs * np.sqrt(Omega0 * a**-3 + OmegaL)
    rho_crit  = 3.0 * Hz_cgs**2 / (8.0 * np.pi * G_cgs)
    r200_cm   = (3.0 * m200_code * um_cgs / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
    return r200_cm / ul_cm  # ckpc/h

def glob_snap_chunks(path):
    """Return sorted HDF5 chunk files for a snapshot directory or base path."""
    p = Path(path)
    if p.is_dir():
        chunks = sorted(p.glob("*.hdf5"))
    else:
        stem   = re.sub(r"(\.\d+)?\.hdf5$", "", str(p))
        chunks = sorted(Path(stem).parent.glob(Path(stem).name + "*.hdf5"))
    if not chunks:
        raise FileNotFoundError(f"No HDF5 chunks found at: {path}")
    return chunks


def glob_catalog_chunks(groups_dir, snap_num):
    """Return sorted FOF catalog chunks for snap_num inside groups_dir."""
    chunks = sorted(Path(groups_dir).glob(
        f"fof_subhalo_tab_{snap_num:03d}*.hdf5"))
    if not chunks:
        raise FileNotFoundError(
            f"No catalog for snap {snap_num:03d} in {groups_dir}")
    return chunks


def read_snap_header(snap_path):
    """Read cosmological header. Returns dict: h, a, z, box."""
    chunks = glob_snap_chunks(snap_path)
    with h5py.File(chunks[0], "r") as f:
        h   = float(f["Parameters"].attrs["HubbleParam"])
        a   = float(f["Header"].attrs["Time"])
        z   = float(f["Header"].attrs["Redshift"])
        box = float(f["Header"].attrs["BoxSize"])
    return dict(h=h, a=a, z=z, box=box)


def read_fof_catalog(groups_dir, snap_num):
    """
    Read ALL FOF group entries from a multi-chunk catalog.

    Returns dict {'pos', 'r200', 'm200', 'mstar'} in Gadget native units
    (ckpc/h, 1e10 Msun/h), or None if no groups exist yet.

    mstar is GroupMassType[:,4] — stellar mass within the FOF group.
    Used for stellar-mass-based halo identification.
    """
    try:
        chunks = glob_catalog_chunks(groups_dir, snap_num)
    except FileNotFoundError:
        return None

    pos_l   = []
    r200_l  = []
    m200_l  = []
    mstar_l = []

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_l.append(grp["GroupPos"][:])
            r200_l.append(grp["Group_R_Crit200"][:])
            m200_l.append(grp["Group_M_Crit200"][:])
            if "GroupMassType" in grp:
                mstar_l.append(grp["GroupMassType"][:, 4])
            else:
                mstar_l.append(np.zeros(len(grp["GroupPos"])))

    if not pos_l:
        return None

    return dict(
        pos   = np.concatenate(pos_l,   axis=0),
        r200  = np.concatenate(r200_l,  axis=0),
        m200  = np.concatenate(m200_l,  axis=0),
        mstar = np.concatenate(mstar_l, axis=0),
    )


def get_unit_mass(snap_path):
    """UnitMass_in_g from snapshot Parameters; falls back to 1e10 Msun."""
    chunks = glob_snap_chunks(snap_path)
    with h5py.File(chunks[0], "r") as f:
        params = f.get("Parameters", {})
        um = params.attrs.get("UnitMass_in_g", None) if params else None
    return float(um) if um is not None else 1.989e43


def find_snapshots(output_dir):
    """
    Return sorted list of (snap_num, snapdir_path, groups_dir_path) for all
    snapshots that have both a snapdir and a groups directory with catalogs.
    """
    output_dir = Path(output_dir)
    entries    = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num   = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        if not groups_dir.exists():
            continue
        if not list(groups_dir.glob("fof_subhalo_tab_*.hdf5")):
            continue
        entries.append((snap_num, snapdir, groups_dir))
    return entries


# ==============================================================================
# PRIMARY API -- position-based Halo 569 tracking
# ==============================================================================

def _select_primary_halo_idx(cat):
    """
    Select the primary halo index from a catalog dict.

    Uses stellar mass argmax (GroupMassType[:,4]) — correctly identifies
    the zoom target galaxy rather than the most massive dark matter halo.
    Falls back to M200 argmax when no stars have formed yet.

    Parameters
    ----------
    cat : dict from read_fof_catalog() with keys pos, r200, m200, mstar

    Returns
    -------
    idx : int
    selection_by : str  ('M*' or 'M200')
    """
    if cat["mstar"].max() > 0:
        return int(np.argmax(cat["mstar"])), "M*"
    else:
        return int(np.argmax(cat["m200"])), "M200"


# ---------------------------------------------------------------------------
# Per-resolution z=0 override table for Halo 569
# ---------------------------------------------------------------------------
# At some resolutions the stellar mass argmax selects the wrong FOF group
# (e.g. a massive but star-poor merger companion).  These hardcoded overrides
# force the correct group index at z=0, derived by cross-matching with the
# known 1024^3 comoving position and visual inspection of the top candidates.
#
# Format: output_dir_substring -> group_idx_at_snap49
#
# To add a new resolution:
#   1. Run the diagnostic in the comments at the bottom of this file
#   2. Identify the group closest in position to (34312, 34415, 35122) pkpc
#      with physically reasonable R200 (100-150 pkpc) and logM*~9.7
#   3. Add its index here
#
_HALO569_Z0_OVERRIDES = {
    # key: substring of output_dir path  →  value: FOF group index at snap 49
    "2048": 4,    # Idx=4: pos~(34436,34050,34899), R200=110.4 pkpc, logM*=9.63
                  # Idx=0 (stellar argmax) is a merger-inflated neighbour with
                  # R200=318.9 pkpc and logM*=11.10 -- wrong object
}

def find_shrinking_sphere_center(snap_path, initial_center, box,
                                  start_r=300.0, shrink=0.75,
                                  n_min=50, n_iter=40, verbose=False):
    """
    Density-weighted shrinking sphere center from gas particles.
    
    Starting from initial_center (ckpc/h), iteratively shrinks a sphere
    and recomputes the density-weighted centroid until fewer than n_min
    particles remain or n_iter is reached.
    
    Returns converged center in ckpc/h.
    """
    chunks = glob_snap_chunks(snap_path)
    all_pos, all_rho = [], []
    for chunk in chunks:
        with h5py.File(chunk, 'r') as f:
            if 'PartType0' not in f: continue
            pos = f['PartType0/Coordinates'][:]
            rho = f['PartType0/Density'][:]
            # Pre-filter to 2x start_r to avoid loading entire box
            dx = pos - initial_center
            dx -= box * np.round(dx / box)
            mask = np.sqrt((dx**2).sum(1)) < 2 * start_r
            if mask.any():
                all_pos.append(pos[mask])
                all_rho.append(rho[mask])

    if not all_pos:
        if verbose:
            print(f'  [shrinking_sphere] no gas found near {initial_center}')
        return initial_center.copy()

    pos = np.concatenate(all_pos)
    rho = np.concatenate(all_rho)
    ctr = initial_center.copy().astype(float)
    r   = float(start_r)

    for i in range(n_iter):
        dx   = pos - ctr
        dx  -= box * np.round(dx / box)
        d    = np.sqrt((dx**2).sum(1))
        mask = d < r
        if mask.sum() < n_min:
            break
        ctr = np.average(pos[mask], weights=rho[mask], axis=0)
        r  *= shrink
        if verbose:
            print(f'    iter {i:2d}: r={r:.1f} N={mask.sum()} ctr={ctr}')

    return ctr

def _get_z0_override(output_dir):
    """Return hardcoded group index override for output_dir, or None."""
    s = str(output_dir)
    for key, idx in _HALO569_Z0_OVERRIDES.items():
        if key in s:
            return idx
    return None


def get_halo569_reference(output_dir, snap_num_z0=None):
    """
    Establish Halo 569's z=0 (or last available) comoving position.
    If snap_num_z0 is None, automatically uses the highest snapshot
    number with both a snapdir and a catalog.
    """
    output_dir = Path(output_dir)

    if snap_num_z0 is None:
        snap_num_z0 = find_last_snap_num(output_dir)
        if snap_num_z0 is None:
            raise RuntimeError(
                f"No complete snapshot (snapdir+catalog) found in {output_dir}")

    groups_dir = output_dir / f"groups_{snap_num_z0:03d}"
    snapdir    = output_dir / f"snapdir_{snap_num_z0:03d}"

    hdr = read_snap_header(snapdir)
    cat = read_fof_catalog(groups_dir, snap_num_z0)
    if cat is None:
        raise RuntimeError(
            f"No FOF groups in {groups_dir} -- cannot establish z=0 reference")

    h = hdr["h"]
    a = hdr["a"]

    override_idx = _get_z0_override(output_dir)
    if override_idx is not None:
        idx    = override_idx
        sel_by = f"hardcoded override (idx={idx})"
    else:
        idx, sel_by = _select_primary_halo_idx(cat)

    fof_center = cat["pos"][idx].astype(float)
    r200       = float(cat["r200"][idx])
    m200       = float(cat["m200"][idx])
    mstar      = float(cat["mstar"][idx]) * MSUN_PER_CODE / h

    # Refine center with shrinking sphere on gas density peak
    refined_center = find_shrinking_sphere_center(
        snapdir, fof_center, hdr["box"], verbose=True)
    offset = np.linalg.norm(refined_center - fof_center)

    print(f"[halo_utils] Halo 569 z=0 reference  (selected by {sel_by})")
    print(f"  FOF group idx   : {idx}  (across {len(cat['pos'])} total groups)")
    print(f"  FOF centre      : [{fof_center[0]:.1f}, {fof_center[1]:.1f}, "
          f"{fof_center[2]:.1f}] ckpc/h")
    print(f"  Refined centre  : [{refined_center[0]:.1f}, {refined_center[1]:.1f}, "
          f"{refined_center[2]:.1f}] ckpc/h  (offset={offset:.1f} ckpc/h)")
    print(f"  R_Crit200       : {r200:.1f} ckpc/h  ({r200/h*a:.1f} pkpc)")
    print(f"  M_Crit200       : {m200:.3e} [1e10 Msun/h]  ({m200*1e10/h:.3e} Msun)")
    print(f"  M_star          : {mstar:.3e} Msun")
    print(f"  search radius   : {HALO569_SEARCH_RADIUS_CKPCH:.0f} ckpc/h"
          f"  ({HALO569_SEARCH_RADIUS_CKPCH/h:.0f} pkpc)")

    return dict(
        center_ckpch = refined_center,  # shrinking-sphere, not FOF
        box_ckpch    = hdr["box"],
        h            = h,
        a            = a,
        r200_ckpch   = r200,
        r200_pkpc    = r200 / h * a,
        m200_code    = m200,
        snap_num_z0  = snap_num_z0,
    )


def get_halo569(groups_dir, snap_num, ref,
                search_radius_ckpch=HALO569_SEARCH_RADIUS_CKPCH,
                verbose=True):
    cat = read_fof_catalog(groups_dir, snap_num)
    if cat is None:
        return None

    ref_pos = ref["center_ckpch"]
    box     = ref["box_ckpch"]
    h       = ref["h"]
    a       = _read_catalog_a(groups_dir, snap_num)

    dx   = cat["pos"] - ref_pos[None, :]
    dx  -= box * np.round(dx / box)
    dist = np.sqrt((dx**2).sum(axis=1))

    within   = dist <= search_radius_ckpch
    n_within = int(within.sum())
    fallback = False

    if n_within == 0:
        idx, sel_by = _select_primary_halo_idx(cat)
        fallback    = True
        if verbose:
            print(f"  [halo569] snap {snap_num:03d}: no group within "
                  f"{search_radius_ckpch:.0f} ckpc/h. "
                  f"Falling back to {sel_by} argmax (idx={idx}).")
    else:
        # Among groups within search radius, prefer those with substantial M200
        # Minimum threshold: 1.0 code unit = 1.5e10 Msun (excludes satellites)
        M200_MIN = 1.0
        valid = within & (cat["m200"] >= M200_MIN) & (cat["r200"] > 0)
        if valid.any():
            dist_valid = np.where(valid, dist, np.inf)
            idx = int(np.argmin(dist_valid))
            if verbose:
                print(f"  [halo569] snap {snap_num:03d}: selected group {idx} "
                      f"with M200={cat['m200'][idx]:.2f}, "
                      f"dist={dist[idx]:.0f} ckpc/h")
        else:
            # Fall back to stellar mass argmax
            idx, sel_by = _select_primary_halo_idx(cat)
            fallback = True
            if verbose:
                print(f"  [halo569] snap {snap_num:03d}: no group with "
                      f"M200>{M200_MIN} within search radius, "
                      f"falling back to {sel_by} argmax (idx={idx})")

    fof_center = cat["pos"][idx].astype(float)
    r200 = float(cat["r200"][idx])
    m200 = float(cat["m200"][idx])
    
    # R200 can be zero during merger fragmentation — compute from M200
    if r200 <= 0 and m200 > 0:
        r200 = _compute_r200_from_m200_simple(m200, h, a)
        if verbose:
            print(f"  [halo569] snap {snap_num:03d}: R200=0 in catalog, "
                      f"computed from M200={m200:.3f}: R200={r200:.1f} ckpc/h")

    # Refine center with shrinking sphere on gas density peak
    groups_path = Path(groups_dir)
    output_dir  = groups_path.parent
    snapdir     = output_dir / f"snapdir_{snap_num:03d}"

    if snapdir.exists():
        refined_center = find_shrinking_sphere_center(
            snapdir, fof_center, box, verbose=False)
        if verbose:
            offset = np.linalg.norm(refined_center - fof_center)
            print(f"  [halo569] snap {snap_num:03d}: FOF={fof_center}, "
                  f"refined={refined_center}, offset={offset:.1f} ckpc/h")
        center = refined_center
    else:
        if verbose:
            print(f"  [halo569] snap {snap_num:03d}: no snapdir found, "
                  f"using FOF center")
        center = fof_center

    return dict(
        center       = center,
        r200_ckpch   = r200,
        r200_pkpc    = r200 / h * a,
        m200_code    = m200,
        dist_ckpch   = float(dist[idx]),
        n_within     = n_within,
        used_fallback= fallback,
    )


def _read_catalog_a(groups_dir, snap_num):
    """Scale factor from catalog header; falls back to 1.0."""
    try:
        chunks = glob_catalog_chunks(groups_dir, snap_num)
        with h5py.File(chunks[0], "r") as f:
            return float(f["Header"].attrs["Time"])
    except Exception:
        return 1.0


def load_particles_within_r200(snap_path, halo, part_types=(4, 6)):
    """
    Spatially load particles within R_Crit200 of Halo 569.

    Parameters
    ----------
    snap_path  : str or Path (snapdir or snapshot base)
    halo       : dict from get_halo569()
    part_types : tuple of ints

    Returns
    -------
    dict {ptype: {field: array, ...}}
    """
    center = halo["center"]
    r200   = halo["r200_ckpch"]
    chunks = glob_snap_chunks(snap_path)
    box    = None

    buffers = {pt: {f: [] for f in _SPATIAL_FIELDS.get(
                    pt, ["Coordinates", "Masses"])}
               for pt in part_types}

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if box is None:
                box = float(f["Header"].attrs["BoxSize"])
            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                grp    = f[key]
                coords = grp["Coordinates"][:]
                dx     = coords - center[None, :]
                dx    -= box * np.round(dx / box)
                mask   = np.sqrt((dx**2).sum(axis=1)) <= r200
                if not mask.any():
                    continue
                for field in _SPATIAL_FIELDS.get(pt, ["Coordinates","Masses"]):
                    if field in grp:
                        arr = grp[field][:]
                        buffers[pt][field].append(
                            arr[mask] if arr.ndim == 1 else arr[mask])

    result = {}
    for pt in part_types:
        result[pt] = {}
        for field, bufs in buffers[pt].items():
            if bufs:
                result[pt][field] = (np.concatenate(bufs)
                                     if bufs[0].ndim == 1
                                     else np.vstack(bufs))
    return result


# ==============================================================================
# BACKWARD-COMPATIBLE API
# ==============================================================================

def load_target_halo(catalog_file, snapshot_base, particle_types='all',
                     output_file=None, verbose=True, ref=None):
    """
    Extract target halo particles using Subhalo offset/length tables.
    Original signature fully preserved.
    Optional `ref` from get_halo569_reference() improves centre/R200.
    """
    cat_path   = Path(catalog_file)
    stem_base  = re.sub(r"\.\d+$", "", cat_path.stem)
    cat_chunks = sorted(cat_path.parent.glob(f"{stem_base}*.hdf5"))
    if not cat_chunks:
        cat_chunks = [cat_path]

    subhalo_data = None
    for chunk in cat_chunks:
        with h5py.File(chunk, "r") as f:
            if "Subhalo" in f and "SubhaloMass" in f["Subhalo"]:
                subhalo_data = {
                    "mass"    : f["Subhalo"]["SubhaloMass"][:],
                    "pos"     : f["Subhalo"]["SubhaloPos"][:],
                    "vel"     : f["Subhalo"]["SubhaloVel"][:],
                    "halfmass": f["Subhalo"]["SubhaloHalfmassRad"][:],
                    "vmax"    : f["Subhalo"]["SubhaloVmax"][:],
                    "spin"    : f["Subhalo"]["SubhaloSpin"][:],
                    "offset"  : f["Subhalo"]["SubhaloOffsetType"][:],
                    "length"  : f["Subhalo"]["SubhaloLenType"][:],
                }
                break

    if subhalo_data is None:
        raise RuntimeError(
            f"No Subhalo table found in {catalog_file}. "
            "Use get_halo569() + load_particles_within_r200() instead.")

    target_id = int(np.argmax(subhalo_data["mass"]))

    if ref is not None:
        m = re.search(r"fof_subhalo_tab_(\d+)", cat_path.name)
        snap_num     = int(m.group(1)) if m else 49
        halo_result  = get_halo569(cat_path.parent, snap_num, ref,
                                   verbose=verbose)
        if halo_result is not None:
            position = halo_result["center"]
            r200     = halo_result["r200_ckpch"]
            r200_pk  = halo_result["r200_pkpc"]
            m200     = halo_result["m200_code"]
        else:
            position = subhalo_data["pos"][target_id]
            r200 = r200_pk = m200 = None
    else:
        position = subhalo_data["pos"][target_id]
        r200 = r200_pk = m200 = None
        for chunk in cat_chunks:
            with h5py.File(chunk, "r") as f:
                if "Group" in f and "Group_R_Crit200" in f["Group"]:
                    r200 = float(f["Group"]["Group_R_Crit200"][0])
                    m200 = float(f["Group"]["Group_M_Crit200"][0])
                    snap_chunks = sorted(
                        Path(snapshot_base).parent.glob(
                            Path(snapshot_base).name + "*.hdf5"))
                    if snap_chunks:
                        with h5py.File(snap_chunks[0], "r") as sf:
                            h  = float(sf["Parameters"].attrs["HubbleParam"])
                            a  = float(sf["Header"].attrs["Time"])
                        r200_pk = r200 / h * a
                    break

    halo_info = {
        "id"          : target_id,
        "mass"        : float(subhalo_data["mass"][target_id]),
        "position"    : position,
        "velocity"    : subhalo_data["vel"][target_id],
        "halfmass_rad": float(subhalo_data["halfmass"][target_id]),
        "vmax"        : float(subhalo_data["vmax"][target_id]),
        "spin"        : subhalo_data["spin"][target_id],
        "r200"        : r200,
        "r200_pkpc"   : r200_pk,
        "m200"        : m200,
    }

    if verbose:
        print(f"Target subhalo {target_id}")
        print(f"  Mass      : {halo_info['mass']:.2e} [1e10 Msun/h]")
        print(f"  Position  : {halo_info['position']}")
        if r200 is not None:
            print(f"  R_Crit200 : {r200:.2f} ckpc/h  ({r200_pk:.1f} pkpc)")

    result = {"halo_info": halo_info}

    if particle_types == "all":
        particle_types = range(7)

    ptype_names = {0:"gas", 1:"dm", 2:"dm2", 4:"stars", 5:"bh", 6:"dust"}
    dm_masses   = _get_dm_particle_masses(snapshot_base)

    for ptype in particle_types:
        offset = int(subhalo_data["offset"][target_id, ptype])
        length = int(subhalo_data["length"][target_id, ptype])
        if length == 0:
            continue
        pname  = ptype_names.get(ptype, f"parttype{ptype}")
        fields = PARTICLE_TYPE_FIELDS.get(ptype, ["Coordinates","Velocities"])
        fields = _check_available_fields(snapshot_base, ptype, fields)
        if verbose:
            print(f"Extracting {length} {pname} particles...")
        result[pname] = _extract_particles(
            snapshot_base, ptype, offset, length, fields)
        if ptype in [1, 2] and dm_masses[ptype] > 0:
            result[pname]["Masses"] = np.full(length, dm_masses[ptype])

    if output_file:
        _save_to_hdf5(result, output_file)

    return result


def extract_dust_spatially(snapshot_base, halo_center, radius_kpc=None,
                           verbose=True):
    """
    Spatially extract PartType6 dust particles near a halo centre.
    Original signature preserved. Units: ckpc/h throughout.
    """
    files = sorted(_glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = sorted(_glob.glob(f"{snapshot_base}/*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No snapshot chunks for: {snapshot_base}")

    buffers    = {}
    total_dust = 0

    for fname in files:
        with h5py.File(fname, "r") as f:
            if "PartType6" not in f:
                continue
            dust  = f["PartType6"]
            npart = len(dust["Coordinates"])
            total_dust += npart
            coords = dust["Coordinates"][:]
            r      = np.sqrt(np.sum((coords - halo_center)**2, axis=1))
            mask   = (r < radius_kpc) if radius_kpc is not None \
                     else np.ones(npart, dtype=bool)
            if not mask.any():
                continue
            for field in dust.keys():
                arr = dust[field][:]
                sel = arr[mask] if arr.ndim == 1 else arr[mask]
                buffers.setdefault(field, []).append(sel)

    if not buffers:
        return None

    result = {k: (np.concatenate(v) if v[0].ndim == 1 else np.vstack(v))
              for k, v in buffers.items()}

    if verbose:
        r_all = np.sqrt(np.sum(
            (result["Coordinates"] - halo_center)**2, axis=1))
        label = f"within {radius_kpc:.1f} ckpc/h" if radius_kpc else "total"
        print(f"  Extracted {len(r_all):,} dust particles ({label})")
        print(f"  Radial range : {r_all.min():.2f} - {r_all.max():.2f} ckpc/h")
        print(f"  Total mass   : {result['Masses'].sum():.2e} [1e10 Msun/h]")

    return result


def compute_radial_distance(coords, center):
    """Euclidean distance of each row in coords from center."""
    return np.sqrt(np.sum((coords - np.asarray(center)[None, :])**2, axis=1))


def compute_radial_profile(coords, masses, center, rbins):
    """Radial mass profile. Returns (r_centers, mass_profile)."""
    r             = compute_radial_distance(coords, center)
    mass_profile, _ = np.histogram(r, bins=rbins, weights=masses)
    r_centers     = 0.5 * (rbins[1:] + rbins[:-1])
    return r_centers, mass_profile


def convert_to_physical_units(data, mass_in_msun=True):
    """Convert Gadget code masses to solar masses in-place."""
    if "Masses" in data and mass_in_msun:
        data["Masses"] = data["Masses"] * UNIT_MASS
    return data


# ==============================================================================
# Private helpers (unchanged)
# ==============================================================================

def _get_dm_particle_masses(snapshot_base):
    files = sorted(_glob.glob(f"{snapshot_base}.*.hdf5"))
    with h5py.File(files[0], "r") as f:
        return f["Header"].attrs["MassTable"]


def _check_available_fields(snapshot_base, parttype, requested_fields):
    files = sorted(_glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        return requested_fields
    with h5py.File(files[0], "r") as f:
        key = f"PartType{parttype}"
        if key not in f:
            return []
        available = list(f[key].keys())
    return [field for field in requested_fields if field in available]


def _extract_particles(snapshot_base, parttype, global_offset, length, fields):
    files = sorted(_glob.glob(f"{snapshot_base}.*.hdf5"))
    cumulative = [0]
    for fname in files:
        with h5py.File(fname, "r") as f:
            cumulative.append(
                cumulative[-1] +
                int(f["Header"].attrs["NumPart_ThisFile"][parttype]))

    global_end = global_offset + length
    all_data   = {field: [] for field in fields}

    for i, fname in enumerate(files):
        fs, fe = cumulative[i], cumulative[i + 1]
        if fe <= global_offset or fs >= global_end:
            continue
        ls = max(0, global_offset - fs)
        le = min(fe - fs, global_end - fs)
        with h5py.File(fname, "r") as f:
            grp = f[f"PartType{parttype}"]
            for field in fields:
                all_data[field].append(grp[field][ls:le])

    result = {}
    for field in fields:
        bufs = all_data[field]
        if not bufs:
            continue
        result[field] = (np.concatenate(bufs)
                         if bufs[0].ndim == 1 else np.vstack(bufs))
    return result


def _save_to_hdf5(data, filename):
    ptype_map = {"gas":0, "dm":1, "dm2":2, "stars":4, "bh":5, "dust":6}
    with h5py.File(filename, "w") as f:
        for key, val in data["halo_info"].items():
            try:
                f.attrs[key] = val
            except Exception:
                pass
        for name, ptype_num in ptype_map.items():
            if name in data:
                grp = f.create_group(f"PartType{ptype_num}")
                for key, val in data[name].items():
                    grp.create_dataset(key, data=val)
