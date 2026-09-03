"""
halo_utils.py
=============
Canonical shared halo utilities for CosmicGrain analyses.

Core philosophy
---------------
1. Use the FOF/Subfind catalog only to identify the target group and get an
   initial center.
2. Compute R200c/M200c directly from particles around that center using a true
   spherical-overdensity calculation.
3. Keep units explicit everywhere.
4. Detect (not just document) FOF bridging.
5. Track any explicitly anchored zoom target backward from the final snapshot
   with a conserved dark-matter CORE ParticleID set. FOF position is not used
   to discover identity; exact DM IDs are located globally in the earlier
   snapshot and their densest spatial cluster defines the progenitor center.
   get_zoom_halo_series() deliberately walks through every intervening
   snapshot, even when the caller requests only a sparse subset. This prevents
   merger/bridging epochs from jumping between nearby FOF structures.

Unit conventions
----------------
Snapshot/catalog positions : comoving kpc/h  (ckpc/h)
Snapshot/catalog masses    : 1e10 Msun/h     (Gadget code mass units)
Physical distance          : pkpc = ckpc/h * a / h
Physical mass               : Msun = code_mass * 1e10 / h
Returned halo['center']    : ckpc/h
Returned halo['r200_ckpch']: ckpc/h
Returned halo['r200_pkpc'] : physical kpc
Returned halo['m200_code'] : 1e10 Msun/h
Returned halo['m200_msun'] : Msun
Returned halo['mass_ratio_group_to_so']: GroupMass / M200c_SO (dimensionless)
Returned halo['likely_bridged']        : bool, True if that ratio is large
                                          enough to indicate FOF bridging
                                          onto a neighboring structure --
                                          treat center/R200/M200 as unreliable
                                          when this is True

Primary API
-----------
    # One explicitly identified target at one snapshot:
    ref = get_zoom_halo(output_dir, snap, group_index=group_index)

    # Recommended for the 12-halo suite. group_index identifies the target
    # only at the final/reference snapshot; earlier identity follows HRDM IDs:
    halos = get_zoom_halo_series(
        output_dir, snap_nums, group_index=group_index
    )

Legacy Halo 569 APIs remain available for existing analysis scripts. New suite
code should use get_zoom_halo() and get_zoom_halo_series().

Optional particle loader:
    pdata = load_particles_within_radius(snapdir, halo['center'], halo['r200_ckpch'])

Notes
-----
This version deliberately does NOT use GroupMass as M200 and does NOT derive
R200 from GroupMass. FOF mass can include bridges/companions and is not a
spherical-overdensity mass. Catalog Group_R_Crit200/Group_M_Crit200 are kept
as diagnostics, and -- new in this version -- GroupMass is actively compared
against the particle-based M200c_SO to flag likely bridging (see
GROUP_MASS_RATIO_WARN below).

Catalog naming -- FOF-only vs SUBFIND runs
-------------------------------------------
Gadget-4 writes "fof_subhalo_tab_*.hdf5" catalogs when SUBFIND is enabled, and
plain "fof_tab_*.hdf5" catalogs when SUBFIND is disabled (FOF group-finding
only, e.g. to avoid SUBFIND hangs at high resolution). Every catalog read in
this module goes through glob_catalog_chunks(), find_last_snap_num(), or
find_snapshots(), all three of which check for "fof_subhalo_tab_*" first and
fall back to "fof_tab_*". Nothing here reads "Subhalo" fields -- only the
"Group" table (GroupPos, GroupMassType, Group_M_Crit200, Group_R_Crit200,
GroupMass) -- so FOF-only catalogs are fully sufficient for this module's
purposes, regardless of which naming convention a given run produced.
"""

from __future__ import annotations

import re
import glob as _glob
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MSUN_PER_CODE = 1.0e10          # Gadget mass unit: 1e10 Msun/h
MSUN_IN_G     = 1.98847e33
KPC_IN_CM     = 3.085677581e21
MPC_IN_CM     = 3.085677581e24
KM_IN_CM      = 1.0e5
G_CGS         = 6.67430e-8

HALO569_SEARCH_RADIUS_CKPCH = 3000.0
DEFAULT_SO_RMAX_CKPCH       = 2000.0

# GroupMass / M200c_SO above this ratio is flagged as likely FOF bridging
# onto a neighboring structure rather than genuine FOF-vs-SO mass excess.
# Empirically, on a clean (non-bridged) snapshot this ratio sits ~1.0-1.2x
# for Halo 569; bridged snapshots seen so far jump to 70-230x. 20.0 is a
# conservative cut sitting in the wide gap between those two populations --
# re-check it against your own run's clean-snapshot scatter if this module
# is reused for a different halo/box.
GROUP_MASS_RATIO_WARN = 20.0

# Fallback shrinking-sphere offset cap (ckpc/h) used only when no prior
# validated halo size is available to scale against (e.g. the very first
# snapshot processed, or get_halo569_reference() itself). Deliberately much
# tighter than the old fixed 300.0 ckpc/h default, which was larger than
# Halo 569's own R200c at every epoch in this run and so never actually
# caught anything.
FALLBACK_MAX_OFFSET_CKPCH = 100.0

# Main-progenitor DM-ID tracking controls.
#
# Identity is based on conserved PartType1 ParticleIDs, not FOF proximity.
# We extract a compact DM core from the last TRUSTED halo, locate those exact
# IDs globally in the earlier snapshot, and identify the densest spatial
# cluster of those conserved particles.
DM_TRACK_CORE_RADIUS_FACTOR = 0.50
DM_TRACK_CLUSTER_RADIUS_FACTOR = 0.35
DM_TRACK_CLUSTER_RADIUS_MIN_CKPCH = 12.0
DM_TRACK_CLUSTER_RADIUS_MAX_CKPCH = 60.0
DM_TRACK_MIN_CORE_IDS = 16
DM_TRACK_MIN_CLUSTER_IDS = 8
DM_TRACK_MIN_CLUSTER_FRACTION = 0.05
DM_TRACK_MAX_FOF_ASSOC_DIST_FACTOR = 2.0

# For loading spatial particle subsets.
_SPATIAL_FIELDS = {
    0: ["Coordinates", "Masses", "Density", "Metallicity", "InternalEnergy", "StarFormationRate"],
    1: ["Coordinates", "Velocities", "ParticleIDs"],
    2: ["Coordinates", "Velocities", "ParticleIDs"],
    4: ["Coordinates", "Masses", "Velocities", "Metallicity", "StellarFormationTime", "ParticleIDs"],
    5: ["Coordinates", "Masses", "Velocities", "ParticleIDs"],
    6: ["Coordinates", "Masses", "GrainRadius", "DustSource", "DustTemperature", "CarbonMassFraction", "Velocities", "ParticleIDs"],
}

# Per-resolution override for z=0 target selection.
# These are global FOF group indices across all catalog chunks.
_HALO569_Z0_OVERRIDES = {
    "2048": 4,
}

# Catalog filename prefixes to try, in priority order. "fof_subhalo_tab" is
# written when SUBFIND is enabled; "fof_tab" is written in FOF-only mode
# (SUBFIND disabled). Both expose the same "Group" table fields this module
# relies on -- see module docstring.
_CATALOG_PREFIXES = ("fof_subhalo_tab", "fof_tab")


def _catalog_glob_patterns(snap_num: int, suffix: str = "*.hdf5") -> list[str]:
    """Filename glob patterns to try for a given snapshot's catalog,
    in priority order (fof_subhalo_tab_* first, fof_tab_* fallback)."""
    return [f"{prefix}_{snap_num:03d}{suffix}" for prefix in _CATALOG_PREFIXES]


def _has_any_catalog(groups_dir: Path) -> bool:
    """True if groups_dir contains catalog chunks under either naming
    convention (no snap_num filtering -- used for directory-level checks)."""
    return any(list(groups_dir.glob(f"{prefix}_*.hdf5")) for prefix in _CATALOG_PREFIXES)


# -----------------------------------------------------------------------------
# File discovery and headers
# -----------------------------------------------------------------------------

def glob_snap_chunks(path: str | Path) -> list[Path]:
    """Return sorted HDF5 chunk files for a snapdir or snapshot base path."""
    p = Path(path)
    if p.is_dir():
        chunks = sorted(p.glob("*.hdf5"))
    else:
        stem = re.sub(r"(\.\d+)?\.hdf5$", "", str(p))
        chunks = sorted(Path(stem).parent.glob(Path(stem).name + "*.hdf5"))
    if not chunks:
        raise FileNotFoundError(f"No HDF5 chunks found at: {path}")
    return chunks


def glob_catalog_chunks(groups_dir: str | Path, snap_num: int) -> list[Path]:
    """Return sorted FOF/Subfind catalog chunks for a snapshot.

    Tries "fof_subhalo_tab_{snap_num}*.hdf5" first (SUBFIND enabled), then
    falls back to "fof_tab_{snap_num}*.hdf5" (FOF-only mode).
    """
    groups_dir = Path(groups_dir)
    for pattern in _catalog_glob_patterns(snap_num):
        chunks = sorted(groups_dir.glob(pattern))
        if chunks:
            return chunks
    raise FileNotFoundError(f"No catalog for snap {snap_num:03d} in {groups_dir}")


def find_last_snap_num(output_dir: str | Path) -> Optional[int]:
    """Highest snapshot number with both snapdir_NNN and groups_NNN."""
    output_dir = Path(output_dir)
    last = None
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        if groups_dir.exists() and _has_any_catalog(groups_dir):
            last = snap_num
    return last


def find_snapshots(output_dir: str | Path) -> list[tuple[int, Path, Path]]:
    """Sorted list of (snap_num, snapdir, groups_dir) with both snapshots/catalogs."""
    output_dir = Path(output_dir)
    out = []
    for snapdir in sorted(output_dir.glob("snapdir_*")):
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        if groups_dir.exists() and _has_any_catalog(groups_dir):
            out.append((snap_num, snapdir, groups_dir))
    return out


def read_snap_header(snap_path: str | Path) -> dict:
    """Read h, a, z, box, Omega0, OmegaLambda from a snapshot/snapdir."""
    with h5py.File(glob_snap_chunks(snap_path)[0], "r") as f:
        h = float(f["Parameters"].attrs.get("HubbleParam", f["Header"].attrs.get("HubbleParam", 1.0)))
        hdr = f["Header"].attrs
        return {
            "h": h,
            "a": float(hdr["Time"]),
            "z": float(hdr["Redshift"]),
            "box": float(hdr["BoxSize"]),
            "Omega0": float(hdr.get("Omega0", 0.3158)),
            "OmegaLambda": float(hdr.get("OmegaLambda", 0.6842)),
        }


def _read_catalog_a(groups_dir: str | Path, snap_num: int) -> float:
    try:
        with h5py.File(glob_catalog_chunks(groups_dir, snap_num)[0], "r") as f:
            return float(f["Header"].attrs["Time"])
    except Exception:
        return 1.0


# -----------------------------------------------------------------------------
# Catalog reading and target selection
# -----------------------------------------------------------------------------

def read_fof_catalog(groups_dir: str | Path, snap_num: int) -> Optional[dict]:
    """Read all FOF groups from all catalog chunks."""
    try:
        chunks = glob_catalog_chunks(groups_dir, snap_num)
    except FileNotFoundError:
        return None

    pos_l, r200_l, m200_l, mstar_l, gmass_l = [], [], [], [], []
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f or "GroupPos" not in f["Group"]:
                continue
            g = f["Group"]
            n = len(g["GroupPos"])
            if n == 0:
                continue
            pos_l.append(g["GroupPos"][:])
            r200_l.append(g["Group_R_Crit200"][:] if "Group_R_Crit200" in g else np.zeros(n))
            m200_l.append(g["Group_M_Crit200"][:] if "Group_M_Crit200" in g else np.zeros(n))
            if "GroupMassType" in g:
                mstar_l.append(g["GroupMassType"][:, 4])
            else:
                mstar_l.append(np.zeros(n))
            gmass_l.append(g["GroupMass"][:] if "GroupMass" in g else np.zeros(n))

    if not pos_l:
        return None

    return {
        "pos": np.concatenate(pos_l, axis=0),
        "r200_catalog": np.concatenate(r200_l),
        "m200_catalog": np.concatenate(m200_l),
        "mstar": np.concatenate(mstar_l),
        "group_mass": np.concatenate(gmass_l),
    }


def _select_primary_halo_idx(cat: dict) -> tuple[int, str]:
    """Select by stellar mass if present, otherwise by GroupMass."""
    if np.nanmax(cat["mstar"]) > 0:
        return int(np.nanargmax(cat["mstar"])), "Mstar"
    return int(np.nanargmax(cat["group_mass"])), "GroupMass"


def _get_z0_override(output_dir: str | Path) -> Optional[int]:
    s = str(output_dir)
    for key, idx in _HALO569_Z0_OVERRIDES.items():
        if key in s:
            return idx
    return None


def _periodic_delta(pos: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    dx = pos - center
    dx -= box * np.round(dx / box)
    return dx


# -----------------------------------------------------------------------------
# Spherical-overdensity calculation
# -----------------------------------------------------------------------------

def rho_crit_cgs(a: float, h: float, Omega0: float = 0.3158, OmegaLambda: float = 0.6842) -> float:
    """Critical density at scale factor a in g/cm^3."""
    H0 = 100.0 * h * KM_IN_CM / MPC_IN_CM
    Hz = H0 * np.sqrt(Omega0 * a**-3 + OmegaLambda)
    return 3.0 * Hz**2 / (8.0 * np.pi * G_CGS)


def compute_spherical_overdensity(
    snap_path: str | Path,
    center_ckpch: np.ndarray,
    rmax_ckpch: float = DEFAULT_SO_RMAX_CKPCH,
    Delta: float = 200.0,
    rho_ref: str = "crit",
    part_types: Sequence[int] = (0, 1, 2, 4, 5, 6),
    verbose: bool = False,
) -> Optional[dict]:
    """
    Compute R_Delta and M_Delta directly from particles around center.

    The SO crossing is found by sorting particles by radius and finding where
    mean enclosed density first drops below Delta*rho_ref. A linear interpolation
    in log(rho) versus log(r) is used at the crossing.
    """
    chunks = glob_snap_chunks(snap_path)
    hdr = read_snap_header(snap_path)
    h, a, box = hdr["h"], hdr["a"], hdr["box"]
    Om, OL = hdr["Omega0"], hdr["OmegaLambda"]

    radii_l, masses_l = [], []
    center = np.asarray(center_ckpch, dtype=float)

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            mass_table = f["Header"].attrs.get("MassTable", np.zeros(7))
            for ptype in part_types:
                key = f"PartType{ptype}"
                if key not in f or "Coordinates" not in f[key]:
                    continue
                coords = f[key]["Coordinates"][:]
                dx = _periodic_delta(coords, center, box)
                r = np.sqrt((dx * dx).sum(axis=1))
                mask = r <= rmax_ckpch
                if not mask.any():
                    continue

                if "Masses" in f[key]:
                    m = f[key]["Masses"][:][mask]
                elif len(mass_table) > ptype and mass_table[ptype] > 0:
                    m = np.full(mask.sum(), float(mass_table[ptype]))
                else:
                    continue

                radii_l.append(r[mask])
                masses_l.append(m.astype(float))

    if not radii_l:
        return None

    r_ckpch = np.concatenate(radii_l)
    m_code = np.concatenate(masses_l)
    ok = (r_ckpch > 0) & np.isfinite(r_ckpch) & np.isfinite(m_code) & (m_code > 0)
    if ok.sum() < 10:
        return None

    r_ckpch = r_ckpch[ok]
    m_code = m_code[ok]
    order = np.argsort(r_ckpch)
    r_ckpch = r_ckpch[order]
    m_code = m_code[order]
    m_enc_code = np.cumsum(m_code)

    r_pkpc = r_ckpch * a / h
    r_cm = r_pkpc * KPC_IN_CM
    m_enc_g = m_enc_code * MSUN_PER_CODE * MSUN_IN_G / h
    rho_enc = m_enc_g / ((4.0 / 3.0) * np.pi * r_cm**3)

    rhoc = rho_crit_cgs(a, h, Om, OL)
    if rho_ref == "crit":
        target = Delta * rhoc
    elif rho_ref == "mean":
        # rho_m(z) = Omega_m(z) rho_crit(z)
        Omz = (Om * a**-3) / (Om * a**-3 + OL)
        target = Delta * Omz * rhoc
    else:
        raise ValueError("rho_ref must be 'crit' or 'mean'")

    above = rho_enc >= target
    if not np.any(above):
        if verbose:
            print("  [SO] enclosed density is already below threshold at innermost particle")
        return None
    if above[-1]:
        if verbose:
            print(f"  [SO] WARNING: still above threshold at rmax={rmax_ckpch:.1f} ckpc/h; increase rmax")
        # Return outermost as lower limit rather than silently failing.
        i = len(r_ckpch) - 1
        return {
            "r_delta_ckpch": float(r_ckpch[i]),
            "r_delta_pkpc": float(r_pkpc[i]),
            "m_delta_code": float(m_enc_code[i]),
            "m_delta_msun": float(m_enc_code[i] * MSUN_PER_CODE / h),
            "rho_enc_cgs": float(rho_enc[i]),
            "target_rho_cgs": float(target),
            "is_lower_limit": True,
            "n_particles": int(len(r_ckpch)),
        }

    # Crossing between i1 (above target) and i2 (below target), i2 = i1 + 1.
    #
    # We deliberately take i1 as the LAST (largest-radius) index where
    # rho_enc is still above target, rather than i2 = first index where
    # rho_enc drops below target. With few particles very close to the
    # center, the enclosed density computed from only a handful of
    # particles is noisy and can dip below target right at the innermost
    # radius before climbing back above it as more particles are enclosed,
    # then falling below target again further out at the genuine halo
    # edge. Taking the first below-threshold index would latch onto that
    # spurious inner dip (i2=0, i1=-1) and return None even though a
    # perfectly good outer SO crossing exists. Taking the last
    # above-threshold index instead finds the true, physically meaningful
    # edge of the halo, regardless of what the density profile does deep
    # inside it.
    above_idx = np.where(above)[0]
    i1 = int(above_idx[-1])
    i2 = i1 + 1
    if i2 >= len(r_ckpch):
        # above[-1] was already handled above, so this shouldn't happen,
        # but guard anyway in case of edge effects at the boundary.
        return None

    # Interpolate log rho versus log r to the target density.
    x1, x2 = np.log(r_ckpch[i1]), np.log(r_ckpch[i2])
    y1, y2 = np.log(rho_enc[i1]), np.log(rho_enc[i2])
    yt = np.log(target)
    if y2 == y1:
        xt = x1
    else:
        xt = x1 + (yt - y1) * (x2 - x1) / (y2 - y1)
    r_delta_ckpch = float(np.exp(xt))

    # Use exact SO mass implied by radius and target density, converted to code units.
    r_delta_pkpc = r_delta_ckpch * a / h
    r_delta_cm = r_delta_pkpc * KPC_IN_CM
    m_delta_g = (4.0 / 3.0) * np.pi * r_delta_cm**3 * target
    m_delta_code = m_delta_g / (MSUN_PER_CODE * MSUN_IN_G / h)

    return {
        "r_delta_ckpch": r_delta_ckpch,
        "r_delta_pkpc": float(r_delta_pkpc),
        "m_delta_code": float(m_delta_code),
        "m_delta_msun": float(m_delta_code * MSUN_PER_CODE / h),
        "rho_enc_cgs": float(target),
        "target_rho_cgs": float(target),
        "is_lower_limit": False,
        "n_particles": int(len(r_ckpch)),
    }


# -----------------------------------------------------------------------------
# Center refinement
# -----------------------------------------------------------------------------

def find_shrinking_sphere_center(
    snap_path: str | Path,
    initial_center: np.ndarray,
    box: float,
    start_r: float = 300.0,
    shrink: float = 0.75,
    n_min: int = 50,
    n_iter: int = 40,
    r_min_ckpch: float = 5.0,
    max_offset_ckpch: Optional[float] = 300.0,
    verbose: bool = False,
) -> np.ndarray:
    """Density-weighted shrinking-sphere center using gas particles."""
    chunks = glob_snap_chunks(snap_path)
    initial_center = np.asarray(initial_center, dtype=float)
    all_pos, all_w = [], []

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "PartType0" not in f or "Coordinates" not in f["PartType0"]:
                continue
            pos = f["PartType0/Coordinates"][:]
            dx = _periodic_delta(pos, initial_center, box)
            r = np.sqrt((dx * dx).sum(axis=1))
            mask = r < 2.0 * start_r
            if not mask.any():
                continue
            all_pos.append(pos[mask])
            if "Density" in f["PartType0"]:
                all_w.append(f["PartType0/Density"][:][mask])
            elif "Masses" in f["PartType0"]:
                all_w.append(f["PartType0/Masses"][:][mask])
            else:
                all_w.append(np.ones(mask.sum()))

    if not all_pos:
        return initial_center.copy()

    pos = np.concatenate(all_pos)
    w = np.concatenate(all_w).astype(float)
    ctr = initial_center.copy()
    rad = float(start_r)

    for i in range(n_iter):
        if rad < r_min_ckpch:
            break
        dx = _periodic_delta(pos, ctr, box)
        d = np.sqrt((dx * dx).sum(axis=1))
        mask = d < rad
        if mask.sum() < n_min:
            break
        ctr = np.average(pos[mask], weights=w[mask], axis=0)
        rad *= shrink
        if verbose:
            print(f"    [shrink] iter={i:02d} r={rad:.2f} N={mask.sum()} ctr={ctr}")

    if max_offset_ckpch is not None:
        dx = _periodic_delta(ctr[None, :], initial_center, box)[0]
        offset = float(np.sqrt((dx * dx).sum()))
        if offset > max_offset_ckpch:
            print(f"  [shrink] WARNING: rejected refined center; offset={offset:.1f} ckpc/h > {max_offset_ckpch:.1f}")
            return initial_center.copy()

    return ctr



# -----------------------------------------------------------------------------
# Dark-matter ParticleID main-progenitor tracking
# -----------------------------------------------------------------------------

def _load_dm_snapshot(snap_path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load all PartType1 coordinates and IDs for one snapshot.

    Returns
    -------
    coords : (N,3) float array, ckpc/h
    ids    : (N,) uint64 array
    box    : float, ckpc/h

    The DM particle set is the most stable identity tracer in a merger and is
    therefore used for main-progenitor matching.  This routine is intentionally
    separate from load_particles_within_radius() so a snapshot is read only once
    per tracking step.
    """
    chunks = glob_snap_chunks(snap_path)
    pos_l, id_l = [], []
    box = None

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if box is None:
                box = float(f["Header"].attrs["BoxSize"])
            if "PartType1" not in f:
                continue
            g = f["PartType1"]
            if "Coordinates" not in g or "ParticleIDs" not in g:
                continue
            pos_l.append(np.asarray(g["Coordinates"][:], dtype=float))
            id_l.append(np.asarray(g["ParticleIDs"][:], dtype=np.uint64))

    if not pos_l:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=np.uint64), float(box or 0.0)

    return np.vstack(pos_l), np.concatenate(id_l), float(box)


def _query_periodic_indices(
    coords: np.ndarray,
    center: np.ndarray,
    radius_ckpch: float,
    box: float,
    tree=None,
) -> np.ndarray:
    """Indices of particles inside a periodic sphere.

    Uses scipy's periodic cKDTree when available; falls back to a vectorized
    periodic-distance calculation otherwise.
    """
    if len(coords) == 0 or radius_ckpch <= 0:
        return np.empty(0, dtype=np.int64)

    center = np.mod(np.asarray(center, dtype=float), box)

    if tree is not None:
        return np.asarray(tree.query_ball_point(center, float(radius_ckpch)), dtype=np.int64)

    d = _periodic_delta(coords, center, box)
    r2 = np.einsum("ij,ij->i", d, d)
    return np.where(r2 <= float(radius_ckpch) ** 2)[0]


def load_halo_dm_ids(
    output_dir: str | Path,
    snap_num: int,
    halo: dict,
    radius_factor: float = DM_TRACK_CORE_RADIUS_FACTOR,
) -> np.ndarray:
    """Return PartType1 ParticleIDs within radius_factor * R200c of halo."""
    output_dir = Path(output_dir)
    snapdir = output_dir / f"snapdir_{snap_num:03d}"
    coords, ids, box = _load_dm_snapshot(snapdir)
    if len(ids) == 0:
        return np.empty(0, dtype=np.uint64)

    radius = float(radius_factor) * float(halo["r200_ckpch"])
    if radius <= 0:
        return np.empty(0, dtype=np.uint64)

    tree = cKDTree(np.mod(coords, box), boxsize=box) if cKDTree is not None else None
    ii = _query_periodic_indices(coords, halo["center"], radius, box, tree=tree)
    return np.unique(ids[ii])


def dm_overlap_metrics(ids_earlier: np.ndarray, ids_later: np.ndarray) -> dict:
    """Dark-matter continuity metrics for an earlier candidate and later halo.

    later_retained_fraction
        Fraction of the later halo's DM IDs already present in the earlier
        candidate.  This is the most useful backward-tracking quantity.

    earlier_to_later_fraction
        Fraction of the earlier candidate's DM IDs that end up in the later
        halo.  A high value identifies a compact genuine progenitor even when
        the later object has acquired substantial new material.
    """
    a = np.unique(np.asarray(ids_earlier, dtype=np.uint64))
    b = np.unique(np.asarray(ids_later, dtype=np.uint64))

    if len(a) == 0 or len(b) == 0:
        return {
            "n_earlier": int(len(a)),
            "n_later": int(len(b)),
            "n_shared": 0,
            "later_retained_fraction": np.nan,
            "earlier_to_later_fraction": np.nan,
        }

    shared = np.intersect1d(a, b, assume_unique=True)
    ns = int(len(shared))
    return {
        "n_earlier": int(len(a)),
        "n_later": int(len(b)),
        "n_shared": ns,
        "later_retained_fraction": ns / len(b),
        "earlier_to_later_fraction": ns / len(a),
    }


def _match_ids_to_positions(
    coords: np.ndarray,
    ids_all: np.ndarray,
    wanted_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and IDs for wanted ParticleIDs present in a snapshot.

    ParticleIDs are conserved for collisionless DM, so this is the cleanest
    possible cross-snapshot identity operation.  The implementation uses sorted
    integer lookup rather than repeated np.isin() scans over the full snapshot.
    """
    if len(ids_all) == 0 or len(wanted_ids) == 0:
        return np.empty((0, 3), float), np.empty(0, np.uint64)

    ids_all = np.asarray(ids_all, dtype=np.uint64)
    wanted = np.unique(np.asarray(wanted_ids, dtype=np.uint64))

    order = np.argsort(ids_all)
    sorted_ids = ids_all[order]
    loc = np.searchsorted(sorted_ids, wanted)
    good = loc < len(sorted_ids)
    good[good] &= sorted_ids[loc[good]] == wanted[good]

    if not np.any(good):
        return np.empty((0, 3), float), np.empty(0, np.uint64)

    snap_idx = order[loc[good]]
    return np.asarray(coords[snap_idx], float), wanted[good]


def _periodic_cluster_center(
    positions: np.ndarray,
    box: float,
    search_center: np.ndarray,
    cluster_radius_ckpch: float,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Find the densest local cluster in a set of conserved DM-core particles.

    The later halo's core particles can be split among multiple earlier
    structures during a merger.  Taking one global mean/median can therefore
    land between halos.  Instead, find the particle with the largest number of
    conserved-core neighbors within a physically motivated radius, then take
    the periodic median of that local cluster.

    Returns
    -------
    center : ndarray or None
        Robust periodic cluster center.
    member_indices : ndarray
        Indices into ``positions`` belonging to the selected cluster.
    """
    if len(positions) == 0:
        return None, np.empty(0, dtype=np.int64)

    pos = np.mod(np.asarray(positions, float), box)
    r = float(cluster_radius_ckpch)

    if len(pos) == 1:
        return pos[0].copy(), np.array([0], dtype=np.int64)

    if cKDTree is not None:
        tree = cKDTree(pos, boxsize=box)

        # For modest core sets, query every point.  This directly finds the
        # densest conserved-ID clump without assuming a FOF center.
        neighborhoods = tree.query_ball_point(pos, r)
        counts = np.fromiter((len(x) for x in neighborhoods), dtype=int)
        seed = int(np.argmax(counts))
        members = np.asarray(neighborhoods[seed], dtype=np.int64)
    else:
        # Vectorized fallback.  Use the particle nearest the predicted center
        # as a first seed, then gather its local neighbors.
        dd0 = _periodic_delta(pos, np.asarray(search_center, float), box)
        seed = int(np.argmin(np.einsum("ij,ij->i", dd0, dd0)))
        dd = _periodic_delta(pos, pos[seed], box)
        members = np.where(np.einsum("ij,ij->i", dd, dd) <= r*r)[0]

    if len(members) == 0:
        return None, members

    seed_pos = pos[members[0]]
    dd = _periodic_delta(pos[members], seed_pos, box)
    shift = np.median(dd, axis=0)
    center = np.mod(seed_pos + shift, box)

    # One recentering iteration makes the member set less dependent on the
    # arbitrary first member chosen above.
    if cKDTree is not None:
        members2 = np.asarray(
            cKDTree(pos, boxsize=box).query_ball_point(center, r),
            dtype=np.int64,
        )
    else:
        dd2 = _periodic_delta(pos, center, box)
        members2 = np.where(np.einsum("ij,ij->i", dd2, dd2) <= r*r)[0]

    if len(members2) > 0:
        dd = _periodic_delta(pos[members2], center, box)
        center = np.mod(center + np.median(dd, axis=0), box)
        members = members2

    return center, members


def _nearest_fof_to_center(
    cat: dict,
    center: np.ndarray,
    box: float,
) -> tuple[int, float]:
    """Return nearest FOF group index and periodic distance to a trusted center."""
    d = _periodic_delta(cat["pos"], np.asarray(center, float), box)
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def _select_dm_main_progenitor(
    output_dir: str | Path,
    earlier_snap: int,
    later_snap: int,
    later_halo: dict,
    refine_center: bool = False,
    verbose: bool = True,
) -> Optional[dict]:
    """Recover the earlier main progenitor from conserved DM core ParticleIDs.

    This routine intentionally does *not* use FOF proximity to establish halo
    identity.  During a bridge/merger, FOF GroupPos and GroupMass can be badly
    misleading, while PartType1 IDs remain exact.

    Algorithm
    ---------
    1. Extract DM IDs inside 0.5 R200 of the last TRUSTED halo.
    2. Locate those exact IDs globally in the earlier snapshot.
    3. Find the densest spatial cluster of the conserved IDs.
    4. Use that cluster center for a fresh particle-based SO calculation.
    5. Associate the nearest FOF group only for diagnostics/catalog fallback.
    6. Reject weak matches.  A weak/zero match is NEVER allowed to become the
       next chain anchor.

    This naturally handles catalog gaps: get_halo569_series() can try 044->043,
    reject it, then try 044->042 using the same trusted 044 core.
    """
    output_dir = Path(output_dir)
    earlier_snapdir = output_dir / f"snapdir_{earlier_snap:03d}"
    later_snapdir = output_dir / f"snapdir_{later_snap:03d}"
    earlier_groups = output_dir / f"groups_{earlier_snap:03d}"

    cat = read_fof_catalog(earlier_groups, earlier_snap)
    if cat is None:
        if verbose:
            print(f"  [halo569-ID] {earlier_snap:03d}: no FOF catalog")
        return None

    hdr_earlier = read_snap_header(earlier_snapdir)
    box = float(hdr_earlier["box"])

    # --------------------------------------------------------------
    # 1. Trusted later-halo DM core.
    # --------------------------------------------------------------
    lcoords, lids_all, lbox = _load_dm_snapshot(later_snapdir)
    if len(lids_all) == 0:
        return None

    core_radius = max(
        DM_TRACK_CORE_RADIUS_FACTOR * float(later_halo["r200_ckpch"]),
        DM_TRACK_CLUSTER_RADIUS_MIN_CKPCH,
    )

    ltree = (
        cKDTree(np.mod(lcoords, lbox), boxsize=lbox)
        if cKDTree is not None else None
    )
    jj = _query_periodic_indices(
        lcoords,
        later_halo["center"],
        core_radius,
        lbox,
        tree=ltree,
    )
    trusted_core_ids = np.unique(lids_all[jj])

    if len(trusted_core_ids) < DM_TRACK_MIN_CORE_IDS:
        if verbose:
            print(
                f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
                f"trusted halo has only {len(trusted_core_ids)} DM core IDs; "
                "not enough for robust tracking."
            )
        return None

    # --------------------------------------------------------------
    # 2. Locate the exact same IDs in the earlier snapshot.
    # --------------------------------------------------------------
    ecoords, eids_all, ebox = _load_dm_snapshot(earlier_snapdir)
    if len(eids_all) == 0:
        return None

    matched_pos, matched_ids = _match_ids_to_positions(
        ecoords, eids_all, trusted_core_ids
    )
    global_retained = len(matched_ids) / len(trusted_core_ids)

    if len(matched_ids) < DM_TRACK_MIN_CLUSTER_IDS:
        if verbose:
            print(
                f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
                f"only {len(matched_ids)}/{len(trusted_core_ids)} trusted "
                "core IDs exist in earlier snapshot; rejecting."
            )
        return None

    # --------------------------------------------------------------
    # 3. Find the densest conserved-ID cluster.
    # --------------------------------------------------------------
    cluster_radius = np.clip(
        DM_TRACK_CLUSTER_RADIUS_FACTOR * float(later_halo["r200_ckpch"]),
        DM_TRACK_CLUSTER_RADIUS_MIN_CKPCH,
        DM_TRACK_CLUSTER_RADIUS_MAX_CKPCH,
    )

    core_center, members = _periodic_cluster_center(
        matched_pos,
        ebox,
        np.asarray(later_halo["center"], float),
        cluster_radius,
    )
    if core_center is None or len(members) == 0:
        return None

    n_cluster = int(len(members))
    cluster_fraction = n_cluster / len(trusted_core_ids)

    if (
        n_cluster < DM_TRACK_MIN_CLUSTER_IDS
        or cluster_fraction < DM_TRACK_MIN_CLUSTER_FRACTION
    ):
        if verbose:
            print(
                f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
                f"weak conserved-core cluster: N={n_cluster}, "
                f"fraction={cluster_fraction:.3f}; rejecting and keeping "
                f"snapshot {later_snap:03d} as trusted anchor."
            )
        return None

    # --------------------------------------------------------------
    # 4/5. FOF association is diagnostic only; SO center is the DM core.
    # --------------------------------------------------------------
    idx, fof_dist = _nearest_fof_to_center(cat, core_center, ebox)

    # If the nearest catalog group is absurdly far from the conserved-DM core,
    # we still attempt a pure particle SO center, but mark the association weak.
    assoc_scale = max(
        float(cat["r200_catalog"][idx]) if cat["r200_catalog"][idx] > 0 else 0.0,
        cluster_radius,
    )
    fof_assoc_weak = fof_dist > DM_TRACK_MAX_FOF_ASSOC_DIST_FACTOR * assoc_scale

    offset_cap = max(
        20.0,
        3.0 * float(later_halo.get("r200_ckpch", 0.0)),
    )

    halo = None
    build_errors = []

    # First and preferred center: conserved DM core.
    # Fallback: nearest FOF position only if the direct SO calculation fails.
    for center_label, trial_center in (
        ("conserved-DM core", core_center),
        ("nearest FOF fallback", np.asarray(cat["pos"][idx], float)),
    ):
        try:
            halo = _build_halo_result(
                earlier_snapdir,
                trial_center,
                hdr_earlier,
                idx,
                cat,
                refine_center=refine_center,
                verbose=False,
                max_offset_ckpch=offset_cap,
            )
            halo["tracking_center_method"] = center_label
            break
        except RuntimeError as exc:
            build_errors.append(f"{center_label}: {exc}")

    if halo is None:
        if verbose:
            print(
                f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
                "conserved DM core found, but SO construction failed."
            )
            for msg in build_errors:
                print(f"               {msg}")
            print(
                f"               snapshot {later_snap:03d} remains the trusted anchor."
            )
        return None

    # --------------------------------------------------------------
    # 6. Structural sanity check.  Do not impose monotonic mass growth:
    # mergers/pseudo-evolution make that too restrictive.  Only reject truly
    # catastrophic solutions that are inconsistent with the conserved core.
    # --------------------------------------------------------------
    later_mass = float(later_halo.get("m200_msun", np.nan))
    mass_ratio = (
        float(halo["m200_msun"]) / later_mass
        if np.isfinite(later_mass) and later_mass > 0 else np.nan
    )

    catastrophic_mass = (
        np.isfinite(mass_ratio)
        and (mass_ratio < 0.01 or mass_ratio > 3.0)
        and cluster_fraction < 0.25
    )

    if catastrophic_mass:
        if verbose:
            print(
                f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
                f"rejecting structurally implausible solution "
                f"(Mearlier/Mlater={mass_ratio:.3e}, "
                f"core cluster fraction={cluster_fraction:.3f})."
            )
        return None

    halo.update({
        "center_ckpch": halo["center"],
        "center_dm_shared": np.asarray(core_center, float),
        "center_fof_catalog": np.asarray(cat["pos"][idx], float),
        "selection": "conserved DM-core main progenitor",
        "tracking_method": "conserved_dm_core_ids",
        "tracking_from_snap": int(later_snap),
        "tracking_gap": int(later_snap - earlier_snap),
        "dm_n_trusted_core": int(len(trusted_core_ids)),
        "dm_n_found_global": int(len(matched_ids)),
        "dm_global_retained_fraction": float(global_retained),
        "dm_n_shared": int(n_cluster),
        "dm_n_shared_for_center": int(n_cluster),
        "dm_later_retained_fraction": float(cluster_fraction),
        "dm_earlier_to_later_fraction": np.nan,
        "dm_cluster_fraction": float(cluster_fraction),
        "dm_cluster_radius_ckpch": float(cluster_radius),
        "dm_tracking_weak": False,
        "fof_association_distance_ckpch": float(fof_dist),
        "fof_association_weak": bool(fof_assoc_weak),
        "refine_center": bool(refine_center),
    })

    if verbose:
        print(
            f"  [halo569-ID] {later_snap:03d}->{earlier_snap:03d}: "
            f"group={idx:4d}, core={n_cluster:5d}/{len(trusted_core_ids):5d} "
            f"({cluster_fraction:.3f}), global-ID={global_retained:.3f}, "
            f"FOFdist={fof_dist:.1f} ckpc/h"
        )
        print(
            f"               R200={halo['r200_pkpc']:.1f} pkpc, "
            f"M200={halo['m200_msun']:.3e} Msun, "
            f"Mearlier/Mlater={mass_ratio:.3f}"
            if np.isfinite(mass_ratio)
            else
            f"               R200={halo['r200_pkpc']:.1f} pkpc, "
            f"M200={halo['m200_msun']:.3e} Msun"
        )
        print(
            f"               center={halo.get('tracking_center_method','unknown')}, "
            f"cluster-radius={cluster_radius:.1f} ckpc/h"
        )
        if fof_assoc_weak:
            print(
                "               NOTE: nearest FOF group is far from the conserved "
                "DM core; trust the particle-based SO center, not GroupPos."
            )
        if halo.get("likely_bridged", False):
            print(
                "               NOTE: FOF bridging flagged; identity/center came "
                "from conserved DM core IDs."
            )

    return halo


# -----------------------------------------------------------------------------
# Primary API
# -----------------------------------------------------------------------------

def _build_halo_result(
    snapdir: Path,
    fof_center: np.ndarray,
    hdr: dict,
    group_idx: int,
    cat: dict,
    refine_center: bool = True,
    verbose: bool = True,
    max_offset_ckpch: float = 300.0,
) -> dict:
    """Refine center if requested, compute SO R200c/M200c, return standard halo dict.

    max_offset_ckpch is forwarded to find_shrinking_sphere_center's own
    rejection guard. Callers that know the halo's approximate current size
    (e.g. get_halo569 chaining from a previous validated snapshot via
    get_halo569_series) should pass a cap scaled to that size rather than
    relying on the old fixed default -- see module docstring.
    """
    center0 = np.asarray(fof_center, dtype=float)
    center = center0.copy()

    if refine_center and snapdir.exists():
        center = find_shrinking_sphere_center(
            snapdir,
            center0,
            hdr["box"],
            start_r=300.0,
            max_offset_ckpch=max_offset_ckpch,
            verbose=False,
        )

    so = compute_spherical_overdensity(
        snapdir,
        center,
        rmax_ckpch=DEFAULT_SO_RMAX_CKPCH,
        Delta=200.0,
        rho_ref="crit",
        verbose=verbose,
    )
    if so is None:
        rcat = float(cat["r200_catalog"][group_idx])
        mcat = float(cat["m200_catalog"][group_idx])

        if rcat <= 0 or mcat <= 0:
            raise RuntimeError(
                "Could not compute spherical-overdensity radius from particles "
                "and catalog Group_R_Crit200/Group_M_Crit200 are invalid"
            )

        if verbose:
            print(
                "  [halo_utils] WARNING: particle SO failed; "
                "using catalog Group_R_Crit200/Group_M_Crit200 fallback"
            )

        so = {
            "r_delta_ckpch": rcat,
            "r_delta_pkpc": rcat * hdr["a"] / hdr["h"],
            "m_delta_code": mcat,
            "m_delta_msun": mcat * MSUN_PER_CODE / hdr["h"],
            "is_lower_limit": False,
            "n_particles": 0,
            "used_catalog_fallback": True,
        }
    else:
        so["used_catalog_fallback"] = False

    # --- Bridging check -----------------------------------------------------
    # GroupMass was previously kept as a diagnostic-only field with nothing
    # comparing it to anything. A real galaxy's FOF group mass shouldn't
    # wildly exceed its own SO M200 -- some excess is normal (FOF isn't SO,
    # and picks up some unbound/CGM/companion mass), but a large ratio is
    # the signature of FOF bridging onto a neighboring structure. See
    # GROUP_MASS_RATIO_WARN's docstring for the empirical basis of the cut.
    group_mass_msun = float(cat["group_mass"][group_idx] * MSUN_PER_CODE / hdr["h"])
    m200_msun = so["m_delta_msun"]
    mass_ratio = group_mass_msun / m200_msun if m200_msun > 0 else np.inf
    likely_bridged = mass_ratio > GROUP_MASS_RATIO_WARN

    # Independent sanity check: compare the particle-based SO solution against
    # the catalog's Crit200 quantities. These are not used as the primary halo
    # definition, but strong disagreement is useful for detecting a bad center
    # or an SO crossing that has swallowed neighboring structure.
    catalog_r200_ckpch = float(cat["r200_catalog"][group_idx])
    catalog_m200_code = float(cat["m200_catalog"][group_idx])
    catalog_r200_pkpc = catalog_r200_ckpch * hdr["a"] / hdr["h"]
    catalog_m200_msun = catalog_m200_code * MSUN_PER_CODE / hdr["h"]

    so_to_catalog_r_ratio = (
        so["r_delta_pkpc"] / catalog_r200_pkpc
        if catalog_r200_pkpc > 0 else np.nan
    )
    so_to_catalog_m_ratio = (
        m200_msun / catalog_m200_msun
        if catalog_m200_msun > 0 else np.nan
    )

    so_catalog_mismatch = bool(
        (np.isfinite(so_to_catalog_r_ratio) and
         (so_to_catalog_r_ratio > 2.0 or so_to_catalog_r_ratio < 0.5))
        or
        (np.isfinite(so_to_catalog_m_ratio) and
         (so_to_catalog_m_ratio > 3.0 or so_to_catalog_m_ratio < (1.0 / 3.0)))
    )

    if likely_bridged and verbose:
        print(
            f"  [halo_utils] WARNING: GroupMass/M200c_SO = {mass_ratio:.1f}x "
            f"(> {GROUP_MASS_RATIO_WARN:.0f}x) -- likely FOF bridging onto a "
            f"neighboring structure. GroupMass is unreliable as a halo-mass "
            f"proxy; inspect the center and independent SO diagnostics."
        )

    if so_catalog_mismatch and verbose:
        print(
            f"  [halo_utils] WARNING: particle-SO/catalog-SO mismatch: "
            f"R ratio={so_to_catalog_r_ratio:.2f}, M ratio={so_to_catalog_m_ratio:.2f}. "
            f"Treat this center/SO solution as suspect."
        )

    return {
        "center": center,
        "center_fof": center0,
        "r200_ckpch": so["r_delta_ckpch"],
        "r200_pkpc": so["r_delta_pkpc"],
        "m200_code": so["m_delta_code"],
        "m200_msun": m200_msun,
        "group_idx": int(group_idx),
        "group_mass_code": float(cat["group_mass"][group_idx]),
        "group_mass_msun": group_mass_msun,
        "mstar_code": float(cat["mstar"][group_idx]),
        "mstar_msun": float(cat["mstar"][group_idx] * MSUN_PER_CODE / hdr["h"]),
        "catalog_r200_ckpch": catalog_r200_ckpch,
        "catalog_r200_pkpc": float(catalog_r200_pkpc),
        "catalog_m200_code": catalog_m200_code,
        "catalog_m200_msun": float(catalog_m200_msun),
        "so_to_catalog_r_ratio": float(so_to_catalog_r_ratio),
        "so_to_catalog_m_ratio": float(so_to_catalog_m_ratio),
        "so_catalog_mismatch": bool(so_catalog_mismatch),
        "so_is_lower_limit": bool(so.get("is_lower_limit", False)),
        "so_n_particles": int(so.get("n_particles", 0)),
        "used_catalog_fallback": bool(so.get("used_catalog_fallback", False)),
        "mass_ratio_group_to_so": float(mass_ratio),
        "likely_bridged": bool(likely_bridged),
        "max_offset_ckpch_used": float(max_offset_ckpch),
        "h": hdr["h"],
        "a": hdr["a"],
    }


def get_halo569_reference(
    output_dir: str | Path,
    snap_num_z0: Optional[int] = None,
    verbose: bool = True,
    refine_center: bool = True,
) -> dict:
    """Establish the z=0/last-snapshot reference for Halo 569.

    Parameters
    ----------
    refine_center : bool, default True
        If True, refine the catalog/FOF center with the gas-density shrinking
        sphere. If False, freeze the center definition to the catalog/FOF
        center. This is useful for comparing several runs with an identical
        centering convention.

    Note
    ----
    This is the anchor every other snapshot gets matched against, so if IT
    is bridged, everything downstream inherits the problem silently -- there
    is no earlier "previous validated halo" to scale the offset cap against
    here, so this call uses the conservative FALLBACK_MAX_OFFSET_CKPCH. If
    the returned dict has likely_bridged=True, treat the reference itself as
    untrustworthy: try a different snap_num_z0 (e.g. one snapshot earlier)
    rather than proceeding.
    """
    output_dir = Path(output_dir)
    if snap_num_z0 is None:
        snap_num_z0 = find_last_snap_num(output_dir)
        if snap_num_z0 is None:
            raise RuntimeError(f"No complete snapshot found in {output_dir}")

    snapdir = output_dir / f"snapdir_{snap_num_z0:03d}"
    groups_dir = output_dir / f"groups_{snap_num_z0:03d}"
    hdr = read_snap_header(snapdir)
    cat = read_fof_catalog(groups_dir, snap_num_z0)
    if cat is None:
        raise RuntimeError(f"No FOF groups in {groups_dir}")

    override_idx = _get_z0_override(output_dir)
    if override_idx is not None:
        idx, sel_by = int(override_idx), f"override idx={override_idx}"
    else:
        idx, sel_by = _select_primary_halo_idx(cat)

    halo = _build_halo_result(
        snapdir,
        cat["pos"][idx],
        hdr,
        idx,
        cat,
        refine_center=refine_center,
        verbose=verbose,
        max_offset_ckpch=FALLBACK_MAX_OFFSET_CKPCH,
    )
    halo.update({
        "center_ckpch": halo["center"],      # backward-compatible alias
        "box_ckpch": hdr["box"],
        "snap_num_z0": int(snap_num_z0),
        "selection": sel_by,
        "refine_center": bool(refine_center),
    })

    if verbose:
        off = np.linalg.norm(_periodic_delta(halo["center"][None, :], halo["center_fof"], hdr["box"])[0])
        center_label = "refined center" if refine_center else "FOF center (frozen)"
        print(f"[halo_utils] Halo 569 reference, snap {snap_num_z0:03d}, selected by {sel_by}")
        print(f"  group idx        : {idx} / {len(cat['pos'])}")
        print(f"  FOF center       : {halo['center_fof']}")
        print(f"  {center_label:16s}: {halo['center']}  offset={off:.1f} ckpc/h")
        print(f"  R200c SO         : {halo['r200_ckpch']:.1f} ckpc/h  ({halo['r200_pkpc']:.1f} pkpc)")
        print(f"  M200c SO         : {halo['m200_msun']:.3e} Msun")
        print(f"  GroupMass diag   : {halo['group_mass_msun']:.3e} Msun  (ratio to M200c_SO: {halo['mass_ratio_group_to_so']:.1f}x)")
        print(f"  Catalog R/M diag : {halo['catalog_r200_pkpc']:.1f} pkpc, {halo['catalog_m200_msun']:.3e} Msun")
        print(f"  SO/catalog ratio : R={halo['so_to_catalog_r_ratio']:.3f}, M={halo['so_to_catalog_m_ratio']:.3f}")
        if halo["so_is_lower_limit"]:
            print("  WARNING: SO radius is a lower limit; increase DEFAULT_SO_RMAX_CKPCH")
        if halo.get("used_catalog_fallback", False):
            print("  WARNING: used catalog Group_R_Crit200/Group_M_Crit200 fallback")
        if halo["likely_bridged"]:
            print(
                "  WARNING: reference FOF GroupMass is likely bridged. This does "
                "not automatically invalidate the FOF position or particle-based "
                "SO result; check SO/catalog agreement below."
            )
        if halo.get("so_catalog_mismatch", False):
            print(
                "  WARNING: reference particle-SO and catalog-SO disagree strongly. "
                "This is a genuinely suspect anchor; consider another snap_num_z0 "
                "or disable center refinement."
            )

    return halo


def get_halo569(
    groups_dir: str | Path,
    snap_num: int,
    ref: dict,
    search_radius_ckpch: float = HALO569_SEARCH_RADIUS_CKPCH,
    verbose: bool = True,
    refine_center: bool = True,
    prev_halo: Optional[dict] = None,
) -> Optional[dict]:
    """LEGACY position-based matcher; compute particle SO R200c around its selected group.

    Parameters
    ----------
    refine_center : bool, default True
        Set False to freeze the center definition to the catalog/FOF center
        for this snapshot.
    prev_halo : dict, optional
        The last accepted halo dict in the tracking chain. Its center is used
        as the positional reference for choosing the nearest FOF group, and its
        R200 is used to scale the shrinking-sphere offset cap
        (max(20, 3*r200)). This is what makes the tracking genuinely chained
        rather than repeatedly matching every snapshot to the z=0 coordinate. If omitted, the
        conservative FALLBACK_MAX_OFFSET_CKPCH is used instead. Prefer
        get_halo569_series() over calling this directly in a loop, since it
        handles robust main-progenitor tracking for you. New code should
        prefer get_halo569_series() or get_halo569_at_snapshot().
    """
    groups_dir = Path(groups_dir)
    output_dir = groups_dir.parent
    snapdir = output_dir / f"snapdir_{snap_num:03d}"
    cat = read_fof_catalog(groups_dir, snap_num)
    if cat is None:
        return None
    hdr = read_snap_header(snapdir)
    hdr["box"] = ref.get("box_ckpch", hdr["box"])

    # True chained positional matching:
    # if a previously validated halo exists, use ITS center as the positional
    # reference. Only fall back to the original z=0 reference when no prior
    # accepted halo is available.
    match_ref = prev_halo if prev_halo is not None else ref
    ref_pos = np.asarray(
        match_ref["center_ckpch"] if "center_ckpch" in match_ref else match_ref["center"],
        dtype=float,
    )
    dx = _periodic_delta(cat["pos"], ref_pos, hdr["box"])
    dist = np.sqrt((dx * dx).sum(axis=1))
    within = dist <= search_radius_ckpch

    if within.any():
        # Choose nearest non-tiny group. This avoids selecting small satellites near the reference.
        valid = within & (cat["group_mass"] > 1.0)
        if valid.any():
            idx = int(np.argmin(np.where(valid, dist, np.inf)))
            sel_by = "nearest to reference"
        else:
            idx, sel_by = _select_primary_halo_idx(cat)
    else:
        idx, sel_by = _select_primary_halo_idx(cat)

    if prev_halo is not None and prev_halo.get("r200_ckpch", 0) > 0:
        offset_cap = max(20.0, 3.0 * prev_halo["r200_ckpch"])
    else:
        offset_cap = FALLBACK_MAX_OFFSET_CKPCH

    halo = _build_halo_result(
        snapdir,
        cat["pos"][idx],
        hdr,
        idx,
        cat,
        refine_center=refine_center,
        verbose=False,
        max_offset_ckpch=offset_cap,
    )
    halo.update({
        "dist_ckpch": float(dist[idx]),
        "n_within": int(within.sum()),
        "used_fallback": not bool(within.any()),
        "selection": sel_by,
        "refine_center": bool(refine_center),
    })

    if verbose:
        center_mode = "refined" if refine_center else "FOF/frozen"
        ref_kind = "previous halo" if prev_halo is not None else "z=0 reference"
        print(f"  [halo569] snap {snap_num:03d}: group {idx}, {sel_by} to {ref_kind}, "
              f"dist={dist[idx]:.0f} ckpc/h, center={center_mode}")
        print(f"  [halo569] R200c={halo['r200_pkpc']:.1f} pkpc, M200c={halo['m200_msun']:.3e} Msun")
        print(f"  [halo569] diagnostics: GroupMass={halo['group_mass_msun']:.3e} Msun "
              f"(ratio {halo['mass_ratio_group_to_so']:.1f}x), "
              f"catalog R200c={halo['catalog_r200_pkpc']:.1f} pkpc, "
              f"SO/catalog R={halo['so_to_catalog_r_ratio']:.2f}, "
              f"M={halo['so_to_catalog_m_ratio']:.2f}")
        if halo["likely_bridged"]:
            print(f"  [halo569] WARNING: snap {snap_num:03d} likely FOF-bridged; do not trust this center/R200/M200")

    return halo


def get_halo569_series(
    output_dir: str | Path,
    snap_nums: Sequence[int],
    ref: Optional[dict] = None,
    verbose: bool = True,
    refine_center: bool = False,
) -> Dict[int, Optional[dict]]:
    """Track Halo 569 backward using a STRICT trusted DM-core anchor.

    The final/reference halo is the initial trusted anchor.  Each earlier
    snapshot is tested against the conserved PartType1 core IDs of the most
    recent trusted halo.

    Crucially, failed or weak snapshots DO NOT advance the chain.  If 044->043
    fails, the next attempt is 044->042, then 044->041, etc.  This turns bad
    catalogs/merger phases into gaps rather than branch switches.

    The tracker walks every complete intervening snapshot even for sparse user
    requests, but returns only the requested epochs.
    """
    output_dir = Path(output_dir)
    requested = sorted(set(int(s) for s in snap_nums))
    if not requested:
        return {}

    if ref is None:
        ref = get_halo569_reference(
            output_dir,
            snap_num_z0=None,
            verbose=verbose,
            refine_center=refine_center,
        )

    ref_snap = int(ref["snap_num_z0"])
    if max(requested) > ref_snap:
        raise ValueError(
            f"Requested snapshot {max(requested)} is later than reference "
            f"snapshot {ref_snap}."
        )

    available = [s for s, _, _ in find_snapshots(output_dir)]
    available_set = set(available)
    min_requested = min(requested)

    chain = sorted(
        [s for s in available if min_requested <= s <= ref_snap],
        reverse=True,
    )
    if ref_snap not in chain:
        chain.insert(0, ref_snap)

    tracked_all: Dict[int, Optional[dict]] = {ref_snap: ref}

    trusted_snap = ref_snap
    trusted_halo = ref

    if verbose:
        print(
            f"[halo569-ID] STRICT conserved-DM-core tracking: "
            f"{ref_snap:03d} -> {min_requested:03d}"
        )
        print(
            f"[halo569-ID] Walking {len(chain)} complete snapshots; "
            "failed matches become gaps and NEVER replace the trusted anchor."
        )

    for earlier_snap in chain:
        if earlier_snap == ref_snap:
            continue

        halo = _select_dm_main_progenitor(
            output_dir,
            earlier_snap=earlier_snap,
            later_snap=trusted_snap,
            later_halo=trusted_halo,
            refine_center=refine_center,
            verbose=verbose,
        )

        if halo is None:
            tracked_all[earlier_snap] = None
            if verbose:
                print(
                    f"  [halo569-ID] snap {earlier_snap:03d}: rejected/gap; "
                    f"trusted anchor remains {trusted_snap:03d}"
                )
            continue

        # Only accepted strong matches become the next trusted anchor.
        tracked_all[earlier_snap] = halo
        trusted_snap = earlier_snap
        trusted_halo = halo

    missing = [s for s in requested if s not in available_set and s != ref_snap]
    if missing and verbose:
        print(
            f"[halo569-ID] WARNING: requested snapshots without complete "
            f"snapshot/catalog pairs: {missing}"
        )

    return {s: tracked_all.get(s) for s in requested}


def get_halo569_at_snapshot(
    output_dir: str | Path,
    snap_num: int,
    ref: Optional[dict] = None,
    verbose: bool = True,
    refine_center: bool = False,
) -> Optional[dict]:
    """Safely return Halo 569 at one historical snapshot.

    Unlike calling get_halo569_reference(..., snap_num_z0=snap_num), this does
    NOT re-select the most stellar-massive object at that epoch. It anchors at
    the true final Halo 569 and walks the DM-ID main-progenitor chain through
    every intervening snapshot.

    Use this helper in single-snapshot analysis scripts.
    """
    result = get_halo569_series(
        output_dir,
        [int(snap_num)],
        ref=ref,
        verbose=verbose,
        refine_center=refine_center,
    )
    return result.get(int(snap_num))


# -----------------------------------------------------------------------------
# Particle loading helpers
# -----------------------------------------------------------------------------

def load_particles_within_radius(
    snap_path: str | Path,
    center_ckpch: np.ndarray,
    radius_ckpch: float,
    part_types: Iterable[int] = (0, 4, 6),
    fields_by_type: Optional[dict[int, list[str]]] = None,
) -> dict:
    """Load selected particle fields within a ckpc/h aperture."""
    chunks = glob_snap_chunks(snap_path)
    hdr = read_snap_header(snap_path)
    center = np.asarray(center_ckpch, dtype=float)
    if fields_by_type is None:
        fields_by_type = _SPATIAL_FIELDS

    buffers = {pt: {field: [] for field in fields_by_type.get(pt, ["Coordinates", "Masses"])} for pt in part_types}

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            box = float(f["Header"].attrs["BoxSize"])
            mass_table = f["Header"].attrs.get("MassTable", np.zeros(7))
            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f or "Coordinates" not in f[key]:
                    continue
                grp = f[key]
                coords = grp["Coordinates"][:]
                dx = _periodic_delta(coords, center, box)
                r = np.sqrt((dx * dx).sum(axis=1))
                mask = r <= radius_ckpch
                if not mask.any():
                    continue
                for field in fields_by_type.get(pt, ["Coordinates", "Masses"]):
                    if field in grp:
                        arr = grp[field][:]
                        buffers[pt][field].append(arr[mask] if arr.ndim == 1 else arr[mask])
                    elif field == "Masses" and len(mass_table) > pt and mass_table[pt] > 0:
                        buffers[pt][field].append(np.full(mask.sum(), float(mass_table[pt])))

    out = {}
    for pt, fieldbufs in buffers.items():
        out[pt] = {}
        for field, bufs in fieldbufs.items():
            if bufs:
                out[pt][field] = np.concatenate(bufs) if bufs[0].ndim == 1 else np.vstack(bufs)
    return out


def load_particles_within_r200(snap_path: str | Path, halo: dict, part_types: Iterable[int] = (4, 6)) -> dict:
    """Backward-compatible wrapper around load_particles_within_radius."""
    return load_particles_within_radius(snap_path, halo["center"], halo["r200_ckpch"], part_types=part_types)


# -----------------------------------------------------------------------------
# Backward-compatible legacy-ish helpers
# -----------------------------------------------------------------------------

def compute_radial_distance(coords: np.ndarray, center: np.ndarray, box: Optional[float] = None) -> np.ndarray:
    """Distance from center, optionally periodic if box is supplied."""
    dx = np.asarray(coords) - np.asarray(center)[None, :]
    if box is not None:
        dx -= box * np.round(dx / box)
    return np.sqrt((dx * dx).sum(axis=1))


def compute_radial_profile(coords: np.ndarray, masses: np.ndarray, center: np.ndarray, rbins: np.ndarray, box: Optional[float] = None):
    r = compute_radial_distance(coords, center, box=box)
    prof, _ = np.histogram(r, bins=rbins, weights=masses)
    return 0.5 * (rbins[1:] + rbins[:-1]), prof


def convert_code_mass_to_msun(m_code: np.ndarray | float, h: float) -> np.ndarray | float:
    return np.asarray(m_code) * MSUN_PER_CODE / h


def convert_ckpch_to_pkpc(r_ckpch: np.ndarray | float, a: float, h: float) -> np.ndarray | float:
    return np.asarray(r_ckpch) * a / h


def extract_dust_spatially(snapshot_base, halo_center, radius_kpc=None, verbose=True):
    """Legacy wrapper. radius_kpc is interpreted as ckpc/h for compatibility."""
    files = sorted(_glob.glob(f"{snapshot_base}.*.hdf5"))
    if not files:
        files = sorted(_glob.glob(f"{snapshot_base}/*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No snapshot chunks for: {snapshot_base}")
    snap_path = Path(files[0]).parent if Path(files[0]).parent.exists() else snapshot_base
    radius = float(radius_kpc) if radius_kpc is not None else np.inf
    data = load_particles_within_radius(snap_path, np.asarray(halo_center), radius, part_types=(6,))
    result = data.get(6, {})
    if verbose and result and "Coordinates" in result:
        print(f"  Extracted {len(result['Coordinates']):,} dust particles")
    return result if result else None


# =============================================================================
# Generic multi-halo zoom API
# =============================================================================

@dataclass
class HaloReference:
    """Single-snapshot halo definition used by the generic zoom-suite tools."""

    group_index: int
    catalog_center_ckpch: np.ndarray
    catalog_m200_code: float
    catalog_r200_ckpch: float
    chosen_center_ckpch: np.ndarray
    refined_center_ckpch: np.ndarray
    refinement_shift_ckpch: float
    refinement_accepted: bool
    so_m200_code: float
    so_r200_ckpch: float
    boxsize_ckpch: float
    a: float
    z: float
    h: float
    omega_m: float
    omega_lambda: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _piece_key(filename: str):
    base = os.path.basename(filename)
    try:
        return int(base.split(".")[-2])
    except Exception:
        return base


def find_snapshot_and_group_files(
    output_dir: str | Path,
    snap: int,
) -> Tuple[List[str], List[str]]:
    """Return numerically sorted snapshot and FOF/SUBFIND catalog chunks."""
    output_dir = str(output_dir)
    s = f"{int(snap):03d}"
    snap_files = sorted(
        glob.glob(os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.*.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapshot_{s}.*.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapshot_{s}.hdf5")),
        key=_piece_key,
    )

    group_files: List[str] = []
    for prefix in _CATALOG_PREFIXES:
        found = sorted(
            glob.glob(
                os.path.join(
                    output_dir,
                    f"groups_{s}",
                    f"{prefix}_{s}.*.hdf5",
                )
            )
            + glob.glob(
                os.path.join(
                    output_dir,
                    f"groups_{s}",
                    f"{prefix}_{s}.hdf5",
                )
            ),
            key=_piece_key,
        )
        if found:
            group_files = found
            break

    if not snap_files:
        raise FileNotFoundError(
            f"No snapshot files found for snap {s} in {output_dir}"
        )
    if not group_files:
        raise FileNotFoundError(
            f"No FOF/SUBFIND catalog files found for snap {s} in {output_dir}"
        )
    return snap_files, group_files


def read_header(snapshot_file: str | Path) -> Dict[str, float]:
    """Read expansion state, box size, and cosmology from one snapshot chunk."""
    with h5py.File(snapshot_file, "r") as f:
        header = f["Header"].attrs
        params = f["Parameters"].attrs if "Parameters" in f else {}
        a = float(np.asarray(header["Time"]).squeeze())
        z = float(np.asarray(header.get("Redshift", 1.0 / a - 1.0)).squeeze())

        def get_param(name: str) -> float:
            if name in params:
                return float(np.asarray(params[name]).squeeze())
            if name in header:
                return float(np.asarray(header[name]).squeeze())
            raise KeyError(
                f"Required cosmology parameter {name!r} is absent from "
                f"/Parameters and /Header of {snapshot_file}"
            )

        return {
            "box": float(np.asarray(header["BoxSize"]).squeeze()),
            "a": a,
            "z": z,
            "h": get_param("HubbleParam"),
            "omega_m": get_param("Omega0"),
            "omega_lambda": get_param("OmegaLambda"),
        }


def concat_group_dataset(
    group_files: Sequence[str | Path],
    dataset: str,
) -> np.ndarray:
    arrays = []
    for filename in group_files:
        with h5py.File(filename, "r") as f:
            if "Group" in f and dataset in f["Group"]:
                arrays.append(f["Group"][dataset][()])
    if not arrays:
        raise KeyError(f"Group/{dataset} not found")
    return np.concatenate(arrays, axis=0)


def read_catalog_halos(
    group_files: Sequence[str | Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(
        concat_group_dataset(group_files, "GroupPos"), dtype=np.float64
    )
    m200 = np.asarray(
        concat_group_dataset(group_files, "Group_M_Crit200"), dtype=np.float64
    )
    r200 = np.asarray(
        concat_group_dataset(group_files, "Group_R_Crit200"), dtype=np.float64
    )
    return pos, m200, r200


def periodic_delta(
    coords: np.ndarray,
    center: np.ndarray,
    box: float,
) -> np.ndarray:
    """Minimum-image displacement from center in a periodic box."""
    return _periodic_delta(
        np.asarray(coords, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        float(box),
    )


def wrap_position(position: np.ndarray, box: float) -> np.ndarray:
    return np.mod(np.asarray(position, dtype=np.float64), float(box))


def read_particles_within(
    snapshot_files: Sequence[str | Path],
    center_ckpch: np.ndarray,
    radius_ckpch: float,
    part_types: Iterable[int],
    box_ckpch: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Load coordinates, masses, and IDs inside one periodic sphere."""
    requested = tuple(int(pt) for pt in part_types)
    buffers: Dict[int, Dict[str, list[np.ndarray]]] = {
        pt: {"Coordinates": [], "Masses": [], "ParticleIDs": []}
        for pt in requested
    }

    for filename in snapshot_files:
        with h5py.File(filename, "r") as f:
            mass_table = np.asarray(
                f["Header"].attrs.get("MassTable", np.zeros(7)),
                dtype=np.float64,
            )
            for pt in requested:
                name = f"PartType{pt}"
                if name not in f or "Coordinates" not in f[name]:
                    continue
                group = f[name]
                coords = np.asarray(group["Coordinates"], dtype=np.float64)
                delta = periodic_delta(coords, center_ckpch, box_ckpch)
                radius2 = np.einsum("ij,ij->i", delta, delta)
                selected = radius2 <= float(radius_ckpch) ** 2
                if not np.any(selected):
                    continue

                buffers[pt]["Coordinates"].append(coords[selected])
                if "Masses" in group:
                    buffers[pt]["Masses"].append(
                        np.asarray(group["Masses"], dtype=np.float64)[selected]
                    )
                elif pt < len(mass_table) and mass_table[pt] > 0:
                    buffers[pt]["Masses"].append(
                        np.full(np.count_nonzero(selected), mass_table[pt])
                    )
                if "ParticleIDs" in group:
                    buffers[pt]["ParticleIDs"].append(
                        np.asarray(group["ParticleIDs"])[selected]
                    )

    packed: Dict[int, Dict[str, np.ndarray]] = {}
    for pt in requested:
        if not buffers[pt]["Coordinates"]:
            continue
        packed[pt] = {
            "Coordinates": np.concatenate(buffers[pt]["Coordinates"]),
            "Masses": np.concatenate(buffers[pt]["Masses"]),
        }
        if buffers[pt]["ParticleIDs"]:
            packed[pt]["ParticleIDs"] = np.concatenate(
                buffers[pt]["ParticleIDs"]
            )
    return packed


def combine_particle_sets(
    data: Dict[int, Dict[str, np.ndarray]],
    part_types: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    coords, masses = [], []
    for pt in part_types:
        if pt in data:
            coords.append(data[pt]["Coordinates"])
            masses.append(data[pt]["Masses"])
    if not coords:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(coords), np.concatenate(masses)


def shrinking_sphere_center(
    coords: np.ndarray,
    masses: np.ndarray,
    initial_center: np.ndarray,
    initial_radius: float,
    box: float,
    shrink_factor: float = 0.85,
    min_particles: int = 500,
    min_radius_fraction: float = 0.03,
    max_iter: int = 64,
) -> np.ndarray:
    """Conservative mass-weighted shrinking-sphere candidate center."""
    center = np.asarray(initial_center, dtype=np.float64).copy()
    radius = float(initial_radius)
    min_radius = radius * float(min_radius_fraction)

    for _ in range(max_iter):
        delta = periodic_delta(coords, center, box)
        radius2 = np.einsum("ij,ij->i", delta, delta)
        selected = radius2 <= radius * radius
        if np.count_nonzero(selected) < min_particles or radius <= min_radius:
            break
        weights = masses[selected]
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
            break
        center = wrap_position(
            center + np.average(delta[selected], axis=0, weights=weights),
            box,
        )
        radius *= shrink_factor
    return center


def critical_density_msun_pkpc3(
    a: float,
    h: float,
    omega_m: float,
    omega_lambda: float,
) -> float:
    """Critical density in Msun/pkpc^3."""
    omega_k = 1.0 - omega_m - omega_lambda
    e2 = omega_m / a**3 + omega_k / a**2 + omega_lambda
    hubble_km_s_kpc = 100.0 * h * np.sqrt(e2) / 1000.0
    g_kpc_kms2_msun = 4.30091e-6
    return 3.0 * hubble_km_s_kpc**2 / (
        8.0 * np.pi * g_kpc_kms2_msun
    )


def spherical_overdensity_200c(
    coords: np.ndarray,
    masses_code: np.ndarray,
    center_ckpch: np.ndarray,
    box_ckpch: float,
    a: float,
    h: float,
    omega_m: float,
    omega_lambda: float,
) -> Tuple[float, float]:
    """Return ``(M200c_code, R200c_ckpch)`` from loaded particle data."""
    delta = periodic_delta(coords, center_ckpch, box_ckpch)
    radii = np.linalg.norm(delta, axis=1)
    good = (
        np.isfinite(radii)
        & np.isfinite(masses_code)
        & (masses_code >= 0)
        & (radii > 0)
    )
    radii = radii[good]
    masses = np.asarray(masses_code, dtype=float)[good]
    if len(radii) < 10:
        raise RuntimeError("Too few particles for the SO calculation")

    order = np.argsort(radii)
    radii = radii[order]
    mass_enclosed_code = np.cumsum(masses[order])
    radii_pkpc = radii * a / h
    mass_enclosed_msun = mass_enclosed_code * MSUN_PER_CODE / h
    mean_density = mass_enclosed_msun / (
        (4.0 / 3.0) * np.pi * radii_pkpc**3
    )
    target = 200.0 * critical_density_msun_pkpc3(
        a, h, omega_m, omega_lambda
    )
    above = np.where(mean_density >= target)[0]
    if len(above) == 0:
        raise RuntimeError(
            "Mean enclosed density is below 200 rho_crit at the innermost particle"
        )
    i = int(above[-1])
    if i >= len(radii) - 1:
        raise RuntimeError(
            "SO crossing lies beyond loaded radius; enlarge the search region"
        )

    x1, x2 = np.log(radii_pkpc[i]), np.log(radii_pkpc[i + 1])
    y1, y2 = np.log(mean_density[i]), np.log(mean_density[i + 1])
    yt = np.log(target)
    fraction = 0.0 if y2 == y1 else np.clip((yt-y1)/(y2-y1), 0.0, 1.0)
    r200_pkpc = float(np.exp(x1 + fraction * (x2 - x1)))
    r200_ckpch = r200_pkpc * h / a
    m200_msun = (4.0 / 3.0) * np.pi * r200_pkpc**3 * target
    return float(m200_msun * h / MSUN_PER_CODE), float(r200_ckpch)


def get_zoom_halo(
    output_dir: str | Path,
    snap: int,
    group_index: Optional[int] = None,
    refine_center: bool = True,
    max_refine_shift_r200: float = 0.10,
    refine_types: Sequence[int] = (0, 1, 4),
    so_types: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    verbose: bool = True,
) -> HaloReference:
    """Define one explicitly selected zoom halo and recompute particle SO values.

    ``group_index`` is deliberately required.  FOF indices are local to an
    output and snapshot; a parent halo ID is not a valid substitute.  This
    prevents a dwarf analysis from silently selecting the largest halo in the
    periodic volume.
    """
    if group_index is None:
        raise ValueError(
            "group_index is required for get_zoom_halo(); identify the final "
            "target from its MUSIC2-shifted parent position and expected mass"
        )

    snapshot_files, group_files = find_snapshot_and_group_files(output_dir, snap)
    header = read_header(snapshot_files[0])
    positions, catalog_m200, catalog_r200 = read_catalog_halos(group_files)
    index = int(group_index)
    if index < 0 or index >= len(catalog_m200):
        raise IndexError(
            f"group_index={index} outside 0..{len(catalog_m200)-1}"
        )

    catalog_center = np.asarray(positions[index], dtype=np.float64)
    catalog_mass = float(catalog_m200[index])
    catalog_radius = float(catalog_r200[index])
    if catalog_mass <= 0 or catalog_radius <= 0:
        raise RuntimeError(
            f"Group {index} has invalid catalog M200c/R200c"
        )

    centering_data = read_particles_within(
        snapshot_files,
        catalog_center,
        1.25 * catalog_radius,
        refine_types,
        header["box"],
    )
    center_coords, center_masses = combine_particle_sets(
        centering_data, refine_types
    )
    refined = catalog_center.copy()
    shift = 0.0
    accepted = False
    if refine_center and len(center_coords) >= 500:
        refined = shrinking_sphere_center(
            center_coords,
            center_masses,
            catalog_center,
            catalog_radius,
            header["box"],
        )
        shift = float(
            np.linalg.norm(
                periodic_delta(
                    refined[None, :], catalog_center, header["box"]
                )[0]
            )
        )
        accepted = shift <= max_refine_shift_r200 * catalog_radius
    chosen = refined if accepted else catalog_center

    last_error: Optional[Exception] = None
    so_mass = so_radius = np.nan
    for factor in (2.5, 4.0):
        so_data = read_particles_within(
            snapshot_files,
            chosen,
            factor * catalog_radius,
            so_types,
            header["box"],
        )
        so_coords, so_masses = combine_particle_sets(so_data, so_types)
        try:
            so_mass, so_radius = spherical_overdensity_200c(
                so_coords,
                so_masses,
                chosen,
                header["box"],
                header["a"],
                header["h"],
                header["omega_m"],
                header["omega_lambda"],
            )
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error

    reference = HaloReference(
        group_index=index,
        catalog_center_ckpch=catalog_center,
        catalog_m200_code=catalog_mass,
        catalog_r200_ckpch=catalog_radius,
        chosen_center_ckpch=chosen,
        refined_center_ckpch=refined,
        refinement_shift_ckpch=shift,
        refinement_accepted=accepted,
        so_m200_code=float(so_mass),
        so_r200_ckpch=float(so_radius),
        boxsize_ckpch=header["box"],
        a=header["a"],
        z=header["z"],
        h=header["h"],
        omega_m=header["omega_m"],
        omega_lambda=header["omega_lambda"],
    )
    if verbose:
        print(f"[halo_utils] snap={snap:03d} target group={index}")
        print(
            f"  M200c={so_mass*MSUN_PER_CODE/header['h']:.6e} Msun  "
            f"R200c={so_radius*header['a']/header['h']:.3f} pkpc"
        )
        print(
            f"  center shift={shift:.3f} ckpc/h; accepted={accepted}"
        )
    return reference


def _zoom_reference_to_tracking_dict(
    reference: HaloReference,
    snap_num: int,
) -> dict:
    """Adapt the public dataclass to the established DM-core tracker."""
    h = reference.h
    a = reference.a
    return {
        "center": reference.chosen_center_ckpch.copy(),
        "center_ckpch": reference.chosen_center_ckpch.copy(),
        "center_fof": reference.catalog_center_ckpch.copy(),
        "r200_ckpch": reference.so_r200_ckpch,
        "r200_pkpc": reference.so_r200_ckpch * a / h,
        "m200_code": reference.so_m200_code,
        "m200_msun": reference.so_m200_code * MSUN_PER_CODE / h,
        "group_idx": reference.group_index,
        "catalog_r200_ckpch": reference.catalog_r200_ckpch,
        "catalog_r200_pkpc": reference.catalog_r200_ckpch * a / h,
        "catalog_m200_code": reference.catalog_m200_code,
        "catalog_m200_msun": reference.catalog_m200_code * MSUN_PER_CODE / h,
        "box_ckpch": reference.boxsize_ckpch,
        "snap_num_z0": int(snap_num),
        "h": h,
        "a": a,
        "z": reference.z,
        "selection": "explicit final target group",
        "refine_center": True,
    }


def get_zoom_halo_series(
    output_dir: str | Path,
    snap_nums: Sequence[int],
    group_index: int,
    reference_snap: Optional[int] = None,
    verbose: bool = True,
    refine_reference_center: bool = True,
    refine_historical_centers: bool = False,
) -> Dict[int, Optional[dict]]:
    """Track an explicitly anchored zoom target backward with HRDM core IDs.

    ``group_index`` identifies the target only at ``reference_snap`` (normally
    the final snapshot).  Earlier FOF indices are discovered from conserved
    PartType1 IDs and must never be assumed equal to the final index.
    """
    output_dir = Path(output_dir)
    requested = sorted(set(int(value) for value in snap_nums))
    if not requested:
        return {}

    if reference_snap is None:
        reference_snap = find_last_snap_num(output_dir)
    if reference_snap is None:
        raise RuntimeError(f"No complete snapshot/catalog pair in {output_dir}")
    reference_snap = int(reference_snap)
    if max(requested) > reference_snap:
        raise ValueError(
            f"Requested snapshot {max(requested)} is later than reference "
            f"snapshot {reference_snap}"
        )

    public_reference = get_zoom_halo(
        output_dir,
        reference_snap,
        group_index=int(group_index),
        refine_center=refine_reference_center,
        verbose=verbose,
    )
    reference = _zoom_reference_to_tracking_dict(
        public_reference, reference_snap
    )

    available = [snap for snap, _, _ in find_snapshots(output_dir)]
    available_set = set(available)
    minimum = min(requested)
    chain = sorted(
        [snap for snap in available if minimum <= snap <= reference_snap],
        reverse=True,
    )
    if reference_snap not in chain:
        chain.insert(0, reference_snap)

    tracked: Dict[int, Optional[dict]] = {reference_snap: reference}
    trusted_snap = reference_snap
    trusted_halo = reference

    if verbose:
        print(
            f"[halo-track] conserved-HRDM-core history: "
            f"{reference_snap:03d} -> {minimum:03d}"
        )

    for earlier_snap in chain:
        if earlier_snap == reference_snap:
            continue
        halo = _select_dm_main_progenitor(
            output_dir,
            earlier_snap=earlier_snap,
            later_snap=trusted_snap,
            later_halo=trusted_halo,
            refine_center=refine_historical_centers,
            verbose=verbose,
        )
        tracked[earlier_snap] = halo
        if halo is not None:
            trusted_snap = earlier_snap
            trusted_halo = halo

    missing = [
        snap for snap in requested
        if snap not in available_set and snap != reference_snap
    ]
    if missing and verbose:
        print(
            f"[halo-track] requested snapshots without complete catalogs: {missing}"
        )
    return {snap: tracked.get(snap) for snap in requested}
