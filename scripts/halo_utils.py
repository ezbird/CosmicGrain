"""
halo_utils.py
=============
Clean shared utilities for CosmicGrain Halo 569 analysis.

Core philosophy
---------------
1. Use the FOF/Subfind catalog only to identify the target group and get an
   initial center.
2. Compute R200c/M200c directly from particles around that center using a true
   spherical-overdensity calculation.
3. Keep units explicit everywhere.

Unit conventions
----------------
Snapshot/catalog positions : comoving kpc/h  (ckpc/h)
Snapshot/catalog masses    : 1e10 Msun/h     (Gadget code mass units)
Physical distance          : pkpc = ckpc/h * a / h
Physical mass              : Msun = code_mass * 1e10 / h
Returned halo['center']    : ckpc/h
Returned halo['r200_ckpch']: ckpc/h
Returned halo['r200_pkpc'] : physical kpc
Returned halo['m200_code'] : 1e10 Msun/h
Returned halo['m200_msun'] : Msun

Primary API
-----------
    ref  = get_halo569_reference(output_dir)
    halo = get_halo569(groups_dir, snap_num, ref)

Optional particle loader:
    pdata = load_particles_within_radius(snapdir, halo['center'], halo['r200_ckpch'])

Notes
-----
This version deliberately does NOT use GroupMass as M200 and does NOT derive
R200 from GroupMass. FOF mass can include bridges/companions and is not a
spherical-overdensity mass. Catalog Group_R_Crit200/Group_M_Crit200 are kept
as diagnostics only.
"""

from __future__ import annotations

import re
import glob as _glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np

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

# For loading spatial particle subsets.
_SPATIAL_FIELDS = {
    0: ["Coordinates", "Masses", "Density", "Metallicity", "InternalEnergy", "StarFormationRate"],
    1: ["Coordinates", "Velocities", "ParticleIDs"],
    2: ["Coordinates", "Velocities", "ParticleIDs"],
    4: ["Coordinates", "Masses", "Velocities", "Metallicity", "StellarFormationTime", "ParticleIDs"],
    5: ["Coordinates", "Masses", "Velocities", "ParticleIDs"],
    6: ["Coordinates", "Masses", "GrainRadius", "GrainType", "DustTemperature", "CarbonFraction", "Velocities", "ParticleIDs"],
}

# Per-resolution override for z=0 target selection.
# These are global FOF group indices across all catalog chunks.
_HALO569_Z0_OVERRIDES = {
    "2048": 4,
}


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
    """Return sorted FOF/Subfind catalog chunks for a snapshot."""
    chunks = sorted(Path(groups_dir).glob(f"fof_subhalo_tab_{snap_num:03d}*.hdf5"))
    if not chunks:
        raise FileNotFoundError(f"No catalog for snap {snap_num:03d} in {groups_dir}")
    return chunks


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
        if groups_dir.exists() and list(groups_dir.glob("fof_subhalo_tab_*.hdf5")):
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
        if groups_dir.exists() and list(groups_dir.glob("fof_subhalo_tab_*.hdf5")):
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

    # Crossing between i1 above and i2 below.
    i2 = int(np.argmax(~above))
    i1 = i2 - 1
    if i1 < 0:
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
) -> dict:
    """Refine center if requested, compute SO R200c/M200c, return standard halo dict."""
    center0 = np.asarray(fof_center, dtype=float)
    center = center0.copy()

    if refine_center and snapdir.exists():
        center = find_shrinking_sphere_center(
            snapdir,
            center0,
            hdr["box"],
            start_r=300.0,
            max_offset_ckpch=300.0,
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

    return {
        "center": center,
        "center_fof": center0,
        "r200_ckpch": so["r_delta_ckpch"],
        "r200_pkpc": so["r_delta_pkpc"],
        "m200_code": so["m_delta_code"],
        "m200_msun": so["m_delta_msun"],
        "group_idx": int(group_idx),
        "group_mass_code": float(cat["group_mass"][group_idx]),
        "group_mass_msun": float(cat["group_mass"][group_idx] * MSUN_PER_CODE / hdr["h"]),
        "mstar_code": float(cat["mstar"][group_idx]),
        "mstar_msun": float(cat["mstar"][group_idx] * MSUN_PER_CODE / hdr["h"]),
        "catalog_r200_ckpch": float(cat["r200_catalog"][group_idx]),
        "catalog_r200_pkpc": float(cat["r200_catalog"][group_idx] * hdr["a"] / hdr["h"]),
        "catalog_m200_code": float(cat["m200_catalog"][group_idx]),
        "catalog_m200_msun": float(cat["m200_catalog"][group_idx] * MSUN_PER_CODE / hdr["h"]),
        "so_is_lower_limit": bool(so.get("is_lower_limit", False)),
        "so_n_particles": int(so.get("n_particles", 0)),
        "used_catalog_fallback": bool(so.get("used_catalog_fallback", False)),
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
        print(f"  GroupMass diag   : {halo['group_mass_msun']:.3e} Msun")
        print(f"  Catalog R/M diag : {halo['catalog_r200_pkpc']:.1f} pkpc, {halo['catalog_m200_msun']:.3e} Msun")
        if halo["so_is_lower_limit"]:
            print("  WARNING: SO radius is a lower limit; increase DEFAULT_SO_RMAX_CKPCH")
        if halo.get("used_catalog_fallback", False):
            print("  WARNING: used catalog Group_R_Crit200/Group_M_Crit200 fallback")

    return halo


def get_halo569(
    groups_dir: str | Path,
    snap_num: int,
    ref: dict,
    search_radius_ckpch: float = HALO569_SEARCH_RADIUS_CKPCH,
    verbose: bool = True,
    refine_center: bool = True,
) -> Optional[dict]:
    """Find Halo 569 near the reference position and compute particle SO R200c.

    Set refine_center=False to freeze the center definition to the catalog/FOF
    center for this snapshot.
    """
    groups_dir = Path(groups_dir)
    output_dir = groups_dir.parent
    snapdir = output_dir / f"snapdir_{snap_num:03d}"
    cat = read_fof_catalog(groups_dir, snap_num)
    if cat is None:
        return None
    hdr = read_snap_header(snapdir)
    hdr["box"] = ref.get("box_ckpch", hdr["box"])

    ref_pos = np.asarray(ref["center_ckpch"] if "center_ckpch" in ref else ref["center"], dtype=float)
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

    halo = _build_halo_result(
        snapdir,
        cat["pos"][idx],
        hdr,
        idx,
        cat,
        refine_center=refine_center,
        verbose=False,
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
        print(f"  [halo569] snap {snap_num:03d}: group {idx}, {sel_by}, dist={dist[idx]:.0f} ckpc/h, center={center_mode}")
        print(f"  [halo569] R200c={halo['r200_pkpc']:.1f} pkpc, M200c={halo['m200_msun']:.3e} Msun")
        print(f"  [halo569] diagnostics: GroupMass={halo['group_mass_msun']:.3e} Msun, catalog R200c={halo['catalog_r200_pkpc']:.1f} pkpc")

    return halo


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
