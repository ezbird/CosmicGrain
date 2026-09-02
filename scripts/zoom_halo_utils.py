#!/usr/bin/env python3
"""
zoom_halo_utils.py

Generic halo utilities for the CosmicGrain zoom suite.

Purpose
-------
Provide one consistent halo definition for all zoom targets without hard-coding
Halo 569. The workflow is:

1. Identify a FOF group from the catalog (default: largest Group_M_Crit200).
2. Use GroupPos as the robust reference center.
3. Optionally compute a shrinking-sphere center from high-resolution matter.
4. Reject the refined center automatically if it shifts implausibly far.
5. Recompute R200c and M200c from particle masses via a true spherical-
   overdensity (SO) calculation.
6. Report catalog and particle-based values side by side.

Units assumed
-------------
Coordinates: ckpc/h
Masses:      Gadget code mass = 1e10 Msun/h
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

G_KPC_KMS2_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun


@dataclass
class HaloReference:
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


def _piece_key(fn: str):
    base = os.path.basename(fn)
    try:
        return int(base.split(".")[-2])
    except Exception:
        return base


def find_snapshot_and_group_files(output_dir: str, snap: int) -> Tuple[List[str], List[str]]:
    s = f"{snap:03d}"
    snap_files = sorted(
        glob.glob(os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.*.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapshot_{s}.*.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.hdf5"))
        + glob.glob(os.path.join(output_dir, f"snapshot_{s}.hdf5")),
        key=_piece_key,
    )
    group_files = sorted(
        glob.glob(os.path.join(output_dir, f"groups_{s}", f"fof_subhalo_tab_{s}.*.hdf5"))
        + glob.glob(os.path.join(output_dir, f"groups_{s}", f"fof_subhalo_tab_{s}.hdf5")),
        key=_piece_key,
    )
    if not snap_files:
        raise FileNotFoundError(f"No snapshot files found for snap {s} in {output_dir}")
    if not group_files:
        raise FileNotFoundError(f"No group catalog files found for snap {s} in {output_dir}")
    return snap_files, group_files


def read_header(snapshot_file: str) -> Dict[str, float]:
    """
    Read expansion state from /Header and cosmological parameters from
    /Parameters when present.

    GADGET-4 commonly stores Time/Redshift/BoxSize in /Header, while
    HubbleParam/Omega0/OmegaLambda live in /Parameters rather than /Header.
    """
    with h5py.File(snapshot_file, "r") as f:
        header = f["Header"].attrs
        params = f["Parameters"].attrs if "Parameters" in f else {}

        a = float(np.asarray(header["Time"]).squeeze())
        z = float(np.asarray(header.get("Redshift", 1.0 / a - 1.0)).squeeze())

        def get_param(name):
            if name in params:
                return float(np.asarray(params[name]).squeeze())
            if name in header:
                return float(np.asarray(header[name]).squeeze())
            raise KeyError(
                f"Required cosmology parameter '{name}' not found in "
                f"/Parameters or /Header of {snapshot_file}"
            )

        return {
            "box": float(np.asarray(header["BoxSize"]).squeeze()),
            "a": a,
            "z": z,
            "h": get_param("HubbleParam"),
            "omega_m": get_param("Omega0"),
            "omega_lambda": get_param("OmegaLambda"),
        }


def concat_group_dataset(group_files: Sequence[str], dataset: str) -> np.ndarray:
    out = []
    for fn in group_files:
        with h5py.File(fn, "r") as f:
            if "Group" in f and dataset in f["Group"]:
                out.append(f["Group"][dataset][()])
    if not out:
        raise KeyError(f"Group/{dataset} not found")
    return np.concatenate(out, axis=0)


def read_catalog_halos(group_files: Sequence[str]):
    pos = np.asarray(concat_group_dataset(group_files, "GroupPos"), dtype=np.float64)
    m200 = np.asarray(concat_group_dataset(group_files, "Group_M_Crit200"), dtype=np.float64)
    r200 = np.asarray(concat_group_dataset(group_files, "Group_R_Crit200"), dtype=np.float64)
    return pos, m200, r200


def periodic_delta(coords: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    d = np.asarray(coords, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    d -= box * np.rint(d / box)
    return d


def wrap_position(pos: np.ndarray, box: float) -> np.ndarray:
    return np.mod(pos, box)


def read_particles_within(
    snapshot_files: Sequence[str],
    center_ckpch: np.ndarray,
    radius_ckpch: float,
    part_types: Iterable[int],
    box_ckpch: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Read only particles within radius_ckpch of center, using periodic distances.
    Returns Coordinates, Masses, and ParticleIDs when available.
    """
    result: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for pt in part_types:
        result[int(pt)] = {"Coordinates": [], "Masses": [], "ParticleIDs": []}

    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            mt = np.asarray(f["Header"].attrs["MassTable"], dtype=np.float64)
            for pt in part_types:
                name = f"PartType{pt}"
                if name not in f:
                    continue
                g = f[name]
                coords = np.asarray(g["Coordinates"], dtype=np.float64)
                d = periodic_delta(coords, center_ckpch, box_ckpch)
                rr2 = np.einsum("ij,ij->i", d, d)
                q = rr2 <= radius_ckpch * radius_ckpch
                if not np.any(q):
                    continue

                result[pt]["Coordinates"].append(coords[q])
                if "Masses" in g:
                    result[pt]["Masses"].append(np.asarray(g["Masses"], dtype=np.float64)[q])
                else:
                    result[pt]["Masses"].append(np.full(np.count_nonzero(q), mt[pt], dtype=np.float64))

                if "ParticleIDs" in g:
                    result[pt]["ParticleIDs"].append(np.asarray(g["ParticleIDs"])[q])

    packed: Dict[int, Dict[str, np.ndarray]] = {}
    for pt in part_types:
        if not result[pt]["Coordinates"]:
            continue
        packed[pt] = {
            "Coordinates": np.concatenate(result[pt]["Coordinates"], axis=0),
            "Masses": np.concatenate(result[pt]["Masses"], axis=0),
        }
        if result[pt]["ParticleIDs"]:
            packed[pt]["ParticleIDs"] = np.concatenate(result[pt]["ParticleIDs"], axis=0)
    return packed


def combine_particle_sets(data: Dict[int, Dict[str, np.ndarray]], part_types: Iterable[int]):
    coords, masses = [], []
    for pt in part_types:
        if pt not in data:
            continue
        coords.append(data[pt]["Coordinates"])
        masses.append(data[pt]["Masses"])
    if not coords:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(coords, axis=0), np.concatenate(masses, axis=0)


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
    """
    Mass-weighted shrinking-sphere center.

    This is deliberately conservative. It is only a candidate center; callers
    should compare it with the catalog center before accepting it.
    """
    center = np.asarray(initial_center, dtype=np.float64).copy()
    radius = float(initial_radius)
    min_radius = initial_radius * float(min_radius_fraction)

    for _ in range(max_iter):
        d = periodic_delta(coords, center, box)
        r2 = np.einsum("ij,ij->i", d, d)
        q = r2 <= radius * radius
        n = int(np.count_nonzero(q))
        if n < min_particles or radius <= min_radius:
            break

        w = masses[q]
        if not np.all(np.isfinite(w)) or np.sum(w) <= 0:
            break

        shift = np.average(d[q], axis=0, weights=w)
        center = wrap_position(center + shift, box)
        radius *= shrink_factor

    return center


def critical_density_msun_pkpc3(a: float, h: float, omega_m: float, omega_lambda: float) -> float:
    omega_k = 1.0 - omega_m - omega_lambda
    Ez2 = omega_m / a**3 + omega_k / a**2 + omega_lambda
    H_km_s_Mpc = 100.0 * h * np.sqrt(Ez2)
    H_km_s_kpc = H_km_s_Mpc / 1000.0
    return 3.0 * H_km_s_kpc**2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)


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
    """
    Return (M200c_code, R200c_ckpch) from particle data around center.

    Coordinates are ckpc/h and masses are 1e10 Msun/h.
    """
    d = periodic_delta(coords, center_ckpch, box_ckpch)
    r_ckpch = np.linalg.norm(d, axis=1)
    good = np.isfinite(r_ckpch) & np.isfinite(masses_code) & (masses_code >= 0) & (r_ckpch > 0)
    r_ckpch = r_ckpch[good]
    m_code = masses_code[good]

    if len(r_ckpch) < 10:
        raise RuntimeError("Too few particles for spherical-overdensity calculation")

    order = np.argsort(r_ckpch)
    r_ckpch = r_ckpch[order]
    m_code = m_code[order]
    menc_code = np.cumsum(m_code)

    r_pkpc = r_ckpch * a / h
    menc_msun = menc_code * 1e10 / h
    rho_mean = menc_msun / ((4.0 / 3.0) * np.pi * r_pkpc**3)

    target = 200.0 * critical_density_msun_pkpc3(a, h, omega_m, omega_lambda)

    above = np.where(rho_mean >= target)[0]
    if len(above) == 0:
        raise RuntimeError("Mean enclosed density is already below 200 rho_crit at innermost particle")

    i = int(above[-1])
    if i >= len(r_ckpch) - 1:
        raise RuntimeError(
            "SO crossing lies beyond loaded particle radius; increase the SO search radius"
        )

    # Log-linear interpolation of density crossing between particles i and i+1.
    x1, x2 = np.log(r_pkpc[i]), np.log(r_pkpc[i + 1])
    y1, y2 = np.log(rho_mean[i]), np.log(rho_mean[i + 1])
    yt = np.log(target)

    if y2 == y1:
        frac = 0.0
    else:
        frac = np.clip((yt - y1) / (y2 - y1), 0.0, 1.0)

    log_r200_pkpc = x1 + frac * (x2 - x1)
    r200_pkpc = float(np.exp(log_r200_pkpc))
    r200_ckpch = r200_pkpc * h / a

    # By definition at 200 rho_crit.
    m200_msun = (4.0 / 3.0) * np.pi * r200_pkpc**3 * target
    m200_code = m200_msun * h / 1e10
    return float(m200_code), float(r200_ckpch)


def get_zoom_halo(
    output_dir: str,
    snap: int,
    group_index: Optional[int] = None,
    refine_center: bool = True,
    max_refine_shift_r200: float = 0.10,
    refine_types: Sequence[int] = (0, 1, 4),
    so_types: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
    verbose: bool = True,
) -> HaloReference:
    snap_files, group_files = find_snapshot_and_group_files(output_dir, snap)
    hdr = read_header(snap_files[0])
    gpos, gm200, gr200 = read_catalog_halos(group_files)

    gi = int(np.nanargmax(gm200)) if group_index is None else int(group_index)
    if gi < 0 or gi >= len(gm200):
        raise IndexError(f"group_index={gi} outside 0..{len(gm200)-1}")

    cat_center = np.asarray(gpos[gi], dtype=np.float64)
    cat_m200 = float(gm200[gi])
    cat_r200 = float(gr200[gi])

    # Load enough high-resolution matter for centering.
    centering_data = read_particles_within(
        snap_files, cat_center, 1.25 * cat_r200, refine_types, hdr["box"]
    )
    ccoords, cmasses = combine_particle_sets(centering_data, refine_types)

    refined = cat_center.copy()
    shift = 0.0
    accepted = False

    if refine_center and len(ccoords) >= 500:
        refined = shrinking_sphere_center(
            ccoords, cmasses, cat_center, cat_r200, hdr["box"]
        )
        shift = float(np.linalg.norm(periodic_delta(refined[None, :], cat_center, hdr["box"])[0]))
        accepted = shift <= max_refine_shift_r200 * cat_r200

    chosen = refined if accepted else cat_center

    # Load a broad region for SO. 2.5 catalog R200 is normally ample and still
    # cheap because spatial filtering is performed piece-by-piece.
    so_radius = 2.5 * cat_r200
    so_data = read_particles_within(
        snap_files, chosen, so_radius, so_types, hdr["box"]
    )
    scoords, smasses = combine_particle_sets(so_data, so_types)

    try:
        so_m200, so_r200 = spherical_overdensity_200c(
            scoords, smasses, chosen, hdr["box"],
            hdr["a"], hdr["h"], hdr["omega_m"], hdr["omega_lambda"],
        )
    except RuntimeError:
        # One retry with a larger region.
        so_radius = 4.0 * cat_r200
        so_data = read_particles_within(
            snap_files, chosen, so_radius, so_types, hdr["box"]
        )
        scoords, smasses = combine_particle_sets(so_data, so_types)
        so_m200, so_r200 = spherical_overdensity_200c(
            scoords, smasses, chosen, hdr["box"],
            hdr["a"], hdr["h"], hdr["omega_m"], hdr["omega_lambda"],
        )

    ref = HaloReference(
        group_index=gi,
        catalog_center_ckpch=cat_center,
        catalog_m200_code=cat_m200,
        catalog_r200_ckpch=cat_r200,
        chosen_center_ckpch=chosen,
        refined_center_ckpch=refined,
        refinement_shift_ckpch=shift,
        refinement_accepted=accepted,
        so_m200_code=so_m200,
        so_r200_ckpch=so_r200,
        boxsize_ckpch=hdr["box"],
        a=hdr["a"],
        z=hdr["z"],
        h=hdr["h"],
        omega_m=hdr["omega_m"],
        omega_lambda=hdr["omega_lambda"],
    )

    if verbose:
        print("=" * 72)
        print("GENERIC ZOOM HALO REFERENCE")
        print("=" * 72)
        print(f"group index        : {gi}")
        print(f"catalog center     : {cat_center}")
        print(f"refined center     : {refined}")
        print(f"refinement shift   : {shift:.3f} ckpc/h = "
              f"{shift * hdr['a'] / hdr['h']:.3f} pkpc "
              f"({shift/cat_r200:.4f} catalog R200)")
        print(f"refinement accepted: {accepted}")
        print(f"chosen center      : {chosen}")
        print()
        print(f"catalog M200c      : {cat_m200*1e10/hdr['h']:.6e} Msun")
        print(f"particle SO M200c  : {so_m200*1e10/hdr['h']:.6e} Msun")
        print(f"catalog R200c      : {cat_r200*hdr['a']/hdr['h']:.3f} pkpc")
        print(f"particle SO R200c  : {so_r200*hdr['a']/hdr['h']:.3f} pkpc")
        print("=" * 72)

    return ref
