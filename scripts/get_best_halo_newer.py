#!/usr/bin/env python3
"""
Select halos by mass + isolation from (one or many) SUBFIND/FOF HDF5 catalogs,
print a ranked table (now including halo_id), and optionally save a CSV.

Isolation = distance to the nearest MORE MASSIVE halo (periodic boundary).

Examples
--------
python get_best_halo.py catalogs/groups_*.hdf5 \
  --mmin 5e11 --mmax 2e12 --nmax 20 --csv best_halos.csv

# Just print the chosen halo IDs (one per line)
python get_best_halo.py catalogs/groups_*.hdf5 --print-ids-only --nmax 10
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


# ------------------------------ HDF5 helpers ------------------------------ #

def _find_first_dataset(grp: h5py.Group, candidates: List[str]) -> Optional[np.ndarray]:
    """Return the first existing dataset in candidates (as ndarray) or None."""
    for key in candidates:
        if key in grp:
            return np.array(grp.get(key))
    return None


def _read_header_attrs(f: h5py.File) -> Tuple[float, float]:
    """Return (BoxSize [Mpc/h], Redshift). Falls back to 0 if missing."""
    box = 0.0
    z = 0.0
    if "Header" in f:
        hdr = f["Header"]
        if "BoxSize" in hdr.attrs:
            # Many catalogs store BoxSize in comoving Mpc/h
            box = float(np.atleast_1d(hdr.attrs["BoxSize"])[0])
        if "Redshift" in hdr.attrs:
            z = float(np.atleast_1d(hdr.attrs["Redshift"])[0])
    return box, z


def _read_one_catalog(fn: str) -> Dict[str, np.ndarray]:
    """
    Read one HDF5 group catalog and return dict with positions, masses, R200,
    halo IDs (if present), and header info.
    """
    with h5py.File(fn, "r") as f:
        box, redshift = _read_header_attrs(f)

        # Support both "Group/*" and top-level variants
        if "Group" in f:
            grp = f["Group"]
        else:
            grp = f  # some catalogs flatten datasets at file root

        # Positions (comoving, Mpc/h typically)
        pos = _find_first_dataset(grp, [
            "GroupPos", "GroupCM", "GroupPos_Mean",
        ])
        if pos is None:
            raise RuntimeError(f"{fn}: could not find any GroupPos-like dataset")
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise RuntimeError(f"{fn}: GroupPos-like dataset has wrong shape {pos.shape}")

        # Masses (various conventions)
        # Preferred: M200c, M200m, otherwise GroupMass.
        mass = _find_first_dataset(grp, [
            "Group_M_Crit200", "Group_M_TopHat200", "GroupMass",
        ])
        if mass is None:
            raise RuntimeError(f"{fn}: could not find any GroupMass-like dataset")
        mass = np.asarray(mass, dtype=np.float64).reshape(-1)

        # R200 if available (kpc/h typically). Optional.
        r200 = _find_first_dataset(grp, [
            "Group_R_Crit200", "Group_R_TopHat200", "Group_R_Mean200",
        ])
        if r200 is not None:
            r200 = np.asarray(r200, dtype=np.float64).reshape(-1)
        else:
            r200 = np.full(mass.shape, np.nan, dtype=np.float64)

        # Group IDs (optional)
        gid = _find_first_dataset(grp, [
            "GroupID", "GroupIDs", "GroupNumber", "GroupNr", "GroupIndex",
        ])
        if gid is not None:
            gid = np.asarray(gid, dtype=np.int64).reshape(-1)

    return dict(
        filename=np.array([fn]),  # keep file name (not used downstream, but handy)
        pos=pos,
        mass=mass,
        r200_kpc=r200,     # units as in file (often kpc/h)
        box_Mpch=np.array([box], dtype=np.float64),
        z=np.array([redshift], dtype=np.float64),
        ids=gid if gid is not None else None,
    )


def _stack_catalogs(files: List[str]) -> Dict[str, np.ndarray]:
    """Stack multiple catalogs; create unique fallback IDs if none given."""
    all_pos, all_mass, all_r200, all_ids = [], [], [], []
    box_Mpch = None
    z = None
    running = 0  # for global stable ID fallback

    for fn in files:
        d = _read_one_catalog(fn)

        if box_Mpch is None:
            box_Mpch = float(d["box_Mpch"][0])
            z = float(d["z"][0])
        else:
            # Warn if box size differs (we proceed anyway)
            if abs(float(d["box_Mpch"][0]) - box_Mpch) > 1e-6:
                print(f"WARNING: {fn} has BoxSize={float(d['box_Mpch'][0])} (Mpc/h), "
                      f"differs from first={box_Mpch} (Mpc/h)")

        pos = d["pos"]
        mass = d["mass"]
        r200 = d["r200_kpc"]
        N = mass.shape[0]

        if d["ids"] is not None and d["ids"].shape[0] == N:
            ids = d["ids"].astype(np.int64, copy=False)
        else:
            # Stable global row index across all input files
            ids = (np.arange(N, dtype=np.int64) + running)
        running += N

        all_pos.append(pos)
        all_mass.append(mass)
        all_r200.append(r200)
        all_ids.append(ids)

    if not all_mass:
        raise RuntimeError("No groups found in provided files.")

    pos = np.vstack(all_pos)
    mass = np.hstack(all_mass)
    r200 = np.hstack(all_r200)
    ids = np.hstack(all_ids)

    return dict(pos=pos, mass=mass, r200_kpc=r200, ids=ids,
                box_Mpch=np.array([box_Mpch]), z=np.array([z]))


# ------------------------------ Core logic ------------------------------ #

def _periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    """Return minimum-image displacement vector a-b (periodic with side 'box')."""
    d = a - b
    d -= np.round(d / box) * box
    return d


def _nearest_more_massive_distance(pos: np.ndarray,
                                   mass: np.ndarray,
                                   box: float) -> np.ndarray:
    """
    For each halo i, compute distance (Mpc/h) to nearest halo j with mass[j] > mass[i].
    Returns np.inf if none more massive exists.
    """
    N = pos.shape[0]
    dmin = np.full(N, np.inf, dtype=np.float64)

    # Brute force is OK for a few ×1e4. If you anticipate more, consider cKDTree tiling.
    for i in range(N):
        # Mask: strictly more massive
        mask = mass > mass[i]
        if not np.any(mask):
            continue
        dp = _periodic_delta(pos[mask], pos[i], box)
        dist = np.sqrt(np.einsum("ij,ij->i", dp, dp))
        if dist.size:
            dmin[i] = np.min(dist)

    return dmin


def select_and_rank(pos_Mpch: np.ndarray,
                    mass: np.ndarray,
                    r200_kpc: np.ndarray,
                    ids: np.ndarray,
                    box_Mpch: float,
                    mmin: float,
                    mmax: float,
                    nmax: int) -> List[Dict[str, float]]:
    """
    Apply mass cut, compute isolation, and return top 'nmax' halos with fields:
    rank, halo_id, mass, r200_kpc, iso_dist_Mpch, x,y,z (Mpc/h).
    """
    sel = (mass >= mmin) & (mass <= mmax)
    if not np.any(sel):
        return []

    pos = pos_Mpch[sel]
    mass_sel = mass[sel]
    r200_sel = r200_kpc[sel]
    ids_sel = ids[sel]

    # Isolation = distance to nearest MORE MASSIVE halo
    iso = _nearest_more_massive_distance(pos, mass_sel, box_Mpch)

    # Rank by isolation DESC (most isolated first)
    order = np.argsort(-iso)
    if nmax > 0:
        order = order[:nmax]

    rows = []
    for rank, ii in enumerate(order, start=1):
        rows.append(dict(
            rank=rank,
            halo_id=int(ids_sel[ii]),
            mass=float(mass_sel[ii]),
            r200_kpc=float(r200_sel[ii]) if math.isfinite(r200_sel[ii]) else float("nan"),
            iso_dist_Mpch=float(iso[ii]) if math.isfinite(iso[ii]) else float("inf"),
            x_Mpch=float(pos[ii, 0]),
            y_Mpch=float(pos[ii, 1]),
            z_Mpch=float(pos[ii, 2]),
        ))
    return rows


# ------------------------------ CLI / I/O ------------------------------ #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pick halos by mass and isolation; print a ranked table including halo_id."
    )
    ap.add_argument("files", nargs="+",
                    help="HDF5 group catalogs (paths or globs).")
    ap.add_argument("--mmin", type=float, required=True,
                    help="Minimum halo mass (catalog units).")
    ap.add_argument("--mmax", type=float, required=True,
                    help="Maximum halo mass (catalog units).")
    ap.add_argument("--nmax", type=int, default=20,
                    help="How many halos to print/save (default: 20).")
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional path to write a CSV of the selected halos.")
    ap.add_argument("--print-ids-only", action="store_true",
                    help="Only print selected halo IDs (one per line).")
    return ap.parse_args()


def _expand_globs(files: List[str]) -> List[str]:
    out = []
    for pat in files:
        hits = glob.glob(pat)
        if hits:
            out.extend(sorted(hits))
        else:
            # Keep literal path if no glob expansion
            out.append(pat)
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def _print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No halos matched the selection.")
        return

    print("\nTop halos (ranked by isolation):")
    print(f"{'rk':>2}  {'halo_id':>12}  {'Mass':>12}  {'R200[Mpc/h]':>10}  {'iso[Mpc/h]':>11}   {'x,y,z [Mpc/h]':>29}")
    for r in rows:
        R = f"{r['r200_kpc']:.3f}" if math.isfinite(r['r200_kpc']) else "—"
        xyz = f"({r['x_Mpch']:.3f}, {r['y_Mpch']:.3f}, {r['z_Mpch']:.3f})"
        print(f"{r['rank']:2d}  {r['halo_id']:12d}  {r['mass']:12.3e}  {R:>10}  {r['iso_dist_Mpch']:11.3f}   {xyz:>29}")


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    cols = ["rank", "halo_id", "mass", "r200_kpc", "iso_dist_Mpch", "x_Mpch", "y_Mpch", "z_Mpch"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"\nWrote CSV: {path}")


def main() -> None:
    args = _parse_args()
    files = _expand_globs(args.files)
    if not files:
        raise SystemExit("No input files found.")

    data = _stack_catalogs(files)
    pos = data["pos"]                  # (N,3), Mpc/h (as catalog)
    mass = data["mass"]                # (N,), catalog units
    r200 = data["r200_kpc"]            # (N,), kpc/h typically
    ids = data["ids"]                  # (N,), int64
    box = float(data["box_Mpch"][0])   # Mpc/h box (0 if unknown)
    z = float(data["z"][0])

    if box <= 0:
        # If box is unknown, we cannot do periodic isolation; we’ll warn and use non-periodic distance.
        print("WARNING: BoxSize missing/invalid; distances will not be minimum-image periodic.")
        # In that (rare) case, treat 'box' as np.inf so no wrapping occurs
        box = float("inf")

    rows = select_and_rank(pos, mass, r200, ids, box, args.mmin, args.mmax, args.nmax)

    if args.print_ids_only:
        for r in rows:
            print(r["halo_id"])
        return

    _print_table(rows)

    if args.csv:
        _write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
