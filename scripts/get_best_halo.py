#!/usr/bin/env python3
"""
Select & rank "best" FoF halos by isolation in a cosmological box.

- Positions -> Mpc, masses -> Msun (auto-detected; robust to Mpc or kpc/h inputs)
- Isolation metric: distance (Mpc) to the nearest neighbor with mass >= iso_frac * M_target
- Periodic minimum-image distances
- Multi-file support (e.g., fof_subhalo_tab_*.hdf5)

Examples
--------
# Milky Way–like halos, very isolated (>= 3 Mpc from equal/heavier neighbors)
python get_best_halo.py fof_subhalo_tab_*.hdf5 \
  --mmin 5e11 --mmax 2e12 --isolation 3.0 --iso-frac 1.0 --top 20 -o best_halos.csv

# Neighbors counted if >= 0.5*M, slightly looser isolation
python get_best_halo.py fof_*.hdf5 --mmin 8e11 --mmax 2e12 --isolation 2.0 --iso-frac 0.5
"""

import argparse
import glob
import h5py
import numpy as np
import os
import sys
import csv
from typing import Dict

MSUN_IN_G  = 1.98847e33
CM_PER_KPC = 3.085678e21
CM_PER_MPC = 3.085678e24


# -------------------------- I/O & Unit Handling -------------------------- #

def _read_one_catalog(fn: str) -> Dict[str, np.ndarray]:
    """Read one FoF HDF5 catalog and return arrays in Mpc (pos/box) and Msun (mass)."""
    with h5py.File(fn, "r") as f:
        hdr = f["Header"].attrs

        # Units — default to Mpc & 1e10 Msun if absent (common in cosmology HDF5)
        unit_len_cm   = float(hdr.get("UnitLength_in_cm", CM_PER_MPC))
        unit_mass_g   = float(hdr.get("UnitMass_in_g", 1.989e43))  # 1e10 Msun if code uses that
        redshift      = float(hdr.get("Redshift", 0.0))
        box_size_code = float(hdr.get("BoxSize"))

        grp = f.get("Group")
        if grp is None:
            raise RuntimeError(f"{fn}: no /Group group found")

        pos  = np.array(grp.get("GroupPos"))
        mass = np.array(grp.get("GroupMass"))
        if pos is None or mass is None:
            raise RuntimeError(f"{fn}: missing GroupPos or GroupMass datasets")

        # Optional R200 (often kpc/h; handle heuristically)
        r200 = grp.get("Group_R_Crit200")
        if r200 is not None:
            r200 = np.array(r200)
        else:
            r200 = np.full(mass.shape[0], np.nan, dtype=np.float64)

        # ---- Convert to desired physical units: Mpc (pos/box) and Msun (mass) ----
        to_Mpc = unit_len_cm / CM_PER_MPC
        box_Mpc = box_size_code * to_Mpc
        pos_Mpc = pos * to_Mpc

        # Mass: many FoF outputs store GroupMass in 1e10 Msun units.
        mmax = float(np.nanmax(mass)) if mass.size else 0.0
        if mmax < 1e8:
            mass_Msun = mass * 1.0e10
        else:
            mass_Msun = mass * (unit_mass_g / MSUN_IN_G)

        # Heuristic for R200 to kpc
        if np.isfinite(r200).any():
            med = float(np.nanmedian(r200))
            if med < 5.0:            # looks like Mpc
                r200_kpc = r200 * 1000.0
            elif med > 5e4:          # could be code units -> convert with header length
                r200_kpc = r200 * (unit_len_cm / CM_PER_KPC)
            else:                     # assume already kpc
                r200_kpc = r200
        else:
            r200_kpc = r200

        return dict(
            filename=fn,
            pos=pos_Mpc,
            mass=mass_Msun,
            r200_kpc=r200_kpc,
            box_Mpc=box_Mpc,
            z=redshift
        )


def _stack_catalogs(files) -> Dict[str, np.ndarray]:
    """Concatenate multiple catalogs (same snapshot) into single arrays."""
    all_pos, all_mass, all_r200 = [], [], []
    box_Mpc = None
    z = None

    for fn in files:
        d = _read_one_catalog(fn)
        if box_Mpc is None:
            box_Mpc = d["box_Mpc"]
            z = d["z"]
        else:
            if abs(d["box_Mpc"] - box_Mpc) > 1e-6:
                print(f"WARNING: {fn} has BoxSize={d['box_Mpc']} Mpc, differs from first={box_Mpc} Mpc")
        all_pos.append(d["pos"])
        all_mass.append(d["mass"])
        all_r200.append(d["r200_kpc"])

    pos  = np.vstack(all_pos)  if all_pos else np.empty((0,3))
    mass = np.hstack(all_mass) if all_mass else np.empty((0,))
    r200 = np.hstack(all_r200) if all_r200 else np.empty((0,))
    return dict(pos=pos, mass=mass, r200_kpc=r200, box_Mpc=box_Mpc, z=z)


# ----------------------------- Geometry utils ---------------------------- #

def _pbc_delta(a, b, box):
    """Minimum-image separation vector with periodic BCs (Mpc). a,b: (...,3)"""
    d = a - b
    d -= np.round(d / box) * box
    return d


# -------------------------- Isolation computations ----------------------- #

def _nearest_heavy_neighbor_for_candidates(pos_all, mass_all, pos_sel, mass_sel, box, iso_frac):
    """
    For each candidate halo (pos_sel, mass_sel), return the distance [Mpc]
    to the nearest halo in (pos_all, mass_all) with mass >= iso_frac * M_candidate.
    Uses periodic minimum-image distances.
    """
    N_sel = pos_sel.shape[0]
    out = np.full(N_sel, np.inf, dtype=np.float64)

    for i in range(N_sel):
        thr = iso_frac * mass_sel[i]
        dvec = _pbc_delta(pos_all, pos_sel[i], box)
        rr = np.sqrt(np.sum(dvec**2, axis=1))
        m = (mass_all >= thr) & (rr > 0)          # exclude self (rr==0)
        if np.any(m):
            out[i] = rr[m].min()
    return out


# --------------------------------- Main ---------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Rank FoF halos by isolation (Mpc/Msun units).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("catalogs", nargs="+", help="FoF HDF5 files (accepts globs)")
    ap.add_argument("--mmin", type=float, required=True, help="Minimum halo mass [Msun]")
    ap.add_argument("--mmax", type=float, required=True, help="Maximum halo mass [Msun]")
    ap.add_argument("--iso-frac", type=float, default=1.0,
                    help="Neighbor mass threshold as a fraction of candidate mass (>= f * M)")
    ap.add_argument("--isolation", type=float, default=3.0,
                    help="Minimum isolation distance [Mpc] to accept")
    ap.add_argument("--top", type=int, default=20, help="Number of best halos to print/save")
    ap.add_argument("-o","--output", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    # Expand globs
    files = []
    for pat in args.catalogs:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"WARNING: no files matched {pat}")
        files += hits
    if not files:
        print("ERROR: no input files found", file=sys.stderr)
        sys.exit(1)

    data = _stack_catalogs(files)
    pos, mass, r200_kpc, box, z = data["pos"], data["mass"], data["r200_kpc"], data["box_Mpc"], data["z"]

    if pos.size == 0:
        print("No halos found.")
        sys.exit(0)

    print(f"Loaded {len(mass)} FoF halos | z={z:.3f} | Box = {box:.3f} Mpc")
    print(f"Mass range (Msun): min={mass.min():.3e}, max={mass.max():.3e}")

    # Mass cut
    sel = (mass >= args.mmin) & (mass <= args.mmax)
    if not np.any(sel):
        print("No halos in the requested mass range.")
        sys.exit(0)

    pos_sel  = pos[sel]
    mass_sel = mass[sel]
    r200_sel = r200_kpc[sel]

    # Distance (Mpc) to nearest neighbor >= iso_frac * M for each candidate
    d_iso = _nearest_heavy_neighbor_for_candidates(
        pos_all=pos, mass_all=mass,
        pos_sel=pos_sel, mass_sel=mass_sel,
        box=box, iso_frac=args.iso_frac
    )

    # Keep only those meeting the isolation distance
    iso_ok = d_iso >= args.isolation
    if not np.any(iso_ok):
        print("No halos satisfy the isolation criterion.")
        sys.exit(0)

    pos_ok  = pos_sel[iso_ok]
    mass_ok = mass_sel[iso_ok]
    r200_ok = r200_sel[iso_ok]
    d_ok    = d_iso[iso_ok]

    # Rank by isolation distance descending; tie-breaker: closeness to mid-mass
    mid_mass = 0.5*(args.mmin + args.mmax)
    score = np.vstack([-d_ok, np.abs(np.log(mass_ok/mid_mass))]).T
    order = np.lexsort((score[:,1], score[:,0]))

    pick = order[:args.top]
    rows = []
    for i, r in enumerate(pick, start=1):
        rows.append(dict(
            rank=i,
            mass_Msun=float(mass_ok[r]),
            r200_kpc=float(r200_ok[r]) if np.isfinite(r200_ok[r]) else np.nan,
            iso_dist_Mpc=float(d_ok[r]),
            x_Mpc=float(pos_ok[r,0]),
            y_Mpc=float(pos_ok[r,1]),
            z_Mpc=float(pos_ok[r,2])
        ))

    # Print table
    print("\nTop halos (ranked by isolation):")
    print(f"{'rk':>2}  {'M [Msun]':>12}  {'R200 [kpc]':>10}  {'iso [Mpc]':>9}   {'x,y,z [Mpc]':>25}")
    for r in rows:
        xyz = f"({r['x_Mpc']:.3f}, {r['y_Mpc']:.3f}, {r['z_Mpc']:.3f})"
        R = f"{r['r200_kpc']:.1f}" if np.isfinite(r['r200_kpc']) else "—"
        print(f"{r['rank']:2d}  {r['mass_Msun']:12.3e}  {R:>10}  {r['iso_dist_Mpc']:9.3f}   {xyz:>25}")

    # CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
