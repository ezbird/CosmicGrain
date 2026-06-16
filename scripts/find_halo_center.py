#!/usr/bin/env python3
"""
find_halo569_center.py
======================
Computes the density-weighted shrinking-sphere center of Halo 569
from the LAST snapshot of a given run.  Prints the result in ckpc/h
ready to paste into HALO569_CENTERS_CKPC_H in make_zoom_movie.py.

Run this ONCE per resolution tier on a post-fix snapshot (z=0 preferred).

Usage
-----
  python find_halo569_center.py --snapdir ../S10_output_2048 --resolution 2048
  python find_halo569_center.py --snapdir ../S10_output_1024 --resolution 1024

The script:
  1. Finds the last available snapshot
  2. Loads ALL gas particle positions + densities
  3. Seeds a shrinking sphere near the old hardcoded center (or box center)
  4. Iterates until convergence
  5. Prints the new center in ckpc/h
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Old centers as initial seeds (ckpc/h) — used as starting guess only
SEED_CENTERS = {
    512:  np.array([23052.975, 23205.770, 23703.861]),
    1024: np.array([23048.920, 23163.650, 23699.611]),
    2048: np.array([23085.406, 23512.129, 23653.939]),
}

def find_last_snap(snapdir):
    """Return snap_entry for the highest-numbered snapshot."""
    snaps = {}
    for pat in ["snap_???.hdf5", "snap_????.hdf5"]:
        for p in sorted(glob.glob(os.path.join(snapdir, pat))):
            n = int(Path(p).stem.split("_")[-1])
            snaps[n] = ("single", p)
    for pat in ["snap_???", "snap_????"]:
        for d in sorted(glob.glob(os.path.join(snapdir, pat))):
            if os.path.isdir(d):
                pieces = sorted(glob.glob(os.path.join(d, "*.hdf5")))
                if pieces:
                    n = int(Path(d).name.split("_")[-1])
                    snaps[n] = ("multi", pieces)
    # Gadget-4 default layout
    if not snaps:
        for d in sorted(glob.glob(os.path.join(snapdir, "snapdir_???"))):
            if os.path.isdir(d):
                n = int(Path(d).name.split("_")[-1])
                pieces = sorted(glob.glob(os.path.join(d, "snapshot_???.*.hdf5")))
                if not pieces:
                    pieces = sorted(glob.glob(os.path.join(d, "snapshot_???.hdf5")))
                if pieces:
                    snaps[n] = ("multi", pieces) if len(pieces) > 1 else ("single", pieces[0])
    if not snaps:
        sys.exit(f"No snapshots found in {snapdir}")
    last_num = max(snaps)
    print(f"Using snapshot {last_num} (highest available)")
    return last_num, snaps[last_num]

def read_gas(snap_entry):
    """Read gas Coordinates and Density across all chunks."""
    kind, path = snap_entry
    files = path if kind == "multi" else [path]
    pos_chunks, rho_chunks = [], []
    hdr, params = None, None
    for fname in files:
        with h5py.File(fname, "r") as f:
            if hdr is None:
                hdr = dict(f["Header"].attrs)
                params = dict(f["Parameters"].attrs) if "Parameters" in f else {}
            if "PartType0" not in f:
                continue
            pos_chunks.append(f["PartType0"]["Coordinates"][:].astype(np.float64))
            rho_chunks.append(f["PartType0"]["Density"][:].astype(np.float64))
    pos = np.concatenate(pos_chunks)
    rho = np.concatenate(rho_chunks)
    h = float(params.get("HubbleParam", hdr.get("HubbleParam", 0.6774)))
    a = float(hdr["Time"])
    boxsize = float(hdr["BoxSize"])
    return pos, rho, h, a, boxsize

def shrinking_sphere(pos, weights, seed_ckpc_h, r_init=300.0,
                     r_min=5.0, shrink=0.95, n_min=50):
    """
    Density-weighted shrinking sphere.
    pos, seed in ckpc/h (comoving).  r_init, r_min in ckpc/h.
    """
    cen = seed_ckpc_h.copy()
    r = r_init
    iteration = 0
    while r > r_min:
        dx = pos - cen
        dist = np.sqrt((dx**2).sum(axis=1))
        mask = dist < r
        n_in = mask.sum()
        if n_in < n_min:
            print(f"  iter {iteration:3d}: r={r:.1f} ckpc/h  n={n_in} — too few particles, stopping")
            break
        w = weights[mask]
        cen = np.average(pos[mask], weights=w, axis=0)
        print(f"  iter {iteration:3d}: r={r:7.1f} ckpc/h  n={n_in:8,}  cen={cen}")
        r *= shrink
        iteration += 1
    return cen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapdir",   required=True)
    p.add_argument("--resolution",type=int, default=1024,
                   help="512 / 1024 / 2048 — used to pick the seed center")
    p.add_argument("--r_init",    type=float, default=300.0,
                   help="Initial sphere radius in ckpc/h (default 300)")
    p.add_argument("--r_min",     type=float, default=8.0,
                   help="Stop shrinking below this radius in ckpc/h (default 8)")
    p.add_argument("--snap",      type=int, default=None,
                   help="Specific snapshot number (default: last available)")
    args = p.parse_args()

    if args.snap is not None:
        # find that specific snap
        snap_num, snap_entry = args.snap, None
        snapdir = args.snapdir
        for d in [f"snapdir_{args.snap:03d}", f"snapdir_{args.snap:04d}"]:
            full = os.path.join(snapdir, d)
            if os.path.isdir(full):
                pieces = sorted(glob.glob(os.path.join(full, "*.hdf5")))
                if pieces:
                    snap_entry = ("multi", pieces)
                    break
        if snap_entry is None:
            sys.exit(f"Snapshot {args.snap} not found")
    else:
        snap_num, snap_entry = find_last_snap(args.snapdir)

    print(f"\nReading gas particles...")
    pos, rho, h, a, boxsize = read_gas(snap_entry)
    print(f"  {len(pos):,} gas particles  |  h={h}  a={a:.4f}  z={1/a-1:.3f}")
    print(f"  Box: {boxsize:.1f} ckpc/h = {boxsize/h:.1f} ckpc = {boxsize/h/1000:.2f} Mpc")

    # Seed
    if args.resolution in SEED_CENTERS:
        seed = SEED_CENTERS[args.resolution]
        print(f"\nSeed center ({args.resolution}³): {seed} ckpc/h")
    else:
        seed = np.array([boxsize/2, boxsize/2, boxsize/2])
        print(f"\nNo seed for resolution {args.resolution} — using box center: {seed}")

    print(f"\nRunning shrinking sphere (r_init={args.r_init}, r_min={args.r_min} ckpc/h)...")
    center = shrinking_sphere(pos, rho, seed,
                              r_init=args.r_init, r_min=args.r_min)

    center_pkpc = center * a / h
    print(f"\n{'='*60}")
    print(f"RESULT for {args.resolution}³ at snap {snap_num} (z={1/a-1:.3f})")
    print(f"  Center (ckpc/h) : [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Center (pkpc)   : [{center_pkpc[0]:.2f}, {center_pkpc[1]:.2f}, {center_pkpc[2]:.2f}]")
    print(f"\nPaste into make_zoom_movie.py HALO569_CENTERS_CKPC_H:")
    print(f"    {args.resolution}: np.array([{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]),")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
