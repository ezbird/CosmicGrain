#!/usr/bin/env python3
"""
snap_inventory.py  —  Quick survey of all snapshots in a Gadget-4 run
======================================================================
Prints a table:  snap# | z | t_Gyr | N_DM | N_gas | N_star | N_dust
Useful for picking key epochs before rendering the full movie.

Usage
-----
  python snap_inventory.py --snapdir /scratch/cygnus/CosmicGrain/output_s5_1024
  python snap_inventory.py --snapdir ... --snapbase snap --csv inventory.csv
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from scipy.integrate import quad
    def a_to_Gyr(a, H0=67.74, Om=0.3089, Ol=0.6911):
        def integrand(z):
            E = np.sqrt(Om*(1+z)**3 + Ol)
            return 1.0 / ((1+z)*E)
        z = 1.0/a - 1.0
        t, _ = quad(integrand, z, np.inf)
        return t * 977.8 / H0
except ImportError:
    def a_to_Gyr(a, **kw):
        return float("nan")


def find_all(snapdir, snapbase="snap"):
    snaps = {}
    for pat in [f"{snapbase}_???.hdf5", f"{snapbase}_????.hdf5"]:
        for p in sorted(glob.glob(os.path.join(snapdir, pat))):
            n = int(Path(p).stem.split("_")[-1])
            snaps[n] = ("single", p)
    for pat in [f"{snapbase}_???", f"{snapbase}_????"]:
        for d in sorted(glob.glob(os.path.join(snapdir, pat))):
            if os.path.isdir(d):
                pieces = sorted(glob.glob(os.path.join(d, "*.hdf5")))
                if pieces:
                    n = int(Path(d).name.split("_")[-1])
                    snaps[n] = ("multi", pieces)
    return sorted(snaps.items())


def count_part(snap_entry, ptype):
    kind, path = snap_entry
    fname = path[0] if kind == "multi" else path
    try:
        with h5py.File(fname, "r") as f:
            hdr = dict(f["Header"].attrs)
            np_arr = list(hdr.get("NumPart_Total", [0]*8))
            if ptype < len(np_arr):
                n = int(np_arr[ptype])
                # for multi-file add NumPart_Total_HighWord * 2^32
                hw = list(hdr.get("NumPart_Total_HighWord", [0]*8))
                if ptype < len(hw):
                    n += int(hw[ptype]) * (2**32)
                return n
    except Exception:
        pass
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapdir",  required=True)
    p.add_argument("--snapbase", default="snap")
    p.add_argument("--csv",      default=None, help="Also write CSV to this path")
    args = p.parse_args()

    snaps = find_all(args.snapdir, args.snapbase)
    if not snaps:
        sys.exit(f"No snapshots found in {args.snapdir}")

    header = ["snap", "z", "t_Gyr", "N_DM", "N_gas", "N_star", "N_dust", "path"]
    rows = []

    print(f"\n{'snap':>5}  {'z':>8}  {'t_Gyr':>8}  "
          f"{'N_DM':>10}  {'N_gas':>10}  {'N_star':>10}  {'N_dust':>10}")
    print("-" * 75)

    for snap_num, snap_entry in snaps:
        kind, path = snap_entry
        fname = path[0] if kind == "multi" else path
        try:
            with h5py.File(fname, "r") as f:
                hdr = dict(f["Header"].attrs)
        except Exception as e:
            print(f"  [{snap_num:4d}]  ERROR reading header: {e}")
            continue

        a     = float(hdr["Time"])
        z     = 1.0/a - 1.0
        t_Gyr = a_to_Gyr(a)

        n_dm   = count_part(snap_entry, 1)
        n_gas  = count_part(snap_entry, 0)
        n_star = count_part(snap_entry, 4)
        n_dust = count_part(snap_entry, 6)

        print(f"  {snap_num:4d}  {z:8.3f}  {t_Gyr:8.3f}  "
              f"{n_dm:10,}  {n_gas:10,}  {n_star:10,}  {n_dust:10,}")

        rows.append([snap_num, f"{z:.4f}", f"{t_Gyr:.4f}",
                     n_dm, n_gas, n_star, n_dust,
                     fname])

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        print(f"\nCSV written to {args.csv}")

    print(f"\nTotal: {len(rows)} snapshots")


if __name__ == "__main__":
    main()
