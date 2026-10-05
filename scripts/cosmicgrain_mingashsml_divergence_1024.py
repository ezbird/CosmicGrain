#!/usr/bin/env python3
"""Find the first differing snapshot among CG10 1024^3 MinGasHsml runs."""

import argparse
from pathlib import Path

import h5py
import numpy as np


ROOTS = {
    "1.0": Path("halo441_output_1024"),
    "0.25": Path("halo441_output_1024_mingashsml0.25"),
    "0.0": Path("halo441_output_1024_mingashsml0.0"),
}
PAIRS = (("0.25", "0.0"), ("1.0", "0.25"))
FIELDS = {
    "PartType0": ("ParticleIDs", "Coordinates", "Velocities", "Masses",
                  "Density", "SmoothingLength", "InternalEnergy",
                  "StarFormationRate", "Metallicity"),
    "PartType4": ("ParticleIDs", "Coordinates", "Velocities", "Masses",
                  "StellarFormationTime", "Metallicity"),
    "PartType6": ("ParticleIDs", "Coordinates", "Velocities", "Masses",
                  "GrainRadius", "CarbonMassFraction", "DustSource"),
}


def path(value, snap, piece):
    return (ROOTS[value] / f"snapdir_{snap:03d}" /
            f"snapshot_{snap:03d}.{piece}.hdf5")


def issue(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if np.array_equal(a, b, equal_nan=True):
        return None
    same = ((a == b) | (np.isnan(a) & np.isnan(b))) if a.dtype.kind == "f" else (a == b)
    return f"{a.size - np.count_nonzero(same)}/{a.size} elements differ"


def compare(left, right, snap, pieces):
    differences = []
    for piece in range(pieces):
        with h5py.File(path(left, snap, piece)) as a, h5py.File(path(right, snap, piece)) as b:
            for group, fields in FIELDS.items():
                if (group in a) != (group in b):
                    differences.append(f"piece {piece}: {group} present in one run only")
                    continue
                if group not in a:
                    continue
                for field in fields:
                    if field not in a[group] or field not in b[group]:
                        continue
                    found = issue(a[group][field][:], b[group][field][:])
                    if found:
                        differences.append(f"piece {piece}: {group}/{field}: {found}")
    return differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=29)
    p.add_argument("--pieces", type=int, default=4)
    args = p.parse_args()

    for left, right in PAIRS:
        print(f"\n=== {left} vs {right} ===", flush=True)
        for snap in range(args.first, args.last + 1):
            differences = compare(left, right, snap, args.pieces)
            if differences:
                with h5py.File(path(left, snap, 0)) as f:
                    a = float(f["Header"].attrs["Time"])
                print(f"First differing snapshot: {snap:03d}; a={a:.6f}, z={1/a-1:.3f}")
                for item in differences[:12]:
                    print("  " + item)
                if len(differences) > 12:
                    print(f"  ... and {len(differences)-12} more")
                break
            print(f"  snapshot {snap:03d}: identical", flush=True)
        else:
            print(f"No differences in selected fields through snapshot {args.last:03d}.")


if __name__ == "__main__":
    main()
