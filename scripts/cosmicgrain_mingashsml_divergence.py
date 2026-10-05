#!/usr/bin/env python3
"""Find the first differing saved snapshot in CG10's four 512^3 floor runs.

Run from the gadget4 directory. Requires numpy and h5py. Snapshot comparisons
are exact, field by field and piece by piece; this does not compare integration
steps between saved snapshots. Do not compare HDF5 file hashes: output paths and
metadata differ by construction.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


PAIRS = (("1.0", "0.25"),)
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
    if value == "1.0":
        root = Path("halo441_output_1024")
    else:
        root = Path("halo441_output_1024_mingashsml0.25")

    return (root / f"snapdir_{snap:03d}" /
            f"snapshot_{snap:03d}.{piece}.hdf5")


def details(left, right):
    if left.shape != right.shape:
        return f"shape {left.shape} vs {right.shape}"
    if np.array_equal(left, right, equal_nan=True):
        return None
    if left.dtype.kind in "f" and right.dtype.kind in "f":
        same = (left == right) | (np.isnan(left) & np.isnan(right))
    else:
        same = left == right
    bad = int(np.size(same) - np.count_nonzero(same))
    return f"{bad}/{left.size} elements differ; first index {tuple(np.argwhere(~same)[0])}"


def compare(value_a, value_b, snap, pieces):
    differences = []
    for piece in range(pieces):
        pa, pb = path(value_a, snap, piece), path(value_b, snap, piece)
        if not pa.exists() or not pb.exists():
            raise FileNotFoundError(f"Missing snapshot piece: {pa} or {pb}")
        with h5py.File(pa) as fa, h5py.File(pb) as fb:
            if not np.isclose(fa["Header"].attrs["Time"],
                              fb["Header"].attrs["Time"], atol=1e-12):
                differences.append(f"piece {piece}: snapshot times differ")
            for group, fields in FIELDS.items():
                if (group in fa) != (group in fb):
                    differences.append(f"piece {piece}: {group} present in one run only")
                    continue
                if group not in fa:
                    continue
                for field in fields:
                    if (field in fa[group]) != (field in fb[group]):
                        differences.append(f"piece {piece}: {group}/{field} missing in one run")
                        continue
                    if field not in fa[group]:
                        continue
                    issue = details(fa[group][field][:], fb[group][field][:])
                    if issue:
                        differences.append(f"piece {piece}: {group}/{field}: {issue}")
    return differences


def gas_stats(value, snap, pieces):
    hsml, density, sfr = [], [], []
    for piece in range(pieces):
        with h5py.File(path(value, snap, piece)) as f:
            if "PartType0" not in f:
                continue
            gas = f["PartType0"]
            hsml.append(gas["SmoothingLength"][:])
            density.append(gas["Density"][:])
            sfr.append(gas["StarFormationRate"][:])
    if not hsml:
        return "no gas particles"
    h = np.concatenate(hsml)
    d = np.concatenate(density)
    s = np.concatenate(sfr)
    return (f"N={len(h)}, h_min={h.min():.9g}, h_p01={np.percentile(h, 1):.9g}, "
            f"rho_p99={np.percentile(d, 99):.9g}, SF_N={np.count_nonzero(s > 0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=29)
    parser.add_argument("--pieces", type=int, default=4)
    args = parser.parse_args()
    print("Exact comparisons of selected scientific fields; matching piece/order assumed.")
    for value_a, value_b in PAIRS:
        print(f"\n=== {value_a} vs {value_b} ===", flush=True)
        first_difference = None
        for snap in range(args.first, args.last + 1):
            differences = compare(value_a, value_b, snap, args.pieces)
            if differences:
                first_difference = snap
                print(f"First differing saved snapshot: {snap:03d}")
                for item in differences[:16]:
                    print("  " + item)
                if len(differences) > 16:
                    print(f"  ... and {len(differences) - 16} more differing fields/pieces")
                print(f"  {value_a}: {gas_stats(value_a, snap, args.pieces)}")
                print(f"  {value_b}: {gas_stats(value_b, snap, args.pieces)}")
                break
            print(f"  snapshot {snap:03d}: identical", flush=True)
        if first_difference is None:
            print(f"No differences detected in selected fields, snapshots "
                  f"{args.first:03d}–{args.last:03d}.")


if __name__ == "__main__":
    main()
