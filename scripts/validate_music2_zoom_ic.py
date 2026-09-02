#!/usr/bin/env python3
"""
validate_music2_zoom_ic.py

Validate a MUSIC2 zoom IC written in AREPO/Gadget HDF5 format.

Checks:
  - Header metadata and particle counts
  - Presence of expected PartType groups
  - Coordinate / velocity / ID / mass datasets
  - Particle mass statistics by type
  - ID uniqueness across all particle types
  - Coordinate bounds relative to BoxSize
  - Bounding boxes for gas and high-resolution DM
  - Gas/HR-DM spatial overlap
  - Coarse-DM mass hierarchy
  - Basic sanity checks for NaN/Inf values

Usage:
    python3 validate_music2_zoom_ic.py /path/to/IC_halo3886_level10

If MUSIC2 wrote a multi-file IC, pass the base filename or the first file.
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import h5py


def resolve_files(path_string):
    p = Path(path_string).expanduser()

    if p.exists():
        return [str(p)]

    # Common multi-file conventions.
    candidates = sorted(glob.glob(str(p) + ".*"))
    candidates = [
        x for x in candidates
        if Path(x).is_file()
        and not x.endswith((".log", ".txt"))
    ]
    if candidates:
        return candidates

    raise FileNotFoundError(f"Could not find IC file or file set: {p}")


def fmt_vec(v):
    return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"


def get_mass_array(g, header_mass, n):
    if "Masses" in g:
        return np.asarray(g["Masses"][:], dtype=np.float64)
    if header_mass > 0:
        return np.full(n, header_mass, dtype=np.float64)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ic", help="MUSIC2 IC file or basename")
    ap.add_argument("--h", type=float, default=0.6732,
                    help="Hubble parameter for optional Msun conversion")
    args = ap.parse_args()

    files = resolve_files(args.ic)
    print("=== MUSIC2 zoom IC validation ===")
    print("Files:")
    for f in files:
        print(f"  {f}")
    print()

    type_counts = np.zeros(6, dtype=np.int64)
    type_mass_values = {i: [] for i in range(6)}
    type_coord_min = {i: np.full(3, np.inf) for i in range(6)}
    type_coord_max = {i: np.full(3, -np.inf) for i in range(6)}
    type_id_min = {i: None for i in range(6)}
    type_id_max = {i: None for i in range(6)}

    all_ids = []
    boxsize = None
    time = None
    redshift = None
    mass_table = None
    num_total_header = None

    bad_finite = []

    for ifile, filename in enumerate(files):
        with h5py.File(filename, "r") as f:
            h = f["Header"].attrs

            if ifile == 0:
                boxsize = float(h.get("BoxSize", np.nan))
                time = float(h.get("Time", np.nan))
                redshift = float(h.get("Redshift", np.nan))
                mass_table = np.asarray(h.get("MassTable", np.zeros(6)),
                                        dtype=np.float64)
                num_total_header = np.asarray(
                    h.get("NumPart_Total", np.zeros(6)),
                    dtype=np.uint64
                )

                print("Header:")
                print(f"  BoxSize       = {boxsize}")
                print(f"  Time          = {time}")
                print(f"  Redshift      = {redshift}")
                print(f"  NumPart_Total = {num_total_header.tolist()}")
                print(f"  MassTable     = {mass_table.tolist()}")
                for key in (
                    "NumFilesPerSnapshot",
                    "Omega0",
                    "OmegaLambda",
                    "HubbleParam",
                ):
                    if key in h:
                        print(f"  {key:14s}= {h[key]}")
                print()

            npart_this = np.asarray(
                h.get("NumPart_ThisFile", np.zeros(6)),
                dtype=np.int64
            )

            for ptype in range(6):
                name = f"PartType{ptype}"
                if name not in f:
                    continue

                g = f[name]
                n = len(g["Coordinates"])
                type_counts[ptype] += n

                if n != npart_this[ptype]:
                    print(
                        f"WARNING: {filename} {name}: "
                        f"group has {n} particles but NumPart_ThisFile says "
                        f"{npart_this[ptype]}"
                    )

                coords = np.asarray(g["Coordinates"][:], dtype=np.float64)
                type_coord_min[ptype] = np.minimum(
                    type_coord_min[ptype], np.min(coords, axis=0)
                )
                type_coord_max[ptype] = np.maximum(
                    type_coord_max[ptype], np.max(coords, axis=0)
                )

                if not np.all(np.isfinite(coords)):
                    bad_finite.append(f"{filename}:{name}/Coordinates")

                if "Velocities" in g:
                    vel = np.asarray(g["Velocities"][:], dtype=np.float64)
                    if not np.all(np.isfinite(vel)):
                        bad_finite.append(f"{filename}:{name}/Velocities")

                if "ParticleIDs" not in g:
                    print(f"WARNING: {name} has no ParticleIDs dataset")
                else:
                    ids = np.asarray(g["ParticleIDs"][:])
                    all_ids.append(ids)
                    imin = int(np.min(ids))
                    imax = int(np.max(ids))
                    if type_id_min[ptype] is None:
                        type_id_min[ptype] = imin
                        type_id_max[ptype] = imax
                    else:
                        type_id_min[ptype] = min(type_id_min[ptype], imin)
                        type_id_max[ptype] = max(type_id_max[ptype], imax)

                masses = get_mass_array(g, mass_table[ptype], n)
                if masses is not None:
                    if not np.all(np.isfinite(masses)):
                        bad_finite.append(f"{filename}:{name}/Masses")
                    type_mass_values[ptype].append(masses)

    print("Particle inventory:")
    print(" type       count            ID min            ID max")
    print("-------------------------------------------------------")
    for ptype in range(6):
        if type_counts[ptype] == 0:
            continue
        print(
            f"  {ptype:1d}   {type_counts[ptype]:12,d}   "
            f"{str(type_id_min[ptype]):>14s}   "
            f"{str(type_id_max[ptype]):>14s}"
        )
    print()

    if num_total_header is not None:
        header_counts = num_total_header.astype(np.int64)
        if np.array_equal(type_counts, header_counts):
            print("PASS: particle counts match Header/NumPart_Total.")
        else:
            print("WARNING: accumulated particle counts do NOT match Header.")
            print("  accumulated:", type_counts.tolist())
            print("  header     :", header_counts.tolist())
    print()

    if all_ids:
        ids = np.concatenate(all_ids)
        n_unique = np.unique(ids).size
        if n_unique == ids.size:
            print(f"PASS: all {ids.size:,} ParticleIDs are globally unique.")
        else:
            print(
                f"WARNING: ParticleIDs are not unique: "
                f"{ids.size - n_unique:,} duplicates."
            )
    print()

    print("Mass statistics:")
    print("Units below are the IC's Gadget mass units (expected 1e10 Msun/h).")
    print(" type      Nmass       min             median          max")
    print("----------------------------------------------------------------")
    for ptype in range(6):
        if not type_mass_values[ptype]:
            continue
        m = np.concatenate(type_mass_values[ptype])
        print(
            f"  {ptype:1d}   {m.size:10,d}   "
            f"{np.min(m):.8e}   {np.median(m):.8e}   {np.max(m):.8e}"
        )
        print(
            f"       median physical scale ~= "
            f"{np.median(m)*1e10/args.h:.6e} Msun "
            f"(assuming mass unit = 1e10 Msun/h)"
        )
    print()

    print("Coordinate bounding boxes:")
    print("Coordinates are reported exactly in the IC's native length units.")
    for ptype in range(6):
        if type_counts[ptype] == 0:
            continue
        cmin = type_coord_min[ptype]
        cmax = type_coord_max[ptype]
        width = cmax - cmin
        print(
            f"  Type {ptype}: min={fmt_vec(cmin)}  "
            f"max={fmt_vec(cmax)}  width={fmt_vec(width)}"
        )
    print()

    if np.isfinite(boxsize):
        coord_ok = True
        for ptype in range(6):
            if type_counts[ptype] == 0:
                continue
            if np.any(type_coord_min[ptype] < 0) or np.any(
                type_coord_max[ptype] >= boxsize
            ):
                coord_ok = False
                print(
                    f"WARNING: Type {ptype} contains coordinates outside "
                    f"[0, BoxSize)."
                )
        if coord_ok:
            print("PASS: all coordinates lie inside [0, BoxSize).")
        print()

    # Type 0 vs Type 1 overlap is expected for baryonic split of HR cells.
    if type_counts[0] > 0 and type_counts[1] > 0:
        lo = np.maximum(type_coord_min[0], type_coord_min[1])
        hi = np.minimum(type_coord_max[0], type_coord_max[1])
        overlap = np.maximum(0.0, hi - lo)
        print("High-resolution gas / DM bbox overlap:")
        print(f"  overlap width = {fmt_vec(overlap)}")
        if np.all(overlap > 0):
            print("PASS: Type 0 and Type 1 occupy overlapping HR regions.")
        else:
            print("WARNING: gas and HR-DM bounding boxes do not overlap.")
        print()

    if type_mass_values[0] and type_mass_values[1]:
        m0 = np.median(np.concatenate(type_mass_values[0]))
        m1 = np.median(np.concatenate(type_mass_values[1]))
        print("High-resolution mass ratio:")
        print(f"  m_DM / m_gas = {m1/m0:.6f}")
        expected = (0.3158 - 0.04936) / 0.04936
        print(f"  cosmological expectation = {expected:.6f}")
        print(f"  fractional difference = {(m1/m0/expected - 1):+.3e}")
        print()

    if type_mass_values[1] and type_mass_values[2]:
        m1 = np.median(np.concatenate(type_mass_values[1]))
        m2 = np.median(np.concatenate(type_mass_values[2]))
        print("Coarse-to-HR DM mass hierarchy:")
        print(f"  median Type2 / Type1 mass = {m2/m1:.6f}")
        print()

    if bad_finite:
        print("WARNING: non-finite values found in:")
        for x in bad_finite:
            print(" ", x)
    else:
        print("PASS: checked coordinates/velocities/masses contain no NaN/Inf.")

    print()
    print("=== Validation complete ===")


if __name__ == "__main__":
    main()
