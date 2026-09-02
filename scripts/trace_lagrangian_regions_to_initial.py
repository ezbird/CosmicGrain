#!/usr/bin/env python3
"""
trace_lagrangian_regions_to_initial.py

Stage 2 of CosmicGrain zoom-region preparation.

For each halo ParticleID set created by prepare_lagrangian_particle_sets_v2.py:
  1. Read the initial parent snapshot (typically snapshot_000.hdf5).
  2. Match the target ParticleIDs to PartType1/ParticleIDs.
  3. Extract their initial coordinates.
  4. Compute a compact periodic-unwrapped Lagrangian region.
  5. Report center, min/max bounds, widths, and a padded MUSIC2-friendly box.

Outputs:
    lagrangian_regions_initial/lagrangian_regions_summary.txt
    lagrangian_regions_initial/lagrangian_regions_summary.csv
    one coordinate .npy file per halo

Coordinates are reported in Mpc/h.

The periodic-unwrapping method is robust for compact Lagrangian patches that
may straddle a periodic boundary: each dimension is shifted relative to a
circular mean, then unwrapped into a continuous interval.
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def header_attr(f, name, default=None):
    if "Header" in f and name in f["Header"].attrs:
        return f["Header"].attrs[name]
    if name in f.attrs:
        return f.attrs[name]
    return default


def infer_length_scale_to_mpc_h(boxsize):
    if boxsize > 1000.0:
        return 1.0e-3, "kpc/h"
    return 1.0, "Mpc/h"


def circular_unwrap_1d(x, box):
    """
    Unwrap periodic coordinates into a compact continuous interval.

    Uses circular mean as an anchor and maps all points into +/- box/2 of it.
    """
    theta = 2.0 * np.pi * x / box
    c = np.mean(np.cos(theta))
    s = np.mean(np.sin(theta))
    ang = np.arctan2(s, c)
    if ang < 0:
        ang += 2.0 * np.pi
    anchor = box * ang / (2.0 * np.pi)

    dx = x - anchor
    dx -= box * np.rint(dx / box)
    xu = anchor + dx
    return xu, anchor


def periodic_wrap(x, box):
    return np.mod(x, box)


def parse_halo_id(path):
    m = re.search(r"halo_(\d+)_traceIDs\.npy$", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse halo id from {path}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(
        description="Trace z=0 halo ParticleID sets back to the initial snapshot."
    )
    ap.add_argument("initial_snapshot", help="Initial parent snapshot, e.g. snapshot_000.hdf5")
    ap.add_argument(
        "--id-dir", default="lagrangian_particle_sets",
        help="Directory containing halo_*_traceIDs.npy"
    )
    ap.add_argument(
        "--output-dir", default="lagrangian_regions_initial",
        help="Output directory"
    )
    ap.add_argument(
        "--padding-frac", type=float, default=0.15,
        help="Fractional padding added to each side-length [default: 0.15]"
    )
    ap.add_argument(
        "--padding-absolute", type=float, default=0.25,
        help="Minimum total extra padding per dimension in Mpc/h [default: 0.25]"
    )
    args = ap.parse_args()

    id_files = sorted(
        glob.glob(os.path.join(args.id_dir, "halo_*_traceIDs.npy")),
        key=parse_halo_id
    )

    if not id_files:
        raise RuntimeError(f"No halo_*_traceIDs.npy files found in {args.id_dir}")

    halo_ids = [parse_halo_id(p) for p in id_files]
    target_ids = {hid: np.load(path) for hid, path in zip(halo_ids, id_files)}

    # Build one combined ID array so the 134M-particle initial snapshot is read only once.
    all_target_ids = np.unique(np.concatenate([target_ids[h] for h in halo_ids]))
    all_target_ids.sort()

    print("")
    print("=== Initial-condition Lagrangian tracing ===")
    print(f"Initial snapshot : {args.initial_snapshot}")
    print(f"Target halos     : {len(halo_ids)}")
    print(f"Unique target IDs: {len(all_target_ids):,}")
    print("")

    with h5py.File(args.initial_snapshot, "r") as f:
        if "PartType1/ParticleIDs" not in f or "PartType1/Coordinates" not in f:
            raise RuntimeError("Initial snapshot must contain PartType1/ParticleIDs and Coordinates.")

        ids_ds = f["PartType1/ParticleIDs"]
        xyz_ds = f["PartType1/Coordinates"]
        n_dm = ids_ds.shape[0]

        box_raw = header_attr(f, "BoxSize", None)
        if box_raw is None:
            print("WARNING: BoxSize absent; assuming 50 Mpc/h.", file=sys.stderr)
            box_raw = 50.0

        box_raw = float(np.asarray(box_raw).ravel()[0])
        scale, raw_unit = infer_length_scale_to_mpc_h(box_raw)
        box = box_raw * scale

        print(f"Initial DM count  : {n_dm:,}")
        print(f"Box size          : {box:.3f} Mpc/h")
        print(f"Raw unit          : inferred {raw_unit}")
        print("")

        # Match in chunks using searchsorted on sorted target IDs.
        matched_ids = []
        matched_xyz = []

        chunk = 2_000_000
        ntarget = len(all_target_ids)

        for start in range(0, n_dm, chunk):
            stop = min(start + chunk, n_dm)
            ids = np.asarray(ids_ds[start:stop])

            # For each snapshot ID, test membership in sorted target list.
            loc = np.searchsorted(all_target_ids, ids)
            mask = loc < ntarget
            idx = np.where(mask)[0]
            if len(idx):
                ok = all_target_ids[loc[idx]] == ids[idx]
                idx = idx[ok]

            if len(idx):
                xyz = np.asarray(xyz_ds[start:stop][idx], dtype=np.float64) * scale
                matched_ids.append(ids[idx].copy())
                matched_xyz.append(xyz)

            print(f"processed {stop:,}/{n_dm:,} initial DM particles", end="\r", flush=True)

    print(" " * 90, end="\r")

    if matched_ids:
        matched_ids = np.concatenate(matched_ids)
        matched_xyz = np.concatenate(matched_xyz, axis=0)
    else:
        raise RuntimeError("No target ParticleIDs were found in the initial snapshot.")

    # Sort once by ID for efficient per-halo lookup.
    order = np.argsort(matched_ids)
    matched_ids = matched_ids[order]
    matched_xyz = matched_xyz[order]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for hid in halo_ids:
        ids = np.asarray(target_ids[hid])
        ids_sorted = np.sort(ids)

        loc = np.searchsorted(matched_ids, ids_sorted)
        valid = loc < len(matched_ids)
        loc2 = loc[valid]
        ids2 = ids_sorted[valid]
        good = matched_ids[loc2] == ids2
        loc2 = loc2[good]

        xyz = matched_xyz[loc2]

        n_expected = len(ids)
        n_found = len(xyz)

        if n_found == 0:
            print(f"WARNING: halo {hid}: no IDs found", file=sys.stderr)
            continue

        # Periodically unwrap each dimension independently.
        xyz_u = np.empty_like(xyz)
        anchors = np.zeros(3)
        for dim in range(3):
            xyz_u[:, dim], anchors[dim] = circular_unwrap_1d(xyz[:, dim], box)

        mins = xyz_u.min(axis=0)
        maxs = xyz_u.max(axis=0)
        widths = maxs - mins
        center_u = 0.5 * (mins + maxs)
        center_wrapped = periodic_wrap(center_u, box)

        # MUSIC2-style padded rectangular bounds.
        # Add at least padding_absolute total extra width, or padding_frac * width,
        # whichever is larger.
        extra = np.maximum(args.padding_frac * widths, args.padding_absolute)
        padded_widths = widths + extra
        pad_mins = center_u - 0.5 * padded_widths
        pad_maxs = center_u + 0.5 * padded_widths

        # Fractional coordinates in box units, useful for IC generators.
        center_frac = periodic_wrap(center_u, box) / box
        width_frac = padded_widths / box

        np.save(outdir / f"halo_{hid}_initial_coords_unwrapped.npy", xyz_u)

        row = {
            "group_index": hid,
            "n_expected": n_expected,
            "n_found": n_found,
            "match_fraction": n_found / n_expected,
            "center_x_Mpc_h": center_wrapped[0],
            "center_y_Mpc_h": center_wrapped[1],
            "center_z_Mpc_h": center_wrapped[2],
            "width_x_Mpc_h": widths[0],
            "width_y_Mpc_h": widths[1],
            "width_z_Mpc_h": widths[2],
            "max_width_Mpc_h": widths.max(),
            "padded_width_x_Mpc_h": padded_widths[0],
            "padded_width_y_Mpc_h": padded_widths[1],
            "padded_width_z_Mpc_h": padded_widths[2],
            "center_x_frac": center_frac[0],
            "center_y_frac": center_frac[1],
            "center_z_frac": center_frac[2],
            "padded_width_x_frac": width_frac[0],
            "padded_width_y_frac": width_frac[1],
            "padded_width_z_frac": width_frac[2],
            "unwrapped_min_x": mins[0],
            "unwrapped_min_y": mins[1],
            "unwrapped_min_z": mins[2],
            "unwrapped_max_x": maxs[0],
            "unwrapped_max_y": maxs[1],
            "unwrapped_max_z": maxs[2],
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No halo regions were successfully constructed.")

    csv_path = outdir / "lagrangian_regions_summary.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    txt_path = outdir / "lagrangian_regions_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("# Initial Lagrangian region summary\n")
        fh.write("# center coordinates are wrapped to [0, BoxSize) in Mpc/h\n")
        fh.write("# widths are compact periodic-unwrapped extents\n")
        fh.write(f"# padding_frac={args.padding_frac}, padding_absolute={args.padding_absolute} Mpc/h\n")
        fh.write("#\n")
        fh.write(
            "# halo  Nfound/Nexp   center_x  center_y  center_z   "
            "Wx    Wy    Wz   Wmax   padded_Wx padded_Wy padded_Wz\n"
        )
        for r in rows:
            fh.write(
                f"{r['group_index']:6d} "
                f"{r['n_found']:7d}/{r['n_expected']:<7d} "
                f"{r['center_x_Mpc_h']:9.4f} "
                f"{r['center_y_Mpc_h']:9.4f} "
                f"{r['center_z_Mpc_h']:9.4f} "
                f"{r['width_x_Mpc_h']:6.3f} "
                f"{r['width_y_Mpc_h']:6.3f} "
                f"{r['width_z_Mpc_h']:6.3f} "
                f"{r['max_width_Mpc_h']:6.3f} "
                f"{r['padded_width_x_Mpc_h']:9.3f} "
                f"{r['padded_width_y_Mpc_h']:9.3f} "
                f"{r['padded_width_z_Mpc_h']:9.3f}\n"
            )

    print("=== Initial Lagrangian regions ===")
    print(
        " halo   matched       center [Mpc/h]              "
        "extent [Mpc/h]        maxW"
    )
    print("-" * 100)

    for r in rows:
        print(
            f"{r['group_index']:6d} "
            f"{r['n_found']:6d}/{r['n_expected']:<6d} "
            f"({r['center_x_Mpc_h']:6.2f}, "
            f"{r['center_y_Mpc_h']:6.2f}, "
            f"{r['center_z_Mpc_h']:6.2f})   "
            f"({r['width_x_Mpc_h']:5.2f}, "
            f"{r['width_y_Mpc_h']:5.2f}, "
            f"{r['width_z_Mpc_h']:5.2f})   "
            f"{r['max_width_Mpc_h']:5.2f}"
        )

    print("")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {csv_path}")
    print("")
    print("Important:")
    print("  Every halo should ideally have match_fraction = 1.0.")
    print("  The reported center and padded fractional widths can be used")
    print("  as the starting geometry for each halo's MUSIC2 refinement region.")
    print("")


if __name__ == "__main__":
    main()
