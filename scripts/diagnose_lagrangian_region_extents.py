#!/usr/bin/env python3
"""
diagnose_lagrangian_region_extents.py

Recompute the initial Lagrangian extents using an exact minimum periodic
interval in each coordinate dimension (largest-gap method).

This is a diagnostic / replacement for the circular-mean unwrapping used in
trace_lagrangian_regions_to_initial.py. For each halo, the algorithm finds the
shortest interval on a periodic axis that contains all traced particles.

Inputs:
    initial snapshot, e.g. snapshot_000.hdf5
    lagrangian_particle_sets/halo_*_traceIDs.npy

Outputs:
    lagrangian_regions_minimal/lagrangian_regions_minimal_summary.txt
    lagrangian_regions_minimal/lagrangian_regions_minimal_summary.csv
"""

import argparse
import csv
import glob
import os
import re
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
    return (1.0e-3, "kpc/h") if boxsize > 1000.0 else (1.0, "Mpc/h")


def parse_halo_id(path):
    m = re.search(r"halo_(\d+)_traceIDs\.npy$", os.path.basename(path))
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def minimal_periodic_interval(x, box):
    """
    Return exact shortest periodic interval containing all x.

    Sort wrapped positions. The complement of the largest empty periodic gap
    is the minimum containing interval.
    """
    x = np.mod(np.asarray(x, dtype=np.float64), box)
    xs = np.sort(x)

    if len(xs) == 1:
        return xs.copy(), xs[0], 0.0, xs[0], xs[0]

    gaps = np.diff(xs)
    wrap_gap = xs[0] + box - xs[-1]
    all_gaps = np.concatenate([gaps, [wrap_gap]])
    igap = int(np.argmax(all_gaps))

    # Interval begins immediately after the largest gap.
    if igap == len(xs) - 1:
        start = xs[0]
    else:
        start = xs[igap + 1]

    xu = x.copy()
    xu[xu < start] += box

    xmin = xu.min()
    xmax = xu.max()
    width = xmax - xmin
    center_unwrapped = 0.5 * (xmin + xmax)
    center_wrapped = center_unwrapped % box

    return xu, center_wrapped, width, xmin, xmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("initial_snapshot")
    ap.add_argument("--id-dir", default="lagrangian_particle_sets")
    ap.add_argument("--output-dir", default="lagrangian_regions_minimal")
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    ap.add_argument("--padding-frac", type=float, default=0.15)
    ap.add_argument("--padding-absolute", type=float, default=0.25)
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.id_dir, "halo_*_traceIDs.npy")),
        key=parse_halo_id
    )
    if not files:
        raise RuntimeError("No halo trace ID files found.")

    halo_ids = [parse_halo_id(p) for p in files]
    per_halo_ids = {h: np.load(p) for h, p in zip(halo_ids, files)}
    target = np.unique(np.concatenate([per_halo_ids[h] for h in halo_ids]))
    target.sort()

    with h5py.File(args.initial_snapshot, "r") as f:
        ids_ds = f["PartType1/ParticleIDs"]
        xyz_ds = f["PartType1/Coordinates"]

        box_raw = float(np.asarray(header_attr(f, "BoxSize", 50.0)).ravel()[0])
        scale, raw_unit = infer_length_scale_to_mpc_h(box_raw)
        box = box_raw * scale

        mids = []
        mxyz = []
        n = len(ids_ds)
        nt = len(target)

        for start in range(0, n, args.chunk_size):
            stop = min(start + args.chunk_size, n)
            ids = np.asarray(ids_ds[start:stop])

            loc = np.searchsorted(target, ids)
            valid = loc < nt
            idx = np.where(valid)[0]
            if len(idx):
                ok = target[loc[idx]] == ids[idx]
                idx = idx[ok]

            if len(idx):
                mids.append(ids[idx].copy())
                mxyz.append(np.asarray(xyz_ds[start:stop][idx], dtype=np.float64) * scale)

            print(f"processed {stop:,}/{n:,}", end="\r", flush=True)

    print(" " * 60, end="\r")

    mids = np.concatenate(mids)
    mxyz = np.concatenate(mxyz, axis=0)
    order = np.argsort(mids)
    mids = mids[order]
    mxyz = mxyz[order]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for hid in halo_ids:
        ids = np.sort(per_halo_ids[hid])
        loc = np.searchsorted(mids, ids)
        good = (loc < len(mids))
        loc2 = loc[good]
        ids2 = ids[good]
        good2 = mids[loc2] == ids2
        xyz = mxyz[loc2[good2]]

        nfound = len(xyz)
        nexp = len(ids)

        centers = np.zeros(3)
        widths = np.zeros(3)
        mins = np.zeros(3)
        maxs = np.zeros(3)
        xyz_u = np.empty_like(xyz)

        for d in range(3):
            xyz_u[:, d], centers[d], widths[d], mins[d], maxs[d] = \
                minimal_periodic_interval(xyz[:, d], box)

        # Note: each dimension's unwrapping is independently optimal.
        # These coordinates are suitable for axis-aligned bounding boxes.
        np.save(outdir / f"halo_{hid}_initial_coords_minimal_unwrapped.npy", xyz_u)

        extra = np.maximum(args.padding_frac * widths, args.padding_absolute)
        pwidth = widths + extra

        row = {
            "group_index": hid,
            "n_expected": nexp,
            "n_found": nfound,
            "match_fraction": nfound / nexp,
            "center_x_Mpc_h": centers[0],
            "center_y_Mpc_h": centers[1],
            "center_z_Mpc_h": centers[2],
            "width_x_Mpc_h": widths[0],
            "width_y_Mpc_h": widths[1],
            "width_z_Mpc_h": widths[2],
            "max_width_Mpc_h": widths.max(),
            "padded_width_x_Mpc_h": pwidth[0],
            "padded_width_y_Mpc_h": pwidth[1],
            "padded_width_z_Mpc_h": pwidth[2],
            "center_x_frac": centers[0] / box,
            "center_y_frac": centers[1] / box,
            "center_z_frac": centers[2] / box,
            "padded_width_x_frac": pwidth[0] / box,
            "padded_width_y_frac": pwidth[1] / box,
            "padded_width_z_frac": pwidth[2] / box,
        }
        rows.append(row)

    csv_path = outdir / "lagrangian_regions_minimal_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    txt_path = outdir / "lagrangian_regions_minimal_summary.txt"
    with open(txt_path, "w") as fh:
        fh.write("# Exact minimum periodic-axis Lagrangian extents\n")
        fh.write("#\n")
        fh.write("# halo matched center_x center_y center_z Wx Wy Wz Wmax padded_Wx padded_Wy padded_Wz\n")
        for r in rows:
            fh.write(
                f"{r['group_index']:6d} {r['n_found']:6d}/{r['n_expected']:<6d} "
                f"{r['center_x_Mpc_h']:8.3f} {r['center_y_Mpc_h']:8.3f} {r['center_z_Mpc_h']:8.3f} "
                f"{r['width_x_Mpc_h']:7.3f} {r['width_y_Mpc_h']:7.3f} {r['width_z_Mpc_h']:7.3f} "
                f"{r['max_width_Mpc_h']:7.3f} "
                f"{r['padded_width_x_Mpc_h']:9.3f} {r['padded_width_y_Mpc_h']:9.3f} {r['padded_width_z_Mpc_h']:9.3f}\n"
            )

    print("=== Exact minimum-periodic Lagrangian regions ===")
    print(" halo   matched       center [Mpc/h]              extent [Mpc/h]        maxW")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['group_index']:6d} "
            f"{r['n_found']:6d}/{r['n_expected']:<6d} "
            f"({r['center_x_Mpc_h']:6.2f}, {r['center_y_Mpc_h']:6.2f}, {r['center_z_Mpc_h']:6.2f})   "
            f"({r['width_x_Mpc_h']:5.2f}, {r['width_y_Mpc_h']:5.2f}, {r['width_z_Mpc_h']:5.2f})   "
            f"{r['max_width_Mpc_h']:5.2f}"
        )

    print("")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
