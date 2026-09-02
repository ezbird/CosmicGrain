#!/usr/bin/env python3
"""
prepare_lagrangian_particle_sets_v2.py

For a fixed list of selected FOF halos in a GADGET-4 parent simulation:
  1. Read halo centers and R200c from the final FOF/SUBFIND catalog.
  2. Read DM particle coordinates and ParticleIDs from the final snapshot.
  3. Select DM particles within trace_factor * R200c of each target halo,
     using periodic distances.
  4. Save one ParticleID file per halo plus a summary table.

This is stage 1 of Lagrangian-region construction.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np


DEFAULT_HALOS = [
    9235, 5834, 7723,
    3352, 3886, 3879,
    859, 1481, 1534,
    295, 308, 441,
]


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


def periodic_delta(pos, center, box):
    d = pos - center
    d -= box * np.rint(d / box)
    return d


def main():
    ap = argparse.ArgumentParser(
        description="Prepare z=0 DM particle-ID sets for halo Lagrangian tracing."
    )
    ap.add_argument("catalog", help="Final fof_subhalo_tab_XXX.hdf5")
    ap.add_argument("snapshot", help="Final snapshot_XXX.hdf5")
    ap.add_argument(
        "--halos", nargs="+", type=int, default=DEFAULT_HALOS,
        help="FOF group indices to process"
    )
    ap.add_argument(
        "--trace-factor", type=float, default=3.0,
        help="Select DM particles within trace_factor * R200c [default: 3]"
    )
    ap.add_argument(
        "--output-dir", default="lagrangian_particle_sets",
        help="Directory for ParticleID files and summary"
    )
    ap.add_argument(
        "--chunk-size", type=int, default=2_000_000,
        help="Number of DM particles read per chunk [default: 2000000]"
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    halos = [int(x) for x in args.halos]
    halo_idx = np.asarray(halos, dtype=np.int64)

    # --- Read catalog metadata for selected halos ---
    with h5py.File(args.catalog, "r") as fc:
        required = [
            "Group/GroupPos",
            "Group/Group_R_Crit200",
            "Group/Group_M_Crit200",
        ]
        missing = [x for x in required if x not in fc]
        if missing:
            raise RuntimeError("Catalog missing required datasets: " + ", ".join(missing))

        n_groups = fc["Group/GroupPos"].shape[0]
        if np.any(halo_idx < 0) or np.any(halo_idx >= n_groups):
            raise RuntimeError("One or more requested halo indices are outside the group catalog.")

        # IMPORTANT:
        # h5py fancy indexing requires indices in strictly increasing order.
        # Our halo list is intentionally ordered by mass bin, so read these
        # relatively small group arrays first and index them with NumPy.
        all_gpos = fc["Group/GroupPos"][:]
        all_r200 = fc["Group/Group_R_Crit200"][:]
        all_m200 = fc["Group/Group_M_Crit200"][:]

        gpos_raw = all_gpos[halo_idx].astype(np.float64)
        r200_raw = all_r200[halo_idx].astype(np.float64)
        m200_raw = all_m200[halo_idx].astype(np.float64)

        cat_box = header_attr(fc, "BoxSize", None)
        h = header_attr(fc, "HubbleParam", None)

    # --- Read snapshot header ---
    with h5py.File(args.snapshot, "r") as fs:
        if "PartType1/Coordinates" not in fs:
            raise RuntimeError("Snapshot has no PartType1/Coordinates.")
        if "PartType1/ParticleIDs" not in fs:
            raise RuntimeError("Snapshot has no PartType1/ParticleIDs.")

        snap_box = header_attr(fs, "BoxSize", None)
        n_dm = fs["PartType1/Coordinates"].shape[0]

    box_raw = snap_box if snap_box is not None else cat_box
    if box_raw is None:
        print("WARNING: BoxSize absent; assuming 50 Mpc/h.", file=sys.stderr)
        box_raw = 50.0

    box_raw = float(np.asarray(box_raw).ravel()[0])
    scale, raw_unit = infer_length_scale_to_mpc_h(box_raw)
    box_mpc_h = box_raw * scale

    if h is None:
        h = 0.6732
        print("WARNING: HubbleParam absent; using h=0.6732 for masses.", file=sys.stderr)
    h = float(np.asarray(h).ravel()[0])

    centers = gpos_raw * scale
    r200 = r200_raw * scale
    rtrace = args.trace_factor * r200
    m200 = m200_raw * 1.0e10 / h

    if np.any(rtrace <= 0):
        bad = [halos[i] for i in np.where(rtrace <= 0)[0]]
        raise RuntimeError(f"Non-positive R200c for halos: {bad}")

    print("")
    print("=== Lagrangian particle-set preparation ===")
    print(f"Catalog       : {args.catalog}")
    print(f"Snapshot      : {args.snapshot}")
    print(f"DM particles  : {n_dm:,}")
    print(f"Box size      : {box_mpc_h:.3f} Mpc/h")
    print(f"Raw unit      : inferred {raw_unit}")
    print(f"Trace radius  : {args.trace_factor:.2f} R200c")
    print(f"Targets       : {len(halos)}")
    print("")

    selected_id_chunks = {hid: [] for hid in halos}
    selected_count = {hid: 0 for hid in halos}

    with h5py.File(args.snapshot, "r") as fs:
        dcoord = fs["PartType1/Coordinates"]
        did = fs["PartType1/ParticleIDs"]

        for start in range(0, n_dm, args.chunk_size):
            stop = min(start + args.chunk_size, n_dm)

            xyz = dcoord[start:stop].astype(np.float64) * scale
            ids = did[start:stop]

            for k, hid in enumerate(halos):
                d = periodic_delta(xyz, centers[k], box_mpc_h)
                r2 = np.einsum("ij,ij->i", d, d)
                mask = r2 <= rtrace[k] * rtrace[k]

                if np.any(mask):
                    chunk_ids = np.asarray(ids[mask])
                    selected_id_chunks[hid].append(chunk_ids)
                    selected_count[hid] += len(chunk_ids)

            print(
                f"processed {stop:,}/{n_dm:,} DM particles",
                end="\r", flush=True
            )

    print(" " * 80, end="\r")

    rows = []

    for k, hid in enumerate(halos):
        if selected_id_chunks[hid]:
            ids = np.concatenate(selected_id_chunks[hid])
        else:
            ids = np.empty(0, dtype=np.uint64)

        ids = np.unique(ids)

        outfile = outdir / f"halo_{hid}_traceIDs.npy"
        np.save(outfile, ids)

        txtfile = outdir / f"halo_{hid}_traceIDs.txt"
        np.savetxt(txtfile, ids, fmt="%d")

        row = {
            "group_index": hid,
            "M200c_Msun": m200[k],
            "R200c_kpc_h": r200[k] * 1000.0,
            "trace_factor": args.trace_factor,
            "trace_radius_kpc_h": rtrace[k] * 1000.0,
            "x_Mpc_h": centers[k, 0],
            "y_Mpc_h": centers[k, 1],
            "z_Mpc_h": centers[k, 2],
            "n_trace_DM": len(ids),
            "n_trace_DM_raw": selected_count[hid],
            "id_npy": str(outfile),
            "id_txt": str(txtfile),
        }
        rows.append(row)

    summary_csv = outdir / "lagrangian_trace_summary.csv"
    fields = list(rows[0].keys())

    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_txt = outdir / "lagrangian_trace_summary.txt"
    with open(summary_txt, "w") as fh:
        fh.write("# Stage-1 Lagrangian tracing summary\n")
        fh.write("# IDs selected from z=0 DM particles within trace_factor * R200c\n")
        fh.write("#\n")
        fh.write(
            "# halo   log10M200   R200[kpc/h]   Rtrace[kpc/h]   "
            "NtraceDM    x[Mpc/h]    y[Mpc/h]    z[Mpc/h]\n"
        )
        for r in rows:
            fh.write(
                f"{r['group_index']:6d} "
                f"{np.log10(r['M200c_Msun']):10.4f} "
                f"{r['R200c_kpc_h']:12.3f} "
                f"{r['trace_radius_kpc_h']:14.3f} "
                f"{r['n_trace_DM']:10d} "
                f"{r['x_Mpc_h']:11.5f} "
                f"{r['y_Mpc_h']:11.5f} "
                f"{r['z_Mpc_h']:11.5f}\n"
            )

    print("=== Trace sets written ===")
    print(
        " halo   log10M200  R200[kpc/h]  Rtrace[kpc/h]  "
        "NtraceDM"
    )
    print("-" * 65)

    for r in rows:
        print(
            f"{r['group_index']:6d} "
            f"{np.log10(r['M200c_Msun']):10.3f} "
            f"{r['R200c_kpc_h']:12.2f} "
            f"{r['trace_radius_kpc_h']:14.2f} "
            f"{r['n_trace_DM']:9d}"
        )

    print("")
    print(f"Wrote summary: {summary_txt}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Particle-ID files are in: {outdir}/")
    print("")
    print("Next step:")
    print("  Match each halo_*_traceIDs.npy set in snapshot_000.hdf5")
    print("  and compute the periodic initial Lagrangian bounds for MUSIC2.")
    print("")


if __name__ == "__main__":
    main()
