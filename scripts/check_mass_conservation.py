#!/usr/bin/env python3

# Run with: python3 check_mass_conservation.py ../simulation_runs_used_for_paper/S10_output_2048/

import argparse
import glob
import os
import h5py
import numpy as np
import pandas as pd


PTYPES = {
    0: "gas",
    1: "dm_hr",
    2: "dm_lr_or_shell",
    3: "ptype3",
    4: "stars",
    5: "bh_or_boundary",
    6: "dust",
}


def read_header(f):
    h = f["Header"].attrs
    return {
        "time": float(h.get("Time", np.nan)),
        "redshift": float(h.get("Redshift", np.nan)),
        "mass_table": np.array(h.get("MassTable", np.zeros(7)), dtype=np.float64),
    }


def sum_particle_type(f, ptype, mass_table):
    gname = f"PartType{ptype}"

    if gname not in f:
        return 0.0, 0

    g = f[gname]

    if "ParticleIDs" in g:
        n = len(g["ParticleIDs"])
    elif "Coordinates" in g:
        n = len(g["Coordinates"])
    else:
        n = 0

    if "Masses" in g:
        mass = np.asarray(g["Masses"], dtype=np.float64).sum()
    else:
        mass = n * mass_table[ptype]

    return float(mass), int(n)


def analyze_piece(path):
    with h5py.File(path, "r") as f:
        header = read_header(f)

        row = {
            "time": header["time"],
            "redshift": header["redshift"],
        }

        for ptype, name in PTYPES.items():
            mass, count = sum_particle_type(f, ptype, header["mass_table"])
            row[f"M_{name}"] = mass
            row[f"N_{name}"] = count

        return row


def find_multifile_snapshots(output_dir):
    snapdirs = sorted(glob.glob(os.path.join(output_dir, "snapdir_*")))
    snapshots = []

    for snapdir in snapdirs:
        snapnum = os.path.basename(snapdir).split("_")[-1]

        files = sorted(glob.glob(os.path.join(
            snapdir, f"snapshot_{snapnum}.*.hdf5"
        )))

        if not files:
            files = sorted(glob.glob(os.path.join(
                snapdir, f"snap_{snapnum}.*.hdf5"
            )))

        if files:
            snapshots.append((snapnum, files))

    return snapshots


def analyze_snapshot(snapnum, files):
    pieces = [analyze_piece(path) for path in files]

    row = {
        "snapnum": int(snapnum),
        "snapshot": f"snapdir_{snapnum}",
        "num_files": len(files),
        "time": pieces[0]["time"],
        "redshift": pieces[0]["redshift"],
    }

    for ptype, name in PTYPES.items():
        row[f"M_{name}"] = sum(p[f"M_{name}"] for p in pieces)
        row[f"N_{name}"] = sum(p[f"N_{name}"] for p in pieces)

    row["M_total"] = sum(row[f"M_{name}"] for name in PTYPES.values())

    row["M_baryon_gas_stars_dust"] = (
        row["M_gas"] +
        row["M_stars"] +
        row["M_dust"]
    )

    row["M_collisionless"] = (
        row["M_dm_hr"] +
        row["M_dm_lr_or_shell"] +
        row["M_ptype3"] +
        row["M_bh_or_boundary"]
    )

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Check mass conservation across GADGET-4 multifile snapshots."
    )
    parser.add_argument("output_dir", help="Directory containing snapdir_### folders")
    parser.add_argument("--out", default="mass_conservation.csv")
    parser.add_argument("--warn-frac", type=float, default=1e-5)
    parser.add_argument("--top", type=int, default=12,
                        help="Number of worst step changes to print")
    args = parser.parse_args()

    snapshots = find_multifile_snapshots(args.output_dir)

    if not snapshots:
        raise RuntimeError("No snapdir_### multifile snapshots found.")

    rows = [analyze_snapshot(snapnum, files) for snapnum, files in snapshots]
    df = pd.DataFrame(rows).sort_values("snapnum").reset_index(drop=True)

    baseline_total = df["M_total"].iloc[0]
    baseline_baryon = df["M_baryon_gas_stars_dust"].iloc[0]

    df["dM_total"] = df["M_total"] - baseline_total
    df["frac_dM_total"] = df["dM_total"] / baseline_total

    df["dM_baryon"] = df["M_baryon_gas_stars_dust"] - baseline_baryon
    df["frac_dM_baryon"] = df["dM_baryon"] / baseline_baryon

    # Per-step deltas
    mass_cols = [
        "M_total",
        "M_baryon_gas_stars_dust",
        "M_collisionless",
        "M_gas",
        "M_stars",
        "M_dust",
        "M_dm_hr",
        "M_dm_lr_or_shell",
        "M_ptype3",
        "M_bh_or_boundary",
    ]

    for col in mass_cols:
        df[f"delta_{col}_step"] = df[col].diff().fillna(0.0)

    # Convenience column: did baryon loss/gain balance collisionless?
    df["delta_baryon_plus_collisionless_step"] = (
        df["delta_M_baryon_gas_stars_dust_step"] +
        df["delta_M_collisionless_step"]
    )

    outpath = os.path.join(args.output_dir, args.out)
    df.to_csv(outpath, index=False)

    print(f"\nWrote: {outpath}\n")

    summary_cols = [
        "snapnum",
        "redshift",
        "num_files",
        "M_total",
        "frac_dM_total",
        "M_baryon_gas_stars_dust",
        "frac_dM_baryon",
        "M_gas",
        "M_stars",
        "M_dust",
    ]

    print("=== Snapshot mass summary ===")
    print(df[summary_cols].to_string(index=False))

    print("\n=== Step-by-step component deltas ===")
    delta_cols = [
        "snapnum",
        "redshift",
        "delta_M_total_step",
        "delta_M_baryon_gas_stars_dust_step",
        "delta_M_collisionless_step",
        "delta_baryon_plus_collisionless_step",
        "delta_M_gas_step",
        "delta_M_stars_step",
        "delta_M_dust_step",
        "delta_M_ptype3_step",
    ]
    print(df[delta_cols].to_string(index=False))

    print(f"\n=== Worst {args.top} baryon step changes ===")
    worst_b = df.reindex(
        df["delta_M_baryon_gas_stars_dust_step"].abs()
        .sort_values(ascending=False)
        .index
    ).head(args.top)

    print(worst_b[[
        "snapnum",
        "redshift",
        "delta_M_baryon_gas_stars_dust_step",
        "delta_M_total_step",
        "delta_M_gas_step",
        "delta_M_stars_step",
        "delta_M_dust_step",
        "delta_M_ptype3_step",
    ]].to_string(index=False))

    print(f"\n=== Worst {args.top} total-mass step changes ===")
    worst_t = df.reindex(
        df["delta_M_total_step"].abs()
        .sort_values(ascending=False)
        .index
    ).head(args.top)

    print(worst_t[[
        "snapnum",
        "redshift",
        "delta_M_total_step",
        "delta_M_baryon_gas_stars_dust_step",
        "delta_M_collisionless_step",
        "delta_M_gas_step",
        "delta_M_stars_step",
        "delta_M_dust_step",
        "delta_M_ptype3_step",
    ]].to_string(index=False))

    bad_total = df[np.abs(df["frac_dM_total"]) > args.warn_frac]
    bad_baryon = df[np.abs(df["frac_dM_baryon"]) > args.warn_frac]

    if len(bad_total):
        print("\nWARNING: total mass drift exceeds threshold:")
        print(bad_total[[
            "snapnum",
            "redshift",
            "frac_dM_total",
            "delta_M_total_step",
        ]].to_string(index=False))

    if len(bad_baryon):
        print("\nWARNING: baryonic mass drift exceeds threshold:")
        print(bad_baryon[[
            "snapnum",
            "redshift",
            "frac_dM_baryon",
            "delta_M_baryon_gas_stars_dust_step",
        ]].to_string(index=False))

    # ── Final summary: first snapshot -> last snapshot, in Msun and % ────────
    # Gadget-4 code units: mass in 1e10 Msun/h. HubbleParam read from any
    # snapshot file's Parameters group (consistent across all snapshots).
    first_file = snapshots[0][1][0]
    with h5py.File(first_file, "r") as f:
        params = f["Parameters"].attrs if "Parameters" in f else {}
        h = float(params.get("HubbleParam", 0.6732))

    msun_per_code = 1.0e10 / h

    final_total  = df["M_total"].iloc[-1]
    final_baryon = df["M_baryon_gas_stars_dust"].iloc[-1]

    drift_total_code  = final_total - baseline_total
    drift_baryon_code = final_baryon - baseline_baryon

    drift_total_msun  = drift_total_code * msun_per_code
    drift_baryon_msun = drift_baryon_code * msun_per_code

    pct_total  = 100.0 * drift_total_code / baseline_total
    pct_baryon = 100.0 * drift_baryon_code / baseline_baryon

    z_first = df["redshift"].iloc[0]
    z_last  = df["redshift"].iloc[-1]

    print("\n" + "=" * 70)
    print("=== FINAL MASS CONSERVATION SUMMARY (first snap -> last snap) ===")
    print("=" * 70)
    print(f"  HubbleParam used        : {h:.4f}")
    print(f"  Redshift range          : z={z_first:.3f}  ->  z={z_last:.3f}")
    print(f"  Initial baryon mass     : {baseline_baryon*msun_per_code:.4e} Msun")
    print(f"  Final baryon mass       : {final_baryon*msun_per_code:.4e} Msun")
    print(f"  Baryon mass drift       : {drift_baryon_msun:+.4e} Msun  "
          f"({pct_baryon:+.5f}%)")
    print(f"  Initial total mass      : {baseline_total*msun_per_code:.4e} Msun")
    print(f"  Final total mass        : {final_total*msun_per_code:.4e} Msun")
    print(f"  Total mass drift        : {drift_total_msun:+.4e} Msun  "
          f"({pct_total:+.7f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
