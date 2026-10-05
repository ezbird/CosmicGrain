#!/usr/bin/env python3
"""
CosmicGrain -- SAGAbg dwarf-galaxy comparison.

Measures z=0 galaxy properties directly from final CosmicGrain snapshots,
using the same halo definition / center refinement as the standardized
post-run audit. The target FOF group index is read from each run's existing
post-run audit JSON rather than hard-coded.

For every completed CG halo this script measures within a 30 pkpc aperture:
    Mstar
    Mgas
    SFR
    12+log10(O/H), all gas, gas-mass weighted
    12+log10(O/H), star-forming gas, gas-mass weighted
    12+log10(O/H), star-forming gas, SFR weighted

The SFR-weighted abundance is intended as the primary comparison to
nebular/emission-line SAGAbg metallicities.

CosmicGrain convention:
    X_H = 0.76
    O/H = (X_O / 16) / X_H

Averaging is performed in linear O/H and converted to 12+log10(O/H)
only afterward.

Run from the gadget4 root:
    python3 scripts/cosmicgrain_sagabg_comparison.py

Test one halo:
    python3 scripts/cosmicgrain_sagabg_comparison.py --only CG06
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Dict, List, Sequence

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from halo_utils import find_snapshot_and_group_files, get_zoom_halo, periodic_delta


CG_HALOS = {
    "CG01": 9235,
    "CG02": 7723,
    "CG03": 5834,
    "CG04": 3886,
    "CG05": 3879,
    "CG06": 3352,
    "CG07": 1534,
    "CG08": 1481,
    "CG09": 859,
    "CG10": 441,
    "CG11": 308,
    "CG12": 295,
}

# Parent-box values are used only as a gross target-identification sanity check.
# Zoom values can differ, so the allowed ratio is deliberately broad.
PARENT_M200_MSUN = {
    "CG01": 5.60e10, "CG02": 6.50e10, "CG03": 8.77e10, "CG04": 1.33e11,
    "CG05": 1.44e11, "CG06": 1.59e11, "CG07": 4.14e11, "CG08": 4.24e11,
    "CG09": 6.81e11, "CG10": 1.06e12, "CG11": 1.35e12, "CG12": 1.95e12,
}

X_H = 0.76
A_O = 16.0
OXYGEN_FIELD = "GasOxygenMassFraction"
SFR_FIELD = "StarFormationRate"


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    return str(value)


def get_group_index_from_audit(output_dir: str):
    patterns = [
        os.path.join(output_dir, "*_postrun_audit.json"),
        os.path.join(output_dir, "*postrun*audit*.json"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))
    if not matches:
        raise RuntimeError(f"no post-run audit JSON found in {output_dir}")

    audit_file = max(matches, key=os.path.getmtime)
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    try:
        group_index = int(audit["halo"]["group_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{audit_file}: missing/invalid halo.group_index"
        ) from exc

    return group_index, audit_file


def read_header(snapshot_file: str) -> Dict:
    with h5py.File(snapshot_file, "r") as f:
        hdr = f["Header"].attrs
        if "Parameters" not in f:
            raise RuntimeError(f"{snapshot_file}: Parameters group not found")
        params = f["Parameters"].attrs
        if "HubbleParam" not in params:
            raise RuntimeError(f"{snapshot_file}: Parameters/HubbleParam not found")
        return {
            "a": float(hdr["Time"]),
            "z": float(hdr.get("Redshift", 1.0 / float(hdr["Time"]) - 1.0)),
            "h": float(params["HubbleParam"]),
            "box": float(hdr["BoxSize"]),
            "git_commit": decode_attr(hdr.get("Git_commit", "UNKNOWN")),
            "git_date": decode_attr(hdr.get("Git_date", "UNKNOWN")),
        }


def read_particle_fields(
    snapshot_files: Sequence[str], ptype: int, fields: Sequence[str]
) -> Dict[str, np.ndarray]:
    output: Dict[str, List[np.ndarray]] = {field: [] for field in fields}
    output["_Masses"] = []
    found_particles = False

    for filename in snapshot_files:
        with h5py.File(filename, "r") as f:
            group_name = f"PartType{ptype}"
            if group_name not in f:
                continue

            found_particles = True
            g = f[group_name]
            n = len(g["Coordinates"])

            if "Masses" in g:
                masses = np.asarray(g["Masses"], dtype=np.float64)
            else:
                mass_table = np.asarray(f["Header"].attrs["MassTable"], dtype=np.float64)
                masses = np.full(n, mass_table[ptype], dtype=np.float64)
            output["_Masses"].append(masses)

            for field in fields:
                if field not in g:
                    raise RuntimeError(f"{filename}: {group_name}/{field} not found")
                output[field].append(np.asarray(g[field], dtype=np.float64))

    if not found_particles:
        raise RuntimeError(f"PartType{ptype} not present")

    return {key: np.concatenate(pieces) for key, pieces in output.items() if pieces}


def read_coords_and_fields(
    snapshot_files: Sequence[str], ptype: int, fields: Sequence[str]
) -> Dict[str, np.ndarray]:
    return read_particle_fields(snapshot_files, ptype, ["Coordinates"] + list(fields))


def aperture_mask(
    coords: np.ndarray, center: np.ndarray, box: float, radius_ckpch: float
) -> np.ndarray:
    delta = periodic_delta(coords, center, box)
    r2 = np.einsum("ij,ij->i", delta, delta)
    return r2 < radius_ckpch**2


def oxygen_number_ratio(xo: np.ndarray) -> np.ndarray:
    return (np.asarray(xo, dtype=np.float64) / A_O) / X_H


def weighted_linear_oh(oh: np.ndarray, weights: np.ndarray) -> float:
    oh = np.asarray(oh, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = (
        np.isfinite(oh)
        & np.isfinite(weights)
        & (oh >= 0.0)
        & (weights > 0.0)
    )
    if not np.any(good):
        return float("nan")
    return float(np.sum(oh[good] * weights[good]) / np.sum(weights[good]))


def oh12(oh: float) -> float:
    if not np.isfinite(oh) or oh <= 0:
        return float("nan")
    return 12.0 + math.log10(oh)


def finite_minmax(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    q = np.isfinite(x)
    if not np.any(q):
        return float("nan"), float("nan")
    return float(np.min(x[q])), float(np.max(x[q]))


def analyze_halo(
    cg_name: str,
    halo_id: int,
    output_dir: str,
    group_index: int,
    snap: int,
    aperture_pkpc: float,
    refine_center: bool = True,
) -> Dict:
    snapshot_files, _ = find_snapshot_and_group_files(output_dir, snap)
    if not snapshot_files:
        raise RuntimeError(f"snapshot {snap:03d} not found")

    header = read_header(snapshot_files[0])

    halo = get_zoom_halo(
        output_dir,
        snap,
        group_index=group_index,
        refine_center=refine_center,
        verbose=False,
    )

    center = np.asarray(halo.chosen_center_ckpch, dtype=np.float64)
    box = float(halo.boxsize_ckpch)
    a, h = float(halo.a), float(halo.h)
    aperture_ckpch = aperture_pkpc * h / a
    mass_to_msun = 1.0e10 / h
    length_to_pkpc = a / h

    gas = read_coords_and_fields(
        snapshot_files, 0, [OXYGEN_FIELD, SFR_FIELD]
    )
    gas_coords = gas["Coordinates"]
    gas_mass = gas["_Masses"]
    gas_xo = gas[OXYGEN_FIELD]
    gas_sfr = gas[SFR_FIELD]

    qgas = aperture_mask(gas_coords, center, box, aperture_ckpch)
    mass_ap = gas_mass[qgas]
    xo_ap = gas_xo[qgas]
    sfr_ap = gas_sfr[qgas]
    oh_ap = oxygen_number_ratio(xo_ap)

    qsf = np.isfinite(sfr_ap) & (sfr_ap > 0.0)

    oh_all_mass = weighted_linear_oh(oh_ap, mass_ap)
    oh_sf_mass = weighted_linear_oh(oh_ap[qsf], mass_ap[qsf])
    oh_sf_sfr = weighted_linear_oh(oh_ap[qsf], sfr_ap[qsf])

    stars = read_coords_and_fields(snapshot_files, 4, [])
    qstar = aperture_mask(stars["Coordinates"], center, box, aperture_ckpch)

    mstar_msun = float(np.sum(stars["_Masses"][qstar]) * mass_to_msun)
    mgas_msun = float(np.sum(mass_ap) * mass_to_msun)
    sfr_msun_yr = float(np.sum(sfr_ap))
    m200_msun = float(halo.so_m200_code * mass_to_msun)
    r200_pkpc = float(halo.so_r200_ckpch * length_to_pkpc)
    catalog_m200_msun = float(halo.catalog_m200_code * mass_to_msun)
    catalog_r200_pkpc = float(halo.catalog_r200_ckpch * length_to_pkpc)

    xo_min, xo_max = finite_minmax(xo_ap)
    n_bad_xo = int(np.count_nonzero(
        ~np.isfinite(xo_ap) | (xo_ap < 0.0) | (xo_ap > 1.0)
    ))
    n_bad_sfr = int(np.count_nonzero(
        ~np.isfinite(sfr_ap) | (sfr_ap < 0.0)
    ))

    parent_m200 = PARENT_M200_MSUN[cg_name]
    m200_parent_ratio = m200_msun / parent_m200
    target_mass_sanity = 0.4 <= m200_parent_ratio <= 2.5

    return {
        "CG": cg_name,
        "halo_id": int(halo_id),
        "group_index": int(group_index),
        "snapshot": int(snap),
        "z": float(halo.z),
        "a": a,
        "h": h,
        "git_commit": header["git_commit"],
        "git_date": header["git_date"],
        "aperture_pkpc": float(aperture_pkpc),
        "center_x_ckpch": float(center[0]),
        "center_y_ckpch": float(center[1]),
        "center_z_ckpch": float(center[2]),
        "center_refinement_accepted": bool(halo.refinement_accepted),
        "center_shift_pkpc": float(halo.refinement_shift_ckpch * length_to_pkpc),
        "M200c_Msun": m200_msun,
        "R200c_pkpc": r200_pkpc,
        "catalog_M200c_Msun": catalog_m200_msun,
        "catalog_R200c_pkpc": catalog_r200_pkpc,
        "parent_M200c_Msun": parent_m200,
        "zoom_to_parent_M200_ratio": m200_parent_ratio,
        "target_mass_sanity": target_mass_sanity,
        "Mstar_Msun": mstar_msun,
        "Mgas_Msun": mgas_msun,
        "SFR_Msun_yr": sfr_msun_yr,
        "log10_Mstar_Msun": math.log10(mstar_msun) if mstar_msun > 0 else float("nan"),
        "N_gas_30pkpc": int(np.sum(qgas)),
        "N_SFgas_30pkpc": int(np.sum(qsf)),
        "N_star_30pkpc": int(np.sum(qstar)),
        "OH_linear_allgas_massweighted": oh_all_mass,
        "OH12_allgas_massweighted": oh12(oh_all_mass),
        "OH_linear_SFgas_massweighted": oh_sf_mass,
        "OH12_SFgas_massweighted": oh12(oh_sf_mass),
        "OH_linear_SFRweighted": oh_sf_sfr,
        "OH12_SFRweighted": oh12(oh_sf_sfr),
        "XO_min_30pkpc": xo_min,
        "XO_max_30pkpc": xo_max,
        "bad_oxygen_values": n_bad_xo,
        "bad_sfr_values": n_bad_sfr,
    }


def print_halo(row: Dict) -> None:
    print()
    print("=" * 76)
    print(f"{row['CG']}  (halo {row['halo_id']}, audited group {row['group_index']})")
    print("=" * 76)
    print(f"audit source          = {row['audit_file']}")
    print(f"z                     = {row['z']:.8f}")
    print(f"Git commit            = {row['git_commit']}")
    print(f"Git date              = {row['git_date']}")
    print(f"center refinement     = {row['center_refinement_accepted']}")
    print(f"center shift          = {row['center_shift_pkpc']:.3f} pkpc")
    print(f"M200c                 = {row['M200c_Msun']:.6e} Msun")
    print(f"R200c                 = {row['R200c_pkpc']:.3f} pkpc")
    print(f"parent M200c          = {row['parent_M200c_Msun']:.6e} Msun")
    print(f"zoom / parent M200    = {row['zoom_to_parent_M200_ratio']:.3f}")
    print(f"target mass sanity    = {'PASS' if row['target_mass_sanity'] else 'WARNING'}")
    print(f"aperture              = {row['aperture_pkpc']:.1f} pkpc")
    print()
    print(f"N gas                 = {row['N_gas_30pkpc']:,}")
    print(f"N SF gas              = {row['N_SFgas_30pkpc']:,}")
    print(f"N stars               = {row['N_star_30pkpc']:,}")
    print()
    print(f"Mstar                 = {row['Mstar_Msun']:.6e} Msun")
    print(f"Mgas                  = {row['Mgas_Msun']:.6e} Msun")
    print(f"SFR                   = {row['SFR_Msun_yr']:.6e} Msun/yr")
    print()
    print("12 + log10(O/H)")
    print(f"  all gas, mass wt    = {row['OH12_allgas_massweighted']:.5f}")
    print(f"  SF gas, mass wt     = {row['OH12_SFgas_massweighted']:.5f}")
    print(f"  SFR weighted        = {row['OH12_SFRweighted']:.5f}   <-- primary SAGAbg comparison")
    print()
    print(f"X_O range             = {row['XO_min_30pkpc']:.6e} .. {row['XO_max_30pkpc']:.6e}")
    print(f"bad oxygen / SFR      = {row['bad_oxygen_values']} / {row['bad_sfr_values']}")
    if row["N_SFgas_30pkpc"] < 20:
        print("NOTE                  = low SF-gas particle count; treat SFR-weighted O/H cautiously")


def write_csv(rows: List[Dict], filename: str) -> None:
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure CosmicGrain z=0 oxygen abundances for comparison to SAGAbg."
    )
    ap.add_argument("--root", default=".", help="gadget4 root containing halo*_output_1024")
    ap.add_argument("--snap", type=int, default=29, help="final snapshot number (default: 29)")
    ap.add_argument("--aperture-pkpc", type=float, default=30.0)
    ap.add_argument("--only", nargs="*", default=None, help="optional labels, e.g. --only CG06 CG10")
    ap.add_argument("--output", default="cosmicgrain_sagabg_z0.csv")
    ap.add_argument("--no-refine-center", action="store_true")
    args = ap.parse_args()

    selected = set(args.only) if args.only else set(CG_HALOS)
    unknown = selected - set(CG_HALOS)
    if unknown:
        ap.error("unknown CG labels: " + ", ".join(sorted(unknown)))

    rows: List[Dict] = []

    print("=" * 76)
    print("COSMICGRAIN -- SAGAbg OXYGEN COMPARISON")
    print("=" * 76)
    print(f"snapshot              = {args.snap:03d}")
    print(f"galaxy aperture       = {args.aperture_pkpc:g} pkpc")
    print(f"hydrogen mass frac    = {X_H}")
    print("target selection      = group index from post-run audit JSON")
    print("primary abundance     = SFR-weighted linear O/H")

    for cg_name, halo_id in CG_HALOS.items():
        if cg_name not in selected:
            continue

        output_dir = os.path.join(args.root, f"halo{halo_id}_output_1024")
        snap_pattern = os.path.join(
            output_dir, f"snapdir_{args.snap:03d}", f"snapshot_{args.snap:03d}.*.hdf5"
        )
        snap_single = os.path.join(output_dir, f"snapshot_{args.snap:03d}.hdf5")
        has_snapshot = bool(glob.glob(snap_pattern)) or os.path.isfile(snap_single)

        if not has_snapshot:
            print(f"\n{cg_name}: SKIP -- snapshot {args.snap:03d} not found")
            continue

        try:
            group_index, audit_file = get_group_index_from_audit(output_dir)
        except Exception as exc:
            print(f"\n{cg_name}: SKIP -- could not determine audited target group: {exc}")
            continue

        try:
            row = analyze_halo(
                cg_name=cg_name,
                halo_id=halo_id,
                output_dir=output_dir,
                group_index=group_index,
                snap=args.snap,
                aperture_pkpc=args.aperture_pkpc,
                refine_center=not args.no_refine_center,
            )
            row["audit_file"] = audit_file
            rows.append(row)
            print_halo(row)
        except Exception as exc:
            print(f"\n{cg_name}: ERROR -- {type(exc).__name__}: {exc}")

    if not rows:
        print("\nNo completed halos were analyzed.")
        return 1

    write_csv(rows, args.output)

    print()
    print("=" * 76)
    print("SUITE SUMMARY")
    print("=" * 76)
    print(f"Successfully analyzed: {len(rows)} / {len(selected)} selected halos")
    print()
    print(
        f"{'CG':<5s} {'grp':>4s} {'logMstar':>9s} {'M200':>10s} {'SFR':>11s} "
        f"{'O/H all':>9s} {'O/H SF':>9s} {'O/H SFR':>9s} {'N_SF':>7s} {'target':>8s}"
    )
    print("-" * 101)

    for row in rows:
        print(
            f"{row['CG']:<5s} "
            f"{row['group_index']:4d} "
            f"{row['log10_Mstar_Msun']:9.3f} "
            f"{row['M200c_Msun']:10.3e} "
            f"{row['SFR_Msun_yr']:11.3e} "
            f"{row['OH12_allgas_massweighted']:9.3f} "
            f"{row['OH12_SFgas_massweighted']:9.3f} "
            f"{row['OH12_SFRweighted']:9.3f} "
            f"{row['N_SFgas_30pkpc']:7d} "
            f"{'PASS' if row['target_mass_sanity'] else 'WARNING':>8s}"
        )

    print()
    print(f"Wrote: {args.output}")

    bad_targets = [r for r in rows if not r["target_mass_sanity"]]
    if bad_targets:
        print()
        print("WARNING: target-mass sanity check failed for:")
        for row in bad_targets:
            print(
                f"  {row['CG']}: M200={row['M200c_Msun']:.3e}, "
                f"parent={row['parent_M200c_Msun']:.3e}, "
                f"ratio={row['zoom_to_parent_M200_ratio']:.3f}"
            )
        print("Verify these audit target selections before scientific use.")

    commits = {row["git_commit"] for row in rows}
    if len(commits) == 1:
        print(f"Git consistency: PASS -- all analyzed halos use {next(iter(commits))}")
    else:
        print()
        print("WARNING: multiple Git commits occur among analyzed halos:")
        for row in rows:
            print(f"  {row['CG']}: {row['git_commit']}  {row['git_date']}")
        print(
            "Different commits are not automatically invalid, but verify that all "
            "use the intended chemistry implementation."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
