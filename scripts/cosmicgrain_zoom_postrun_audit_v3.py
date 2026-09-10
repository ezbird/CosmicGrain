#!/usr/bin/env python3
"""Standardized acceptance audit for completed CosmicGrain zoom simulations.

The script checks the complete snapshot/catalog cadence, audits the final halo,
galaxy, dust population, contamination, and numerical integrity, and writes:

  <output_dir>/<run_name>_postrun_audit.txt
  <output_dir>/<run_name>_postrun_audit.json
  <output_dir>/<run_name>_paper_row.csv

It intentionally delegates halo selection, center refinement, and particle-SO
M200c/R200c to the unified halo_utils.py so every suite script uses one halo
definition.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import re
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from halo_utils import find_snapshot_and_group_files, get_zoom_halo, periodic_delta


PTYPES = (0, 1, 2, 4, 6)
PTYPE_NAMES = {0: "gas", 1: "hr_dm", 2: "lr_dm", 4: "stars", 6: "dust"}
Z_FIELDS = ("Metallicity", "GFM_Metallicity", "Z")
SFR_FIELDS = ("StarFormationRate", "SFR")
DUST_RADIUS_FIELDS = ("GrainRadius", "DustGrainRadius", "GRAD")
DUST_CF_FIELDS = ("CarbonMassFraction", "CarbonFraction", "DustCarbonFraction", "CFRC")
DUST_SOURCE_FIELDS = ("DustSource", "Source", "DustSourceType", "DSRC")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def first_present(names: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    names = set(names)
    return next((x for x in candidates if x in names), None)


def discover_indices(output_dir: str, kind: str) -> List[int]:
    if kind == "snapshot":
        patterns = ("snapdir_*", "snapshot_*.hdf5", "snapshot_.*.hdf5")
        regex = re.compile(r"(?:snapdir|snapshot)_(\d+)")
    else:
        patterns = ("groups_*", "fof_subhalo_tab_*.hdf5", "fof_subhalo_tab_.*.hdf5")
        regex = re.compile(r"(?:groups|fof_subhalo_tab)_(\d+)")
    found = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            match = regex.search(os.path.basename(path))
            if match:
                found.add(int(match.group(1)))
    return sorted(found)


def missing_from_contiguous(indices: Sequence[int]) -> List[int]:
    if not indices:
        return []
    return sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))


def snapshot_header(snapshot_files: Sequence[str]) -> Dict[str, Any]:
    with h5py.File(snapshot_files[0], "r") as f:
        attrs = f["Header"].attrs
        params = f["Parameters"].attrs
        return {
            "a": float(attrs["Time"]),
            "z": float(attrs.get("Redshift", 1.0 / float(attrs["Time"]) - 1.0)),
            "h": float(params["HubbleParam"]),
            "box": float(attrs["BoxSize"]),
            "mass_table": np.asarray(attrs["MassTable"], dtype=float),
            "num_total": np.asarray(attrs.get("NumPart_Total", np.zeros(6)), dtype=np.uint64),
        }


def cadence_audit(output_dir: str, snap_indices: Sequence[int]) -> Dict[str, Any]:
    rows, errors = [], []
    previous_a = -np.inf
    for snap in snap_indices:
        try:
            snapshot_files, group_files = find_snapshot_and_group_files(output_dir, snap)
            hdr = snapshot_header(snapshot_files)
            monotonic = hdr["a"] > previous_a
            previous_a = hdr["a"]
            if not monotonic:
                errors.append(f"snapshot {snap:03d}: scale factor is not increasing")
            rows.append({"snap": snap, "a": hdr["a"], "z": hdr["z"],
                         "snapshot_files": len(snapshot_files),
                         "group_files": len(group_files), "time_monotonic": monotonic})
        except Exception as exc:
            errors.append(f"snapshot {snap:03d}: {exc}")
    return {"rows": rows, "errors": errors}


def read_type(snapshot_files: Sequence[str], pt: int, extra_fields: Sequence[str] = ()):
    coords, masses, ids = [], [], []
    extras: Dict[str, List[np.ndarray]] = {x: [] for x in extra_fields}
    available = set()
    mt = None
    for filename in snapshot_files:
        with h5py.File(filename, "r") as f:
            if mt is None:
                mt = np.asarray(f["Header"].attrs["MassTable"], dtype=float)
            name = f"PartType{pt}"
            if name not in f:
                continue
            group = f[name]
            available.update(group.keys())
            c = np.asarray(group["Coordinates"], dtype=float)
            coords.append(c)
            masses.append(np.asarray(group["Masses"], dtype=float) if "Masses" in group
                          else np.full(len(c), mt[pt], dtype=float))
            if "ParticleIDs" in group:
                ids.append(np.asarray(group["ParticleIDs"]))
            for field in extra_fields:
                if field in group:
                    extras[field].append(np.asarray(group[field]))
    if not coords:
        return None
    return {
        "coords": np.concatenate(coords),
        "masses": np.concatenate(masses),
        "ids": np.concatenate(ids) if ids else None,
        "fields": {k: np.concatenate(v) for k, v in extras.items() if v},
        "available": sorted(available),
    }


def radial_mask(coords: np.ndarray, center: np.ndarray, box: float, radius: float):
    delta = periodic_delta(coords, center, box)
    r = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return r, r < radius


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    q = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[q], weights=weights[q])) if np.any(q) else float("nan")


def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                        percentiles=(5.0, 50.0, 95.0)) -> List[float]:
    q = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(q):
        return [float("nan")] * len(percentiles)
    v, w = values[q], weights[q]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return [float(np.interp(p / 100.0, cdf, v)) for p in percentiles]


def issue(bucket: List[Dict[str, str]], severity: str, check: str, detail: str):
    bucket.append({"severity": severity, "check": check, "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standardized 30-snapshot CosmicGrain zoom acceptance audit")
    ap.add_argument("output_dir", help="Completed GADGET-4 output directory")
    ap.add_argument("--snap", type=int, default=None,
                    help="Final snapshot (default: highest discovered snapshot)")
    ap.add_argument("--group-index", type=int, required=True,
                    help="Explicit FOF group index identified from the shifted parent target")
    ap.add_argument("--galaxy-aperture-pkpc", type=float, default=30.0)
    ap.add_argument("--expected-snapshots", type=int, default=30)
    ap.add_argument("--parent-m200-msun", type=float, default=None,
                    help="Selected parent-box M200c for parent-versus-zoom comparison")
    ap.add_argument("--run-name", default=None,
                    help="Report stem (default: basename of output_dir)")
    ap.add_argument("--output-prefix", default=None,
                    help="Override report path prefix")
    ap.add_argument("--source-map", default="0:SNII,1:AGB,2:LRN",
                    help="Dust source integer labels")
    ap.add_argument("--no-refine-center", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="Print only")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    run_name = args.run_name or os.path.basename(output_dir.rstrip(os.sep))
    prefix = args.output_prefix or os.path.join(output_dir, f"{run_name}_postrun_audit")
    source_map = {}
    try:
        for item in args.source_map.split(","):
            key, label = item.split(":", 1)
            source_map[int(key.strip())] = label.strip()
    except Exception as exc:
        ap.error(f"invalid --source-map: {exc}")

    snapshots = discover_indices(output_dir, "snapshot")
    catalogs = discover_indices(output_dir, "catalog")
    if not snapshots:
        ap.error(f"no snapshots found under {output_dir}")
    final_snap = args.snap if args.snap is not None else snapshots[-1]
    issues: List[Dict[str, str]] = []

    missing_snaps = missing_from_contiguous(snapshots)
    missing_cats = missing_from_contiguous(catalogs)
    if len(snapshots) != args.expected_snapshots:
        issue(issues, "FAIL", "snapshot_count",
              f"found {len(snapshots)}, expected {args.expected_snapshots}")
    if len(catalogs) != args.expected_snapshots:
        issue(issues, "FAIL", "catalog_count",
              f"found {len(catalogs)}, expected {args.expected_snapshots}")
    if snapshots != catalogs:
        issue(issues, "FAIL", "snapshot_catalog_pairing", "snapshot and catalog indices differ")
    if missing_snaps or missing_cats:
        issue(issues, "FAIL", "contiguous_cadence",
              f"missing snapshots={missing_snaps}; missing catalogs={missing_cats}")

    cadence = cadence_audit(output_dir, sorted(set(snapshots) & set(catalogs)))
    for error in cadence["errors"]:
        issue(issues, "FAIL", "cadence_read", error)

    snapshot_files, _ = find_snapshot_and_group_files(output_dir, final_snap)
    final_header = snapshot_header(snapshot_files)
    halo = get_zoom_halo(output_dir, final_snap, group_index=args.group_index,
                         refine_center=not args.no_refine_center, verbose=False)
    center, r200, box = halo.chosen_center_ckpch, halo.so_r200_ckpch, halo.boxsize_ckpch
    a, h = halo.a, halo.h
    aperture = args.galaxy_aperture_pkpc * h / a
    mass_to_msun = 1.0e10 / h
    length_to_pkpc = a / h

    extra = {
        0: Z_FIELDS + SFR_FIELDS,
        1: (), 2: (), 4: Z_FIELDS,
        6: DUST_RADIUS_FIELDS + DUST_CF_FIELDS + DUST_SOURCE_FIELDS,
    }
    particles = {pt: read_type(snapshot_files, pt, extra[pt]) for pt in PTYPES}
    for required_pt in (0, 1, 4, 6):
        if particles[required_pt] is None:
            issue(issues, "FAIL", f"PartType{required_pt}_presence",
                  f"required final-snapshot {PTYPE_NAMES[required_pt]} population is absent")
    regions: Dict[int, Dict[str, Any]] = {}
    integrity: Dict[str, Any] = {}
    all_ids = []
    for pt, data in particles.items():
        if data is None:
            regions[pt] = {"present": False, "n_r200": 0, "mass_r200_msun": 0.0,
                           "n_aperture": 0, "mass_aperture_msun": 0.0}
            continue
        r, q200 = radial_mask(data["coords"], center, box, r200)
        _, qap = radial_mask(data["coords"], center, box, aperture)
        regions[pt] = {
            "present": True, "r_ckpch": r, "q_r200": q200, "q_aperture": qap,
            "n_total": len(data["masses"]), "mass_total_msun": float(data["masses"].sum() * mass_to_msun),
            "n_r200": int(q200.sum()), "mass_r200_msun": float(data["masses"][q200].sum() * mass_to_msun),
            "n_aperture": int(qap.sum()), "mass_aperture_msun": float(data["masses"][qap].sum() * mass_to_msun),
        }
        checks = {
            "nonfinite_coordinates": int(np.count_nonzero(~np.isfinite(data["coords"]))),
            "nonfinite_masses": int(np.count_nonzero(~np.isfinite(data["masses"]))),
            "negative_masses": int(np.count_nonzero(data["masses"] < 0)),
            "nonpositive_ids": int(np.count_nonzero(data["ids"] <= 0)) if data["ids"] is not None else None,
        }
        integrity[f"PartType{pt}"] = checks
        if any(v for v in checks.values() if v is not None):
            issue(issues, "FAIL", f"PartType{pt}_basic_integrity", str(checks))
        if data["ids"] is not None:
            all_ids.append(data["ids"])
        header_count = (int(final_header["num_total"][pt])
                        if pt < len(final_header["num_total"]) else None)
        if header_count is not None:
            integrity[f"PartType{pt}"]["header_count"] = header_count
            integrity[f"PartType{pt}"]["loaded_count"] = len(data["masses"])
            if header_count != len(data["masses"]):
                issue(issues, "FAIL", f"PartType{pt}_header_count",
                      f"header={header_count}, loaded={len(data['masses'])}")
    if all_ids:
        ids = np.concatenate(all_ids)
        duplicate_ids = int(len(ids) - len(np.unique(ids)))
        integrity["duplicate_particle_ids"] = duplicate_ids
        if duplicate_ids:
            issue(issues, "FAIL", "unique_particle_ids", f"{duplicate_ids} duplicates")

    gas = particles[0]
    gas_q = regions[0].get("q_aperture", np.zeros(0, dtype=bool))
    gas_mass = gas["masses"][gas_q] if gas is not None else np.array([])
    zfield = first_present(gas["fields"] if gas else (), Z_FIELDS)
    sfrfield = first_present(gas["fields"] if gas else (), SFR_FIELDS)
    if not zfield:
        issue(issues, "FAIL", "gas_metallicity_field", "field not found")
    if not sfrfield:
        issue(issues, "FAIL", "gas_sfr_field", "field not found")
    zgas = weighted_mean(gas["fields"][zfield][gas_q], gas_mass) if zfield else float("nan")
    sfr = float(np.sum(gas["fields"][sfrfield][gas_q])) if sfrfield else float("nan")
    gas_metal_mass = (float(np.sum(gas_mass * gas["fields"][zfield][gas_q]) * mass_to_msun)
                      if zfield else float("nan"))

    dust = particles[6]
    dust_q = regions[6].get("q_aperture", np.zeros(0, dtype=bool))
    dust_mass_code = dust["masses"][dust_q] if dust is not None else np.array([])
    mdust = float(dust_mass_code.sum() * mass_to_msun)
    mgas = regions[0]["mass_aperture_msun"]
    dg = mdust / mgas if mgas > 0 else float("nan")
    dz = mdust / gas_metal_mass if gas_metal_mass > 0 else float("nan")
    dust_result: Dict[str, Any] = {"mass_msun": mdust, "count": int(dust_q.sum()),
                                   "dust_to_gas": dg, "dust_to_gas_metals": dz}
    if dust is not None:
        radius_field = first_present(dust["fields"], DUST_RADIUS_FIELDS)
        cf_field = first_present(dust["fields"], DUST_CF_FIELDS)
        source_field = first_present(dust["fields"], DUST_SOURCE_FIELDS)
        if radius_field:
            radii = np.asarray(dust["fields"][radius_field][dust_q], dtype=float)
            dust_result.update({"radius_field": radius_field,
                                "radius_number_p05_p50_p95": np.percentile(radii[np.isfinite(radii)], [5, 50, 95]).tolist() if np.any(np.isfinite(radii)) else [None]*3,
                                "radius_mass_p05_p50_p95": weighted_percentile(radii, dust_mass_code)})
            bad_radius = int(np.count_nonzero(~np.isfinite(radii) | (radii <= 0)))
            integrity["dust_bad_radius"] = bad_radius
            if bad_radius:
                issue(issues, "FAIL", "dust_radius", f"{bad_radius} nonfinite/nonpositive values")
        else:
            issue(issues, "FAIL", "dust_radius_field", "field not found")
        if cf_field:
            cf = np.asarray(dust["fields"][cf_field][dust_q], dtype=float)
            bad_cf = int(np.count_nonzero(~np.isfinite(cf) | (cf < 0) | (cf > 1)))
            integrity["dust_bad_carbon_fraction"] = bad_cf
            dust_result.update({"carbon_fraction_field": cf_field,
                                "carbon_mass_msun": float(np.sum(dust_mass_code * cf) * mass_to_msun),
                                "silicate_mass_msun": float(np.sum(dust_mass_code * (1.0 - cf)) * mass_to_msun),
                                "carbon_mass_fraction": weighted_mean(cf, dust_mass_code)})
            if bad_cf:
                issue(issues, "FAIL", "dust_composition_bounds", f"{bad_cf} values outside [0,1]")
        else:
            issue(issues, "FAIL", "dust_carbon_fraction_field", "field not found")
        if source_field:
            src = np.asarray(dust["fields"][source_field][dust_q]).reshape(-1)
            source_rows = {}
            finite_integer = np.isfinite(src) & (src == np.floor(src))
            invalid = int(np.count_nonzero(~finite_integer))
            for value in np.unique(src[finite_integer]):
                ivalue = int(value)
                q = src == value
                label = source_map.get(ivalue, f"UNKNOWN_{ivalue}")
                if ivalue not in source_map:
                    invalid += int(q.sum())
                source_rows[label] = {"code": ivalue, "count": int(q.sum()),
                                      "mass_msun": float(dust_mass_code[q].sum() * mass_to_msun),
                                      "mass_fraction": float(dust_mass_code[q].sum()/dust_mass_code.sum()) if dust_mass_code.sum() > 0 else None}
            dust_result.update({"source_field": source_field, "sources": source_rows})
            integrity["dust_invalid_source"] = invalid
            if invalid:
                issue(issues, "FAIL", "dust_source_validity", f"{invalid} unknown source codes")
        else:
            issue(issues, "FAIL", "dust_source_field", "field not found")

    if abs(float(halo.z)) > 1.0e-4:
        issue(issues, "FAIL", "final_redshift", f"final snapshot is z={halo.z:.6g}, expected z=0")

    contamination = {"present": particles[2] is not None, "mass_levels": []}
    if particles[2] is not None:
        rlr, mlr = regions[2]["r_ckpch"], particles[2]["masses"]
        contamination["nearest_r200"] = float(rlr.min() / r200) if len(rlr) else None
        contamination["nearest_pkpc"] = float(rlr.min() * length_to_pkpc) if len(rlr) else None
        contamination["n_within_1r200"] = int(np.count_nonzero(rlr < r200))
        contamination["n_within_2r200"] = int(np.count_nonzero(rlr < 2*r200))
        contamination["n_within_3r200"] = int(np.count_nonzero(rlr < 3*r200))
        for level in np.unique(mlr):
            q = np.isclose(mlr, level, rtol=1e-7, atol=0)
            contamination["mass_levels"].append({"particle_mass_msun": float(level*mass_to_msun),
                "count": int(q.sum()), "n_1r200": int(np.count_nonzero(rlr[q] < r200)),
                "n_2r200": int(np.count_nonzero(rlr[q] < 2*r200)),
                "n_3r200": int(np.count_nonzero(rlr[q] < 3*r200)),
                "nearest_r200": float(rlr[q].min()/r200)})
        if contamination["n_within_1r200"]:
            issue(issues, "FAIL", "low_resolution_contamination",
                  f"{contamination['n_within_1r200']} PartType2 particles within R200")

    parent_delta = None
    m200 = float(halo.so_m200_code * mass_to_msun)
    if args.parent_m200_msun:
        parent_delta = 100.0 * (m200 / args.parent_m200_msun - 1.0)

    result = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name, "output_dir": output_dir,
        "settings": {"expected_snapshots": args.expected_snapshots, "final_snap": final_snap,
                     "group_index_requested": args.group_index,
                     "galaxy_aperture_pkpc": args.galaxy_aperture_pkpc,
                     "refine_center": not args.no_refine_center,
                     "dust_source_map": source_map},
        "cadence": {"snapshot_count": len(snapshots), "catalog_count": len(catalogs),
                    "snapshot_indices": snapshots, "catalog_indices": catalogs,
                    "missing_snapshot_indices": missing_snaps, "missing_catalog_indices": missing_cats,
                    **cadence},
        "halo": {"group_index": int(halo.group_index), "z": float(halo.z), "a": float(a), "h": float(h),
                 "catalog_center_ckpch": halo.catalog_center_ckpch, "refined_center_ckpch": halo.refined_center_ckpch,
                 "chosen_center_ckpch": center, "refinement_accepted": bool(halo.refinement_accepted),
                 "refinement_shift_pkpc": float(halo.refinement_shift_ckpch*length_to_pkpc),
                 "refinement_shift_catalog_r200": float(halo.refinement_shift_ckpch/halo.catalog_r200_ckpch),
                 "catalog_m200_msun": float(halo.catalog_m200_code*mass_to_msun), "particle_so_m200_msun": m200,
                 "catalog_r200_pkpc": float(halo.catalog_r200_ckpch*length_to_pkpc),
                 "particle_so_r200_pkpc": float(r200*length_to_pkpc),
                 "particle_vs_catalog_m200_percent": float(100*(halo.so_m200_code/halo.catalog_m200_code-1)),
                 "parent_m200_msun": args.parent_m200_msun, "zoom_vs_parent_m200_percent": parent_delta},
        "galaxy": {"aperture_pkpc": args.galaxy_aperture_pkpc,
                   "stellar_mass_msun": regions[4]["mass_aperture_msun"],
                   "gas_mass_msun": mgas, "sfr_msun_per_yr": sfr,
                   "gas_metallicity_mass_fraction": zgas, "gas_metallicity_field": zfield,
                   "sfr_field": sfrfield,
                   "particle_counts": {PTYPE_NAMES[p]: regions[p]["n_aperture"] for p in PTYPES}},
        "dust": dust_result, "contamination": contamination,
        "particles": {PTYPE_NAMES[p]: {k: v for k, v in regions[p].items() if k not in ("r_ckpch", "q_r200", "q_aperture")} for p in PTYPES},
        "integrity": integrity, "issues": issues,
    }
    fails = sum(x["severity"] == "FAIL" for x in issues)
    warns = sum(x["severity"] == "WARN" for x in issues)
    result["acceptance"] = {"status": "PASS" if fails == 0 else "FAIL", "failures": fails, "warnings": warns}
    result = jsonable(result)

    out = io.StringIO()
    with redirect_stdout(out):
        print("=" * 88)
        print("COSMICGRAIN STANDARDIZED ZOOM POST-RUN ACCEPTANCE AUDIT")
        print("=" * 88)
        print(f"Run: {run_name} | final snapshot: {final_snap:03d} | z={halo.z:.6f} | group={halo.group_index}")
        print(f"Overall: {result['acceptance']['status']} ({fails} failures, {warns} warnings)")
        print("\n--- CADENCE ---")
        print(f"Snapshots: {len(snapshots)}/{args.expected_snapshots}; catalogs: {len(catalogs)}/{args.expected_snapshots}; paired={snapshots == catalogs}")
        print(f"Snapshot range: {snapshots[0]:03d}..{snapshots[-1]:03d}; missing={missing_snaps or 'none'}")
        print(f"Catalog range:  {catalogs[0]:03d}..{catalogs[-1]:03d}; missing={missing_cats or 'none'}")
        print("\n--- HALO ---")
        print(f"Catalog center: {np.asarray(halo.catalog_center_ckpch)} ckpc/h")
        print(f"Refined center: {np.asarray(halo.refined_center_ckpch)} ckpc/h; accepted={halo.refinement_accepted}")
        print(f"Center shift: {halo.refinement_shift_ckpch*length_to_pkpc:.3f} pkpc = {halo.refinement_shift_ckpch/halo.catalog_r200_ckpch:.4f} catalog R200")
        print(f"M200c catalog / particle-SO: {halo.catalog_m200_code*mass_to_msun:.6e} / {m200:.6e} Msun")
        print(f"R200c catalog / particle-SO: {halo.catalog_r200_ckpch*length_to_pkpc:.3f} / {r200*length_to_pkpc:.3f} pkpc")
        if args.parent_m200_msun:
            print(f"M200c parent / zoom difference: {parent_delta:+.2f}%")
        else:
            print("M200c parent / zoom difference: not supplied")
        print(f"Nearest low-resolution particle: {contamination.get('nearest_r200', 'n/a')} R200")
        print(f"Low-resolution counts <1/<2/<3 R200: {contamination.get('n_within_1r200', 0)}/{contamination.get('n_within_2r200', 0)}/{contamination.get('n_within_3r200', 0)}")
        print("\n--- GALAXY ---")
        print(f"Aperture: {args.galaxy_aperture_pkpc:g} pkpc")
        print(f"Mstar={regions[4]['mass_aperture_msun']:.6e} Msun  Mgas={mgas:.6e} Msun  SFR={sfr:.6e} Msun/yr")
        print(f"Zgas={zgas:.6e} ({zfield or 'field absent'})")
        print("Counts: " + ", ".join(f"{PTYPE_NAMES[p]}={regions[p]['n_aperture']}" for p in PTYPES))
        print("\n--- DUST ---")
        print(f"Mdust={mdust:.6e} Msun  D/G={dg:.6e}  D/Z={dz:.6e}  Ndust={int(dust_q.sum())}")
        if result["dust"].get("radius_mass_p05_p50_p95"):
            print(f"Grain radius mass-weighted p05/p50/p95: {result['dust']['radius_mass_p05_p50_p95']}")
        if result["dust"].get("carbon_mass_fraction") is not None:
            print(f"Carbon/silicate mass fractions: {result['dust']['carbon_mass_fraction']:.6f} / {1-result['dust']['carbon_mass_fraction']:.6f}")
        for label, row in result["dust"].get("sources", {}).items():
            print(f"{label}: N={row['count']} M={row['mass_msun']:.6e} Msun fraction={row['mass_fraction']:.6f}")
        print("\n--- INTEGRITY / ACCEPTANCE ---")
        if issues:
            for row in issues:
                print(f"[{row['severity']}] {row['check']}: {row['detail']}")
        else:
            print("[PASS] cadence, contamination, particle values, composition, and source checks")
        print("\nRESULT: " + result["acceptance"]["status"])
        print("=" * 88)
    report_text = out.getvalue()
    print(report_text, end="")

    paper_row = {
        "run_name": run_name, "group_index": halo.group_index, "z": halo.z,
        "M200c_Msun": m200, "R200c_pkpc": r200*length_to_pkpc,
        "Mstar_Msun": regions[4]["mass_aperture_msun"], "Mgas_Msun": mgas,
        "SFR_Msun_yr": sfr, "Zgas_mass_fraction": zgas, "Mdust_Msun": mdust,
        "dust_to_gas": dg, "dust_to_gas_metals": dz,
        "nearest_lowres_R200": contamination.get("nearest_r200"),
        "N_lowres_within_R200": contamination.get("n_within_1r200", 0),
        "acceptance": result["acceptance"]["status"],
    }
    if not args.no_write:
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        with open(prefix + ".txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        csv_path = prefix.replace("_postrun_audit", "") + "_paper_row.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paper_row))
            writer.writeheader(); writer.writerow(jsonable(paper_row))
        print(f"Wrote {prefix}.txt")
        print(f"Wrote {prefix}.json")
        print(f"Wrote {csv_path}")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
