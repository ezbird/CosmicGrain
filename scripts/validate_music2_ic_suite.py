#!/usr/bin/env python3
"""
Validate the complete CosmicGrain MUSIC2 zoom-IC suite.

The default suite contains 12 halos at four effective resolutions.  The
validator understands the post-processed seven-type GADGET header and the
intentionally empty PartType6 dust group.

Checks performed for every file include:
  * expected HDF5 groups, datasets, shapes, and header particle counts
  * finite coordinates, velocities, and masses, scanned in chunks
  * coordinates inside the periodic box and positive particle masses
  * an exact global ParticleID uniqueness check (default)
  * gas/high-resolution-DM spatial overlap
  * the high-resolution DM-to-gas particle-mass ratio
  * a valid empty PartType6 dust placeholder

The suite-level checks compare header cosmology between all files and verify
that high-resolution particle masses scale by approximately eight whenever
the nominal linear resolution doubles.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


HALOS = (295, 308, 441, 859, 1481, 1534, 3352, 3879, 3886, 5834, 7723, 9235)
RESOLUTIONS = (512, 1024, 2048, 4096)
NPART_TYPES = 7
CORE_DATASETS = ("Coordinates", "Velocities", "ParticleIDs")
MASS_SCALING_RTOL = 0.03
COSMIC_RATIO_RTOL = 0.01


@dataclass
class FileResult:
    halo: int
    resolution: int
    path: Path
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: np.ndarray = field(
        default_factory=lambda: np.zeros(NPART_TYPES, dtype=np.uint64)
    )
    total_particles: int = 0
    box_size: float = math.nan
    time: float = math.nan
    redshift: float = math.nan
    omega0: float = math.nan
    omega_lambda: float = math.nan
    omega_baryon: float = math.nan
    hubble_param: float = math.nan
    gas_mass: float = math.nan
    hrdm_mass: float = math.nan
    type2_mass_min: float = math.nan
    type2_mass_max: float = math.nan
    dm_gas_ratio: float = math.nan
    expected_dm_gas_ratio: float = math.nan
    id_unique: str = "not checked"
    finite: bool = True
    bounds_ok: bool = True
    type6_valid: bool = False
    overlap_x: float = math.nan
    overlap_y: float = math.nan
    overlap_z: float = math.nan

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "FAIL"

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "PASS":
            self.status = "WARN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all 48 CosmicGrain MUSIC2 zoom IC files."
    )
    parser.add_argument(
        "--ic-root",
        type=Path,
        default=Path("~/gadget4/ICs").expanduser(),
        help="Root containing halo<ID>/IC_halo<ID>_zoom_<RES>.hdf5",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV destination (default: IC_ROOT/MUSIC2_logs/ic_suite_validation.csv)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows read per numerical scan chunk (default: 1000000)",
    )
    parser.add_argument(
        "--id-check",
        choices=("exact", "none"),
        default="exact",
        help="Exact global ID uniqueness or skip it (default: exact)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure if any warning is produced.",
    )
    return parser.parse_args()


def scalar(attrs: h5py.AttributeManager, name: str, default: float = math.nan) -> float:
    if name not in attrs:
        return default
    value = np.asarray(attrs[name]).reshape(-1)
    return float(value[0]) if value.size else default


def header_counts(attrs: h5py.AttributeManager, result: FileResult) -> np.ndarray:
    if "NumPart_Total" not in attrs:
        result.error("Header is missing NumPart_Total")
        return np.zeros(NPART_TYPES, dtype=np.uint64)

    low = np.asarray(attrs["NumPart_Total"], dtype=np.uint64).reshape(-1)
    high = np.asarray(
        attrs.get("NumPart_Total_HighWord", np.zeros_like(low)), dtype=np.uint64
    ).reshape(-1)
    if len(low) != NPART_TYPES:
        result.error(f"NumPart_Total has length {len(low)}, expected {NPART_TYPES}")
    if len(high) != len(low):
        result.error("NumPart_Total_HighWord length differs from NumPart_Total")
        high = np.zeros_like(low)

    combined = low + (high << np.uint64(32))
    padded = np.zeros(NPART_TYPES, dtype=np.uint64)
    padded[: min(NPART_TYPES, len(combined))] = combined[:NPART_TYPES]

    if "NumPart_ThisFile" not in attrs:
        result.error("Header is missing NumPart_ThisFile")
    else:
        this_file = np.asarray(attrs["NumPart_ThisFile"], dtype=np.uint64).reshape(-1)
        if len(this_file) != NPART_TYPES:
            result.error(
                f"NumPart_ThisFile has length {len(this_file)}, expected {NPART_TYPES}"
            )
        elif not np.array_equal(this_file, padded):
            result.error("NumPart_ThisFile differs from NumPart_Total")

    mass_table = np.asarray(
        attrs.get("MassTable", np.zeros(NPART_TYPES)), dtype=np.float64
    ).reshape(-1)
    if len(mass_table) != NPART_TYPES:
        result.error(f"MassTable has length {len(mass_table)}, expected {NPART_TYPES}")

    return padded


def iter_slices(size: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, size, chunk_size):
        yield slice(start, min(start + chunk_size, size))


def scan_vectors(
    dataset: h5py.Dataset,
    result: FileResult,
    label: str,
    chunk_size: int,
    check_bounds: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for section in iter_slices(dataset.shape[0], chunk_size):
        values = np.asarray(dataset[section])
        if not np.all(np.isfinite(values)):
            result.finite = False
            result.error(f"{label} contains non-finite values")
            continue
        low = np.minimum(low, np.min(values, axis=0))
        high = np.maximum(high, np.max(values, axis=0))
        if check_bounds and (
            np.any(values < 0.0) or np.any(values >= result.box_size)
        ):
            result.bounds_ok = False
            result.error(f"{label} has coordinates outside [0, BoxSize)")
    return low, high


def scan_masses(
    dataset: h5py.Dataset,
    result: FileResult,
    label: str,
    chunk_size: int,
) -> tuple[float, float]:
    minimum = np.inf
    maximum = -np.inf
    for section in iter_slices(dataset.shape[0], chunk_size):
        values = np.asarray(dataset[section], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            result.finite = False
            result.error(f"{label} contains non-finite values")
            continue
        if np.any(values <= 0.0):
            result.error(f"{label} contains non-positive values")
        minimum = min(minimum, float(np.min(values)))
        maximum = max(maximum, float(np.max(values)))
    return minimum, maximum


def representative_mass(
    handle: h5py.File,
    ptype: int,
    mass_table: np.ndarray,
    result: FileResult,
    chunk_size: int,
) -> tuple[float, float, float]:
    count = int(result.counts[ptype])
    if count == 0:
        return math.nan, math.nan, math.nan
    if ptype < len(mass_table) and mass_table[ptype] > 0.0:
        value = float(mass_table[ptype])
        return value, value, value
    group = handle[f"PartType{ptype}"]
    if "Masses" not in group:
        result.error(f"PartType{ptype} has no Masses dataset and zero MassTable entry")
        return math.nan, math.nan, math.nan
    minimum, maximum = scan_masses(
        group["Masses"], result, f"PartType{ptype}/Masses", chunk_size
    )
    representative = 0.5 * (minimum + maximum)
    return representative, minimum, maximum


def validate_type6(handle: h5py.File, result: FileResult) -> None:
    if "PartType6" not in handle:
        result.error("PartType6 dust placeholder is missing")
        return
    group = handle["PartType6"]
    expected = {
        "Coordinates": (0, 3),
        "Velocities": (0, 3),
        "ParticleIDs": (0,),
        "Masses": (0,),
    }
    valid = True
    for name, shape in expected.items():
        if name not in group:
            result.error(f"PartType6 is missing {name}")
            valid = False
        elif tuple(group[name].shape) != shape:
            result.error(
                f"PartType6/{name} shape is {group[name].shape}, expected {shape}"
            )
            valid = False
    result.type6_valid = valid


def check_exact_ids(
    handle: h5py.File, result: FileResult, chunk_size: int
) -> None:
    arrays: list[np.ndarray] = []
    for ptype in range(NPART_TYPES):
        group_name = f"PartType{ptype}"
        if group_name not in handle or result.counts[ptype] == 0:
            continue
        dataset = handle[group_name]["ParticleIDs"]
        for section in iter_slices(dataset.shape[0], chunk_size):
            ids = np.asarray(dataset[section], dtype=np.uint64)
            arrays.append(ids)
    if not arrays:
        result.error("No particle IDs were found")
        result.id_unique = "no"
        return
    all_ids = np.concatenate(arrays)
    if np.any(all_ids == 0):
        result.warn("ParticleIDs include zero")
    unique_count = int(np.unique(all_ids).size)
    if unique_count != len(all_ids):
        result.error(
            f"ParticleIDs are not globally unique: {len(all_ids)-unique_count} duplicates"
        )
        result.id_unique = "no"
    else:
        result.id_unique = "yes"


def validate_file(
    path: Path,
    halo: int,
    resolution: int,
    chunk_size: int,
    id_check: str,
) -> FileResult:
    result = FileResult(halo=halo, resolution=resolution, path=path)
    if not path.is_file():
        result.error("File is missing")
        return result

    try:
        with h5py.File(path, "r") as handle:
            if "Header" not in handle:
                result.error("Header group is missing")
                return result
            attrs = handle["Header"].attrs
            result.counts = header_counts(attrs, result)
            result.total_particles = int(np.sum(result.counts))
            result.box_size = scalar(attrs, "BoxSize")
            result.time = scalar(attrs, "Time")
            result.redshift = scalar(attrs, "Redshift")
            result.omega0 = scalar(attrs, "Omega0")
            result.omega_lambda = scalar(attrs, "OmegaLambda")
            result.omega_baryon = scalar(attrs, "OmegaBaryon")
            result.hubble_param = scalar(attrs, "HubbleParam")

            if scalar(attrs, "NumFilesPerSnapshot") != 1.0:
                result.error("NumFilesPerSnapshot is not 1")
            if not np.isfinite(result.box_size) or result.box_size <= 0.0:
                result.error("BoxSize is missing or non-positive")
            if not np.isfinite(result.time) or result.time <= 0.0:
                result.error("Time is missing or non-positive")
            if np.isfinite(result.time) and np.isfinite(result.redshift):
                expected_redshift = 1.0 / result.time - 1.0
                if not np.isclose(result.redshift, expected_redshift, rtol=1e-8, atol=1e-8):
                    result.error("Time and Redshift are inconsistent")

            mass_table = np.asarray(
                attrs.get("MassTable", np.zeros(NPART_TYPES)), dtype=np.float64
            ).reshape(-1)
            bboxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            actual_counts = np.zeros(NPART_TYPES, dtype=np.uint64)

            for ptype in range(NPART_TYPES):
                group_name = f"PartType{ptype}"
                expected_count = int(result.counts[ptype])
                if group_name not in handle:
                    if expected_count:
                        result.error(
                            f"{group_name} is missing but header count is {expected_count}"
                        )
                    continue
                group = handle[group_name]
                for name in CORE_DATASETS:
                    if name not in group:
                        result.error(f"{group_name} is missing {name}")
                if not all(name in group for name in CORE_DATASETS):
                    continue

                coordinates = group["Coordinates"]
                velocities = group["Velocities"]
                particle_ids = group["ParticleIDs"]
                actual_count = coordinates.shape[0]
                actual_counts[ptype] = actual_count
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    result.error(f"{group_name}/Coordinates is not shaped (N, 3)")
                if velocities.shape != coordinates.shape:
                    result.error(f"{group_name}/Velocities shape differs from Coordinates")
                if particle_ids.shape != (actual_count,):
                    result.error(f"{group_name}/ParticleIDs is not shaped (N,)")
                if actual_count != expected_count:
                    result.error(
                        f"{group_name} count {actual_count} differs from header {expected_count}"
                    )
                if actual_count and coordinates.ndim == 2 and coordinates.shape[1] == 3:
                    bboxes[ptype] = scan_vectors(
                        coordinates,
                        result,
                        f"{group_name}/Coordinates",
                        chunk_size,
                        check_bounds=True,
                    )
                    scan_vectors(
                        velocities,
                        result,
                        f"{group_name}/Velocities",
                        chunk_size,
                    )

            if not np.array_equal(actual_counts, result.counts):
                result.error("Summed dataset particle counts differ from the header")
            for required_type in (0, 1, 2):
                if result.counts[required_type] == 0:
                    result.error(f"Required PartType{required_type} is empty")
            for unused_type in (3, 4, 5):
                if result.counts[unused_type] != 0:
                    result.warn(f"Unexpected non-empty PartType{unused_type}")

            validate_type6(handle, result)

            gas, gas_min, gas_max = representative_mass(
                handle, 0, mass_table, result, chunk_size
            )
            hrdm, hrdm_min, hrdm_max = representative_mass(
                handle, 1, mass_table, result, chunk_size
            )
            _, type2_min, type2_max = representative_mass(
                handle, 2, mass_table, result, chunk_size
            )
            result.gas_mass = gas
            result.hrdm_mass = hrdm
            result.type2_mass_min = type2_min
            result.type2_mass_max = type2_max

            if np.isfinite(gas_min) and not np.isclose(
                gas_min, gas_max, rtol=1e-10, atol=0.0
            ):
                result.warn("PartType0 particle masses are not uniform")
            if np.isfinite(hrdm_min) and not np.isclose(
                hrdm_min, hrdm_max, rtol=1e-10, atol=0.0
            ):
                result.warn("PartType1 particle masses are not uniform")
            if np.isfinite(gas) and gas > 0.0 and np.isfinite(hrdm):
                result.dm_gas_ratio = hrdm / gas
                omega_b = result.omega_baryon
                if not np.isfinite(omega_b):
                    omega_b = 0.04936
                if np.isfinite(result.omega0) and omega_b > 0.0:
                    result.expected_dm_gas_ratio = (
                        result.omega0 - omega_b
                    ) / omega_b
                    if not np.isclose(
                        result.dm_gas_ratio,
                        result.expected_dm_gas_ratio,
                        rtol=COSMIC_RATIO_RTOL,
                        atol=0.0,
                    ):
                        result.error(
                            "High-resolution DM/gas mass ratio differs from cosmology"
                        )

            if (
                np.isfinite(type2_min)
                and np.isfinite(hrdm_max)
                and type2_min <= hrdm_max
            ):
                result.error("PartType2 minimum mass is not greater than HRDM mass")

            if 0 in bboxes and 1 in bboxes:
                overlap = np.minimum(bboxes[0][1], bboxes[1][1]) - np.maximum(
                    bboxes[0][0], bboxes[1][0]
                )
                result.overlap_x, result.overlap_y, result.overlap_z = map(
                    float, overlap
                )
                if np.any(overlap <= 0.0):
                    result.error("Gas and high-resolution DM bounding boxes do not overlap")

            if id_check == "exact":
                check_exact_ids(handle, result, chunk_size)
            else:
                result.id_unique = "skipped"
                result.warn("Exact global ParticleID uniqueness check was skipped")

    except (OSError, KeyError, ValueError) as exc:
        result.error(f"Could not validate HDF5 content: {exc}")
    return result


def compare_suite(results: list[FileResult]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    valid = [result for result in results if result.status != "FAIL"]
    if not valid:
        return ["No valid IC files were available for suite comparisons"], warnings

    reference = valid[0]
    fields = (
        ("BoxSize", "box_size"),
        ("Time", "time"),
        ("Redshift", "redshift"),
        ("Omega0", "omega0"),
        ("OmegaLambda", "omega_lambda"),
        ("HubbleParam", "hubble_param"),
    )
    for label, attribute in fields:
        expected = getattr(reference, attribute)
        for result in valid[1:]:
            value = getattr(result, attribute)
            if not (
                np.isfinite(expected)
                and np.isfinite(value)
                and np.isclose(value, expected, rtol=1e-10, atol=1e-12)
            ):
                errors.append(
                    f"{label} differs between {reference.path.name} and {result.path.name}"
                )

    by_key = {(result.halo, result.resolution): result for result in results}
    for halo in HALOS:
        previous: FileResult | None = None
        for resolution in RESOLUTIONS:
            current = by_key[(halo, resolution)]
            if previous is not None and all(
                np.isfinite(value)
                for value in (
                    previous.gas_mass,
                    current.gas_mass,
                    previous.hrdm_mass,
                    current.hrdm_mass,
                )
            ):
                for species, old_mass, new_mass in (
                    ("gas", previous.gas_mass, current.gas_mass),
                    ("HRDM", previous.hrdm_mass, current.hrdm_mass),
                ):
                    ratio = old_mass / new_mass
                    if not np.isclose(
                        ratio, 8.0, rtol=MASS_SCALING_RTOL, atol=0.0
                    ):
                        errors.append(
                            f"Halo {halo} {species} mass scaling "
                            f"{previous.resolution}->{resolution} is {ratio:.5g}, not 8"
                        )
                if current.total_particles <= previous.total_particles:
                    warnings.append(
                        f"Halo {halo} particle count did not increase from "
                        f"{previous.resolution} to {resolution}"
                    )
            previous = current

    for resolution in RESOLUTIONS:
        resolution_results = [
            by_key[(halo, resolution)]
            for halo in HALOS
            if np.isfinite(by_key[(halo, resolution)].gas_mass)
        ]
        if len(resolution_results) > 1:
            ref = resolution_results[0]
            for result in resolution_results[1:]:
                for species, reference_mass, mass in (
                    ("gas", ref.gas_mass, result.gas_mass),
                    ("HRDM", ref.hrdm_mass, result.hrdm_mass),
                ):
                    if not np.isclose(mass, reference_mass, rtol=1e-10, atol=0.0):
                        errors.append(
                            f"{species} mass differs across halos at resolution {resolution}"
                        )
                        break
    return sorted(set(errors)), sorted(set(warnings))


CSV_FIELDS = (
    "halo",
    "resolution",
    "status",
    "path",
    "errors",
    "warnings",
    "total_particles",
    "PartType0",
    "PartType1",
    "PartType2",
    "PartType3",
    "PartType4",
    "PartType5",
    "PartType6",
    "gas_mass",
    "hrdm_mass",
    "type2_mass_min",
    "type2_mass_max",
    "dm_gas_ratio",
    "expected_dm_gas_ratio",
    "id_unique",
    "finite",
    "bounds_ok",
    "type6_valid",
    "overlap_x",
    "overlap_y",
    "overlap_z",
    "BoxSize",
    "Time",
    "Redshift",
    "Omega0",
    "OmegaLambda",
    "OmegaBaryon",
    "HubbleParam",
)


def write_csv(results: list[FileResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            row = {
                "halo": result.halo,
                "resolution": result.resolution,
                "status": result.status,
                "path": str(result.path),
                "errors": " | ".join(result.errors),
                "warnings": " | ".join(result.warnings),
                "total_particles": result.total_particles,
                **{
                    f"PartType{ptype}": int(result.counts[ptype])
                    for ptype in range(NPART_TYPES)
                },
                "gas_mass": result.gas_mass,
                "hrdm_mass": result.hrdm_mass,
                "type2_mass_min": result.type2_mass_min,
                "type2_mass_max": result.type2_mass_max,
                "dm_gas_ratio": result.dm_gas_ratio,
                "expected_dm_gas_ratio": result.expected_dm_gas_ratio,
                "id_unique": result.id_unique,
                "finite": result.finite,
                "bounds_ok": result.bounds_ok,
                "type6_valid": result.type6_valid,
                "overlap_x": result.overlap_x,
                "overlap_y": result.overlap_y,
                "overlap_z": result.overlap_z,
                "BoxSize": result.box_size,
                "Time": result.time,
                "Redshift": result.redshift,
                "Omega0": result.omega0,
                "OmegaLambda": result.omega_lambda,
                "OmegaBaryon": result.omega_baryon,
                "HubbleParam": result.hubble_param,
            }
            writer.writerow(row)
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        print("ERROR: --chunk-size must be positive", file=sys.stderr)
        return 2

    ic_root = args.ic_root.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else ic_root / "MUSIC2_logs" / "ic_suite_validation.csv"
    )
    results: list[FileResult] = []

    print(f"Validating {len(HALOS) * len(RESOLUTIONS)} IC files under {ic_root}")
    print(f"ParticleID check: {args.id_check}")
    for halo in HALOS:
        for resolution in RESOLUTIONS:
            path = (
                ic_root
                / f"halo{halo}"
                / f"IC_halo{halo}_zoom_{resolution}.hdf5"
            )
            result = validate_file(
                path, halo, resolution, args.chunk_size, args.id_check
            )
            results.append(result)
            details = f"N={result.total_particles:,}"
            if result.errors:
                details += f"; {result.errors[0]}"
            elif result.warnings:
                details += f"; {result.warnings[0]}"
            print(
                f"[{result.status:4}] halo {halo:4d}  res {resolution:4d}  {details}",
                flush=True,
            )

    suite_errors, suite_warnings = compare_suite(results)
    write_csv(results, output_csv)

    fail_count = sum(result.status == "FAIL" for result in results)
    warn_count = sum(result.status == "WARN" for result in results)
    pass_count = sum(result.status == "PASS" for result in results)
    print()
    print(
        f"Files: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL "
        f"({len(results)} total)"
    )
    if suite_errors:
        print("Suite errors:")
        for message in suite_errors:
            print(f"  ERROR: {message}")
    if suite_warnings:
        print("Suite warnings:")
        for message in suite_warnings:
            print(f"  WARN: {message}")
    print(f"CSV summary: {output_csv}")

    failed = fail_count > 0 or bool(suite_errors)
    if args.strict:
        failed = failed or warn_count > 0 or bool(suite_warnings)
    if failed:
        print("FINAL STATUS: FAIL")
        return 1
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
