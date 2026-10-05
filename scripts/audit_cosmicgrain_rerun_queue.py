#!/usr/bin/env python3
"""Read-only preflight audit for the 12-halo CosmicGrain production queue.

The audit checks the frozen halo registry, generated parameter files,
MinGasHsml, ICs, output lists, unique canonical output directories, shared
same-resolution physics parameters, available storage, and safety/provenance
features in ``runCompleteGrid.sh``.  It never compiles, moves, archives,
deletes, resumes, or starts a simulation.

Example::

    cd ~/gadget4
    python3 scripts/audit_cosmicgrain_rerun_queue.py

Optionally enforce a site-specific free-space requirement::

    python3 scripts/audit_cosmicgrain_rerun_queue.py --minimum-free-tib 1.5
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from cosmicgrain_paper_contract import DEFAULT_CONTRACT, load_contract


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REQUIRED_PARAMETER_KEYS = (
    "InitCondFile",
    "OutputDir",
    "OutputListFilename",
    "MinGasHsml",
    "MinimumComovingHydroSoftening",
    "SofteningComovingClass0",
    "SofteningMaxPhysClass0",
    "SofteningComovingClass1",
    "SofteningMaxPhysClass1",
    "SofteningComovingClass4",
    "SofteningMaxPhysClass4",
    "SofteningComovingClass6",
    "SofteningMaxPhysClass6",
    "DustParticlesPerSNII",
    "DustParticlesPerAGB",
    "DustParticlesPerLRN",
    "DustCoagulationCalibration",
    "DustShatteringCalibration",
    "DustGrowthCalibration",
    "DustAstrationCalibration",
    "DustShockAmbientDensity",
    "DustCollisionDensityThresh",
)

SAME_RESOLUTION_KEYS = (
    "MinGasHsml",
    "MinimumComovingHydroSoftening",
    "SofteningComovingClass0",
    "SofteningMaxPhysClass0",
    "SofteningComovingClass1",
    "SofteningMaxPhysClass1",
    "SofteningComovingClass4",
    "SofteningMaxPhysClass4",
    "SofteningComovingClass6",
    "SofteningMaxPhysClass6",
    "CritPhysDensity",
    "MaxSfrTimescale",
    "SNEfficiency",
    "DustParticlesPerSNII",
    "DustParticlesPerAGB",
    "DustParticlesPerLRN",
    "DustCoagulationCalibration",
    "DustShatteringCalibration",
    "DustGrowthCalibration",
    "DustAstrationCalibration",
    "DustShockAmbientDensity",
    "DustCollisionDensityThresh",
    "DustYieldSNII",
    "DustYieldAGB",
    "DustEnableCreation",
    "DustEnableCooling",
    "DustEnableDrag",
    "DustEnableAstration",
    "DustEnableSputtering",
    "DustEnableGrowth",
    "DustEnableClumping",
    "DustEnableShockDestruction",
    "DustEnableCoagulation",
    "DustEnableShattering",
    "DustEnableRadiationPressure",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def strip_comment(line: str) -> str:
    return line.split("%", 1)[0].split("#", 1)[0].strip()


def parse_parameter_file(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    duplicates = []
    for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = strip_comment(raw)
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        key, value = fields[0], fields[1]
        if key in values:
            duplicates.append(f"{key} (line {number})")
        values[key] = value
    return values, duplicates


def resolve_from_root(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def matching_ic_files(root: Path, raw: str) -> list[Path]:
    path = resolve_from_root(root, raw)
    if path.exists():
        return [path]
    return [Path(item) for item in sorted(glob.glob(str(path) + "*"))
            if Path(item).is_file()]


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def human_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


def snapshot_indices(output: Path) -> list[int]:
    found = set()
    patterns = (
        output / "snapdir_*",
        output / "snapshot_*.hdf5",
        output / "snapshot_.*.hdf5",
    )
    for pattern in patterns:
        for item in glob.glob(str(pattern)):
            match = re.search(r"(?:snapdir|snapshot)_(\d+)", Path(item).name)
            if match:
                found.add(int(match.group(1)))
    return sorted(found)


def restart_files(output: Path) -> list[Path]:
    patterns = (
        output / "restartfiles" / "*",
        output / "restart.*",
        output / "restart_*",
    )
    result = []
    for pattern in patterns:
        result.extend(Path(item) for item in glob.glob(str(pattern))
                      if Path(item).is_file())
    return sorted(set(result))


def runner_audit(path: Path, expected_registry: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path), "exists": path.is_file(), "checks": {},
        "errors": [], "warnings": [],
    }
    if not path.is_file():
        result["errors"].append(f"runner not found: {path}")
        return result
    text = path.read_text(errors="replace")
    checks = {
        "strict_shell_mode": "set -euo pipefail" in text,
        "safe_archive_function": "archive_output" in text and "mv --" in text,
        "timestamped_backups": "_backup_" in text and "%Y%m%d_%H%M%S" in text,
        "metadata_start": "write_metadata" in text,
        "metadata_finish": "finish_metadata" in text,
        "git_provenance": "git_commit" in text and "git_branch" in text,
        "mpi_pipeline_status": "PIPESTATUS[0]" in text,
        "preflight_before_archive": "Check every selected run before compiling or moving" in text,
        "unsafe_recursive_delete_absent": not bool(re.search(r"\brm\s+-[A-Za-z]*r[A-Za-z]*f", text)),
        "resume_option": bool(re.search(r"--resume|restart\s+mode|RESTART_MODE", text, re.I)),
    }
    result["checks"] = checks
    for name, passed in checks.items():
        if name == "resume_option":
            continue
        if not passed:
            result["errors"].append(f"runner safety/provenance check failed: {name}")
    if checks["resume_option"]:
        result["restart_behavior"] = "runner advertises a resume/restart mode; inspect before use"
    else:
        result["restart_behavior"] = (
            "fresh-run only: an existing OutputDir is archived and a new run starts"
        )
        result["warnings"].append(
            "runner has no explicit resume mode; restart files will be archived, not resumed"
        )

    map_match = re.search(r"declare\s+-A\s+CG_HALO=\((.*?)\)", text, re.S)
    runner_registry = {}
    if map_match:
        runner_registry = {
            label: int(halo)
            for label, halo in re.findall(r"\[(CG\d{2})\]\s*=\s*(\d+)", map_match.group(1))
        }
    result["halo_registry"] = runner_registry
    if runner_registry != expected_registry:
        result["errors"].append(
            f"runner halo registry differs from contract: {runner_registry}"
        )
    return result


def add_issue(record: dict[str, Any], level: str, message: str) -> None:
    record[level].append(message)


def audit_halo(root: Path, params_dir: Path, label: str, halo_id: int,
               resolution: int, expected_snapshots: int,
               measure_sizes: bool) -> dict[str, Any]:
    parameter = params_dir / f"param_halo{halo_id}_{resolution}.txt"
    record: dict[str, Any] = {
        "label": label, "halo_id": halo_id,
        "parameter_file": str(parameter),
        "errors": [], "warnings": [], "info": [],
    }
    if not parameter.is_file():
        add_issue(record, "errors", f"missing parameter file: {parameter}")
        return record
    values, duplicates = parse_parameter_file(parameter)
    record["parameters"] = values
    if duplicates:
        add_issue(record, "errors", "duplicate parameter keys: " + ", ".join(duplicates))
    missing = [key for key in REQUIRED_PARAMETER_KEYS if key not in values]
    if missing:
        add_issue(record, "errors", "missing required keys: " + ", ".join(missing))

    try:
        min_hsml = float(values.get("MinGasHsml", "nan"))
        if min_hsml != 0.0:
            add_issue(record, "errors", f"MinGasHsml must be 0 for final rerun; got {min_hsml:g}")
    except ValueError:
        add_issue(record, "errors", f"invalid MinGasHsml: {values.get('MinGasHsml')!r}")

    expected_output = f"halo{halo_id}_output_{resolution}"
    actual_output = values.get("OutputDir")
    if actual_output != expected_output:
        add_issue(record, "errors",
                  f"OutputDir must be canonical {expected_output!r}; got {actual_output!r}")
    output = resolve_from_root(root, actual_output or expected_output)
    record["output_directory"] = str(output)

    initcond = values.get("InitCondFile", "")
    ic_files = matching_ic_files(root, initcond) if initcond else []
    record["initial_condition_files"] = [str(path) for path in ic_files]
    if not ic_files:
        add_issue(record, "errors", f"missing initial conditions: {initcond}*")
    if str(halo_id) not in initcond or str(resolution) not in initcond:
        add_issue(record, "warnings",
                  "InitCondFile name does not contain both halo ID and resolution")

    output_list_raw = values.get("OutputListFilename", "")
    output_list = resolve_from_root(root, output_list_raw) if output_list_raw else None
    record["output_list"] = str(output_list) if output_list else None
    if output_list is None or not output_list.is_file():
        add_issue(record, "errors", f"missing output list: {output_list_raw!r}")
    else:
        entries = [line for line in output_list.read_text().splitlines()
                   if strip_comment(line)]
        record["output_list_entries"] = len(entries)
        if len(entries) != expected_snapshots:
            add_issue(record, "warnings",
                      f"output list has {len(entries)} entries; expected {expected_snapshots}")

    record["output_exists"] = output.is_dir()
    snaps = snapshot_indices(output) if output.is_dir() else []
    restarts = restart_files(output) if output.is_dir() else []
    record["existing_snapshots"] = snaps
    record["restart_file_count"] = len(restarts)
    if output.exists() and not output.is_dir():
        add_issue(record, "errors", f"OutputDir exists but is not a directory: {output}")
    elif output.is_dir():
        add_issue(record, "warnings",
                  f"existing output will require archival before a fresh run: {output}")
        if restarts:
            add_issue(record, "warnings",
                      f"{len(restarts)} restart files exist; current runner will archive rather than resume")
        if snaps:
            add_issue(record, "info",
                      f"existing snapshots span {snaps[0]:03d}--{snaps[-1]:03d}")

    if measure_sizes:
        record["initial_condition_size_bytes"] = sum(
            path.stat().st_size for path in ic_files if path.is_file()
        )
        record["existing_output_size_bytes"] = directory_size(output)
    return record


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--params-dir", type=Path,
                        help="Defaults to ROOT/params")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--runner", type=Path,
                        help="Defaults to ROOT/runCompleteGrid.sh")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--expected-snapshots", type=int, default=30)
    parser.add_argument("--minimum-free-tib", type=float, default=0.0,
                        help="Site-specific hard free-space requirement; zero only reports free space")
    parser.add_argument("--measure-directory-sizes", action="store_true",
                        help="Walk existing outputs to estimate storage use; may be slow")
    parser.add_argument("--json", type=Path,
                        help="Defaults to ROOT/cosmicgrain_rerun_queue_audit.json")
    parser.add_argument("--csv", type=Path,
                        help="Defaults to ROOT/cosmicgrain_rerun_queue_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.resolution < 1 or args.expected_snapshots < 1 or args.minimum_free_tib < 0:
        raise SystemExit("resolution/expected snapshots must be positive and free-space threshold nonnegative")
    root = args.root.resolve()
    params_dir = (args.params_dir or root / "params").resolve()
    runner_path = (args.runner or root / "runCompleteGrid.sh").resolve()
    json_path = (args.json or root / "cosmicgrain_rerun_queue_audit.json").resolve()
    csv_path = (args.csv or root / "cosmicgrain_rerun_queue_audit.csv").resolve()
    contract, provenance = load_contract(args.contract)
    registry = {key: int(value) for key, value in contract["halo_registry"].items()}

    halos = [
        audit_halo(root, params_dir, label, halo_id, args.resolution,
                   args.expected_snapshots, args.measure_directory_sizes)
        for label, halo_id in registry.items()
    ]
    outputs = [item.get("output_directory") for item in halos
               if item.get("output_directory")]
    duplicate_outputs = sorted({path for path in outputs if outputs.count(path) > 1})
    global_errors = []
    global_warnings = []
    if duplicate_outputs:
        global_errors.append("duplicate OutputDir values: " + ", ".join(duplicate_outputs))

    # Same-resolution physics must be identical across the full suite.
    for key in SAME_RESOLUTION_KEYS:
        present = {
            item["label"]: item.get("parameters", {}).get(key)
            for item in halos if key in item.get("parameters", {})
        }
        distinct = sorted(set(present.values()))
        if len(distinct) > 1:
            global_errors.append(
                f"inconsistent {key}: "
                + ", ".join(f"{label}={value}" for label, value in present.items())
            )

    runner = runner_audit(runner_path, registry)
    global_errors.extend(runner["errors"])
    global_warnings.extend(runner["warnings"])

    disk = shutil.disk_usage(root)
    minimum_bytes = int(args.minimum_free_tib * 1024**4)
    if minimum_bytes and disk.free < minimum_bytes:
        global_errors.append(
            f"free storage {human_bytes(disk.free)} is below requested "
            f"minimum {args.minimum_free_tib:g} TiB"
        )
    if args.minimum_free_tib == 0:
        global_warnings.append(
            "storage is informational only; rerun with --minimum-free-tib VALUE "
            "to enforce the site allocation requirement"
        )

    errors = len(global_errors) + sum(len(item["errors"]) for item in halos)
    warnings = len(global_warnings) + sum(len(item["warnings"]) for item in halos)
    status = "PASS" if errors == 0 else "FAIL"
    report = {
        "audit": "CosmicGrain 12-halo final rerun queue",
        "created_utc": utc_now(),
        "status": status,
        "error_count": errors,
        "warning_count": warnings,
        "contract": provenance,
        "root": str(root),
        "resolution": args.resolution,
        "expected_snapshots": args.expected_snapshots,
        "storage": {
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "filesystem_free_human": human_bytes(disk.free),
            "minimum_required_tib": args.minimum_free_tib,
        },
        "runner": runner,
        "global_errors": global_errors,
        "global_warnings": global_warnings,
        "halos": halos,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "label", "halo_id", "parameter_file", "output_directory",
            "MinGasHsml", "output_exists", "existing_snapshots",
            "restart_file_count", "errors", "warnings",
        ])
        for item in halos:
            writer.writerow([
                item["label"], item["halo_id"], item["parameter_file"],
                item.get("output_directory", ""),
                item.get("parameters", {}).get("MinGasHsml", ""),
                item.get("output_exists", False),
                ";".join(str(x) for x in item.get("existing_snapshots", [])),
                item.get("restart_file_count", 0),
                " | ".join(item["errors"]), " | ".join(item["warnings"]),
            ])

    print("=" * 88)
    print("COSMICGRAIN FINAL RERUN QUEUE AUDIT")
    print("=" * 88)
    print(f"Contract: {provenance['contract_name']} v{provenance['contract_version']}")
    print(f"Resolution: {args.resolution}")
    print(f"Free storage: {human_bytes(disk.free)}")
    print(f"Runner restart behavior: {runner.get('restart_behavior', 'unknown')}")
    print()
    print(f"{'HALO':<6} {'ID':>6} {'HSML':>8} {'OUTPUT':<10} {'SNAPS':>7} {'RESTART':>8} RESULT")
    for item in halos:
        params = item.get("parameters", {})
        result = "FAIL" if item["errors"] else ("WARN" if item["warnings"] else "PASS")
        print(
            f"{item['label']:<6} {item['halo_id']:>6} "
            f"{params.get('MinGasHsml', '-'):>8} "
            f"{('exists' if item.get('output_exists') else 'fresh'):<10} "
            f"{len(item.get('existing_snapshots', [])):>7} "
            f"{item.get('restart_file_count', 0):>8} {result}"
        )
        for message in item["errors"]:
            print(f"       ERROR: {message}")
        for message in item["warnings"]:
            print(f"       WARN:  {message}")
    for message in global_errors:
        print(f"GLOBAL ERROR: {message}")
    for message in global_warnings:
        print(f"GLOBAL WARN:  {message}")
    print("-" * 88)
    print(f"STATUS: {status} ({errors} errors, {warnings} warnings)")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("=" * 88)
    raise SystemExit(0 if errors == 0 else 2)


if __name__ == "__main__":
    main()
