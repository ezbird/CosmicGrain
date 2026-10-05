#!/usr/bin/env python3
"""Build the CosmicGrain manuscript figures from the frozen analysis contract.

The driver owns final analysis arguments.  Command-line options identify data
locations and snapshots, but cannot change apertures, pixel scales, cuts,
normalizations, or statistical conventions fixed by the contract.

Examples
--------
Preflight the complete final build::

    python3 scripts/build_cosmicgrain_paper.py --preflight

Build only currently available suite products::

    python3 scripts/build_cosmicgrain_paper.py --figures 1 2 3 5

Build all figures once the final CG10 and Zhang--Green inputs exist::

    python3 scripts/build_cosmicgrain_paper.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

from cosmicgrain_paper_contract import (
    DEFAULT_CONTRACT,
    figure_contract,
    load_contract,
    measurement,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def shell_join(command: list[str]) -> str:
    return shlex.join(str(item) for item in command)


def existing_snapshot_pattern(output: Path, snapshot: int) -> str:
    candidates = [
        output / f"snapdir_{snapshot:03d}" / f"snapshot_{snapshot:03d}.*.hdf5",
        output / f"snapdir_{snapshot:03d}" / f"snapshot_{snapshot:03d}.hdf5",
        output / f"snapshot_{snapshot:03d}.*.hdf5",
        output / f"snapshot_{snapshot:03d}.hdf5",
    ]
    import glob
    for candidate in candidates:
        if glob.glob(str(candidate)):
            return str(candidate)
    return str(candidates[0])


def require_file(path: Path, description: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {description}: {path}")


def require_dir(path: Path, description: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing {description}: {path}")


def task(name: str, figures: list[int], command: list[str], cwd: Path,
         output_dir: Path, requirements: list[tuple[Path, str]]) -> dict[str, Any]:
    return {
        "name": name,
        "figures": figures,
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "output_dir": str(output_dir),
        "requirements": [(str(path), description) for path, description in requirements],
        "status": "pending",
    }


def build_tasks(args, contract: dict[str, Any], build_dir: Path) -> list[dict[str, Any]]:
    py = sys.executable
    scripts = args.scripts_dir.resolve()
    root = args.root.resolve()
    selected = set(args.figures)
    tasks = []

    integrated = measurement(contract, "integrated_galaxy")
    gsd = measurement(contract, "grain_size_distribution")
    metallicity = measurement(contract, "metallicity")
    optics = measurement(contract, "optical_constants")
    orientation = measurement(contract, "orientation")

    if selected & {1, 2, 3}:
        out = build_dir / "figures_1_to_3"
        command = [
            py, str(scripts / "cosmicgrain_multihalo_comparison.py"),
            "--root", str(root),
            "--resolution", str(args.suite_resolution),
            "--outdir", str(out),
            "--aperture-pkpc", str(integrated["aperture_pkpc"]),
            "--radial-max-pkpc", str(integrated["aperture_pkpc"]),
            "--xh-mode", str(metallicity["hydrogen_prescription"]),
            "--dataset-label", "",
            "--grain-min-micron", str(gsd["minimum_radius_nm"] * 1.0e-3),
            "--grain-max-micron", str(gsd["maximum_radius_nm"] * 1.0e-3),
            "--grain-bins", str(gsd["logarithmic_bins"]),
            "--grain-evolution-redshift", "2.0",
            "--include-grain-evolution",
        ]
        if args.include_incomplete_suite:
            command.append("--include-incomplete")
        tasks.append(task(
            "suite_figures_1_to_3", sorted(selected & {1, 2, 3}), command,
            root, out,
            [(scripts / "cosmicgrain_multihalo_comparison.py", "suite analysis script")],
        ))

    cg10_output = (root / args.cg10_output).resolve()
    snapshot_pattern = existing_snapshot_pattern(cg10_output, args.cg10_snapshot)

    if 4 in selected:
        cfg = measurement(contract, "cg10_grain_rv")
        out = build_dir / "figure_4"
        bins = round(2.0 * cfg["projected_aperture_pkpc"] / cfg["pixel_size_pkpc"])
        command = [
            py, str(scripts / "cosmicgrain_figure4_rv_grains.py"),
            snapshot_pattern,
            "--center-file", str(args.cg10_center_file.resolve()),
            "--rotation-file", str(args.cg10_rotation_file.resolve()),
            "--carbon-table", str(root / optics["carbonaceous_table"]),
            "--silicate-table", str(root / optics["silicate_table"]),
            "--plot-radius", str(cfg["projected_aperture_pkpc"]),
            "--zmax", str(cfg["projection_half_depth_pkpc"]),
            "--bins", str(bins),
            "--min-count", str(cfg["minimum_dust_particles_per_pixel"]),
            "--min-signal-fraction", str(cfg["minimum_tauV_fraction_of_aperture_maximum"]),
            "--min-ebv-fraction", str(cfg["minimum_EBV_fraction_of_aperture_maximum"]),
            "--rv-min", str(cfg["displayed_RV_range"][0]),
            "--rv-max", str(cfg["displayed_RV_range"][1]),
            "--small-grain-nm", str(gsd["small_grain_threshold_nm"]),
            "--prefix", str(out / "CG10_figure4"),
        ]
        tasks.append(task(
            "figure_4_grains_behind_rv", [4], command, root, out,
            [
                (scripts / "cosmicgrain_figure4_rv_grains.py", "Figure 4 script"),
                (cg10_output, "final CG10 output directory"),
                (args.cg10_center_file.resolve(), "audited CG10 center"),
                (args.cg10_rotation_file.resolve(), "audited CG10 rotation matrix"),
            ],
        ))

    if 5 in selected:
        cfg = measurement(contract, "suite_extinction")
        out = build_dir / "figure_5"
        command = [
            py, str(scripts / "cosmicgrain_figure5_direct.py"),
            "--output-root", str(root),
            "--resolution", str(args.suite_resolution),
            "--snap", str(args.suite_snapshot),
            "--pixel-size-pkpc", str(cfg["pixel_size_pkpc"]),
            "--aperture-r200", str(float(str(cfg["projected_aperture"]).split("_")[0])),
            "--depth-r200", str(float(str(cfg["projection_half_depth"]).split("_")[0])),
            "--min-count", str(cfg["minimum_dust_particles_per_pixel"]),
            "--min-signal-fraction", str(cfg["minimum_tauV_fraction_of_aperture_maximum"]),
            "--min-ebv-fraction", str(cfg["minimum_EBV_fraction_of_aperture_maximum"]),
            "--tail-rv", str(cfg["displayed_RV_range"][1]),
            "--orientation-radius-pkpc", str(orientation["radius_pkpc"]),
            "--min-radial-pixels", str(cfg["minimum_pixels_per_radial_bin"]),
            "--carbon-table", str(root / optics["carbonaceous_table"]),
            "--silicate-table", str(root / optics["silicate_table"]),
            "--prefix", str(out / "CG_figure5_suite_rv"),
            "--dataset-label", "",
        ]
        tasks.append(task(
            "figure_5_suite_extinction", [5], command, root, out,
            [(scripts / "cosmicgrain_figure5_direct.py", "Figure 5 script")],
        ))

    if 6 in selected:
        cfg = measurement(contract, "zhang_green_comparison")
        out = build_dir / "figure_6"
        command = [
            py, str(scripts / "zg_cg10_items1to4_final.py"),
            str(args.zg_h5.resolve()),
            "--root", str(root),
            "--snapshot", str(args.cg10_snapshot),
            "--output-dir", str(cg10_output),
            "--carbon-table", str(root / optics["carbonaceous_table"]),
            "--silicate-table", str(root / optics["silicate_table"]),
            "--output-prefix", str(out / "CG10_ZG_figure6"),
            "--half-size", str(cfg["projected_aperture_pkpc"]),
            "--radius", str(cfg["projected_aperture_pkpc"]),
            "--zmax", str(cfg["projection_half_depth_pkpc"]),
            "--pixel", str(cfg["pixel_size_pkpc"]),
            "--dz", str(cfg["layer_depth_pkpc"]),
            "--paper-rv-min", "1.5",
            "--paper-rv-max", "10.0",
        ]
        tasks.append(task(
            "figure_6_milky_way_benchmark", [6], command, root, out,
            [
                (scripts / "zg_cg10_items1to4_final.py", "Figure 6 script"),
                (scripts / "cosmicgrain_z0_z6_gsd_rv.py", "Figure 6 support module"),
                (cg10_output, "final CG10 output directory"),
                (args.zg_h5.resolve(), "Zhang--Green HDF5 map"),
            ],
        ))
    return tasks


def preflight(tasks: list[dict[str, Any]], root: Path,
              contract_path: Path, contract: dict[str, Any],
              suite_resolution: int) -> list[str]:
    errors = []
    require_file(contract_path, "paper contract", errors)
    for task_spec in tasks:
        for raw_path, description in task_spec["requirements"]:
            path = Path(raw_path)
            if "directory" in description:
                require_dir(path, description, errors)
            else:
                require_file(path, description, errors)
        command = task_spec["command"]
        if len(command) >= 2 and not Path(command[1]).is_file():
            errors.append(f"missing executable script for {task_spec['name']}: {command[1]}")
        if task_spec["name"] == "figure_4_grains_behind_rv":
            import glob
            if not glob.glob(command[2]):
                errors.append(f"missing CG10 snapshot pieces: {command[2]}")
    if any(spec["name"] in {
        "suite_figures_1_to_3", "figure_5_suite_extinction"
    } for spec in tasks):
        for label, halo_id in contract["halo_registry"].items():
            output = root / f"halo{halo_id}_output_{suite_resolution}"
            if not output.is_dir():
                errors.append(
                    f"missing canonical final-suite output for {label}: {output}"
                )
    if shutil.which(Path(sys.executable).name) is None and not Path(sys.executable).is_file():
        errors.append(f"Python executable is unavailable: {sys.executable}")
    if not root.is_dir():
        errors.append(f"project root is not a directory: {root}")
    return sorted(set(errors))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_task(spec: dict[str, Any], log_path: Path) -> int:
    Path(spec["output_dir"]).mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 88)
    print(f"RUNNING {spec['name']} (paper figures {spec['figures']})")
    print(shell_join(spec["command"]))
    print("=" * 88)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            spec["command"], cwd=spec["cwd"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def product_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"paper_build_manifest.json"}:
            records.append({
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 7),
                        default=list(range(1, 7)))
    parser.add_argument("--build-root", type=Path,
                        default=PROJECT_ROOT / "cosmicgrain_paper_builds")
    parser.add_argument("--build-name", default=None)
    parser.add_argument("--suite-resolution", type=int, default=1024)
    parser.add_argument("--suite-snapshot", type=int, default=29)
    parser.add_argument("--include-incomplete-suite", action="store_true",
                        help="Exploratory only: allow incomplete suite runs")
    parser.add_argument("--cg10-output",
                        default="halo441_output_2048_mingashsml0.0")
    parser.add_argument("--cg10-snapshot", type=int, default=29)
    parser.add_argument("--cg10-center-file", type=Path,
                        default=SCRIPT_DIR / "halo441_2048_final_snap029_audited_center.txt")
    parser.add_argument("--cg10-rotation-file", type=Path,
                        default=SCRIPT_DIR / "halo441_2048_final_snap029_audited_rotation_matrix.txt")
    parser.add_argument("--zg-h5", type=Path, default=SCRIPT_DIR / "Rv_map_new.h5")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate all inputs and print commands without running")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the manifest and print commands without running")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.figures = sorted(set(args.figures))
    contract, provenance = load_contract(args.contract)
    if args.include_incomplete_suite and contract["contract"]["status"] == "frozen":
        print("WARNING: --include-incomplete-suite makes this an exploratory, "
              "non-final build.")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build_name = args.build_name or (
        f"contract_v{provenance['contract_version']}_{stamp}"
    )
    build_dir = args.build_root.resolve() / build_name
    if build_dir.exists() and any(build_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty paper build directory: {build_dir}"
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, contract, build_dir)
    errors = preflight(
        tasks, args.root.resolve(), args.contract.resolve(), contract,
        args.suite_resolution,
    )

    manifest = {
        "analysis": "CosmicGrain manuscript paper build",
        "created_utc": utc_now(),
        "status": "preflight_failed" if errors else "ready",
        "final_contract_compliant": not args.include_incomplete_suite,
        "contract": provenance,
        "requested_figures": args.figures,
        "figure_contracts": {
            str(number): figure_contract(contract, number)
            for number in args.figures
        },
        "project_root": str(args.root.resolve()),
        "scripts_dir": str(args.scripts_dir.resolve()),
        "git_commit": git_value(args.root.resolve(), "rev-parse", "HEAD"),
        "git_branch": git_value(args.root.resolve(), "rev-parse", "--abbrev-ref", "HEAD"),
        "preflight_errors": errors,
        "tasks": tasks,
        "products": [],
    }
    manifest_path = build_dir / "paper_build_manifest.json"
    write_manifest(manifest_path, manifest)

    print(f"Contract: {provenance['contract_name']} v{provenance['contract_version']}")
    print(f"SHA-256:  {provenance['contract_sha256']}")
    print(f"Build:    {build_dir}")
    for spec in tasks:
        print(f"\n[{spec['name']}] figures {spec['figures']}")
        print("  " + shell_join(spec["command"]))

    if errors:
        print("\nPREFLIGHT FAILED")
        for error in errors:
            print(f"  - {error}")
        print(f"Manifest: {manifest_path}")
        raise SystemExit(2)
    print("\nPREFLIGHT PASSED")
    if args.preflight or args.dry_run:
        manifest["status"] = "preflight_passed" if args.preflight else "dry_run"
        write_manifest(manifest_path, manifest)
        print(f"Manifest: {manifest_path}")
        return

    manifest["status"] = "running"
    manifest["started_utc"] = utc_now()
    write_manifest(manifest_path, manifest)
    any_failure = False
    for spec in tasks:
        spec["started_utc"] = utc_now()
        spec["status"] = "running"
        write_manifest(manifest_path, manifest)
        status = run_task(spec, build_dir / f"{spec['name']}.log")
        spec["finished_utc"] = utc_now()
        spec["exit_code"] = status
        spec["status"] = "completed" if status == 0 else "failed"
        write_manifest(manifest_path, manifest)
        if status != 0:
            any_failure = True
            if not args.keep_going:
                break

    manifest["finished_utc"] = utc_now()
    manifest["products"] = product_records(build_dir)
    manifest["status"] = "failed" if any_failure else "completed"
    write_manifest(manifest_path, manifest)
    print("\n" + "=" * 88)
    print(f"BUILD {manifest['status'].upper()}: {build_dir}")
    print(f"Products recorded: {len(manifest['products'])}")
    print(f"Manifest: {manifest_path}")
    print("=" * 88)
    if any_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
