#!/usr/bin/env python3
"""Check unrestricted gas smoothing lengths and compare hydro time bins."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np


TIMEBIN_RE = re.compile(r"TIMEBIN_HYDRO\s+\[z=([0-9.eE+\-]+)\]:\s*(.*)")
BIN_RE = re.compile(r"\[(\d+):([0-9.eE+\-]+)=(\d+)\]")


def snapshot_files(root: Path, snap: int) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(str(root / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.*.hdf5")))
    if not files:
        single = root / f"snapshot_{snap:03d}.hdf5"
        if single.exists():
            files = [single]
    if not files:
        raise FileNotFoundError(f"No snapshot {snap:03d} found beneath {root}")
    return files


def percentile_text(values: np.ndarray) -> str:
    q = np.percentile(values, [0, 0.1, 1, 10, 50, 90, 99, 100])
    names = ("min", "p0.1", "p1", "p10", "p50", "p90", "p99", "max")
    return "  ".join(f"{name}={value:.6g}" for name, value in zip(names, q))


def smoothing_report(root: Path, snap: int, thresholds: list[float]) -> None:
    hsml_parts, sfr_parts = [], []
    files = snapshot_files(root, snap)
    with h5py.File(files[0], "r") as f:
        a = float(f["Header"].attrs["Time"])
        hubble = float(f["Parameters"].attrs["HubbleParam"])
    for path in files:
        with h5py.File(path, "r") as f:
            if "PartType0" not in f:
                continue
            hsml_parts.append(np.asarray(f["PartType0/SmoothingLength"], dtype=np.float64))
            sfr_parts.append(np.asarray(f["PartType0/StarFormationRate"], dtype=np.float64))
    hsml = np.concatenate(hsml_parts)
    sfr = np.concatenate(sfr_parts)
    sf = sfr > 0
    physical = hsml * a / hubble

    print(f"\nSMOOTHING LENGTHS: {root}, snapshot {snap:03d}")
    print(f"a={a:.8g}  z={1/a-1:.6g}  HubbleParam={hubble:.6g}  N_gas={len(hsml):,}")
    print("Stored code units, all gas:")
    print("  " + percentile_text(hsml))
    print("Physical pkpc, all gas:")
    print("  " + percentile_text(physical))
    print(f"Star-forming gas: {sf.sum():,}; total SFR={sfr.sum():.8g} Msun/yr")
    if sf.any():
        print("Stored code units, star-forming gas:")
        print("  " + percentile_text(hsml[sf]))
        print("Physical pkpc, star-forming gas:")
        print("  " + percentile_text(physical[sf]))

    print("\nThreshold occupancy:")
    print(" h_limit    N_all   frac_all       N_SF    frac_SF    SFR_frac")
    for limit in thresholds:
        low = hsml < limit
        low_sf = low & sf
        sfr_frac = sfr[low].sum() / sfr.sum() if sfr.sum() > 0 else np.nan
        print(f" {limit:7g} {low.sum():9,d} {low.mean():10.4%} "
              f"{low_sf.sum():9,d} {low_sf.sum()/max(sf.sum(),1):10.4%} {sfr_frac:10.4%}")


def parse_timebins(path: Path) -> list[dict]:
    records = []
    with path.open("r", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = TIMEBIN_RE.search(line)
            if not match:
                continue
            bins = {
                int(number): (float(dt), int(count))
                for number, dt, count in BIN_RE.findall(match.group(2))
            }
            if bins:
                records.append({"z": float(match.group(1)), "bins": bins, "line": line_number})
    if not records:
        raise RuntimeError(f"No TIMEBIN_HYDRO records found in {path}")
    return records


def nearest_record(records: list[dict], target_z: float) -> dict:
    # On equal rounded redshifts, prefer the latest record in the log.
    return min(records, key=lambda record: (abs(record["z"] - target_z), -record["line"]))


def timebin_report(old_log: Path, new_log: Path, target_z: float) -> None:
    old = nearest_record(parse_timebins(old_log), target_z)
    new = nearest_record(parse_timebins(new_log), target_z)
    print(f"\nMATCHED HYDRO TIMEBINS requested near z={target_z:g}")
    print(f"old: {old_log}  selected z={old['z']:g}, line={old['line']}")
    print(f"new: {new_log}  selected z={new['z']:g}, line={new['line']}")
    print("\n bin       dt_old      N_old       dt_new      N_new")
    for number in sorted(set(old["bins"]) | set(new["bins"])):
        odt, on = old["bins"].get(number, (np.nan, 0))
        ndt, nn = new["bins"].get(number, (np.nan, 0))
        print(f" {number:3d} {odt:12.4e} {on:10,d} {ndt:12.4e} {nn:10,d}")
    for label, record in (("old", old), ("new", new)):
        smallest = min(record["bins"])
        dt, count = record["bins"][smallest]
        total = sum(n for _, n in record["bins"].values())
        short = sum(n for b, (_, n) in record["bins"].items() if b <= 12)
        print(f"{label}: smallest occupied bin={smallest}, dt={dt:.4e}, N={count:,}; "
              f"N(bin<=12)={short:,}/{total:,} ({short/max(total,1):.4%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root", type=Path, help="new MinGasHsml=0 output directory")
    parser.add_argument("--snapshot", type=int, default=14)
    parser.add_argument("--old-log", type=Path)
    parser.add_argument("--new-log", type=Path)
    parser.add_argument("--target-z", type=float, default=2.49,
                        help="redshift at which to compare TIMEBIN_HYDRO records")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.0625])
    args = parser.parse_args()

    smoothing_report(args.snapshot_root, args.snapshot, args.thresholds)
    if (args.old_log is None) != (args.new_log is None):
        parser.error("supply both --old-log and --new-log, or neither")
    if args.old_log is not None:
        timebin_report(args.old_log, args.new_log, args.target_z)


if __name__ == "__main__":
    main()
