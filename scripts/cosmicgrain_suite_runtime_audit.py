#!/usr/bin/env python3

"""
CosmicGrain suite runtime / computational-cost audit.

For each CG01--CG12 zoom run, measure:

  * parent DM-only M200c
  * first/latest snapshot and redshift
  * initial gas / HRDM / LRDM particle counts
  * latest gas / star / dust particle counts
  * peak dust particle count over available snapshots
  * wall-clock elapsed time inferred from snapshot timestamps
  * synchronization-point count
  * domain decompositions
  * FOF / SUBFIND executions
  * cumulative active gravity and hydro particle updates
  * feedback / dust log-line counts
  * debug levels from the parameter file
  * normalized runtime diagnostics

The purpose is QC/performance diagnosis, NOT physical galaxy analysis.

Run from ~/gadget4:

    python3 scripts/cosmicgrain_suite_runtime_audit.py

Optional:

    python3 scripts/cosmicgrain_suite_runtime_audit.py \
        --root . \
        --resolution 1024 \
        --csv cosmicgrain_suite_runtime_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Suite definition
# ---------------------------------------------------------------------------

SUITE = {
    "CG01": {"halo": 9235, "m200_parent": 5.601e10},
    "CG02": {"halo": 7723, "m200_parent": 6.499e10},
    "CG03": {"halo": 5834, "m200_parent": 8.766e10},
    "CG04": {"halo": 3886, "m200_parent": 1.332e11},
    "CG05": {"halo": 3879, "m200_parent": 1.438e11},
    "CG06": {"halo": 3352, "m200_parent": 1.593e11},
    "CG07": {"halo": 1534, "m200_parent": 4.139e11},
    "CG08": {"halo": 1481, "m200_parent": 4.241e11},
    "CG09": {"halo": 859,  "m200_parent": 6.808e11},
    "CG10": {"halo": 441,  "m200_parent": 1.061e12},
    "CG11": {"halo": 308,  "m200_parent": 1.353e12},
    "CG12": {"halo": 295,  "m200_parent": 1.954e12},
}


DEBUG_KEYS = (
    "StarformationDebugLevel",
    "FeedbackDebugLevel",
    "DustDebugLevel",
    "SpatialHashDebugLevel",
    "CheckpointDebugLevel",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt_int(x):
    if x is None:
        return "--"
    return f"{int(x):,}"


def fmt_float(x, fmt=".3g"):
    if x is None or not np.isfinite(x):
        return "--"
    return format(x, fmt)


def snapnum_from_path(path: Path) -> Optional[int]:
    m = re.search(r"snapdir_(\d+)", str(path))
    if m:
        return int(m.group(1))
    m = re.search(r"snapshot_(\d+)", str(path))
    if m:
        return int(m.group(1))
    return None


def snapshot_piece_files(output_dir: Path, snap: int):
    d = output_dir / f"snapdir_{snap:03d}"

    if d.is_dir():
        files = sorted(d.glob(f"snapshot_{snap:03d}.*.hdf5"))
        if files:
            return files

        f = d / f"snapshot_{snap:03d}.hdf5"
        if f.exists():
            return [f]

    # non-snapdir fallback
    files = sorted(output_dir.glob(f"snapshot_{snap:03d}.*.hdf5"))
    if files:
        return files

    f = output_dir / f"snapshot_{snap:03d}.hdf5"
    if f.exists():
        return [f]

    return []


def discover_snapshots(output_dir: Path):
    nums = set()

    for p in output_dir.glob("snapdir_*"):
        if p.is_dir():
            n = snapnum_from_path(p)
            if n is not None and snapshot_piece_files(output_dir, n):
                nums.add(n)

    for p in output_dir.glob("snapshot_*.hdf5"):
        n = snapnum_from_path(p)
        if n is not None:
            nums.add(n)

    return sorted(nums)


def read_snapshot_header(output_dir: Path, snap: int):
    files = snapshot_piece_files(output_dir, snap)
    if not files:
        return None

    with h5py.File(files[0], "r") as f:
        h = f["Header"].attrs

        z = float(h.get("Redshift", np.nan))
        a = float(h.get("Time", np.nan))

        # NumPart_Total is the preferred source.
        total = np.asarray(h.get("NumPart_Total", np.zeros(6)), dtype=np.uint64)

        # GADGET may provide high words for >2^32 particles.
        high = np.asarray(
            h.get("NumPart_Total_HighWord", np.zeros_like(total)),
            dtype=np.uint64,
        )

        if len(high) == len(total):
            total = total + (high << np.uint64(32))

        counts = np.zeros(max(7, len(total)), dtype=np.uint64)
        counts[:len(total)] = total

    return {
        "z": z,
        "a": a,
        "counts": counts,
        "mtime": max(p.stat().st_mtime for p in files),
    }


def find_output_dir(root: Path, halo: int, resolution: int):
    """
    Prefer canonical halo<ID>_output_<resolution>, then suffix variants.
    Choose the candidate with the highest available snapshot number.
    """

    patterns = [
        f"halo{halo}_output_{resolution}",
        f"halo{halo}_output_{resolution}_*",
    ]

    candidates = []
    seen = set()

    for pattern in patterns:
        for p in root.glob(pattern):
            if not p.is_dir():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)

            snaps = discover_snapshots(p)
            if snaps:
                candidates.append((max(snaps), p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return candidates[0][1]


def find_log(output_dir: Path):
    logs = list(output_dir.glob("*.log"))
    if not logs:
        return None

    # largest log is usually the actual simulation output log
    logs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return logs[0]


def find_param_file(output_dir: Path, root: Path, halo: int, resolution: int):
    candidates = list(output_dir.glob("param*.txt"))

    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    p = root / "params" / f"param_halo{halo}_{resolution}.txt"
    if p.exists():
        return p

    return None


def read_debug_levels(param_file: Optional[Path]):
    result = {k: None for k in DEBUG_KEYS}

    if param_file is None or not param_file.exists():
        return result

    with param_file.open("r", errors="replace") as f:
        for line in f:
            # remove comments
            line = line.split("%", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            key = parts[0]

            if key in result:
                try:
                    result[key] = int(float(parts[1]))
                except ValueError:
                    pass

    return result


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

SYNC_RE = re.compile(
    r"^Sync-Point\s+(\d+).*?"
    r"Nsync-grv:\s*(\d+).*?"
    r"Nsync-hyd:\s*(\d+)"
)


@dataclass
class LogStats:
    sync_points: int = 0
    last_sync: Optional[int] = None

    gravity_updates: int = 0
    hydro_updates: int = 0

    domain_decomp: int = 0
    fof: int = 0
    subfind: int = 0

    feedback_lines: int = 0
    dust_lines: int = 0

    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None


def parse_log(logfile: Optional[Path]) -> LogStats:
    stats = LogStats()

    if logfile is None or not logfile.exists():
        return stats

    with logfile.open("r", errors="replace") as f:
        for line in f:

            if line.startswith("Sync-Point"):
                stats.sync_points += 1

                m = SYNC_RE.search(line)
                if m:
                    stats.last_sync = int(m.group(1))
                    stats.gravity_updates += int(m.group(2))
                    stats.hydro_updates += int(m.group(3))

            if "DOMAIN: Begin domain decomposition" in line:
                stats.domain_decomp += 1

            if "FOF: We shall first compute" in line:
                stats.fof += 1

            if "SUBFIND: We now execute" in line:
                stats.subfind += 1

            lower = line.lower()

            if "feedback" in lower:
                stats.feedback_lines += 1

            if "dust" in lower:
                stats.dust_lines += 1

    return stats


# ---------------------------------------------------------------------------
# Runtime extraction
# ---------------------------------------------------------------------------

def snapshot_runtime_hours(headers):
    """
    Use filesystem timestamps of the first and latest available snapshots.

    This measures elapsed wall-clock span represented by the output files.
    It can include pauses/restarts, so it is not identical to CPU time.
    """

    valid = [h for h in headers if h is not None]

    if len(valid) < 2:
        return None

    t0 = min(h["mtime"] for h in valid)
    t1 = max(h["mtime"] for h in valid)

    return (t1 - t0) / 3600.0


def count_peak_dust(headers):
    vals = []

    for h in headers:
        if h is None:
            continue

        c = h["counts"]

        if len(c) > 6:
            vals.append(int(c[6]))

    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Audit one run
# ---------------------------------------------------------------------------

def audit_run(label, meta, root, resolution):
    halo = meta["halo"]

    d = find_output_dir(root, halo, resolution)

    if d is None:
        return {
            "CG": label,
            "halo": halo,
            "M200_parent_Msun": meta["m200_parent"],
            "status": "NOT FOUND",
        }

    snaps = discover_snapshots(d)

    if not snaps:
        return {
            "CG": label,
            "halo": halo,
            "M200_parent_Msun": meta["m200_parent"],
            "output_dir": str(d),
            "status": "NO SNAPSHOTS",
        }

    headers = []

    for s in snaps:
        h = read_snapshot_header(d, s)
        headers.append(h)

    first_h = headers[0]
    last_h = headers[-1]

    log = find_log(d)
    logstats = parse_log(log)

    param = find_param_file(d, root, halo, resolution)
    debug = read_debug_levels(param)

    first_counts = first_h["counts"]
    last_counts = last_h["counts"]

    runtime_hr = snapshot_runtime_hours(headers)

    total_updates = logstats.gravity_updates + logstats.hydro_updates

    sec_per_sync = None
    if runtime_hr is not None and logstats.sync_points > 0:
        sec_per_sync = runtime_hr * 3600.0 / logstats.sync_points

    sec_per_million_updates = None
    if runtime_hr is not None and total_updates > 0:
        sec_per_million_updates = (
            runtime_hr * 3600.0 / (total_updates / 1.0e6)
        )

    hydro_updates_per_initial_gas = None
    if len(first_counts) > 0 and first_counts[0] > 0:
        hydro_updates_per_initial_gas = (
            logstats.hydro_updates / float(first_counts[0])
        )

    grav_updates_per_initial_hrdm = None
    if len(first_counts) > 1 and first_counts[1] > 0:
        grav_updates_per_initial_hrdm = (
            logstats.gravity_updates / float(first_counts[1])
        )

    # Latest snapshot may still be incomplete relative to expected 029.
    status = "COMPLETE" if max(snaps) >= 29 else "RUNNING/INCOMPLETE"

    row = {
        "CG": label,
        "halo": halo,
        "M200_parent_Msun": meta["m200_parent"],
        "status": status,

        "output_dir": str(d),
        "log_file": str(log) if log else "",
        "param_file": str(param) if param else "",

        "first_snap": min(snaps),
        "latest_snap": max(snaps),
        "latest_z": last_h["z"],

        "N_gas_initial": int(first_counts[0]) if len(first_counts) > 0 else None,
        "N_HRDM_initial": int(first_counts[1]) if len(first_counts) > 1 else None,
        "N_LRDM_initial": int(first_counts[2]) if len(first_counts) > 2 else None,

        "N_gas_latest": int(last_counts[0]) if len(last_counts) > 0 else None,
        "N_star_latest": int(last_counts[4]) if len(last_counts) > 4 else None,
        "N_dust_latest": int(last_counts[6]) if len(last_counts) > 6 else None,

        "N_dust_peak": count_peak_dust(headers),

        "runtime_hours_snapshot_span": runtime_hr,

        "sync_points": logstats.sync_points,
        "last_sync": logstats.last_sync,
        "domain_decompositions": logstats.domain_decomp,
        "FOF_executions": logstats.fof,
        "SUBFIND_executions": logstats.subfind,

        "gravity_active_updates": logstats.gravity_updates,
        "hydro_active_updates": logstats.hydro_updates,
        "total_active_updates": total_updates,

        "feedback_log_lines": logstats.feedback_lines,
        "dust_log_lines": logstats.dust_lines,

        "seconds_per_sync": sec_per_sync,
        "seconds_per_million_active_updates": sec_per_million_updates,

        "hydro_updates_per_initial_gas": hydro_updates_per_initial_gas,
        "gravity_updates_per_initial_HRDM": grav_updates_per_initial_hrdm,
    }

    for key in DEBUG_KEYS:
        row[key] = debug[key]

    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows):
    print()
    print("=" * 180)
    print("COSMICGRAIN SUITE RUNTIME AUDIT")
    print("=" * 180)

    header = (
        f"{'CG':<5}"
        f"{'halo':>7}"
        f"{'zlast':>8}"
        f"{'Ngas0':>12}"
        f"{'NHRDM0':>12}"
        f"{'Nstar':>11}"
        f"{'NdustPk':>11}"
        f"{'Nsync':>9}"
        f"{'HydUpd':>13}"
        f"{'GravUpd':>13}"
        f"{'Ndom':>7}"
        f"{'hours':>9}"
        f"{'s/Mupd':>10}"
        f"{'FBdbg':>7}"
        f"{'Ddbg':>6}"
        f"{'SHdbg':>7}"
    )

    print(header)
    print("-" * len(header))

    for r in rows:

        if r.get("status") == "NOT FOUND":
            print(
                f"{r['CG']:<5}{r['halo']:>7}"
                f"   -- NOT FOUND"
            )
            continue

        if r.get("status") == "NO SNAPSHOTS":
            print(
                f"{r['CG']:<5}{r['halo']:>7}"
                f"   -- NO SNAPSHOTS"
            )
            continue

        print(
            f"{r['CG']:<5}"
            f"{r['halo']:>7}"
            f"{fmt_float(r.get('latest_z'), '.3f'):>8}"
            f"{fmt_int(r.get('N_gas_initial')):>12}"
            f"{fmt_int(r.get('N_HRDM_initial')):>12}"
            f"{fmt_int(r.get('N_star_latest')):>11}"
            f"{fmt_int(r.get('N_dust_peak')):>11}"
            f"{fmt_int(r.get('sync_points')):>9}"
            f"{fmt_float(r.get('hydro_active_updates') / 1e9 if r.get('hydro_active_updates') else None, '.2f'):>13}"
            f"{fmt_float(r.get('gravity_active_updates') / 1e9 if r.get('gravity_active_updates') else None, '.2f'):>13}"
            f"{fmt_int(r.get('domain_decompositions')):>7}"
            f"{fmt_float(r.get('runtime_hours_snapshot_span'), '.2f'):>9}"
            f"{fmt_float(r.get('seconds_per_million_active_updates'), '.3f'):>10}"
            f"{str(r.get('FeedbackDebugLevel', '--')):>7}"
            f"{str(r.get('DustDebugLevel', '--')):>6}"
            f"{str(r.get('SpatialHashDebugLevel', '--')):>7}"
        )

    print()
    print("HydUpd / GravUpd are billions of cumulative active-particle updates.")
    print("hours = filesystem time span between first and latest snapshots.")
    print("s/Mupd = elapsed seconds per million active-particle updates.")
    print()


def print_detailed(rows):
    print("=" * 80)
    print("DETAILED RUN DIAGNOSTICS")
    print("=" * 80)

    for r in rows:
        if r.get("status") in ("NOT FOUND", "NO SNAPSHOTS"):
            continue

        print()
        print(
            f"{r['CG']} / halo{r['halo']} "
            f"[{r['status']}]"
        )
        print("-" * 80)

        print(f"Output directory:          {r.get('output_dir')}")
        print(f"Latest snapshot:           {r.get('latest_snap')}")
        print(f"Latest redshift:           {fmt_float(r.get('latest_z'), '.6f')}")
        print(f"Parent M200:               {r['M200_parent_Msun']:.4e} Msun")

        print()
        print(f"Initial gas:               {fmt_int(r.get('N_gas_initial'))}")
        print(f"Initial HRDM:              {fmt_int(r.get('N_HRDM_initial'))}")
        print(f"Initial LRDM:              {fmt_int(r.get('N_LRDM_initial'))}")
        print(f"Latest stars:              {fmt_int(r.get('N_star_latest'))}")
        print(f"Latest dust:               {fmt_int(r.get('N_dust_latest'))}")
        print(f"Peak dust:                 {fmt_int(r.get('N_dust_peak'))}")

        print()
        print(f"Sync points:               {fmt_int(r.get('sync_points'))}")
        print(f"Domain decompositions:     {fmt_int(r.get('domain_decompositions'))}")
        print(f"FOF executions:            {fmt_int(r.get('FOF_executions'))}")
        print(f"SUBFIND executions:        {fmt_int(r.get('SUBFIND_executions'))}")

        print()
        print(
            f"Gravity active updates:    "
            f"{fmt_int(r.get('gravity_active_updates'))}"
        )
        print(
            f"Hydro active updates:      "
            f"{fmt_int(r.get('hydro_active_updates'))}"
        )
        print(
            f"Total active updates:      "
            f"{fmt_int(r.get('total_active_updates'))}"
        )

        print(
            f"Hydro updates / initial gas: "
            f"{fmt_float(r.get('hydro_updates_per_initial_gas'), '.1f')}"
        )
        print(
            f"Grav updates / initial HRDM: "
            f"{fmt_float(r.get('gravity_updates_per_initial_HRDM'), '.1f')}"
        )

        print()
        print(
            f"Snapshot wall span:        "
            f"{fmt_float(r.get('runtime_hours_snapshot_span'), '.2f')} hr"
        )
        print(
            f"Seconds / sync:            "
            f"{fmt_float(r.get('seconds_per_sync'), '.3f')}"
        )
        print(
            f"Seconds / million updates: "
            f"{fmt_float(r.get('seconds_per_million_active_updates'), '.3f')}"
        )

        print()
        print(
            f"Feedback log lines:        "
            f"{fmt_int(r.get('feedback_log_lines'))}"
        )
        print(
            f"Dust log lines:            "
            f"{fmt_int(r.get('dust_log_lines'))}"
        )

        print()
        print("Debug levels:")
        for key in DEBUG_KEYS:
            print(f"  {key:<28} {r.get(key)}")


def write_csv(rows, filename):
    keys = []

    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit computational cost of CosmicGrain 12-halo suite."
    )

    parser.add_argument(
        "--root",
        default=".",
        help="GADGET-4 root containing halo*_output_* directories",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--halos",
        nargs="*",
        default=None,
        help="Optional subset, e.g. --halos CG07 CG08 CG10",
    )

    parser.add_argument(
        "--csv",
        default="cosmicgrain_suite_runtime_audit.csv",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    wanted = set(args.halos) if args.halos else set(SUITE)

    rows = []

    for label, meta in SUITE.items():
        if label not in wanted:
            continue

        print(
            f"Auditing {label} / halo{meta['halo']}...",
            flush=True,
        )

        row = audit_run(
            label,
            meta,
            root,
            args.resolution,
        )

        rows.append(row)

    print_table(rows)
    print_detailed(rows)

    write_csv(rows, args.csv)

    print()
    print(f"Wrote: {args.csv}")


if __name__ == "__main__":
    main()
