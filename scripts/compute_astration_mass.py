#!/usr/bin/env python3
"""
compute_astration_mass.py
============================
Aggregate astration (event_type=2) mass and event counts from the
per-task dust_log_taskN.txt files, summed across ALL tasks -- the same
multi-task aggregation pattern used earlier for shock-destruction
analysis (parse_dust_event_log_all_tasks).

WHY THIS EXISTS: the dust_log_taskN.txt files are split per-MPI-task,
so summing only task 0's file undercounts total astrated mass by a
large factor on a multi-node run. The [ASTRATION]/[ASTRATION_CHECK]/
[ASTRATION_LARGE] lines in the stdout log are a SPARSE DIAGNOSTIC SAMPLE
(only the first 20 calls per task for ASTRATION_CHECK, only every 100th
event for ASTRATION, only large-consumption events for ASTRATION_LARGE)
-- summing those will also undercount, in a different and less
predictable way, since they are not a complete record at all.

This script reads the COMPLETE per-event record from dust_log_taskN.txt
(event_type == 2 rows), which IS a full account of every astration event
across the whole run, once all task files are combined.

USAGE
-----
  python compute_astration_mass.py --log-dir ../S3_output_1024/dust_logs

  # Restrict to events before/after a given scale factor (e.g. to match
  # a specific table row's z=0 snapshot, or to isolate early-universe
  # astration from late-time):
  python compute_astration_mass.py --log-dir ../S3_output_1024/dust_logs --a-max 0.3

OUTPUT
------
Prints: total astration event count, total astrated mass (code units AND
Msun, using UnitMass_in_g if you supply --unit-mass-g), and a breakdown
by grain_type (SNII-silicate vs AGB-carbon vs mixed) since the dust log
already carries that field per-event at no extra cost.
"""

import argparse
import glob
import os
import sys

import numpy as np

EVENT_TYPE_ASTRATION = 2

GRAIN_TYPE_NAMES = {
    0: "SNII-silicate",
    1: "AGB-carbon",
    2: "mixed",
}

# Column indices (0-based) in dust_log_taskN.txt, per the documented format:
#  1 ID  2 birth_a  3 event_a  4-6 birth_xyz  7-9 event_xyz
#  10 displacement  11 mass  12 grain_radius  13 carbon_fraction
#  14 gas_density  15 grain_type  16 event_type
COL_EVENT_A    = 2
COL_MASS       = 10
COL_GRAIN_TYPE = 14
COL_EVENT_TYPE = 15


def find_task_log_files(log_dir):
    """
    Find all dust_log_taskN.txt files in log_dir. Returns sorted list of
    full paths. Raises if none found.
    """
    pattern = os.path.join(log_dir, "dust_log_task*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No dust_log_task*.txt files found in {log_dir}. "
            f"Check the path, or that this run actually wrote per-task "
            f"event logs (open_dust_particle_log() must have been called)."
        )
    return files


def aggregate_astration(log_dir, a_min=None, a_max=None, verbose=True):
    """
    Read every dust_log_taskN.txt in log_dir, sum astration (event_type=2)
    events and mass across ALL tasks, optionally restricted to a scale-
    factor range [a_min, a_max).

    Returns dict with:
        n_events_total      : int
        mass_total_code     : float, code units
        n_by_grain_type      : dict {grain_type_name: count}
        mass_by_grain_type   : dict {grain_type_name: mass in code units}
        n_files_read         : int
        n_lines_parsed       : int
        n_lines_skipped      : int  (malformed/short lines)
    """
    files = find_task_log_files(log_dir)
    if verbose:
        print(f"Found {len(files)} per-task log file(s) in {log_dir}")

    n_events_total  = 0
    mass_total_code = 0.0
    n_by_type    = {name: 0 for name in GRAIN_TYPE_NAMES.values()}
    mass_by_type = {name: 0.0 for name in GRAIN_TYPE_NAMES.values()}
    n_lines_parsed  = 0
    n_lines_skipped = 0

    for fpath in files:
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 16:
                    n_lines_skipped += 1
                    continue
                try:
                    event_a    = float(parts[COL_EVENT_A])
                    mass       = float(parts[COL_MASS])
                    grain_type = int(parts[COL_GRAIN_TYPE])
                    event_type = int(parts[COL_EVENT_TYPE])
                except (ValueError, IndexError):
                    n_lines_skipped += 1
                    continue

                n_lines_parsed += 1

                if event_type != EVENT_TYPE_ASTRATION:
                    continue
                if a_min is not None and event_a < a_min:
                    continue
                if a_max is not None and event_a >= a_max:
                    continue

                n_events_total  += 1
                mass_total_code += mass

                type_name = GRAIN_TYPE_NAMES.get(grain_type, f"unknown_type_{grain_type}")
                if type_name not in n_by_type:
                    n_by_type[type_name] = 0
                    mass_by_type[type_name] = 0.0
                n_by_type[type_name]    += 1
                mass_by_type[type_name] += mass

    return dict(
        n_events_total=n_events_total,
        mass_total_code=mass_total_code,
        n_by_grain_type=n_by_type,
        mass_by_grain_type=mass_by_type,
        n_files_read=len(files),
        n_lines_parsed=n_lines_parsed,
        n_lines_skipped=n_lines_skipped,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True,
                     help="Directory containing dust_log_task*.txt files "
                          "(e.g. ../S3_output_1024/dust_logs)")
    ap.add_argument("--a-min", type=float, default=None,
                     help="Only count events with event_a >= this value")
    ap.add_argument("--a-max", type=float, default=None,
                     help="Only count events with event_a < this value")
    ap.add_argument("--unit-mass-g", type=float, default=1.989e43,
                     help="UnitMass_in_g for code-units-to-Msun conversion "
                          "(default 1.989e43 = 1e10 Msun, the project's "
                          "documented default -- verify against your "
                          "actual parameter file).")
    ap.add_argument("--hubble-param", type=float, default=None,
                     help="If your code mass convention is 1e10 Msun/h "
                          "rather than 1e10 Msun directly, supply h here "
                          "to additionally divide by h. Omit if the dust "
                          "log's 'mass' column is already in absolute "
                          "code mass units with no h dependence -- check "
                          "dust.cc's Sp->P[i].getMass() convention.")
    args = ap.parse_args()

    try:
        result = aggregate_astration(
            args.log_dir, a_min=args.a_min, a_max=args.a_max, verbose=True)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    msun_per_code = args.unit_mass_g / 1.989e33
    if args.hubble_param:
        msun_per_code /= args.hubble_param

    print(f"\nParsed {result['n_lines_parsed']} valid lines across "
          f"{result['n_files_read']} file(s) "
          f"({result['n_lines_skipped']} lines skipped as malformed).")

    print(f"\n=== Astration totals"
          f"{f' (a in [{args.a_min}, {args.a_max})' if (args.a_min or args.a_max) else ''} ===")
    print(f"  Total events: {result['n_events_total']}")
    print(f"  Total mass:   {result['mass_total_code']:.6e} code units "
          f"= {result['mass_total_code'] * msun_per_code:.6e} Msun "
          f"(using UnitMass_in_g={args.unit_mass_g:.3e}"
          f"{f', h={args.hubble_param}' if args.hubble_param else ''})")

    print(f"\n=== Breakdown by grain type ===")
    for type_name in GRAIN_TYPE_NAMES.values():
        n = result['n_by_grain_type'].get(type_name, 0)
        m = result['mass_by_grain_type'].get(type_name, 0.0)
        if n == 0:
            continue
        print(f"  {type_name:16s}: {n:6d} events, "
              f"{m:.6e} code units = {m * msun_per_code:.6e} Msun")

    print(f"\nNOTE: this is the COMPLETE astration record (all tasks, all")
    print(f"events), unlike the sparse [ASTRATION]/[ASTRATION_CHECK]/")
    print(f"[ASTRATION_LARGE] stdout diagnostic lines, which only sample a")
    print(f"fraction of events and would undercount total astrated mass if")
    print(f"summed directly. Use THIS number in the paper, not a sum of the")
    print(f"stdout diagnostic lines.")
    print(f"\nIMPORTANT: verify --unit-mass-g and --hubble-param against your")
    print(f"actual run's parameter file before trusting the Msun conversion --")
    print(f"this script cannot read that from the dust_log_taskN.txt files")
    print(f"themselves, since they only store mass in code units.")


if __name__ == "__main__":
    main()
