#!/usr/bin/env python3
"""
compute_shatter_stats.py
===========================
Comprehensive shattering analysis, combining TWO complementary sources --
same two-source pattern as compute_shock_stats.py, but with the roles
reversed:

  1. dust_log_taskN.txt (event_type == 5 rows): the COMPLETE record of
     every FULL shattering destruction (a grain whose fragment radius,
     a/3, fell below DUST_MIN_GRAIN_SIZE=1nm, dissolving the superparticle
     entirely). Gives true counts, true total mass, true carbon/silicate
     split for shattering-driven destructions specifically.

  2. stdout [SHATTERING] diagnostic lines: fire on every NON-destructive
     shattering event (grain radius drops by the standard factor of 3 but
     SURVIVES above the floor). This is NOT recorded in dust_log_taskN.txt
     at all -- the dust log only logs the destructive case (event_type=5).
     Skipping this source would tell you nothing about the vastly more
     common "grain shrinks but survives" outcome -- only the comparatively
     rare full-destruction tail.

============================================================================
SAMPLING CAVEAT -- DIFFERENT FROM compute_shock_stats.py / compute_growth_stats.py
============================================================================
The [SHATTERING] stdout print cadence (confirmed from dust.cc) is:
    if(NShatteringEvents <= 100 || NShatteringEvents % 10000 == 0)
i.e. the FIRST 100 events are logged unconditionally, then only every
10,000th event thereafter. This is NEITHER a uniform sample (unlike HK11
growth's flat 1-in-10000) NOR a complete record (unlike the dust log).
The first-100 block is HEAVILY weighted toward early simulation time
(whenever NShatteringEvents was small) and is NOT representative of the
event distribution at later times/different gas conditions. CONCRETELY:
do not pool the first 100 events with the every-10000th events and treat
the combination as a uniform sample -- consider analyzing the early block
(events 1-100) and the periodic block (multiples of 10000) SEPARATELY, or
note explicitly which regime a given printed sample falls in. This script
reports both pieces but does not blend them into one statistic.

This script CANNOT compute a true total "mass eroded via partial
shattering" (analogous to shock's M_lost), because shattering events
shown in the [SHATTERING] line do not print a mass-lost value at all --
mass is conserved during a non-destructive shattering event (only the
radius changes), so there IS no mass to report for those events. The only
mass change shattering produces, in any of the available logs, is the
FULL destruction case in the dust log (event_type=5), which this script
DOES report completely and exactly.

USAGE
-----
  python compute_shatter_stats.py --dust-log-dir ../S9_output_1024/dust_logs \\
      --stdout-log ../S9_output_1024/output_S9_1024.log
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

EVENT_TYPE_SHATTER = 5

GRAIN_TYPE_NAMES = {
    0: "SNII-silicate",
    1: "AGB-carbon",
    2: "mixed",
}

COL_EVENT_A    = 2
COL_MASS       = 10
COL_GRAIN_TYPE = 14
COL_EVENT_TYPE = 15


# -----------------------------------------------------------------------
# Part 1: complete dust_log_taskN.txt analysis (full shattering destructions)
# -----------------------------------------------------------------------

def find_task_log_files(log_dir):
    pattern = os.path.join(log_dir, "dust_log_task*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No dust_log_task*.txt files found in {log_dir}.")
    return files


def aggregate_full_shatter_destructions(log_dir, a_min=None, a_max=None, verbose=True):
    """
    Sum event_type==5 (shattering below minimum size) rows across ALL task
    files. Complete record of FULL shattering-driven destructions only --
    does NOT include the much more common non-destructive shrink-by-3x
    events (see Part 2).

    Also tracks a PER-TASK event count (one dust_log_taskN.txt file per
    task, so this is trivial and exact, unlike the stdout-based per-task
    breakdown in Part 2 which depends on which stdout log files you
    happen to have). This gives a complete-record way to check whether
    shattering DESTRUCTIONS specifically are concentrated on certain
    tasks/regions, complementing (but not replacing) the stdout-based
    check on total CALLS in Part 2 -- a task could show concentrated
    destructions while still making evenly-spread total calls, if its
    region simply has a higher destruction-per-call rate, so these two
    checks answer related but distinct questions.
    """
    files = find_task_log_files(log_dir)
    if verbose:
        print(f"[Part 1] Found {len(files)} per-task dust log file(s) in {log_dir}")

    n_events  = 0
    mass_total_code = 0.0
    n_by_type    = {name: 0 for name in GRAIN_TYPE_NAMES.values()}
    mass_by_type = {name: 0.0 for name in GRAIN_TYPE_NAMES.values()}
    n_parsed, n_skipped = 0, 0
    n_events_per_task = {}   # task_id (from filename) -> shatter-destruction count

    for fpath in files:
        # Extract task ID from filename, e.g. "dust_log_task7.txt" -> 7
        fname = os.path.basename(fpath)
        m_task = re.search(r'task(\d+)', fname)
        task_id = int(m_task.group(1)) if m_task else -1
        n_events_per_task[task_id] = 0

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 16:
                    n_skipped += 1
                    continue
                try:
                    event_a    = float(parts[COL_EVENT_A])
                    mass       = float(parts[COL_MASS])
                    grain_type = int(parts[COL_GRAIN_TYPE])
                    event_type = int(parts[COL_EVENT_TYPE])
                except (ValueError, IndexError):
                    n_skipped += 1
                    continue
                n_parsed += 1

                if event_type != EVENT_TYPE_SHATTER:
                    continue
                if a_min is not None and event_a < a_min:
                    continue
                if a_max is not None and event_a >= a_max:
                    continue

                n_events   += 1
                mass_total_code += mass
                n_events_per_task[task_id] += 1
                type_name = GRAIN_TYPE_NAMES.get(grain_type, f"unknown_{grain_type}")
                if type_name not in n_by_type:
                    n_by_type[type_name] = 0
                    mass_by_type[type_name] = 0.0
                n_by_type[type_name]    += 1
                mass_by_type[type_name] += mass

    if verbose and n_events_per_task:
        counts = np.array(list(n_events_per_task.values()))
        print(f"\n  Per-task shattering-DESTRUCTION counts (from dust_log_taskN.txt, "
              f"complete record):")
        for task_id in sorted(n_events_per_task.keys()):
            print(f"    Task {task_id:3d}: {n_events_per_task[task_id]:6d} destructions")
        if counts.max() > 0:
            print(f"\n  Spread across tasks: min={counts.min()}  max={counts.max()}  "
                  f"(max/min ratio={counts.max()/max(1,counts.min()):.1f}x)")

    return dict(
        n_events=n_events, mass_total_code=mass_total_code,
        n_by_grain_type=n_by_type, mass_by_grain_type=mass_by_type,
        n_files=len(files), n_parsed=n_parsed, n_skipped=n_skipped,
        n_events_per_task=n_events_per_task,
    )


# -----------------------------------------------------------------------
# Part 2: stdout [SHATTERING] non-destructive event analysis
# -----------------------------------------------------------------------
# CONFIRMED format (verified against real S9_output_1024 log sample --
# this is a DIFFERENT format from an earlier version of this code; the
# field names and structure changed at some point between when this
# project's notes were written and the current run):
#   [SHATTERING] Event #1: a=12.7→4.2 nm  n_H=2.77 n_eff=82.97 cm^-3  T=34719 K  v_turb=21.86 km/s  tau=685.1 Myr  P=5.262e-04
SHATTER_LINE_RE = re.compile(
    r'\[SHATTERING\]\s+Event\s+#(\d+):\s+'
    r'a=([\d.eE+-]+)\s*(?:→|->)\s*([\d.eE+-]+)\s*nm\s+'
    r'n_H=([\d.eE+-]+)\s+'
    r'n_eff=([\d.eE+-]+)\s*cm\^-3\s+'
    r'T=([\d.eE+-]+)\s*K\s+'
    r'v_turb=([\d.eE+-]+)\s*km/s\s+'
    r'tau=([\d.eE+-]+)\s*Myr\s+'
    r'P=([\d.eE+-]+)'
)
# Group map: 1=event_num 2=a_old 3=a_new 4=n_H 5=n_eff 6=T_gas
#            7=v_turb 8=tau_shatter_myr 9=shatter_prob
PREFIX_A_RE = re.compile(r'a=([\d.eE+-]+)\s')
# Task ID, e.g. "[DUST|T=3|a=0.275 z=2.6]" -- REQUIRED for correct
# aggregation across multiple tasks, since NShatteringEvents is a
# task-LOCAL static counter in dust.cc: each task independently logs its
# own "first 100" events and its own "every 10000th" events. Pooling
# lines from different tasks without tracking which task each event_num
# belongs to would silently corrupt the early/periodic block split (e.g.
# task 3's event #50 and task 7's event #50 are unrelated events that
# both happen to satisfy event_num<=100, which is fine for the EARLY
# block since the cutoff is task-independent; but a GLOBAL "is this the
# Nth event overall" question cannot be answered without per-task
# numbering, which this script tracks via task_id below).
PREFIX_T_RE = re.compile(r'\[DUST\|T=(\d+)\|')

# [SHAT_DIAG] line, e.g.:
#   [SHAT_DIAG] calls=61200000  failed: vel=3(0.0%) dens=5094052(8.3%) size=4832(0.0%)  passed=4788(0.0%)
SHAT_DIAG_RE = re.compile(
    r'\[SHAT_DIAG\]\s+calls=(\d+)\s+'
    r'failed:\s+vel=(\d+)\([\d.]+%\)\s+'
    r'dens=(\d+)\([\d.]+%\)\s+'
    r'size=(\d+)\([\d.]+%\)\s+'
    r'passed=(\d+)\([\d.]+%\)'
)


def aggregate_shat_diag(stdout_log, verbose=True):
    """
    Parse [SHAT_DIAG] lines (printed every 50000 CALLS, per task, per
    dust.cc) across ALL tasks present in the log. Returns each task's
    FINAL (highest-calls) snapshot -- since these are cumulative
    per-task counters, the last line for a given task already reflects
    that task's running total, so no summing across lines is needed
    (and summing WOULD double-count, since each line already includes
    all prior calls for that task).
    """
    last_per_task = {}  # task_id -> dict(calls, vel, dens, size, passed)

    with open(stdout_log, errors='replace') as f:
        for line in f:
            if '[SHAT_DIAG]' not in line:
                continue
            m = SHAT_DIAG_RE.search(line)
            if not m:
                continue
            m_t = PREFIX_T_RE.search(line)
            task_id = int(m_t.group(1)) if m_t else -1

            last_per_task[task_id] = dict(
                calls=int(m.group(1)), vel=int(m.group(2)),
                dens=int(m.group(3)), size=int(m.group(4)),
                passed=int(m.group(5)),
            )

    if verbose and last_per_task:
        print(f"\n  [SHAT_DIAG] final per-task snapshot ({len(last_per_task)} "
              f"task(s) reporting):")
        total_calls, total_passed = 0, 0
        for task_id in sorted(last_per_task.keys()):
            d = last_per_task[task_id]
            pass_rate = 100.0 * d['passed'] / d['calls'] if d['calls'] > 0 else 0.0
            print(f"    Task {task_id:3d}: calls={d['calls']:10,d}  "
                  f"passed={d['passed']:6,d} ({pass_rate:.4f}%)")
            total_calls  += d['calls']
            total_passed += d['passed']
        if total_calls > 0:
            print(f"    {'TOTAL':>8s}: calls={total_calls:10,d}  "
                  f"passed={total_passed:6,d} "
                  f"({100.0*total_passed/total_calls:.4f}%)")
            # Spread check: how unevenly is 'passed' distributed across tasks?
            passed_vals = np.array([d['passed'] for d in last_per_task.values()])
            if passed_vals.max() > 0:
                print(f"    Spread of 'passed' across tasks: "
                      f"min={passed_vals.min()}  max={passed_vals.max()}  "
                      f"(max/min ratio={passed_vals.max()/max(1,passed_vals.min()):.1f}x "
                      f"-- a large ratio indicates shattering activity IS "
                      f"concentrated on specific tasks)")

    return last_per_task

# The print cadence boundary: events with event_num <= this are from the
# "first 100 unconditional" block; events with event_num > this are from
# the "every 10000th" block. Confirmed from dust.cc's print condition.
EARLY_BLOCK_CUTOFF = 100
PERIODIC_INTERVAL  = 10000


def print_unmatched_sample(stdout_log, n=15):
    print(f"\n[Format check] First {n} lines containing 'SHATTERING' in {stdout_log}:\n")
    count = 0
    with open(stdout_log, errors='replace') as f:
        for line in f:
            if 'SHATTERING' in line and '[SHAT_DIAG]' not in line:
                print(f"  {line.rstrip()}")
                count += 1
                if count >= n:
                    break
    if count == 0:
        print("  (no matching lines found -- check the log path, or that "
              "shattering is enabled and has fired at least once)")


def aggregate_shatter_events(stdout_log, a_min=None, a_max=None, verbose=True):
    """
    Parse [SHATTERING] lines (non-destructive shrink events) from stdout,
    across ALL tasks present in the log (NOT just task 0). Returns
    early-block and periodic-block statistics SEPARATELY, AND a per-task
    breakdown of how many early/periodic events each task contributed --
    since NShatteringEvents is a task-local counter, this per-task
    breakdown is the only way to tell whether shattering activity is
    concentrated on a few tasks (which would explain a persistently-empty
    periodic block even with substantial total shattering activity) or
    genuinely spread evenly (in which case an empty periodic block more
    likely means total activity per task has simply not yet reached
    10,000, e.g. on a still-short or low-activity run).
    """
    early_records, periodic_records = [], []
    n_lines_seen = 0
    per_task_max_event_num = {}   # task_id -> highest event_num seen
    per_task_early_count   = {}   # task_id -> count of early-block events
    per_task_periodic_count = {}  # task_id -> count of periodic-block events

    with open(stdout_log, errors='replace') as f:
        for line in f:
            if '[SHATTERING]' not in line or '[SHAT_DIAG]' in line:
                continue
            n_lines_seen += 1
            m = SHATTER_LINE_RE.search(line)
            if not m:
                continue

            m_t = PREFIX_T_RE.search(line)
            task_id = int(m_t.group(1)) if m_t else -1

            m_a = PREFIX_A_RE.search(line)
            a_val = float(m_a.group(1)) if m_a else None
            if a_val is not None:
                if a_min is not None and a_val < a_min:
                    continue
                if a_max is not None and a_val >= a_max:
                    continue

            event_num   = int(m.group(1))
            record = dict(
                event_num=event_num,
                task_id=task_id,
                a_old=float(m.group(2)),
                a_new=float(m.group(3)),
                n_H=float(m.group(4)),
                n_eff=float(m.group(5)),
                T_gas=float(m.group(6)),
                v_turb=float(m.group(7)),
                tau_shatter_myr=float(m.group(8)),
                shatter_prob=float(m.group(9)),
            )

            per_task_max_event_num[task_id] = max(
                per_task_max_event_num.get(task_id, 0), event_num)

            if event_num <= EARLY_BLOCK_CUTOFF:
                early_records.append(record)
                per_task_early_count[task_id] = per_task_early_count.get(task_id, 0) + 1
            else:
                periodic_records.append(record)
                per_task_periodic_count[task_id] = per_task_periodic_count.get(task_id, 0) + 1

    if verbose:
        n_tasks_seen = len(per_task_max_event_num)
        print(f"[Part 2] Saw {n_lines_seen} line(s) containing '[SHATTERING]' "
              f"across {n_tasks_seen} distinct task(s); "
              f"{len(early_records)} in the early (event<={EARLY_BLOCK_CUTOFF}) "
              f"block, {len(periodic_records)} in the periodic "
              f"(every {PERIODIC_INTERVAL}th) block")
        if n_lines_seen > 0 and not early_records and not periodic_records:
            print(f"  WARNING: lines found but none matched SHATTER_LINE_RE -- "
                  f"re-run with --print-unmatched-sample and check the format.")

        if n_tasks_seen > 0:
            print(f"\n  Per-task breakdown (max event_num reached, i.e. this "
                  f"task's own local NShatteringEvents counter at log's end):")
            for task_id in sorted(per_task_max_event_num.keys()):
                max_evt = per_task_max_event_num[task_id]
                n_early = per_task_early_count.get(task_id, 0)
                n_period = per_task_periodic_count.get(task_id, 0)
                flag = "  <-- reached periodic block!" if max_evt > EARLY_BLOCK_CUTOFF else ""
                print(f"    Task {task_id:3d}: max_event_num={max_evt:6d}  "
                      f"early_logged={n_early:3d}  periodic_logged={n_period:3d}{flag}")
            max_overall = max(per_task_max_event_num.values())
            min_overall = min(per_task_max_event_num.values())
            print(f"\n  Spread across tasks: min max_event_num={min_overall}, "
                  f"max max_event_num={max_overall} "
                  f"(a large spread indicates shattering activity is "
                  f"concentrated on specific tasks/regions rather than spread "
                  f"evenly across the zoom volume)")

    return dict(early=early_records, periodic=periodic_records,
                n_lines_seen=n_lines_seen,
                per_task_max_event_num=per_task_max_event_num,
                per_task_early_count=per_task_early_count,
                per_task_periodic_count=per_task_periodic_count)


def _summarize_block(records, label):
    if not records:
        print(f"  {label}: no events")
        return
    a_old  = np.array([r['a_old'] for r in records])
    a_new  = np.array([r['a_new'] for r in records])
    ratio  = a_new / a_old
    v_turb = np.array([r['v_turb'] for r in records])
    n_eff  = np.array([r['n_eff'] for r in records])
    tau    = np.array([r['tau_shatter_myr'] for r in records])

    print(f"  {label}: N={len(records)}")
    print(f"    a_new/a_old ratio: mean={ratio.mean():.4f}  "
          f"median={np.median(ratio):.4f}  "
          f"(expected ~0.33 from the fixed 1/3 reduction per event)")
    print(f"    v_turb (km/s):     mean={v_turb.mean():.2f}  median={np.median(v_turb):.2f}  "
          f"min={v_turb.min():.2f}  max={v_turb.max():.2f}")
    print(f"    n_eff (cm^-3):     mean={n_eff.mean():.3f}  median={np.median(n_eff):.3f}  "
          f"max={n_eff.max():.3f}")
    print(f"    tau_shatter (Myr): mean={tau.mean():.1f}  median={np.median(tau):.1f}")

    # How many of these were already small enough to be at risk of crossing
    # the 1nm floor on THIS event (a_old < 3nm, since a_old/3 < 1nm requires
    # a_old < 3.0). This directly informs the destruction-floor timing
    # question (was it really "two to three events", or different?).
    near_floor = a_old < 3.0
    if near_floor.any():
        print(f"    Grains with a_old<3nm (would cross 1nm floor THIS event): "
              f"{near_floor.sum()}/{len(records)} ({100*near_floor.sum()/len(records):.1f}%)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dust-log-dir", default=None,
                     help="Directory with dust_log_task*.txt (Part 1: full "
                          "shattering destructions, complete record)")
    ap.add_argument("--stdout-log", default=None,
                     help="Main run stdout log file (Part 2: non-destructive "
                          "shattering events, [SHATTERING] lines)")
    ap.add_argument("--print-unmatched-sample", action="store_true")
    ap.add_argument("--a-min", type=float, default=None)
    ap.add_argument("--a-max", type=float, default=None)
    ap.add_argument("--unit-mass-g", type=float, default=1.989e43,
                     help="UnitMass_in_g (default 1e10 Msun/h per this "
                          "project's parameter files)")
    ap.add_argument("--hubble-param", type=float, default=0.6732,
                     help="h, REQUIRED for correct Msun conversion given "
                          "this project's UnitMass_in_g convention. "
                          "Default 0.6732 matches this project.")
    args = ap.parse_args()

    if args.print_unmatched_sample:
        if not args.stdout_log:
            print("ERROR: --print-unmatched-sample requires --stdout-log", file=sys.stderr)
            sys.exit(1)
        print_unmatched_sample(args.stdout_log)
        return

    if not args.dust_log_dir and not args.stdout_log:
        print("ERROR: supply at least one of --dust-log-dir or --stdout-log",
              file=sys.stderr)
        sys.exit(1)

    msun_per_code = args.unit_mass_g / 1.989e33 / args.hubble_param

    full_destr = None
    if args.dust_log_dir:
        try:
            full_destr = aggregate_full_shatter_destructions(
                args.dust_log_dir, a_min=args.a_min, a_max=args.a_max)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)

    shatter_events = None
    shat_diag = None
    if args.stdout_log:
        if not os.path.exists(args.stdout_log):
            print(f"ERROR: {args.stdout_log} not found", file=sys.stderr)
        else:
            shatter_events = aggregate_shatter_events(
                args.stdout_log, a_min=args.a_min, a_max=args.a_max)
            shat_diag = aggregate_shat_diag(args.stdout_log)

    print("\n" + "=" * 72)
    print("SHATTERING SUMMARY")
    print("=" * 72)

    if full_destr is not None:
        print(f"\n--- Part 1: FULL shattering destructions (complete record, "
              f"dust_log_task*.txt) ---")
        print(f"  Total events: {full_destr['n_events']}")
        print(f"  Total mass destroyed: {full_destr['mass_total_code']:.6e} "
              f"code units = "
              f"{full_destr['mass_total_code'] * msun_per_code:.6e} Msun")
        for type_name, n in full_destr['n_by_grain_type'].items():
            if n == 0:
                continue
            m = full_destr['mass_by_grain_type'][type_name]
            print(f"    {type_name:16s}: {n:6d} events, "
                  f"{m * msun_per_code:.4e} Msun")
    else:
        print("\n--- Part 1: skipped (no --dust-log-dir given or not found) ---")

    if shatter_events is not None:
        print(f"\n--- Part 2: non-destructive shrink events (stdout "
              f"[SHATTERING] lines, NOT in dust log) ---")
        print(f"  NOTE: early block (event<=100) and periodic block "
              f"(every {PERIODIC_INTERVAL}th) are reported SEPARATELY -- "
              f"see module docstring for why they must not be pooled.")
        _summarize_block(shatter_events['early'],    "Early block   (event<=100)")
        _summarize_block(shatter_events['periodic'], "Periodic block (every 10000th)")
    else:
        print("\n--- Part 2: skipped (no --stdout-log given or not found) ---")

    if full_destr is not None and shatter_events is not None:
        n_periodic = len(shatter_events['periodic'])
        n_early    = len(shatter_events['early'])
        if n_periodic > 0:
            implied_total_nondestructive = n_periodic * PERIODIC_INTERVAL
            print(f"\n--- Rough scale comparison ---")
            print(f"  Full destructions (exact):                {full_destr['n_events']}")
            print(f"  Non-destructive events (EXTRAPOLATED from "
                  f"periodic block only, rough estimate): "
                  f"~{implied_total_nondestructive:,}")
            if full_destr['n_events'] > 0:
                ratio = implied_total_nondestructive / full_destr['n_events']
                print(f"  Implied ratio (non-destructive : full destruction): "
                      f"~{ratio:.0f} : 1")
                print(f"  (Treat this ratio as order-of-magnitude only -- the "
                      f"periodic block's extrapolation factor of {PERIODIC_INTERVAL} "
                      f"is itself an approximation, and the early block, excluded "
                      f"here, would bias the count if it dominated.)")


if __name__ == "__main__":
    main()
