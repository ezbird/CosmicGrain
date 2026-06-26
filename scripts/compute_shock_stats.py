#!/usr/bin/env python3
"""
compute_shock_stats.py
=========================
Comprehensive shock-destruction analysis, combining TWO complementary
sources -- neither one alone tells the full story:

  1. dust_log_taskN.txt (event_type == 1 rows): the COMPLETE record of
     every FULL shock destruction (a superparticle fully destroyed by
     shock erosion crossing the minimum-mass floor). Gives true counts,
     true total mass, and a true carbon/silicate split for destructions
     -- the same complete-record guarantee as the astration analysis.

  2. stdout [DUST_SN] diagnostic lines (if present in your run's main
     output log): shock erosion that does NOT fully destroy a grain --
     i.e. partial mass loss -- is NOT recorded in dust_log_taskN.txt at
     all (the dust log's own header says "one row per DESTRUCTION
     event"; partial erosion that doesn't cross the destruction floor
     is not a destruction event). Per this project's own investigation,
     partial erosion is expected to dominate total shock-eroded mass by
     a large factor over full destructions, given f_vol suppression
     from the spatial-hash cell-size issue (see Methods/Discussion).
     Skipping this source would systematically and badly UNDERCOUNT
     total shock-eroded mass.

============================================================================
IMPORTANT -- VERIFY THE STDOUT FORMAT BEFORE TRUSTING PART 2's OUTPUT
============================================================================
This script's stdout-line regex is written from memory of this project's
earlier investigation (a [DUST_SN] line format including M_local,
M_target, M_lost, f_vol, effective_search_radius, cell_size fields) but
has NOT been verified against an actual pasted sample of that log output
in this session. Run with --print-unmatched-sample first to see a
handful of raw lines containing "DUST_SN" or "SN_SHOCK" from your real
log, and update STDOUT_LINE_RE below if it doesn't match -- do NOT trust
Part 2's totals until you've confirmed the regex actually matches real
lines (a silent zero-match result will just report "0 events found",
which is easy to mistake for "shocks really do nothing" rather than
"the regex is wrong").

Part 1 (dust_log_taskN.txt) does NOT have this caveat -- its format is
fully documented and confirmed from real pasted samples tonight.
============================================================================

USAGE
-----
  python compute_shock_stats.py --dust-log-dir ../S7_output_1024/dust_logs \\
      --stdout-log ../S7_output_1024/output_S7_1024.log

  # Dust-log analysis only (skip stdout partial-erosion parsing):
  python compute_shock_stats.py --dust-log-dir ../S7_output_1024/dust_logs

  # Sanity-check the stdout regex against your real log first:
  python compute_shock_stats.py --stdout-log ../S7_output_1024/output_S7_1024.log \\
      --print-unmatched-sample
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

EVENT_TYPE_SHOCK = 1

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
# Part 1: complete dust_log_taskN.txt analysis (full destructions)
# -----------------------------------------------------------------------

def find_task_log_files(log_dir):
    pattern = os.path.join(log_dir, "dust_log_task*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No dust_log_task*.txt files found in {log_dir}.")
    return files


def aggregate_full_destructions(log_dir, a_min=None, a_max=None, verbose=True):
    """
    Sum event_type==1 (SN shock destruction) rows across ALL task files.
    This is the COMPLETE record of full shock destructions only -- it
    does not include partial erosion (see Part 2).
    """
    files = find_task_log_files(log_dir)
    if verbose:
        print(f"[Part 1] Found {len(files)} per-task dust log file(s) in {log_dir}")

    n_events  = 0
    mass_total = 0.0
    n_by_type    = {name: 0 for name in GRAIN_TYPE_NAMES.values()}
    mass_by_type = {name: 0.0 for name in GRAIN_TYPE_NAMES.values()}
    n_parsed, n_skipped = 0, 0

    for fpath in files:
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

                if event_type != EVENT_TYPE_SHOCK:
                    continue
                if a_min is not None and event_a < a_min:
                    continue
                if a_max is not None and event_a >= a_max:
                    continue

                n_events   += 1
                mass_total += mass
                type_name = GRAIN_TYPE_NAMES.get(grain_type, f"unknown_{grain_type}")
                if type_name not in n_by_type:
                    n_by_type[type_name] = 0
                    mass_by_type[type_name] = 0.0
                n_by_type[type_name]    += 1
                mass_by_type[type_name] += mass

    return dict(
        n_events=n_events, mass_total_code=mass_total,
        n_by_grain_type=n_by_type, mass_by_grain_type=mass_by_type,
        n_files=len(files), n_parsed=n_parsed, n_skipped=n_skipped,
    )


# -----------------------------------------------------------------------
# Part 2: stdout [DUST_SN] partial-erosion analysis
# -----------------------------------------------------------------------
# CONFIRMED format (verified against real S7_output_1024 log sample):
#   [DUST|T=0|a=0.0833486 z=10.998] [DUST_SN] physical_r=0.0538 kpc  search_r=3.000 kpc  v=70.1 km/s  f_vol=5.767e-06  eff=0.172  M_local=6.140e-07  M_target=6.101e-13  M_lost=6.101e-13 Msun  n_dust=7  destroyed=0  eroded=7
#
# IMPORTANT: M_lost (and M_local, M_target) are reported directly in
# Msun in this line -- NOT code units. Do NOT apply unit_mass_g/h
# conversion to these fields; they are already physical Msun.
STDOUT_LINE_RE = re.compile(
    r'\[DUST_SN\]\s+physical_r=([\d.eE+-]+)\s*kpc\s+'
    r'search_r=([\d.eE+-]+)\s*kpc\s+'
    r'v=([\d.eE+-]+)\s*km/s\s+'
    r'f_vol=([\d.eE+-]+)\s+'
    r'eff=([\d.eE+-]+)\s+'
    r'M_local=([\d.eE+-]+)\s*(?:code\s*\(([\d.eE+-]+)\s*Msun\))?\s+'
    r'M_target=([\d.eE+-]+)\s*(?:code\s*\(([\d.eE+-]+)\s*Msun\))?\s+'
    r'M_lost=([\d.eE+-]+)\s*(?:Msun|code\s*\(([\d.eE+-]+)\s*Msun\))\s+'
    r'n_dust=(\d+)\s+'
    r'destroyed=(\d+)\s+'
    r'eroded=(\d+)'
)
# Group map: 1=physical_r 2=search_r 3=v 4=f_vol 5=eff
#            6=M_local(code) 7=M_local(Msun, Format B only, else None)
#            8=M_target(code) 9=M_target(Msun, Format B only, else None)
#            10=M_lost(value -- Msun directly in Format A, code units in Format B)
#            11=M_lost(Msun, Format B only, else None)
#            12=n_dust 13=destroyed 14=eroded
# Companion debug line with local vs. Sedov-Taylor comparison density and
# the DENSITY_CAPPED flag, e.g.:
#   [SN_SHOCK_DEBUG] Call #1: physical_r=0.054 kpc  search_r=3.000 kpc  v_shock=70.1 km/s  rho_local=1.387e-23  rho_sedov=8.363e-25 [DENSITY_CAPPED]
DEBUG_LINE_RE = re.compile(
    r'\[SN_SHOCK_DEBUG\]\s+Call\s+#(\d+):.*?'
    r'rho_local=([\d.eE+-]+)\s+rho_sedov=([\d.eE+-]+)'
    r'(\s*\[DENSITY_CAPPED\])?'
)
# Scale-factor prefix, e.g. "[DUST|T=0|a=0.275 z=2.6]"
PREFIX_A_RE = re.compile(r'a=([\d.eE+-]+)\s')


def print_unmatched_sample(stdout_log, n=15):
    """Print raw lines containing DUST_SN or SN_SHOCK for manual format checking."""
    print(f"\n[Format check] First {n} lines containing 'DUST_SN' or 'SN_SHOCK' "
          f"in {stdout_log}:\n")
    count = 0
    with open(stdout_log, errors='replace') as f:
        for line in f:
            if 'DUST_SN' in line or 'SN_SHOCK' in line:
                print(f"  {line.rstrip()}")
                count += 1
                if count >= n:
                    break
    if count == 0:
        print("  (no matching lines found at all -- check the log file path, "
              "or that this run/rung actually has SN shock destruction "
              "enabled and producing stdout diagnostics)")
    else:
        print(f"\n  Compare these {count} raw line(s) against STDOUT_LINE_RE "
              f"in this script's source -- if the field names/separators "
              f"don't match (e.g. 'M_local=' vs 'M_local: ' vs a different "
              f"field order), edit STDOUT_LINE_RE before trusting --stdout-log "
              f"results.")


def aggregate_partial_erosion(stdout_log, a_min=None, a_max=None, verbose=True):
    """
    Parse [DUST_SN] lines from a run's stdout log for M_lost (partial
    erosion mass per event, ALREADY IN MSUN -- no unit_mass_g/h
    conversion needed for this field). Also parses companion
    [SN_SHOCK_DEBUG] lines for the DENSITY_CAPPED frequency. This is the
    ONLY source for partial-erosion mass -- dust_log_taskN.txt does not
    record it (see module docstring).
    """
    n_events = 0
    mass_total_msun = 0.0
    n_dust_total = 0
    n_destroyed_total = 0
    n_eroded_total = 0
    f_vol_list = []
    physical_r_list = []   # NEW: track for max-blast-radius hash-sizing decision
    n_lines_seen = 0

    n_debug_lines = 0
    n_density_capped = 0

    with open(stdout_log, errors='replace') as f:
        for line in f:
            if 'SN_SHOCK_DEBUG' in line:
                n_debug_lines += 1
                m_dbg = DEBUG_LINE_RE.search(line)
                if m_dbg and m_dbg.group(4):
                    n_density_capped += 1
                continue

            if 'DUST_SN' not in line:
                continue
            n_lines_seen += 1
            m = STDOUT_LINE_RE.search(line)
            if not m:
                continue
            m_a = PREFIX_A_RE.search(line)
            a_val = float(m_a.group(1)) if m_a else None
            if a_val is not None:
                if a_min is not None and a_val < a_min:
                    continue
                if a_max is not None and a_val >= a_max:
                    continue

            f_vol      = float(m.group(4))
            physical_r = float(m.group(1))
            # M_lost: group 10 is Msun directly (Format A) OR code units
            # (Format B, in which case group 11 holds the parenthetical Msun
            # value -- prefer that if present).
            m_lost = float(m.group(11)) if m.group(11) is not None else float(m.group(10))
            n_dust     = int(m.group(12))
            destroyed  = int(m.group(13))
            eroded     = int(m.group(14))

            n_events += 1
            mass_total_msun += m_lost
            n_dust_total += n_dust
            n_destroyed_total += destroyed
            n_eroded_total += eroded
            f_vol_list.append(f_vol)
            physical_r_list.append(physical_r)   # NEW

    if verbose:
        print(f"[Part 2] Saw {n_lines_seen} line(s) containing 'DUST_SN', "
              f"{n_events} matched the expected format")
        if n_lines_seen > 0 and n_events == 0:
            print(f"  WARNING: lines containing 'DUST_SN' were found but NONE "
                  f"matched STDOUT_LINE_RE -- re-run with "
                  f"--print-unmatched-sample and fix STDOUT_LINE_RE.")
        if n_debug_lines > 0:
            print(f"[Part 2] Saw {n_debug_lines} [SN_SHOCK_DEBUG] line(s), "
                  f"{n_density_capped} ({100*n_density_capped/n_debug_lines:.1f}%) "
                  f"flagged [DENSITY_CAPPED]")

    return dict(
        n_events=n_events,
        mass_total_msun=mass_total_msun,   # NOTE: already Msun, not code units
        n_dust_total=n_dust_total,
        n_destroyed_total=n_destroyed_total,
        n_eroded_total=n_eroded_total,
        f_vol_array=np.array(f_vol_list) if f_vol_list else np.array([]),
        physical_r_array=np.array(physical_r_list) if physical_r_list else np.array([]),  # NEW
        n_debug_lines=n_debug_lines,
        n_density_capped=n_density_capped,
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dust-log-dir", default=None,
                     help="Directory with dust_log_task*.txt (Part 1: full "
                          "destructions, complete record)")
    ap.add_argument("--stdout-log", default=None,
                     help="Main run stdout log file (Part 2: partial erosion, "
                          "[DUST_SN] lines)")
    ap.add_argument("--print-unmatched-sample", action="store_true",
                     help="Print raw DUST_SN/SN_SHOCK lines from --stdout-log "
                          "for manual format verification, then exit.")
    ap.add_argument("--a-min", type=float, default=None)
    ap.add_argument("--a-max", type=float, default=None)
    ap.add_argument("--unit-mass-g", type=float, default=1.989e43,
                     help="UnitMass_in_g (default 1e10 Msun/h per this "
                          "project's parameter files)")
    ap.add_argument("--hubble-param", type=float, default=0.6732,
                     help="h, REQUIRED for a correct Msun conversion given "
                          "this project's UnitMass_in_g convention "
                          "('1e10 Msun/h') -- see prior session findings. "
                          "Default 0.6732 matches this project; pass "
                          "explicitly to be sure.")
    args = ap.parse_args()

    if args.print_unmatched_sample:
        if not args.stdout_log:
            print("ERROR: --print-unmatched-sample requires --stdout-log",
                  file=sys.stderr)
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
            full_destr = aggregate_full_destructions(
                args.dust_log_dir, a_min=args.a_min, a_max=args.a_max)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            full_destr = None

    partial = None
    if args.stdout_log:
        if not os.path.exists(args.stdout_log):
            print(f"ERROR: {args.stdout_log} not found", file=sys.stderr)
        else:
            partial = aggregate_partial_erosion(
                args.stdout_log, a_min=args.a_min, a_max=args.a_max)

    print("\n" + "=" * 72)
    print("SHOCK DESTRUCTION SUMMARY")
    print("=" * 72)

    if full_destr is not None:
        print(f"\n--- Part 1: FULL destructions (complete record, "
              f"dust_log_task*.txt) ---")
        print(f"  Total full-destruction events: {full_destr['n_events']}")
        print(f"  Total mass fully destroyed: "
              f"{full_destr['mass_total_code']:.6e} code units = "
              f"{full_destr['mass_total_code'] * msun_per_code:.6e} Msun")
        for type_name, n in full_destr['n_by_grain_type'].items():
            if n == 0:
                continue
            m = full_destr['mass_by_grain_type'][type_name]
            print(f"    {type_name:16s}: {n:6d} events, "
                  f"{m * msun_per_code:.4e} Msun")
    else:
        print("\n--- Part 1: skipped (no --dust-log-dir given or not found) ---")

    if partial is not None:
        print(f"\n--- Part 2: PARTIAL erosion (stdout [DUST_SN] lines, "
              f"NOT in dust log) ---")
        print(f"  Total partial-erosion events matched: {partial['n_events']}")
        print(f"  Total dust particles touched (n_dust, cumulative): "
              f"{partial['n_dust_total']}")
        print(f"  Total eroded (partial, not fully destroyed): "
              f"{partial['n_eroded_total']}")
        print(f"  Total fully destroyed (via this channel's own counter): "
              f"{partial['n_destroyed_total']}")
        print(f"  Total mass eroded (partial, never crossed destruction floor): "
              f"{partial['mass_total_msun']:.6e} Msun "
              f"(reported directly in Msun by the log line -- no unit "
              f"conversion applied)")
        if partial['f_vol_array'].size > 0:
            fv = partial['f_vol_array']
            print(f"  f_vol: mean={fv.mean():.3e}  median={np.median(fv):.3e}  "
                  f"max={fv.max():.3e}")
        if partial['physical_r_array'].size > 0:
            pr = partial['physical_r_array']
            print(f"  physical_r (true blast radius, kpc): "
                  f"mean={pr.mean():.4f}  median={np.median(pr):.4f}  "
                  f"max={pr.max():.4f}  p99={np.percentile(pr, 99):.4f}")
            print(f"\n  HASH SIZING NOTE: if you are sizing a dedicated, finer")
            print(f"  spatial hash for shock destruction (see prior discussion),")
            print(f"  use the MAX (or a safety-padded value above the p99) of")
            print(f"  physical_r printed above as max_search_radius -- NOT the")
            print(f"  mean/median -- since Method 2 in calculate_optimal_cells()")
            print(f"  needs to guarantee adequate resolution for the LARGEST")
            print(f"  blast radius this hash will ever be queried with, not the")
            print(f"  typical one. Re-run this across ALL rungs/resolutions you")
            print(f"  intend to support with the same hash sizing, since shock")
            print(f"  velocities (and therefore radii) may differ across them.")
        if partial['n_debug_lines'] > 0:
            pct = 100 * partial['n_density_capped'] / partial['n_debug_lines']
            print(f"  [DENSITY_CAPPED] frequency: {partial['n_density_capped']}/"
                  f"{partial['n_debug_lines']} calls ({pct:.1f}%)")
    else:
        print("\n--- Part 2: skipped (no --stdout-log given, not found, or "
              "regex unverified -- see warnings above) ---")

    if full_destr is not None and partial is not None and partial['n_events'] > 0:
        full_destr_msun = full_destr['mass_total_code'] * msun_per_code
        total_mass_msun = full_destr_msun + partial['mass_total_msun']
        frac_partial = (partial['mass_total_msun'] / total_mass_msun
                         if total_mass_msun > 0 else float('nan'))
        print(f"\n--- Combined ---")
        print(f"  Full destructions:  {full_destr_msun:.6e} Msun")
        print(f"  Partial erosion:    {partial['mass_total_msun']:.6e} Msun")
        print(f"  Total shock-eroded mass (full + partial): "
              f"{total_mass_msun:.6e} Msun")
        print(f"  Fraction of total from PARTIAL erosion (not in dust log): "
              f"{frac_partial*100:.1f}%")
        print(f"\n  NOTE: if this fraction is large (likely, per this project's")
        print(f"  prior findings), quoting ONLY the dust-log full-destruction")
        print(f"  total in the paper would badly understate total shock")
        print(f"  activity -- use the COMBINED total for any 'how much mass")
        print(f"  has shock destruction removed' claim.")


if __name__ == "__main__":
    main()
