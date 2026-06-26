#!/usr/bin/env python3
"""
compute_growth_stats.py
==========================
Aggregate HK11 grain-growth diagnostic events from stdout log files
(the [HK11_GROWTH] lines), computing per-species statistics: fractional
mass growth per event, accretion timescale, n_eff distribution, and
absolute mass gained.

============================================================================
CRITICAL CAVEAT -- READ BEFORE TRUSTING ANY "TOTAL MASS" NUMBER
============================================================================
Unlike the astration dust_log_taskN.txt files (which record EVERY
astration event completely), the [HK11_GROWTH] stdout lines are printed
only once every 10,000 attempted growth events (confirmed directly from
the sample: event numbers increment by exactly 10000 between consecutive
printed events, e.g. #33010000, #33020000, ...). This script can compute
accurate PER-EVENT statistics (mean fractional growth, mean tau_acc,
species ratios) from the sampled events, since those are simple averages
and a representative 1-in-10000 sample gives a reasonable estimate of
them, PROVIDED growth conditions are not systematically different at the
specific events that happen to land on a multiple of 10000 (unlikely,
but not something this script can verify).

This script CANNOT compute a true total accreted mass the way
compute_astration_mass.py computed a true total astrated mass, because
the dust_log_taskN.txt files contain a COMPLETE record while this stdout
log is a SPARSE SAMPLE. If you naively multiply summed sampled mass by
10000, you get an EXTRAPOLATED total mass, not a measured one -- this
script reports that extrapolation but labels it clearly as such, and
the extrapolation will be wrong if growth rates vary systematically with
cosmic time/environment in ways a uniform 1-in-10000 sample doesn't
evenly capture (e.g. if growth events cluster in bursts).

If a COMPLETE per-event growth log exists elsewhere (a dust_log_taskN.txt
equivalent with growth event rows, or a dedicated growth log directory),
use that instead for any total-mass figure quoted in the paper -- this
script's extrapolated total is a rough estimate only, suitable for an
order-of-magnitude sanity check, not for a precise quoted number.
============================================================================

INPUT FORMAT
------------
Three-line stdout blocks of the form:
  [DUST|T=<task>|a=<a> z=<z>] [HK11_GROWTH] Event #<n>: species=<sp> CF=<cf> f_mol=<f> n_H=<nh> -> n_eff=<neff> cm^-3 (C=<c>)
  [DUST|T=<task>|a=<a> z=<z>] [HK11_GROWTH] tau_acc=<tau> yr | n_eff=<neff2> cm^-3 T_eff=<teff> K Z=<z_gas>
  [DUST|T=<task>|a=<a> z=<z>] [HK11_GROWTH] a=<a_old>-><a_new> nm | dm=<dm> (M: <m_old>-><m_new>) | Z=<z_old>-><z_new>

The three lines for one event are tied together by appearing consecutively
with the same [DUST|T=...|a=... z=...] prefix; this script groups them by
that assumption (i.e. expects them in order, one event's 3 lines never
interleaved with another event's lines -- true for single-threaded stdout
output, may break if multiple MPI ranks write to a SHARED stdout stream
with interleaved buffering; check your log structure if results look
inconsistent).

USAGE
-----
  python compute_growth_stats.py --logfile path/to/output_S5_1024.log
  python compute_growth_stats.py --logfile path/to/output_S5_1024.log --a-min 0.5 --a-max 0.6
"""

import argparse
import re
import sys
from collections import defaultdict

import numpy as np

EVENT_LINE_RE = re.compile(
    r'\[HK11_GROWTH\]\s+Event\s+#(\d+):\s+species=(\w+)\s+CF=([\d.]+)\s+'
    r'f_mol=([\d.]+)\s+n_H=([\d.eE+-]+)\s*(?:→|->)\s*n_eff=([\d.eE+-]+)'
)
TAU_LINE_RE = re.compile(
    r'\[HK11_GROWTH\]\s+tau_acc=([\d.eE+-]+)\s+yr\s*\|\s*n_eff=([\d.eE+-]+)\s*'
    r'cm\^-3\s+T_eff=([\d.eE+-]+)\s+K\s+Z=([\d.eE+-]+)'
)
MASS_LINE_RE = re.compile(
    r'\[HK11_GROWTH\]\s+a=([\d.eE+-]+)\s*(?:→|->)\s*([\d.eE+-]+)\s+nm\s*\|\s*'
    r'dm=([\d.eE+-]+)\s*\(M:\s*([\d.eE+-]+)\s*(?:→|->)\s*([\d.eE+-]+)\)'
)
PREFIX_RE = re.compile(r'\[DUST\|T=(\d+)\|a=([\d.eE+-]+)\s+z=([\d.eE+-]+)\]')


def parse_growth_log(logfile, a_min=None, a_max=None):
    """
    Parse a stdout log file for [HK11_GROWTH] three-line event blocks.
    Returns a dict of per-event records: list of dicts with keys
    species, cf, f_mol, n_H, n_eff, tau_acc, T_eff, Z_gas, a_old, a_new,
    dm, m_old, m_new, a_scale, z.

    Lines are matched independently (not strictly assuming adjacency),
    then re-paired by event number proximity -- since the event number
    only appears on the FIRST of the three lines, this script instead
    pairs sequentially: each "Event #" line starts a new record, and the
    next "tau_acc=" line and next "dm=" line encountered are assumed to
    belong to it. This matches the sample format exactly (3 lines always
    appear together, in order) but will silently mispair records if any
    of the three lines for an event is ever missing (e.g. truncated by
    a concurrent write) -- the n_records_incomplete counter below tracks
    how often a started record never got all 3 lines.
    """
    records = []
    n_incomplete = 0
    pending = None

    with open(logfile, errors='replace') as f:
        for line in f:
            m_evt = EVENT_LINE_RE.search(line)
            if m_evt:
                if pending is not None:
                    n_incomplete += 1
                m_prefix = PREFIX_RE.search(line)
                a_scale = float(m_prefix.group(2)) if m_prefix else None
                z_val = float(m_prefix.group(3)) if m_prefix else None
                pending = dict(
                    event_num=int(m_evt.group(1)),
                    species=m_evt.group(2),
                    cf=float(m_evt.group(3)),
                    f_mol=float(m_evt.group(4)),
                    n_H=float(m_evt.group(5)),
                    n_eff_1=float(m_evt.group(6)),
                    a_scale=a_scale,
                    z=z_val,
                )
                continue

            m_tau = TAU_LINE_RE.search(line)
            if m_tau and pending is not None and 'tau_acc' not in pending:
                pending['tau_acc'] = float(m_tau.group(1))
                pending['n_eff_2'] = float(m_tau.group(2))
                pending['T_eff'] = float(m_tau.group(3))
                pending['Z_gas'] = float(m_tau.group(4))
                continue

            m_mass = MASS_LINE_RE.search(line)
            if m_mass and pending is not None and 'dm' not in pending:
                pending['a_old'] = float(m_mass.group(1))
                pending['a_new'] = float(m_mass.group(2))
                pending['dm'] = float(m_mass.group(3))
                pending['m_old'] = float(m_mass.group(4))
                pending['m_new'] = float(m_mass.group(5))

                # Record is complete -- apply a-range filter and commit.
                if (a_min is None or pending['a_scale'] >= a_min) and \
                   (a_max is None or pending['a_scale'] < a_max):
                    records.append(pending)
                pending = None
                continue

    if pending is not None:
        n_incomplete += 1

    return records, n_incomplete


def summarize(records, sample_interval=10000):
    """Compute per-species summary statistics from parsed event records."""
    by_species = defaultdict(list)
    for r in records:
        by_species[r['species']].append(r)

    print(f"\nTotal complete event records parsed: {len(records)}")
    print(f"Sample interval (events between printed lines): {sample_interval}")
    print(f"Implied total attempted events represented by this sample: "
          f"~{len(records) * sample_interval:,}")

    overall_dm_sum = sum(r['dm'] for r in records)
    print(f"\n=== EXTRAPOLATED total mass gained (CAVEAT: see module "
          f"docstring -- this is sampled_mass_sum * {sample_interval}, "
          f"NOT a measured total) ===")
    print(f"  Sum of sampled dm: {overall_dm_sum:.6e} code units")
    print(f"  Extrapolated total: {overall_dm_sum * sample_interval:.6e} "
          f"code units (rough estimate only)")

    print(f"\n=== Per-species statistics (from sampled events only) ===")
    for species, recs in sorted(by_species.items()):
        n = len(recs)
        dm_arr = np.array([r['dm'] for r in recs])
        mold_arr = np.array([r['m_old'] for r in recs])
        tau_arr = np.array([r['tau_acc'] for r in recs])
        neff_arr = np.array([r['n_eff_2'] for r in recs])
        frac_growth = dm_arr / mold_arr

        print(f"\n  Species: {species}  (N={n} sampled events)")
        print(f"    Mean dm (absolute, code units):     {dm_arr.mean():.4e}")
        print(f"    Mean fractional growth per event:   {frac_growth.mean():.4e}")
        print(f"    Median fractional growth per event: {np.median(frac_growth):.4e}")
        print(f"    Mean tau_acc (yr):                  {tau_arr.mean():.4e}")
        print(f"    Mean n_eff (cm^-3):                 {neff_arr.mean():.2f}")
        print(f"    Median n_eff (cm^-3):                {np.median(neff_arr):.2f}")

    species_list = sorted(by_species.keys())
    if len(species_list) == 2:
        s1, s2 = species_list
        frac1 = np.array([r['dm']/r['m_old'] for r in by_species[s1]])
        frac2 = np.array([r['dm']/r['m_old'] for r in by_species[s2]])
        mean1, mean2     = frac1.mean(), frac2.mean()
        median1, median2 = np.median(frac1), np.median(frac2)
        print(f"\n=== Species comparison ===")
        print(f"  MEAN fractional growth ratio   ({s2}/{s1}): {mean2/mean1:.3f}x")
        print(f"  MEDIAN fractional growth ratio ({s2}/{s1}): {median2/median1:.3f}x")
        print(f"\n  NOTE: if the mean and median ratios differ substantially")
        print(f"  (as they often will -- growth rates are typically heavily")
        print(f"  right-skewed by a tail of high-n_eff events), prefer the")
        print(f"  MEDIAN ratio when reporting a representative per-species")
        print(f"  growth-rate comparison. The mean ratio is driven")
        print(f"  disproportionately by rare high-density events and can")
        print(f"  overstate the TYPICAL species difference by an order of")
        print(f"  magnitude or more. Check this directly: each species'")
        print(f"  own mean-to-median ratio below shows how skewed its")
        print(f"  distribution is internally:")
        print(f"    {s1}: mean/median = {mean1/median1:.2f}x")
        print(f"    {s2}: mean/median = {mean2/median2:.2f}x")
        print(f"\n  Also compare the mean/median n_eff actually sampled for")
        print(f"  each species (printed above) -- a large mean ratio")
        print(f"  combined with SIMILAR median n_eff between species")
        print(f"  (as opposed to differing n_eff) points to the skew being")
        print(f"  driven by a handful of high-density outlier events for")
        print(f"  one species rather than a systematic prefactor effect.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logfile", required=True,
                     help="Path to the stdout log file containing "
                          "[HK11_GROWTH] lines (e.g. output_S5_1024.log)")
    ap.add_argument("--a-min", type=float, default=None,
                     help="Only include events with scale factor >= this")
    ap.add_argument("--a-max", type=float, default=None,
                     help="Only include events with scale factor < this")
    ap.add_argument("--sample-interval", type=int, default=10000,
                     help="Events between printed log lines (default 10000, "
                          "confirmed from the event-number increments in "
                          "the sample log -- verify this matches your run "
                          "if event numbers increment differently).")
    args = ap.parse_args()

    try:
        records, n_incomplete = parse_growth_log(
            args.logfile, a_min=args.a_min, a_max=args.a_max)
    except FileNotFoundError:
        print(f"ERROR: could not find {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if n_incomplete > 0:
        print(f"WARNING: {n_incomplete} event(s) started but never got all "
              f"3 lines (possibly truncated by log rotation/interleaving) "
              f"-- these were dropped, not counted.")

    if not records:
        print("No complete [HK11_GROWTH] event records found. Check the "
              "log format matches what this script expects (see module "
              "docstring), or that --a-min/--a-max aren't excluding "
              "everything.")
        sys.exit(1)

    summarize(records, sample_interval=args.sample_interval)


if __name__ == "__main__":
    main()
