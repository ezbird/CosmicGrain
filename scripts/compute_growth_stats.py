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
only once every 10,000 attempted growth events (event numbers increment
by exactly 10000 between consecutive printed events, e.g. #33890000,
#33900000, ...). This script can compute accurate PER-EVENT statistics
(mean fractional growth, mean tau_acc, species ratios) from the sampled
events, since those are simple averages and a representative 1-in-10000
sample gives a reasonable estimate of them, PROVIDED growth conditions
are not systematically different at the specific events that happen to
land on a multiple of 10000 (unlikely, but not something this script can
verify).

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

INPUT FORMAT (fixed -- see below)
----------------------------------
FIX: this script previously expected an obsolete THREE-line block format
("Event #<n>: species=<sp> CF=<cf> ... -> n_eff=...", then a separate
"tau_acc=..." line, then a separate "a=...->..." line) that does not
match the current source. The current [HK11_GROWTH] print statement in
dust.cc emits ONE consolidated line per (sampled) event:

  [DUST|T=<task>|a=<a> z=<z>] [HK11_GROWTH] event=<n> species=<sp>
      carbon_fraction=<cf> f_mol=<f> n_H_cm3=<nh> n_eff_cm3=<neff>
      clumping_factor=<c> tau_acc_yr=<tau> T_eff_K=<teff>
      Z_gas_before=<zb> Z_gas_after=<za> a_nm_old=<aold> a_nm_new=<anew>
      dm_code=<dm> M_dust_before=<mold> M_dust_after=<mnew>

This regex has already been verified against real S5 log lines.

USAGE
-----
  python compute_growth_stats.py --logfile path/to/output_S5_1024.log
  python compute_growth_stats.py --logfile path/to/output_S5_1024.log --a-min 0.5 --a-max 0.6
  python compute_growth_stats.py --logfile path/to/output_S5_1024.log --print-unmatched-sample
"""

import argparse
import re
import sys
from collections import defaultdict

import numpy as np

GROWTH_LINE_RE = re.compile(
    r"\[HK11_GROWTH\] event=(\d+) species=(\w+) carbon_fraction=([\d.]+) f_mol=([\d.]+) "
    r"n_H_cm3=([\deE+\-.]+) n_eff_cm3=([\deE+\-.]+) clumping_factor=([\d.]+) "
    r"tau_acc_yr=([\deE+\-.]+) T_eff_K=([\d.]+) Z_gas_before=([\deE+\-.]+) Z_gas_after=([\deE+\-.]+) "
    r"a_nm_old=([\deE+\-.]+) a_nm_new=([\deE+\-.]+) dm_code=([\deE+\-.]+) "
    r"M_dust_before=([\deE+\-.]+) M_dust_after=([\deE+\-.]+)"
)
PREFIX_RE = re.compile(r'\[DUST\|T=(\d+)\|a=([\d.eE+-]+)\s+z=([\d.eE+-]+)\]')


def parse_growth_log(logfile, a_min=None, a_max=None, print_unmatched_sample=False):
    """
    Parse a stdout log file for single-line [HK11_GROWTH] events.
    Returns (records, n_unmatched_hk11_lines).
    """
    records = []
    n_unmatched = 0
    unmatched_samples = []

    with open(logfile, errors='replace') as f:
        for line in f:
            if "[HK11_GROWTH]" not in line:
                continue

            m = GROWTH_LINE_RE.search(line)
            if not m:
                n_unmatched += 1
                if print_unmatched_sample and len(unmatched_samples) < 5:
                    unmatched_samples.append(line.rstrip("\n"))
                continue

            m_prefix = PREFIX_RE.search(line)
            a_scale = float(m_prefix.group(2)) if m_prefix else None

            (event_num, species, cf, f_mol, n_H, n_eff, clump,
             tau_acc, T_eff, Z_before, Z_after, a_old, a_new, dm,
             m_old, m_new) = m.groups()

            rec = dict(
                event_num=int(event_num), species=species, cf=float(cf),
                f_mol=float(f_mol), n_H=float(n_H), n_eff_2=float(n_eff),
                clumping_factor=float(clump), tau_acc=float(tau_acc),
                T_eff=float(T_eff), Z_gas=float(Z_after),
                a_old=float(a_old), a_new=float(a_new), dm=float(dm),
                m_old=float(m_old), m_new=float(m_new), a_scale=a_scale,
            )

            if (a_min is None or (a_scale is not None and a_scale >= a_min)) and \
               (a_max is None or (a_scale is not None and a_scale < a_max)):
                records.append(rec)

    if print_unmatched_sample and unmatched_samples:
        print(f"\n--- Sample of {len(unmatched_samples)} unmatched [HK11_GROWTH] line(s) ---")
        for s in unmatched_samples:
            print(f"  {s}")
        print()

    return records, n_unmatched


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
    ap.add_argument("--print-unmatched-sample", action="store_true",
                     help="Print up to 5 raw [HK11_GROWTH] lines that were "
                          "found but failed to match the expected format, "
                          "for debugging a future format drift.")
    args = ap.parse_args()

    try:
        records, n_unmatched = parse_growth_log(
            args.logfile, a_min=args.a_min, a_max=args.a_max,
            print_unmatched_sample=args.print_unmatched_sample)
    except FileNotFoundError:
        print(f"ERROR: could not find {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if n_unmatched > 0:
        print(f"WARNING: {n_unmatched} [HK11_GROWTH] line(s) found but did not "
              f"match the expected format -- these were skipped, not counted. "
              f"Re-run with --print-unmatched-sample to see examples.")

    if not records:
        print("No complete [HK11_GROWTH] event records found. Check the "
              "log format matches what this script expects (see module "
              "docstring), or that --a-min/--a-max aren't excluding "
              "everything.")
        sys.exit(1)

    summarize(records, sample_interval=args.sample_interval)


if __name__ == "__main__":
    main()
