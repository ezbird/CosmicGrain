#!/usr/bin/env python3
"""
compute_coag_stats.py
========================
Coagulation analysis, combining TWO complementary sources -- but with a
DIFFERENT structure from every other compute_*_stats.py script tonight,
because coagulation never destroys a superparticle (mass is conserved;
only grain radius changes), so there is NO dust_log_taskN.txt event type
for it at all (see dust.cc's event-type list: 0-5 cover sputtering,
shock, astration, sublimation, cleanup, shattering -- nothing for
coagulation). There is no "Part 1: complete destruction record" for this
channel, because coagulation doesn't destroy anything.

  1. [COAG_HIST] lines: a COMPLETE, MPI-reduced, cumulative n_H / n_eff
     histogram of every coagulation event across the WHOLE run, built
     specifically (per dust_particle_log.cc's header) because per-event
     rows would be "enormous at 2048^3". Unlike every other channel's
     stdout diagnostic tonight, this is NOT a sparse sample -- it is a
     complete accounting, just binned (12 log-spaced bins from
     0.01-3.16e3 cm^-3) rather than itemized per-event. This is the
     PRIMARY, trustworthy source for "how many coagulation events
     occurred, and at what density".

  2. [COAGULATION] lines: sparse, itemized per-event detail (grain size
     before/after, swept fraction, local tau, clumping factor) for the
     first 100 events then every 10000th (SAME cadence as shattering --
     see compute_shatter_stats.py's caveat about early-block vs
     periodic-block pooling, which applies identically here). Useful for
     understanding the MECHANISM (typical swept fractions, how much
     clumping factor varies event-to-event) but should not be treated as
     a representative sample of when/where coagulation occurs -- use the
     histogram for that.

============================================================================
IMPORTANT: COAG_HIST CAN PLATEAU -- CHECK FOR THIS EXPLICITLY
============================================================================
Because [COAG_HIST] is cumulative and MPI-reduced, repeated identical
"Cumulative coagulation events" totals across consecutive printed
snapshots mean coagulation has STOPPED ACCUMULATING new events during
that interval -- not a logging bug, a real statement about when/whether
this channel is still active. This script detects and reports any such
plateau explicitly rather than silently treating the LAST snapshot as
"the total" without comment.
============================================================================

USAGE
-----
  python compute_coag_stats.py --stdout-log ../S8_output_1024/output_S8_1024.log
"""

import argparse
import re
import sys

import numpy as np

# -----------------------------------------------------------------------
# [COAG_HIST] parsing -- complete, cumulative, MPI-reduced histogram
# -----------------------------------------------------------------------
# CONFIRMED format (verified against real S10_output_1024 log sample):
#   [COAG_HIST|a=0.8902] Cumulative coagulation events: 328521
#   [COAG_HIST]  n_H edges (cm^-3):      0.01    0.0316 ... (12 values)
#   [COAG_HIST]  raw n_H counts   :         0         0 ... (12 values)
#   [COAG_HIST]  n_eff counts     :         0         0 ... (12 values)
#   [COAG_HIST]  Peak raw n_H bin : ~32 cm^-3  | Peak n_eff bin: ~1e+03 cm^-3
HIST_HEADER_RE = re.compile(
    r'\[COAG_HIST\|a=([\d.eE+-]+)\]\s+Cumulative coagulation events:\s+(\d+)'
)
HIST_EDGES_RE  = re.compile(r'\[COAG_HIST\]\s+n_H edges \(cm\^-3\):\s+(.+)')
HIST_RAW_RE    = re.compile(r'\[COAG_HIST\]\s+raw n_H counts\s+:\s+(.+)')
HIST_NEFF_RE   = re.compile(r'\[COAG_HIST\]\s+n_eff counts\s+:\s+(.+)')


def parse_coag_hist(stdout_log, verbose=True):
    """
    Parse all [COAG_HIST] snapshots. Returns a list of dicts (one per
    printed snapshot, in file order) with keys: a, cumulative_events,
    edges, raw_counts, neff_counts. Also detects and reports plateaus
    (consecutive snapshots with identical cumulative_events).
    """
    snapshots = []
    pending_a = None
    pending_cum = None
    pending_edges = None
    pending_raw = None

    with open(stdout_log, errors='replace') as f:
        for line in f:
            m_hdr = HIST_HEADER_RE.search(line)
            if m_hdr:
                pending_a   = float(m_hdr.group(1))
                pending_cum = int(m_hdr.group(2))
                pending_edges = None
                pending_raw   = None
                continue

            m_edges = HIST_EDGES_RE.search(line)
            if m_edges and pending_a is not None:
                pending_edges = [float(x) for x in m_edges.group(1).split()]
                continue

            m_raw = HIST_RAW_RE.search(line)
            if m_raw and pending_a is not None:
                pending_raw = [int(x) for x in m_raw.group(1).split()]
                continue

            m_neff = HIST_NEFF_RE.search(line)
            if m_neff and pending_a is not None and pending_edges is not None \
               and pending_raw is not None:
                neff_counts = [int(x) for x in m_neff.group(1).split()]
                snapshots.append(dict(
                    a=pending_a, cumulative_events=pending_cum,
                    edges=pending_edges, raw_counts=pending_raw,
                    neff_counts=neff_counts,
                ))
                pending_a = None
                continue

    if verbose:
        print(f"[COAG_HIST] Parsed {len(snapshots)} snapshot(s)")
        if snapshots:
            # Detect plateaus: consecutive snapshots with identical cumulative count
            plateau_start = None
            n_plateaus = 0
            for i in range(1, len(snapshots)):
                same = snapshots[i]['cumulative_events'] == snapshots[i-1]['cumulative_events']
                if same and plateau_start is None:
                    plateau_start = i - 1
                elif not same and plateau_start is not None:
                    n_plateaus += 1
                    a_start = snapshots[plateau_start]['a']
                    a_end   = snapshots[i-1]['a']
                    print(f"  PLATEAU detected: cumulative_events unchanged "
                          f"({snapshots[plateau_start]['cumulative_events']}) "
                          f"from a={a_start:.4f} to a={a_end:.4f} "
                          f"({i - plateau_start} consecutive snapshots) -- "
                          f"coagulation produced ZERO new events in this interval.")
                    plateau_start = None
            if plateau_start is not None:
                a_start = snapshots[plateau_start]['a']
                a_end   = snapshots[-1]['a']
                print(f"  PLATEAU detected (extends to end of log): "
                      f"cumulative_events unchanged "
                      f"({snapshots[plateau_start]['cumulative_events']}) "
                      f"from a={a_start:.4f} to a={a_end:.4f} -- coagulation "
                      f"produced ZERO new events for the remainder of this log.")

    return snapshots


def summarize_coag_hist(snapshots):
    if not snapshots:
        print("  No [COAG_HIST] snapshots found.")
        return

    final = snapshots[-1]
    print(f"\n  Final snapshot: a={final['a']:.4f}  "
          f"total cumulative events={final['cumulative_events']:,}")

    edges = final['edges']
    raw   = final['raw_counts']
    neff  = final['neff_counts']

    print(f"\n  n_H distribution (RAW, not clumping-boosted):")
    total_raw = sum(raw)
    for i, (lo, n) in enumerate(zip(edges, raw)):
        if n == 0:
            continue
        hi = edges[i+1] if i + 1 < len(edges) else None
        bin_label = f"{lo:.3g}-{hi:.3g}" if hi else f">{lo:.3g}"
        print(f"    {bin_label:>16s} cm^-3: {n:8,d} ({100*n/total_raw:5.1f}%)")

    print(f"\n  n_eff distribution (clumping-boosted):")
    total_neff = sum(neff)
    for i, (lo, n) in enumerate(zip(edges, neff)):
        if n == 0:
            continue
        hi = edges[i+1] if i + 1 < len(edges) else None
        bin_label = f"{lo:.3g}-{hi:.3g}" if hi else f">{lo:.3g}"
        print(f"    {bin_label:>16s} cm^-3: {n:8,d} ({100*n/total_neff:5.1f}%)")

    # Weighted median/peak bin (using bin LEFT edge as representative value)
    def weighted_stats(edges, counts):
        vals = np.array(edges[:len(counts)])
        wts  = np.array(counts, dtype=float)
        if wts.sum() == 0:
            return None, None
        peak_idx = int(np.argmax(wts))
        cum = np.cumsum(wts) / wts.sum()
        median_idx = int(np.searchsorted(cum, 0.5))
        median_idx = min(median_idx, len(vals) - 1)
        return vals[peak_idx], vals[median_idx]

    peak_raw, med_raw   = weighted_stats(edges, raw)
    peak_neff, med_neff = weighted_stats(edges, neff)
    print(f"\n  Peak bin (raw n_H):  ~{peak_raw:.3g} cm^-3   "
          f"Median bin (raw n_H):  ~{med_raw:.3g} cm^-3")
    print(f"  Peak bin (n_eff):    ~{peak_neff:.3g} cm^-3   "
          f"Median bin (n_eff):    ~{med_neff:.3g} cm^-3")
    if peak_neff is not None and peak_raw is not None and peak_raw > 0:
        print(f"  Clumping shift (peak n_eff / peak raw n_H): "
              f"~{peak_neff/peak_raw:.1f}x")


# -----------------------------------------------------------------------
# [COAGULATION] sparse per-event line parsing
# -----------------------------------------------------------------------
# CONFIRMED format (verified against real S10_output_1024 log sample):
#   [COAGULATION] Event #1: a=12.4→12.5 nm  M=1.686e-07 Msun (conserved)  n_H=67.66 n_eff=2029.85 (C=30) cm^-3  T=2561 K  tau=27.8 Myr  swept_f=8.046e-03
COAG_LINE_RE = re.compile(
    r'\[COAGULATION\]\s+Event\s+#(\d+):\s+'
    r'a=([\d.eE+-]+)\s*(?:→|->)\s*([\d.eE+-]+)\s*nm\s+'
    r'M=([\d.eE+-]+)\s*Msun\s*\(conserved\)\s+'
    r'n_H=([\d.eE+-]+)\s+'
    r'n_eff=([\d.eE+-]+)\s*\(C=([\d.eE+-]+)\)\s*cm\^-3\s+'
    r'T=([\d.eE+-]+)\s*K\s+'
    r'tau=([\d.eE+-]+)\s*Myr\s+'
    r'swept_f=([\d.eE+-]+)'
)
PREFIX_T_RE = re.compile(r'\[DUST\|T=(\d+)\|')
EARLY_BLOCK_CUTOFF = 100
PERIODIC_INTERVAL  = 10000


def print_unmatched_sample(stdout_log, n=15):
    print(f"\n[Format check] First {n} lines containing 'COAGULATION' "
          f"(excluding COAG_HIST/COAG_DIAG) in {stdout_log}:\n")
    count = 0
    with open(stdout_log, errors='replace') as f:
        for line in f:
            if '[COAGULATION]' in line:
                print(f"  {line.rstrip()}")
                count += 1
                if count >= n:
                    break
    if count == 0:
        print("  (no matching lines found)")


def aggregate_coag_events(stdout_log, verbose=True):
    early, periodic = [], []
    n_lines_seen = 0
    per_task_max = {}

    with open(stdout_log, errors='replace') as f:
        for line in f:
            if '[COAGULATION]' not in line:
                continue
            n_lines_seen += 1
            m = COAG_LINE_RE.search(line)
            if not m:
                continue
            m_t = PREFIX_T_RE.search(line)
            task_id = int(m_t.group(1)) if m_t else -1

            event_num = int(m.group(1))
            record = dict(
                event_num=event_num, task_id=task_id,
                a_old=float(m.group(2)), a_new=float(m.group(3)),
                mass_msun=float(m.group(4)),
                n_H=float(m.group(5)), n_eff=float(m.group(6)),
                clumping_factor=float(m.group(7)),
                T_gas=float(m.group(8)), tau_coag_myr=float(m.group(9)),
                swept_fraction=float(m.group(10)),
            )
            per_task_max[task_id] = max(per_task_max.get(task_id, 0), event_num)
            if event_num <= EARLY_BLOCK_CUTOFF:
                early.append(record)
            else:
                periodic.append(record)

    if verbose:
        print(f"\n[COAGULATION events] Saw {n_lines_seen} line(s) across "
              f"{len(per_task_max)} task(s); {len(early)} early-block, "
              f"{len(periodic)} periodic-block")
        if n_lines_seen > 0 and not early and not periodic:
            print(f"  WARNING: lines found but none matched COAG_LINE_RE -- "
                  f"re-run with --print-unmatched-sample.")

    return dict(early=early, periodic=periodic, per_task_max=per_task_max)


def _summarize_event_block(records, label):
    if not records:
        print(f"  {label}: no events")
        return
    a_old = np.array([r['a_old'] for r in records])
    a_new = np.array([r['a_new'] for r in records])
    swept = np.array([r['swept_fraction'] for r in records])
    cf    = np.array([r['clumping_factor'] for r in records])
    n_eff = np.array([r['n_eff'] for r in records])
    tau   = np.array([r['tau_coag_myr'] for r in records])
    growth_pct = 100 * (a_new - a_old) / a_old

    print(f"  {label}: N={len(records)}")
    print(f"    Size growth per event: mean={growth_pct.mean():.3f}%  "
          f"median={np.median(growth_pct):.3f}%  max={growth_pct.max():.3f}%")
    print(f"    Swept fraction:        mean={swept.mean():.4f}  "
          f"median={np.median(swept):.4f}  max={swept.max():.4f}")
    print(f"    Clumping factor (C):   mean={cf.mean():.1f}  "
          f"unique values seen: {sorted(set(cf.tolist()))}")
    print(f"    n_eff (cm^-3):         mean={n_eff.mean():.1f}  "
          f"median={np.median(n_eff):.1f}  max={n_eff.max():.1f}")
    print(f"    tau_coag (Myr):        mean={tau.mean():.1f}  "
          f"median={np.median(tau):.1f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout-log", required=True,
                     help="Main run stdout log file")
    ap.add_argument("--print-unmatched-sample", action="store_true")
    args = ap.parse_args()

    if args.print_unmatched_sample:
        print_unmatched_sample(args.stdout_log)
        return

    print("=" * 72)
    print("COAGULATION SUMMARY")
    print("=" * 72)

    print("\n--- Source 1: [COAG_HIST] complete cumulative histogram "
          "(PRIMARY, trustworthy total) ---")
    snapshots = parse_coag_hist(args.stdout_log)
    summarize_coag_hist(snapshots)

    print("\n--- Source 2: [COAGULATION] sparse itemized events "
          "(mechanism detail only -- NOT representative sampling, "
          "same early/periodic caveat as shattering) ---")
    events = aggregate_coag_events(args.stdout_log)
    _summarize_event_block(events['early'],    "Early block   (event<=100)")
    _summarize_event_block(events['periodic'], "Periodic block (every 10000th)")

    if snapshots and events['per_task_max']:
        final_total = snapshots[-1]['cumulative_events']
        print(f"\n--- Cross-check ---")
        print(f"  [COAG_HIST] reports {final_total:,} TOTAL cumulative events "
              f"(complete, all tasks, MPI-reduced)")
        print(f"  [COAGULATION] sparse log shows individual tasks reaching "
              f"local event_num up to {max(events['per_task_max'].values())} "
              f"-- this is each task's OWN local counter (not the global "
              f"total), consistent with [COAG_HIST] being the correct "
              f"source for any global count quoted in the paper.")


if __name__ == "__main__":
    main()
