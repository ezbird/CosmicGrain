#!/usr/bin/env python3
import argparse, re, sys
from statistics import mean

SYNC_RE = re.compile(r'\bSync-Point\b')
# Match any "... <number> sec" token to accumulate total reported seconds
SEC_RE = re.compile(r'(?<![A-Za-z0-9_.-])([0-9]+(?:\.[0-9]+)?)\s+sec\b')
# DOMAIN imbalance lines, e.g.: "maximum imbalance of 1.0893|1.0894"
IMBAL_RE = re.compile(r'maximum imbalance of\s+([0-9.]+)\|([0-9.]+)', re.IGNORECASE)

def parse_file(path):
    syncs = 0
    total_sec = 0.0
    imbals_left = []
    imbals_right = []

    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                if SYNC_RE.search(line):
                    syncs += 1
                for m in SEC_RE.finditer(line):
                    total_sec += float(m.group(1))
                m = IMBAL_RE.search(line)
                if m:
                    try:
                        l = float(m.group(1))
                        r = float(m.group(2))
                        imbals_left.append(l)
                        imbals_right.append(r)
                    except ValueError:
                        pass
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(2)

    return syncs, total_sec, imbals_left, imbals_right

def main():
    ap = argparse.ArgumentParser(
        description="Gadget-4 log quick summary: average time per sync and average domain imbalance."
    )
    ap.add_argument("logfile", help="Path to Gadget-4 output log")
    args = ap.parse_args()

    syncs, total_sec, imbals_left, imbals_right = parse_file(args.logfile)

    if syncs == 0:
        print("No Sync-Point lines found. Cannot compute avg time per sync.")
        avg_per_sync = float('nan')
    else:
        avg_per_sync = total_sec / syncs

    if imbals_left and imbals_right:
        avg_imb_left = mean(imbals_left)
        avg_imb_right = mean(imbals_right)
        avg_imb_both = (avg_imb_left + avg_imb_right) / 2.0
    else:
        avg_imb_left = avg_imb_right = avg_imb_both = float('nan')

    # Pretty print
    print(f"file: {args.logfile}")
    print(f"sync_points: {syncs}")
    print(f"total_reported_time: {total_sec:.3f} sec")
    print(f"avg_time_per_sync: {avg_per_sync:.3f} sec  "
          f"(proxy = sum of all '* sec' timings / #syncs)")
    if imbals_left and imbals_right:
        print(f"domain_imbalance_avg: left={avg_imb_left:.4f}  right={avg_imb_right:.4f}  mean={avg_imb_both:.4f}")
        print(f"samples_used_for_imbalance: {len(imbals_left)}")
    else:
        print("domain_imbalance_avg: n/a (no 'maximum imbalance of X|Y' lines found)")

if __name__ == "__main__":
    main()
