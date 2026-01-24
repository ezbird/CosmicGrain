#!/usr/bin/env python3
import re, argparse, sys, math
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

SYNC_RE = re.compile(
    r'^Sync-Point\s+(\d+),\s*Time:\s*([0-9.eE+-]+),\s*Redshift:\s*([0-9.eE+-]+)',
    re.M
)

# Phase regexes (float seconds)
PHASE_PATTERNS = OrderedDict([
  ("gravtree_build_s",   re.compile(r"^GRAVTREE: Tree construction done\. took\s+([0-9.]+)\s+sec", re.M)),
  ("gravtree_force_s",   re.compile(r"^GRAVTREE: tree-forces.*took\s+([0-9.]+)\s+sec", re.M)),
  ("timestep_walk_s",    re.compile(r"^TIMESTEP-TREEWALK: took\s+([0-9.]+)\s+sec", re.M)),
  ("hydro_force_s",      re.compile(r"^SPH-HYDRO: .*hydro-force.*took\s+([0-9.]+)", re.M)),
  ("density_total_s",    re.compile(r"^SPH-DENSITY: density computation done\. took\s+([0-9.]+)", re.M)),
  ("ngbtree_build_s",    re.compile(r"^NGBTREE: Ngb-tree construction done\. took\s+([0-9.]+)\s+sec", re.M)),
  ("domain_total_s",     re.compile(r"^DOMAIN: domain decomposition done\. \(took in total\s+([0-9.]+)\s+sec\)", re.M)),
  ("domain_exchange_s",  re.compile(r"^DOMAIN: particle exchange done\. \(took\s+([0-9.]+)\s+sec\)", re.M)),
])

# Optional: capture tasks/nodes changes (best effort)
TASKS_RE = re.compile(r"tasks=(\d+),\s*nnodes=(\d+)")

def parse_files(paths):
    # Concatenate in file order, but we’ll still detect sync boundaries.
    text = ""
    for p in paths:
        text += Path(p).read_text(errors="ignore") + "\n"

    # Split by Sync-Point blocks (keep the header in each chunk)
    blocks = []
    for m in SYNC_RE.finditer(text):
        blocks.append((m.start(), m.end(), m.group(1), m.group(2), m.group(3)))
    chunks = []
    for i, (s, e, step, t, z) in enumerate(blocks):
        s0 = s
        s1 = blocks[i+1][0] if i+1 < len(blocks) else len(text)
        chunks.append((int(step), float(t), float(z), text[s0:s1]))

    rows = []
    current_tasks = None
    current_nodes = None

    for step, a_time, z, chunk in chunks:
        row = {
            "step": step,
            "a_time": a_time,
            "redshift": z,
            "tasks": current_tasks,
            "nodes": current_nodes,
        }

        # If tasks/nodes appear inside the chunk, update
        mtn = TASKS_RE.search(chunk)
        if mtn:
            current_tasks = int(mtn.group(1))
            current_nodes = int(mtn.group(2))
            row["tasks"] = current_tasks
            row["nodes"] = current_nodes

        # Find each phase time (first match per chunk)
        for key, rx in PHASE_PATTERNS.items():
            m = rx.search(chunk)
            row[key] = float(m.group(1)) if m else np.nan

        # A simple “total” (sum of known phases). Not perfect but useful.
        phase_sum = 0.0
        count = 0
        for k in ["gravtree_build_s","gravtree_force_s","timestep_walk_s",
                  "ngbtree_build_s","density_total_s","hydro_force_s",
                  "domain_exchange_s","domain_total_s"]:
            v = row.get(k, np.nan)
            if not (v is None or math.isnan(v)):
                phase_sum += v
                count += 1
        row["total_known_phases_s"] = phase_sum if count > 0 else np.nan

        rows.append(row)

    # Sort by scale factor (or step)
    rows.sort(key=lambda r: (r["a_time"], r["step"]))
    return rows

def write_csv(rows, path):
    cols = ["step","a_time","redshift","tasks","nodes"] + list(PHASE_PATTERNS.keys()) + ["total_known_phases_s"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, np.nan)
                if v is None:
                    vals.append("")
                elif isinstance(v, float) and math.isnan(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

def plot_phases(rows, outpath):
    # Plot key phases vs. redshift (single axes, multiple lines)
    z = np.array([r["redshift"] for r in rows])
    # Ensure monotonic for plotting: sort by decreasing z
    order = np.argsort(-z)
    z = z[order]

    series = {
        "Gravity tree force (s)": np.array([rows[i].get("gravtree_force_s", np.nan) for i in order]),
        "Hydro force (s)":        np.array([rows[i].get("hydro_force_s", np.nan) for i in order]),
        "Density (s)":            np.array([rows[i].get("density_total_s", np.nan) for i in order]),
        "Domain total (s)":       np.array([rows[i].get("domain_total_s", np.nan) for i in order]),
    }

    plt.figure()
    for label, y in series.items():
        plt.plot(z, y, label=label)
    plt.xlabel("Redshift")
    plt.ylabel("Time per phase (s)")
    plt.title("Gadget-4 per-phase wall times vs redshift")
    plt.legend()
    plt.gca().invert_xaxis()  # decreasing z over time
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_total(rows, outpath):
    z = np.array([r["redshift"] for r in rows])
    order = np.argsort(-z)
    z = z[order]
    y = np.array([rows[i].get("total_known_phases_s", np.nan) for i in order])

    plt.figure()
    plt.plot(z, y)
    plt.xlabel("Redshift")
    plt.ylabel("Sum of known phases (s)")
    plt.title("Gadget-4 total (known phases) vs redshift")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Parse Gadget-4 logs and plot per-phase times vs redshift.")
    ap.add_argument("logs", nargs="+", help="Log file(s) to parse (in order).")
    ap.add_argument("--csv", help="Write merged CSV here.")
    ap.add_argument("--plot-phases", help="Write phases-vs-redshift PNG here.")
    ap.add_argument("--plot-total", help="Write total-known-phases PNG here.")
    args = ap.parse_args()

    rows = parse_files(args.logs)
    if not rows:
        print("No Sync-Point blocks found.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote CSV: {args.csv}")

    if args.plot_phases:
        plot_phases(rows, args.plot_phases)
        print(f"Wrote plot: {args.plot_phases}")

    if args.plot_total:
        plot_total(rows, args.plot_total)
        print(f"Wrote plot: {args.plot_total}")

if __name__ == "__main__":
    main()
