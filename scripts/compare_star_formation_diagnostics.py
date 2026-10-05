#!/usr/bin/env python3
"""Compare two CosmicGrain star-formation diagnostic CSV products."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", help="Reference run directory or diagnostic CSV")
    p.add_argument("test", help="Test run directory or diagnostic CSV")
    p.add_argument("--reference-label", default="reference")
    p.add_argument("--test-label", default="test")
    p.add_argument("--output-dir", default="comparison")
    p.add_argument("--prefix", default="sf_run_comparison")
    return p.parse_args()


def locate_csv(value):
    path = Path(value)
    if path.is_file():
        return path
    candidates = sorted(path.glob("*_sf_diagnostic.csv"))
    if not candidates:
        candidates = sorted((path / "star_formation_diagnostic").glob("*_sf_diagnostic.csv"))
    if len(candidates) != 1:
        raise SystemExit(f"Expected exactly one diagnostic CSV under {path}; found {len(candidates)}")
    return candidates[0]


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"birthmass_field", "gas_missing", "star_missing"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return sorted(rows, key=lambda r: r["z"], reverse=True)


def matched_rows(reference, test, tolerance=0.03):
    pairs = []
    used = set()
    for ref in reference:
        choices = [(abs(ref["z"] - candidate["z"]), i, candidate)
                   for i, candidate in enumerate(test) if i not in used]
        if not choices:
            continue
        delta, index, candidate = min(choices)
        if delta <= tolerance:
            used.add(index)
            pairs.append((ref, candidate))
    return pairs


METRICS = [
    ("m200_msun", "M200c", "Msun"),
    ("star_30pkpc_msun", "Mstar(<30 pkpc)", "Msun"),
    ("gas_30pkpc_msun", "Mgas(<30 pkpc)", "Msun"),
    ("sfr_30pkpc_msunyr", "SFR(<30 pkpc)", "Msun/yr"),
    ("max_nH_cm3", "maximum nH", "cm^-3"),
    ("gas30_nh_ge_sf_msun", "SF-eligible gas", "Msun"),
    ("gas30_nh_ge_1_msun", "gas at nH>=1", "Msun"),
    ("gas30_nh_ge_10_msun", "gas at nH>=10", "Msun"),
    ("gas30_nh_ge_100_msun", "gas at nH>=100", "Msun"),
    ("gas30_sf_effective_eos_msun", "SF effective-EOS gas", "Msun"),
    ("gas30_nonSF_T_ge_1e6_msun", "hot non-SF gas", "Msun"),
    ("median_hsml30_pkpc", "median gas Hsml", "pkpc"),
    ("min_hsml_fraction30", "fraction at Hsml floor", "fraction"),
]


def number(row, key):
    value = row.get(key, np.nan)
    return float(value) if isinstance(value, (int, float)) else np.nan


def write_comparison(path, pairs, ref_label, test_label):
    columns = ["snapshot_reference", "snapshot_test", "z_reference", "z_test"]
    for key, _, _ in METRICS:
        columns += [f"{key}_{ref_label}", f"{key}_{test_label}",
                    f"{key}_ratio_test_to_reference", f"{key}_percent_change"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for ref, test in pairs:
            out = {"snapshot_reference": int(ref["snapshot"]),
                   "snapshot_test": int(test["snapshot"]),
                   "z_reference": ref["z"], "z_test": test["z"]}
            for key, _, _ in METRICS:
                a, b = number(ref, key), number(test, key)
                ratio = b / a if np.isfinite(a) and a != 0 and np.isfinite(b) else np.nan
                out[f"{key}_{ref_label}"] = a
                out[f"{key}_{test_label}"] = b
                out[f"{key}_ratio_test_to_reference"] = ratio
                out[f"{key}_percent_change"] = 100 * (ratio - 1) if np.isfinite(ratio) else np.nan
            writer.writerow(out)


def make_plot(path, pairs, ref_label, test_label):
    panels = [
        ("star_30pkpc_msun", "Stellar mass", True),
        ("gas_30pkpc_msun", "Central gas mass", True),
        ("sfr_30pkpc_msunyr", "SFR", True),
        ("max_nH_cm3", "Maximum nH", True),
        ("gas30_nh_ge_sf_msun", "SF-eligible gas mass", True),
        ("gas30_nonSF_T_ge_1e6_msun", "Hot non-SF gas mass", True),
    ]
    z_ref = np.array([a["z"] for a, _ in pairs])
    z_test = np.array([b["z"] for _, b in pairs])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title, log) in zip(axes.flat, panels):
        a = np.array([number(ref, key) for ref, _ in pairs])
        b = np.array([number(test, key) for _, test in pairs])
        ax.plot(z_ref, a, "o-", label=ref_label)
        ax.plot(z_test, b, "s-", label=test_label)
        if log and np.any(a > 0) and np.any(b > 0):
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Redshift")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("CosmicGrain matched-run star-formation comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments()
    reference_path, test_path = locate_csv(args.reference), locate_csv(args.test)
    reference, test = read_rows(reference_path), read_rows(test_path)
    pairs = matched_rows(reference, test)
    if not pairs:
        raise SystemExit("No matched redshift pairs found")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{args.prefix}.csv"
    png_path = outdir / f"{args.prefix}.png"
    write_comparison(csv_path, pairs, args.reference_label, args.test_label)
    make_plot(png_path, pairs, args.reference_label, args.test_label)
    print(f"Reference: {reference_path}")
    print(f"Test:      {test_path}")
    print(f"Matched epochs: {len(pairs)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
