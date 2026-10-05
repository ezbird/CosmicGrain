#!/usr/bin/env python3
"""
Plot the observed SAGAbg-I direct-Te mass-metallicity relation together with
CosmicGrain z=0 galaxies.

Expected files (defaults assume this script is run from gadget4/):
    scripts/apjad3042t1_mrt.txt
    cosmicgrain_sagabg_z0.csv

Example:
    python3 scripts/plot_cosmicgrain_sagabg_mzr.py

Outputs:
    cosmicgrain_sagabg_mzr.png
    cosmicgrain_sagabg_mzr.pdf

By default only CG01--CG06 are shown, since these are the dwarf comparison
sample. Use --all-cg to plot every available CosmicGrain halo.

SAGAbg source:
Kado-Fong et al. 2024, ApJ 966, 129, Table 1
"Galaxy Properties and Emission Line Measurements of 120 SAGAbg-A
 Galaxies with Secure Auroral Line Metallicities"

The SAGAbg table is fixed-width. This script reads only:
    OBJID        bytes   1-19
    z            bytes  41-46
    logM*        bytes  48-51
    P16logOH12   bytes 422-425
    P50logOH12   bytes 427-430
    P84logOH12   bytes 432-435

CosmicGrain y-values use OH12_SFRweighted, i.e. the SFR-weighted O/H
computed in linear O/H space by cosmicgrain_sagabg_comparison.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DWARF_LABELS = {f"CG{i:02d}" for i in range(1, 7)}


def parse_float(text):
    text = text.strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_sagabg_table(filename):
    """Read the needed columns directly from the published fixed-width MRT."""
    rows = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Actual catalog rows begin with the 19-character integer OBJID.
            if len(line) < 435:
                continue

            obj_text = line[0:19].strip()
            if not obj_text.isdigit():
                continue

            row = {
                "OBJID": obj_text,
                "z": parse_float(line[40:46]),
                "logMstar": parse_float(line[47:51]),
                "P16logOH12": parse_float(line[421:425]),
                "P50logOH12": parse_float(line[426:430]),
                "P84logOH12": parse_float(line[431:435]),
            }

            if np.isfinite(row["logMstar"]) and np.isfinite(row["P50logOH12"]):
                rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No SAGAbg catalog rows were parsed from {filename}. "
            "Check that this is apjad3042t1_mrt.txt."
        )

    return rows


def read_cosmicgrain_csv(filename):
    rows = []
    with open(filename, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            def num(key):
                try:
                    return float(row[key])
                except (KeyError, TypeError, ValueError):
                    return np.nan

            rows.append({
                "CG": row["CG"].strip(),
                "logMstar": num("log10_Mstar_Msun"),
                "OH12_SFRweighted": num("OH12_SFRweighted"),
                "OH12_allgas": num("OH12_allgas_massweighted"),
                "N_SF": int(float(row.get("N_SFgas_30pkpc", 0) or 0)),
                "SFR": num("SFR_Msun_yr"),
            })

    if not rows:
        raise RuntimeError(f"No CosmicGrain rows found in {filename}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sagabg",
        default="scripts/apjad3042t1_mrt.txt",
        help="published SAGAbg-I Table 1 machine-readable file",
    )
    parser.add_argument(
        "--cosmicgrain",
        default="cosmicgrain_sagabg_z0.csv",
        help="output from cosmicgrain_sagabg_comparison.py",
    )
    parser.add_argument(
        "--prefix",
        default="cosmicgrain_sagabg_mzr",
        help="output filename prefix",
    )
    parser.add_argument(
        "--z-local",
        type=float,
        default=0.05,
        help="redshift threshold highlighted as the low-z SAGAbg subset",
    )
    parser.add_argument(
        "--low-nsf",
        type=int,
        default=20,
        help="mark CG measurements with fewer than this many SF gas particles as low-N",
    )
    parser.add_argument(
        "--all-cg",
        action="store_true",
        help="plot all available CG halos rather than CG01--CG06 only",
    )
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    saga = read_sagabg_table(args.sagabg)
    cg = read_cosmicgrain_csv(args.cosmicgrain)

    if len(saga) != 120:
        print(f"WARNING: parsed {len(saga)} SAGAbg rows; published table contains 120.")
    else:
        print("SAGAbg catalog check: PASS -- parsed all 120 galaxies")

    z = np.array([r["z"] for r in saga])
    sm = np.array([r["logMstar"] for r in saga])
    oh16 = np.array([r["P16logOH12"] for r in saga])
    oh50 = np.array([r["P50logOH12"] for r in saga])
    oh84 = np.array([r["P84logOH12"] for r in saga])

    local = np.isfinite(z) & (z < args.z_local)
    print(f"SAGAbg z < {args.z_local:g}: {np.sum(local)} / {len(saga)}")

    if not args.all_cg:
        cg = [r for r in cg if r["CG"] in DWARF_LABELS]

    finite_cg = [
        r for r in cg
        if np.isfinite(r["logMstar"]) and np.isfinite(r["OH12_SFRweighted"])
    ]
    no_nebular = [
        r for r in cg
        if np.isfinite(r["logMstar"]) and not np.isfinite(r["OH12_SFRweighted"])
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.6))

    # Full SAGAbg sample: neutral background population.
    ax.scatter(
        sm, oh50,
        s=28,
        c="0.72",
        alpha=0.45,
        linewidths=0,
        label=rf"SAGAbg direct-$T_e$ (all {len(saga)})",
        zorder=1,
    )

    # Low-redshift subset: make these unmistakable rather than using
    # nearly invisible open circles.
    ax.scatter(
        sm[local], oh50[local],
        s=42,
        color="tab:blue",
        alpha=0.90,
        linewidths=0,
        label=rf"SAGAbg $z<{args.z_local:g}$ ({np.sum(local)})",
        zorder=3,
    )

    # Posterior metallicity uncertainties for the highlighted low-z subset.
    yerr_low = oh50[local] - oh16[local]
    yerr_high = oh84[local] - oh50[local]
    good_err = (
        np.isfinite(yerr_low)
        & np.isfinite(yerr_high)
        & (yerr_low >= 0)
        & (yerr_high >= 0)
    )
    if np.any(good_err):
        ax.errorbar(
            sm[local][good_err],
            oh50[local][good_err],
            yerr=np.vstack((yerr_low[good_err], yerr_high[good_err])),
            fmt="none",
            ecolor="tab:blue",
            alpha=0.42,
            linewidth=0.8,
            capsize=0,
            zorder=2,
        )

    # CosmicGrain: every simulated galaxy is a circle and uses one color.
    # Filled circles have >= low_nsf SF gas particles; open circles flag
    # low-N_SF measurements without implying a different physical quantity.
    first_normal = True
    first_low = True

    for r in finite_cg:
        low_n = r["N_SF"] < args.low_nsf

        if low_n:
            ax.scatter(
                r["logMstar"], r["OH12_SFRweighted"],
                s=145,
                marker="o",
                facecolors="white",
                edgecolors="tab:red",
                linewidths=2.4,
                label=(
                    rf"CosmicGrain ($N_{{\rm SF}}<{args.low_nsf}$)"
                    if first_low else None
                ),
                zorder=6,
            )
            first_low = False
        else:
            ax.scatter(
                r["logMstar"], r["OH12_SFRweighted"],
                s=145,
                marker="o",
                facecolors="tab:red",
                edgecolors="black",
                linewidths=1.0,
                label=(
                    rf"CosmicGrain ($N_{{\rm SF}}\geq {args.low_nsf}$)"
                    if first_normal else None
                ),
                zorder=6,
            )
            first_normal = False

        ax.annotate(
            r["CG"],
            (r["logMstar"], r["OH12_SFRweighted"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            zorder=7,
        )

    # A zero-SFR system has no nebular-like SFR-weighted abundance.
    # Do not substitute the all-gas O/H onto the observed nebular relation.
    for r in no_nebular:
        print(
            f"{r['CG']}: not plotted -- no finite SFR-weighted O/H "
            f"(SFR={r['SFR']:.3e}, N_SF={r['N_SF']}); "
            f"all-gas O/H={r['OH12_allgas']:.3f}"
        )

    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$", fontsize=13)
    ax.set_ylabel(r"$12+\log_{10}(\mathrm{O/H})$", fontsize=13)
    ax.set_title("CosmicGrain dwarfs vs. SAGAbg direct-$T_e$ metallicities", fontsize=14)

    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.18)
    ax.legend(frameon=True, framealpha=0.95, fontsize=10, loc="best", scatterpoints=1)

    # Let the data set the useful range, with modest padding.
    xvals = np.concatenate((sm[np.isfinite(sm)],
                            np.array([r["logMstar"] for r in finite_cg])))
    yvals = np.concatenate((oh50[np.isfinite(oh50)],
                            np.array([r["OH12_SFRweighted"] for r in finite_cg])))

    if len(xvals):
        ax.set_xlim(np.min(xvals) - 0.18, np.max(xvals) + 0.18)
    if len(yvals):
        ax.set_ylim(np.min(yvals) - 0.12, np.max(yvals) + 0.12)

    fig.tight_layout()

    png = args.prefix + ".png"
    pdf = args.prefix + ".pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")

    print("\nCosmicGrain points plotted:")
    for r in finite_cg:
        flag = f" LOW-N_SF({r['N_SF']})" if r["N_SF"] < args.low_nsf else ""
        print(
            f"  {r['CG']}: logM*={r['logMstar']:.3f}, "
            f"12+log(O/H)_SFR={r['OH12_SFRweighted']:.3f}, "
            f"N_SF={r['N_SF']}{flag}"
        )


if __name__ == "__main__":
    main()
