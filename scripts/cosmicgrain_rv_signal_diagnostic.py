#!/usr/bin/env python3
"""
Diagnose whether extreme CosmicGrain R(V) pixels are associated with
poor color-excess signal and/or sparse dust-superparticle sampling.

Uses ASCII maps from export_halo441_maps_for_ft.py.

R(V) = tau_V / (tau_B - tau_V)

Diagnostics inside a circular aperture:
  1. R(V) histogram
  2. R(V) vs projected dust-particle count
  3. R(V) vs color-excess proxy tau_B - tau_V
  4. R(V) distributions after relative color-excess cuts

No interpolation or hole filling is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("tau_b")
    p.add_argument("tau_v")
    p.add_argument("rv")
    p.add_argument("dust_count")

    p.add_argument(
        "--pixel",
        type=float,
        default=0.25,
        help="Pixel size in physical kpc",
    )

    p.add_argument(
        "--radius",
        type=float,
        default=2.5,
        help="Circular analysis radius in physical kpc",
    )

    p.add_argument(
        "--cuts",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 1e-3, 1e-2],
        help="Minimum (tau_B-tau_V)/max(tau_B-tau_V)",
    )

    p.add_argument(
        "--prefix",
        default="halo441_rv_250pc_signal",
    )

    return p.parse_args()


def load(fn):
    a = np.loadtxt(fn)

    if a.ndim != 2:
        raise ValueError(f"{fn} is not a 2-D matrix")

    return np.asarray(a, dtype=float)


def main():

    a = parse_args()

    tb = load(a.tau_b)
    tv = load(a.tau_v)
    rv = load(a.rv)
    nd = load(a.dust_count)

    if not (tb.shape == tv.shape == rv.shape == nd.shape):
        raise ValueError(
            f"Shapes differ: "
            f"tauB={tb.shape}, tauV={tv.shape}, "
            f"RV={rv.shape}, count={nd.shape}"
        )

    ny, nx = rv.shape

    x = (np.arange(nx) - 0.5 * (nx - 1)) * a.pixel
    y = (np.arange(ny) - 0.5 * (ny - 1)) * a.pixel

    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.hypot(xx, yy)

    footprint = rr <= a.radius

    # Color-excess / differential-extinction proxy.
    ebv = tb - tv

    base = (
        footprint
        & np.isfinite(tb)
        & np.isfinite(tv)
        & np.isfinite(rv)
        & np.isfinite(nd)
        & (tv > 0)
        & (ebv > 0)
        & (nd >= 1)
    )

    if not np.any(base):
        raise RuntimeError("No valid pixels in the selected aperture.")

    ebv_max = np.max(ebv[base])
    rel = np.full_like(ebv, np.nan)

    rel[base] = ebv[base] / ebv_max

    print()
    print("CosmicGrain R(V) signal diagnostic")
    print("----------------------------------")
    print(f"grid                  = {nx} x {ny}")
    print(f"pixel                 = {a.pixel:.4f} kpc")
    print(f"analysis radius       = {a.radius:.4f} kpc")
    print(f"pixels in footprint   = {footprint.sum():,}")
    print(f"base valid pixels     = {base.sum():,}")
    print(f"base valid fraction   = {base.sum()/footprint.sum():.6f}")
    print()
    print(f"max(tau_B-tau_V)      = {ebv_max:.8e}")

    vals = rv[base]

    print()
    print("Uncut R(V):")
    print(f"  mean                = {np.mean(vals):.6f}")
    print(f"  median              = {np.median(vals):.6f}")
    print(f"  std                 = {np.std(vals):.6f}")
    print(f"  p16                 = {np.percentile(vals,16):.6f}")
    print(f"  p84                 = {np.percentile(vals,84):.6f}")
    print(f"  p95                 = {np.percentile(vals,95):.6f}")
    print(f"  p99                 = {np.percentile(vals,99):.6f}")
    print(f"  max                 = {np.max(vals):.6f}")

    results = []

    print()
    print("Relative color-excess cuts")
    print("--------------------------")

    for cut in a.cuts:

        q = base & (rel >= cut)

        n = q.sum()

        if n == 0:
            print(f"{cut:.1e}: 0 pixels")
            continue

        v = rv[q]

        result = dict(
            cut=cut,
            n=n,
            fraction=n / footprint.sum(),
            retained=n / base.sum(),
            mean=np.mean(v),
            median=np.median(v),
            std=np.std(v),
            p16=np.percentile(v, 16),
            p84=np.percentile(v, 84),
            p95=np.percentile(v, 95),
            maximum=np.max(v),
        )

        results.append(result)

        print(
            f"{cut:8.1e}: "
            f"N={n:4d}  "
            f"footprint={result['fraction']:.4f}  "
            f"retained={result['retained']:.4f}  "
            f"mean={result['mean']:.4f}  "
            f"median={result['median']:.4f}  "
            f"std={result['std']:.4f}  "
            f"p95={result['p95']:.4f}  "
            f"max={result['maximum']:.4f}"
        )

    # -------------------------------------------------------
    # Save per-pixel diagnostic table
    # -------------------------------------------------------

    iy, ix = np.where(base)

    table = np.column_stack([
        x[ix],
        y[iy],
        rr[iy, ix],
        nd[iy, ix],
        tb[iy, ix],
        tv[iy, ix],
        ebv[iy, ix],
        rel[iy, ix],
        rv[iy, ix],
    ])

    np.savetxt(
        f"{a.prefix}_pixels.txt",
        table,
        header=(
            "x_kpc y_kpc R_kpc N_dust "
            "tau_B tau_V tauB_minus_tauV "
            "relative_color_excess RV"
        ),
    )

    # -------------------------------------------------------
    # Figure
    # -------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    ax = axes[0, 0]

    ax.hist(
        rv[base],
        bins=30,
        histtype="step",
        linewidth=2,
    )

    ax.axvline(
        np.median(rv[base]),
        linestyle="--",
        linewidth=1,
        label=f"median = {np.median(rv[base]):.2f}",
    )

    ax.set_xlabel(r"$R(V)$")
    ax.set_ylabel("Number of pixels")
    ax.set_title("All valid 250 pc pixels")
    ax.legend()

    # -------------------------------------------------------

    ax = axes[0, 1]

    ax.scatter(
        nd[base],
        rv[base],
        s=25,
        alpha=0.7,
    )

    ax.set_xlabel(r"$N_{\rm dust}$ per 250 pc pixel")
    ax.set_ylabel(r"$R(V)$")
    ax.set_title(r"$R(V)$ vs. dust sampling")

    # -------------------------------------------------------

    ax = axes[1, 0]

    q = base & (rel > 0)

    ax.scatter(
        rel[q],
        rv[q],
        s=25,
        alpha=0.7,
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"$(\tau_B-\tau_V)/"
        r"\max(\tau_B-\tau_V)$"
    )

    ax.set_ylabel(r"$R(V)$")
    ax.set_title(r"$R(V)$ vs. color-excess signal")

    # -------------------------------------------------------

    ax = axes[1, 1]

    for result in results:

        cut = result["cut"]

        q = base & (rel >= cut)

        ax.hist(
            rv[q],
            bins=25,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=(
                rf"$E_{{BV}}/E_{{BV,\max}}\geq"
                f"{cut:.0e}$ "
                f"(N={q.sum()})"
            ),
        )

    ax.set_xlabel(r"$R(V)$")
    ax.set_ylabel("Normalized distribution")
    ax.set_title("Effect of color-excess cut")
    ax.legend(fontsize=9)

    fig.suptitle(
        "CosmicGrain CG10: 250 pc R(V) signal diagnostic",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        f"{a.prefix}_diagnostic.png",
        dpi=200,
    )

    plt.close(fig)

    # -------------------------------------------------------
    # Summary table
    # -------------------------------------------------------

    if results:

        summary = np.array([
            [
                r["cut"],
                r["n"],
                r["fraction"],
                r["retained"],
                r["mean"],
                r["median"],
                r["std"],
                r["p16"],
                r["p84"],
                r["p95"],
                r["maximum"],
            ]
            for r in results
        ])

        np.savetxt(
            f"{a.prefix}_cut_summary.txt",
            summary,
            header=(
                "relative_cut N_pixels footprint_fraction "
                "retained_fraction mean_RV median_RV std_RV "
                "p16_RV p84_RV p95_RV max_RV"
            ),
        )

    print()
    print("Outputs:")
    print(f"  {a.prefix}_diagnostic.png")
    print(f"  {a.prefix}_pixels.txt")
    print(f"  {a.prefix}_cut_summary.txt")
    print()


if __name__ == "__main__":
    main()
