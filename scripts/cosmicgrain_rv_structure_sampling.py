#!/usr/bin/env python3
"""
Test the robustness of a CosmicGrain R(V) structure function against
dust-superparticle sampling.

Input:
    CosmicGrain ASCII maps produced by export_halo441_maps_for_ft.py

For each minimum projected dust-particle count:
    N_dust >= 1, 3, 5, 10

compute

    S2(r) = < [R_V(x+r) - R_V(x)]^2 >

using ONLY pairs for which both pixels satisfy the sampling cut.

No interpolation or hole filling is performed.

The script also reports and plots:
    - valid pixel fraction
    - number of valid pixel pairs at each separation
    - S2(r) for each dust-count cut
    - ratio to the N_dust >= 1 result

This is intended as a sampling/convergence diagnostic before comparing
the CosmicGrain R(V) structure function to Zhang & Green.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "rv_map",
        help="ASCII R(V) map, e.g. halo441_maps_for_FT/250pc_RV_unmasked.txt",
    )

    p.add_argument(
        "count_map",
        help="ASCII projected dust-count map, e.g. "
             "halo441_maps_for_FT/250pc_dust_count.txt",
    )

    p.add_argument(
        "--pixel",
        type=float,
        default=0.25,
        help="Pixel size in physical kpc (default: 0.25)",
    )

    p.add_argument(
        "--radius",
        type=float,
        default=2.5,
        help="Circular analysis radius in kpc (default: 2.5)",
    )

    p.add_argument(
        "--sf-max",
        type=float,
        default=2.5,
        help="Maximum structure-function separation in kpc (default: 2.5)",
    )

    p.add_argument(
        "--sf-bins",
        type=int,
        default=20,
        help="Number of logarithmic separation bins (default: 20)",
    )

    p.add_argument(
        "--min-counts",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="Dust-count thresholds (default: 1 3 5 10)",
    )

    p.add_argument(
        "--field",
        choices=["absolute", "fractional"],
        default="absolute",
        help="Use R(V)-<R(V)> or R(V)/<R(V)>-1",
    )

    p.add_argument(
        "--prefix",
        default="halo441_rv_250pc_sampling",
    )

    return p.parse_args()


def load_map(filename):
    a = np.loadtxt(filename)

    if a.ndim != 2:
        raise ValueError(f"{filename} is not a 2-D matrix")

    return np.asarray(a, dtype=float)


def make_geometry(shape, dx):
    ny, nx = shape

    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dx

    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.hypot(xx, yy)

    return x, y, xx, yy, rr


def structure_function(field, valid, dx, rmax, nbins):
    """
    Direct pair-based second-order structure function.

    Each unordered pixel pair is counted exactly once.

    Returns
    -------
    rcenter
    S2
    npairs
    """

    edges = np.geomspace(dx, rmax, nbins + 1)

    sum_d2 = np.zeros(nbins, dtype=float)
    npairs = np.zeros(nbins, dtype=np.int64)

    ny, nx = field.shape

    max_shift = int(np.floor(rmax / dx))

    for dy in range(max_shift + 1):

        for sx in range(-max_shift, max_shift + 1):

            # Count each unordered pair once.
            if dy == 0 and sx <= 0:
                continue

            r = dx * math.hypot(sx, dy)

            if r < edges[0] or r >= edges[-1]:
                continue

            ib = np.searchsorted(edges, r, side="right") - 1

            if ib < 0 or ib >= nbins:
                continue

            ya = max(0, -dy)
            yb = min(ny, ny - dy)

            xa = max(0, -sx)
            xb = min(nx, nx - sx)

            if yb <= ya or xb <= xa:
                continue

            A = field[ya:yb, xa:xb]
            B = field[
                ya + dy:yb + dy,
                xa + sx:xb + sx
            ]

            va = valid[ya:yb, xa:xb]
            vb = valid[
                ya + dy:yb + dy,
                xa + sx:xb + sx
            ]

            q = (
                va
                & vb
                & np.isfinite(A)
                & np.isfinite(B)
            )

            if not np.any(q):
                continue

            d = A[q] - B[q]

            sum_d2[ib] += np.sum(d * d)
            npairs[ib] += q.sum()

    rcenter = np.sqrt(edges[:-1] * edges[1:])

    s2 = np.full(nbins, np.nan)

    good = npairs > 0
    s2[good] = sum_d2[good] / npairs[good]

    return rcenter, s2, npairs


def main():

    a = parse_args()

    rv = load_map(a.rv_map)
    count = load_map(a.count_map)

    if rv.shape != count.shape:
        raise ValueError(
            f"map shapes differ: R(V)={rv.shape}, count={count.shape}"
        )

    x, y, xx, yy, rr = make_geometry(rv.shape, a.pixel)

    footprint = rr <= a.radius

    finite = np.isfinite(rv)

    print()
    print("CosmicGrain R(V) structure-function sampling test")
    print("-------------------------------------------------")
    print(f"R(V) map          : {a.rv_map}")
    print(f"dust-count map    : {a.count_map}")
    print(f"grid              : {rv.shape[1]} x {rv.shape[0]}")
    print(f"pixel             : {a.pixel:.4f} kpc")
    print(f"analysis radius   : {a.radius:.4f} kpc")
    print(f"SF maximum r      : {a.sf_max:.4f} kpc")
    print(f"field             : {a.field}")
    print(f"pixels in circle  : {footprint.sum():,}")
    print()

    results = {}

    for nmin in a.min_counts:

        valid = (
            footprint
            & finite
            & np.isfinite(count)
            & (count >= nmin)
        )

        nvalid = valid.sum()
        frac = nvalid / footprint.sum()

        if nvalid == 0:
            print(f"N_dust >= {nmin:2d}: NO VALID PIXELS")
            continue

        mean_rv = np.mean(rv[valid])
        median_rv = np.median(rv[valid])
        std_rv = np.std(rv[valid])

        field = np.full_like(rv, np.nan)

        if a.field == "absolute":
            field[valid] = rv[valid] - mean_rv

        else:
            field[valid] = rv[valid] / mean_rv - 1.0

        rs, s2, npairs = structure_function(
            field,
            valid,
            a.pixel,
            a.sf_max,
            a.sf_bins,
        )

        results[nmin] = {
            "valid": valid,
            "field": field,
            "r": rs,
            "s2": s2,
            "npairs": npairs,
            "mean": mean_rv,
            "median": median_rv,
            "std": std_rv,
            "fraction": frac,
        }

        print(
            f"N_dust >= {nmin:2d}: "
            f"{nvalid:4d}/{footprint.sum():4d} pixels "
            f"({frac:.4f}), "
            f"<RV>={mean_rv:.4f}, "
            f"median={median_rv:.4f}, "
            f"sigma={std_rv:.4f}"
        )

    if not results:
        raise RuntimeError("No valid sampling thresholds.")

    prefix = Path(a.prefix)

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    for nmin, r in results.items():

        outfile = f"{prefix}_N{nmin}_structure.txt"

        np.savetxt(
            outfile,
            np.c_[r["r"], r["s2"], r["npairs"]],
            header=(
                "r_kpc S2_RV n_pairs\n"
                f"minimum_dust_count = {nmin}\n"
                f"valid_fraction = {r['fraction']:.8f}\n"
                f"mean_RV = {r['mean']:.8f}\n"
                f"median_RV = {r['median']:.8f}\n"
                f"std_RV = {r['std']:.8f}\n"
                f"field = {a.field}"
            ),
        )

    # --------------------------------------------------------
    # Figure 1: structure functions
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    for nmin, r in results.items():

        q = (
            np.isfinite(r["s2"])
            & (r["s2"] > 0)
            & (r["npairs"] > 0)
        )

        ax.loglog(
            r["r"][q],
            r["s2"][q],
            marker="o",
            ms=5,
            label=rf"$N_{{\rm dust}}\geq {nmin}$",
        )

    ax.set_xlabel(r"$r\ ({\rm kpc})$")
    ax.set_ylabel(r"$S_2^{R(V)}(r)$")

    ax.set_title(
        "CosmicGrain CG10: R(V) sampling convergence"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_structure_convergence.png",
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: pair counts
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    for nmin, r in results.items():

        q = r["npairs"] > 0

        ax.loglog(
            r["r"][q],
            r["npairs"][q],
            marker="o",
            ms=5,
            label=rf"$N_{{\rm dust}}\geq {nmin}$",
        )

    ax.set_xlabel(r"$r\ ({\rm kpc})$")
    ax.set_ylabel("Number of valid pixel pairs")

    ax.set_title(
        "CosmicGrain CG10: structure-function sampling"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_pair_counts.png",
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 3: ratio to N>=1
    # --------------------------------------------------------

    if 1 in results:

        base = results[1]

        fig, ax = plt.subplots(figsize=(8, 5.5))

        for nmin, r in results.items():

            if nmin == 1:
                continue

            if not np.allclose(r["r"], base["r"]):
                continue

            ratio = r["s2"] / base["s2"]

            q = (
                np.isfinite(ratio)
                & (ratio > 0)
                & (r["npairs"] > 0)
                & (base["npairs"] > 0)
            )

            ax.semilogx(
                r["r"][q],
                ratio[q],
                marker="o",
                ms=5,
                label=rf"$N_{{\rm dust}}\geq {nmin}$",
            )

        ax.axhline(1.0, ls="--", lw=1)

        ax.set_xlabel(r"$r\ ({\rm kpc})$")
        ax.set_ylabel(
            r"$S_2(r)\,/\,S_{2,N_{\rm dust}\geq1}(r)$"
        )

        ax.set_title(
            "Sensitivity of R(V) structure to dust sampling"
        )

        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            f"{prefix}_structure_ratio.png",
            dpi=200,
        )
        plt.close(fig)

    # --------------------------------------------------------
    # Figure 4: sampling masks
    # --------------------------------------------------------

    nplot = len(results)

    fig, axes = plt.subplots(
        1,
        nplot,
        figsize=(4.2 * nplot, 4.5),
        squeeze=False,
    )

    extent = [
        x[0] - 0.5 * a.pixel,
        x[-1] + 0.5 * a.pixel,
        y[0] - 0.5 * a.pixel,
        y[-1] + 0.5 * a.pixel,
    ]

    for ax, (nmin, r) in zip(axes[0], results.items()):

        m = np.where(footprint, r["valid"].astype(float), np.nan)

        ax.imshow(
            m,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_title(
            rf"$N_{{\rm dust}}\geq {nmin}$"
            "\n"
            rf"$f_{{\rm valid}}={r['fraction']:.2f}$"
        )

        ax.set_xlabel("x (kpc)")
        ax.set_ylabel("y (kpc)")
        ax.set_aspect("equal")

    fig.suptitle(
        "CG10 R(V): valid-pixel sampling masks"
    )

    fig.tight_layout()

    fig.savefig(
        f"{prefix}_sampling_masks.png",
        dpi=200,
    )

    plt.close(fig)

    print()
    print("Outputs:")
    print(f"  {prefix}_structure_convergence.png")
    print(f"  {prefix}_pair_counts.png")
    print(f"  {prefix}_structure_ratio.png")
    print(f"  {prefix}_sampling_masks.png")
    print(f"  {prefix}_N*_structure.txt")
    print()


if __name__ == "__main__":
    main()
