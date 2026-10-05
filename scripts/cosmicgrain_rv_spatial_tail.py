#!/usr/bin/env python3
"""
Spatial diagnostic for the high-R(V) tail in CosmicGrain.

Uses 250-pc maps exported by export_halo441_maps_for_ft.py.

Panels:
  1. R(V) map for all mathematically valid pixels
  2. Same map with R(V)>6, >10, >15 highlighted
  3. Projected dust-superparticle count
  4. R(V) versus dust count, colored by galactocentric radius

No interpolation or hole filling.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("rv_map")
    p.add_argument("count_map")

    p.add_argument("--pixel", type=float, default=0.25)
    p.add_argument("--radius", type=float, default=2.5)

    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[6.0, 10.0, 15.0],
    )

    p.add_argument(
        "--prefix",
        default="halo441_rv_250pc_spatial_tail",
    )

    return p.parse_args()


def load(fn):
    a = np.loadtxt(fn)

    if a.ndim != 2:
        raise ValueError(f"{fn} is not a 2-D matrix")

    return np.asarray(a, dtype=float)


def main():

    a = parse_args()

    rv = load(a.rv_map)
    nd = load(a.count_map)

    if rv.shape != nd.shape:
        raise ValueError(
            f"Shape mismatch: RV={rv.shape}, dust_count={nd.shape}"
        )

    ny, nx = rv.shape

    x = (np.arange(nx) - 0.5 * (nx - 1)) * a.pixel
    y = (np.arange(ny) - 0.5 * (ny - 1)) * a.pixel

    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.hypot(xx, yy)

    footprint = rr <= a.radius

    valid = (
        footprint
        & np.isfinite(rv)
        & np.isfinite(nd)
        & (nd >= 1)
    )

    nfoot = footprint.sum()
    nvalid = valid.sum()

    print()
    print("CosmicGrain high-R(V) spatial diagnostic")
    print("----------------------------------------")
    print(f"grid               = {nx} x {ny}")
    print(f"pixel              = {a.pixel:.4f} kpc")
    print(f"analysis radius    = {a.radius:.4f} kpc")
    print(f"footprint pixels   = {nfoot}")
    print(f"valid pixels       = {nvalid}")
    print(f"valid fraction     = {nvalid/nfoot:.6f}")

    print()
    print("R(V) thresholds")
    print("---------------")

    for t in a.thresholds:

        q = valid & (rv > t)

        if q.sum() == 0:
            print(f"R(V) > {t:5.1f}: 0 pixels")
            continue

        print(
            f"R(V) > {t:5.1f}: "
            f"N={q.sum():3d}  "
            f"fraction_valid={q.sum()/nvalid:.4f}  "
            f"median_Ndust={np.median(nd[q]):.2f}  "
            f"mean_Ndust={np.mean(nd[q]):.2f}  "
            f"median_R={np.median(rr[q]):.3f} kpc"
        )

    # ---------------------------------------------------------
    # Count distribution by R(V) regime
    # ---------------------------------------------------------

    regimes = [
        ("R(V) <= 6", valid & (rv <= 6)),
        ("6 < R(V) <= 10", valid & (rv > 6) & (rv <= 10)),
        ("10 < R(V) <= 15", valid & (rv > 10) & (rv <= 15)),
        ("R(V) > 15", valid & (rv > 15)),
    ]

    print()
    print("Dust sampling by R(V) regime")
    print("----------------------------")

    for label, q in regimes:

        if q.sum() == 0:
            print(f"{label:18s}: N=0")
            continue

        print(
            f"{label:18s}: "
            f"N={q.sum():3d}  "
            f"Ndust median={np.median(nd[q]):6.2f}  "
            f"mean={np.mean(nd[q]):6.2f}  "
            f"range={np.min(nd[q]):.0f}-{np.max(nd[q]):.0f}"
        )

    # ---------------------------------------------------------
    # Save valid-pixel catalogue
    # ---------------------------------------------------------

    iy, ix = np.where(valid)

    table = np.column_stack([
        x[ix],
        y[iy],
        rr[iy, ix],
        rv[iy, ix],
        nd[iy, ix],
    ])

    np.savetxt(
        f"{a.prefix}_pixels.txt",
        table,
        header="x_kpc y_kpc R_kpc RV N_dust",
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    extent = [
        x[0] - 0.5 * a.pixel,
        x[-1] + 0.5 * a.pixel,
        y[0] - 0.5 * a.pixel,
        y[-1] + 0.5 * a.pixel,
    ]

    # Display only the central requested region.
    lim = a.radius + 0.5 * a.pixel

    rv_display = np.full_like(rv, np.nan)
    rv_display[valid] = rv[valid]

    # Use the actual data range, but cap the displayed scale at
    # the 95th percentile so that the ordinary R(V) structure
    # remains visible. Extreme values are identified separately.
    vmax = np.percentile(rv[valid], 95)
    vmin = np.percentile(rv[valid], 5)

    # ---------------------------------------------------------
    # Panel 1: R(V)
    # ---------------------------------------------------------

    ax = axes[0, 0]

    im = ax.imshow(
        rv_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$R(V)$")

    ax.set_title(
        rf"$R(V)$, valid pixels "
        rf"({nvalid}/{nfoot})"
    )

    # ---------------------------------------------------------
    # Panel 2: high-R(V) locations
    # ---------------------------------------------------------

    ax = axes[0, 1]

    im = ax.imshow(
        rv_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    # Increasing marker size denotes increasingly extreme R(V).
    markers = ["o", "s", "*"]
    sizes = [55, 75, 130]

    for i, t in enumerate(a.thresholds):

        q = valid & (rv > t)

        if not np.any(q):
            continue

        iyq, ixq = np.where(q)

        ax.scatter(
            x[ixq],
            y[iyq],
            s=sizes[min(i, len(sizes)-1)],
            marker=markers[min(i, len(markers)-1)],
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label=rf"$R(V)>{t:g}$ (N={q.sum()})",
        )

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_title("Locations of high-$R(V)$ pixels")

    # ---------------------------------------------------------
    # Panel 3: projected dust counts
    # ---------------------------------------------------------

    ax = axes[1, 0]

    count_display = np.full_like(nd, np.nan)
    count_display[valid] = nd[valid]

    # log10(Ndust) makes low-count structure much easier to see.
    log_count = np.full_like(nd, np.nan)

    q = valid & (nd > 0)
    log_count[q] = np.log10(nd[q])

    im = ax.imshow(
        log_count,
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10} N_{\rm dust}$")

    ax.set_title("Projected dust-superparticle sampling")

    # ---------------------------------------------------------
    # Panel 4: RV versus dust count
    # ---------------------------------------------------------

    ax = axes[1, 1]

    sc = ax.scatter(
        nd[valid],
        rv[valid],
        c=rr[valid],
        s=32,
        alpha=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$R_{\rm gal}\ ({\rm kpc})$")

    ax.set_xscale("log")

    ax.set_xlabel(
        r"$N_{\rm dust}$ per 250 pc pixel"
    )
    ax.set_ylabel(r"$R(V)$")

    for t in a.thresholds:
        ax.axhline(
            t,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_title(r"$R(V)$ versus particle sampling")

    # ---------------------------------------------------------

    for ax in axes.flat:

        if ax is not axes[1, 1]:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")

            ax.set_xlabel(r"$x\ ({\rm kpc})$")
            ax.set_ylabel(r"$y\ ({\rm kpc})$")

    fig.suptitle(
        "CosmicGrain CG10: 250 pc R(V) spatial diagnostic",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        f"{a.prefix}.png",
        dpi=200,
    )

    plt.close(fig)

    print()
    print("Outputs:")
    print(f"  {a.prefix}.png")
    print(f"  {a.prefix}_pixels.txt")
    print()


if __name__ == "__main__":
    main()
