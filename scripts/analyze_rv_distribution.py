#!/usr/bin/env python3
"""
Analyze the distribution of valid R_V pixels in a CosmicGrain R_V map NPZ.

Reports percentiles and Milky-Way-like fractions, and makes:
  * R_V histogram with the MW 2.3--3.9 interval highlighted
  * R_V map with MW-like pixels highlighted
  * R_V versus projected galactocentric radius

The script uses the validity mask stored in the NPZ, so run it on the map
generated with the desired --min-count / --signal-frac cuts.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_first(npz, names, default=None):
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="R_V map NPZ from halo569_rv_sfr_triptych.py")
    p.add_argument("--mw-min", type=float, default=2.3)
    p.add_argument("--mw-max", type=float, default=3.9)
    p.add_argument("--prefix", default=None)
    args = p.parse_args()

    infile = Path(args.npz)
    prefix = args.prefix or str(infile.with_suffix("")) + "_distribution"

    d = np.load(infile, allow_pickle=True)
    print("NPZ keys:", ", ".join(d.files))

    rv = get_first(d, ["rv", "Rv", "R_V"])
    if rv is None:
        raise KeyError("Could not find an R_V array (expected key 'rv').")
    rv = np.asarray(rv, dtype=float)

    mask = get_first(d, ["rv_mask", "mask", "valid_mask"])
    if mask is None:
        mask = np.isfinite(rv)
        print("WARNING: no stored validity mask found; using finite R_V pixels.")
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(rv)

    vals = rv[mask]
    if vals.size == 0:
        raise RuntimeError("No valid R_V pixels.")

    # Coordinates: prefer stored 1-D pixel centers. Otherwise infer from
    # pixel size and put (0,0) at the map center.
    x = get_first(d, ["x_centers", "x_center", "x"])
    y = get_first(d, ["y_centers", "y_center", "y"])
    pixel_size = get_first(d, ["pixel_size_pkpc", "pixel_size", "dx_pkpc", "dx"])

    ny, nx = rv.shape
    if x is not None and np.ndim(x) == 1 and len(x) == nx:
        x = np.asarray(x, dtype=float)
    else:
        dx = float(np.ravel(pixel_size)[0]) if pixel_size is not None else 1.0
        x = (np.arange(nx) - 0.5*(nx-1)) * dx

    if y is not None and np.ndim(y) == 1 and len(y) == ny:
        y = np.asarray(y, dtype=float)
    else:
        dx = float(np.ravel(pixel_size)[0]) if pixel_size is not None else 1.0
        y = (np.arange(ny) - 0.5*(ny-1)) * dx

    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    mw = mask & (rv >= args.mw_min) & (rv <= args.mw_max)

    qlevels = [0, 1, 5, 16, 50, 84, 95, 99, 100]
    q = np.percentile(vals, qlevels)

    print("\nR_V DISTRIBUTION")
    print("=" * 64)
    print(f"Input                    : {infile}")
    print(f"Map shape                : {ny} x {nx}")
    print(f"Valid pixels             : {vals.size:,} / {rv.size:,} "
          f"({100*vals.size/rv.size:.2f}%)")
    for lev, value in zip(qlevels, q):
        label = "min" if lev == 0 else ("max" if lev == 100 else f"p{lev:02d}")
        print(f"{label:24s} : {value:.6f}")

    def frac(condition):
        return np.count_nonzero(condition) / vals.size

    mw_count = np.count_nonzero(mw)
    print("\nMILKY-WAY-LIKE R_V")
    print("=" * 64)
    print(f"MW comparison interval   : {args.mw_min:.3f} <= R_V <= {args.mw_max:.3f}")
    print(f"Pixels in MW interval    : {mw_count:,} / {vals.size:,} "
          f"({100*mw_count/vals.size:.4f}%)")
    print(f"R_V < 4                  : {np.count_nonzero(vals < 4):,} / {vals.size:,} "
          f"({100*frac(vals < 4):.4f}%)")
    print(f"R_V < 5                  : {np.count_nonzero(vals < 5):,} / {vals.size:,} "
          f"({100*frac(vals < 5):.4f}%)")

    if mw_count:
        mw_r = rr[mw]
        print(f"MW-like radius median    : {np.median(mw_r):.3f} pkpc")
        print(f"MW-like radius p16--p84  : "
              f"{np.percentile(mw_r,16):.3f} -- {np.percentile(mw_r,84):.3f} pkpc")
    else:
        print("MW-like radius           : no matching pixels")

    # Save numerical summary.
    txt = Path(prefix + "_summary.txt")
    with txt.open("w") as f:
        f.write(f"input={infile}\n")
        f.write(f"valid_pixels={vals.size}\n")
        f.write(f"total_pixels={rv.size}\n")
        for lev, value in zip(qlevels, q):
            f.write(f"percentile_{lev}={value:.10g}\n")
        f.write(f"mw_min={args.mw_min}\n")
        f.write(f"mw_max={args.mw_max}\n")
        f.write(f"mw_count={mw_count}\n")
        f.write(f"mw_fraction={mw_count/vals.size:.10g}\n")
        f.write(f"fraction_rv_lt_4={frac(vals < 4):.10g}\n")
        f.write(f"fraction_rv_lt_5={frac(vals < 5):.10g}\n")

    # Histogram.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(args.mw_min, args.mw_max, alpha=0.18,
               label=f"MW observed range: {args.mw_min:g}–{args.mw_max:g}")
    ax.hist(vals, bins=70, histtype="step", linewidth=2)
    ax.axvline(np.median(vals), linestyle="--",
               label=f"CosmicGrain median = {np.median(vals):.2f}")
    ax.set_xlabel(r"$R_V$")
    ax.set_ylabel("Valid 250-pc pixels")
    ax.set_title(r"$R_V$ distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(prefix + "_histogram.png", dpi=180)
    plt.close(fig)

    # Map: gray all valid pixels, overlay MW-like pixels.
    fig, ax = plt.subplots(figsize=(8, 7))
    base = np.where(mask, rv, np.nan)
    im = ax.imshow(base, origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   interpolation="nearest", aspect="equal")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$R_V$")
    if mw_count:
        ax.scatter(xx[mw], yy[mw], s=13, facecolors="none",
                   edgecolors="white", linewidths=0.7,
                   label=f"{args.mw_min:g} ≤ $R_V$ ≤ {args.mw_max:g}")
        ax.legend(loc="upper right")
    ax.set_xlabel("x' [pkpc]")
    ax.set_ylabel("y' [pkpc]")
    ax.set_title("Locations of MW-like $R_V$ pixels")
    fig.tight_layout()
    fig.savefig(prefix + "_mw_like_map.png", dpi=180)
    plt.close(fig)

    # Radius scatter.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(rr[mask], rv[mask], s=4, alpha=0.18, rasterized=True)
    ax.axhspan(args.mw_min, args.mw_max, alpha=0.15,
               label=f"MW range {args.mw_min:g}–{args.mw_max:g}")
    ax.set_xlabel("Projected radius [pkpc]")
    ax.set_ylabel(r"$R_V$")
    ax.set_title(r"$R_V$ versus projected galactocentric radius")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(prefix + "_rv_vs_radius.png", dpi=180)
    plt.close(fig)

    print("\nSaved:")
    for name in [
        prefix + "_summary.txt",
        prefix + "_histogram.png",
        prefix + "_mw_like_map.png",
        prefix + "_rv_vs_radius.png",
    ]:
        print(" ", name)


if __name__ == "__main__":
    main()
