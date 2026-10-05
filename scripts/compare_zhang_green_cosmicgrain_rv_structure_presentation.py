#!/usr/bin/env python3
"""
Matched R(V) structure-function comparison:

    Zhang & Green Milky Way
        versus
    CosmicGrain CG10

Both maps are evaluated with exactly the same direct pair estimator:

    S2(r) = < [R_V(x+r) - R_V(x)]^2 >

Only pairs for which BOTH pixels are valid are included.  No
interpolation or hole filling is performed.

Also computes the variance-normalized structure function

    S2_norm(r) = S2(r) / (2 sigma_RV^2)

which separates the spatial correlation shape from the total variance
of the R(V) distribution.

The presentation figure compares only the adopted Zhang & Green and
CosmicGrain selections. No R(V) clipping is applied to CosmicGrain.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

# Use the standard CosmicGrain plotting style.
plt.style.use("cosmicgrain.mplstyle")


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "zg_npz",
        help="Zhang & Green products NPZ, e.g. "
             "zhang_green_rv_250pc_products.npz",
    )

    p.add_argument(
        "cg_rv",
        help="CosmicGrain 250-pc R(V) ASCII map",
    )

    p.add_argument(
        "cg_count",
        help="CosmicGrain 250-pc projected dust-count ASCII map",
    )

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
        "--sf-max",
        type=float,
        default=2.5,
        help="Maximum pair separation in physical kpc",
    )

    p.add_argument(
        "--sf-bins",
        type=int,
        default=20,
        help="Number of logarithmic separation bins",
    )



    p.add_argument(
        "--prefix",
        default="zhang_green_vs_cg10_rv_250pc",
    )

    return p.parse_args()


def structure_function(field, valid, dx, edges):
    """
    Direct second-order structure function.

    Every unordered pixel pair is counted exactly once.

    Parameters
    ----------
    field : 2D array
    valid : 2D boolean array
    dx    : pixel size in kpc
    edges : radial separation bin edges

    Returns
    -------
    rcenter, s2, npairs
    """

    nbins = len(edges) - 1

    sum_d2 = np.zeros(nbins, dtype=float)
    npairs = np.zeros(nbins, dtype=np.int64)

    ny, nx = field.shape

    rmax = edges[-1]
    max_shift = int(np.ceil(rmax / dx))

    for dy in range(max_shift + 1):

        for sx in range(-max_shift, max_shift + 1):

            # Count each unordered pair exactly once.
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


def summarize(name, rv, valid):

    v = rv[valid]

    print()
    print(name)
    print("-" * len(name))
    print(f"valid pixels = {valid.sum():,}")
    print(f"mean R(V)    = {np.mean(v):.6f}")
    print(f"median R(V)  = {np.median(v):.6f}")
    print(f"std R(V)     = {np.std(v):.6f}")
    print(f"p16          = {np.percentile(v, 16):.6f}")
    print(f"p84          = {np.percentile(v, 84):.6f}")
    print(f"min/max      = {np.min(v):.6f} / {np.max(v):.6f}")

    return np.std(v)


def main():

    a = parse_args()

    # ------------------------------------------------------------
    # Load Zhang & Green products
    # ------------------------------------------------------------

    z = np.load(a.zg_npz)

    print()
    print("Zhang & Green NPZ fields:")
    print("  " + ", ".join(z.files))

    if "rv" not in z.files:
        raise KeyError(
            "Z&G NPZ does not contain field 'rv'. "
            f"Available fields: {z.files}"
        )

    zg_rv = np.asarray(z["rv"], dtype=float)

    if "valid_mask" in z.files:
        zg_valid = np.asarray(z["valid_mask"]).astype(bool)
    else:
        zg_valid = np.isfinite(zg_rv)

    # Enforce finite values regardless of stored mask.
    zg_valid &= np.isfinite(zg_rv)

    # ------------------------------------------------------------
    # Load CosmicGrain products
    # ------------------------------------------------------------

    cg_rv = np.loadtxt(a.cg_rv)
    cg_count = np.loadtxt(a.cg_count)

    cg_rv = np.asarray(cg_rv, dtype=float)
    cg_count = np.asarray(cg_count, dtype=float)

    if cg_rv.shape != cg_count.shape:
        raise ValueError(
            f"CG shape mismatch: "
            f"RV={cg_rv.shape}, count={cg_count.shape}"
        )

    # ------------------------------------------------------------
    # Check pixel scale stored in Z&G file
    # ------------------------------------------------------------

    if "pixel_kpc" in z.files:

        zg_dx = float(np.asarray(z["pixel_kpc"]).ravel()[0])

        if not np.isclose(zg_dx, a.pixel):
            raise ValueError(
                f"Z&G NPZ pixel size is {zg_dx} kpc, "
                f"but --pixel={a.pixel} kpc"
            )

    # ------------------------------------------------------------
    # Build identical physical circular footprints
    # independently for each grid.
    #
    # Z&G will normally be 20x20.
    # CG export may be 80x80.
    # ------------------------------------------------------------

    def geometry(shape):

        ny, nx = shape

        x = (
            np.arange(nx) - 0.5 * (nx - 1)
        ) * a.pixel

        y = (
            np.arange(ny) - 0.5 * (ny - 1)
        ) * a.pixel

        xx, yy = np.meshgrid(x, y, indexing="xy")

        rr = np.hypot(xx, yy)

        return rr

    zg_rr = geometry(zg_rv.shape)
    cg_rr = geometry(cg_rv.shape)

    zg_footprint = zg_rr <= a.radius
    cg_footprint = cg_rr <= a.radius

    zg_valid &= zg_footprint

    # Primary CG selection:
    # mathematically valid R(V), inside aperture, >=1 dust particle.
    cg_valid = (
        cg_footprint
        & np.isfinite(cg_rv)
        & np.isfinite(cg_count)
        & (cg_count >= 1)
    )


    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("Matched R(V) structure-function comparison")
    print("------------------------------------------")
    print(f"pixel size          = {a.pixel:.4f} kpc")
    print(f"analysis radius     = {a.radius:.4f} kpc")
    print(f"maximum separation = {a.sf_max:.4f} kpc")
    print(f"SF bins             = {a.sf_bins}")
    print()

    print(
        f"Z&G footprint       = {zg_footprint.sum():,} pixels"
    )
    print(
        f"CG footprint        = {cg_footprint.sum():,} pixels"
    )

    sigma_zg = summarize(
        "Zhang & Green",
        zg_rv,
        zg_valid,
    )

    sigma_cg = summarize(
        "CosmicGrain CG10 -- primary",
        cg_rv,
        cg_valid,
    )




    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Use ONE common set of physical separation bins for all
    # samples.  This makes every plotted point directly comparable.
    # ------------------------------------------------------------

    edges = np.geomspace(
        a.pixel,
        a.sf_max,
        a.sf_bins + 1,
    )

    # ------------------------------------------------------------
    # Structure functions
    # ------------------------------------------------------------

    rz, s2_zg, np_zg = structure_function(
        zg_rv,
        zg_valid,
        a.pixel,
        edges,
    )

    rc, s2_cg, np_cg = structure_function(
        cg_rv,
        cg_valid,
        a.pixel,
        edges,
    )



    if not np.allclose(rz, rc):
        raise RuntimeError(
            "Internal error: separation grids differ."
        )

    r = rz

    # ------------------------------------------------------------
    # Variance-normalized structure functions
    # ------------------------------------------------------------

    norm_zg = s2_zg / (2.0 * sigma_zg**2)
    norm_cg = s2_cg / (2.0 * sigma_cg**2)

    # ------------------------------------------------------------
    # Save matched numerical table
    # ------------------------------------------------------------

    table = np.column_stack([
        r,
        s2_zg,
        norm_zg,
        np_zg,
        s2_cg,
        norm_cg,
        np_cg,
    ])

    np.savetxt(
        f"{a.prefix}_structure.txt",
        table,
        header=(
            "r_kpc "
            "ZG_S2 ZG_S2_over_2sigma2 ZG_Npairs "
            "CG_S2 CG_S2_over_2sigma2 CG_Npairs"
        ),
    )

    # ------------------------------------------------------------
    # Presentation figure: adopted samples only
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    zg_label = rf"Zhang & Green ($\sigma_{{R(V)}}={sigma_zg:.2f}$)"
    cg_label = rf"CosmicGrain CG10 ($\sigma_{{R(V)}}={sigma_cg:.2f}$)"

    # Absolute structure function.
    ax = axes[0]

    q = np.isfinite(s2_zg) & (s2_zg > 0)
    ax.loglog(
        r[q], s2_zg[q],
        marker="o", markersize=5, linewidth=2,
        label=zg_label,
    )

    q = np.isfinite(s2_cg) & (s2_cg > 0)
    ax.loglog(
        r[q], s2_cg[q],
        marker="o", markersize=5, linewidth=2,
        label=cg_label,
    )

    ax.set_ylabel(r"$S_2^{R(V)}(r)$")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left")

    # Variance-normalized structure function.
    ax = axes[1]

    q = np.isfinite(norm_zg) & (norm_zg > 0)
    ax.semilogx(
        r[q], norm_zg[q],
        marker="o", markersize=5, linewidth=2,
        label="Zhang & Green",
    )

    q = np.isfinite(norm_cg) & (norm_cg > 0)
    ax.semilogx(
        r[q], norm_cg[q],
        marker="o", markersize=5, linewidth=2,
        label="CosmicGrain CG10",
    )

    ax.set_xlabel(r"Separation $r$ (kpc)")
    ax.set_ylabel(r"$S_2(r)/(2\sigma_{R(V)}^2)$")
    ax.set_title("Variance-normalized $R(V)$ structure")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")

    fig.suptitle(
        "$R(V)$ spatial structure at 250 pc\n"
        "Milky Way vs. CosmicGrain CG10 Halo",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(f"{a.prefix}.png", dpi=220)
    fig.savefig(f"{a.prefix}.pdf")

    plt.close(fig)

    # ------------------------------------------------------------
    # Useful terminal comparison
    # ------------------------------------------------------------

    print()
    print("Structure-function bins")
    print("-----------------------")
    print(
        " r[kpc]    "
        "ZG_S2       CG_S2       "
        "ZG_norm     CG_norm     "
        "ZG_pairs  CG_pairs"
    )

    for i in range(len(r)):

        if (
            np_zg[i] == 0
            and np_cg[i] == 0
        ):
            continue

        print(
            f"{r[i]:7.3f}  "
            f"{s2_zg[i]:10.4g}  "
            f"{s2_cg[i]:10.4g}  "
            f"{norm_zg[i]:9.4f}  "
            f"{norm_cg[i]:9.4f}  "
            f"{np_zg[i]:8d}  "
            f"{np_cg[i]:8d}"
        )

    print()
    print("Outputs:")
    print(f"  {a.prefix}.png")
    print(f"  {a.prefix}.pdf")
    print(f"  {a.prefix}_structure.txt")
    print()


if __name__ == "__main__":
    main()
