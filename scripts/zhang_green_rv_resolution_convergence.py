#!/usr/bin/env python3
"""
Resolution-convergence test for a single Zhang & Green R(V) realization.

Takes the products NPZ produced by zhang_green_rv_power_structure.py at
native 50-pc sampling and degrades that SAME realization to coarser
spatial resolutions.

Default resolutions:
    0.05, 0.10, 0.20, 0.25 kpc

The coarse R(V) field is NOT an unweighted block average.  We preserve
the original extinction weighting:

    Rv_coarse = sum(Rv_i * W_i) / sum(W_i),

where W_i is the projected AV_diff column weight saved by the original
script.

For every resolution this script computes:
  * the same R <= radius circular footprint
  * the same physical-width cosine apodization
  * masked pseudo-power spectrum k^2 P_R(V)(k)
  * second-order structure function S2_R(V)(r)

The goal is to distinguish real spatial structure from resolution /
pixel-scale behavior by keeping the underlying Z&G realization fixed.
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
        "products",
        nargs="?",
        default="zhang_green_rv_products.npz",
        help="NPZ from native-resolution zhang_green_rv_power_structure.py run",
    )

    p.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.25],
        help="Pixel sizes in kpc",
    )

    p.add_argument("--radius", type=float, default=2.5)
    p.add_argument("--apodize-physical", type=float, default=0.20,
                   help="Physical cosine taper width in kpc")
    p.add_argument("--nbins", type=int, default=35)
    p.add_argument("--sf-bins", type=int, default=30)
    p.add_argument("--sf-max", type=float, default=2.5)

    p.add_argument(
        "--field",
        choices=["absolute", "fractional"],
        default="absolute",
    )

    p.add_argument(
        "--prefix",
        default="zhang_green_rv_resolution_convergence",
    )

    return p.parse_args()


def block_sum(a, factor):
    """
    Exact factor x factor block sum.
    """
    ny, nx = a.shape

    if ny % factor != 0 or nx % factor != 0:
        raise ValueError(
            f"Array shape {a.shape} is not divisible by factor {factor}"
        )

    return a.reshape(
        ny // factor, factor,
        nx // factor, factor
    ).sum(axis=(1, 3))


def degrade(rv_native, weight_native, valid_native, factor):
    """
    Extinction-weighted degradation.

    Fine pixels that were invalid contribute zero weight.
    """
    good = (
        valid_native
        & np.isfinite(rv_native)
        & np.isfinite(weight_native)
        & (weight_native > 0)
    )

    w = np.where(good, weight_native, 0.0)
    rw = np.where(good, rv_native * weight_native, 0.0)

    wc = block_sum(w, factor)
    rwc = block_sum(rw, factor)

    rvc = np.full_like(wc, np.nan, dtype=float)

    q = wc > 0
    rvc[q] = rwc[q] / wc[q]

    return rvc, wc, q


def coordinates(n, dx):
    """
    Pixel-center coordinates for a square n x n map centered on zero.
    """
    half = 0.5 * n * dx
    edges = np.linspace(-half, half, n + 1)
    c = 0.5 * (edges[:-1] + edges[1:])

    xx, yy = np.meshgrid(c, c, indexing="xy")
    rr = np.hypot(xx, yy)

    return c, xx, yy, rr


def window(rr, radius, taper_width):
    """
    Circular cosine apodization with taper_width in physical kpc.
    """
    if taper_width <= 0:
        return (rr <= radius).astype(float)

    w = np.zeros_like(rr)

    inner = max(0.0, radius - taper_width)

    w[rr <= inner] = 1.0

    q = (rr > inner) & (rr < radius)

    x = (rr[q] - inner) / taper_width
    w[q] = 0.5 * (1.0 + np.cos(np.pi * x))

    return w


def power(field, valid, rr, dx, radius, taper_width, nbins):
    win = valid.astype(float) * window(
        rr, radius, taper_width
    )

    denom = np.sum(win[valid])

    if denom <= 0:
        raise RuntimeError("No valid weighted pixels for FFT")

    mu = np.sum(field[valid] * win[valid]) / denom

    f = np.zeros_like(field)
    f[valid] = field[valid] - mu

    F = (
        np.fft.fftshift(
            np.fft.fft2(f * win)
        )
        * dx**2
    )

    area = np.sum(win**2) * dx**2

    P2 = np.abs(F)**2 / area

    n = field.shape[0]

    kv = (
        2.0
        * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(n, d=dx)
        )
    )

    kx, ky = np.meshgrid(kv, kv, indexing="xy")
    kk = np.hypot(kx, ky)

    kfund = 2.0 * np.pi / (n * dx)
    knyq = np.pi / dx

    edges = np.geomspace(
        kfund,
        knyq,
        nbins + 1,
    )

    K = []
    P = []
    N = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        q = (kk >= lo) & (kk < hi)

        if q.any():
            K.append(math.sqrt(lo * hi))
            P.append(P2[q].mean())
            N.append(q.sum())

    return (
        np.asarray(K),
        np.asarray(P),
        np.asarray(N),
        win,
        mu,
        kfund,
        knyq,
        area,
    )


def structure(field, valid, dx, rmax, nbins):
    edges = np.geomspace(
        dx,
        rmax,
        nbins + 1,
    )

    ss = np.zeros(nbins)
    nn = np.zeros(nbins, dtype=np.int64)

    ny, nx = field.shape

    maxshift = int(np.floor(rmax / dx))

    for dy in range(maxshift + 1):
        for sx in range(-maxshift, maxshift + 1):

            if dy == 0 and sx <= 0:
                continue

            r = dx * math.hypot(sx, dy)

            if r < edges[0] or r >= edges[-1]:
                continue

            ib = np.searchsorted(
                edges,
                r,
                side="right",
            ) - 1

            ya = max(0, -dy)
            yb = min(ny, ny - dy)

            xa = max(0, -sx)
            xb = min(nx, nx - sx)

            if yb <= ya or xb <= xa:
                continue

            A = field[ya:yb, xa:xb]

            B = field[
                ya + dy:yb + dy,
                xa + sx:xb + sx,
            ]

            q = (
                valid[ya:yb, xa:xb]
                & valid[
                    ya + dy:yb + dy,
                    xa + sx:xb + sx,
                ]
                & np.isfinite(A)
                & np.isfinite(B)
            )

            if q.any():
                d = A[q] - B[q]

                ss[ib] += np.sum(d * d)
                nn[ib] += q.sum()

    rc = np.sqrt(edges[:-1] * edges[1:])

    s2 = np.full(nbins, np.nan)

    q = nn > 0
    s2[q] = ss[q] / nn[q]

    return rc, s2, nn


def main():
    a = parse_args()

    dat = np.load(a.products, allow_pickle=True)

    rv0 = np.asarray(dat["rv"], dtype=float)
    w0 = np.asarray(dat["column_weight"], dtype=float)
    valid0 = np.asarray(dat["valid_mask"], dtype=bool)

    dx0 = float(dat["pixel_kpc"])

    if rv0.ndim != 2 or rv0.shape[0] != rv0.shape[1]:
        raise ValueError(
            f"Expected square 2-D native map; got {rv0.shape}"
        )

    print()
    print("Native realization")
    print("------------------")
    print(f"file              = {a.products}")
    print(f"shape             = {rv0.shape}")
    print(f"native pixel      = {dx0:.6f} kpc")
    print(f"native valid      = {valid0.sum():,}")
    print(f"radius            = {a.radius:.3f} kpc")
    print(
        f"apodization width = "
        f"{a.apodize_physical:.3f} kpc"
    )
    print()

    results = []

    for target_dx in a.resolutions:

        factor_float = target_dx / dx0
        factor = int(round(factor_float))

        if not np.isclose(
            factor_float,
            factor,
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Requested {target_dx} kpc is not an "
                f"integer multiple of native {dx0} kpc"
            )

        if factor < 1:
            raise ValueError(
                f"Requested resolution {target_dx} kpc "
                f"is finer than native {dx0} kpc"
            )

        rv, weight, valid_data = degrade(
            rv0,
            w0,
            valid0,
            factor,
        )

        n = rv.shape[0]
        dx = dx0 * factor

        c, xx, yy, rr = coordinates(n, dx)

        footprint = rr <= a.radius

        valid = (
            valid_data
            & footprint
            & np.isfinite(rv)
            & (weight > 0)
        )

        field = rv.copy()

        mean = np.nanmean(rv[valid])

        if a.field == "fractional":
            field[valid] = rv[valid] / mean - 1.0

        (
            k,
            P,
            nmodes,
            win,
            mu,
            kfund,
            knyq,
            area,
        ) = power(
            field,
            valid,
            rr,
            dx,
            a.radius,
            a.apodize_physical,
            a.nbins,
        )

        k2P = k * k * P

        rs, s2, npairs = structure(
            field,
            valid,
            dx,
            a.sf_max,
            a.sf_bins,
        )

        print(
            f"{dx*1000:6.0f} pc : "
            f"{n:3d}x{n:<3d}  "
            f"valid={valid.sum():5d}/"
            f"{footprint.sum():5d}  "
            f"kfund={kfund:7.3f}  "
            f"kNyq={knyq:7.3f}  "
            f"Aeff={area:6.3f}"
        )

        results.append(
            dict(
                dx=dx,
                factor=factor,
                rv=rv,
                weight=weight,
                valid=valid,
                rr=rr,
                k=k,
                P=P,
                k2P=k2P,
                nmodes=nmodes,
                rs=rs,
                s2=s2,
                npairs=npairs,
                kfund=kfund,
                knyq=knyq,
                area=area,
                mean=mean,
            )
        )

    pre = Path(a.prefix)

    # ------------------------------------------------------------
    # Structure-function convergence
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for r in results:

        q = (
            np.isfinite(r["s2"])
            & (r["s2"] > 0)
            & (r["npairs"] > 0)
        )

        ax.loglog(
            r["rs"][q],
            r["s2"][q],
            marker="o",
            ms=4,
            label=f'{r["dx"]*1000:.0f} pc',
        )

    ax.set_xlabel(r"$r\ (\mathrm{kpc})$")
    ax.set_ylabel(r"$S_2^{R(V)}(r)$")

    ax.set_title(
        "Zhang & Green R(V): resolution convergence"
    )

    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        f"{pre}_structure_function.png",
        dpi=200,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Power-spectrum convergence
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for r in results:

        q = (
            np.isfinite(r["k2P"])
            & (r["k2P"] > 0)
        )

        ax.loglog(
            r["k"][q],
            r["k2P"][q],
            marker="o",
            ms=4,
            label=(
                f'{r["dx"]*1000:.0f} pc '
                f'(kNyq={r["knyq"]:.1f})'
            ),
        )

    ax.set_xlabel(
        r"$k\ (\mathrm{kpc}^{-1})$"
    )

    ax.set_ylabel(
        r"$k^2P_{R(V)}(k)$"
    )

    ax.set_title(
        "Zhang & Green R(V): resolution convergence"
    )

    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        f"{pre}_power_spectrum.png",
        dpi=200,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Save individual numerical curves
    # ------------------------------------------------------------

    for r in results:

        tag = f'{int(round(r["dx"]*1000)):04d}pc'

        np.savetxt(
            f"{pre}_{tag}_power.txt",
            np.c_[
                r["k"],
                r["P"],
                r["k2P"],
                r["nmodes"],
            ],
            header=(
                "k_kpc^-1 P_kpc2 "
                "k2P n_modes"
            ),
        )

        np.savetxt(
            f"{pre}_{tag}_structure.txt",
            np.c_[
                r["rs"],
                r["s2"],
                r["npairs"],
            ],
            header="r_kpc S2 n_pairs",
        )

    print()
    print("Outputs")
    print("-------")
    print(
        f"{pre}_structure_function.png"
    )
    print(
        f"{pre}_power_spectrum.png"
    )

    print()
    print(
        "Important: all curves are degraded from the "
        "same native realization."
    )


if __name__ == "__main__":
    main()
