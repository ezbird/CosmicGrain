#!/usr/bin/env python3
"""
Physical drivers of the high-R(V) population in CG10.

Starts from the SAME native master-map NPZ that was supplied to
export_halo441_maps_for_ft.py and reproduces its central 20-kpc,
250-pc aggregation.

For each 250-pc pixel, calculate:

  R(V) = sum(tau_V) / [sum(tau_B) - sum(tau_V)]

and dust-mass-weighted coarse quantities:

  <a>_M
  f_small
  <f_C>_M

The weighting is performed correctly from the native 125-pc maps:

  Q_250 = sum(Mdust_125 * Q_125) / sum(Mdust_125)

rather than directly averaging Q.

This therefore uses exactly the same 250-pc pixels as the
Zhang & Green / CosmicGrain structure-function comparison.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("npz")

    p.add_argument("--square-size", type=float, default=20.0)
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
        default="halo441_rv_250pc_physical_drivers",
    )

    return p.parse_args()


def block_sum(x, f):
    ny, nx = x.shape

    if ny % f or nx % f:
        raise ValueError(
            f"Array shape {x.shape} not divisible by factor {f}"
        )

    return x.reshape(
        ny // f, f,
        nx // f, f
    ).sum(axis=(1, 3))


def block_mass_weighted(q, mass, f):
    """
    Coarsen a native map whose values are already dust-mass-weighted
    within each native pixel.

    Qcoarse = sum(M_i Q_i) / sum(M_i)
    """

    good = (
        np.isfinite(q)
        & np.isfinite(mass)
        & (mass > 0)
    )

    numerator = block_sum(
        np.where(good, q * mass, 0.0),
        f,
    )

    denominator = block_sum(
        np.where(good, mass, 0.0),
        f,
    )

    out = np.full_like(numerator, np.nan, dtype=float)

    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]

    return out


def stats(label, q, rv, nd, props):

    n = q.sum()

    print()
    print(label)
    print("-" * len(label))
    print(f"N pixels               = {n}")

    if n == 0:
        return

    print(f"R(V) median            = {np.nanmedian(rv[q]):.5f}")
    print(f"N_dust median          = {np.nanmedian(nd[q]):.3f}")

    for name, arr in props:

        good = q & np.isfinite(arr)

        if not np.any(good):
            print(f"{name:23s} = no valid pixels")
            continue

        p16, p50, p84 = np.nanpercentile(
            arr[good],
            [16, 50, 84],
        )

        print(
            f"{name:23s} = "
            f"{p50:.5g} "
            f"[{p16:.5g}, {p84:.5g}]"
        )


def main():

    a = parse_args()

    z = np.load(a.npz)

    required = [
        "tau_b",
        "tau_v",
        "dust_count",
        "dust_mass_code",
        "mean_grain_radius_massweighted_nm",
        "small_grain_mass_fraction",
        "carbon_mass_fraction_massweighted",
        "pixel_size_pkpc",
    ]

    missing = [k for k in required if k not in z.files]

    if missing:
        raise KeyError(
            "NPZ missing required fields: "
            + ", ".join(missing)
        )

    tb0 = np.asarray(z["tau_b"], float)
    tv0 = np.asarray(z["tau_v"], float)
    nd0 = np.asarray(z["dust_count"], float)
    md0 = np.asarray(z["dust_mass_code"], float)

    grain0 = np.asarray(
        z["mean_grain_radius_massweighted_nm"],
        float,
    )

    small0 = np.asarray(
        z["small_grain_mass_fraction"],
        float,
    )

    carbon0 = np.asarray(
        z["carbon_mass_fraction_massweighted"],
        float,
    )

    dx0 = float(
        np.asarray(z["pixel_size_pkpc"]).ravel()[0]
    )

    print()
    print("Native master map")
    print("-----------------")
    print(f"file              = {a.npz}")
    print(f"native shape      = {tb0.shape}")
    print(f"native pixel      = {dx0:.6f} kpc")

    # -------------------------------------------------------
    # Reproduce exporter crop exactly.
    # -------------------------------------------------------

    N = tb0.shape[0]

    nn = int(round(a.square_size / dx0))

    if not np.isclose(nn * dx0, a.square_size):
        raise ValueError(
            "Requested square size incompatible with native pixel."
        )

    if nn > N:
        raise ValueError(
            "Requested square is larger than native map."
        )

    lo = (N - nn) // 2
    hi = lo + nn

    def crop(x):
        return x[lo:hi, lo:hi]

    tb0 = crop(tb0)
    tv0 = crop(tv0)
    nd0 = crop(nd0)
    md0 = crop(md0)

    grain0 = crop(grain0)
    small0 = crop(small0)
    carbon0 = crop(carbon0)

    # -------------------------------------------------------
    # Reproduce 250-pc aggregation.
    # -------------------------------------------------------

    f = int(round(a.pixel / dx0))

    if not np.isclose(f * dx0, a.pixel):
        raise ValueError(
            f"Requested pixel {a.pixel} incompatible "
            f"with native pixel {dx0}"
        )

    tb = block_sum(tb0, f)
    tv = block_sum(tv0, f)
    nd = block_sum(nd0, f)
    md = block_sum(md0, f)

    delta = tb - tv

    rv = np.full_like(tv, np.nan)

    good_rv = (
        np.isfinite(tb)
        & np.isfinite(tv)
        & (tv > 0)
        & (delta > 0)
    )

    rv[good_rv] = tv[good_rv] / delta[good_rv]

    # -------------------------------------------------------
    # Correct mass-weighted coarse grain properties.
    # -------------------------------------------------------

    grain = block_mass_weighted(
        grain0,
        md0,
        f,
    )

    small = block_mass_weighted(
        small0,
        md0,
        f,
    )

    carbon = block_mass_weighted(
        carbon0,
        md0,
        f,
    )

    # -------------------------------------------------------
    # Geometry
    # -------------------------------------------------------

    ny, nx = rv.shape

    x = (
        np.arange(nx) - 0.5 * (nx - 1)
    ) * a.pixel

    y = (
        np.arange(ny) - 0.5 * (ny - 1)
    ) * a.pixel

    xx, yy = np.meshgrid(x, y, indexing="xy")

    rr = np.hypot(xx, yy)

    footprint = rr <= a.radius

    valid = (
        footprint
        & np.isfinite(rv)
        & (nd >= 1)
    )

    print()
    print("250-pc reconstruction")
    print("---------------------")
    print(f"coarsening factor = {f}")
    print(f"coarse shape      = {rv.shape}")
    print(f"footprint pixels  = {footprint.sum()}")
    print(f"valid R(V) pixels = {valid.sum()}")
    print(f"mean R(V)         = {np.mean(rv[valid]):.6f}")
    print(f"median R(V)       = {np.median(rv[valid]):.6f}")
    print(f"std R(V)          = {np.std(rv[valid]):.6f}")

    # These should reproduce what we already measured:
    #
    # N = 152
    # mean = 5.237092
    # median = 3.375829
    # std = 4.120796

    # -------------------------------------------------------
    # Population statistics
    # -------------------------------------------------------

    props = [
        ("grain radius [nm]", grain),
        ("small-grain fraction", small),
        ("carbon fraction", carbon),
        ("dust mass [code]", md),
    ]

    stats(
        "R(V) <= 6",
        valid & (rv <= 6),
        rv, nd, props,
    )

    stats(
        "R(V) > 6",
        valid & (rv > 6),
        rv, nd, props,
    )

    stats(
        "R(V) > 10",
        valid & (rv > 10),
        rv, nd, props,
    )

    stats(
        "R(V) > 15",
        valid & (rv > 15),
        rv, nd, props,
    )

    # -------------------------------------------------------
    # Save pixel table
    # -------------------------------------------------------

    iy, ix = np.where(valid)

    table = np.column_stack([
        x[ix],
        y[iy],
        rr[iy, ix],
        rv[iy, ix],
        nd[iy, ix],
        md[iy, ix],
        grain[iy, ix],
        small[iy, ix],
        carbon[iy, ix],
    ])

    np.savetxt(
        f"{a.prefix}_pixels.txt",
        table,
        header=(
            "x_kpc y_kpc R_kpc RV N_dust DustMassCode "
            "MeanGrainRadiusMassWeighted_nm "
            "SmallGrainMassFraction "
            "CarbonMassFractionMassWeighted"
        ),
    )

    # -------------------------------------------------------
    # Figure: R(V) versus physical grain properties
    # -------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    relations = [
        (
            grain,
            r"Mass-weighted mean grain radius (nm)",
        ),
        (
            small,
            "Small-grain dust-mass fraction",
        ),
        (
            carbon,
            "Mass-weighted carbon fraction",
        ),
        (
            md,
            "Dust mass per 250 pc pixel (code units)",
        ),
    ]

    for ax, (prop, xlabel) in zip(
        axes.flat,
        relations,
    ):

        q = valid & np.isfinite(prop)

        sc = ax.scatter(
            prop[q],
            rv[q],
            c=nd[q],
            s=38,
            alpha=0.8,
        )

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(
            r"$N_{\rm dust}$"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$R(V)$")

        ax.axhline(
            6,
            linestyle="--",
            linewidth=1,
        )

        ax.axhline(
            10,
            linestyle=":",
            linewidth=1,
        )

        ax.grid(alpha=0.2)

    fig.suptitle(
        "CosmicGrain CG10: physical drivers of 250 pc R(V)",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        f"{a.prefix}_correlations.png",
        dpi=220,
    )

    plt.close(fig)

    # -------------------------------------------------------
    # Figure: distributions for ordinary/high R(V)
    # -------------------------------------------------------

    low = valid & (rv <= 6)
    high = valid & (rv > 6)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
    )

    distributions = [
        (
            grain,
            r"$\langle a\rangle_M$ (nm)",
        ),
        (
            small,
            r"$f_{\rm small}$",
        ),
        (
            carbon,
            r"$f_{\rm C}$",
        ),
    ]

    for ax, (prop, xlabel) in zip(
        axes,
        distributions,
    ):

        q1 = low & np.isfinite(prop)
        q2 = high & np.isfinite(prop)

        combined = prop[q1 | q2]

        if combined.size == 0:
            continue

        edges = np.linspace(
            np.nanmin(combined),
            np.nanmax(combined),
            18,
        )

        ax.hist(
            prop[q1],
            bins=edges,
            histtype="step",
            linewidth=2,
            density=True,
            label=rf"$R(V)\leq6$ (N={q1.sum()})",
        )

        ax.hist(
            prop[q2],
            bins=edges,
            histtype="step",
            linewidth=2,
            density=True,
            label=rf"$R(V)>6$ (N={q2.sum()})",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized pixel distribution")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "CG10 grain populations associated with high R(V)",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        f"{a.prefix}_distributions.png",
        dpi=220,
    )

    plt.close(fig)

    print()
    print("Outputs:")
    print(f"  {a.prefix}_correlations.png")
    print(f"  {a.prefix}_distributions.png")
    print(f"  {a.prefix}_pixels.txt")
    print()


if __name__ == "__main__":
    main()
