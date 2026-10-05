#!/usr/bin/env python3
"""
cosmicgrain_rv_provenance_250pc.py

Trace the exact CG10 250-pc R(V) pixels back to PartType6 stellar provenance.

Critical validation:
  The PartType6 particles are projected using the center, rotation matrix,
  zmax, and geometry stored in the SAME master NPZ that generated the
  exported 250-pc R(V) maps.

Before interpreting provenance, the reconstructed particle-count map is
compared pixel-by-pixel against the exported 250pc_dust_count.txt map.

CosmicGrain DustSource convention:
    0 = SNII
    1 = AGB
    2 = LRN

Reports both particle-number and dust-mass provenance fractions.
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


SOURCE_NAMES = {
    0: "SNII",
    1: "AGB",
    2: "LRN",
}


def block_sum(a, f):
    ny, nx = a.shape
    if ny % f or nx % f:
        raise ValueError(
            f"Shape {a.shape} not divisible by factor {f}"
        )
    return a.reshape(
        ny // f, f,
        nx // f, f
    ).sum(axis=(1, 3))


def periodic_delta(x, center, box):
    d = x - center
    d -= box * np.rint(d / box)
    return d


def read_dust(pattern):
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No snapshot files matched:\n{pattern}"
        )

    chunks = {
        "Coordinates": [],
        "Masses": [],
        "DustSource": [],
        "GrainRadius": [],
        "CarbonMassFraction": [],
    }

    a = None
    h = None
    box = None

    for fn in files:
        with h5py.File(fn, "r") as f:

            if a is None:
                a = float(f["Header"].attrs["Time"])
                box = float(f["Header"].attrs["BoxSize"])

                # CosmicGrain convention: HubbleParam is in Parameters.
                h = float(
                    f["Parameters"].attrs["HubbleParam"]
                )

            if "PartType6" not in f:
                continue

            g = f["PartType6"]

            n = len(g["Coordinates"])

            chunks["Coordinates"].append(
                np.asarray(g["Coordinates"][:], float)
            )

            if "Masses" in g:
                chunks["Masses"].append(
                    np.asarray(g["Masses"][:], float)
                )
            else:
                m6 = float(f["Header"].attrs["MassTable"][6])
                chunks["Masses"].append(
                    np.full(n, m6, dtype=float)
                )

            chunks["DustSource"].append(
                np.asarray(g["DustSource"][:], int)
            )

            chunks["GrainRadius"].append(
                np.asarray(g["GrainRadius"][:], float)
            )

            chunks["CarbonMassFraction"].append(
                np.asarray(
                    g["CarbonMassFraction"][:],
                    float
                )
            )

    dust = {
        k: np.concatenate(v)
        for k, v in chunks.items()
    }

    return dust, a, h, box


def hist2d_sum(x, y, weights, edges):
    H, _, _ = np.histogram2d(
        y,
        x,
        bins=[edges, edges],
        weights=weights,
    )
    return H


def hist2d_count(x, y, edges):
    H, _, _ = np.histogram2d(
        y,
        x,
        bins=[edges, edges],
    )
    return H.astype(int)


def summarize_population(label, mask, rv,
                         nsrc, msrc, total_count, total_mass):

    npix = np.count_nonzero(mask)

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"N pixels = {npix}")

    if npix == 0:
        return

    print(
        f"R(V) median = "
        f"{np.nanmedian(rv[mask]):.5f}"
    )

    print(
        f"N_dust median = "
        f"{np.nanmedian(total_count[mask]):.3f}"
    )

    # Sum over all selected pixels. This answers:
    # "Of all particles/mass represented in this R(V) population,
    #  what fraction came from each stellar source?"

    Ntot = sum(
        np.sum(nsrc[s][mask])
        for s in SOURCE_NAMES
    )

    Mtot = sum(
        np.sum(msrc[s][mask])
        for s in SOURCE_NAMES
    )

    print()
    print("Integrated provenance across these pixels:")
    print()
    print(
        f"{'Source':8s} "
        f"{'Npart':>10s} "
        f"{'N frac':>12s} "
        f"{'Mass frac':>12s}"
    )

    for s, name in SOURCE_NAMES.items():

        ns = np.sum(nsrc[s][mask])
        ms = np.sum(msrc[s][mask])

        nf = ns / Ntot if Ntot > 0 else np.nan
        mf = ms / Mtot if Mtot > 0 else np.nan

        print(
            f"{name:8s} "
            f"{int(ns):10d} "
            f"{nf:12.5f} "
            f"{mf:12.5f}"
        )

    # Also calculate per-pixel mass fractions.
    print()
    print("Median per-pixel dust-mass fractions:")

    for s, name in SOURCE_NAMES.items():

        frac = np.full_like(
            total_mass,
            np.nan,
            dtype=float,
        )

        good = total_mass > 0
        frac[good] = (
            msrc[s][good] / total_mass[good]
        )

        vals = frac[mask & np.isfinite(frac)]

        if vals.size:
            q16, q50, q84 = np.percentile(
                vals,
                [16, 50, 84],
            )

            print(
                f"  {name:5s}: "
                f"{q50:.5f} "
                f"[{q16:.5f}, {q84:.5f}]"
            )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "snapshot",
        help=(
            "Snapshot glob, e.g. "
            "'../halo441_output_1024/"
            "snapdir_029/snapshot_029.*.hdf5'"
        ),
    )

    ap.add_argument(
        "master_npz",
        help="Original 125-pc R(V) master NPZ",
    )

    ap.add_argument(
        "exported_rv",
        help="Exported 250pc_RV_unmasked.txt",
    )

    ap.add_argument(
        "exported_count",
        help="Exported 250pc_dust_count.txt",
    )

    ap.add_argument(
        "--square-size",
        type=float,
        default=20.0,
    )

    ap.add_argument(
        "--pixel",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--radius",
        type=float,
        default=2.5,
    )

    ap.add_argument(
        "--prefix",
        default="halo441_rv_250pc_provenance",
    )

    args = ap.parse_args()

    # ------------------------------------------------------------
    # Master-map geometry
    # ------------------------------------------------------------

    z = np.load(args.master_npz)

    center = np.asarray(z["center"], float)
    rotation = np.asarray(z["rotation"], float)

    native_pixel = float(
        np.asarray(
            z["pixel_size_pkpc"]
        ).ravel()[0]
    )

    master_radius = float(
        np.asarray(
            z["plot_radius_pkpc"]
        ).ravel()[0]
    )

    zmax = float(
        np.asarray(
            z["zmax_pkpc"]
        ).ravel()[0]
    )

    print()
    print("MASTER MAP GEOMETRY")
    print("=" * 72)
    print(f"center [ckpc/h] = {center}")
    print("rotation =")
    print(rotation)
    print(f"native pixel     = {native_pixel:.6f} pkpc")
    print(f"master radius    = {master_radius:.6f} pkpc")
    print(f"zmax             = {zmax:.6f} pkpc")

    # ------------------------------------------------------------
    # Read particles
    # ------------------------------------------------------------

    dust, a, h, box = read_dust(args.snapshot)

    xyz = dust["Coordinates"]
    mass = dust["Masses"]
    source = dust["DustSource"]
    grain = dust["GrainRadius"]
    carbon = dust["CarbonMassFraction"]

    print()
    print("SNAPSHOT")
    print("=" * 72)
    print(f"N PartType6      = {len(xyz):,}")
    print(f"a                = {a:.8f}")
    print(f"h                = {h:.8f}")
    print(f"BoxSize          = {box:.8f} ckpc/h")

    print()
    print("Global DustSource counts:")

    for s, name in SOURCE_NAMES.items():
        q = source == s
        print(
            f"  {name:5s}: "
            f"{q.sum():8d} "
            f"({q.mean():.5f})"
        )

    # ------------------------------------------------------------
    # Exact projection geometry
    # ------------------------------------------------------------

    d = periodic_delta(
        xyz,
        center,
        box,
    )

    # Convert comoving ckpc/h displacement to physical pkpc.
    d *= a / h

    # Same convention as master-map scripts:
    # rotated coordinates = displacement @ rotation.T
    u = d @ rotation.T

    x = u[:, 0]
    y = u[:, 1]
    zz = u[:, 2]

    half = 0.5 * args.square_size

    selected = (
        (x >= -half)
        & (x < half)
        & (y >= -half)
        & (y < half)
        & (np.abs(zz) <= zmax)
    )

    print()
    print("PROJECTION")
    print("=" * 72)
    print(
        f"Particles in {args.square_size:g} x "
        f"{args.square_size:g} pkpc square, "
        f"|z| <= {zmax:g} pkpc: "
        f"{selected.sum():,}"
    )

    xs = x[selected]
    ys = y[selected]
    ms = mass[selected]
    ss = source[selected]
    gs = grain[selected]
    cs = carbon[selected]

    # ------------------------------------------------------------
    # Construct native 125-pc map first.
    #
    # This intentionally follows the lineage:
    #
    # PartType6 -> native 125 pc -> crop/export -> 250 pc
    #
    # rather than directly histogramming at 250 pc.
    # ------------------------------------------------------------

    nn = int(round(
        args.square_size / native_pixel
    ))

    if not np.isclose(
        nn * native_pixel,
        args.square_size,
    ):
        raise ValueError(
            "Square size incompatible with native pixel size"
        )

    edges_native = np.linspace(
        -half,
        half,
        nn + 1,
    )

    count_native = hist2d_count(
        xs,
        ys,
        edges_native,
    )

    mass_native = hist2d_sum(
        xs,
        ys,
        ms,
        edges_native,
    )

    nsrc_native = {}
    msrc_native = {}

    for s in SOURCE_NAMES:

        q = ss == s

        nsrc_native[s] = hist2d_count(
            xs[q],
            ys[q],
            edges_native,
        )

        msrc_native[s] = hist2d_sum(
            xs[q],
            ys[q],
            ms[q],
            edges_native,
        )

    # Optional physical-property maps from particles themselves.
    grain_mass_native = hist2d_sum(
        xs,
        ys,
        ms * gs,
        edges_native,
    )

    carbon_mass_native = hist2d_sum(
        xs,
        ys,
        ms * cs,
        edges_native,
    )

    # ------------------------------------------------------------
    # 125 -> 250 pc
    # ------------------------------------------------------------

    factor = int(round(
        args.pixel / native_pixel
    ))

    if not np.isclose(
        factor * native_pixel,
        args.pixel,
    ):
        raise ValueError(
            "Requested coarse pixel incompatible "
            "with master native pixel."
        )

    count = block_sum(
        count_native,
        factor,
    )

    massmap = block_sum(
        mass_native,
        factor,
    )

    nsrc = {
        s: block_sum(
            nsrc_native[s],
            factor,
        )
        for s in SOURCE_NAMES
    }

    msrc = {
        s: block_sum(
            msrc_native[s],
            factor,
        )
        for s in SOURCE_NAMES
    }

    grain_mass = block_sum(
        grain_mass_native,
        factor,
    )

    carbon_mass = block_sum(
        carbon_mass_native,
        factor,
    )

    mean_grain = np.full_like(
        massmap,
        np.nan,
        dtype=float,
    )

    mean_carbon = np.full_like(
        massmap,
        np.nan,
        dtype=float,
    )

    qm = massmap > 0

    mean_grain[qm] = (
        grain_mass[qm] / massmap[qm]
    )

    mean_carbon[qm] = (
        carbon_mass[qm] / massmap[qm]
    )

    # ------------------------------------------------------------
    # Load exact exported maps
    # ------------------------------------------------------------

    rv = np.loadtxt(args.exported_rv)
    count_export = np.loadtxt(
        args.exported_count
    ).astype(int)

    if rv.shape != count.shape:
        raise RuntimeError(
            f"Shape mismatch: RV {rv.shape}, "
            f"particle map {count.shape}"
        )

    # ------------------------------------------------------------
    # THE CRITICAL VALIDATION
    # ------------------------------------------------------------

    diff = count - count_export

    nbad = np.count_nonzero(diff)
    maxdiff = (
        np.max(np.abs(diff))
        if diff.size else 0
    )

    print()
    print("=" * 72)
    print("EXACT MAP VALIDATION")
    print("=" * 72)
    print(
        f"Reconstructed shape                = {count.shape}"
    )
    print(
        f"Maximum absolute count difference = {maxdiff}"
    )
    print(
        f"Mismatched pixels                 = "
        f"{nbad} / {diff.size}"
    )

    if nbad != 0:

        print()
        print("VALIDATION FAILED.")
        print(
            "Do NOT interpret the provenance results."
        )

        bad = np.argwhere(diff != 0)

        print()
        print("First mismatches:")

        for iy, ix in bad[:20]:
            print(
                f"  iy={iy:3d} ix={ix:3d} "
                f"reconstructed={count[iy,ix]} "
                f"exported={count_export[iy,ix]} "
                f"diff={diff[iy,ix]}"
            )

        raise SystemExit(1)

    print()
    print("VALIDATION PASSED EXACTLY.")

    # ------------------------------------------------------------
    # Circular Z&G comparison footprint
    # ------------------------------------------------------------

    ny, nx = rv.shape

    xc = (
        np.arange(nx)
        - 0.5 * (nx - 1)
    ) * args.pixel

    yc = (
        np.arange(ny)
        - 0.5 * (ny - 1)
    ) * args.pixel

    X, Y = np.meshgrid(
        xc,
        yc,
        indexing="xy",
    )

    R = np.hypot(X, Y)

    footprint = R <= args.radius

    valid = (
        footprint
        & np.isfinite(rv)
        & (count >= 1)
    )

    print()
    print("R(V) FOOTPRINT")
    print("=" * 72)
    print(
        f"footprint pixels = {footprint.sum()}"
    )
    print(
        f"valid pixels     = {valid.sum()}"
    )
    print(
        f"mean R(V)        = "
        f"{np.mean(rv[valid]):.6f}"
    )
    print(
        f"median R(V)      = "
        f"{np.median(rv[valid]):.6f}"
    )
    print(
        f"std R(V)         = "
        f"{np.std(rv[valid]):.6f}"
    )

    # ------------------------------------------------------------
    # Provenance populations
    # ------------------------------------------------------------

    regimes = [
        (
            r"$R(V)\leq6$",
            valid & (rv <= 6),
        ),
        (
            r"$R(V)>6$",
            valid & (rv > 6),
        ),
        (
            r"$R(V)>10$",
            valid & (rv > 10),
        ),
        (
            r"$R(V)>15$",
            valid & (rv > 15),
        ),
    ]

    for label, q in regimes:

        summarize_population(
            label,
            q,
            rv,
            nsrc,
            msrc,
            count,
            massmap,
        )

    # ------------------------------------------------------------
    # Per-pixel provenance fractions
    # ------------------------------------------------------------

    frac_mass = {}

    for s in SOURCE_NAMES:

        arr = np.full_like(
            massmap,
            np.nan,
            dtype=float,
        )

        arr[qm] = (
            msrc[s][qm] / massmap[qm]
        )

        frac_mass[s] = arr

    # ------------------------------------------------------------
    # Save exact pixel table
    # ------------------------------------------------------------

    iy, ix = np.where(valid)

    table = np.column_stack([
        X[iy, ix],
        Y[iy, ix],
        R[iy, ix],
        rv[iy, ix],
        count[iy, ix],
        massmap[iy, ix],
        mean_grain[iy, ix],
        mean_carbon[iy, ix],
        frac_mass[0][iy, ix],
        frac_mass[1][iy, ix],
        frac_mass[2][iy, ix],
        nsrc[0][iy, ix],
        nsrc[1][iy, ix],
        nsrc[2][iy, ix],
    ])

    np.savetxt(
        args.prefix + "_pixels.txt",
        table,
        header=(
            "x_kpc y_kpc R_kpc RV N_dust "
            "DustMassCode "
            "MeanGrainRadiusMassWeighted_nm "
            "CarbonMassFractionMassWeighted "
            "SNII_mass_fraction "
            "AGB_mass_fraction "
            "LRN_mass_fraction "
            "N_SNII N_AGB N_LRN"
        ),
    )

    # ------------------------------------------------------------
    # Plot R(V) versus provenance
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )

    for ax, s in zip(
        axes,
        [0, 1, 2],
    ):

        name = SOURCE_NAMES[s]

        q = (
            valid
            & np.isfinite(frac_mass[s])
        )

        sc = ax.scatter(
            frac_mass[s][q],
            rv[q],
            c=count[q],
            s=42,
            alpha=0.8,
        )

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

        ax.set_xlim(-0.04, 1.04)

        ax.set_xlabel(
            f"{name} dust-mass fraction"
        )

        ax.grid(alpha=0.2)

        cb = fig.colorbar(
            sc,
            ax=ax,
        )

        cb.set_label(
            r"$N_{\rm dust}$"
        )

    axes[0].set_ylabel(
        r"$R(V)$"
    )

    fig.suptitle(
        "CG10: stellar provenance of 250 pc extinction pixels",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        args.prefix
        + "_rv_vs_provenance.png",
        dpi=220,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Spatial map of dominant source
    # ------------------------------------------------------------

    dominant = np.full(
        rv.shape,
        np.nan,
    )

    stack = np.stack([
        frac_mass[0],
        frac_mass[1],
        frac_mass[2],
    ])

    good = (
        valid
        & np.all(
            np.isfinite(stack),
            axis=0,
        )
    )

    dominant[good] = np.argmax(
        stack[:, good],
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    im = ax.imshow(
        dominant,
        origin="lower",
        extent=[
            -args.square_size / 2,
            args.square_size / 2,
            -args.square_size / 2,
            args.square_size / 2,
        ],
        interpolation="nearest",
        vmin=-0.5,
        vmax=2.5,
    )

    circle = plt.Circle(
        (0, 0),
        args.radius,
        fill=False,
        linewidth=1.5,
    )

    ax.add_patch(circle)

    cb = fig.colorbar(
        im,
        ax=ax,
        ticks=[0, 1, 2],
    )

    cb.ax.set_yticklabels([
        "SNII",
        "AGB",
        "LRN",
    ])

    cb.set_label(
        "Dominant dust source by mass"
    )

    ax.set_xlim(
        -args.radius - 0.25,
        args.radius + 0.25,
    )

    ax.set_ylim(
        -args.radius - 0.25,
        args.radius + 0.25,
    )

    ax.set_xlabel("x (pkpc)")
    ax.set_ylabel("y (pkpc)")

    ax.set_title(
        "CG10 — dominant stellar provenance"
    )

    fig.tight_layout()

    fig.savefig(
        args.prefix
        + "_dominant_source_map.png",
        dpi=220,
    )

    plt.close(fig)

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)
    print(
        args.prefix
        + "_rv_vs_provenance.png"
    )
    print(
        args.prefix
        + "_dominant_source_map.png"
    )
    print(
        args.prefix
        + "_pixels.txt"
    )


if __name__ == "__main__":
    main()
