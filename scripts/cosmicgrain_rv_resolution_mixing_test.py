#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def block_sum(a, f):
    """Sum f x f blocks."""
    ny, nx = a.shape
    ny2 = (ny // f) * f
    nx2 = (nx // f) * f
    a = a[:ny2, :nx2]
    return a.reshape(ny2 // f, f, nx2 // f, f).sum(axis=(1, 3))


def rv_from_tau(tau_b, tau_v):
    d = tau_b - tau_v
    rv = np.full_like(tau_v, np.nan, dtype=float)

    good = (
        np.isfinite(tau_b)
        & np.isfinite(tau_v)
        & (tau_v > 0)
        & (d > 0)
    )

    rv[good] = tau_v[good] / d[good]
    return rv


def central_crop(a, n):
    ny, nx = a.shape
    if n > ny or n > nx:
        raise ValueError("Requested crop is larger than input map.")

    y0 = (ny - n) // 2
    x0 = (nx - n) // 2
    return a[y0:y0+n, x0:x0+n]


def percentile_string(x):
    if len(x) == 0:
        return "nan / nan / nan"
    q = np.percentile(x, [16, 50, 84])
    return f"{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_npz")
    ap.add_argument("--square-size", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument(
        "--pixel-sizes",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 1.0, 2.0],
    )
    ap.add_argument(
        "--prefix",
        default="halo441_current_rv_resolution_mixing"
    )
    args = ap.parse_args()

    d = np.load(args.master_npz)

    tau_b = np.asarray(d["tau_b"], dtype=float)
    tau_v = np.asarray(d["tau_v"], dtype=float)
    dust_count = np.asarray(d["dust_count"], dtype=float)

    native_pixel = float(np.asarray(d["pixel_size_pkpc"]).squeeze())

    print()
    print("INPUT")
    print("=" * 72)
    print(f"master file       = {args.master_npz}")
    print(f"master shape      = {tau_b.shape}")
    print(f"native pixel      = {native_pixel:.6f} pkpc")
    print(f"analysis square   = {args.square_size:.3f} pkpc")
    print(f"analysis radius   = {args.radius:.3f} pkpc")

    # First crop to exactly the same central square used in our FT products.
    n_native = int(round(args.square_size / native_pixel))

    if abs(n_native * native_pixel - args.square_size) > 1e-8:
        raise ValueError("square-size must be an integer number of native pixels")

    tb0 = central_crop(tau_b, n_native)
    tv0 = central_crop(tau_v, n_native)
    nd0 = central_crop(dust_count, n_native)

    print(f"cropped shape     = {tb0.shape}")
    print(f"cropped dust sum  = {nd0.sum():.0f}")

    results = []

    # Figure 1: histograms
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    # Figure 2: tail fraction vs resolution
    resolutions = []
    frac6 = []
    frac10 = []
    frac15 = []
    med_nd = []
    med_rv = []
    std_rv = []

    for pix in args.pixel_sizes:

        f = int(round(pix / native_pixel))

        if f < 1 or abs(f * native_pixel - pix) > 1e-8:
            raise ValueError(
                f"Pixel size {pix} is not an integer multiple "
                f"of native pixel {native_pixel}"
            )

        if tb0.shape[0] % f != 0 or tb0.shape[1] % f != 0:
            raise ValueError(
                f"{pix} kpc pixels do not divide the "
                f"{args.square_size} kpc square exactly."
            )

        tb = block_sum(tb0, f)
        tv = block_sum(tv0, f)
        nd = block_sum(nd0, f)

        rv = rv_from_tau(tb, tv)

        ny, nx = rv.shape

        # Pixel centers in physical kpc.
        x = (np.arange(nx) + 0.5 - nx / 2.0) * pix
        y = (np.arange(ny) + 0.5 - ny / 2.0) * pix
        xx, yy = np.meshgrid(x, y)

        aperture = np.hypot(xx, yy) <= args.radius

        valid = (
            aperture
            & np.isfinite(rv)
            & (nd >= 1)
        )

        vals = rv[valid]
        counts = nd[valid]

        nfoot = int(aperture.sum())
        nvalid = int(valid.sum())

        mean = np.mean(vals) if nvalid else np.nan
        median = np.median(vals) if nvalid else np.nan
        std = np.std(vals) if nvalid else np.nan

        p16, p84 = (
            np.percentile(vals, [16, 84])
            if nvalid else (np.nan, np.nan)
        )

        f6 = np.mean(vals > 6) if nvalid else np.nan
        f10 = np.mean(vals > 10) if nvalid else np.nan
        f15 = np.mean(vals > 15) if nvalid else np.nan

        nd_med = np.median(counts) if nvalid else np.nan
        nd_mean = np.mean(counts) if nvalid else np.nan

        resolutions.append(pix)
        frac6.append(f6)
        frac10.append(f10)
        frac15.append(f15)
        med_nd.append(nd_med)
        med_rv.append(median)
        std_rv.append(std)

        results.append([
            pix, nfoot, nvalid,
            mean, median, std, p16, p84,
            f6, f10, f15,
            nd_med, nd_mean
        ])

        print()
        print("=" * 72)
        print(f"{pix * 1000:.0f} pc")
        print("=" * 72)
        print(f"map shape             = {rv.shape}")
        print(f"footprint pixels      = {nfoot}")
        print(f"valid pixels          = {nvalid}")
        print(f"valid fraction        = {nvalid/nfoot:.5f}")
        print(f"mean R(V)             = {mean:.5f}")
        print(f"median R(V)           = {median:.5f}")
        print(f"std R(V)              = {std:.5f}")
        print(f"p16/median/p84        = {percentile_string(vals)}")
        print(f"fraction R(V) > 6     = {f6:.5f}")
        print(f"fraction R(V) > 10    = {f10:.5f}")
        print(f"fraction R(V) > 15    = {f15:.5f}")
        print(f"median N_dust/pixel   = {nd_med:.3f}")
        print(f"mean N_dust/pixel     = {nd_mean:.3f}")

        # Histogram as probability density.
        bins = np.linspace(1, 20, 77)
        ax1.hist(
            vals,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.8,
            label=f"{pix*1000:.0f} pc (N={nvalid})"
        )

    # ------------------------------------------------------------
    # Save numerical table
    # ------------------------------------------------------------

    results = np.asarray(results)

    header = (
        "pixel_kpc footprint_pixels valid_pixels "
        "rv_mean rv_median rv_std rv_p16 rv_p84 "
        "fraction_rv_gt6 fraction_rv_gt10 fraction_rv_gt15 "
        "median_dust_count mean_dust_count"
    )

    np.savetxt(
        args.prefix + "_summary.txt",
        results,
        header=header,
        fmt=[
            "%.6f", "%.0f", "%.0f",
            "%.8f", "%.8f", "%.8f", "%.8f", "%.8f",
            "%.8f", "%.8f", "%.8f",
            "%.8f", "%.8f"
        ]
    )

    # ------------------------------------------------------------
    # Histogram figure
    # ------------------------------------------------------------

    ax1.axvline(6, linestyle="--", linewidth=1)
    ax1.axvline(10, linestyle=":", linewidth=1)

    ax1.set_xlabel(r"$R(V)$")
    ax1.set_ylabel("Probability density")
    ax1.set_title("CG10: effect of physical pixel size on R(V)")
    ax1.legend(frameon=False)
    ax1.set_xlim(1, 20)

    fig1.tight_layout()
    fig1.savefig(args.prefix + "_distributions.png", dpi=220)
    plt.close(fig1)

    # ------------------------------------------------------------
    # Tail fraction figure
    # ------------------------------------------------------------

    resolutions = np.asarray(resolutions)
    frac6 = np.asarray(frac6)
    frac10 = np.asarray(frac10)
    frac15 = np.asarray(frac15)

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))

    ax2.plot(resolutions * 1000, frac6, "o-", label=r"$R(V)>6$")
    ax2.plot(resolutions * 1000, frac10, "o-", label=r"$R(V)>10$")
    ax2.plot(resolutions * 1000, frac15, "o-", label=r"$R(V)>15$")

    ax2.set_xscale("log")
    ax2.set_xlabel("Physical pixel size (pc)")
    ax2.set_ylabel("Fraction of valid pixels")
    ax2.set_title("CG10 high-R(V) tail vs. spatial averaging")
    ax2.legend(frameon=False)

    fig2.tight_layout()
    fig2.savefig(args.prefix + "_tail_fraction.png", dpi=220)
    plt.close(fig2)

    # ------------------------------------------------------------
    # Scatter / sampling figure
    # ------------------------------------------------------------

    fig3, ax3 = plt.subplots(figsize=(7.5, 5.5))

    ax3.plot(
        resolutions * 1000,
        std_rv,
        "o-",
        label=r"$\sigma_{R(V)}$"
    )

    ax3.set_xscale("log")
    ax3.set_xlabel("Physical pixel size (pc)")
    ax3.set_ylabel(r"$\sigma_{R(V)}$")
    ax3.set_title("CG10 R(V) scatter vs. spatial averaging")

    fig3.tight_layout()
    fig3.savefig(args.prefix + "_scatter.png", dpi=220)
    plt.close(fig3)

    # Dust count
    fig4, ax4 = plt.subplots(figsize=(7.5, 5.5))

    ax4.plot(
        resolutions * 1000,
        med_nd,
        "o-"
    )

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Physical pixel size (pc)")
    ax4.set_ylabel(r"Median $N_{\rm dust}$ per valid pixel")
    ax4.set_title("CG10 dust sampling vs. spatial averaging")

    fig4.tight_layout()
    fig4.savefig(args.prefix + "_dust_sampling.png", dpi=220)
    plt.close(fig4)

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)
    print(args.prefix + "_summary.txt")
    print(args.prefix + "_distributions.png")
    print(args.prefix + "_tail_fraction.png")
    print(args.prefix + "_scatter.png")
    print(args.prefix + "_dust_sampling.png")


if __name__ == "__main__":
    main()
