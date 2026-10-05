#!/usr/bin/env python3

import argparse
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


# ================================================================
# Utilities
# ================================================================

def load_dust(snapshot, center, rotation):

    files = sorted(glob.glob(snapshot))
    if not files:
        raise FileNotFoundError(snapshot)

    with h5py.File(files[0], "r") as f:
        a = float(f["Header"].attrs.get("Time", 1.0))
        h = f["Header"].attrs.get("HubbleParam", None)

        if h is None and "Parameters" in f:
            h = f["Parameters"].attrs.get("HubbleParam", None)

        if h is None:
            raise RuntimeError("HubbleParam not found")

        h = float(h)
        box = float(f["Header"].attrs["BoxSize"])

    xyz = []
    mass = []

    for i, fn in enumerate(files, 1):

        print(
            f"\rReading snapshot {i}/{len(files)}",
            end="",
            flush=True
        )

        with h5py.File(fn, "r") as f:

            if "PartType6" not in f:
                continue

            g = f["PartType6"]

            xyz.append(
                np.asarray(g["Coordinates"], float)
            )

            mass.append(
                np.asarray(g["Masses"], float)
            )

    print()

    xyz = np.concatenate(xyz)
    mass = np.concatenate(mass)

    d = xyz - center[None, :]
    d -= box * np.rint(d / box)

    # ckpc/h -> physical pkpc
    d *= a / h

    u = d @ rotation.T

    return u, mass, a, h, box


def cosine_apodization(mask, width_pix):

    mask = np.asarray(mask, bool)

    if width_pix <= 0:
        return mask.astype(float)

    dist = distance_transform_edt(mask)

    w = np.ones_like(dist, dtype=float)

    q = mask & (dist < width_pix)

    w[q] = 0.5 * (
        1.0 - np.cos(np.pi * dist[q] / width_pix)
    )

    w[~mask] = 0.0

    return w


# ================================================================
# Power spectrum
#
# Same basic convention as our Zhang & Green matched calculation.
# ================================================================

def power_spectrum_2d(
    sigma,
    mask,
    dx,
    apodize_pix=4,
    nbins=35
):

    valid = (
        mask
        & np.isfinite(sigma)
    )

    mean = np.mean(sigma[valid])

    if mean <= 0:
        raise RuntimeError(
            "Mean surface density <= 0."
        )

    delta = np.zeros_like(sigma, dtype=float)

    delta[valid] = (
        sigma[valid] / mean - 1.0
    )

    window = cosine_apodization(
        valid,
        apodize_pix
    )

    field = delta * window

    ny, nx = field.shape

    F = (
        dx**2
        * np.fft.fftshift(
            np.fft.fft2(field)
        )
    )

    area_eff = (
        np.sum(window**2) * dx**2
    )

    P2 = np.abs(F)**2 / area_eff

    kx = (
        2.0 * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(nx, d=dx)
        )
    )

    ky = (
        2.0 * np.pi
        * np.fft.fftshift(
            np.fft.fftfreq(ny, d=dx)
        )
    )

    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)

    Lx = nx * dx
    Ly = ny * dx

    kfund = 2.0 * np.pi / max(Lx, Ly)
    kny = np.pi / dx

    edges = np.geomspace(
        kfund,
        kny,
        nbins + 1
    )

    kout = []
    pout = []
    nout = []

    for lo, hi in zip(
        edges[:-1],
        edges[1:]
    ):

        q = (
            (K >= lo)
            & (K < hi)
            & np.isfinite(P2)
        )

        if not np.any(q):
            continue

        kout.append(
            np.exp(
                np.mean(np.log(K[q]))
            )
        )

        pout.append(
            np.mean(P2[q])
        )

        nout.append(
            np.count_nonzero(q)
        )

    k = np.asarray(kout)
    P = np.asarray(pout)
    nmodes = np.asarray(nout)

    return {
        "k": k,
        "P": P,
        "k2P": k**2 * P,
        "nmodes": nmodes,
        "kfund": kfund,
        "kny": kny,
        "mean": mean
    }


# ================================================================
# Map construction
# ================================================================

def make_map(
    x,
    y,
    mass,
    radius,
    dx
):

    nfloat = 2.0 * radius / dx
    n = int(round(nfloat))

    if not np.isclose(nfloat, n):
        raise ValueError(
            f"2R/dx must be integer: "
            f"R={radius}, dx={dx}"
        )

    edges = np.linspace(
        -radius,
        radius,
        n + 1
    )

    mass_map = np.histogram2d(
        y,
        x,
        bins=[edges, edges],
        weights=mass
    )[0]

    count_map = np.histogram2d(
        y,
        x,
        bins=[edges, edges]
    )[0]

    centers = (
        0.5 * (edges[:-1] + edges[1:])
    )

    X, Y = np.meshgrid(
        centers,
        centers
    )

    mask = (
        np.hypot(X, Y) <= radius
    )

    sigma = mass_map / dx**2

    return (
        sigma,
        mass_map,
        count_map,
        mask,
        centers
    )


# ================================================================
# Radial profile
# ================================================================

def radial_profile(
    x,
    y,
    mass,
    rmax=20.0,
    dr=0.5
):

    r = np.hypot(x, y)

    edges = np.arange(
        0.0,
        rmax + dr,
        dr
    )

    rc = 0.5 * (
        edges[:-1] + edges[1:]
    )

    sigma = np.zeros_like(rc)
    counts = np.zeros_like(rc)

    for i in range(len(rc)):

        q = (
            (r >= edges[i])
            & (r < edges[i+1])
        )

        area = np.pi * (
            edges[i+1]**2
            - edges[i]**2
        )

        sigma[i] = (
            np.sum(mass[q]) / area
        )

        counts[i] = np.count_nonzero(q)

    return rc, sigma, counts


# ================================================================
# Randomized controls
# ================================================================

def randomized_xy(
    x,
    y,
    rng
):

    r = np.hypot(x, y)

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=len(r)
    )

    xr = r * np.cos(phi)
    yr = r * np.sin(phi)

    return xr, yr


def random_control(
    x,
    y,
    mass,
    radius,
    dx,
    apodize_pix,
    nbins,
    nrandom,
    seed
):

    rng = np.random.default_rng(seed)

    spectra = []

    reference_k = None

    for i in range(nrandom):

        xr, yr = randomized_xy(
            x,
            y,
            rng
        )

        sigma, _, _, mask, _ = make_map(
            xr,
            yr,
            mass,
            radius,
            dx
        )

        s = power_spectrum_2d(
            sigma,
            mask,
            dx,
            apodize_pix=apodize_pix,
            nbins=nbins
        )

        if reference_k is None:
            reference_k = s["k"]

        spectra.append(
            s["k2P"]
        )

    arr = np.asarray(spectra)

    return {
        "k": reference_k,
        "median": np.nanmedian(
            arr,
            axis=0
        ),
        "p16": np.nanpercentile(
            arr,
            16,
            axis=0
        ),
        "p84": np.nanpercentile(
            arr,
            84,
            axis=0
        ),
        "all": arr
    }


# ================================================================
# Main
# ================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "snapshot"
    )

    ap.add_argument(
        "--center-file",
        required=True
    )

    ap.add_argument(
        "--rotation-file",
        required=True
    )

    ap.add_argument(
        "--zmax",
        type=float,
        default=0.4
    )

    ap.add_argument(
        "--map-radii",
        nargs="+",
        type=float,
        default=[2.5, 5, 10, 20]
    )

    ap.add_argument(
        "--pixel-size",
        type=float,
        default=0.25
    )

    ap.add_argument(
        "--apodize-pix",
        type=int,
        default=1
    )

    ap.add_argument(
        "--nbins",
        type=int,
        default=35
    )

    ap.add_argument(
        "--nrandom",
        type=int,
        default=100
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    ap.add_argument(
        "--radial-dr",
        type=float,
        default=0.5
    )

    ap.add_argument(
        "--prefix",
        default="CG10_dust_spatial_sampling"
    )

    a = ap.parse_args()

    center = np.loadtxt(
        a.center_file
    ).reshape(3)

    rotation = np.loadtxt(
        a.rotation_file
    ).reshape(3, 3)

    u, mass, scale, hubble, box = load_dust(
        a.snapshot,
        center,
        rotation
    )

    # Thin disk selection.
    slab = (
        np.abs(u[:, 2]) <= a.zmax
    )

    xall = u[slab, 0]
    yall = u[slab, 1]
    mall = mass[slab]

    rall = np.hypot(
        xall,
        yall
    )

    print()
    print("=" * 72)
    print("CG10 DUST SPATIAL SAMPLING TEST")
    print("=" * 72)

    print(
        f"Total PartType6       : {len(u):,}"
    )

    print(
        f"|z| < {a.zmax:g} pkpc     : "
        f"{len(xall):,}"
    )

    print(
        f"Dust mass in slab     : "
        f"{mall.sum():.8e}"
    )

    # ------------------------------------------------------------
    # Large-scale 40x40 pkpc map
    # ------------------------------------------------------------

    rlarge = max(a.map_radii)

    qlarge = (
        rall <= rlarge
    )

    sigma_large, mass_large, count_large, mask_large, centers = (
        make_map(
            xall[qlarge],
            yall[qlarge],
            mall[qlarge],
            rlarge,
            a.pixel_size
        )
    )

    extent = [
        -rlarge,
        rlarge,
        -rlarge,
        rlarge
    ]

    # Log Sigma map
    positive = (
        sigma_large > 0
    )

    log_sigma = np.full_like(
        sigma_large,
        np.nan,
        dtype=float
    )

    log_sigma[positive] = np.log10(
        sigma_large[positive]
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im = ax.imshow(
        log_sigma,
        origin="lower",
        extent=extent,
        cmap="magma"
    )

    ax.set_xlabel(
        r"$x\ ({\rm pkpc})$"
    )

    ax.set_ylabel(
        r"$y\ ({\rm pkpc})$"
    )

    ax.set_title(
        rf"CG10 dust: $|z|<{a.zmax:g}$ pkpc"
    )

    cb = fig.colorbar(
        im,
        ax=ax
    )

    cb.set_label(
        r"$\log_{10}\Sigma_{\rm dust}$ "
        r"(code mass / pkpc$^2$)"
    )

    fig.tight_layout()

    fig.savefig(
        a.prefix + "_map.png",
        dpi=250
    )

    plt.close(fig)

    # Particle-count map
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im = ax.imshow(
        np.where(
            mask_large,
            count_large,
            np.nan
        ),
        origin="lower",
        extent=extent,
        cmap="viridis"
    )

    ax.set_xlabel(
        r"$x\ ({\rm pkpc})$"
    )

    ax.set_ylabel(
        r"$y\ ({\rm pkpc})$"
    )

    ax.set_title(
        "CG10 dust-particle counts"
    )

    cb = fig.colorbar(
        im,
        ax=ax
    )

    cb.set_label(
        "PartType6 particles / pixel"
    )

    fig.tight_layout()

    fig.savefig(
        a.prefix + "_counts.png",
        dpi=250
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Radial profile
    # ------------------------------------------------------------

    rc, radial_sigma, radial_n = radial_profile(
        xall,
        yall,
        mall,
        rmax=rlarge,
        dr=a.radial_dr
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.4)
    )

    good = (
        radial_sigma > 0
    )

    ax.semilogy(
        rc[good],
        radial_sigma[good],
        marker="o"
    )

    ax.set_xlabel(
        r"$R\ ({\rm pkpc})$"
    )

    ax.set_ylabel(
        r"$\Sigma_{\rm dust}$ "
        r"(code mass / pkpc$^2$)"
    )

    ax.set_title(
        rf"CG10 radial dust profile: "
        rf"$|z|<{a.zmax:g}$ pkpc"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        a.prefix + "_radial_profile.png",
        dpi=250
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Aperture spectra
    # ------------------------------------------------------------

    results = {}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    axes = axes.ravel()

    for ia, radius in enumerate(
        a.map_radii
    ):

        q = (
            rall <= radius
        )

        x = xall[q]
        y = yall[q]
        m = mall[q]

        sigma, mmap, cmap, mask, _ = make_map(
            x,
            y,
            m,
            radius,
            a.pixel_size
        )

        real = power_spectrum_2d(
            sigma,
            mask,
            a.pixel_size,
            apodize_pix=a.apodize_pix,
            nbins=a.nbins
        )

        rand = random_control(
            x,
            y,
            m,
            radius,
            a.pixel_size,
            a.apodize_pix,
            a.nbins,
            a.nrandom,
            a.seed + ia
        )

        occupied = np.count_nonzero(
            (cmap > 0) & mask
        )

        npix = np.count_nonzero(mask)

        print()
        print(
            f"R < {radius:g} pkpc"
        )
        print("-" * 40)

        print(
            f"N dust particles       : "
            f"{len(x):,}"
        )

        print(
            f"Dust mass              : "
            f"{m.sum():.8e}"
        )

        print(
            f"FFT pixels             : "
            f"{npix:,}"
        )

        print(
            f"Occupied pixels        : "
            f"{occupied:,}"
        )

        print(
            f"Occupied fraction      : "
            f"{occupied/npix:.4f}"
        )

        print(
            f"kfund                   : "
            f"{real['kfund']:.4f}"
        )

        print(
            f"kNyq                    : "
            f"{real['kny']:.4f}"
        )

        # Real/random ratio on common bins.
        ratio = (
            real["k2P"]
            / rand["median"]
        )

        results[radius] = {
            "real": real,
            "random": rand,
            "ratio": ratio,
            "N": len(x),
            "mass": m.sum()
        }

        ax = axes[ia]

        good = (
            (real["k2P"] > 0)
            & np.isfinite(real["k2P"])
        )

        ax.loglog(
            real["k"][good],
            real["k2P"][good],
            marker="o",
            ms=3.5,
            lw=1.5,
            label="CG10"
        )

        rg = (
            (rand["median"] > 0)
            & np.isfinite(rand["median"])
        )

        ax.loglog(
            rand["k"][rg],
            rand["median"][rg],
            lw=1.5,
            label="azimuth randomized"
        )

        ax.fill_between(
            rand["k"][rg],
            rand["p16"][rg],
            rand["p84"][rg],
            alpha=0.2
        )

        ax.set_title(
            rf"$R<{radius:g}$ pkpc "
            rf"($N={len(x)}$)"
        )

        ax.set_xlabel(
            r"$k\ ({\rm kpc^{-1}})$"
        )

        ax.set_ylabel(
            r"$k^2P_{\delta_{\rm dust}}(k)$"
        )

        ax.grid(
            alpha=0.18,
            which="both"
        )

        ax.legend(
            fontsize=8
        )

    # Hide unused panels if fewer than four apertures.
    for j in range(
        len(a.map_radii),
        len(axes)
    ):
        axes[j].axis("off")

    fig.suptitle(
        "CG10 dust structure vs finite-particle control",
        fontsize=15
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    fig.savefig(
        a.prefix + "_power_random_control.png",
        dpi=250
    )

    fig.savefig(
        a.prefix + "_power_random_control.pdf"
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Real/random ratio
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    for radius in a.map_radii:

        r = results[radius]

        k = r["real"]["k"]
        ratio = r["ratio"]

        good = (
            (ratio > 0)
            & np.isfinite(ratio)
        )

        ax.semilogx(
            k[good],
            ratio[good],
            marker="o",
            ms=3.5,
            label=rf"$R<{radius:g}$ pkpc"
        )

    ax.axhline(
        1.0,
        ls="--",
        alpha=0.6
    )

    ax.set_xlabel(
        r"$k\ ({\rm kpc^{-1}})$"
    )

    ax.set_ylabel(
        r"$P_{\rm real}/P_{\rm randomized}$"
    )

    ax.set_title(
        "CG10 excess dust structure above randomized control"
    )

    ax.grid(
        alpha=0.2,
        which="both"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        a.prefix + "_real_random_ratio.png",
        dpi=250
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Save numerical output
    # ------------------------------------------------------------

    out = {
        "radial_R_pkpc": rc,
        "radial_sigma_dust": radial_sigma,
        "radial_N": radial_n,
        "map_sigma_dust": sigma_large,
        "map_dust_count": count_large,
        "map_pixel_size_pkpc": np.array(
            a.pixel_size
        ),
        "zmax_pkpc": np.array(
            a.zmax
        )
    }

    for radius in a.map_radii:

        tag = (
            f"R{radius:g}".replace(".", "p")
        )

        r = results[radius]

        out[f"k_{tag}"] = (
            r["real"]["k"]
        )

        out[f"k2P_real_{tag}"] = (
            r["real"]["k2P"]
        )

        out[f"k2P_random_median_{tag}"] = (
            r["random"]["median"]
        )

        out[f"k2P_random_p16_{tag}"] = (
            r["random"]["p16"]
        )

        out[f"k2P_random_p84_{tag}"] = (
            r["random"]["p84"]
        )

        out[f"ratio_{tag}"] = (
            r["ratio"]
        )

        out[f"N_{tag}"] = np.array(
            r["N"]
        )

        out[f"mass_{tag}"] = np.array(
            r["mass"]
        )

    np.savez_compressed(
        a.prefix + ".npz",
        **out
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Saved {a.prefix}.npz and plots."
    )


if __name__ == "__main__":
    main()
