#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix


# ================================================================
# Zhang & Green map query
# Adapted from their example_query.py
# ================================================================

def query_av_diff(fname, sample_coord, nside="combined"):

    with h5py.File(fname, "r") as f:
        distance_bins = f["distance_bins"][:]

        map_nside = f[f"map_{nside}/nside"][...]
        hp_order = f[f"map_{nside}/hp_order"][...].astype("str")

        AV_map_diff = f[f"map_{nside}/AV_map_diff"][:]

        if nside == "combined":
            nside_diff = f[f"map_{nside}/nside_map_diff"][:]
        else:
            nside_diff = None

    distance_sample = sample_coord.distance.kpc

    distance_idx = np.searchsorted(
        distance_bins,
        distance_sample
    ) - 1

    # Explicitly require points to lie inside the tabulated
    # distance interval.
    valid = (
        (distance_idx >= 0)
        & (distance_idx < len(distance_bins) - 1)
    )

    hp = HEALPix(
        nside=int(map_nside),
        order=str(hp_order)
    )

    healpix_idx = hp.lonlat_to_healpix(
        sample_coord.l,
        sample_coord.b
    )

    av = np.full(sample_coord.shape[0], np.nan)
    ns = np.full(sample_coord.shape[0], np.nan)

    if np.any(valid):

        av[valid] = AV_map_diff[
            distance_idx[valid],
            healpix_idx[valid]
        ]

        if nside_diff is not None:
            ns[valid] = nside_diff[
                distance_idx[valid],
                healpix_idx[valid]
            ]
        else:
            ns[valid] = int(map_nside)

    return av, ns


# ================================================================
# Query one z slice
# ================================================================

def query_slice(fname, X, Y, z, nside="combined"):

    x = X.ravel()
    y = Y.ravel()
    zz = np.full_like(x, z)

    coord = SkyCoord(
        u=x * u.kpc,
        v=y * u.kpc,
        w=zz * u.kpc,
        frame="galactic",
        representation_type="cartesian"
    )

    # Preserve the transformation used in Zhang & Green's example.
    coord = coord.icrs.galactic

    av, ns = query_av_diff(
        fname,
        coord,
        nside=nside
    )

    return (
        av.reshape(X.shape),
        ns.reshape(X.shape)
    )


# ================================================================
# Construct deterministic face-on maps
# ================================================================

def build_faceon_maps(
    fname,
    half_size=2.5,
    pixel_size=0.05,
    zmax=0.4,
    dz=0.05,
    nside="combined"
):

    edges = np.arange(
        -half_size,
        half_size + pixel_size * 0.5,
        pixel_size
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    X, Y = np.meshgrid(
        centers,
        centers,
        indexing="xy"
    )

    # Use cell-centered z samples.
    z_edges = np.arange(
        -zmax,
        zmax + dz * 0.5,
        dz
    )

    z_centers = 0.5 * (
        z_edges[:-1] + z_edges[1:]
    )

    print()
    print("Building Zhang & Green Cartesian map")
    print("------------------------------------")
    print(f"x/y half-size : {half_size:.3f} kpc")
    print(f"pixel size    : {pixel_size:.4f} kpc")
    print(f"map shape     : {X.shape}")
    print(f"z range       : {-zmax:.3f} to {zmax:.3f} kpc")
    print(f"dz            : {dz:.4f} kpc")
    print(f"z slices      : {len(z_centers)}")
    print()

    sum_av = np.zeros_like(X, dtype=float)
    count_av = np.zeros_like(X, dtype=float)

    # Integrated extinction-density proxy.
    sigma_av = np.zeros_like(X, dtype=float)

    # Track effective angular resolution.
    nside_min = np.full_like(X, np.inf, dtype=float)
    nside_max = np.zeros_like(X, dtype=float)

    # Number of valid z samples.
    valid_z_count = np.zeros_like(X, dtype=int)

    for iz, z in enumerate(z_centers):

        print(
            f"\rQuerying z slice "
            f"{iz + 1:3d}/{len(z_centers):3d} "
            f"(z={z:+.3f} kpc)",
            end="",
            flush=True
        )

        av, ns = query_slice(
            fname,
            X,
            Y,
            z,
            nside=nside
        )

        good = np.isfinite(av)

        sum_av[good] += av[good]
        count_av[good] += 1

        # AV_diff is mag/kpc.
        # Multiplication by dz produces mag.
        sigma_av[good] += av[good] * dz

        valid_z_count[good] += 1

        good_ns = good & np.isfinite(ns)

        nside_min[good_ns] = np.minimum(
            nside_min[good_ns],
            ns[good_ns]
        )

        nside_max[good_ns] = np.maximum(
            nside_max[good_ns],
            ns[good_ns]
        )

    print("\n")

    mean_av = np.full_like(X, np.nan)

    good = count_av > 0
    mean_av[good] = sum_av[good] / count_av[good]

    nside_min[~np.isfinite(nside_min)] = np.nan
    nside_max[nside_max == 0] = np.nan

    # Require complete z coverage for the default FFT map.
    complete = valid_z_count == len(z_centers)

    # Circular footprint, analogous to the example.
    R = np.sqrt(X**2 + Y**2)
    circular = R <= half_size

    mask = complete & circular & np.isfinite(sigma_av)

    print("Map statistics")
    print("--------------")
    print(f"total pixels       : {mask.size}")
    print(f"valid FFT pixels   : {np.count_nonzero(mask)}")
    print(f"valid fraction     : {np.mean(mask):.4f}")

    if np.any(mask):
        print(
            f"mean dAV/dd        : "
            f"{np.nanmean(mean_av[mask]):.6g} mag/kpc"
        )
        print(
            f"mean Sigma_AV      : "
            f"{np.nanmean(sigma_av[mask]):.6g} mag"
        )

    return {
        "x": centers,
        "y": centers,
        "X": X,
        "Y": Y,
        "mean_av": mean_av,
        "sigma_av": sigma_av,
        "nside_min": nside_min,
        "nside_max": nside_max,
        "valid_z_count": valid_z_count,
        "mask": mask,
        "pixel_size": pixel_size,
        "dz": dz,
        "zmax": zmax,
        "half_size": half_size,
    }


# ================================================================
# Apodization
# ================================================================

def cosine_apodization(mask, width_pix=4):

    """
    Approximate cosine taper toward the edge of the valid mask.

    Requires scipy for the distance transform.
    """

    from scipy.ndimage import distance_transform_edt

    mask = mask.astype(bool)

    if width_pix <= 0:
        return mask.astype(float)

    d = distance_transform_edt(mask)

    window = np.ones_like(d, dtype=float)

    edge = mask & (d < width_pix)

    window[edge] = 0.5 * (
        1.0
        - np.cos(
            np.pi * d[edge] / width_pix
        )
    )

    window[~mask] = 0.0

    return window


# ================================================================
# 2-D power spectrum
# ================================================================

def power_spectrum_2d(
    image,
    mask,
    pixel_size,
    apodize_pix=4,
    nbins=35,
    fractional=True
):

    valid = mask & np.isfinite(image)

    if np.count_nonzero(valid) < 10:
        raise RuntimeError("Too few valid pixels for FFT.")

    mean_value = np.mean(image[valid])

    field = np.zeros_like(image, dtype=float)

    if fractional:
        if mean_value == 0:
            raise RuntimeError(
                "Cannot construct fractional fluctuations: mean=0."
            )

        field[valid] = (
            image[valid] / mean_value - 1.0
        )

    else:
        field[valid] = (
            image[valid] - mean_value
        )

    window = cosine_apodization(
        valid,
        width_pix=apodize_pix
    )

    field *= window

    ny, nx = field.shape

    # Continuous-transform-like normalization:
    #
    # F(k) ~= dx^2 * FFT(field)
    #
    # P2D ~= |F|^2 / area
    #
    # This gives units of kpc^2 for a dimensionless field.
    fft = np.fft.fft2(field) * pixel_size**2
    fft = np.fft.fftshift(fft)

    effective_area = (
        np.sum(window**2) * pixel_size**2
    )

    power2d = np.abs(fft)**2 / effective_area

    kx = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(nx, d=pixel_size)
    )

    ky = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=pixel_size)
    )

    KX, KY = np.meshgrid(kx, ky)

    K = np.sqrt(KX**2 + KY**2)

    # Fundamental mode based on full map side length.
    Lx = nx * pixel_size
    Ly = ny * pixel_size

    kfund = 2.0 * np.pi / max(Lx, Ly)
    kny = np.pi / pixel_size

    good = (
        np.isfinite(power2d)
        & (K > 0)
        & (K >= kfund)
        & (K <= kny)
    )

    bins = np.geomspace(
        kfund,
        kny,
        nbins + 1
    )

    kval = []
    pval = []
    nmodes = []

    for lo, hi in zip(bins[:-1], bins[1:]):

        m = good & (K >= lo) & (K < hi)

        n = np.count_nonzero(m)

        if n == 0:
            continue

        # Geometric mean is natural for log-k bins.
        kval.append(
            np.exp(np.mean(np.log(K[m])))
        )

        pval.append(
            np.mean(power2d[m])
        )

        nmodes.append(n)

    return {
        "k": np.asarray(kval),
        "P": np.asarray(pval),
        "k2P": np.asarray(kval)**2 * np.asarray(pval),
        "nmodes": np.asarray(nmodes),
        "power2d": power2d,
        "K": K,
        "window": window,
        "mean": mean_value,
        "kfund": kfund,
        "kny": kny,
    }


# ================================================================
# Diagnostics
# ================================================================

def plot_maps(data, prefix):

    extent = [
        data["x"][0] - data["pixel_size"]/2,
        data["x"][-1] + data["pixel_size"]/2,
        data["y"][0] - data["pixel_size"]/2,
        data["y"][-1] + data["pixel_size"]/2,
    ]

    # ------------------------------------------------------------
    # Mean differential extinction density
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        data["mean_av"],
        origin="lower",
        extent=extent,
        cmap="inferno"
    )

    ax.set_xlabel(r"$x\ ({\rm kpc})$")
    ax.set_ylabel(r"$y\ ({\rm kpc})$")
    ax.set_title(
        r"Zhang & Green: mean $dA_V/ds$"
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$dA_V/ds\ ({\rm mag\,kpc^{-1}})$")

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_mean_dAVds.png",
        dpi=220
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Integrated extinction column
    # ------------------------------------------------------------

    sigma_plot = np.where(
        data["mask"],
        data["sigma_av"],
        np.nan
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        sigma_plot,
        origin="lower",
        extent=extent,
        cmap="inferno"
    )

    ax.set_xlabel(r"$x\ ({\rm kpc})$")
    ax.set_ylabel(r"$y\ ({\rm kpc})$")
    ax.set_title(
        r"Zhang & Green: projected extinction column"
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\Sigma_{A_V}\ ({\rm mag})$")

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_SigmaAV.png",
        dpi=220
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Minimum NSIDE encountered through column
    # ------------------------------------------------------------

    ns_plot = np.where(
        data["mask"],
        data["nside_min"],
        np.nan
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        ns_plot,
        origin="lower",
        extent=extent,
        cmap="viridis"
    )

    ax.set_xlabel(r"$x\ ({\rm kpc})$")
    ax.set_ylabel(r"$y\ ({\rm kpc})$")
    ax.set_title(
        "Minimum Zhang & Green NSIDE through column"
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("NSIDE")

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_nside.png",
        dpi=220
    )
    plt.close(fig)


def plot_power(spec, prefix, fractional=True):

    ylabel = (
        r"$P_{\Sigma_{A_V}}(k)\ ({\rm kpc^2})$"
        if fractional
        else r"$P_{\Sigma_{A_V}}(k)$"
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.loglog(
        spec["k"],
        spec["P"],
        marker="o",
        ms=4
    )

    ax.axvline(
        spec["kfund"],
        ls="--",
        alpha=0.5,
        label="fundamental"
    )

    ax.axvline(
        spec["kny"],
        ls=":",
        alpha=0.5,
        label="grid Nyquist"
    )

    ax.set_xlabel(r"$k\ ({\rm kpc^{-1}})$")
    ax.set_ylabel(ylabel)

    ax.set_title(
        "Zhang & Green extinction-column power spectrum"
    )

    ax.grid(alpha=0.2, which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_Pk.png",
        dpi=220
    )
    plt.close(fig)

    # k^2 P(k)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.loglog(
        spec["k"],
        spec["k2P"],
        marker="o",
        ms=4
    )

    ax.axvline(
        spec["kfund"],
        ls="--",
        alpha=0.5
    )

    ax.axvline(
        spec["kny"],
        ls=":",
        alpha=0.5
    )

    ax.set_xlabel(r"$k\ ({\rm kpc^{-1}})$")
    ax.set_ylabel(
        r"$k^2P_{\Sigma_{A_V}}(k)$"
    )

    ax.set_title(
        r"Zhang & Green: $k^2P(k)$"
    )

    ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(
        f"{prefix}_k2Pk.png",
        dpi=220
    )
    plt.close(fig)


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "map_file",
        help="Rv_map_new.h5"
    )

    parser.add_argument(
        "--half-size",
        type=float,
        default=2.5,
        help="Half-width of Cartesian map in kpc"
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.05,
        help="Cartesian pixel size in kpc"
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=0.4,
        help="Vertical half-thickness in kpc"
    )

    parser.add_argument(
        "--dz",
        type=float,
        default=0.05,
        help="Vertical sampling interval in kpc"
    )

    parser.add_argument(
        "--nside",
        default="combined"
    )

    parser.add_argument(
        "--apodize-pix",
        type=int,
        default=4
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=35
    )

    parser.add_argument(
        "--absolute",
        action="store_true",
        help=(
            "Use mean-subtracted absolute field instead "
            "of fractional Sigma_AV/<Sigma_AV>-1"
        )
    )

    parser.add_argument(
        "--prefix",
        default="zhang_green_dust"
    )

    args = parser.parse_args()

    data = build_faceon_maps(
        args.map_file,
        half_size=args.half_size,
        pixel_size=args.pixel_size,
        zmax=args.zmax,
        dz=args.dz,
        nside=args.nside
    )

    plot_maps(
        data,
        args.prefix
    )

    spec = power_spectrum_2d(
        data["sigma_av"],
        data["mask"],
        data["pixel_size"],
        apodize_pix=args.apodize_pix,
        nbins=args.nbins,
        fractional=not args.absolute
    )

    print()
    print("Power-spectrum scales")
    print("---------------------")
    print(
        f"k_fundamental = "
        f"{spec['kfund']:.4f} kpc^-1"
    )
    print(
        f"k_Nyquist(grid) = "
        f"{spec['kny']:.4f} kpc^-1"
    )
    print(
        "NOTE: grid Nyquist is NOT necessarily the "
        "observational resolution limit."
    )

    plot_power(
        spec,
        args.prefix,
        fractional=not args.absolute
    )

    np.savez_compressed(
        f"{args.prefix}.npz",

        x=data["x"],
        y=data["y"],

        mean_av=data["mean_av"],
        sigma_av=data["sigma_av"],

        mask=data["mask"],

        nside_min=data["nside_min"],
        nside_max=data["nside_max"],

        valid_z_count=data["valid_z_count"],

        pixel_size=data["pixel_size"],
        dz=data["dz"],
        zmax=data["zmax"],
        half_size=data["half_size"],

        k=spec["k"],
        P=spec["P"],
        k2P=spec["k2P"],
        nmodes=spec["nmodes"],

        kfund=spec["kfund"],
        kny=spec["kny"],

        fractional=(not args.absolute)
    )

    print()
    print(
        f"Saved {args.prefix}.npz and diagnostic plots."
    )


if __name__ == "__main__":
    main()
