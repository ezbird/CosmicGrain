#!/usr/bin/env python3

"""
cosmicgrain_gaia_matched_dust_power.py

Construct a face-on CosmicGrain dust surface-density map directly from
PartType6 and measure its 2-D power spectrum using the same analysis
conventions as the Zhang & Green extinction-column script.

The primary comparison is

    delta_dust = Sigma_dust / <Sigma_dust> - 1

versus

    delta_AV = Sigma_AV / <Sigma_AV> - 1

For spatially constant V-band opacity, these are identical fractional
fields, so this is the fixed-opacity synthetic-extinction comparison.

Default geometry matches the Zhang & Green analysis:
    x,y = +/- 2.5 pkpc
    |z| <= 0.4 pkpc
    50 pc native pixels
    circular R <= 2.5 pkpc FFT footprint
    4-pixel cosine apodization
    35 logarithmic k bins

Also measures rebinned 100, 250, and 500 pc maps as a convergence /
particle-sampling diagnostic.

Optionally supply the NPZ produced by the Zhang & Green script with
--gaia-npz to make a direct MW-vs-CG10 comparison.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt


# ================================================================
# Small utilities
# ================================================================

def read_vector_file(fname, expected=None):
    arr = np.loadtxt(fname, dtype=float)
    arr = np.asarray(arr, dtype=float)

    if expected is not None:
        arr = arr.reshape(expected)

    return arr


def periodic_delta(pos, center, boxsize):
    d = pos - center[None, :]
    d -= boxsize * np.rint(d / boxsize)
    return d


def block_sum(a, factor):
    ny, nx = a.shape

    if ny % factor or nx % factor:
        raise ValueError(
            f"Array shape {a.shape} is not divisible by "
            f"rebin factor {factor}"
        )

    return a.reshape(
        ny // factor,
        factor,
        nx // factor,
        factor
    ).sum(axis=(1, 3))


def block_sum_int(a, factor):
    return block_sum(a.astype(np.int64), factor).astype(np.int64)


# ================================================================
# Zhang & Green-matched apodization
# ================================================================

def cosine_apodization(mask, width_pix=4):

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
# Zhang & Green-matched 2-D power spectrum
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

    # Same continuous-transform-like normalization as the
    # Zhang & Green script:
    #
    # F(k) ~= dx^2 FFT(field)
    # P2D  ~= |F|^2 / effective_area

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

        kval.append(
            np.exp(np.mean(np.log(K[m])))
        )

        pval.append(
            np.mean(power2d[m])
        )

        nmodes.append(n)

    kval = np.asarray(kval)
    pval = np.asarray(pval)
    nmodes = np.asarray(nmodes)

    return {
        "k": kval,
        "P": pval,
        "k2P": kval**2 * pval,
        "nmodes": nmodes,
        "power2d": power2d,
        "K": K,
        "window": window,
        "field": field,
        "mean": mean_value,
        "kfund": kfund,
        "kny": kny,
        "effective_area": effective_area,
    }


# ================================================================
# Read PartType6 and construct face-on dust map
# ================================================================

def build_dust_map(
    snapshot_pattern,
    center,
    rotation,
    half_size=2.5,
    zmax=0.4,
    pixel_size=0.05,
):

    files = sorted(glob.glob(snapshot_pattern))

    if not files:
        raise FileNotFoundError(
            f"No snapshot files match:\n{snapshot_pattern}"
        )

    # ------------------------------------------------------------
    # Snapshot metadata
    # ------------------------------------------------------------

    with h5py.File(files[0], "r") as f:

        header = f["Header"].attrs

        scale_factor = float(
            header.get("Time", 1.0)
        )

        boxsize = float(
            header["BoxSize"]
        )

        hubble = header.get(
            "HubbleParam",
            None
        )

        if hubble is None and "Parameters" in f:
            hubble = f["Parameters"].attrs.get(
                "HubbleParam",
                None
            )

        if hubble is None:
            raise RuntimeError(
                "Could not determine HubbleParam."
            )

        hubble = float(hubble)

    # Coordinates are converted from ckpc/h to physical pkpc
    # in the same way as the existing CG analysis scripts.

    conv_pos = scale_factor / hubble

    # ------------------------------------------------------------
    # Map geometry
    # ------------------------------------------------------------

    nside_float = 2.0 * half_size / pixel_size
    nside = int(round(nside_float))

    if not np.isclose(nside_float, nside):
        raise ValueError(
            "2*half_size must be an integer multiple "
            "of pixel_size."
        )

    edges = np.linspace(
        -half_size,
        half_size,
        nside + 1
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    X, Y = np.meshgrid(
        centers,
        centers,
        indexing="xy"
    )

    dust_mass = np.zeros(
        (nside, nside),
        dtype=float
    )

    dust_count = np.zeros(
        (nside, nside),
        dtype=np.int64
    )

    total_dust_particles = 0
    slab_particles = 0
    map_particles = 0

    total_map_mass_code = 0.0

    # ------------------------------------------------------------
    # Read all snapshot chunks
    # ------------------------------------------------------------

    for ifile, fname in enumerate(files, 1):

        print(
            f"\rReading snapshot file "
            f"{ifile}/{len(files)}",
            end="",
            flush=True
        )

        with h5py.File(fname, "r") as f:

            if "PartType6" not in f:
                continue

            g = f["PartType6"]

            pos = np.asarray(
                g["Coordinates"],
                dtype=float
            )

            if "Masses" in g:

                mass = np.asarray(
                    g["Masses"],
                    dtype=float
                )

            else:

                mt = np.asarray(
                    f["Header"].attrs["MassTable"],
                    dtype=float
                )

                if len(mt) <= 6 or mt[6] <= 0:
                    raise RuntimeError(
                        "PartType6 has neither Masses "
                        "nor a valid MassTable[6]."
                    )

                mass = np.full(
                    len(pos),
                    mt[6],
                    dtype=float
                )

        total_dust_particles += len(pos)

        # Periodic displacement in simulation coordinates.
        d = periodic_delta(
            pos,
            center,
            boxsize
        )

        # Convert to physical pkpc.
        d *= conv_pos

        # Existing audited rotation matrix maps simulation-frame
        # displacement vectors into face-on coordinates.
        u = d @ rotation.T

        # First restrict to vertical slab.
        slab = (
            np.abs(u[:, 2]) <= zmax
        )

        slab_particles += np.count_nonzero(slab)

        if not np.any(slab):
            continue

        x = u[slab, 0]
        y = u[slab, 1]
        m = mass[slab]

        # Square map selection.
        inside = (
            (x >= -half_size)
            & (x < half_size)
            & (y >= -half_size)
            & (y < half_size)
        )

        if not np.any(inside):
            continue

        x = x[inside]
        y = y[inside]
        m = m[inside]

        map_particles += len(x)
        total_map_mass_code += np.sum(m)

        # histogram2d uses [y,x] so output orientation matches imshow.
        dust_mass += np.histogram2d(
            y,
            x,
            bins=[edges, edges],
            weights=m
        )[0]

        dust_count += np.histogram2d(
            y,
            x,
            bins=[edges, edges]
        )[0].astype(np.int64)

    print()

    # ------------------------------------------------------------
    # Surface density
    # ------------------------------------------------------------

    sigma_dust = (
        dust_mass / pixel_size**2
    )

    R = np.sqrt(X**2 + Y**2)

    circular_mask = (
        R <= half_size
    )

    return {
        "x": centers,
        "y": centers,
        "X": X,
        "Y": Y,
        "dust_mass_code": dust_mass,
        "dust_count": dust_count,
        "sigma_dust": sigma_dust,
        "mask": circular_mask,
        "pixel_size": pixel_size,
        "half_size": half_size,
        "zmax": zmax,
        "scale_factor": scale_factor,
        "hubble": hubble,
        "boxsize": boxsize,
        "total_dust_particles": total_dust_particles,
        "slab_particles": slab_particles,
        "map_particles": map_particles,
        "total_map_mass_code": total_map_mass_code,
    }


# ================================================================
# Diagnostics
# ================================================================

def print_sampling_stats(
    dust_mass,
    dust_count,
    mask,
    pixel_size,
):

    m = mask.astype(bool)

    counts = dust_count[m]

    occupied = counts > 0

    print()
    print(
        f"{pixel_size*1000:6.0f} pc sampling"
    )
    print("-" * 42)

    print(
        f"map shape                : "
        f"{dust_mass.shape}"
    )

    print(
        f"circular FFT pixels      : "
        f"{np.count_nonzero(m):,}"
    )

    print(
        f"occupied FFT pixels      : "
        f"{np.count_nonzero(occupied):,}"
    )

    print(
        f"occupied fraction        : "
        f"{np.mean(occupied):.4f}"
    )

    print(
        f"mean grains / FFT pixel  : "
        f"{np.mean(counts):.4f}"
    )

    print(
        f"median grains / pixel    : "
        f"{np.median(counts):.1f}"
    )

    if np.any(occupied):

        q = np.percentile(
            counts[occupied],
            [16, 50, 84, 95, 99]
        )

        print(
            "counts in occupied pixels: "
            f"p16={q[0]:.1f}, "
            f"p50={q[1]:.1f}, "
            f"p84={q[2]:.1f}, "
            f"p95={q[3]:.1f}, "
            f"p99={q[4]:.1f}"
        )

    print(
        f"total dust mass (code)   : "
        f"{np.sum(dust_mass[m]):.8e}"
    )


# ================================================================
# Plot maps
# ================================================================

def plot_native_maps(data, spec, prefix):

    extent = [
        -data["half_size"],
        data["half_size"],
        -data["half_size"],
        data["half_size"],
    ]

    mask = data["mask"]

    sigma_plot = np.where(
        mask,
        data["sigma_dust"],
        np.nan
    )

    count_plot = np.where(
        mask,
        data["dust_count"],
        np.nan
    )

    delta_plot = np.where(
        mask,
        data["sigma_dust"] / spec["mean"] - 1.0,
        np.nan
    )

    # Dust surface density
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        sigma_plot,
        origin="lower",
        extent=extent,
        cmap="inferno"
    )

    ax.set_xlabel(r"$x\ ({\rm pkpc})$")
    ax.set_ylabel(r"$y\ ({\rm pkpc})$")
    ax.set_title(
        "CG10: dust surface density"
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(
        r"$\Sigma_{\rm dust}$ (code mass / pkpc$^2$)"
    )

    fig.tight_layout()

    fig.savefig(
        prefix + "_SigmaDust.png",
        dpi=220
    )

    plt.close(fig)

    # Fractional field
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        delta_plot,
        origin="lower",
        extent=extent,
        cmap="RdBu_r"
    )

    ax.set_xlabel(r"$x\ ({\rm pkpc})$")
    ax.set_ylabel(r"$y\ ({\rm pkpc})$")
    ax.set_title(
        r"CG10: $\delta_{\rm dust}$"
    )

    cb = fig.colorbar(im, ax=ax)

    cb.set_label(
        r"$\Sigma_{\rm dust}/"
        r"\langle\Sigma_{\rm dust}\rangle-1$"
    )

    fig.tight_layout()

    fig.savefig(
        prefix + "_deltaDust.png",
        dpi=220
    )

    plt.close(fig)

    # Particle counts
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        count_plot,
        origin="lower",
        extent=extent,
        cmap="viridis"
    )

    ax.set_xlabel(r"$x\ ({\rm pkpc})$")
    ax.set_ylabel(r"$y\ ({\rm pkpc})$")
    ax.set_title(
        "CG10: dust-particle sampling"
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(
        "PartType6 particles / pixel"
    )

    fig.tight_layout()

    fig.savefig(
        prefix + "_dustCount.png",
        dpi=220
    )

    plt.close(fig)


# ================================================================
# Plot convergence
# ================================================================

def plot_convergence(
    spectra,
    prefix,
    min_modes=1
):

    fig, ax = plt.subplots(
        figsize=(7.5, 5.8)
    )

    for item in spectra:

        dx = item["pixel_size"]
        spec = item["spec"]

        k = spec["k"]
        y = spec["k2P"]
        n = spec["nmodes"]

        good = (
            (k > 0)
            & (y > 0)
            & np.isfinite(y)
            & (n >= min_modes)
        )

        label = (
            f"{dx*1000:.0f} pc"
            if dx < 1.0
            else f"{dx:g} kpc"
        )

        ax.loglog(
            k[good],
            y[good],
            marker="o",
            ms=3.5,
            lw=1.5,
            label=label
        )

    ax.set_xlabel(
        r"$k\ ({\rm kpc^{-1}})$"
    )

    ax.set_ylabel(
        r"$k^2P_{\delta_{\rm dust}}(k)$"
    )

    ax.set_title(
        "CG10: matched dust-column power spectrum"
    )

    ax.grid(
        alpha=0.2,
        which="both"
    )

    ax.legend(
        title="Pixel size"
    )

    fig.tight_layout()

    fig.savefig(
        prefix + "_convergence.png",
        dpi=240
    )

    fig.savefig(
        prefix + "_convergence.pdf"
    )

    plt.close(fig)


# ================================================================
# Gaia comparison
# ================================================================

def plot_gaia_comparison(
    gaia_npz,
    cg_spec,
    prefix,
    min_modes=1,
):

    g = np.load(
        gaia_npz,
        allow_pickle=True
    )

    required = [
        "k",
        "k2P",
        "nmodes",
        "kfund",
        "kny"
    ]

    for key in required:
        if key not in g:
            raise KeyError(
                f"Gaia NPZ lacks required key '{key}'"
            )

    kg = np.asarray(
        g["k"],
        dtype=float
    )

    yg = np.asarray(
        g["k2P"],
        dtype=float
    )

    ng = np.asarray(
        g["nmodes"],
        dtype=int
    )

    kc = cg_spec["k"]
    yc = cg_spec["k2P"]
    nc = cg_spec["nmodes"]

    gg = (
        (kg > 0)
        & (yg > 0)
        & np.isfinite(yg)
        & (ng >= min_modes)
    )

    cc = (
        (kc > 0)
        & (yc > 0)
        & np.isfinite(yc)
        & (nc >= min_modes)
    )

    # ------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.6, 5.8)
    )

    ax.loglog(
        kg[gg],
        yg[gg],
        marker="o",
        ms=4,
        lw=1.6,
        label="Zhang & Green / Milky Way"
    )

    ax.loglog(
        kc[cc],
        yc[cc],
        marker="s",
        ms=4,
        lw=1.6,
        label="CosmicGrain CG10"
    )

    ax.axvline(
        float(np.asarray(g["kfund"]).ravel()[0]),
        ls="--",
        alpha=0.35
    )

    ax.axvline(
        float(np.asarray(g["kny"]).ravel()[0]),
        ls=":",
        alpha=0.35
    )

    ax.set_xlabel(
        r"$k\ ({\rm kpc^{-1}})$"
    )

    ax.set_ylabel(
        r"$k^2P_{\delta}(k)$"
    )

    ax.set_title(
        "Milky Way extinction vs CG10 dust structure"
    )

    ax.grid(
        alpha=0.2,
        which="both"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        prefix + "_Gaia_vs_CG10.png",
        dpi=250
    )

    fig.savefig(
        prefix + "_Gaia_vs_CG10.pdf"
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Ratio on Gaia k grid over common k range
    # ------------------------------------------------------------

    kgood = kg[gg]
    ygood = yg[gg]

    kcgood = kc[cc]
    ycgood = yc[cc]

    kmin = max(
        np.min(kgood),
        np.min(kcgood)
    )

    kmax = min(
        np.max(kgood),
        np.max(kcgood)
    )

    common = (
        (kgood >= kmin)
        & (kgood <= kmax)
    )

    k_ratio = kgood[common]

    cg_interp = np.interp(
        np.log(k_ratio),
        np.log(kcgood),
        np.log(ycgood)
    )

    cg_interp = np.exp(cg_interp)

    ratio = (
        cg_interp / ygood[common]
    )

    fig, ax = plt.subplots(
        figsize=(7.6, 4.8)
    )

    ax.semilogx(
        k_ratio,
        ratio,
        marker="o",
        ms=4
    )

    ax.axhline(
        1.0,
        ls="--",
        alpha=0.5
    )

    ax.set_xlabel(
        r"$k\ ({\rm kpc^{-1}})$"
    )

    ax.set_ylabel(
        r"$P_{\rm CG10}/P_{\rm MW}$"
    )

    ax.set_title(
        "CG10 / Milky Way fractional power"
    )

    ax.grid(
        alpha=0.2,
        which="both"
    )

    fig.tight_layout()

    fig.savefig(
        prefix + "_Gaia_CG10_ratio.png",
        dpi=240
    )

    plt.close(fig)

    return {
        "k_ratio": k_ratio,
        "ratio": ratio
    }


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snapshot",
        help=(
            "Snapshot glob, e.g. "
            "'../halo441_output_1024/snapdir_029/"
            "snapshot_029.*.hdf5'"
        )
    )

    parser.add_argument(
        "--center-file",
        required=True
    )

    parser.add_argument(
        "--rotation-file",
        required=True
    )

    parser.add_argument(
        "--gaia-npz",
        default=None,
        help=(
            "NPZ produced by the Zhang & Green "
            "power-spectrum script"
        )
    )

    parser.add_argument(
        "--half-size",
        type=float,
        default=2.5
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=0.4
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--convergence-pixels",
        nargs="+",
        type=float,
        default=[
            0.05,
            0.10,
            0.25,
            0.50
        ]
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
        "--min-modes",
        type=int,
        default=1
    )

    parser.add_argument(
        "--prefix",
        default="CG10_gaia_matched"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    center = read_vector_file(
        args.center_file,
        expected=(3,)
    )

    rotation = read_vector_file(
        args.rotation_file,
        expected=(3, 3)
    )

    print()
    print("=" * 72)
    print(
        "COSMICGRAIN / ZHANG-GREEN MATCHED "
        "DUST POWER SPECTRUM"
    )
    print("=" * 72)

    print()
    print("Center:")
    print(center)

    print()
    print("Rotation matrix:")
    print(rotation)

    print()
    print("Matched geometry")
    print("----------------")
    print(
        f"x/y range    : "
        f"{-args.half_size:.3f} to "
        f"{args.half_size:.3f} pkpc"
    )
    print(
        f"z range      : "
        f"{-args.zmax:.3f} to "
        f"{args.zmax:.3f} pkpc"
    )
    print(
        f"native pixel : "
        f"{args.pixel_size:.4f} pkpc "
        f"({args.pixel_size*1000:.0f} pc)"
    )
    print(
        f"apodization  : "
        f"{args.apodize_pix} pixels"
    )
    print(
        f"k bins       : "
        f"{args.nbins} logarithmic"
    )

    # ------------------------------------------------------------
    # Native 50-pc map
    # ------------------------------------------------------------

    data = build_dust_map(
        args.snapshot,
        center,
        rotation,
        half_size=args.half_size,
        zmax=args.zmax,
        pixel_size=args.pixel_size,
    )

    print()
    print("Snapshot / particle statistics")
    print("------------------------------")
    print(
        f"scale factor             : "
        f"{data['scale_factor']:.8f}"
    )
    print(
        f"HubbleParam              : "
        f"{data['hubble']:.8f}"
    )
    print(
        f"all PartType6 particles  : "
        f"{data['total_dust_particles']:,}"
    )
    print(
        f"particles in z slab      : "
        f"{data['slab_particles']:,}"
    )
    print(
        f"particles in square map  : "
        f"{data['map_particles']:,}"
    )

    # Native spectrum.
    native_spec = power_spectrum_2d(
        data["sigma_dust"],
        data["mask"],
        data["pixel_size"],
        apodize_pix=args.apodize_pix,
        nbins=args.nbins,
        fractional=True
    )

    print_sampling_stats(
        data["dust_mass_code"],
        data["dust_count"],
        data["mask"],
        data["pixel_size"]
    )

    print()
    print("Native Fourier scales")
    print("---------------------")
    print(
        f"k_fundamental = "
        f"{native_spec['kfund']:.6f} kpc^-1"
    )
    print(
        f"k_Nyquist     = "
        f"{native_spec['kny']:.6f} kpc^-1"
    )
    print(
        f"lambda_Nyquist= "
        f"{2*np.pi/native_spec['kny']:.6f} kpc"
    )

    plot_native_maps(
        data,
        native_spec,
        args.prefix
    )

    # ------------------------------------------------------------
    # Resolution / sampling convergence
    # ------------------------------------------------------------

    spectra = []

    native_dx = data["pixel_size"]

    for dx in args.convergence_pixels:

        ratio = dx / native_dx
        factor = int(round(ratio))

        if factor < 1 or not np.isclose(
            ratio,
            factor
        ):
            raise ValueError(
                f"Convergence pixel size {dx} "
                f"is not an integer multiple "
                f"of native pixel size {native_dx}."
            )

        if factor == 1:

            mass = data["dust_mass_code"].copy()
            count = data["dust_count"].copy()

        else:

            mass = block_sum(
                data["dust_mass_code"],
                factor
            )

            count = block_sum_int(
                data["dust_count"],
                factor
            )

        ny, nx = mass.shape

        x = (
            np.arange(nx)
            - (nx - 1) / 2.0
        ) * dx

        y = (
            np.arange(ny)
            - (ny - 1) / 2.0
        ) * dx

        X, Y = np.meshgrid(
            x,
            y,
            indexing="xy"
        )

        mask = (
            np.sqrt(X**2 + Y**2)
            <= args.half_size
        )

        sigma = (
            mass / dx**2
        )

        # Keep the taper at approximately the same physical width
        # as the 4-pixel / 50-pc Gaia taper.
        taper_physical = (
            args.apodize_pix
            * native_dx
        )

        taper_this = int(
            round(
                taper_physical / dx
            )
        )

        taper_this = max(
            1,
            taper_this
        )

        spec = power_spectrum_2d(
            sigma,
            mask,
            dx,
            apodize_pix=taper_this,
            nbins=args.nbins,
            fractional=True
        )

        print_sampling_stats(
            mass,
            count,
            mask,
            dx
        )

        print(
            f"  taper = {taper_this} pixels "
            f"({taper_this*dx:.3f} pkpc)"
        )
        print(
            f"  kNyq  = "
            f"{spec['kny']:.4f} kpc^-1"
        )

        spectra.append({
            "pixel_size": dx,
            "mass": mass,
            "count": count,
            "mask": mask,
            "spec": spec,
        })

    plot_convergence(
        spectra,
        args.prefix,
        min_modes=args.min_modes
    )

    # ------------------------------------------------------------
    # Optional direct Gaia comparison
    # ------------------------------------------------------------

    ratio_data = None

    if args.gaia_npz is not None:

        ratio_data = plot_gaia_comparison(
            args.gaia_npz,
            native_spec,
            args.prefix,
            min_modes=args.min_modes
        )

    # ------------------------------------------------------------
    # Save everything useful
    # ------------------------------------------------------------

    out = {
        "x": data["x"],
        "y": data["y"],
        "sigma_dust": data["sigma_dust"],
        "dust_mass_code": data["dust_mass_code"],
        "dust_count": data["dust_count"],
        "mask": data["mask"],
        "pixel_size": data["pixel_size"],
        "half_size": data["half_size"],
        "zmax": data["zmax"],

        "k": native_spec["k"],
        "P": native_spec["P"],
        "k2P": native_spec["k2P"],
        "nmodes": native_spec["nmodes"],
        "kfund": native_spec["kfund"],
        "kny": native_spec["kny"],
    }

    for item in spectra:

        dx = item["pixel_size"]
        tag = f"{round(dx*1000):.0f}pc"
        spec = item["spec"]

        out[f"k_{tag}"] = spec["k"]
        out[f"P_{tag}"] = spec["P"]
        out[f"k2P_{tag}"] = spec["k2P"]
        out[f"nmodes_{tag}"] = spec["nmodes"]
        out[f"kfund_{tag}"] = np.array(
            spec["kfund"]
        )
        out[f"kny_{tag}"] = np.array(
            spec["kny"]
        )

    if ratio_data is not None:
        out["comparison_k"] = (
            ratio_data["k_ratio"]
        )
        out["CG10_over_MW"] = (
            ratio_data["ratio"]
        )

    np.savez_compressed(
        args.prefix + ".npz",
        **out
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Saved {args.prefix}.npz "
        "and diagnostic plots."
    )


if __name__ == "__main__":
    main()
