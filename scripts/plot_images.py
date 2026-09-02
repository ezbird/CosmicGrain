#!/usr/bin/env python3
"""
plot_images.py
--------------------------------------------------------------------------
Visualizes a SKIRT FullInstrument FITS output cube (e.g. _faceon_total.fits).
The primary HDU is a wavelength-resolved image cube -- FITS header lists
dimensions as (NX, NY, NWAVE), which astropy reads back as a numpy array
of shape (NWAVE, NY, NX). The actual wavelength value for each of the
NWAVE planes lives in the "Z-axis coordinate values" table extension, not
assumed from the .ski wavelength grid -- read directly from the file so
this doesn't silently break if the grid ever changes.

Tested against a synthetic cube built to match the real header exactly
((500, 500, 400) FITS dims, 'Z-axis coordinate values' TableHDU with
400 rows / 1 column) -- confirmed the wavelength lookup and slicing are
correct before this was used on real data.

Usage:
    # Single wavelength slice (nearest match):
    python plot_images.py halo569_snap047_lrn_1e8_faceon_total.fits \
        --wavelength 2.0 --output image_2um.png

    # Montage across several representative wavelengths:
    python plot_images.py halo569_snap047_lrn_1e8_faceon_total.fits \
        --montage --output montage.png

    # Bolometric (wavelength-integrated) map:
    python plot_images.py halo569_snap047_lrn_1e8_faceon_total.fits \
        --bolometric --output bolometric.png
--------------------------------------------------------------------------
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def load_cube(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)  # shape: (nwave, ny, nx)
        bunit = hdul[0].header.get("BUNIT", "unknown unit")

        wl = None
        for hdu in hdul[1:]:
            if hdu.data is not None and len(hdu.data.dtype.names or []) >= 1:
                col_name = hdu.data.dtype.names[0]
                wl = np.array(hdu.data[col_name], dtype=float)
                break

        if wl is None:
            raise SystemExit(
                f"Couldn't find a wavelength table extension in {path}. "
                f"HDU list: {[h.name for h in hdul]}"
            )
        if len(wl) != data.shape[0]:
            raise SystemExit(
                f"Wavelength table has {len(wl)} entries but the cube has "
                f"{data.shape[0]} planes -- mismatch, structure may differ "
                f"from what this script expects."
            )

    return data, wl, bunit


def plot_slice(data, wl, bunit, target_wl, output):
    idx = np.argmin(np.abs(wl - target_wl))
    img = data[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.log10(np.clip(img, img[img > 0].min() if (img > 0).any() else 1e-30, None)),
                    origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, label=f"log10 flux ({bunit})")
    ax.set_title(f"Wavelength = {wl[idx]:.3g} micron (requested {target_wl})")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output} (used plane {idx}, wavelength {wl[idx]:.4g})")


def plot_montage(data, wl, bunit, output, n_panels=6):
    # Evenly spaced in log-wavelength across the full range, so the montage
    # naturally spans UV through far-IR regardless of the actual grid.
    target_wls = np.logspace(np.log10(wl.min()), np.log10(wl.max()), n_panels)
    indices = [np.argmin(np.abs(wl - t)) for t in target_wls]

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        img = data[idx]
        positive = img[img > 0]
        vmin = positive.min() if positive.size else 1e-30
        im = ax.imshow(np.log10(np.clip(img, vmin, None)), origin="lower", cmap="inferno")
        ax.set_title(f"{wl[idx]:.3g} micron")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"Flux ({bunit}) across wavelength, log scale")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output} ({len(indices)} panels at wavelengths: "
          f"{', '.join(f'{wl[i]:.3g}' for i in indices)} micron)")


def plot_bolometric(data, wl, bunit, output):
    # Trapezoidal integration over wavelength -- approximate bolometric map.
    # Cube is F_nu-like per the SED file's units; this is a rough integrated
    # view for morphology, not a precision bolometric luminosity map.
    integrated = np.trapezoid(data, x=wl, axis=0) if hasattr(np, "trapezoid") \
        else np.trapz(data, x=wl, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    positive = integrated[integrated > 0]
    vmin = positive.min() if positive.size else 1e-30
    im = ax.imshow(np.log10(np.clip(integrated, vmin, None)), origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, label=f"log10 integrated flux ({bunit} x micron)")
    ax.set_title("Wavelength-integrated (bolometric-ish) map")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fits_file", help="Path to a SKIRT FITS cube (e.g. _faceon_total.fits)")
    p.add_argument("--wavelength", type=float, default=None,
                   help="Wavelength (micron) for a single-slice image")
    p.add_argument("--montage", action="store_true",
                   help="Multi-panel montage across the wavelength range")
    p.add_argument("--bolometric", action="store_true",
                   help="Wavelength-integrated map")
    p.add_argument("--output", default="image_plot.png")
    args = p.parse_args()

    data, wl, bunit = load_cube(args.fits_file)
    print(f"Loaded cube: {data.shape[0]} wavelength planes, "
          f"{data.shape[1]}x{data.shape[2]} pixels, unit={bunit}, "
          f"wavelength range {wl.min():.3g}-{wl.max():.3g} micron")

    if args.montage:
        plot_montage(data, wl, bunit, args.output)
    elif args.bolometric:
        plot_bolometric(data, wl, bunit, args.output)
    elif args.wavelength is not None:
        plot_slice(data, wl, bunit, args.wavelength, args.output)
    else:
        raise SystemExit("Specify one of --wavelength, --montage, or --bolometric")


if __name__ == "__main__":
    main()
