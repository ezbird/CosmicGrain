#!/usr/bin/env python3
"""
plot_skirt_sed_v4.py
CosmicGrain / SKIRT SED plotting utility

VERSION: 2026-08-13-v4-clean-rebuild

Designed for SKIRT 9 FullInstrument *_sed.dat files with columns:
  1 wavelength [micron]
  2 total flux density
  3 transparent/intrinsic primary
  4 primary direct
  5 primary scattered
  6 secondary direct
  7 secondary scattered
  8 secondary transparent

Default presentation:
  - nu F_nu energy flux
  - pastel UV / Optical / NIR / MIR / FIR / Sub-mm bands
  - total face-on and optional edge-on SEDs
  - intrinsic stellar, attenuated stellar, dust-emission components
  - 9.7 and 18 micron silicate-feature guides
  - NO old BPASS 10 micron cutoff annotation
  - NO bottom-right explanatory note
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


VERSION = "2026-08-13-v4-clean-rebuild"
C_MICRON_S = 2.99792458e14  # speed of light [micron s^-1]


def load_sed(filename):
    """Read a SKIRT FullInstrument SED table."""
    filename = Path(filename)
    data = np.loadtxt(filename)

    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(
            f"{filename}: expected >=8 columns from a SKIRT FullInstrument SED; "
            f"found shape {data.shape}."
        )

    lam = data[:, 0]

    return {
        "lambda": lam,
        "total": data[:, 1],
        "intrinsic": data[:, 2],
        "primary_direct": data[:, 3],
        "primary_scattered": data[:, 4],
        "secondary_direct": data[:, 5],
        "secondary_scattered": data[:, 6],
        "secondary_transparent": data[:, 7],
        "attenuated_stellar": data[:, 3] + data[:, 4],
        "dust_emission": data[:, 5] + data[:, 6],
    }


def to_plot_quantity(lam_micron, fnu_jy, quantity):
    """
    Convert SKIRT F_nu in Jy to the selected plotted quantity.

    nufnu:
        nu F_nu in W m^-2
    fnu:
        F_nu in Jy
    """
    if quantity == "fnu":
        return np.asarray(fnu_jy, dtype=float)

    nu_hz = C_MICRON_S / np.asarray(lam_micron, dtype=float)
    return nu_hz * np.asarray(fnu_jy, dtype=float) * 1e-26


def clean_for_log(values, relative_floor=None):
    """Mask non-positive and optionally negligible values for log plotting."""
    y = np.array(values, dtype=float, copy=True)
    y[~np.isfinite(y)] = np.nan
    y[y <= 0] = np.nan

    if relative_floor is not None:
        y[y < relative_floor] = np.nan

    return y


def add_spectral_regions(ax):
    """Add subtle pastel wavelength-region backgrounds."""
    regions = [
        (0.10,   0.40,   "UV",      "#ece3f7"),
        (0.40,   0.75,   "Optical", "#fff1c9"),
        (0.75,   5.0,    "NIR",     "#fae2d7"),
        (5.0,    30.0,   "MIR",     "#f7dce5"),
        (30.0,   300.0,  "FIR",     "#dceff5"),
        (300.0,  1000.0, "Sub-mm",  "#e1eddc"),
    ]

    for left, right, label, color in regions:
        ax.axvspan(left, right, color=color, alpha=0.58, lw=0, zorder=0)

        # Geometric midpoint is visually centered on a log wavelength axis.
        xmid = np.sqrt(left * right)
        ax.text(
            xmid,
            0.985,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#4f4f4f",
            zorder=10,
        )


def add_silicate_guides(ax):
    """Mark the broad 9.7 and 18 micron silicate-feature locations."""
    features = [
        (9.7, r"9.7 $\mu$m silicate"),
        (18.0, r"18 $\mu$m silicate"),
    ]

    # Stagger labels slightly to keep the MIR panel uncluttered.
    heights = [0.77, 0.60]

    for (wave, label), height in zip(features, heights):
        ax.axvline(
            wave,
            color="#686868",
            lw=0.9,
            ls=":",
            alpha=0.62,
            zorder=2,
        )
        ax.text(
            wave * 1.035,
            height,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#666666",
            zorder=9,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-style CosmicGrain / SKIRT SED plot."
    )
    parser.add_argument(
        "faceon",
        help="Face-on (or primary-view) SKIRT *_sed.dat file",
    )
    parser.add_argument(
        "edgeon",
        nargs="?",
        default=None,
        help="Optional edge-on SKIRT *_sed.dat file",
    )
    parser.add_argument(
        "-o", "--output",
        default="halo569_sed.png",
        help="Output image filename (default: halo569_sed.png)",
    )
    parser.add_argument(
        "--quantity",
        choices=("nufnu", "fnu"),
        default="nufnu",
        help="Plot nu F_nu (default) or F_nu",
    )
    parser.add_argument(
        "--title",
        default="CosmicGrain Halo 569 — SKIRT synthetic SED",
        help="Plot title",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=0.1,
        help="Minimum wavelength [micron]",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=1000.0,
        help="Maximum wavelength [micron]",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG output resolution",
    )
    args = parser.parse_args()

    print(f"plot_skirt_sed version: {VERSION}")

    face = load_sed(args.faceon)
    edge = load_sed(args.edgeon) if args.edgeon else None

    lam = face["lambda"]

    face_total_raw = to_plot_quantity(lam, face["total"], args.quantity)
    peak = np.nanmax(face_total_raw)

    # Hide only extremely tiny numerical values while preserving real MIR valleys.
    component_floor = peak * 1e-8

    face_total = clean_for_log(face_total_raw)
    intrinsic = clean_for_log(
        to_plot_quantity(lam, face["intrinsic"], args.quantity),
        component_floor,
    )
    attenuated = clean_for_log(
        to_plot_quantity(lam, face["attenuated_stellar"], args.quantity),
        component_floor,
    )
    dust = clean_for_log(
        to_plot_quantity(lam, face["dust_emission"], args.quantity),
        component_floor,
    )

    fig, ax = plt.subplots(figsize=(12.0, 7.4))

    add_spectral_regions(ax)
    add_silicate_guides(ax)

    # Main total SED.
    ax.plot(
        lam,
        face_total,
        color="#315f9e",
        lw=2.8,
        label="Face-on total",
        zorder=8,
    )

    # Optional viewing-angle comparison.
    if edge is not None:
        edge_total = clean_for_log(
            to_plot_quantity(edge["lambda"], edge["total"], args.quantity)
        )
        ax.plot(
            edge["lambda"],
            edge_total,
            color="#b4617e",
            lw=2.0,
            alpha=0.88,
            label="Edge-on total",
            zorder=7,
        )

    # Physical components.
    ax.plot(
        lam,
        intrinsic,
        color="#4d4d4d",
        lw=1.9,
        ls="--",
        label="Intrinsic stellar",
        zorder=6,
    )

    ax.plot(
        lam,
        attenuated,
        color="#75679b",
        lw=1.9,
        ls=":",
        label="Attenuated stellar",
        zorder=6,
    )

    ax.plot(
        lam,
        dust,
        color="#c7792d",
        lw=2.2,
        ls="-.",
        label="Dust emission",
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(args.xmin, args.xmax)

    # Consistent dynamic range without letting numerical underflow dominate.
    ax.set_ylim(peak * 1e-6, peak * 4.0)

    ax.set_xlabel(
        r"Wavelength  $\lambda$  [$\mu$m]",
        fontsize=13,
    )

    if args.quantity == "nufnu":
        ax.set_ylabel(
            r"Energy flux  $\nu F_\nu$  [W m$^{-2}$]",
            fontsize=13,
        )
    else:
        ax.set_ylabel(
            r"Flux density  $F_\nu$  [Jy]",
            fontsize=13,
        )

    ax.set_title(
        args.title,
        fontsize=16,
        pad=18,
    )

    # Restrained logarithmic grid.
    ax.grid(which="major", color="#777777", alpha=0.16, lw=0.8)
    ax.grid(which="minor", color="#999999", alpha=0.055, lw=0.5)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Two-column legend; no explanatory footer text.
    legend = ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=10,
        frameon=True,
        framealpha=0.93,
        borderpad=0.8,
        columnspacing=1.5,
        handlelength=3.4,
    )
    legend.get_frame().set_edgecolor("#cfcfcf")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
