#!/usr/bin/env python3
"""Make an R(V)-beta plot from precomputed CosmicGrain opacity maps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


H = 6.62607015e-34       # J s
K_B = 1.380649e-23       # J K^-1
C = 299792458.0          # m s^-1


def planck_bnu(nu_hz, temperature_k):
    """Planck function B_nu."""
    x = H * nu_hz / (K_B * temperature_k)
    return (2.0 * H * nu_hz**3 / C**2) / np.expm1(x)


def fit_beta(kappa, nu_hz, temperature_k, nu0_hz=353e9):
    """Fit I_nu = A (nu/nu0)^beta B_nu(T) in every map pixel."""
    bnu = planck_bnu(nu_hz, temperature_k)
    intensity = kappa * bnu[:, None, None]

    # Divide the synthetic emission by the fitted blackbody.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_emissivity = np.log(intensity / bnu[:, None, None])

    log_frequency = np.log(nu_hz / nu0_hz)
    centered_frequency = log_frequency - log_frequency.mean()
    beta = np.sum(
        centered_frequency[:, None, None] * log_emissivity, axis=0
    ) / np.sum(centered_frequency**2)
    beta[~np.all(np.isfinite(log_emissivity), axis=0)] = np.nan
    return beta


def equal_population_trend(rv, beta, nbins=10):
    """Median beta and its 16--84% range in equal-population R(V) bins."""
    rows = []
    for indices in np.array_split(np.argsort(rv), nbins):
        rows.append([
            np.median(rv[indices]),
            np.median(beta[indices]),
            np.percentile(beta[indices], 16),
            np.percentile(beta[indices], 84),
        ])
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Parent *_grain_rv_diagnostics.npz file")
    parser.add_argument("--temperature", type=float, default=17.0)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    source = Path(args.npz)
    prefix = args.prefix or source.with_suffix("").as_posix() + "_post"

    with np.load(source) as data:
        rv_map = data["Rv"]
        valid_map = data["valid_mask"].astype(bool)
        pixel_pc = 1000.0 * float(data["pixel_size_pkpc"])
        kappa = data["mbb_relative_kappa_spectra"]
        frequencies_ghz = data["mbb_frequencies_ghz"]

    beta_map = fit_beta(kappa, frequencies_ghz * 1e9, args.temperature)
    valid = valid_map & np.isfinite(rv_map) & np.isfinite(beta_map)
    rv = rv_map[valid]
    beta = beta_map[valid]
    trend = equal_population_trend(rv, beta)

    print(f"Bands (GHz): {frequencies_ghz}")
    print(f"Fixed temperature: {args.temperature:g} K")
    print(f"Plotted pixels: {len(rv):,}")

    fig, ax = plt.subplots(figsize=(8.2, 6.7))
    ax.scatter(
        rv, beta, s=11, alpha=0.20, linewidths=0,
        color="#2b6f9f", rasterized=True, label="Valid map pixels"
    )

    error = np.vstack((
        trend[:, 1] - trend[:, 2],
        trend[:, 3] - trend[:, 1],
    ))
    ax.errorbar(
        trend[:, 0], trend[:, 1], yerr=error,
        fmt="o-", ms=6.5, lw=2.2, capsize=3.5,
        color="#b43c2f", ecolor="#b43c2f",
        markerfacecolor="white", markeredgewidth=1.8,
        label="Median and 16--84% range",
    )

    ax.set_xlabel(r"$R(V)$")
    ax.set_ylabel(
        rf"$\beta_{{\rm MBB}}$ ($T_d={args.temperature:g}$ K fixed)"
    )
    ax.set_title(rf"Halo 441 (face-on {pixel_pc:g} pc pixels)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.18)
    fig.tight_layout()

    output = prefix + "_Rv_vs_beta_mbb.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
