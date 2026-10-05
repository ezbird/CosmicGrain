#!/usr/bin/env python3
"""Plot R(V) versus MBB beta for temperature-mixed CosmicGrain emission."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


H = 6.62607015e-34       # J s
K_B = 1.380649e-23       # J K^-1
C = 299792458.0          # m s^-1


def planck_bnu(nu_hz, temperature_k):
    """Planck function B_nu."""
    x = H * nu_hz / (K_B * temperature_k)
    return (2.0 * H * nu_hz**3 / C**2) / np.expm1(x)


def fit_one_pixel(log_intensity, nu_hz, temperature_bounds, nu0_hz=353e9):
    """Fit log(A), T, and beta to one four-band intensity spectrum."""
    log_frequency = np.log(nu_hz / nu0_hz)
    design = np.column_stack((np.ones(len(nu_hz)), log_frequency))

    def solve_at_temperature(temperature_k):
        reduced = log_intensity - np.log(planck_bnu(nu_hz, temperature_k))
        coefficients = np.linalg.lstsq(design, reduced, rcond=None)[0]
        residual = reduced - design @ coefficients
        return np.sum(residual**2), coefficients

    result = minimize_scalar(
        lambda temperature: solve_at_temperature(temperature)[0],
        bounds=temperature_bounds,
        method="bounded",
        options={"xatol": 1.0e-4},
    )
    _, coefficients = solve_at_temperature(result.x)
    return result.x, coefficients[1]


def fit_maps(intensity, nu_hz, valid, temperature_bounds):
    """Fit an effective single-temperature MBB in every accepted pixel."""
    temperature = np.full(valid.shape, np.nan)
    beta = np.full(valid.shape, np.nan)

    pixels = np.argwhere(valid & np.all(intensity > 0, axis=0))
    for iy, ix in pixels:
        log_intensity = np.log(intensity[:, iy, ix])
        temperature[iy, ix], beta[iy, ix] = fit_one_pixel(
            log_intensity, nu_hz, temperature_bounds
        )
    return temperature, beta


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
    parser.add_argument("--temperature-min", type=float, default=5.0)
    parser.add_argument("--temperature-max", type=float, default=100.0)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    source = Path(args.npz)
    prefix = args.prefix or source.with_suffix("").as_posix() + "_actualT"

    with np.load(source) as data:
        rv_map = data["Rv"]
        valid_map = data["valid_mask"].astype(bool)
        pixel_pc = 1000.0 * float(data["pixel_size_pkpc"])
        frequencies_ghz = data["mbb_frequencies_ghz"]
        intensity = data["mbb_temperature_mixed_intensity"]

    fitted_temperature, beta_map = fit_maps(
        intensity,
        frequencies_ghz * 1.0e9,
        valid_map,
        (args.temperature_min, args.temperature_max),
    )

    valid = valid_map & np.isfinite(rv_map) & np.isfinite(beta_map)
    rv = rv_map[valid]
    beta = beta_map[valid]
    trend = equal_population_trend(rv, beta)

    print(f"Bands (GHz): {frequencies_ghz}")
    print(f"Fitted pixels: {len(rv):,}")
    print(
        "Effective fitted T range: "
        f"{np.nanmin(fitted_temperature):.3f}--"
        f"{np.nanmax(fitted_temperature):.3f} K"
    )

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
    ax.set_ylabel(r"$\beta_{\rm MBB}$ (free-$T$ fit)")
    ax.set_title(rf"Halo 441 (face-on {pixel_pc:g} pc pixels)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.18)
    fig.tight_layout()

    output = prefix + "_Rv_vs_beta_mbb_actualT.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
