#!/usr/bin/env python3
"""
plot_mzr_comparison.py

Place a CosmicGrain simulated galaxy on the z~0 gas-phase mass-metallicity
relation alongside literature determinations.

Comparison relations
--------------------
Curti et al. (2020), MNRAS 491, 944
    SDSS (~150k galaxies), metallicities on the Curti et al. (2017) T_e-anchored
    empirical strong-line scale.  Smoothly broken power law:

        12 + log(O/H) = Z0 - (gamma/beta) * log10(1 + (M*/M0)^(-beta))

    Best fit: Z0 = 8.793 +/- 0.005, log(M0) = 10.02 +/- 0.09,
              gamma = 0.28 +/- 0.02, beta = 1.2 +/- 0.2
    (Verify against Table 2 of the paper before submission.)

Andrews & Martini (2013), ApJ 765, 140
    Direct T_e method on stacked SDSS spectra.  Moustakas et al. (2011) form:

        12 + log(O/H) = Z_asm - log10(1 + (M_TO/M*)^gamma)

    Best fit (their Table 4, "MZR" row): Z_asm = 8.798, log(M_TO) = 8.901,
    gamma = 0.640, fit over log(M*) = 7.4-10.5.

Tremonti et al. (2004), ApJ 613, 898
    SDSS DR2, photoionization-model calibration.  Quadratic in log(M*):

        12 + log(O/H) = -1.492 + 1.847x - 0.08026x^2,  x = log10(M*/Msun)

    Fit over log(M*) = 8.5-11.5.  Included for lineage; its normalization sits
    ~0.3 dex above the direct-method relations at the high-mass end (see
    Kewley & Ellison 2008 for the full calibration systematic).

Usage
-----
    python plot_mzr_comparison.py
    python plot_mzr_comparison.py --logmstar 10.83 --oh 9.08 --label "Halo 441"
    python plot_mzr_comparison.py --oh-err 0.05 --outdir figures/
    python plot_mzr_comparison.py --points sim_points.csv

CSV format for --points (header required):
    logmstar,oh,label,oh_err,logmstar_err
    10.83,9.08,Halo 441 (z=0),0.05,0.05
    10.79,9.02,Halo 569 (z=0),,
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Asplund et al. (2009) photospheric solar oxygen abundance
OH_SOLAR = 8.69

# Curti et al. (2020) gas-phase MZR
C20 = dict(Z0=8.793, logM0=10.02, gamma=0.28, beta=1.2)
C20_RANGE = (8.5, 11.0)

# Andrews & Martini (2013) direct-method MZR (their Table 4)
AM13 = dict(Z_asm=8.798, logM_TO=8.901, gamma=0.640)
AM13_RANGE = (7.4, 10.5)

# Tremonti et al. (2004)
T04_COEFFS = (-1.492, 1.847, -0.08026)
T04_RANGE = (8.5, 11.5)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

def curti20(logmstar):
    """Curti et al. (2020) gas-phase MZR."""
    ratio = 10.0 ** (np.asarray(logmstar, dtype=float) - C20["logM0"])
    return C20["Z0"] - (C20["gamma"] / C20["beta"]) * np.log10(
        1.0 + ratio ** (-C20["beta"])
    )


def andrews_martini13(logmstar):
    """Andrews & Martini (2013) direct-method MZR."""
    ratio = 10.0 ** (AM13["logM_TO"] - np.asarray(logmstar, dtype=float))
    return AM13["Z_asm"] - np.log10(1.0 + ratio ** AM13["gamma"])


def tremonti04(logmstar):
    """Tremonti et al. (2004) SDSS MZR."""
    a, b, c = T04_COEFFS
    x = np.asarray(logmstar, dtype=float)
    return a + b * x + c * x * x


RELATIONS = [
    # (function, label, fit range, color, linestyle, zorder)
    (curti20, "Curti+2020 (SDSS, $T_e$-anchored)", C20_RANGE, "#1f6fb4", "-", 3),
    (andrews_martini13, "Andrews & Martini 2013 (direct $T_e$)", AM13_RANGE, "#2f8f4e", "-", 3),
    (tremonti04, "Tremonti+2004 (photoion. models)", T04_RANGE, "#9a6fb0", "--", 2),
]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def load_points(args):
    """Return a list of dicts describing the simulation points to plot."""
    if args.points:
        pts = []
        with open(args.points, newline="") as fh:
            for row in csv.DictReader(fh):
                pts.append(
                    dict(
                        logmstar=float(row["logmstar"]),
                        oh=float(row["oh"]),
                        label=row.get("label") or "CosmicGrain",
                        oh_err=_maybe_float(row.get("oh_err")),
                        logmstar_err=_maybe_float(row.get("logmstar_err")),
                    )
                )
        if not pts:
            raise ValueError(f"No rows found in {args.points}")
        return pts

    return [
        dict(
            logmstar=args.logmstar,
            oh=args.oh,
            label=args.label,
            oh_err=args.oh_err,
            logmstar_err=args.logmstar_err,
        )
    ]


def _maybe_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_offsets(points):
    """Print offsets from each relation; return per-point offset dicts."""
    all_offsets = []
    header = f"{'relation':<38} {'predicted':>10} {'offset':>9}  note"
    for pt in points:
        logm, oh = pt["logmstar"], pt["oh"]
        print()
        print(f"{pt['label']}:  log(M*/Msun) = {logm:.2f},  12+log(O/H) = {oh:.2f}")
        print(f"  = {10**(oh - OH_SOLAR):.2f} x solar (Asplund+2009, 12+log(O/H)_sun = {OH_SOLAR})")
        print()
        print("  " + header)
        print("  " + "-" * len(header))
        offsets = {}
        for func, label, frange, _c, _ls, _z in RELATIONS:
            pred = float(func(logm))
            delta = oh - pred
            offsets[label] = delta
            note = ""
            if not (frange[0] <= logm <= frange[1]):
                note = f"EXTRAPOLATED (fit range {frange[0]}-{frange[1]})"
            short = label.split(" (")[0]
            print(f"  {short:<38} {pred:>10.3f} {delta:>+9.3f}  {note}")
        all_offsets.append(offsets)
    print()
    return all_offsets


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(points, args):
    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight))

    grid = np.linspace(args.xmin, args.xmax, 600)

    # Shaded band between the two direct-method-anchored relations, which
    # brackets the dominant systematic (metallicity calibration choice).
    band_lo = np.minimum(curti20(grid), andrews_martini13(grid))
    band_hi = np.maximum(curti20(grid), andrews_martini13(grid))
    ax.fill_between(
        grid, band_lo, band_hi,
        color="#7aa6c2", alpha=0.18, lw=0, zorder=1,
    )

    for func, label, frange, color, ls, zorder in RELATIONS:
        y = func(grid)
        inside = (grid >= frange[0]) & (grid <= frange[1])
        # Extrapolated portions dotted, calibrated portion solid/dashed.
        ax.plot(grid, y, color=color, ls=":", lw=1.4, alpha=0.65, zorder=zorder)
        ax.plot(
            np.where(inside, grid, np.nan), np.where(inside, y, np.nan),
            color=color, ls=ls, lw=2.2, zorder=zorder + 1, label=label,
        )

    # Solar reference
    ax.axhline(OH_SOLAR, color="0.45", ls=(0, (6, 4)), lw=1.0, zorder=1)
    ax.text(
        args.xmin + 0.08, OH_SOLAR + 0.015, r"$Z_\odot$ (Asplund+09)",
        color="0.4", fontsize=8, va="bottom",
    )

    # Simulation point(s)
    markers = ["*", "o", "s", "D", "^", "v", "P"]
    sizes = [420, 130, 120, 110, 130, 130, 140]
    for i, pt in enumerate(points):
        ax.errorbar(
            pt["logmstar"], pt["oh"],
            yerr=pt["oh_err"], xerr=pt["logmstar_err"],
            fmt="none", ecolor="#b3261e", elinewidth=1.4,
            capsize=3, zorder=9,
        )
        ax.scatter(
            pt["logmstar"], pt["oh"],
            marker=markers[i % len(markers)], s=sizes[i % len(sizes)],
            facecolor="#d93025", edgecolor="k", linewidth=0.9,
            zorder=10, label=pt["label"],
        )

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel(r"$\log(M_\star/\mathrm{M_\odot})$")
    ax.set_ylabel(r"$12 + \log(\mathrm{O/H})$")
    if args.title:
        ax.set_title(args.title)

    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(alpha=0.12, lw=0.6)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], color="0.5", ls=":", lw=1.4,
               label="extrapolated beyond fit range")
    )
    labels.append("extrapolated beyond fit range")
    leg = ax.legend(
        handles, labels, loc="lower right", fontsize=8.5,
        frameon=True, framealpha=0.92, borderpad=0.7,
    )
    leg.get_frame().set_linewidth(0.6)

    # Annotate offsets for the first point
    if args.annotate and points:
        pt = points[0]
        dc = pt["oh"] - float(curti20(pt["logmstar"]))
        da = pt["oh"] - float(andrews_martini13(pt["logmstar"]))
        dt = pt["oh"] - float(tremonti04(pt["logmstar"]))
        txt = (
            f"{pt['label']} offsets:\n"
            f"  vs Curti+20:  {dc:+.2f} dex\n"
            f"  vs AM13:      {da:+.2f} dex\n"
            f"  vs T04:       {dt:+.2f} dex"
        )
        ax.text(
            0.025, 0.965, txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec="0.7", lw=0.6, alpha=0.92),
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a CosmicGrain galaxy on the z~0 gas-phase MZR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--logmstar", type=float, default=10.83,
                   help="log10(M*/Msun) of the simulated galaxy")
    p.add_argument("--oh", type=float, default=9.08,
                   help="12 + log(O/H) of the simulated galaxy")
    p.add_argument("--label", type=str, default="CosmicGrain Halo 441",
                   help="legend label for the simulation point")
    p.add_argument("--oh-err", type=float, default=None,
                   help="1-sigma error bar on 12+log(O/H)")
    p.add_argument("--logmstar-err", type=float, default=None,
                   help="1-sigma error bar on log(M*)")
    p.add_argument("--points", type=str, default=None,
                   help="CSV of multiple points (overrides --logmstar/--oh)")

    p.add_argument("--xmin", type=float, default=7.5)
    p.add_argument("--xmax", type=float, default=11.5)
    p.add_argument("--ymin", type=float, default=7.5)
    p.add_argument("--ymax", type=float, default=9.35)
    p.add_argument("--figwidth", type=float, default=7.0)
    p.add_argument("--figheight", type=float, default=5.4)
    p.add_argument("--title", type=str,
                   default=r"Gas-phase mass$-$metallicity relation at $z\sim0$")
    p.add_argument("--no-annotate", dest="annotate", action="store_false",
                   help="suppress the offset text box")

    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--outname", type=str, default="mzr_comparison")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--formats", type=str, default="png,pdf",
                   help="comma-separated output formats")
    return p.parse_args()


def main():
    args = parse_args()
    points = load_points(args)

    report_offsets(points)

    fig = make_figure(points, args)

    os.makedirs(args.outdir, exist_ok=True)
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        path = os.path.join(args.outdir, f"{args.outname}.{fmt}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
