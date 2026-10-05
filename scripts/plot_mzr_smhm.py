#!/usr/bin/env python3
"""
plot_mzr_comparison.py

Two-panel comparison of a CosmicGrain galaxy against z~0 observational /
empirical scaling relations:

    left  -- gas-phase mass-metallicity relation (MZR)
    right -- stellar mass - halo mass relation (SMHM)

MZR comparison relations
------------------------
Curti et al. (2020), MNRAS 491, 944
    SDSS (~150k galaxies), metallicities on the Curti et al. (2017) T_e-anchored
    empirical strong-line scale.  Smoothly broken power law:

        12 + log(O/H) = Z0 - (gamma/beta) * log10(1 + (M*/M0)^(-beta))

    Best fit: Z0 = 8.793 +/- 0.005, log(M0) = 10.02 +/- 0.09,
              gamma = 0.28 +/- 0.02, beta = 1.2 +/- 0.2

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

SMHM comparison relations
-------------------------
Moster et al. (2013), MNRAS 428, 3121
    Multi-epoch abundance matching.  Double power law:

        M*/Mh = 2N [ (Mh/M1)^(-beta) + (Mh/M1)^(gamma) ]^(-1)

    z = 0: log(M1) = 11.590 +/- 0.236, N = 0.0351 +/- 0.0058,
           beta = 1.376 +/- 0.153, gamma = 0.608 +/- 0.059
    Quoted intrinsic scatter 0.15 dex in log(M*) at fixed halo mass.

Behroozi et al. (2013), ApJ 770, 57
    Abundance matching + cosmic SFH.  Their Eq. 3:

        log10(M*) = log10(eps*M1) + f(log10(Mh/M1)) - f(0)
        f(x) = -log10(10^(alpha*x) + 1)
               + delta * [log10(1+exp(x))]^gamma / (1 + exp(10^(-x)))

    z = 0: log(M1) = 11.514, log(eps) = -1.777, alpha = -1.412,
           delta = 3.508, gamma = 0.316

HALO MASS DEFINITION -- worth checking before you quote a number.  Behroozi+13
uses M_vir (Bryan & Norman 1998); Moster+13's halo masses are M_200-like.  The
M_vir <-> M_200c conversion is roughly a factor ~1.1 (~0.04-0.05 dex) at
Mh ~ 1e12, i.e. small compared to the offsets this figure is measuring, but a
referee will ask which definition your M200 uses.

VERIFY the fit constants above against the published tables before submission.

Usage
-----
    python plot_mzr_comparison.py
    python plot_mzr_comparison.py --logmstar 10.83 --oh 9.08 --logmhalo 11.99
    python plot_mzr_comparison.py --panel mzr
    python plot_mzr_comparison.py --points sim_points.csv

CSV format for --points (header required; logmhalo optional):
    logmstar,oh,logmhalo,label
    10.83,9.08,11.99,Halo 441 (z=0)
    10.79,9.02,,Halo 569 (z=0)
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OH_SOLAR = 8.69          # Asplund et al. (2009), used only in stdout reporting
F_BARYON = 0.157         # Omega_b / Omega_m, Planck-like

# --- MZR ---
C20 = dict(Z0=8.793, logM0=10.02, gamma=0.28, beta=1.2)
C20_RANGE = (8.5, 11.0)

AM13 = dict(Z_asm=8.798, logM_TO=8.901, gamma=0.640)
AM13_RANGE = (7.4, 10.5)

T04_COEFFS = (-1.492, 1.847, -0.08026)
T04_RANGE = (8.5, 11.5)

# --- SMHM ---
M13 = dict(logM1=11.590, N=0.0351, beta=1.376, gamma=0.608)
M13_SCATTER = 0.15
M13_RANGE = (10.0, 14.5)

B13 = dict(logM1=11.514, logeps=-1.777, alpha=-1.412, delta=3.508, gamma=0.316)
B13_RANGE = (10.5, 14.5)


# ---------------------------------------------------------------------------
# MZR relations
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


MZR_RELATIONS = [
    # (function, label, fit range, color, linestyle, zorder)
    (curti20, "Curti 2020", C20_RANGE, "#1f6fb4", "-", 3),
    (andrews_martini13, "Andrews & Martini 2013", AM13_RANGE, "#2f8f4e", "-", 3),
    (tremonti04, "Tremonti 2004", T04_RANGE, "#9a6fb0", "--", 2),
]


# ---------------------------------------------------------------------------
# SMHM relations
# ---------------------------------------------------------------------------

def moster13(logmhalo):
    """Moster et al. (2013) z=0 SMHM; returns log10(M*/Msun)."""
    logmhalo = np.asarray(logmhalo, dtype=float)
    ratio = 10.0 ** (logmhalo - M13["logM1"])
    frac = 2.0 * M13["N"] / (ratio ** (-M13["beta"]) + ratio ** M13["gamma"])
    return logmhalo + np.log10(frac)


def behroozi13(logmhalo):
    """Behroozi et al. (2013) z=0 SMHM; returns log10(M*/Msun)."""

    def f(x):
        term1 = -np.log10(10.0 ** (B13["alpha"] * x) + 1.0)
        term2 = (
            B13["delta"]
            * np.log10(1.0 + np.exp(x)) ** B13["gamma"]
            / (1.0 + np.exp(10.0 ** (-x)))
        )
        return term1 + term2

    x = np.asarray(logmhalo, dtype=float) - B13["logM1"]
    return B13["logeps"] + B13["logM1"] + f(x) - f(0.0)


SMHM_RELATIONS = [
    (moster13, "Moster 2013", M13_RANGE, "#1f6fb4", "-", 3),
    (behroozi13, "Behroozi 2013", B13_RANGE, "#2f8f4e", "-", 3),
]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _maybe_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_points(args):
    """Return a list of dicts describing the simulation points to plot."""
    if args.points:
        pts = []
        with open(args.points, newline="") as fh:
            for row in csv.DictReader(fh):
                pts.append(
                    dict(
                        logmstar=float(row["logmstar"]),
                        oh=_maybe_float(row.get("oh")),
                        logmhalo=_maybe_float(row.get("logmhalo")),
                        label=row.get("label") or "CosmicGrain",
                    )
                )
        if not pts:
            raise ValueError(f"No rows found in {args.points}")
        return pts

    return [
        dict(
            logmstar=args.logmstar,
            oh=args.oh,
            logmhalo=args.logmhalo,
            label=args.label,
        )
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report_block(title, relations, x, y, ylabel):
    header = f"{'relation':<38} {'predicted':>10} {'offset':>9}  note"
    print(f"  {title}")
    print("  " + header)
    print("  " + "-" * len(header))
    for func, label, frange, _c, _ls, _z in relations:
        pred = float(func(x))
        note = ""
        if not (frange[0] <= x <= frange[1]):
            note = f"EXTRAPOLATED (fit range {frange[0]}-{frange[1]})"
        short = label.split(" (")[0]
        print(f"  {short:<38} {pred:>10.3f} {y - pred:>+9.3f}  {note}")
    print()


def report(points):
    for pt in points:
        print()
        print("=" * 72)
        print(pt["label"])
        print("=" * 72)

        if pt["oh"] is not None:
            print(
                f"  log(M*/Msun) = {pt['logmstar']:.2f}, "
                f"12+log(O/H) = {pt['oh']:.2f} "
                f"({10 ** (pt['oh'] - OH_SOLAR):.2f} x solar)"
            )
            print()
            _report_block("MZR", MZR_RELATIONS, pt["logmstar"], pt["oh"], "12+log(O/H)")

        if pt["logmhalo"] is not None:
            mh = 10.0 ** pt["logmhalo"]
            ms = 10.0 ** pt["logmstar"]
            print(
                f"  log(Mh/Msun) = {pt['logmhalo']:.2f}, "
                f"M*/Mh = {ms / mh:.4f}, "
                f"M*/(f_b Mh) = {ms / (F_BARYON * mh):.3f}"
            )
            print()
            _report_block("SMHM", SMHM_RELATIONS, pt["logmhalo"], pt["logmstar"], "log M*")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

MARKERS = ["*", "o", "s", "D", "^", "v", "P"]
SIZES = [420, 130, 120, 110, 130, 130, 140]


def _draw_relations(ax, relations, grid):
    for func, label, frange, color, ls, zorder in relations:
        inside = (grid >= frange[0]) & (grid <= frange[1])
        ax.plot(
            grid[inside], func(grid[inside]),
            color=color, ls=ls, lw=2.2, zorder=zorder + 1, label=label,
        )


def _draw_points(ax, points, xkey, ykey):
    for i, pt in enumerate(points):
        if pt.get(xkey) is None or pt.get(ykey) is None:
            continue
        ax.scatter(
            pt[xkey], pt[ykey],
            marker=MARKERS[i % len(MARKERS)], s=SIZES[i % len(SIZES)],
            facecolor="#d93025", edgecolor="k", linewidth=0.9,
            zorder=10, label=pt["label"],
        )


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(alpha=0.12, lw=0.6)


def _legend(ax):
    leg = ax.legend(loc="lower right", fontsize=8.5,
                    frameon=True, framealpha=0.92, borderpad=0.6)
    leg.get_frame().set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_mzr(ax, points, args):
    grid = np.linspace(args.mzr_xmin, args.mzr_xmax, 600)

    # Band between the two direct-method-anchored relations.
    lo = np.minimum(curti20(grid), andrews_martini13(grid))
    hi = np.maximum(curti20(grid), andrews_martini13(grid))
    ax.fill_between(grid, lo, hi, color="#7aa6c2", alpha=0.18, lw=0, zorder=1)

    _draw_relations(ax, MZR_RELATIONS, grid)
    _draw_points(ax, points, "logmstar", "oh")

    ax.set_xlim(args.mzr_xmin, args.mzr_xmax)
    ax.set_ylim(args.mzr_ymin, args.mzr_ymax)
    _style(ax, r"$\log(M_\star/\mathrm{M_\odot})$", r"$12 + \log(\mathrm{O/H})$",
           "Gas-phase mass$-$metallicity")
    _legend(ax)


def panel_smhm(ax, points, args):
    grid = np.linspace(args.smhm_xmin, args.smhm_xmax, 600)

    # Moster+2013 intrinsic scatter
    m13 = moster13(grid)
    ax.fill_between(grid, m13 - M13_SCATTER, m13 + M13_SCATTER,
                    color="#7aa6c2", alpha=0.18, lw=0, zorder=1)

    _draw_relations(ax, SMHM_RELATIONS, grid)
    _draw_points(ax, points, "logmhalo", "logmstar")

    ax.set_xlim(args.smhm_xmin, args.smhm_xmax)
    ax.set_ylim(args.smhm_ymin, args.smhm_ymax)
    _style(ax, r"$\log(M_{200}/\mathrm{M_\odot})$",
           r"$\log(M_\star/\mathrm{M_\odot})$",
           "Stellar mass $-$ halo mass")
    _legend(ax)


def make_figure(points, args):
    panels = {"both": ["mzr", "smhm"], "mzr": ["mzr"], "smhm": ["smhm"]}[args.panel]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(args.figwidth * len(panels), args.figheight),
        squeeze=False,
    )
    for ax, name in zip(axes[0], panels):
        {"mzr": panel_mzr, "smhm": panel_smhm}[name](ax, points, args)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a CosmicGrain galaxy on the z~0 MZR and SMHM relations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--logmstar", type=float, default=10.828,
                   help="log10(M*/Msun)  [6.73e10]")
    p.add_argument("--oh", type=float, default=9.08,
                   help="12 + log(O/H), SFR-weighted")
    p.add_argument("--logmhalo", type=float, default=11.990,
                   help="log10(M200/Msun)  [9.77e11]")
    p.add_argument("--label", type=str, default="CosmicGrain Halo 441")
    p.add_argument("--points", type=str, default=None,
                   help="CSV of multiple points (overrides the scalar args)")

    p.add_argument("--panel", choices=["both", "mzr", "smhm"], default="both")

    p.add_argument("--mzr-xmin", type=float, default=8.5)
    p.add_argument("--mzr-xmax", type=float, default=11.5)
    p.add_argument("--mzr-ymin", type=float, default=8.2)
    p.add_argument("--mzr-ymax", type=float, default=9.25)

    p.add_argument("--smhm-xmin", type=float, default=11.0)
    p.add_argument("--smhm-xmax", type=float, default=13.0)
    p.add_argument("--smhm-ymin", type=float, default=8.8)
    p.add_argument("--smhm-ymax", type=float, default=11.4)

    p.add_argument("--figwidth", type=float, default=6.2,
                   help="width PER PANEL in inches")
    p.add_argument("--figheight", type=float, default=5.2)

    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--outname", type=str, default="mzr_smhm_comparison")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--formats", type=str, default="png,pdf")
    return p.parse_args()


def main():
    args = parse_args()
    points = load_points(args)

    report(points)

    fig = make_figure(points, args)

    os.makedirs(args.outdir, exist_ok=True)
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        path = os.path.join(args.outdir, f"{args.outname}.{fmt}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
