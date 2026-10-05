#!/usr/bin/env python3
"""
CosmicGrain z=0 mass-metallicity relation versus direct-method observations.

Default paths assume execution from ~/gadget4:
  scripts/cosmicgrain.mplstyle
  scripts/apjad3042t1_mrt.txt
  scripts/catalog.dat
  cosmicgrain_sagabg_z0.csv

Observational layers:
  * SAGAbg: all 120 direct-Te galaxies; P16--P84 bars are measurement/posterior
    intervals from the published table.
  * Andrews & Martini (2013): published direct-method MZR fit.
  * PHANGS/Brazzini: one Halpha-weighted direct-O/H summary per galaxy.
    PHANGS vertical bars show the Halpha-weighted REGION-TO-REGION RMS spread
    among the auroral-line-selected HII regions. They are not errors on the mean.

CosmicGrain:
  * All finite SFR-weighted O/H measurements use the same symbol.
  * N_SF is printed as a diagnostic only; no arbitrary N_SF threshold is encoded
    in the figure.
  * A galaxy with no finite SFR-weighted O/H (e.g. zero current SF) is reported
    but not given a substitute nebular-metallicity point.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PHANGS_M = {
    "IC5332": 9.67, "NGC0628": 10.34, "NGC1087": 9.93, "NGC1300": 10.62,
    "NGC1365": 10.99, "NGC1385": 9.98, "NGC1433": 10.87, "NGC1512": 10.71,
    "NGC1566": 10.78, "NGC1672": 10.73, "NGC2835": 10.00, "NGC3351": 10.36,
    "NGC3627": 10.83, "NGC4254": 10.42, "NGC4303": 10.52, "NGC4321": 10.75,
    "NGC4535": 10.53, "NGC5068": 9.40, "NGC7496": 10.00,
}


def fl(s):
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return np.nan


def read_sagabg(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(line) >= 435 and line[:19].strip().isdigit():
                row = (
                    fl(line[40:46]),    # z
                    fl(line[47:51]),    # log Mstar
                    fl(line[421:425]),  # P16 O/H
                    fl(line[426:430]),  # P50 O/H
                    fl(line[431:435]),  # P84 O/H
                )
                if np.isfinite(row[1]) and np.isfinite(row[3]):
                    rows.append(row)
    return rows


def normalize_phangs_name(s):
    s = s.strip().upper().replace(" ", "")
    if s.startswith("NGC") and s[3:].isdigit():
        return "NGC" + s[3:].zfill(4)
    if s.startswith("IC") and s[2:].isdigit():
        return "IC" + s[2:].zfill(4)
    return s


def read_phangs(path):
    """
    CDS J/A+A/691/A173 fixed-width fields used here:
      Galaxy : bytes 1--7
      FHa    : bytes 78--86
      A(O)   : bytes 439--455
    """
    groups = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(line) < 455:
                continue

            galaxy = normalize_phangs_name(line[:7])
            fha = fl(line[77:86])
            oh12 = fl(line[438:455])

            if (
                galaxy in PHANGS_M
                and np.isfinite(fha) and fha > 0
                and np.isfinite(oh12)
            ):
                groups[galaxy].append((fha, oh12))

    out = []
    for galaxy, vals in sorted(groups.items()):
        w = np.array([v[0] for v in vals], dtype=float)
        oh12_regions = np.array([v[1] for v in vals], dtype=float)

        # Halpha-weight the abundance in LINEAR O/H space.
        oh_linear = 10.0 ** (oh12_regions - 12.0)
        mean_linear = np.sum(w * oh_linear) / np.sum(w)
        mean_oh12 = 12.0 + np.log10(mean_linear)

        # Weighted region-to-region RMS in the plotted logarithmic abundance.
        # This is deliberately presented as a spread, NOT uncertainty on mean.
        wmean_log = np.sum(w * oh12_regions) / np.sum(w)
        spread = np.sqrt(
            np.sum(w * (oh12_regions - wmean_log) ** 2) / np.sum(w)
        )

        out.append(
            (galaxy, PHANGS_M[galaxy], mean_oh12, spread, len(vals))
        )

    return out


def read_cosmicgrain(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                nsf = int(float(r.get("N_SFgas_30pkpc", 0) or 0))
            except ValueError:
                nsf = 0

            rows.append((
                r["CG"].strip(),
                fl(r.get("log10_Mstar_Msun")),
                fl(r.get("OH12_SFRweighted")),
                nsf,
                fl(r.get("SFR_Msun_yr")),
                fl(r.get("OH12_allgas_massweighted")),
            ))
    return rows


def andrews_martini_2013(logm):
    z0 = 8.798
    log_mto = 8.901
    gamma = 0.640
    return z0 - np.log10(1.0 + 10.0 ** (gamma * (log_mto - logm)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="scripts/cosmicgrain.mplstyle")
    p.add_argument("--sagabg", default="scripts/apjad3042t1_mrt.txt")
    p.add_argument("--phangs", default="scripts/catalog.dat")
    p.add_argument("--cosmicgrain", default="cosmicgrain_sagabg_z0.csv")
    p.add_argument("--prefix", default="cosmicgrain_fullsuite_mzr")
    p.add_argument("--label-phangs", action="store_true")
    args = p.parse_args()

    style = Path(args.style)
    if not style.exists():
        raise FileNotFoundError(
            f"CosmicGrain stylesheet not found: {style}\n"
            "Put cosmicgrain.mplstyle in scripts/ or pass --style PATH."
        )
    plt.style.use(style)

    saga = read_sagabg(args.sagabg)
    phangs = read_phangs(args.phangs)
    cg = read_cosmicgrain(args.cosmicgrain)

    if len(saga) != 120:
        print(f"WARNING: parsed {len(saga)} SAGAbg galaxies rather than 120.")
    else:
        print("SAGAbg catalog check: PASS -- 120 galaxies")

    print(
        f"PHANGS: {sum(x[4] for x in phangs)} valid direct-abundance regions "
        f"in {len(phangs)} represented galaxies"
    )
    print("\nPHANGS Halpha-weighted selected-region summaries:")
    for name, logm, oh12, spread, nreg in phangs:
        print(
            f"  {name:8s} logM*={logm:.2f} OH12={oh12:.3f} "
            f"spread={spread:.3f} Nreg={nreg}"
        )

    z, sm, p16, p50, p84 = map(np.array, zip(*saga))

    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    # ------------------------------------------------------------------
    # SAGAbg: one population. All 120 get their published P16--P84 bars.
    # Error bars are intentionally faint so the point cloud stays legible.
    # ------------------------------------------------------------------
    lo = p50 - p16
    hi = p84 - p50
    good_err = (
        np.isfinite(lo) & np.isfinite(hi) & (lo >= 0) & (hi >= 0)
    )

    ax.errorbar(
        sm[good_err],
        p50[good_err],
        yerr=np.vstack((lo[good_err], hi[good_err])),
        fmt="none",
        ecolor="0.70",
        elinewidth=0.55,
        alpha=0.32,
        capsize=0,
        zorder=1,
    )

    ax.scatter(
        sm,
        p50,
        s=27,
        color="0.58",
        alpha=0.60,
        linewidths=0,
        label=rf"SAGAbg direct-$T_e$ ({len(saga)})",
        zorder=2,
    )

    # Andrews & Martini direct-method relation.
    am_x = np.linspace(7.4, 10.5, 350)
    ax.plot(
        am_x,
        andrews_martini_2013(am_x),
        linestyle="--",
        color="0.20",
        label="Andrews & Martini (2013)",
        zorder=3,
    )

    # ------------------------------------------------------------------
    # PHANGS: point = Halpha-weighted direct O/H from selected regions.
    # Bar = Halpha-weighted REGION-TO-REGION RMS, not error on the mean.
    # ------------------------------------------------------------------
    if phangs:
        px = np.array([x[1] for x in phangs])
        py = np.array([x[2] for x in phangs])
        ps = np.array([x[3] for x in phangs])

        ax.errorbar(
            px,
            py,
            yerr=ps,
            fmt="none",
            color="#CC79A7",
            elinewidth=1.0,
            alpha=0.70,
            capsize=2.5,
            zorder=4,
        )
        ax.scatter(
            px,
            py,
            s=72,
            marker="D",
            color="#CC79A7",
            edgecolors="black",
            linewidths=0.65,
            label=r"PHANGS direct regions (H$\alpha$-weighted)",
            zorder=5,
        )

        if args.label_phangs:
            for name, logm, oh12, spread, nreg in phangs:
                ax.annotate(
                    name,
                    (logm, oh12),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    # ------------------------------------------------------------------
    # CosmicGrain: one symbol for every halo. N_SF remains diagnostic only.
    # ------------------------------------------------------------------
    finite = [
        r for r in cg if np.isfinite(r[1]) and np.isfinite(r[2])
    ]
    no_nebular = [
        r for r in cg if np.isfinite(r[1]) and not np.isfinite(r[2])
    ]

    if finite:
        cx = np.array([r[1] for r in finite])
        cy = np.array([r[2] for r in finite])

        ax.scatter(
            cx,
            cy,
            s=125,
            marker="o",
            color="#D55E00",
            edgecolors="black",
            linewidths=0.8,
            label="CosmicGrain",
            zorder=7,
        )

        for name, logm, oh12, nsf, sfr, oh_all in finite:
            ax.annotate(
                name,
                (logm, oh12),
                xytext=(7, 6),
                textcoords="offset points",
                fontweight="bold",
                fontsize=10,
                zorder=8,
            )

    for name, logm, oh12, nsf, sfr, oh_all in no_nebular:
        print(
            f"{name}: not plotted -- no finite SFR-weighted O/H "
            f"(SFR={sfr:.3e}, N_SF={nsf}, all-gas OH12={oh_all:.3f})"
        )

    print("\nCosmicGrain plotted:")
    for name, logm, oh12, nsf, sfr, oh_all in finite:
        print(
            f"  {name:4s} logM*={logm:.3f} OH12_SFR={oh12:.3f} "
            f"N_SF={nsf:4d} SFR={sfr:.3e}"
        )

    ax.set_xlim(6.65, 11.45)
    ax.set_ylim(6.95, 9.18)
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$12+\log_{10}(\mathrm{O/H})$")
    ax.set_title(r"CosmicGrain $z=0$ mass--metallicity relation")

    # Minor grid is intentionally much lighter than the stylesheet major grid.
    ax.grid(True, which="minor", color="0.93", linewidth=0.3)

    ax.legend(loc="upper left")
    fig.tight_layout()

    fig.savefig(args.prefix + ".png")
    fig.savefig(args.prefix + ".pdf")
    plt.close(fig)

    print(f"\nWrote {args.prefix}.png")
    print(f"Wrote {args.prefix}.pdf")


if __name__ == "__main__":
    main()
