#!/usr/bin/env python3
"""Score runs of the 1024^3 coagulation-shattering (C-S) grid.

For each run and snapshot (default 27-29) this runs the Paper II figure
pipeline with the mock observers at a FIXED radius (the baseline's dust
half-mass radius, so every run is measured at the same place), then reduces it to:

  s2_logratio    mean over 0.38-1.88 kpc of log10[S2_RV(CG)/S2_RV(ZG)],
                 CG taken as the median over the 8 mock observers (0 = match)
  s2n_logratio   the same for the variance-normalised S2 (shape only)
  sigma_rv_ratio median over observers of sigma_RV(CG)/sigma_RV(ZG)
  gsd_l1_hd23    half the L1 distance between the 5-20 kpc mass-weighted GSD
                 and Hensley & Draine (2023) astrodust, per dex (0 = identical,
                 1 = no overlap)
  gsd_l1_base    the same distance to the baseline run's 5-20 kpc GSD
  f_lt20nm       5-20 kpc dust mass fraction in grains < 20 nm (HD23 ~0.23)
  a_med_nm       5-20 kpc mass-weighted median radius
  rv_median, av_median, n_rv_pix   face-on map (count >= 5, E(B-V) >= 0.03)

Rows are appended to analysis/cs_grid/scores.csv; summary.png shows the
snapshot-averaged scores on the C vs C/S plane.

Usage:  python3 scripts/cs_grid_score.py RUN_DIR [RUN_DIR ...] [--snaps 27 28 29]
        python3 scripts/cs_grid_score.py --summary-only
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "cs_grid"
ZG_MAP = ROOT / "data/observations/zhang_green/Rv_map_new.h5"
OBSERVER_RADIUS_KPC = 2.574     # dust half-mass radius of halo441_output_1024_br06_velfix, z=0
BASELINE = "halo441_output_1024_br06_velfix"
SEP_RANGE = (0.3, 2.0)          # kpc, separations averaged in the S2 score
FIELDS = ["run", "c", "r", "snap", "s2_logratio", "s2n_logratio", "sigma_rv_ratio",
          "gsd_l1_hd23", "gsd_l1_base", "f_lt20nm", "a_med_nm",
          "rv_median", "av_median", "n_rv_pix", "mock_sun_kpc"]


def grid_coords(run):
    m = re.search(r"_c(\d{3})_r(\d{3})", run)
    if m:
        return int(m.group(1)) / 100, int(m.group(2)) / 100
    if run == BASELINE:
        return 1.0, 1.0
    return np.nan, np.nan


def run_pipeline(run_dir, snap, outdir):
    if (outdir / "paper_II_figure_data.npz").exists():
        return True
    cmd = [sys.executable, str(ROOT / "scripts/cosmicgrain_paper_II_figures.py"),
           "--run", str(run_dir), "--snapshot", str(snap),
           "--observer-radius", str(OBSERVER_RADIUS_KPC),
           "--zg-map", str(ZG_MAP), "--outdir", str(outdir)]
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "pipeline.log", "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"  pipeline failed for {run_dir} snap {snap} (see {outdir/'pipeline.log'})")
    return rc == 0


def shell_gsd(outdir, lo=5, hi=20):
    d = np.load(outdir / "figure8c_grain_size_distributions_outside_inner.npz")
    edges = d["cg_edges_nm"]
    p = d[f"cg_density_{lo:g}_{hi:g}"]
    return edges, p, d["hd_radius_nm"], d["hd_density"]


def gsd_metrics(edges, p, hd_r, hd_p, base_p=None):
    logc = 0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1]))
    dlog = np.diff(np.log10(edges))
    hd_on_grid = np.interp(logc, np.log10(hd_r), hd_p, left=0, right=0)
    hd_on_grid /= np.sum(hd_on_grid * dlog)
    l1_hd = 0.5 * np.sum(np.abs(p - hd_on_grid) * dlog)
    l1_base = 0.5 * np.sum(np.abs(p - base_p) * dlog) if base_p is not None else np.nan
    cdf = np.cumsum(p * dlog)
    f20 = np.interp(np.log10(20.0), np.log10(edges[1:]), cdf)
    a_med = 10 ** np.interp(0.5, cdf, np.log10(edges[1:]))
    return l1_hd, l1_base, f20, a_med


def s2_metrics(outdir):
    d = np.load(outdir / "paper_II_figure_data.npz")
    sep = d["sf_separation_kpc"]
    use = (sep >= SEP_RANGE[0]) & (sep <= SEP_RANGE[1])
    with np.errstate(all="ignore"):
        cg = np.nanmedian(d["cg_rv_s2"], axis=0)
        zg = np.nanmedian(d["zg_rv_s2"], axis=0)
        cgn = np.nanmedian(d["cg_rv_s2_norm"], axis=0)
        zgn = np.nanmedian(d["zg_rv_s2_norm"], axis=0)
        lr = np.log10(cg / zg)[use]
        lrn = np.log10(cgn / zgn)[use]
        sig = np.nanmedian(d["sf_sigma_rv_cg"] / d["sf_sigma_rv_zg"])
    rv, av, ebv, cnt = d["R_V"], d["A_V"], d["E_BV"], d["dust_count"]
    ok = np.isfinite(rv) & (cnt >= int(d["min_count"])) & (ebv >= float(d["min_ebv_mag"]))
    return (np.nanmean(lr), np.nanmean(lrn), sig,
            np.nanmedian(rv[ok]) if ok.any() else np.nan,
            np.nanmedian(av[ok]) if ok.any() else np.nan,
            int(ok.sum()), float(d["mock_sun_radius_kpc"]))


def score(run_dir, snaps, base_cache):
    run = Path(run_dir).name
    c, r = grid_coords(run)
    rows = []
    for s in snaps:
        od = OUT / run / f"snap{s:03d}"
        if not (Path(run_dir) / f"snapdir_{s:03d}").exists():
            print(f"  {run}: no snapdir_{s:03d}, skipped")
            continue
        print(f"  {run} snap {s} ...", flush=True)
        if not run_pipeline(run_dir, s, od):
            continue
        edges, p, hd_r, hd_p = shell_gsd(od)
        base_p = base_cache.get(s)
        l1h, l1b, f20, amed = gsd_metrics(edges, p, hd_r, hd_p, base_p)
        s2, s2n, sig, rvm, avm, npx, rsun = s2_metrics(od)
        rows.append(dict(run=run, c=c, r=r, snap=s, s2_logratio=s2, s2n_logratio=s2n,
                         sigma_rv_ratio=sig, gsd_l1_hd23=l1h, gsd_l1_base=l1b,
                         f_lt20nm=f20, a_med_nm=amed, rv_median=rvm, av_median=avm,
                         n_rv_pix=npx, mock_sun_kpc=rsun))
    return rows


def write_rows(rows):
    path = OUT / "scores.csv"
    old = []
    if path.exists():
        with open(path) as f:
            old = [x for x in csv.DictReader(f)]
    keys = {(x["run"], int(x["snap"])) for x in rows}
    old = [x for x in old if (x["run"], int(x["snap"])) not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for x in old + rows:
            w.writerow({k: (f"{x[k]:.4g}" if isinstance(x[k], float) else x[k]) for k in FIELDS})


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = OUT / "scores.csv"
    if not path.exists():
        return
    with open(path) as f:
        rows = [x for x in csv.DictReader(f)]
    runs = {}
    for x in rows:
        runs.setdefault(x["run"], []).append(x)
    tab = []
    for run, xs in runs.items():
        avg = {k: np.nanmean([float(x[k]) for x in xs]) for k in FIELDS[4:]}
        tab.append(dict(run=run, c=float(xs[0]["c"]), r=float(xs[0]["r"]), nsnap=len(xs), **avg))
    tab.sort(key=lambda t: (t["c"], t["r"]))
    with open(OUT / "summary.txt", "w") as f:
        f.write(f"{'run':38s} {'c':>5s} {'r':>5s} {'n':>2s} {'S2 dex':>7s} {'S2n dex':>7s} {'sigRV':>6s} "
                f"{'L1 HD':>6s} {'L1 base':>7s} {'f<20':>5s} {'a_med':>6s} {'R_V':>5s} {'A_V':>5s}\n")
        for t in tab:
            f.write(f"{t['run']:38s} {t['c']:5.2f} {t['r']:5.2f} {t['nsnap']:2d} {t['s2_logratio']:7.2f} "
                    f"{t['s2n_logratio']:7.2f} {t['sigma_rv_ratio']:6.2f} {t['gsd_l1_hd23']:6.3f} "
                    f"{t['gsd_l1_base']:7.3f} {t['f_lt20nm']:5.3f} {t['a_med_nm']:6.1f} "
                    f"{t['rv_median']:5.2f} {t['av_median']:5.2f}\n")
    grid = [t for t in tab if np.isfinite(t["c"]) and "_dup" not in t["run"] and t["run"] != BASELINE]
    if not grid:
        return
    cs, rs = [0.25, 0.5, 1, 2], [0.5, 0.71, 1, 1.41]
    panels = [("s2_logratio", "log S2_RV CG/ZG (0 = ZG)", "RdBu_r", (-1.5, 1.5)),
              ("gsd_l1_hd23", "5-20 kpc GSD distance to HD23", "viridis_r", (0, 0.6)),
              ("gsd_l1_base", "5-20 kpc GSD distance to baseline", "viridis_r", (0, 0.4)),
              ("rv_median", "median face-on R_V", "PuOr_r", (2.1, 4.1))]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    for ax, (k, title, cmap, lim) in zip(axes, panels):
        img = np.full((4, 4), np.nan)
        for t in grid:
            if t["c"] in cs and t["r"] in rs:
                img[rs.index(t["r"]), cs.index(t["c"])] = t[k]
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1])
        for i in range(4):
            for j in range(4):
                if np.isfinite(img[i, j]):
                    ax.text(j, i, f"{img[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, lw=2, ec="k"))
        ax.set_xticks(range(4), [f"{c:g}" for c in cs])
        ax.set_yticks(range(4), [f"{r:g}" for r in rs])
        ax.set_xlabel("c (coagulation rate / baseline)")
        ax.set_ylabel("r (C/S / baseline)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "summary.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--snaps", type=int, nargs="+", default=[27, 28, 29])
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        base_cache = {}
        base_rows = score(ROOT / BASELINE, a.snaps, {})
        for s in a.snaps:
            od = OUT / BASELINE / f"snap{s:03d}"
            if (od / "figure8c_grain_size_distributions_outside_inner.npz").exists():
                base_cache[s] = shell_gsd(od)[1]
        rows = [] if all(Path(r).name == BASELINE for r in a.runs) else []
        for run in a.runs:
            if Path(run).name != BASELINE:
                rows += score(run, a.snaps, base_cache)
        # recompute baseline rows with the self-distance filled in (0)
        for x in base_rows:
            x["gsd_l1_base"] = 0.0
        write_rows(base_rows + rows)
    summary()
    print((OUT / "summary.txt").read_text() if (OUT / "summary.txt").exists() else "no summary yet")


if __name__ == "__main__":
    main()
