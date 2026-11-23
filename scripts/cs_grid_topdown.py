#!/usr/bin/env python3
"""Top-down view of the 1024 C-S grid: one point per run in (c, C/S) space, colored three ways.

  1. S2(R_V) at a chosen separation (default 1 kpc), median over the 8 mock Suns, on a diverging
     log color scale centered on the Zhang & Green value at the same separation.
  2. Median face-on R_V, diverging around the Milky Way value 3.1.
  3. Ratio of the small-grain (<30 nm) to large-grain (>30 nm) peak heights of the 5-20 kpc
     mass-weighted GSD, diverging (log) around the Hensley & Draine (2023) ratio.

Everything is averaged over snapshots 27-29 from the cs_grid_score outputs.  c100_r100 and its
duplicate are averaged into one point.

  python3 scripts/cs_grid_topdown.py [--grid-dir analysis/cs_grid] [--sep 1.0] [--out ...]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SNAPS = (27, 28, 29)
SPLIT_NM = 30.0
RV_MW = 3.1

p = argparse.ArgumentParser()
p.add_argument("--grid-dir", default="analysis/cs_grid")
p.add_argument("--sep", type=float, default=1.0, help="structure-function separation in kpc")
p.add_argument("--out", default=None)
a = p.parse_args()
grid = ROOT / a.grid_dir
out = Path(a.out) if a.out else grid / f"topdown_sep{a.sep:g}.png"
rsun = a.grid_dir.split("_R")[-1] if "_R" in a.grid_dir else "2.57"


def peak_ratio(edges, p, width_dex=0.2):
    c = np.sqrt(edges[1:] * edges[:-1])
    n = max(1, int(round(width_dex / np.median(np.diff(np.log10(edges))))))
    p = np.convolve(p, np.ones(n) / n, mode="same")   # peak heights of a ~0.2 dex running mean
    return p[c < SPLIT_NM].max() / p[c >= SPLIT_NM].max()


def s2_at(sep, s2, x):
    ok = np.isfinite(s2) & (s2 > 0)
    return 10 ** np.interp(x, sep[ok], np.log10(s2[ok]))


pts = {}
zg_vals, hd_ratio = [], None
for d in sorted(grid.glob("halo441_output_1024_cs_c*_r*")):
    m = re.search(r"_c(\d+)_r(\d+)", d.name)
    c, r = int(m.group(1)) / 100, int(m.group(2)) / 100
    s2, rv, pr = [], [], []
    for s in SNAPS:
        od = d / f"snap{s:03d}"
        if not (od / "paper_II_figure_data.npz").exists():
            continue
        f = np.load(od / "paper_II_figure_data.npz")
        sep = f["sf_separation_kpc"]
        with np.errstate(all="ignore"):
            s2.append(s2_at(sep, np.nanmedian(f["cg_rv_s2"], 0), a.sep))
            zg_vals.append(s2_at(sep, np.nanmedian(f["zg_rv_s2"], 0), a.sep))
        ok = np.isfinite(f["R_V"]) & (f["dust_count"] >= int(f["min_count"])) & (f["E_BV"] >= float(f["min_ebv_mag"]))
        rv.append(np.nanmedian(f["R_V"][ok]))
        g = np.load(od / "figure8c_grain_size_distributions_outside_inner.npz")
        pr.append(peak_ratio(g["cg_edges_nm"], g["cg_density_5_20"]))
        if hd_ratio is None:
            hr, hp = g["hd_radius_nm"], g["hd_density"]
            hd_ratio = hp[hr < SPLIT_NM].max() / hp[hr >= SPLIT_NM].max()
    if s2:
        pts.setdefault((c, r), []).append((np.mean(np.log10(s2)), np.mean(rv), np.mean(np.log10(pr))))

keys = sorted(pts)
C = np.array([k[0] for k in keys])
R = np.array([k[1] for k in keys])
V = np.array([np.mean(pts[k], axis=0) for k in keys])   # log S2, R_V, log peak ratio
zg = np.log10(np.mean(zg_vals))
hd = np.log10(hd_ratio)


def span(v, mid):
    h = max(np.max(np.abs(v - mid)), 1e-3)
    return TwoSlopeNorm(mid, mid - h, mid + h)



# Overall miss: mean of the three |log10(run / observed)| values (dex), equal weights
miss = np.column_stack([np.abs(V[:, 0] - zg), np.abs(np.log10(V[:, 1] / RV_MW)), np.abs(V[:, 2] - hd)])
total = miss.mean(axis=1)
rank = np.argsort(np.argsort(total)) + 1
best = np.argmin(total)

panels = [
    (V[:, 0], span(V[:, 0], zg), "RdBu_r", rf"$\log_{{10}} S_2(R_V)$ at {a.sep:g} kpc",
     f"$R_V$ structure function at {a.sep:g} kpc\n(white = Zhang & Green, {10**zg:.3f})",
     lambda v: f"{10**(v - zg):.1f}× ZG"),
    (V[:, 1], span(V[:, 1], RV_MW), "RdBu_r", "median face-on $R_V$",
     f"median $R_V$\n(white = Milky Way, {RV_MW})", lambda v: f"$R_V$ = {v:.2f}"),
    (V[:, 2], span(V[:, 2], hd), "RdBu_r", r"$\log_{10}$ (small / large GSD peak)",
     f"GSD small/large peak ratio, 5–20 kpc\n(white = Hensley & Draine, {hd_ratio:.2f})",
     lambda v: f"{10**v:.2f}"),
    (total, plt.Normalize(0, total.max()), "Reds", "mean |log$_{10}$(run / observed)| (dex)",
     "overall: mean miss of the three panels\n(white = perfect; number = rank)", lambda v: f"{v:.2f} dex"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
for k, (ax, (v, norm, cmap, clab, title, fmt)) in enumerate(zip(axes.flat, panels)):
    sc = ax.scatter(R, C, c=v, cmap=cmap, norm=norm, s=900, edgecolors="0.3", linewidths=0.8, zorder=2)
    ax.scatter(R[best], C[best], s=1500, facecolors="none", edgecolors="k", linewidths=2.5, zorder=1)
    for i, (x, y, val) in enumerate(zip(R, C, v)):
        ax.annotate(fmt(val), (x, y), xytext=(0, -27), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color="0.15",
                    fontweight="bold" if i == best else "normal")
        if k == 3:
            ax.annotate(f"{rank[i]}", (x, y), ha="center", va="center", fontsize=12,
                        fontweight="bold", color="w" if norm(val) > 0.6 else "0.1", zorder=3)
    ax.set(xscale="log", yscale="log", ylabel="coagulation rate $c$ (× baseline)",
           xlabel="coagulation / shattering $C/S$ (× baseline)", title=title)
    ax.set_yticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
    ax.set_xticks([0.5, 0.71, 1, 1.41], ["0.5", "0.71", "1", "1.41"])
    ax.minorticks_off()
    ax.set_ylim(0.16, 2.8)
    ax.set_xlim(0.4, 1.75)
    ax.plot(1, 1, marker="s", ms=36, mfc="none", mec="0.45", ls="none", zorder=0)
    fig.colorbar(sc, ax=ax, label=clab, shrink=0.9)
fig.suptitle(f"1024 C–S grid, z ≈ 0 (snapshots 27–29), mock Suns at {rsun} kpc\n"
             f"black ring = best overall (c = {C[best]:g}, C/S = {R[best]:g});  gray square = baseline",
             fontsize=12)
fig.savefig(out, dpi=150)
print("saved", out)
for i in np.argsort(total):
    k = keys[i]
    print(f"#{rank[i]:<2} c={k[0]:<5} r={k[1]:<5} S2/ZG={10**(V[i,0]-zg):5.2f}  R_V={V[i,1]:.2f}  "
          f"peak_ratio={10**V[i,2]:.2f}  miss={total[i]:.3f} [{miss[i,0]:.2f} {miss[i,1]:.2f} {miss[i,2]:.2f}]")
print(f"ZG S2 at {a.sep:g} kpc = {10**zg:.4f};  HD23 peak ratio = {hd_ratio:.3f}")
