"""Overlay the 5-20 kpc GSD (vs HD23) and the R_V / A_V structure functions (vs ZG) for chosen C-S grid runs.
Uses the grid scoring products (fixed mock-Sun radius 2.574 kpc), averaged over snapshots 27-29.
usage: python3 scripts/cs_grid_compare.py run1 run2 ... [--out analysis/cs_grid/compare/name.png]"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "analysis" / "cs_grid"
p = argparse.ArgumentParser(); p.add_argument("runs", nargs="+"); p.add_argument("--out", default=str(G / "compare" / "compare.png"))
p.add_argument("--grid-dir", default=None, help="scoring dir, e.g. analysis/cs_grid_R3.5 (default analysis/cs_grid)")
a = p.parse_args()
if a.grid_dir: G = ROOT / a.grid_dir
RSUN = "2.57" if not a.grid_dir else a.grid_dir.split("_R")[-1]
cols = ["0.55", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
for k, run in enumerate(a.runs):
    gsd, s2, s2a = [], [], []
    for s in (27, 28, 29):
        od = G / f"halo441_output_1024_cs_{run}" / f"snap{s:03d}"
        if not od.exists():
            od = G / run / f"snap{s:03d}"
        g = np.load(od / "figure8c_grain_size_distributions_outside_inner.npz")
        gsd.append(g["cg_density_5_20"]); edges = g["cg_edges_nm"]; hd_r, hd_p = g["hd_radius_nm"], g["hd_density"]
        d = np.load(od / "paper_II_figure_data.npz")
        sep = d["sf_separation_kpc"]
        with np.errstate(all="ignore"):
            s2.append(np.nanmedian(d["cg_rv_s2"], 0)); s2a.append(np.nanmedian(d["cg_av_s2"], 0))
            zg, zga = np.nanmedian(d["zg_rv_s2"], 0), np.nanmedian(d["zg_av_s2"], 0)
    lab = run.replace("halo441_output_1024_", "")
    lab = {"br06_velfix": "Original CG run"}.get(lab, lab)
    ax[0].stairs(np.mean(gsd, 0), edges, color=cols[k], lw=2, label=lab)
    with np.errstate(all="ignore"):
        ax[1].plot(sep, np.nanmean(s2, 0), "o-", ms=3, color=cols[k], lw=2, label=lab)
        ax[2].plot(sep, np.nanmean(s2a, 0), "o-", ms=3, color=cols[k], lw=2, label=lab)
ax[0].plot(hd_r, hd_p / np.trapz(hd_p, np.log10(hd_r)), color="k", lw=2.5, ls="--", label="Hensley & Draine 2023")
ax[1].plot(sep, zg, "k--", lw=2.5, label="Zhang & Green"); ax[2].plot(sep, zga, "k--", lw=2.5, label="Zhang & Green")
ax[0].set(xscale="log", xlim=(1, 2000), xlabel="grain radius (nm)", ylabel="dust mass fraction per dex", title="GSD, 5–20 kpc")
for i, q in [(1, "R_V"), (2, "A_V")]:
    ax[i].set(yscale="log", xlim=(0.3, 4.6), xlabel="separation (kpc)", ylabel=f"$S_2({q[0]}_{q[2]})$", title=f"${q[0]}_{q[2]}$ structure function (mock Suns at {RSUN} kpc)")
for x in ax: x.legend(frameon=False, fontsize=8)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=150); print("saved", a.out)
