#!/usr/bin/env python3
"""Plot R(V) mean and median versus redshift for the CosmicGrain zoom suite.

Install beside cosmicgrain_z0_z6_gsd_rv.py and the memory-safe halo_utils.py.
Run from gadget4 with --root . and --cg10-output-dir ...mingashsml0.0.
Each halo tracks from the final anchor to the earliest requested snapshot in a
separate process. Its compact result is saved on completion, so reruns resume.
The z=0 M200c determines each halo's fixed color at every redshift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import cosmicgrain_z0_z6_gsd_rv as cg


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--cg10-output-dir", type=Path, default=Path("halo441_output_1024_mingashsml0.0"))
    p.add_argument("--cg10-group", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--snapshots", type=int, nargs="+", default=[8, 9, 11, 13, 16, 20, 24, 29],
                   help="epochs to plot; reference snapshot must be present")
    p.add_argument("--reference-snap", type=int, default=29)
    p.add_argument("--halos", nargs="+", choices=list(cg.CG_HALOS), default=list(cg.CG_HALOS))
    p.add_argument("--pixel-pkpc", type=float, default=.5)
    p.add_argument("--rv-radius-pkpc", type=float, default=20.)
    p.add_argument("--rv-zmax-pkpc", type=float, default=20.)
    p.add_argument("--min-dust-count", type=int, default=1)
    p.add_argument("--min-signal-fraction", type=float, default=1e-3)
    p.add_argument("--orient-radius-pkpc", type=float, default=20.)
    p.add_argument("--min-pixels", type=int, default=10,
                   help="minimum valid map pixels required for an RV point")
    p.add_argument("--plot-min-pixels", type=int, default=None,
                   help="optional stricter display threshold; does not invalidate cached measurements")
    p.add_argument("--rv-min", type=float, default=1.5)
    p.add_argument("--rv-max", type=float, default=20.)
    p.add_argument("--carbon-table", type=Path, default=Path("scripts/dust_optics/callqcomp.out_D16MGemt"))
    p.add_argument("--silicate-table", type=Path, default=Path("scripts/dust_optics/callqcomp.out_DH16asil"))
    p.add_argument("--cache-dir", type=Path, default=Path("cosmicgrain_rv_evolution_cache"))
    p.add_argument("--style-file", type=Path, default=Path("scripts/cosmicgrain.mplstyle"))
    p.add_argument("--layout", choices=("stacked", "shared-y"), default="stacked",
                   help="stacked shared-redshift panels or side-by-side panels on a common R(V) scale")
    p.add_argument("--output", type=Path, default=Path("cosmicgrain_rv_evolution.png"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--worker", choices=list(cg.CG_HALOS), help=argparse.SUPPRESS)
    a = p.parse_args()
    a.root = a.root.resolve()
    a.output = a.output.resolve()
    a.cache_dir = a.cache_dir.resolve()
    a.cg10_output_dir = (a.root / a.cg10_output_dir).resolve()
    a.style_file = (a.root / a.style_file).resolve() if not a.style_file.is_absolute() else a.style_file
    for key in ("carbon_table", "silicate_table"):
        v = getattr(a, key)
        setattr(a, key, (a.root / v).resolve() if not v.is_absolute() else v)
    a.snapshots = sorted(set(a.snapshots))
    if a.reference_snap not in a.snapshots or max(a.snapshots) != a.reference_snap:
        p.error("--snapshots must include --reference-snap as their latest snapshot")
    if a.min_pixels < 1 or a.rv_min <= 0 or a.rv_max <= a.rv_min:
        p.error("Require positive --min-pixels and increasing R(V) bounds")
    if a.plot_min_pixels is not None and a.plot_min_pixels < 1:
        p.error("--plot-min-pixels must be positive")
    return a


def directory(a, name):
    return a.cg10_output_dir if name == "CG10" else cg.run_dir(a.root, name, a.resolution)


def cache_path(a, name):
    args = (str(directory(a, name)), name, a.cg10_group if name == "CG10" else None,
            a.snapshots, a.pixel_pkpc, a.rv_radius_pkpc, a.rv_zmax_pkpc,
            a.min_dust_count, a.min_signal_fraction, a.orient_radius_pkpc,
            a.min_pixels, a.rv_min, a.rv_max, str(a.carbon_table), str(a.silicate_table))
    fingerprint = hashlib.sha256(json.dumps(args).encode()).hexdigest()[:12]
    return a.cache_dir / f"{name}_{fingerprint}.npz"


def worker(a, name):
    out = directory(a, name)
    if not out.is_dir():
        raise FileNotFoundError(out)
    group = a.cg10_group if name == "CG10" else cg.find_audit_group(out)
    print(f"{name}: {out}; final group={group}", flush=True)
    series = cg.get_zoom_halo_series(out, a.snapshots, group_index=group,
                                    reference_snap=a.reference_snap,
                                    verbose=True, refine_reference_center=True,
                                    refine_historical_centers=False)
    if series.get(a.reference_snap) is None:
        raise RuntimeError(f"No anchored final halo for {name}")
    final_mass = float(series[a.reference_snap]["m200_msun"])
    carbon = cg.read_draine_qext_table(a.carbon_table)
    silicate = cg.read_draine_qext_table(a.silicate_table)
    rows = []
    for snap in a.snapshots:
        scale, _, _ = cg.read_header(out, snap)
        z = 1. / scale - 1.
        halo = series.get(snap)
        row = dict(halo=name, snapshot=snap, redshift=z, halo_mass_z0_msun=final_mass,
                   group=int(halo["group_idx"]) if halo is not None else None,
                   r200_pkpc=float(halo["r200_pkpc"]) if halo is not None else None,
                   n_valid_pixels=0, n_pixels_in_range=0,
                   mean_rv=None, median_rv=None, p16_rv=None, p84_rv=None,
                   fraction_outside_range=None, orientation=None,
                   status="no tracked halo" if halo is None else "not computed")
        if halo is not None:
            try:
                rv, info = cg.rv_samples(out, snap, halo, carbon, silicate,
                                         a.pixel_pkpc, a.rv_radius_pkpc,
                                         a.rv_zmax_pkpc, a.min_dust_count,
                                         a.min_signal_fraction, a.orient_radius_pkpc)
                row["n_valid_pixels"] = len(rv)
                row["orientation"] = info["orientation_source"]
                values = rv[np.isfinite(rv) & (rv >= a.rv_min) & (rv <= a.rv_max)]
                row["n_pixels_in_range"] = len(values)
                row["fraction_outside_range"] = 1. - len(values) / len(rv) if len(rv) else None
                if len(values) >= a.min_pixels:
                    row["mean_rv"] = float(np.mean(values))
                    row["median_rv"] = float(np.median(values))
                    row["p16_rv"], row["p84_rv"] = [float(x) for x in np.percentile(values, [16, 84])]
                    row["status"] = "ok"
                else:
                    row["status"] = f"only {len(values)} pixels within R(V) range"
            except (RuntimeError, ValueError, KeyError) as exc:
                row["status"] = str(exc)
        rows.append(row)
        print(f"{name} snap {snap:03d}, z={z:.2f}: {row['status']}, "
              f"Npix={row['n_pixels_in_range']}, mean={row['mean_rv']}, "
              f"median={row['median_rv']}", flush=True)
    dest = cache_path(a, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(temp, rows=json.dumps(rows))
    os.replace(temp, dest)
    print(f"Saved {dest}", flush=True)


def plot(a, cached):
    if a.style_file.is_file():
        plt.style.use(str(a.style_file))
        print(f"Using figure style: {a.style_file}", flush=True)
    rows = []
    for name, fn in cached:
        with np.load(fn, allow_pickle=False) as data:
            rows.extend(json.loads(str(data["rows"])))
    display_min = max(a.min_pixels, a.plot_min_pixels or a.min_pixels)
    good = [r for r in rows if r["status"] == "ok" and r["n_pixels_in_range"] >= display_min]
    if not good:
        raise RuntimeError("No usable R(V) points; inspect the per-halo worker logs")
    focused = len(a.halos) <= 4
    if not focused:
        masses = [r["halo_mass_z0_msun"] for r in rows]
        cmap = plt.get_cmap("viridis")
        vmin, vmax = min(masses), max(masses)
        norm = LogNorm(vmin=vmin * .98, vmax=vmax * 1.02)
    focus_colors = ("#16834B", "#7030A0", "#D57915", "#2866A6")
    side_by_side = a.layout == "shared-y"
    fig, axes = (plt.subplots(1, 2, figsize=(13., 5.5), sharey=True,
                             constrained_layout=True) if side_by_side else
                 plt.subplots(2, 1, figsize=(8.8, 7.8) if focused else (9.5, 8.),
                              sharex=True, constrained_layout=True))
    for index, name in enumerate(a.halos):
        points = sorted((r for r in good if r["halo"] == name), key=lambda r: r["redshift"])
        if not points:
            continue
        color = (focus_colors[index] if focused else
                 cmap(norm(points[0]["halo_mass_z0_msun"])))
        mantissa, exponent = f"{points[0]['halo_mass_z0_msun']:.1e}".split("e")
        halo_label = (f"{name}  ($M_{{200c}}={mantissa}\\times"
                      f"10^{{{int(exponent)}}}\\,M_\\odot$)")
        xx = np.asarray([r["redshift"] for r in points])
        sizes = np.asarray([r["n_pixels_in_range"] for r in points])
        marker_sizes = np.clip(10 + 7 * np.sqrt(sizes), 14, 75)
        for ax, key in zip(axes, ("mean_rv", "median_rv")):
            yy = np.asarray([r[key] for r in points])
            ax.plot(xx, yy, color=color, lw=2.0 if focused else 1.4,
                    alpha=1.0 if focused else .8,
                    label=halo_label if focused else None)
            ax.scatter(xx, yy, c=[color], s=marker_sizes,
                       edgecolors="white", linewidths=.35, zorder=3)
    for ax, title in zip(axes, ("Means", "Medians")):
        ax.text(.5, .96, title, transform=ax.transAxes, ha="center", va="top",
                fontsize=12, bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
        ax.grid(alpha=.16)
        ax.set_xlim(max(r["redshift"] for r in rows) + .2, -.1)
    axes[0].set_ylabel(r"$R_V$")
    if side_by_side:
        for ax in axes:
            ax.set_xlabel("Redshift")
    else:
        axes[1].set_ylabel(r"$R_V$")
        axes[1].set_xlabel("Redshift")
    if focused:
        legend_ax = axes[0] if side_by_side else axes[1]
        legend_ax.legend(loc="lower left" if side_by_side else "upper left",
                         fontsize=10, framealpha=.9)
    else:
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                            shrink=.82, pad=.02)
        cbar.set_label(r"Halo $M_{200c}$ at $z=0$ ($M_\odot$)")
    fig.suptitle("CosmicGrain extinction-curve evolution", fontsize=14)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220, bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with a.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {a.output}, PDF, and CSV ({len(good)} valid halo-epoch points)", flush=True)
    missing = sorted(set(a.halos) - {name for name, _ in cached})
    if missing:
        print("INCOMPLETE suite; rerun to resume: " + ", ".join(missing), flush=True)


def main():
    a = arguments()
    if a.worker:
        worker(a, a.worker)
        return
    cached = []
    for name in a.halos:
        dest = cache_path(a, name)
        if a.force or not dest.exists():
            print(f"Running {name} in its own process", flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     *sys.argv[1:], "--worker", name], check=False)
            if result.returncode:
                print(f"{name}: failed (exit {result.returncode})", file=sys.stderr, flush=True)
                if result.returncode in (-9, 137):
                    print("Stopping after RAM kill; prior halos remain cached", file=sys.stderr)
                    break
        if dest.exists():
            cached.append((name, dest))
    plot(a, cached)


if __name__ == "__main__":
    main()
