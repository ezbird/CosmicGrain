#!/usr/bin/env python3
"""Resumable two-panel CosmicGrain figure: suite GSD and tracked-halo R(V).

Place beside cosmicgrain_z0_z6_gsd_rv.py in gadget4/scripts. Run from gadget4:
  python3 scripts/cosmicgrain_proposal_two_panel.py --root . \
    --cg10-output-dir halo441_output_1024_mingashsml0.0 --cg10-group 1 \
    --output cosmicgrain_proposal_two_panel.png

Every halo runs in its own process and saves a compact cache upon completion.
Re-running resumes missing halos. --force recomputes; --halos selects a subset.
The suite GSD sums dust mass inside R200c of selected zoom halos, and is not a
volume-weighted cosmic mean. R(V) is computed from face-on map pixels for a
single halo at both epochs; the default early snapshot is 008 (z about 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cosmicgrain_z0_z6_gsd_rv as cg


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--cg10-output-dir", type=Path, default=Path("halo441_output_1024_mingashsml0.0"))
    p.add_argument("--cg10-group", type=int, default=1)
    p.add_argument("--rv-halo", choices=["auto", *cg.CG_HALOS], default="CG11",
                   help="halo for the paired R(V) distributions; auto selects a dusty z=6 progenitor")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--snap-z0", type=int, default=29)
    p.add_argument("--snap-early", "--snap-z6", dest="snap_z6", type=int, default=8,
                   help="early snapshot (008 is z about 5; --snap-z6 remains an alias)")
    p.add_argument("--halos", nargs="+", choices=list(cg.CG_HALOS), default=list(cg.CG_HALOS))
    p.add_argument("--amin-nm", type=float, default=1.)
    p.add_argument("--amax-nm", type=float, default=160.)
    p.add_argument("--n-gsd-bins", type=int, default=45)
    p.add_argument("--pixel-pkpc", type=float, default=0.5)
    p.add_argument("--rv-radius-pkpc", type=float, default=20.)
    p.add_argument("--rv-zmax-pkpc", type=float, default=20.)
    p.add_argument("--min-dust-count", type=int, default=1)
    p.add_argument("--min-signal-fraction", type=float, default=1e-3)
    p.add_argument("--orient-radius-pkpc", type=float, default=20.)
    p.add_argument("--carbon-table", type=Path, default=Path("scripts/dust_optics/callqcomp.out_D16MGemt"))
    p.add_argument("--silicate-table", type=Path, default=Path("scripts/dust_optics/callqcomp.out_DH16asil"))
    p.add_argument("--style-file", type=Path, default=Path("scripts/cosmicgrain.mplstyle"),
                   help="apply existing CosmicGrain matplotlib style if found")
    p.add_argument("--show-local-references", action=argparse.BooleanOptionalAction, default=True,
                   help="show Gordon et al. 2003 LMC-average and SMC-Bar R(V) guides")
    p.add_argument("--log-gsd", action=argparse.BooleanOptionalAction, default=True,
                   help="use log y for the grain-size histogram")
    p.add_argument("--log-rv", action=argparse.BooleanOptionalAction, default=False,
                   help="use log y for the R(V) histogram (hides zero-count bins)")
    p.add_argument("--log-y", action="store_true",
                   help="legacy option: use log y on both panels")
    p.add_argument("--output", type=Path, default=Path("cosmicgrain_proposal_two_panel.png"))
    p.add_argument("--cache-dir", type=Path, default=Path("cosmicgrain_proposal_cache"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--worker", choices=list(cg.CG_HALOS), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.log_y:
        a.log_gsd = True
        a.log_rv = True
    a.root = a.root.resolve()
    a.cache_dir = a.cache_dir.resolve()
    a.output = a.output.resolve()
    if not a.cg10_output_dir.is_absolute():
        a.cg10_output_dir = a.root / a.cg10_output_dir
    a.cg10_output_dir = a.cg10_output_dir.resolve()
    for name in ("carbon_table", "silicate_table", "style_file"):
        path = getattr(a, name)
        setattr(a, name, (a.root / path).resolve() if not path.is_absolute() else path)
    if a.snap_z0 == a.snap_z6 or a.amin_nm <= 0 or a.amax_nm <= a.amin_nm or a.n_gsd_bins < 2:
        p.error("Choose distinct snapshots and increasing positive GSD bin bounds")
    return a


def outdir(a, name):
    return a.cg10_output_dir if name == "CG10" else cg.run_dir(a.root, name, a.resolution)


def cache_path(a, name):
    # Configuration changes cannot silently reuse an incompatible cached histogram.
    opts = [str(outdir(a, name)), a.snap_z0, a.snap_z6, a.cg10_group if name == "CG10" else None,
            a.amin_nm, a.amax_nm, a.n_gsd_bins, a.pixel_pkpc, a.rv_radius_pkpc,
            a.rv_zmax_pkpc, a.min_dust_count, a.min_signal_fraction,
            a.orient_radius_pkpc, str(a.carbon_table), str(a.silicate_table)]
    if a.rv_halo == name and name != "CG10":
        opts.append("compute_rv_for_this_halo")
    digest = hashlib.sha256(json.dumps(opts).encode()).hexdigest()[:12]
    return a.cache_dir / f"{name}_{digest}.npz"


def mass_hist(a, directory, snap, halo, edges):
    """Accumulate one particle block at a time; do not retain the full snapshot."""
    scale, h, box = cg.read_header(directory, snap)
    histogram = np.zeros(len(edges) - 1, dtype=float)
    count = 0
    total_mass = 0.
    volume_count = 0
    volume_mass = 0.
    nearest_pkpc = float("inf")
    within_2r = 0
    within_5r = 0
    for filename in cg.expand_snapshot(directory, snap):
        with h5py.File(filename) as f:
            if "PartType6" not in f:
                continue
            dust = f["PartType6"]
            for key in ("Coordinates", "Masses", "GrainRadius"):
                if key not in dust:
                    raise KeyError(f"{filename}: missing PartType6/{key}")
            n = len(dust["Masses"])
            for lo in range(0, n, 100_000):
                hi = min(n, lo + 100_000)
                pos = np.asarray(dust["Coordinates"][lo:hi])
                mass = np.asarray(dust["Masses"][lo:hi], dtype=float) * (1e10 / h)
                radii = np.asarray(dust["GrainRadius"][lo:hi], dtype=float)
                near, distances = cg.halo_mask(pos, halo, scale, h, box)
                if len(distances):
                    nearest_pkpc = min(nearest_pkpc, float(np.min(distances)))
                within_2r += int(np.count_nonzero(distances < 2 * halo["r200_pkpc"]))
                within_5r += int(np.count_nonzero(distances < 5 * halo["r200_pkpc"]))
                volume_count += hi - lo
                volume_mass += float(np.sum(mass))
                good = near & np.isfinite(mass) & (mass > 0) & np.isfinite(radii) & (radii > 0)
                count += int(good.sum())
                total_mass += float(mass[good].sum())
                histogram += np.histogram(radii[good], bins=edges, weights=mass[good])[0]
    diagnostic = dict(volume_particles=volume_count, volume_mass_msun=volume_mass,
                      nearest_dust_pkpc=nearest_pkpc if np.isfinite(nearest_pkpc) else None,
                      dust_within_2r200=within_2r, dust_within_5r200=within_5r)
    return histogram, count, total_mass, 1. / scale - 1., diagnostic


def worker(a, name):
    directory = outdir(a, name)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    group = a.cg10_group if name == "CG10" else cg.find_audit_group(directory)
    print(f"{name}: {directory}; final group={group}", flush=True)
    series = cg.get_zoom_halo_series(directory, [a.snap_z6, a.snap_z0],
                                    group_index=group, reference_snap=a.snap_z0,
                                    verbose=True, refine_reference_center=True,
                                    refine_historical_centers=False)
    edges = np.geomspace(a.amin_nm, a.amax_nm, a.n_gsd_bins + 1)
    results = {}
    metadata = {"halo": name, "output_dir": str(directory), "final_group": group, "epochs": {}}
    for tag, snap in (("z0", a.snap_z0), ("z6", a.snap_z6)):
        if snap not in series:
            raise RuntimeError(f"No tracked progenitor at snapshot {snap}")
        hist, count, mass, redshift, diagnostic = mass_hist(a, directory, snap, series[snap], edges)
        results[f"hist_{tag}"] = hist
        metadata["epochs"][tag] = dict(snapshot=snap, redshift=redshift,
                                        dust_particles=count, dust_mass_msun=mass,
                                        group=series[snap].get("group_idx"),
                                        r200_pkpc=float(series[snap]["r200_pkpc"]))
        metadata["epochs"][tag].update(diagnostic)
        print(f"{name} snap {snap:03d}: z={redshift:.3f}, Ndust={count:,}, Mdust={mass:.3e} Msun", flush=True)
        if name == "CG10" and tag == "z6":
            print(f"CG10 z=6 spatial check: whole volume={diagnostic['volume_particles']:,}; "
                  f"inside R200={count:,}; inside 2R200={diagnostic['dust_within_2r200']:,}; "
                  f"inside 5R200={diagnostic['dust_within_5r200']:,}; "
                  f"nearest dust={diagnostic['nearest_dust_pkpc']} pkpc; "
                  f"R200={series[snap]['r200_pkpc']:.2f} pkpc", flush=True)
        if name == a.rv_halo or (a.rv_halo == "auto" and name == "CG10"):
            if count == 0:
                results[f"rv_{tag}"] = np.empty(0)
                metadata["epochs"][tag]["rv_status"] = "No dust in R200c; R(V) undefined"
            else:
                carb = cg.read_draine_qext_table(a.carbon_table)
                sil = cg.read_draine_qext_table(a.silicate_table)
                try:
                    rv, info = cg.rv_samples(directory, snap, series[snap], carb, sil,
                                             a.pixel_pkpc, a.rv_radius_pkpc, a.rv_zmax_pkpc,
                                             a.min_dust_count, a.min_signal_fraction,
                                             a.orient_radius_pkpc)
                    results[f"rv_{tag}"] = rv
                    metadata["epochs"][tag]["rv_status"] = f"{len(rv)} valid pixels"
                    metadata["epochs"][tag]["orientation"] = info["orientation_source"]
                except (RuntimeError, ValueError) as exc:
                    results[f"rv_{tag}"] = np.empty(0)
                    metadata["epochs"][tag]["rv_status"] = str(exc)
                    print(f"{name} snap {snap:03d}: R(V) unavailable: {exc}", flush=True)
    dest = cache_path(a, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **results, metadata=json.dumps(metadata))
    os.replace(tmp, dest)
    print(f"Saved {dest}", flush=True)


def make_figure(a, cached, rv_halo):
    if a.style_file.is_file():
        plt.style.use(str(a.style_file))
        print(f"Using figure style: {a.style_file}", flush=True)
    records = []
    gsd = {"z0": np.zeros(a.n_gsd_bins), "z6": np.zeros(a.n_gsd_bins)}
    chosen_rv = {}
    chosen_epoch = {}
    for name, filename in cached:
        with np.load(filename, allow_pickle=False) as data:
            record = json.loads(str(data["metadata"]))
            records.append(record)
            for tag in gsd:
                gsd[tag] += data[f"hist_{tag}"]
                if name == rv_halo and f"rv_{tag}" in data:
                    chosen_rv[tag] = data[f"rv_{tag}"].copy()
                    chosen_epoch[tag] = record["epochs"][tag]
    if not records:
        raise RuntimeError("No successful halo results; see worker logs above")
    edges = np.geomspace(a.amin_nm, a.amax_nm, a.n_gsd_bins + 1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = {"z0": "#7030A0", "z6": "#16834B"}
    for tag in ("z0", "z6"):
        total = gsd[tag].sum()
        z = float(np.median([r["epochs"][tag]["redshift"] for r in records]))
        label = f"z={z:.2f} ({len(records)} halos)"
        if total > 0:
            y = 100. * gsd[tag] / total
            if a.log_gsd:
                ax.stairs(np.where(y > 0, y, np.nan), edges,
                          color=colors[tag], lw=2, label=label)
            else:
                ax.stairs(y, edges, color=colors[tag], lw=2, label=label)
        else:
            ax.plot([], [], color=colors[tag], lw=2, label=f"{label}: no dust")
            ax.text(.04, .90 if tag == "z6" else .78, f"{label}: no dust in selected halos",
                    color=colors[tag], transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Grain radius (nm)")
    ax.set_ylabel("Dust mass per size bin (%)")
    if a.log_gsd:
        ax.set_yscale("log")
        ax.set_ylim(.03, 100.)
    ax.set_title("Dust grain sizes across zoom halos")
    ax.legend(loc="upper left", fontsize=8)
    if rv_halo in [r["halo"] for r in records] and "z0" in chosen_rv and "z6" in chosen_rv:
        rv0 = chosen_rv["z0"]
        rv6 = chosen_rv["z6"]
        all_rv = np.r_[rv0, rv6]
        rv_edges = np.linspace(1.5, 20., 61)
        rv_centers = .5 * (rv_edges[1:] + rv_edges[:-1])
        tail = bx.inset_axes([.59, .63, .38, .28])
        tail_max = 0.
        for tag, rv in (("z0", rv0), ("z6", rv6)):
            if len(rv):
                counts, _ = np.histogram(rv, bins=rv_edges)
                fraction = 100. * counts / len(rv)
                z_label = f"z={chosen_epoch[tag]['redshift']:.2f}"
                shown = np.where(fraction > 0, fraction, np.nan) if a.log_rv else fraction
                bx.plot(rv_centers, shown, "o-", color=colors[tag], lw=1.6,
                        ms=3.3, label=f"{z_label} ({len(rv)} valid pixels)")
                tail.plot(rv_centers, shown, "o-", color=colors[tag], lw=1.3, ms=2)
                tail_max = max(tail_max, float(np.max(fraction[rv_centers >= 5.5])))
            else:
                status = next(r for r in records if r["halo"] == rv_halo)["epochs"][tag]["rv_status"]
                bx.plot([], [], color=colors[tag], lw=2, label=f"z={chosen_epoch[tag]['redshift']:.2f}: unavailable")
                bx.text(.04, .90 if tag == "z6" else .78, f"z={chosen_epoch[tag]['redshift']:.2f}: {status}",
                        transform=bx.transAxes, color=colors[tag], va="top", fontsize=9, wrap=True)
        bx.set_xlim(1.5, 5.5)
        tail.set_xlim(5.5, 20.)
        if a.log_rv:
            bx.set_yscale("log")
            bx.set_ylim(.03, 100.)
            tail.set_yscale("log")
            tail.set_ylim(.03, max(.2, 1.25 * tail_max))
        else:
            tail.set_ylim(0, max(.2, 1.25 * tail_max))
        tail.set_xticks([6, 10, 15, 20])
        tail.tick_params(labelsize=7)
        tail.set_title(r"Tail to $R_V=20$", fontsize=8)
        if a.show_local_references:
            # Gordon et al. (2003), LMC-average and SMC-Bar stellar sightlines.
            for value, label in ((2.74, "SMC Bar"), (3.41, "LMC avg.")):
                bx.axvline(value, ls="--", lw=.9, color="0.48", alpha=.75, zorder=0)
                bx.text(value+.025, .76, label, transform=bx.get_xaxis_transform(),
                        fontsize=8, color="0.38", rotation=90, va="top")
        bx.set_xlabel(r"$R_V=A_V/(A_B-A_V)$")
        bx.set_ylabel(r"Sightlines per $R_V$ bin (%)")
        bx.legend(loc="upper left", fontsize=8)
    else:
        bx.text(.5, .5, "No halo with usable R(V) at both epochs", ha="center", va="center", transform=bx.transAxes)
    if rv_halo and "z6" in chosen_epoch:
        early_z = chosen_epoch["z6"]["redshift"]
        bx.set_title(
            f"{rv_halo}: $z=0$ halo and $z\\approx{early_z:.0f}$ progenitor"
        )
    else:
        bx.set_title("Halo and its progenitor")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220)
    fig.savefig(a.output.with_suffix(".pdf"))
    plt.close(fig)
    summary = {"definition": "Dust-mass-weighted suite sum inside each tracked R200c; not a cosmic mean",
               "reference_rv": ({"SMC_Bar": 2.74, "LMC_average": 3.41,
                                  "source": "Gordon et al. (2003), doi:10.1086/376774"}
                                 if a.show_local_references else None),
               "configuration": {"cg10_output_dir": str(a.cg10_output_dir), "snap_z0": a.snap_z0,
                                 "snap_early": a.snap_z6, "halos_requested": a.halos,
                                 "rv_halo": rv_halo, "log_gsd": a.log_gsd,
                                 "log_rv": a.log_rv},
               "halos_completed": records,
               "halos_missing": [n for n in a.halos if n not in {r["halo"] for r in records}],
               "gsd_mass_msun_in_plot_bins": {k: float(v.sum()) for k, v in gsd.items()}}
    a.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    products = dict(grain_edges_nm=edges, snapshot_early=a.snap_z6,
                    mass_hist_z0_msun=gsd["z0"], mass_hist_early_msun=gsd["z6"],
                    rv_z0=chosen_rv.get("z0", np.empty(0)),
                    rv_early=chosen_rv.get("z6", np.empty(0)))
    if a.snap_z6 == 7:
        products["mass_hist_z6_msun"] = gsd["z6"]
        products["rv_z6"] = products["rv_early"]
    np.savez_compressed(a.output.with_suffix(".npz"), **products)
    print(f"Wrote {a.output}, {a.output.with_suffix('.pdf')}, .json and .npz", flush=True)
    if summary["halos_missing"]:
        print(f"INCOMPLETE suite: {', '.join(summary['halos_missing'])}; rerun to resume", flush=True)


def main():
    a = arguments()
    if a.worker:
        worker(a, a.worker)
        return
    cached = []
    killed = None
    for name in a.halos:
        dest = cache_path(a, name)
        if a.force or not dest.exists():
            cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", name]
            print(f"Running {name} in a separate process", flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode:
                print(f"{name} failed (exit {result.returncode}); its cache remains absent", file=sys.stderr, flush=True)
                if result.returncode == -9 or result.returncode == 137:
                    killed = name
        if dest.exists():
            cached.append((name, dest))
            print(f"Using {name}: {dest}", flush=True)
        if killed:
            print(f"Stopping after {killed} was killed; no more halo trackers will launch. "
                  "Previously completed halos remain cached.", file=sys.stderr, flush=True)
            break
    selected = a.rv_halo
    if selected == "auto":
        ranked = []
        for name, dest in cached:
            with np.load(dest, allow_pickle=False) as data:
                record = json.loads(str(data["metadata"]))
                ranked.append((int(record["epochs"]["z6"]["dust_particles"]), name))
        selected = None
        for n, name in sorted(ranked, reverse=True):
            if n < 10:
                continue
            if name == "CG10":
                candidate = cache_path(a, name)
            else:
                cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                       "--rv-halo", name, "--worker", name]
                candidate_args = argparse.Namespace(**vars(a))
                candidate_args.rv_halo = name
                candidate = cache_path(candidate_args, name)
                if not candidate.exists():
                    print(f"Computing R(V) for dusty z=6 progenitor {name} ({n} particles)", flush=True)
                    result = subprocess.run(cmd, check=False)
                    if result.returncode:
                        print(f"{name} R(V) failed (exit {result.returncode})", file=sys.stderr, flush=True)
                        if result.returncode in (-9, 137):
                            break
                        continue
            if not candidate.exists():
                continue
            with np.load(candidate, allow_pickle=False) as data:
                if "rv_z6" not in data or "rv_z0" not in data or not len(data["rv_z6"]) or not len(data["rv_z0"]):
                    continue
            cached = [(key, candidate if key == name else path) for key, path in cached]
            selected = name
            break
        if selected is None:
            print("No halo has usable R(V) at both epochs; inspect the cached dust counts", file=sys.stderr, flush=True)
    make_figure(a, cached, selected)


if __name__ == "__main__":
    main()
