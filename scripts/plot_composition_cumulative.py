"""
plot_composition_cumulative.py
--------------------------------------------------------------------------
Cumulative CarbonMassFraction diagnostics for CosmicGrain dust.

For each stellar dust source (SNII, AGB, LRN), this script measures how much
of the surviving dust population lies below a given CarbonMassFraction.

It produces:
  1. Cumulative surviving DUST MASS fraction vs CarbonMassFraction.
  2. Cumulative surviving PARTICLE fraction vs CarbonMassFraction.

It also prints threshold statistics for CF < 0.50, 0.30, 0.10, 0.03, 0.01.

Usage:
    python plot_composition_cumulative.py ../S10_output_1024
    python plot_composition_cumulative.py ../S10_output_1024 --snap 47
    python plot_composition_cumulative.py ../S10_output_1024 \
        --output cf_cumulative.png

Compare runs:
    python plot_composition_cumulative.py \
        ../S10_output_512 ../S10_output_1024 \
        --labels 512^3 1024^3 \
        --output cf_cumulative_compare.png
--------------------------------------------------------------------------
"""

import os
import re
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

DESTROYED_MASS_FLOOR = 1e-20
SOLAR_MASS_G = 1.989e33

SOURCE_INFO = {
    0: ("SNII", 0.1),
    1: ("AGB",  0.6),
    2: ("LRN",  0.1),
}

DEFAULT_THRESHOLDS = [0.50, 0.30, 0.10, 0.03, 0.01]


def snapshot_number_from_path(path):
    base = os.path.basename(path)
    m = re.search(r"snapshot_(\d+)", base)
    if not m:
        raise ValueError(f"Could not determine snapshot number from {path}")
    return int(m.group(1))


def find_snapshot_files(output_dir, snap_num=None):
    if snap_num is not None:
        tag = f"{snap_num:03d}"
        files = sorted(glob.glob(
            os.path.join(output_dir, f"snapdir_{tag}", f"snapshot_{tag}.*.hdf5")
        ))
        if not files:
            files = sorted(glob.glob(
                os.path.join(output_dir, f"snapshot_{tag}.hdf5")
            ))
        if not files:
            raise FileNotFoundError(f"No snapshot {tag} found under {output_dir}")
        return files, snap_num

    files = sorted(glob.glob(
        os.path.join(output_dir, "snapdir_*", "snapshot_*.hdf5")
    ))
    if not files:
        files = sorted(glob.glob(
            os.path.join(output_dir, "snapshot_*.hdf5")
        ))
    if not files:
        raise FileNotFoundError(f"No snapshots found under {output_dir}")

    latest = max(snapshot_number_from_path(p) for p in files)
    return [p for p in files if snapshot_number_from_path(p) == latest], latest


def load_dust(snapshot_files):
    all_mass, all_source, all_cf, all_radius = [], [], [], []
    unit_mass_g = hubble_param = redshift = None

    for path in snapshot_files:
        with h5py.File(path, "r") as f:
            if unit_mass_g is None:
                params = f["Parameters"].attrs
                unit_mass_g = float(params["UnitMass_in_g"])
                hubble_param = float(params["HubbleParam"])
                if "Redshift" in f["Header"].attrs:
                    redshift = float(f["Header"].attrs["Redshift"])

            if "PartType6" not in f:
                continue

            pt6 = f["PartType6"]
            needed = ["Masses", "DustSource", "CarbonMassFraction", "GrainRadius"]
            missing = [x for x in needed if x not in pt6]
            if missing:
                print(f"WARNING: {path} missing {missing}; skipping")
                continue

            all_mass.append(pt6["Masses"][:])
            all_source.append(pt6["DustSource"][:])
            all_cf.append(pt6["CarbonMassFraction"][:])
            all_radius.append(pt6["GrainRadius"][:])

    if not all_mass:
        return np.array([]), np.array([]), np.array([]), np.array([]), redshift

    mass = np.concatenate(all_mass)
    source = np.concatenate(all_source)
    cf = np.concatenate(all_cf)
    radius = np.concatenate(all_radius)

    mass_msun = mass * unit_mass_g / hubble_param / SOLAR_MASS_G
    floor_msun = DESTROYED_MASS_FLOOR * unit_mass_g / hubble_param / SOLAR_MASS_G

    alive = (
        (mass_msun > floor_msun)
        & np.isfinite(mass_msun)
        & np.isfinite(cf)
        & np.isfinite(radius)
        & (radius > 0.0)
        & (cf >= 0.0)
        & (cf <= 1.0)
    )

    return mass_msun[alive], source[alive], cf[alive], radius[alive], redshift


def cumulative_distribution(cf, mass):
    order = np.argsort(cf)
    cf_sorted = cf[order]
    mass_sorted = mass[order]

    particle_fraction = np.arange(1, len(cf_sorted) + 1) / len(cf_sorted)
    mass_fraction = np.cumsum(mass_sorted) / mass_sorted.sum()

    return cf_sorted, particle_fraction, mass_fraction


def threshold_stats(cf, mass, threshold):
    mask = cf < threshold
    n_below = int(mask.sum())
    pfrac = n_below / len(cf) if len(cf) else np.nan
    mbelow = mass[mask].sum()
    mfrac = mbelow / mass.sum() if mass.sum() > 0 else np.nan
    return n_below, pfrac, mbelow, mfrac


def analyze_run(output_dir, snap_num=None):
    files, snap_num = find_snapshot_files(output_dir, snap_num)
    mass, source, cf, radius, z = load_dust(files)

    if len(mass) == 0:
        raise RuntimeError(f"No live PartType6 particles found in {output_dir}")

    return {
        "output_dir": output_dir,
        "snap": snap_num,
        "z": z,
        "mass": mass,
        "source": source,
        "cf": cf,
        "radius": radius,
    }


def print_stats(run, label, thresholds):
    ztxt = f"z={run['z']:.3f}" if run["z"] is not None else f"snap={run['snap']:03d}"

    print("\n" + "=" * 80)
    print(f"{label}: snapshot {run['snap']:03d}, {ztxt}")
    print("=" * 80)

    for src_id, (name, birth_cf) in SOURCE_INFO.items():
        sel = run["source"] == src_id
        if not np.any(sel):
            continue

        cf = run["cf"][sel]
        mass = run["mass"][sel]
        radius = run["radius"][sel]

        print(f"\n[{name}] N={len(cf)}  M={mass.sum():.4e} Msun  birth_CF={birth_cf:.3f}")
        print(f"  median_CF={np.median(cf):.5f}")
        print(f"  mass-weighted_CF={np.average(cf, weights=mass):.5f}")
        print(f"  mass-weighted_radius={np.average(radius, weights=mass):.2f} nm")
        print("  threshold    N below   particle frac   mass below [Msun]   mass frac")

        for t in thresholds:
            n, pf, mb, mf = threshold_stats(cf, mass, t)
            print(f"  CF < {t:4.2f}   {n:8d}     {100*pf:8.3f}%     {mb:12.4e}     {100*mf:8.3f}%")


def plot_cumulative(runs, labels, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col", sharey="row")
    linestyles = ["-", "--", "-.", ":"]

    for col, (src_id, (name, birth_cf)) in enumerate(SOURCE_INFO.items()):
        axm = axes[0, col]
        axn = axes[1, col]

        for j, (run, label) in enumerate(zip(runs, labels)):
            sel = run["source"] == src_id
            if not np.any(sel):
                continue

            cf = run["cf"][sel]
            mass = run["mass"][sel]
            xs, npfrac, mfrac = cumulative_distribution(cf, mass)

            ls = linestyles[j % len(linestyles)]
            axm.plot(xs, mfrac, linestyle=ls, linewidth=1.8, label=label)
            axn.plot(xs, npfrac, linestyle=ls, linewidth=1.8, label=label)

        axm.axvline(birth_cf, linestyle=":", linewidth=1.2, label=f"birth CF={birth_cf}")
        axn.axvline(birth_cf, linestyle=":", linewidth=1.2)

        axm.set_title(name)
        axm.set_xlim(0, 1)
        axn.set_xlim(0, 1)
        axm.set_ylim(0, 1.02)
        axn.set_ylim(0, 1.02)
        axm.grid(alpha=0.2)
        axn.grid(alpha=0.2)
        axn.set_xlabel("CarbonMassFraction")
        axm.legend(fontsize=8)

    axes[0, 0].set_ylabel("Cumulative surviving dust mass fraction")
    axes[1, 0].set_ylabel("Cumulative surviving particle fraction")

    if len(runs) == 1 and runs[0]["z"] is not None:
        suffix = f" -- z={runs[0]['z']:.3f}"
    else:
        suffix = ""

    fig.suptitle("CosmicGrain cumulative CarbonMassFraction distributions" + suffix)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"\nWrote {output_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dirs", nargs="+")
    p.add_argument("--snap", type=int, default=None)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--output", default="cf_cumulative.png")
    args = p.parse_args()

    if args.labels is not None and len(args.labels) != len(args.output_dirs):
        p.error("--labels must contain one label per output directory")

    labels = args.labels or [
        os.path.basename(os.path.normpath(d)) for d in args.output_dirs
    ]

    runs = []
    for d, label in zip(args.output_dirs, labels):
        print(f"Loading {label}: {d}")
        run = analyze_run(d, args.snap)
        runs.append(run)
        print_stats(run, label, args.thresholds)

    plot_cumulative(runs, labels, args.output)


if __name__ == "__main__":
    main()
