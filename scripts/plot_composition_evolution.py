"""
plot_composition_evolution.py
--------------------------------------------------------------------------
CosmicGrain composition-evolution diagnostics.

Produces two figures:

1) Delta CarbonMassFraction vs grain radius for one snapshot:
       dCF = CarbonMassFraction - birth CarbonMassFraction
   separated by DustSource (SNII, AGB, LRN).

2) Mass-weighted mean CarbonMassFraction vs redshift across all snapshots,
   again separated by DustSource.

The loading logic follows plot_composition_histogram.py:
- supports split or single-file snapshots
- converts PartType6 mass to Msun
- removes destroyed / negligible dust particles
- uses DustSource birth compositions:
      SNII = 0.1
      AGB  = 0.6
      LRN  = 0.1

Usage:
    python plot_composition_evolution.py S10_output_512

    python plot_composition_evolution.py S10_output_512 \
        --snap 47 \
        --scatter-output cf_delta_vs_radius.png \
        --history-output cf_vs_redshift.png
--------------------------------------------------------------------------
"""

import sys
import glob
import os
import re
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

DESTROYED_MASS_FLOOR = 1e-20
SOLAR_MASS_G = 1.989e33

SOURCE_INFO = {
    0: ("SNII", 0.1, "tab:red"),
    1: ("AGB",  0.6, "tab:orange"),
    2: ("LRN",  0.1, "tab:blue"),
}


def snapshot_number_from_path(path):
    """Extract snapshot number from snapshot_047.0.hdf5 or snapshot_047.hdf5."""
    base = os.path.basename(path)
    m = re.search(r"snapshot_(\d+)", base)
    if not m:
        raise ValueError(f"Could not determine snapshot number from: {path}")
    return int(m.group(1))


def find_snapshot_files(output_dir, snap_num=None):
    """Find a requested snapshot or the highest-numbered snapshot available."""
    if snap_num is not None:
        tag = f"{snap_num:03d}"
        candidates = sorted(glob.glob(os.path.join(
            output_dir, f"snapdir_{tag}", f"snapshot_{tag}.*.hdf5")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(output_dir, f"snapshot_{tag}.hdf5")))
        if not candidates:
            raise FileNotFoundError(f"No snapshot {tag} found under {output_dir}")
        return candidates, snap_num

    candidates = sorted(glob.glob(os.path.join(output_dir, "snapdir_*", "snapshot_*.hdf5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(output_dir, "snapshot_*.hdf5")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot_*.hdf5 files found under {output_dir}")

    max_num = max(snapshot_number_from_path(p) for p in candidates)
    return [p for p in candidates if snapshot_number_from_path(p) == max_num], max_num


def find_all_snapshots(output_dir):
    """Return {snap_num: [snapshot pieces]} for all available snapshots."""
    candidates = sorted(glob.glob(os.path.join(output_dir, "snapdir_*", "snapshot_*.hdf5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(output_dir, "snapshot_*.hdf5")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot_*.hdf5 files found under {output_dir}")

    grouped = {}
    for path in candidates:
        grouped.setdefault(snapshot_number_from_path(path), []).append(path)
    for snap in grouped:
        grouped[snap] = sorted(grouped[snap])
    return dict(sorted(grouped.items()))


def load_dust_snapshot(snapshot_files):
    """Load live PartType6 mass, source, CF, radius, and redshift."""
    all_mass, all_source, all_cf, all_radius = [], [], [], []
    unit_mass_g, hubble_param, redshift = None, None, None

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
            required = ["Masses", "DustSource", "CarbonMassFraction", "GrainRadius"]
            missing = [name for name in required if name not in pt6]
            if missing:
                print(f"  WARNING: {path} missing PartType6 fields {missing}; skipping")
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

    alive = ((mass_msun > floor_msun) & np.isfinite(mass_msun) &
             np.isfinite(cf) & np.isfinite(radius) & (radius > 0.0))

    return mass_msun[alive], source[alive], cf[alive], radius[alive], redshift


def plot_delta_cf_vs_radius(output_dir, snap_num, output_path,
                            max_points=20000, delta_limit=None):
    """Plot dCF from birth against grain radius for each source."""
    files, snap_num = find_snapshot_files(output_dir, snap_num)
    print(f"Scatter snapshot {snap_num:03d}: {len(files)} file(s)")

    mass, source, cf, radius, z = load_dust_snapshot(files)
    if len(mass) == 0:
        raise RuntimeError("No live PartType6 particles found for scatter plot.")

    z_str = f"z={z:.3f}" if z is not None else f"snap {snap_num:03d}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    rng = np.random.default_rng(12345)

    global_abs_max = 0.0
    for src_id, (_, birth_cf, _) in SOURCE_INFO.items():
        sel = source == src_id
        if np.any(sel):
            global_abs_max = max(global_abs_max,
                                 float(np.nanmax(np.abs(cf[sel] - birth_cf))))

    ylim = float(delta_limit) if delta_limit is not None else max(0.01, 1.08 * global_abs_max)

    for ax, (src_id, (label, birth_cf, color)) in zip(axes, SOURCE_INFO.items()):
        sel = source == src_id
        n = int(sel.sum())
        if n == 0:
            ax.set_title(f"{label} (0 particles)")
            ax.axis("off")
            continue

        r_src = radius[sel]
        cf_src = cf[sel]
        m_src = mass[sel]
        delta = cf_src - birth_cf

        if len(r_src) > max_points:
            idx = rng.choice(len(r_src), size=max_points, replace=False)
        else:
            idx = np.arange(len(r_src))

        ax.scatter(r_src[idx], delta[idx], s=8, alpha=0.30,
                   color=color, edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)

        mean_delta = np.average(delta, weights=m_src)
        median_delta = np.median(delta)
        ax.axhline(mean_delta, color=color, linewidth=1.5,
                   label=f"mass-wtd mean dCF={mean_delta:+.4f}")

        ax.set_xscale("log")
        ax.set_xlabel("GrainRadius (nm)")
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.2)
        ax.set_title(f"{label}: n={n}\nmedian dCF={median_delta:+.4f}")
        ax.legend(fontsize=8)

        frac_up = np.mean(delta > 1e-3)
        frac_down = np.mean(delta < -1e-3)
        print(f"  [{label}] mass-weighted mean dCF={mean_delta:+.5f}, "
              f"median dCF={median_delta:+.5f}, "
              f"up>1e-3: {100*frac_up:.1f}%, down<-1e-3: {100*frac_down:.1f}%")

    axes[0].set_ylabel("dCarbonMassFraction = CF - birth CF")
    fig.suptitle(f"CosmicGrain composition change vs grain radius -- {z_str}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")


def build_composition_history(output_dir):
    """Calculate per-source composition statistics for every snapshot."""
    snapshots = find_all_snapshots(output_dir)
    history = {
        src_id: {"z": [], "snap": [], "mean_cf": [], "mean_abs_delta": [], "mass": [], "n": []}
        for src_id in SOURCE_INFO
    }

    print(f"Found {len(snapshots)} snapshots from {min(snapshots):03d} to {max(snapshots):03d}")

    for snap_num, files in snapshots.items():
        mass, source, cf, radius, z = load_dust_snapshot(files)
        if len(mass) == 0:
            print(f"  snap {snap_num:03d}: no live dust; skipping")
            continue
        if z is None:
            print(f"  snap {snap_num:03d}: missing redshift; skipping")
            continue

        print(f"  snap {snap_num:03d}: z={z:.3f}, live dust={len(mass)}")

        for src_id, (label, birth_cf, _) in SOURCE_INFO.items():
            sel = source == src_id
            if not np.any(sel):
                continue
            m_src = mass[sel]
            cf_src = cf[sel]
            history[src_id]["z"].append(z)
            history[src_id]["snap"].append(snap_num)
            history[src_id]["mean_cf"].append(np.average(cf_src, weights=m_src))
            history[src_id]["mean_abs_delta"].append(
                np.average(np.abs(cf_src - birth_cf), weights=m_src))
            history[src_id]["mass"].append(m_src.sum())
            history[src_id]["n"].append(int(sel.sum()))

    return history


def plot_cf_history(output_dir, output_path):
    """Plot mass-weighted mean CF versus redshift for each stellar source."""
    history = build_composition_history(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

    for ax, (src_id, (label, birth_cf, color)) in zip(axes, SOURCE_INFO.items()):
        data = history[src_id]
        if len(data["z"]) == 0:
            ax.set_title(f"{label} (no data)")
            ax.axis("off")
            continue

        z = np.asarray(data["z"])
        mean_cf = np.asarray(data["mean_cf"])
        mean_abs_delta = np.asarray(data["mean_abs_delta"])
        order = np.argsort(z)[::-1]
        z, mean_cf, mean_abs_delta = z[order], mean_cf[order], mean_abs_delta[order]

        ax.plot(z, mean_cf, marker="o", markersize=3.5, linewidth=1.5,
                color=color, label="mass-weighted mean CF")
        ax.axhline(birth_cf, color="black", linestyle="--", linewidth=1.2,
                   label=f"birth CF = {birth_cf}")
        ax.set_xlabel("Redshift")
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.invert_xaxis()

        print(f"  [{label}] z={z[-1]:.3f}, <CF>_M={mean_cf[-1]:.5f}, "
              f"delta from birth={mean_cf[-1]-birth_cf:+.5f}, "
              f"<|dCF|>_M={mean_abs_delta[-1]:.5f}")

    axes[0].set_ylabel("Mass-weighted mean CarbonMassFraction")
    fig.suptitle("CosmicGrain dust composition evolution by stellar source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="CosmicGrain output directory, e.g. S10_output_512")
    p.add_argument("--snap", type=int, default=None,
                   help="Snapshot for dCF-vs-radius plot. Default: latest available.")
    p.add_argument("--scatter-output", default="cf_delta_vs_radius.png")
    p.add_argument("--history-output", default="cf_vs_redshift.png")
    p.add_argument("--max-points", type=int, default=20000,
                   help="Maximum scatter points per source; statistics use all particles.")
    p.add_argument("--delta-limit", type=float, default=None,
                   help="Optional fixed symmetric dCF y-limit, e.g. 0.1")
    args = p.parse_args()

    plot_delta_cf_vs_radius(args.output_dir, args.snap, args.scatter_output,
                            max_points=args.max_points,
                            delta_limit=args.delta_limit)
    plot_cf_history(args.output_dir, args.history_output)


if __name__ == "__main__":
    main()
