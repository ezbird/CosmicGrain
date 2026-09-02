"""
plot_composition_histogram.py
--------------------------------------------------------------------------
DustSource-binned CarbonMassFraction histogram for the latest available
snapshot in a CosmicGrain output directory -- the composition-evolution
sanity check: are AGB-born grains (birth CF=0.6) and SNII/LRN-born grains
(birth CF=0.1) actually drifting away from their birth values, and in
the expected direction?

Reuses the same snapshot-finding, unit-conversion, and destroyed-particle
filtering logic already validated in compare_lrn_runs.py against real
data -- not new/unverified loading code.

Usage:
    python plot_composition_histogram.py S10_output_512 --output cf_hist.png
    python plot_composition_histogram.py S10_output_512 --snap 30 --output cf_hist_snap30.png
--------------------------------------------------------------------------
"""

import sys
import glob
import os
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


def find_snapshot_files(output_dir, snap_num=None):
    """Find the requested snapshot, or the highest-numbered one available
    if snap_num is None -- same pattern as compare_lrn_runs.py."""
    if snap_num is not None:
        tag = f"{snap_num:03d}"
        candidates = sorted(glob.glob(os.path.join(output_dir, f"snapdir_{tag}", f"snapshot_{tag}.*.hdf5")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(output_dir, f"snapshot_{tag}.hdf5")))
        if not candidates:
            raise FileNotFoundError(f"No snapshot {tag} found under {output_dir}")
        return candidates

    candidates = sorted(glob.glob(os.path.join(output_dir, "snapdir_*", "snapshot_*.hdf5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(output_dir, "snapshot_*.hdf5")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot_*.hdf5 files found under {output_dir}")

    def snap_number(path):
        base = os.path.basename(path).replace(".hdf5", "")
        return int(base.split(".")[0].split("_")[1])

    max_num = max(snap_number(p) for p in candidates)
    return [p for p in candidates if snap_number(p) == max_num], max_num


def load_dust_composition(snapshot_files):
    all_cf, all_source, all_mass = [], [], []
    unit_mass_g, hubble_param, redshift = None, None, None

    for path in snapshot_files:
        with h5py.File(path, "r") as f:
            if unit_mass_g is None:
                params = f["Parameters"].attrs
                unit_mass_g = float(params["UnitMass_in_g"])
                hubble_param = float(params["HubbleParam"])
                redshift = float(f["Header"].attrs["Redshift"]) \
                    if "Redshift" in f["Header"].attrs else None

            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]
            if "Masses" not in pt6 or "DustSource" not in pt6 or "CarbonMassFraction" not in pt6:
                print(f"  WARNING: PartType6 missing expected fields in {path}, skipping")
                continue
            all_mass.append(pt6["Masses"][:])
            all_source.append(pt6["DustSource"][:])
            all_cf.append(pt6["CarbonMassFraction"][:])

    if not all_mass:
        return np.array([]), np.array([]), np.array([]), unit_mass_g, hubble_param, redshift

    mass = np.concatenate(all_mass)
    source = np.concatenate(all_source)
    cf = np.concatenate(all_cf)

    mass_msun = mass * unit_mass_g / hubble_param / SOLAR_MASS_G
    floor_msun = DESTROYED_MASS_FLOOR * unit_mass_g / hubble_param / SOLAR_MASS_G
    alive = mass_msun > floor_msun

    return mass_msun[alive], source[alive], cf[alive], unit_mass_g, hubble_param, redshift


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="CosmicGrain output directory (e.g. S10_output_512)")
    p.add_argument("--snap", type=int, default=None,
                   help="Specific snapshot number. Default: latest available.")
    p.add_argument("--output", default="cf_histogram.png")
    p.add_argument("--bins", type=int, default=40)
    args = p.parse_args()

    result = find_snapshot_files(args.output_dir, args.snap)
    if args.snap is not None:
        files, snap_num = result, args.snap
    else:
        files, snap_num = result

    print(f"Snapshot {snap_num:03d}: {len(files)} file(s)")
    mass, source, cf, unit_mass_g, h, z = load_dust_composition(files)
    if len(mass) == 0:
        sys.exit("No live PartType6 particles found -- nothing to plot.")

    z_str = f"z={z:.3f}" if z is not None else f"snap {snap_num:03d}"
    print(f"  {len(mass)} live dust particles, {z_str}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, (src_id, (label, birth_cf, color)) in zip(axes, SOURCE_INFO.items()):
        sel = source == src_id
        n = int(sel.sum())
        if n == 0:
            ax.set_title(f"{label} (0 particles)")
            ax.axis("off")
            continue

        cf_src = cf[sel]
        mass_src = mass[sel]
        total_mass = mass_src.sum()

        ax.hist(cf_src, bins=args.bins, range=(0, 1), weights=mass_src,
                color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.axvline(birth_cf, color="black", linestyle="--", linewidth=1.5,
                   label=f"birth CF = {birth_cf}")

        mean_cf = np.average(cf_src, weights=mass_src)
        delta = cf_src - birth_cf
        mean_abs_delta = np.average(np.abs(delta), weights=mass_src)
        frac_moved = float((np.abs(delta) > 1e-3).sum()) / n

        # Signed breakdown -- the actual question for coagulation attribution:
        # did ANY mass move ABOVE birth CF? Sputtering/shock destruction can
        # only push CF down, so upward movement (delta > 0) can only come
        # from coagulation's ambient mixing. This is a real diagnostic gap
        # the |delta|-only stats above can't answer.
        up_mask = delta > 1e-3
        down_mask = delta < -1e-3
        n_up, n_down = int(up_mask.sum()), int(down_mask.sum())
        mass_up = float(mass_src[up_mask].sum()) if n_up > 0 else 0.0
        mass_down = float(mass_src[down_mask].sum()) if n_down > 0 else 0.0
        max_delta_up = float(delta[up_mask].max()) if n_up > 0 else 0.0

        ax.set_title(f"{label}: {n} particles, {total_mass:.2e} Msun")
        ax.set_xlabel("CarbonMassFraction")
        ax.set_ylabel("Mass (Msun) per bin")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

        print(f"  [{label}] n={n}  total_mass={total_mass:.4e} Msun  "
              f"mean_CF={mean_cf:.4f} (birth={birth_cf})  "
              f"mean|ΔCF|={mean_abs_delta:.4f}  "
              f"frac with |ΔCF|>1e-3: {100*frac_moved:.1f}%")
        print(f"    -> ABOVE birth CF (coagulation signature): "
              f"{n_up} particles, {mass_up:.3e} Msun, max ΔCF={max_delta_up:.4f}")
        print(f"    -> BELOW birth CF (sputtering/shock, and/or coagulation): "
              f"{n_down} particles, {mass_down:.3e} Msun")

    fig.suptitle(f"CosmicGrain dust composition by source -- {z_str}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
