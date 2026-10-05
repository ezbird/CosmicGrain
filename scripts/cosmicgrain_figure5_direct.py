#!/usr/bin/env python3
"""CosmicGrain paper Figure 5: projected R(V) across the full halo suite.

Run from gadget4/scripts with ``python3 cosmicgrain_figure5_direct.py``.
The script reads canonical halo<ID>_output_<resolution> directories directly;
no manifest or Figure 4 run is required. Every halo uses the same physical
pixel size, optical tables, signal cut, dust-count cut, and scaled aperture.

Requires halo441_rv_grain_physics.py beside this script.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import halo441_rv_grain_physics as core

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


TARGETS = {
    "CG01": (9235, 7), "CG02": (7723, 2), "CG03": (5834, 6),
    "CG04": (3886, 1), "CG05": (3879, 0), "CG06": (3352, 8),
    "CG07": (1653, 0), "CG08": (1481, 0), "CG09": (859, 0),
    "CG10": (441, 1), "CG11": (308, 0), "CG12": (295, 0),
}


def apply_cosmicgrain_style(style_file=None):
    candidates = []
    if style_file:
        candidates.append(os.path.abspath(style_file))
    candidates.extend([
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "cosmicgrain.mplstyle"),
        os.path.join(os.getcwd(), "cosmicgrain.mplstyle"),
    ])
    for path in candidates:
        if os.path.isfile(path):
            plt.style.use(path)
            print(f"Using Matplotlib style: {path}")
            return path
    raise FileNotFoundError(
        "cosmicgrain.mplstyle was not found beside this script or in the "
        "current directory; pass --style-file PATH."
    )


def discover_output(root: Path, halo_id: int, resolution: int) -> Path:
    """Select only the exact canonical output; never choose suffixed tests."""
    output = root / f"halo{halo_id}_output_{resolution}"
    if not output.is_dir():
        raise FileNotFoundError(
            f"Missing canonical output directory: {output}. Experimental "
            "directories with suffixes are intentionally not auto-selected."
        )
    return output


def snapshot_files(output: Path, snap: int):
    patterns = [
        output / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.*.hdf5",
        output / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.hdf5",
        output / f"snapshot_{snap:03d}.*.hdf5",
        output / f"snapshot_{snap:03d}.hdf5",
    ]
    for pattern in patterns:
        files = sorted(glob.glob(str(pattern)))
        if files:
            return files
    raise FileNotFoundError(f"{output}: no snapshot files for {snap:03d}")


def catalog_group(output: Path, snap: int, group_index: int):
    candidates = [
        output / f"groups_{snap:03d}" / f"fof_subhalo_tab_{snap:03d}.*.hdf5",
        output / f"groups_{snap:03d}" / f"fof_subhalo_tab_{snap:03d}.hdf5",
        output / f"groups_{snap:03d}" / f"group_tab_{snap:03d}.*.hdf5",
    ]
    files = []
    for candidate in candidates:
        files = sorted(glob.glob(str(candidate)))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"{output}: no group catalog for snapshot {snap:03d}")
    pieces = {key: [] for key in
              ("GroupPos", "Group_M_Crit200", "Group_R_Crit200")}
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "Group" not in handle:
                continue
            for key in pieces:
                if key not in handle["Group"]:
                    raise KeyError(f"{filename}: missing Group/{key}")
                pieces[key].append(np.asarray(handle["Group"][key][:]))
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    if group_index >= len(arrays["GroupPos"]):
        raise IndexError(f"{output}: group {group_index} absent in catalog")
    return (arrays["GroupPos"][group_index],
            float(arrays["Group_M_Crit200"][group_index]),
            float(arrays["Group_R_Crit200"][group_index]))


def disk_rotation(files, center, box, a, h, radius_pkpc):
    coords, velocities, masses = [], [], []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "PartType4" not in handle:
                continue
            stars = handle["PartType4"]
            for field in ("Coordinates", "Velocities", "Masses"):
                if field not in stars:
                    raise KeyError(f"{filename}: PartType4/{field} missing")
            coords.append(stars["Coordinates"][:])
            velocities.append(stars["Velocities"][:])
            masses.append(stars["Masses"][:])
    if not coords:
        raise RuntimeError("No stars available for face-on orientation")
    xyz = core.rotate_positions(
        np.concatenate(coords), center, box, a, h, np.eye(3))
    velocity = np.concatenate(velocities)
    mass = np.concatenate(masses)
    use = (np.linalg.norm(xyz, axis=1) < radius_pkpc) & (mass > 0)
    if np.count_nonzero(use) < 20:
        raise RuntimeError(
            f"Only {np.count_nonzero(use)} stars within {radius_pkpc:g} pkpc "
            "for disk orientation")
    vbulk = np.average(velocity[use], axis=0, weights=mass[use])
    angular = np.sum(
        np.cross(xyz[use], velocity[use] - vbulk) * mass[use, None], axis=0)
    norm = np.linalg.norm(angular)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Stellar angular-momentum orientation is undefined")
    zhat = angular / norm
    reference = (np.array([1.0, 0.0, 0.0]) if abs(zhat[0]) < 0.9
                 else np.array([0.0, 1.0, 0.0]))
    xhat = reference - np.dot(reference, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    return np.vstack([xhat, np.cross(zhat, xhat), zhat])


def analyze_halo(label, output, snap, group_index, carbon_table,
                 silicate_table, args):
    files = snapshot_files(output, snap)
    pattern = str(output / f"snapdir_{snap:03d}" /
                  f"snapshot_{snap:03d}.*.hdf5")
    dust = core.read_dust_snapshot(pattern if glob.glob(pattern) else files[0])
    center, mcode, rcode = catalog_group(output, snap, group_index)
    m200 = mcode * 1.0e10 / dust["h"]
    r200 = rcode * dust["a"] / dust["h"]
    aperture = (args.aperture_pkpc if args.aperture_pkpc is not None
                else args.aperture_r200 * r200)
    depth = (args.zmax_pkpc if args.zmax_pkpc is not None
             else args.depth_r200 * r200)
    half_pixels = int(np.ceil(aperture / args.pixel_size_pkpc))
    map_radius = half_pixels * args.pixel_size_pkpc
    bins = 2 * half_pixels

    rotation = disk_rotation(files, center, dust["box"], dust["a"], dust["h"],
                             args.orientation_radius_pkpc)
    xyz = core.rotate_positions(
        dust["coords"], center, dust["box"], dust["a"], dust["h"], rotation)
    choose = (
        (np.abs(xyz[:, 0]) < map_radius) &
        (np.abs(xyz[:, 1]) < map_radius) &
        (np.abs(xyz[:, 2]) < depth) &
        np.isfinite(dust["mass"]) & (dust["mass"] > 0) &
        np.isfinite(dust["radius_nm"]) & (dust["radius_nm"] > 0) &
        np.isfinite(dust["carbon"])
    )
    x, y = xyz[choose, 0], xyz[choose, 1]
    mass = dust["mass"][choose]
    radius = dust["radius_nm"][choose]
    carbon_fraction = np.clip(dust["carbon"][choose], 0, 1)
    wb, wv = core.extinction_weights(
        mass, radius, carbon_fraction, carbon_table, silicate_table)
    maps = core.build_maps(
        x, y, mass, wb, wv, map_radius, bins, 0.0,
        args.min_signal_fraction)

    centers = -map_radius + (np.arange(bins) + 0.5) * args.pixel_size_pkpc
    gridx, gridy = np.meshgrid(centers, centers)
    projected_radius = np.hypot(gridx, gridy)
    tau_v = maps["tauV"]
    ebv = maps["tauB"] - maps["tauV"]
    in_aperture = projected_radius < aperture
    finite = np.isfinite(maps["rv"])
    count_pass = maps["count_raw"] >= args.min_count
    positive_ebv = np.isfinite(ebv) & (ebv > 0)
    aperture_ebv = ebv[in_aperture & positive_ebv]
    ebv_reference = (float(np.nanmax(aperture_ebv))
                     if aperture_ebv.size else np.nan)
    ebv_floor = args.min_ebv_fraction * ebv_reference
    ebv_pass = positive_ebv & (ebv >= ebv_floor)
    quality = finite & count_pass & in_aperture & ebv_pass
    valid = quality
    nvalid = int(np.count_nonzero(valid))
    if nvalid < 3:
        print(f"WARNING: {label} has only {nvalid} quality-controlled pixels; "
              "omitting it instead of aborting the suite.")
        return None

    print("\n" + "-" * 78)
    print(f"{label}: {output.name}; group={group_index}; "
          f"M200={m200:.3e} Msun; R200={r200:.2f} pkpc")
    print(f"  aperture={aperture:.2f} pkpc ({aperture/r200:.4f} R200); "
          f"depth=±{depth:.2f} pkpc; map={bins}x{bins}; "
          f"pixel={args.pixel_size_pkpc:g} pkpc")
    n_quality = int(np.count_nonzero(quality))
    n_high_rv = int(np.count_nonzero(quality & (maps["rv"] > args.tail_rv)))
    high_rv_fraction = n_high_rv / n_quality if n_quality else np.nan
    print(f"  dust in map slab={len(mass):,}; finite R(V) pixels="
          f"{np.count_nonzero(finite):,}; quality-pass pixels={n_quality:,}; "
          f"eligible pixels={nvalid:,}")
    print(f"  E(B-V) floor={ebv_floor:.4e} "
          f"({args.min_ebv_fraction:g} of aperture maximum); "
          f"R(V) > {args.tail_rv:g}: {n_high_rv:,}/{n_quality:,} "
          f"({high_rv_fraction:.1%}); retained in analysis")
    for threshold in (1, 3, 5, 10):
        count = np.count_nonzero(
            finite & (maps["count_raw"] >= threshold) &
            (projected_radius < aperture))
        print(f"    finite R(V), N_dust >= {threshold}: {count:,}")

    return {
        "label": label, "halo_id": TARGETS[label][0], "output": str(output),
        "group_index": group_index, "m200_msun": m200, "r200_pkpc": r200,
        "aperture_pkpc": aperture, "depth_pkpc": depth,
        "pixel_size_pkpc": args.pixel_size_pkpc,
        "ebv_reference": ebv_reference, "ebv_floor": ebv_floor,
        "N_quality": n_quality, "N_high_Rv": n_high_rv,
        "high_Rv_fraction": high_rv_fraction,
        "x_pkpc": gridx[valid], "y_pkpc": gridy[valid],
        "radius_pkpc": projected_radius[valid],
        "radius_r200": projected_radius[valid] / r200,
        "rv": maps["rv"][valid],
        "tau_v": tau_v[valid], "ebv": ebv[valid],
        "dust_count": maps["count_raw"][valid].astype(int),
        "dust_mass_code": maps["mass_raw"][valid],
    }


def radial_profile(result, maximum_radius_r200, bins, minimum_pixels):
    edges = np.linspace(0.0, maximum_radius_r200, bins + 1)
    rows = []
    radius, rv = result["radius_r200"], result["rv"]
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        use = (radius >= left) & (radius < right)
        count = int(np.count_nonzero(use))
        if count < minimum_pixels:
            continue
        p16, median, p84 = np.percentile(rv[use], [16, 50, 84])
        rows.append({
            "bin": index + 1, "radius_min_r200": left,
            "radius_max_r200": right,
            "radius_median_r200": float(np.median(radius[use])),
            "N_pixels": count, "Rv_p16": p16, "Rv_median": median,
            "Rv_p84": p84,
        })
    return rows


def write_products(results, profiles, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pixel_path = Path(str(prefix) + "_pixels.csv")
    with pixel_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "CG", "halo_id", "M200c_Msun", "R200c_pkpc", "x_pkpc",
            "y_pkpc", "projected_radius_pkpc", "projected_radius_over_R200c",
            "Rv", "tauV_relative", "EBV_relative", "dust_count",
            "dust_mass_code"])
        for result in results:
            values = zip(result["x_pkpc"], result["y_pkpc"],
                         result["radius_pkpc"], result["radius_r200"],
                         result["rv"], result["tau_v"], result["ebv"],
                         result["dust_count"],
                         result["dust_mass_code"])
            for row in values:
                writer.writerow([result["label"], result["halo_id"],
                                 result["m200_msun"], result["r200_pkpc"], *row])

    summary_path = Path(str(prefix) + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "CG", "halo_id", "output", "group_index", "M200c_Msun",
            "R200c_pkpc", "aperture_pkpc", "depth_pkpc", "pixel_size_pkpc",
            "N_quality", "N_above_Rv_tail_threshold",
            "above_tail_fraction", "EBV_reference", "EBV_floor",
            "N_pixels", "Rv_p16", "Rv_median", "Rv_p84"])
        for result in results:
            p16, median, p84 = np.percentile(result["rv"], [16, 50, 84])
            writer.writerow([
                result["label"], result["halo_id"], result["output"],
                result["group_index"], result["m200_msun"],
                result["r200_pkpc"], result["aperture_pkpc"],
                result["depth_pkpc"], result["pixel_size_pkpc"],
                result["N_quality"], result["N_high_Rv"],
                result["high_Rv_fraction"], result["ebv_reference"],
                result["ebv_floor"],
                len(result["rv"]), p16, median, p84])

    radial_path = Path(str(prefix) + "_radial_profiles.csv")
    with radial_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["CG", "M200c_Msun", "bin", "radius_min_r200",
                  "radius_max_r200", "radius_median_r200", "N_pixels",
                  "Rv_p16", "Rv_median", "Rv_p84"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for row in profiles[result["label"]]:
                writer.writerow({"CG": result["label"],
                                 "M200c_Msun": result["m200_msun"], **row})
    return pixel_path, summary_path, radial_path


def make_figure(results, profiles, prefix, dataset_label):
    results = sorted(results, key=lambda item: item["m200_msun"])
    logmass = np.array([np.log10(item["m200_msun"]) for item in results])
    norm = Normalize(vmin=logmass.min(), vmax=logmass.max())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(norm(value)) for value in logmass]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    positions = np.arange(len(results))
    parts = ax.violinplot(
        [item["rv"] for item in results], positions=positions, vert=False,
        widths=0.78, showmeans=False, showmedians=False, showextrema=False,
        bw_method=0.25)
    for body, color, result in zip(parts["bodies"], colors, results):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
        body.set_linewidth(1.5 if result["label"] == "CG10" else 0.7)
    for y, color, result in zip(positions, colors, results):
        p16, median, p84 = np.percentile(result["rv"], [16, 50, 84])
        ax.plot([p16, p84], [y, y], color="black", lw=1.5, zorder=4)
        marker = "*" if result["label"] == "CG10" else "o"
        ax.scatter(median, y, marker=marker, s=105 if marker == "*" else 35,
                   color=color, edgecolor="black", linewidth=0.7, zorder=5)
    ax.set_yticks(positions)
    ax.set_yticklabels([
        f'{item["label"]}  (N={len(item["rv"]):,})' for item in results])
    ax.set_xlabel(r"Projected $R(V)$")
    ax.set_ylabel("Halo (eligible pixels)")
    ax.set_title("(a) Sightline distributions")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xscale("log")

    ax = axes[1]
    for color, result in zip(colors, results):
        rows = profiles[result["label"]]
        if not rows:
            print(f'{result["label"]}: no radial bins meet the pixel-count cut')
            continue
        x = np.array([row["radius_median_r200"] for row in rows])
        median = np.array([row["Rv_median"] for row in rows])
        p16 = np.array([row["Rv_p16"] for row in rows])
        p84 = np.array([row["Rv_p84"] for row in rows])
        marker = "*" if result["label"] == "CG10" else "o"
        ax.vlines(x, p16, p84, color=color, alpha=0.23, lw=1.0)
        ax.plot(x, median, marker=marker, ms=8 if marker == "*" else 3.5,
                lw=2.7 if marker == "*" else 1.45, color=color, alpha=0.92)
    ax.set_xlabel(r"Projected radius $R_{\rm proj}/R_{200c}$")
    ax.set_ylabel(r"Median projected $R(V)$")
    ax.set_title("(b) Radial profiles")
    ax.grid(alpha=0.22)
    ax.set_yscale("log")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axes, fraction=0.035, pad=0.035)
    cbar.set_label(r"$\log_{10}(M_{200c}/M_\odot)$")
    fig.subplots_adjust(left=0.13, right=0.91, bottom=0.12, wspace=0.30,
                        top=0.89 if dataset_label else 0.94)
    if dataset_label:
        fig.suptitle(dataset_label, color="#b22222", fontweight="bold",
                     fontsize=12)
    png, pdf = str(prefix) + ".png", str(prefix) + ".pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                        help="Directory containing canonical halo outputs")
    parser.add_argument("--halos", nargs="+", choices=sorted(TARGETS),
                        default=list(TARGETS))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--snap", type=int, default=29)
    parser.add_argument("--pixel-size-pkpc", type=float, default=1.0)
    parser.add_argument("--aperture-r200", type=float, default=0.1)
    parser.add_argument("--aperture-pkpc", type=float, default=None)
    parser.add_argument("--depth-r200", type=float, default=0.1)
    parser.add_argument("--zmax-pkpc", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-signal-fraction", type=float, default=1.0e-3)
    parser.add_argument("--min-ebv-fraction", type=float, default=1.0e-3,
                        help="Minimum E(B-V) as a fraction of the maximum "
                             "positive E(B-V) inside each halo aperture")
    parser.add_argument("--tail-rv", type=float, default=10.0,
                        help="Report, but do not discard, pixels above this R(V)")
    parser.add_argument("--orientation-radius-pkpc", type=float, default=20.0)
    parser.add_argument("--radial-bins", type=int, default=5)
    parser.add_argument("--min-radial-pixels", type=int, default=20)
    parser.add_argument("--carbon-table",
                        default=str(SCRIPT_DIR / "dust_optics/callqcomp.out_D16MGemt"))
    parser.add_argument("--silicate-table",
                        default=str(SCRIPT_DIR / "dust_optics/callqcomp.out_DH16asil"))
    parser.add_argument("--style-file", default=None)
    parser.add_argument("--prefix", default=str(PROJECT_ROOT / "CG_figure5_suite_rv"))
    parser.add_argument("--dataset-label", default="PROVISIONAL — PRE-RERUN SUITE",
                        help="Text above figure; pass an empty string to suppress")
    args = parser.parse_args()

    if args.resolution < 1 or args.snap < 0:
        parser.error("resolution must be positive and snap nonnegative")
    if args.pixel_size_pkpc <= 0 or args.aperture_r200 <= 0:
        parser.error("pixel size and aperture must be positive")
    if args.aperture_pkpc is not None and args.aperture_pkpc <= 0:
        parser.error("--aperture-pkpc must be positive")
    if args.depth_r200 <= 0 or (args.zmax_pkpc is not None and args.zmax_pkpc <= 0):
        parser.error("projection depth must be positive")
    if args.min_count < 1 or args.radial_bins < 1 or args.min_radial_pixels < 1:
        parser.error("count cuts and radial-bin count must be positive")
    if not (0 <= args.min_ebv_fraction < 1):
        parser.error("--min-ebv-fraction must be in [0, 1)")
    if args.tail_rv <= 0:
        parser.error("--tail-rv must be positive")

    apply_cosmicgrain_style(args.style_file)
    root = args.output_root.resolve()
    carbon = core.read_draine_qext_table(args.carbon_table)
    silicate = core.read_draine_qext_table(args.silicate_table)
    results = []
    for label in args.halos:
        halo_id, group_index = TARGETS[label]
        output = discover_output(root, halo_id, args.resolution)
        result = analyze_halo(
            label, output, args.snap, group_index, carbon, silicate, args)
        if result is not None:
            results.append(result)
    if not results:
        raise RuntimeError("No halo has enough quality-controlled sightlines")
    results.sort(key=lambda item: item["m200_msun"])
    maximum_radius_r200 = max(
        item["aperture_pkpc"] / item["r200_pkpc"] for item in results)
    profiles = {
        item["label"]: radial_profile(
            item, maximum_radius_r200, args.radial_bins, args.min_radial_pixels)
        for item in results}
    pixels, summary, radial = write_products(results, profiles, args.prefix)
    png, pdf = make_figure(results, profiles, args.prefix, args.dataset_label)
    print("\n" + "=" * 78)
    print(f"Saved {png} and {pdf}")
    print(f"Saved {pixels}\nSaved {summary}\nSaved {radial}")
    print("=" * 78)


if __name__ == "__main__":
    main()
