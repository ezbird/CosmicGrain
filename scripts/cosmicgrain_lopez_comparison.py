#!/usr/bin/env python3
"""CosmicGrain comparison to Lopez et al. (2026), arXiv:2609.37358.

The observational comparison follows their Table 3 and Figure 11: per-pixel
dust-to-gas, dust-to-stellar, and dust-to-metals surface-density ratios versus
HII-CHI-Mistry oxygen abundance.  The simulation is projected face-on and
binned to approximately matched 1 kpc pixels.  Earlier halo identity is found
with halo_utils.get_zoom_halo_series, which tracks the final target backward
using conserved high-resolution dark-matter IDs; a z=0 FOF index is never
silently reused at high redshift.

Run from the GADGET-4 root, for the current CG10/Halo 441 2048 run:

    python3 scripts/cosmicgrain_lopez_comparison.py \
        --output-dir halo441_output_2048_mingashsml0.0 \
        --final-group-index 1 --target-redshift 4.5 \
        --outdir lopez_halo441_2048

The final group index must be the accepted index at the latest complete
snapshot (normally taken from the post-run audit).  Outputs are two figures,
pixel and summary CSV files, and a JSON provenance record.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from halo_utils import (  # noqa: E402
    find_snapshot_and_group_files,
    find_snapshots,
    get_zoom_halo_series,
    periodic_delta,
)

X_H = 0.76
A_O = 16.0
MASS_CODE_TO_MSUN = 1.0e10

Z_FIELDS = ("GFM_Metallicity", "Metallicity", "MetalMassFraction")
O_FIELDS = ("GasOxygenMassFraction", "OxygenMassFraction")
SFR_FIELDS = ("StarFormationRate", "SFR")
RADIUS_FIELDS = ("GrainRadius", "DustGrainRadius", "GRAD")
CARBON_FIELDS = ("CarbonMassFraction", "CarbonFraction", "DustCarbonFraction", "CFRC")
SOURCE_FIELDS = ("DustSource", "Source", "DustSourceType", "DSRC")

# Table 2 (fiducial HCM metallicity) and Table 3 of Lopez et al. (2026).
# Errors in Table 3 are the dispersion of per-pixel logarithmic ratios.
LOPEZ = (
    dict(id="C02", name="DC-848185", oh12=8.27, oherr=0.15,
         log_dg=-2.69, dg_lo=0.16, dg_hi=0.21,
         log_ds=-2.91, ds_lo=0.13, ds_hi=0.13,
         log_dz=-0.53, dz_lo=0.14, dz_hi=0.17,
         area_kpc2=35.40, pixels=66, strong_outflow=True),
    dict(id="C03", name="DC-536534", oh12=8.14, oherr=0.20,
         log_dg=-2.52, dg_lo=0.17, dg_hi=0.23,
         log_ds=-2.65, ds_lo=0.14, ds_hi=0.17,
         log_dz=-0.30, dz_lo=0.14, dz_hi=0.16,
         area_kpc2=9.32, pixels=12, strong_outflow=True),
    dict(id="C04a", name="VC-5100822662", oh12=8.28, oherr=0.17,
         log_dg=-2.29, dg_lo=0.15, dg_hi=0.21,
         log_ds=-2.01, ds_lo=0.12, ds_hi=0.13,
         log_dz=-0.31, dz_lo=0.13, dz_hi=0.15,
         area_kpc2=31.27, pixels=32, strong_outflow=False),
    dict(id="C07ab", name="DC-873321", oh12=8.21, oherr=0.16,
         log_dg=-2.20, dg_lo=0.19, dg_hi=0.24,
         log_ds=-2.53, ds_lo=0.15, ds_hi=0.15,
         log_dz=-0.22, dz_lo=0.16, dz_hi=0.17,
         area_kpc2=16.53, pixels=30, strong_outflow=True),
    dict(id="C11", name="DC-630594", oh12=8.21, oherr=0.17,
         log_dg=-2.61, dg_lo=0.18, dg_hi=0.23,
         log_ds=-2.58, ds_lo=0.16, ds_hi=0.16,
         log_dz=-0.53, dz_lo=0.18, dz_hi=0.21,
         area_kpc2=6.99, pixels=11, strong_outflow=False),
    dict(id="C13", name="VC-5100994794", oh12=8.25, oherr=0.18,
         log_dg=-2.41, dg_lo=0.16, dg_hi=0.22,
         log_ds=-2.21, ds_lo=0.12, ds_hi=0.10,
         log_dz=-0.36, dz_lo=0.14, dz_hi=0.16,
         area_kpc2=12.97, pixels=21, strong_outflow=False),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="halo441_output_2048_mingashsml0.0")
    p.add_argument("--final-group-index", type=int, required=True,
                   help="Accepted target FOF index at the latest complete snapshot")
    p.add_argument("--reference-snapshot", type=int, default=None,
                   help="Final tracking anchor (default: latest complete snapshot)")
    p.add_argument("--snapshot", type=int, default=None,
                   help="Analyze an explicit snapshot instead of closest redshift")
    p.add_argument("--target-redshift", type=float, default=4.5)
    p.add_argument("--pixel-kpc", type=float, default=1.0)
    p.add_argument("--half-size-kpc", type=float, default=5.0)
    p.add_argument("--half-depth-kpc", type=float, default=5.0)
    p.add_argument("--orientation-radius-kpc", type=float, default=5.0)
    p.add_argument("--min-dust-particles", type=int, default=3,
                   help="Simulation dust-detection proxy per projected pixel")
    p.add_argument("--grain-radius-to-micron", type=float, default=1.0e-3,
                   help="Stored PartType6 grain radius multiplier (nm -> micron by default)")
    p.add_argument("--small-grain-micron", type=float, default=0.03)
    p.add_argument("--outflow-shell-r200", nargs=2, type=float, default=(0.1, 0.5),
                   metavar=("RMIN", "RMAX"))
    p.add_argument("--outflow-speed-kms", type=float, default=100.0)
    p.add_argument("--outdir", default="cosmicgrain_lopez_comparison")
    p.add_argument("--style-file", default=None)
    return p.parse_args()


def choose_field(available: Iterable[str], aliases: Sequence[str], required=True) -> Optional[str]:
    available = set(available)
    for name in aliases:
        if name in available:
            return name
    if required:
        raise KeyError(f"None of {aliases} found; available fields: {sorted(available)}")
    return None


def snapshot_files(output_dir: str, snap: int) -> List[str]:
    files, _ = find_snapshot_and_group_files(output_dir, snap)
    if not files:
        raise RuntimeError(f"No snapshot chunks for snapshot {snap:03d}")
    return list(files)


def snapshot_redshift(path: Path) -> float:
    first = sorted(path.glob("snapshot_*.hdf5"))[0]
    with h5py.File(first, "r") as f:
        h = f["Header"].attrs
        a = float(h["Time"])
        return float(h.get("Redshift", 1.0 / a - 1.0))


def select_snapshot(output_dir: str, target_z: float, explicit: Optional[int]) -> Tuple[int, float]:
    available = find_snapshots(output_dir)
    if not available:
        raise RuntimeError(f"No complete snapshot/catalog pairs in {output_dir}")
    candidates = []
    for snap, snap_path, _ in available:
        z = snapshot_redshift(Path(snap_path))
        candidates.append((int(snap), z))
    if explicit is not None:
        for snap, z in candidates:
            if snap == explicit:
                return snap, z
        raise RuntimeError(f"Snapshot {explicit:03d} is not a complete snapshot/catalog pair")
    return min(candidates, key=lambda item: abs(item[1] - target_z))


def available_fields(files: Sequence[str], ptype: int) -> set:
    result = set()
    for filename in files:
        with h5py.File(filename, "r") as f:
            key = f"PartType{ptype}"
            if key in f:
                result.update(f[key].keys())
    return result


def read_type(files: Sequence[str], ptype: int, requested: Sequence[str]) -> Dict[str, np.ndarray]:
    output: Dict[str, List[np.ndarray]] = {name: [] for name in requested}
    output["Masses"] = []
    mass_table = None
    for filename in files:
        with h5py.File(filename, "r") as f:
            if mass_table is None:
                mass_table = np.asarray(f["Header"].attrs["MassTable"], dtype=float)
            key = f"PartType{ptype}"
            if key not in f:
                continue
            g = f[key]
            n = len(g["Coordinates"])
            output["Masses"].append(
                np.asarray(g["Masses"], dtype=float)
                if "Masses" in g else np.full(n, mass_table[ptype], dtype=float)
            )
            for name in requested:
                if name not in g:
                    raise KeyError(f"{filename}: {key}/{name} missing")
                output[name].append(np.asarray(g[name]))
    if not output["Masses"]:
        raise RuntimeError(f"PartType{ptype} is absent")
    return {key: np.concatenate(value) for key, value in output.items() if value}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    q = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[q], weights=weights[q])) if np.any(q) else float("nan")


def rotation_to_face_on(coords: np.ndarray, velocities: np.ndarray, masses: np.ndarray,
                        radius: float, component: str = "particles") -> np.ndarray:
    r = np.linalg.norm(coords, axis=1)
    q = np.isfinite(r) & (r < radius) & np.isfinite(masses) & (masses > 0)
    if np.count_nonzero(q) < 10:
        raise RuntimeError(
            f"Too few {component} within {radius:.3f} pkpc to define orientation "
            f"({np.count_nonzero(q)} < 10)"
        )
    vbulk = np.average(velocities[q], axis=0, weights=masses[q])
    ang = np.sum(np.cross(coords[q], velocities[q] - vbulk) * masses[q, None], axis=0)
    norm = np.linalg.norm(ang)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Stellar angular momentum is undefined")
    ez = ang / norm
    trial = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = np.cross(trial, ez)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.vstack((ex, ey, ez))


def choose_orientation(sxyz: np.ndarray, svel: np.ndarray, sm: np.ndarray,
                       gxyz: np.ndarray, gvel: np.ndarray, gm: np.ndarray,
                       requested_radius: float, r200: float) -> Tuple[np.ndarray, str, float]:
    """Prefer stellar angular momentum, then gas, then the simulation z axis."""
    stellar_radius = min(requested_radius, 0.1 * r200)
    try:
        return (
            rotation_to_face_on(sxyz, svel, sm, stellar_radius, "stars"),
            "stellar_angular_momentum",
            stellar_radius,
        )
    except RuntimeError as stellar_error:
        print(f"[orientation] WARNING: {stellar_error}")

    # Young high-z progenitors can contain very few stars.  A somewhat wider
    # inner-gas aperture gives a stable disk/flow axis without using the CGM.
    gas_radius = min(requested_radius, 0.25 * r200)
    try:
        rotation = rotation_to_face_on(gxyz, gvel, gm, gas_radius, "gas particles")
        print(f"[orientation] using inner-gas angular momentum within {gas_radius:.3f} pkpc")
        return rotation, "gas_angular_momentum", gas_radius
    except RuntimeError as gas_error:
        print(f"[orientation] WARNING: {gas_error}")
        print("[orientation] WARNING: using the simulation z axis; treat map morphology cautiously")
        return np.eye(3), "simulation_z_axis_fallback", float("nan")


def project_sum(xyz: np.ndarray, weights: np.ndarray, edges: np.ndarray,
                half_depth: float) -> np.ndarray:
    q = np.all(np.isfinite(xyz), axis=1) & np.isfinite(weights) & (np.abs(xyz[:, 2]) <= half_depth)
    return np.histogram2d(xyz[q, 0], xyz[q, 1], bins=(edges, edges), weights=weights[q])[0].T


def project_count(xyz: np.ndarray, edges: np.ndarray, half_depth: float) -> np.ndarray:
    return project_sum(xyz, np.ones(len(xyz)), edges, half_depth)


def project_weighted(xyz: np.ndarray, values: np.ndarray, weights: np.ndarray,
                     edges: np.ndarray, half_depth: float) -> np.ndarray:
    numerator = project_sum(xyz, values * weights, edges, half_depth)
    denominator = project_sum(xyz, weights, edges, half_depth)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.log10(np.divide(num, den, out=np.full_like(num, np.nan), where=(num > 0) & (den > 0)))


def robust_stats(values: np.ndarray) -> Dict[str, float]:
    x = values[np.isfinite(values)]
    if not len(x):
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"), p16=float("nan"), p84=float("nan"))
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), std=float(np.std(x)),
                p16=float(np.percentile(x, 16)), p84=float(np.percentile(x, 84)))


def observational_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LOPEZ[0].keys()))
        writer.writeheader()
        writer.writerows(LOPEZ)


def plot_comparison(path: Path, pixels: Dict[str, np.ndarray], valid: np.ndarray,
                    summary: dict, dataset_label: str) -> None:
    panels = (
        ("log_dust_to_gas", "log_dg", "dg_lo", "dg_hi", r"$\log_{10}(\Sigma_{\rm dust}/\Sigma_{\rm gas})$"),
        ("log_dust_to_star", "log_ds", "ds_lo", "ds_hi", r"$\log_{10}(\Sigma_{\rm dust}/\Sigma_\star)$"),
        ("log_dust_to_metals", "log_dz", "dz_lo", "dz_hi", r"$\log_{10}[\Sigma_{\rm dust}/(\Sigma_{Z,gas}+\Sigma_{\rm dust})]$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    for ax, (simkey, obskey, elo, ehi, ylabel) in zip(axes, panels):
        x, y = pixels["oh12"][valid], pixels[simkey][valid]
        q = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(q) >= 3:
            hb = ax.hexbin(x[q], y[q], gridsize=18, mincnt=1, cmap="Blues", linewidths=0.2)
            if ax is axes[-1]:
                fig.colorbar(hb, ax=ax, label="CosmicGrain pixels")
        for row in LOPEZ:
            marker = "s" if row["strong_outflow"] else "o"
            face = "#d95f02" if row["strong_outflow"] else "#1b9e77"
            ax.errorbar(row["oh12"], row[obskey], xerr=row["oherr"],
                        yerr=np.array([[row[elo]], [row[ehi]]]), fmt=marker,
                        ms=6, mfc=face, mec="black", mew=0.5, ecolor=face,
                        elinewidth=0.8, capsize=2, zorder=4)
        sy = summary[simkey]["mean"]
        sx = summary["oh12"]["mean"]
        ax.scatter(sx, sy, marker="*", s=210, color="#6a3d9a", edgecolor="black",
                   linewidth=0.7, zorder=6, label=dataset_label)
        ax.set_xlabel(r"$12+\log_{10}({\rm O/H})$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].scatter([], [], marker="o", color="#1b9e77", edgecolor="black", label="Lopez: no strong outflow")
    axes[0].scatter([], [], marker="s", color="#d95f02", edgecolor="black", label="Lopez: strong outflow/AGN")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("CosmicGrain vs. Lopez et al. (2026): matched ~1 kpc resolved ratios")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_maps(path: Path, maps: Dict[str, np.ndarray], extent: Sequence[float], valid: np.ndarray) -> None:
    panels = (
        ("sigma_dust", r"$\log\Sigma_{dust}$", "magma"),
        ("sigma_gas", r"$\log\Sigma_{gas}$", "viridis"),
        ("sigma_star", r"$\log\Sigma_\star$", "cividis"),
        ("oh12", r"$12+\log({\rm O/H})$", "plasma"),
        ("log_dust_to_gas", r"$\log(\Sigma_d/\Sigma_g)$", "coolwarm"),
        ("log_dust_to_star", r"$\log(\Sigma_d/\Sigma_\star)$", "coolwarm"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes.flat, panels):
        data = maps[key].copy()
        if key.startswith("sigma_"):
            data = np.log10(np.where(data > 0, data, np.nan))
        data[~valid] = np.nan
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [pkpc]")
        ax.set_ylabel("y [pkpc]")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.style_file:
        plt.style.use(args.style_file)
    if args.pixel_kpc <= 0 or args.half_size_kpc <= 0 or args.half_depth_kpc <= 0:
        raise ValueError("Pixel size and map dimensions must be positive")

    output_dir = os.path.abspath(args.output_dir)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    snap, z_selected = select_snapshot(output_dir, args.target_redshift, args.snapshot)
    tracked = get_zoom_halo_series(
        output_dir, [snap], group_index=args.final_group_index,
        reference_snap=args.reference_snapshot, verbose=True,
        refine_reference_center=True, refine_historical_centers=False,
    )[snap]
    if tracked is None:
        raise RuntimeError(f"DM-ID progenitor tracking failed at snapshot {snap:03d}")

    files = snapshot_files(output_dir, snap)
    with h5py.File(files[0], "r") as f:
        header = f["Header"].attrs
        params = f["Parameters"].attrs
        a = float(header["Time"])
        h = float(params["HubbleParam"])
        box = float(header["BoxSize"])
    center = np.asarray(tracked.get("center_ckpch", tracked["center"]), dtype=float)
    r200 = float(tracked["r200_ckpch"]) * a / h
    mass_factor = MASS_CODE_TO_MSUN / h
    length_factor = a / h

    gas_avail, star_avail, dust_avail = (available_fields(files, pt) for pt in (0, 4, 6))
    zfield = choose_field(gas_avail, Z_FIELDS)
    ofield = choose_field(gas_avail, O_FIELDS)
    sfrfield = choose_field(gas_avail, SFR_FIELDS, required=False)
    gas_fields = ["Coordinates", "Velocities", zfield, ofield] + ([sfrfield] if sfrfield else [])
    gas = read_type(files, 0, gas_fields)
    stars = read_type(files, 4, ["Coordinates", "Velocities"])
    rfield = choose_field(dust_avail, RADIUS_FIELDS, required=False)
    cfield = choose_field(dust_avail, CARBON_FIELDS, required=False)
    srcfield = choose_field(dust_avail, SOURCE_FIELDS, required=False)
    dust_fields = ["Coordinates"] + [x for x in (rfield, cfield, srcfield) if x]
    dust = read_type(files, 6, dust_fields)

    def centered_pkpc(data):
        return periodic_delta(data["Coordinates"], center, box) * length_factor

    gxyz, sxyz, dxyz = centered_pkpc(gas), centered_pkpc(stars), centered_pkpc(dust)
    gm, sm, dm = (x["Masses"] * mass_factor for x in (gas, stars, dust))
    rotation, orientation_method, orientation_radius = choose_orientation(
        sxyz, stars["Velocities"], sm,
        gxyz, gas["Velocities"], gm,
        args.orientation_radius_kpc, r200,
    )
    gxyz, sxyz, dxyz = (xyz @ rotation.T for xyz in (gxyz, sxyz, dxyz))

    nside = max(2, int(round(2 * args.half_size_kpc / args.pixel_kpc)))
    edges = np.linspace(-args.half_size_kpc, args.half_size_kpc, nside + 1)
    pixel_area_pc2 = ((edges[1] - edges[0]) * 1.0e3) ** 2
    sigma_gas = project_sum(gxyz, gm, edges, args.half_depth_kpc) / pixel_area_pc2
    sigma_star = project_sum(sxyz, sm, edges, args.half_depth_kpc) / pixel_area_pc2
    sigma_dust = project_sum(dxyz, dm, edges, args.half_depth_kpc) / pixel_area_pc2
    dust_count = project_count(dxyz, edges, args.half_depth_kpc)
    sigma_zgas = project_sum(gxyz, gm * np.asarray(gas[zfield], float), edges, args.half_depth_kpc) / pixel_area_pc2

    oh_linear = (np.asarray(gas[ofield], float) / A_O) / X_H
    oh_weights = np.asarray(gas[sfrfield], float) if sfrfield else gm
    oh_map_linear = project_weighted(gxyz, oh_linear, oh_weights, edges, args.half_depth_kpc)
    oh12_map = 12.0 + np.log10(np.where(oh_map_linear > 0, oh_map_linear, np.nan))
    maps = dict(
        sigma_gas=sigma_gas, sigma_star=sigma_star, sigma_dust=sigma_dust,
        sigma_zgas=sigma_zgas, oh12=oh12_map,
        log_dust_to_gas=log_ratio(sigma_dust, sigma_gas),
        log_dust_to_star=log_ratio(sigma_dust, sigma_star),
        log_dust_to_metals=log_ratio(sigma_dust, sigma_zgas + sigma_dust),
    )

    # CosmicGrain-only diagnostics are mass-weighted in the same projected pixels.
    dust_mass_map = project_sum(dxyz, dm, edges, args.half_depth_kpc)
    dust_mass_den = np.where(dust_mass_map > 0, dust_mass_map, np.nan)
    if rfield:
        radius_um = np.asarray(dust[rfield], float).reshape(-1) * args.grain_radius_to_micron
        maps["small_grain_fraction"] = project_sum(
            dxyz, dm * (radius_um < args.small_grain_micron), edges, args.half_depth_kpc
        ) / dust_mass_den
    else:
        maps["small_grain_fraction"] = np.full_like(sigma_dust, np.nan)
    maps["carbon_fraction"] = (
        project_weighted(dxyz, np.asarray(dust[cfield], float).reshape(-1), dm, edges, args.half_depth_kpc)
        if cfield else np.full_like(sigma_dust, np.nan)
    )
    for code, label in ((0, "snii"), (1, "agb"), (2, "lrn")):
        maps[f"source_{label}_fraction"] = (
            project_sum(dxyz, dm * (np.asarray(dust[srcfield]).reshape(-1) == code), edges, args.half_depth_kpc)
            / dust_mass_den
            if srcfield else np.full_like(sigma_dust, np.nan)
        )

    valid = ((dust_count >= args.min_dust_particles) & (sigma_gas > 0) &
             (sigma_star > 0) & np.isfinite(oh12_map))
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(
            f"Only {np.count_nonzero(valid)} valid pixels. Lower --min-dust-particles "
            "or inspect the tracked halo and map geometry."
        )

    # A transparent simulation outflow proxy; not treated as equivalent to the
    # broad-line/AGN classification used by Lopez et al.
    gr = np.linalg.norm(gxyz, axis=1)
    radial_hat = np.divide(gxyz, gr[:, None], out=np.zeros_like(gxyz), where=gr[:, None] > 0)
    vbulk_q = gr < min(0.1 * r200, args.orientation_radius_kpc)
    vbulk = np.average(gas["Velocities"][vbulk_q], axis=0, weights=gm[vbulk_q]) if np.any(vbulk_q) else 0.0
    velocity_rot = (gas["Velocities"] - vbulk) @ rotation.T
    vr = np.einsum("ij,ij->i", velocity_rot, radial_hat) * math.sqrt(a)
    shell = (gr >= args.outflow_shell_r200[0] * r200) & (gr < args.outflow_shell_r200[1] * r200)
    shell_mass = np.sum(gm[shell])
    outflow_fraction = float(np.sum(gm[shell & (vr > args.outflow_speed_kms)]) / shell_mass) if shell_mass > 0 else float("nan")

    keys = ["oh12", "log_dust_to_gas", "log_dust_to_star", "log_dust_to_metals",
            "small_grain_fraction", "carbon_fraction", "source_snii_fraction",
            "source_agb_fraction", "source_lrn_fraction"]
    summary = {key: robust_stats(maps[key][valid]) for key in keys}
    summary.update(dict(snapshot=snap, redshift=float(z_selected), tracked_group_index=int(tracked["group_idx"]),
                        final_group_index=args.final_group_index, r200_pkpc=float(r200),
                        m200_msun=float(tracked["m200_msun"]),
                        orientation_method=orientation_method,
                        orientation_radius_pkpc=orientation_radius,
                        valid_pixels=int(np.count_nonzero(valid)), pixel_kpc=float(edges[1] - edges[0]),
                        footprint_kpc2=float(np.count_nonzero(valid) * (edges[1] - edges[0]) ** 2),
                        outflow_mass_fraction_gt_threshold=outflow_fraction,
                        outflow_speed_threshold_kms=args.outflow_speed_kms))

    yy, xx = np.indices(valid.shape)
    xc = 0.5 * (edges[:-1] + edges[1:])
    pixel_fields = ["x_pkpc", "y_pkpc", "dust_particles", "sigma_dust_msun_pc2",
                    "sigma_gas_msun_pc2", "sigma_star_msun_pc2"] + keys
    with (outdir / "cosmicgrain_lopez_pixels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pixel_fields)
        writer.writeheader()
        for j, i in zip(*np.where(valid)):
            row = dict(x_pkpc=xc[i], y_pkpc=xc[j], dust_particles=int(dust_count[j, i]),
                       sigma_dust_msun_pc2=sigma_dust[j, i], sigma_gas_msun_pc2=sigma_gas[j, i],
                       sigma_star_msun_pc2=sigma_star[j, i])
            row.update({key: maps[key][j, i] for key in keys})
            writer.writerow(row)
    observational_csv(outdir / "lopez_2026_table2_table3.csv")
    with (outdir / "cosmicgrain_lopez_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    dataset_label = f"CosmicGrain 2048, z={z_selected:.2f}"
    plot_comparison(outdir / "cosmicgrain_lopez_figure11_comparison.png", maps, valid, summary, dataset_label)
    plot_maps(outdir / "cosmicgrain_lopez_resolved_maps.png", maps,
              (-args.half_size_kpc, args.half_size_kpc, -args.half_size_kpc, args.half_size_kpc), valid)

    print("=" * 78)
    print("COSMICGRAIN -- LOPEZ ET AL. (2026) COMPARISON")
    print("=" * 78)
    print(f"snapshot={snap:03d}  z={z_selected:.4f}  tracked group={tracked['group_idx']}")
    print(f"M200c={tracked['m200_msun']:.3e} Msun  R200c={r200:.3f} pkpc")
    print(f"orientation={orientation_method}  radius={orientation_radius:.3f} pkpc")
    print(f"valid pixels={np.count_nonzero(valid)}  "
          f"footprint={summary['footprint_kpc2']:.2f} kpc^2")
    print(f"<12+log(O/H)>={summary['oh12']['mean']:.3f}")
    print(f"<log D/G>={summary['log_dust_to_gas']['mean']:.3f}")
    print(f"<log D/star>={summary['log_dust_to_star']['mean']:.3f}")
    print(f"<log D/Z>={summary['log_dust_to_metals']['mean']:.3f}")
    print(f"outflow proxy f(Mgas, vr>{args.outflow_speed_kms:g} km/s)={outflow_fraction:.4f}")
    print(f"Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
