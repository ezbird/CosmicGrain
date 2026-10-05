#!/usr/bin/env python3
"""Matched-redshift CG10 convergence test: 1024 versus 2048 at snapshot 019.

Place this script beside ``halo_utils.py``, ``halo441_rv_grain_physics.py``,
``cosmicgrain.mplstyle``, and the ``dust_optics`` directory.  Run from the
GADGET-4 repository root::

    python3 scripts/cosmicgrain_cg10_convergence_snap019.py

The comparison deliberately uses fixed physical selections in both runs:

* integrated and grain-population measurements: r < 30 pkpc (3D sphere),
* orientation: stellar angular momentum inside 20 pkpc,
* extinction projection: R < 20 pkpc, |z| < 20 pkpc, 0.5 pkpc pixels,
* valid R(V): identical optical tables, signal cuts, and dust-count cuts.

Outputs include PNG/PDF figures plus summary, GSD, CDF, and pixel CSV files and
a JSON record of the complete analysis contract.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

import halo441_rv_grain_physics as core
from halo_utils import get_zoom_halo


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MSUN_PER_CODE = 1.0e10


def apply_style(style_file: str | None) -> str:
    candidates = []
    if style_file:
        candidates.append(Path(style_file))
    candidates.extend([
        SCRIPT_DIR / "cosmicgrain.mplstyle",
        Path.cwd() / "cosmicgrain.mplstyle",
    ])
    for candidate in candidates:
        if candidate.is_file():
            plt.style.use(candidate)
            print(f"Using Matplotlib style: {candidate.resolve()}")
            return str(candidate.resolve())
    raise FileNotFoundError(
        "cosmicgrain.mplstyle not found beside the script or in the current "
        "directory; pass --style-file PATH"
    )


def snapshot_files(output: Path, snap: int) -> list[str]:
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
    raise FileNotFoundError(f"{output}: snapshot {snap:03d} not found")


def snapshot_pattern(output: Path, snap: int, files: list[str]) -> str:
    pattern = str(
        output / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.*.hdf5"
    )
    return pattern if glob.glob(pattern) else files[0]


def disk_rotation(files: list[str], center: np.ndarray, box: float,
                  a: float, h: float, radius_pkpc: float) -> np.ndarray:
    coords, velocities, masses = [], [], []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "PartType4" not in handle:
                continue
            stars = handle["PartType4"]
            for field in ("Coordinates", "Velocities", "Masses"):
                if field not in stars:
                    raise KeyError(f"{filename}: PartType4/{field} missing")
            coords.append(np.asarray(stars["Coordinates"], dtype=np.float64))
            velocities.append(np.asarray(stars["Velocities"], dtype=np.float64))
            masses.append(np.asarray(stars["Masses"], dtype=np.float64))
    if not coords:
        raise RuntimeError("No stars available for face-on orientation")

    coords = np.concatenate(coords)
    velocities = np.concatenate(velocities)
    masses = np.concatenate(masses)
    xyz = core.rotate_positions(coords, center, box, a, h, np.eye(3))
    use = (
        (np.linalg.norm(xyz, axis=1) < radius_pkpc)
        & np.isfinite(masses) & (masses > 0)
    )
    if np.count_nonzero(use) < 20:
        raise RuntimeError(
            f"Only {np.count_nonzero(use)} stars within {radius_pkpc:g} "
            "pkpc for orientation"
        )
    vbulk = np.average(velocities[use], axis=0, weights=masses[use])
    angular = np.sum(
        np.cross(xyz[use], velocities[use] - vbulk) * masses[use, None],
        axis=0,
    )
    norm = np.linalg.norm(angular)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Stellar angular momentum is undefined")
    zhat = angular / norm
    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(zhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    xhat = reference - np.dot(reference, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    return np.vstack([xhat, np.cross(zhat, xhat), zhat])


def _metallicity_vector(dataset: h5py.Dataset, filename: str) -> np.ndarray:
    values = np.asarray(dataset, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"{filename}: PartType0/Metallicity has shape {values.shape}; "
            "expected the scalar total-metal mass fraction"
        )
    return values


def aperture_census(files: list[str], center: np.ndarray, box: float,
                    a: float, h: float, aperture_pkpc: float) -> dict:
    totals = {
        "gas_mass_code": 0.0,
        "star_mass_code": 0.0,
        "gas_metal_mass_code": 0.0,
        "sfr_msun_yr": 0.0,
        "n_gas": 0,
        "n_star": 0,
    }
    for filename in files:
        with h5py.File(filename, "r") as handle:
            for ptype, mass_key, count_key in (
                (0, "gas_mass_code", "n_gas"),
                (4, "star_mass_code", "n_star"),
            ):
                name = f"PartType{ptype}"
                if name not in handle:
                    continue
                group = handle[name]
                if "Coordinates" not in group or "Masses" not in group:
                    raise KeyError(f"{filename}: {name} lacks Coordinates/Masses")
                coords = np.asarray(group["Coordinates"], dtype=np.float64)
                mass = np.asarray(group["Masses"], dtype=np.float64)
                xyz = core.rotate_positions(
                    coords, center, box, a, h, np.eye(3)
                )
                use = (
                    (np.linalg.norm(xyz, axis=1) < aperture_pkpc)
                    & np.isfinite(mass) & (mass > 0)
                )
                totals[mass_key] += float(np.sum(mass[use], dtype=np.float64))
                totals[count_key] += int(np.count_nonzero(use))
                if ptype == 0:
                    if "Metallicity" not in group:
                        raise KeyError(f"{filename}: PartType0/Metallicity missing")
                    metallicity = _metallicity_vector(
                        group["Metallicity"], filename
                    )
                    totals["gas_metal_mass_code"] += float(
                        np.sum(mass[use] * metallicity[use], dtype=np.float64)
                    )
                    if "StarFormationRate" not in group:
                        raise KeyError(
                            f"{filename}: PartType0/StarFormationRate missing"
                        )
                    sfr = np.asarray(
                        group["StarFormationRate"], dtype=np.float64
                    )
                    totals["sfr_msun_yr"] += float(
                        np.sum(sfr[use], dtype=np.float64)
                    )
    conversion = MSUN_PER_CODE / h
    totals["mgas_msun"] = totals["gas_mass_code"] * conversion
    totals["mstar_msun"] = totals["star_mass_code"] * conversion
    totals["mgas_metals_msun"] = (
        totals["gas_metal_mass_code"] * conversion
    )
    return totals


def sfr_diagnostics(files: list[str], center: np.ndarray, box: float,
                    a: float, h: float,
                    apertures_pkpc: tuple[float, ...] = (10.0, 20.0, 30.0),
                    averaging_myr: tuple[float, ...] = (50.0, 100.0)) -> dict:
    """Measure instantaneous and recent-time-averaged SFR consistently.

    Instantaneous rates are summed from gas-particle ``StarFormationRate``.
    Recent averages use stellar mass currently inside each aperture divided by
    the requested age window.  Initial stellar mass is preferred when the
    snapshot provides it; otherwise current stellar mass is used and the
    adopted field is reported in the output.
    """
    apertures = tuple(sorted(set(float(x) for x in apertures_pkpc)))
    windows = tuple(sorted(set(float(x) for x in averaging_myr)))
    result = {}
    for radius in apertures:
        tag = f"{radius:g}pkpc"
        result[f"sfr_instantaneous_{tag}_msun_yr"] = 0.0
        result[f"n_star_forming_gas_{tag}"] = 0
        for window in windows:
            result[f"sfr_stars_{window:g}myr_{tag}_msun_yr"] = 0.0
            result[f"n_stars_{window:g}myr_{tag}"] = 0

    sf_radii, sf_rates = [], []
    stellar_mass_field = None
    omega_m, omega_l = 0.3158, 0.6842
    with h5py.File(files[0], "r") as handle:
        if "Parameters" in handle:
            params = handle["Parameters"].attrs
            omega_m = float(params.get("Omega0", omega_m))
            omega_l = float(params.get("OmegaLambda", omega_l))
    now_gyr = core.cosmic_time_gyr(a, h, omega_m, omega_l)

    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "PartType0" in handle:
                gas = handle["PartType0"]
                for field in ("Coordinates", "StarFormationRate"):
                    if field not in gas:
                        raise KeyError(f"{filename}: PartType0/{field} missing")
                xyz = core.rotate_positions(
                    np.asarray(gas["Coordinates"], dtype=np.float64),
                    center, box, a, h, np.eye(3),
                )
                radius = np.linalg.norm(xyz, axis=1)
                rate = np.asarray(gas["StarFormationRate"], dtype=np.float64)
                active = np.isfinite(rate) & (rate > 0)
                central_active = active & (radius < max(apertures))
                sf_radii.append(radius[central_active])
                sf_rates.append(rate[central_active])
                for aperture in apertures:
                    tag = f"{aperture:g}pkpc"
                    use = active & (radius < aperture)
                    result[f"sfr_instantaneous_{tag}_msun_yr"] += float(
                        np.sum(rate[use], dtype=np.float64)
                    )
                    result[f"n_star_forming_gas_{tag}"] += int(np.count_nonzero(use))

            if "PartType4" not in handle:
                continue
            stars = handle["PartType4"]
            form_name = next((name for name in (
                "StellarFormationTime", "GFM_StellarFormationTime"
            ) if name in stars), None)
            if form_name is None or "Coordinates" not in stars or "Masses" not in stars:
                continue
            mass_name = next((name for name in (
                "InitialMass", "GFM_InitialMass", "BirthMass"
            ) if name in stars), "Masses")
            if stellar_mass_field is None:
                stellar_mass_field = mass_name
            elif stellar_mass_field != mass_name:
                raise RuntimeError("Inconsistent stellar mass fields across snapshot pieces")
            xyz = core.rotate_positions(
                np.asarray(stars["Coordinates"], dtype=np.float64),
                center, box, a, h, np.eye(3),
            )
            radius = np.linalg.norm(xyz, axis=1)
            formation_a = np.asarray(stars[form_name], dtype=np.float64)
            mass = np.asarray(stars[mass_name], dtype=np.float64)
            valid = (
                np.isfinite(formation_a) & (formation_a > 0) & (formation_a <= a)
                & np.isfinite(mass) & (mass > 0)
            )
            age_myr = np.full(formation_a.shape, np.nan, dtype=np.float64)
            age_myr[valid] = 1000.0 * (
                now_gyr - core.cosmic_time_gyr(formation_a[valid], h, omega_m, omega_l)
            )
            mass_msun = mass * MSUN_PER_CODE / h
            for aperture in apertures:
                tag = f"{aperture:g}pkpc"
                spatial = radius < aperture
                for window in windows:
                    use = valid & spatial & (age_myr <= window)
                    result[f"sfr_stars_{window:g}myr_{tag}_msun_yr"] += float(
                        np.sum(mass_msun[use], dtype=np.float64) / (window * 1.0e6)
                    )
                    result[f"n_stars_{window:g}myr_{tag}"] += int(np.count_nonzero(use))

    if sf_radii:
        radii = np.concatenate(sf_radii)
        rates = np.concatenate(sf_rates)
        if len(radii) and np.sum(rates) > 0:
            order = np.argsort(radii)
            cumulative = np.cumsum(rates[order])
            result["sfr_weighted_median_radius_pkpc"] = float(
                radii[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]
            )
            result["maximum_star_forming_gas_radius_pkpc"] = float(np.max(radii))
        else:
            result["sfr_weighted_median_radius_pkpc"] = np.nan
            result["maximum_star_forming_gas_radius_pkpc"] = np.nan
    result["recent_sfr_stellar_mass_field"] = stellar_mass_field or "unavailable"
    result["recent_sfr_note"] = (
        "formed stellar mass divided by age window; current stellar mass "
        "underestimates formed mass when no initial/birth-mass field is available"
    )
    return result


def distribution_per_dlog(radius: np.ndarray, weights: np.ndarray,
                          edges: np.ndarray, normalization: float) -> np.ndarray:
    hist = np.histogram(radius, bins=edges, weights=weights)[0]
    return hist / normalization / np.diff(np.log10(edges))


def weighted_cdf(values: np.ndarray, weights: np.ndarray,
                 grid: np.ndarray) -> np.ndarray:
    good = (
        np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    )
    values, weights = values[good], weights[good]
    if not len(values) or np.sum(weights) <= 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    indices = np.searchsorted(values, grid, side="right") - 1
    result = np.zeros_like(grid, dtype=float)
    use = indices >= 0
    result[use] = cumulative[indices[use]]
    return result


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    good = (
        np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    )
    values, weights = values[good], weights[good]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5*np.sum(weights))])


def analyze_run(label: str, output: Path, snap: int, group_index: int,
                carbon_table, silicate_table, args) -> dict:
    print("\n" + "=" * 80)
    print(f"ANALYZING {label}: {output}")
    print("=" * 80)
    files = snapshot_files(output, snap)
    halo = get_zoom_halo(
        output, snap, group_index=group_index,
        refine_center=True, verbose=True,
    )
    center = np.asarray(halo.chosen_center_ckpch, dtype=float)
    a, h, box = halo.a, halo.h, halo.boxsize_ckpch
    census = aperture_census(
        files, center, box, a, h, args.aperture_pkpc
    )
    sfr_info = sfr_diagnostics(
        files, center, box, a, h,
        apertures_pkpc=(10.0, 20.0, args.aperture_pkpc),
        averaging_myr=(50.0, 100.0),
    )

    pattern = snapshot_pattern(output, snap, files)
    dust = core.read_dust_snapshot(pattern)
    dust_xyz = core.rotate_positions(
        dust["coords"], center, box, a, h, np.eye(3)
    )
    dust_ok = (
        np.isfinite(dust["mass"]) & (dust["mass"] > 0)
        & np.isfinite(dust["radius_nm"]) & (dust["radius_nm"] > 0)
        & np.isfinite(dust["carbon"])
    )
    in_sphere = dust_ok & (
        np.linalg.norm(dust_xyz, axis=1) < args.aperture_pkpc
    )
    mass = dust["mass"][in_sphere]
    radius = dust["radius_nm"][in_sphere]
    carbon = np.clip(dust["carbon"][in_sphere], 0.0, 1.0)
    if not len(mass) or np.sum(mass) <= 0:
        raise RuntimeError(f"{label}: no valid dust inside the 3D aperture")

    conversion = MSUN_PER_CODE / h
    mdust_msun = float(np.sum(mass) * conversion)
    mgas = census["mgas_msun"]
    mzgas = census["mgas_metals_msun"]
    dgr = mdust_msun / mgas if mgas > 0 else np.nan
    dztot = mdust_msun / (mzgas + mdust_msun) if mzgas + mdust_msun > 0 else np.nan

    gsd_edges = np.logspace(
        np.log10(args.gsd_min_nm), np.log10(args.gsd_max_nm),
        args.gsd_bins + 1,
    )
    gsd_centers = np.sqrt(gsd_edges[:-1] * gsd_edges[1:])
    dust_mass_code = float(np.sum(mass))
    gsd_total = distribution_per_dlog(
        radius, mass, gsd_edges, dust_mass_code
    )
    gsd_carbon = distribution_per_dlog(
        radius, mass * carbon, gsd_edges, dust_mass_code
    )
    gsd_silicate = distribution_per_dlog(
        radius, mass * (1.0-carbon), gsd_edges, dust_mass_code
    )

    orientation = disk_rotation(
        files, center, box, a, h, args.orientation_radius_pkpc
    )
    projected = core.rotate_positions(
        dust["coords"], center, box, a, h, orientation
    )
    slab = (
        dust_ok
        & (np.abs(projected[:, 0]) < args.plot_radius_pkpc)
        & (np.abs(projected[:, 1]) < args.plot_radius_pkpc)
        & (np.abs(projected[:, 2]) < args.zmax_pkpc)
    )
    x, y = projected[slab, 0], projected[slab, 1]
    slab_mass = dust["mass"][slab]
    slab_radius = dust["radius_nm"][slab]
    slab_carbon = np.clip(dust["carbon"][slab], 0.0, 1.0)
    wb, wv = core.extinction_weights(
        slab_mass, slab_radius, slab_carbon,
        carbon_table, silicate_table,
    )
    maps = core.build_maps(
        x, y, slab_mass, wb, wv,
        args.plot_radius_pkpc, args.map_bins, 0.0,
        args.min_signal_fraction,
    )
    pixel_size = 2.0 * args.plot_radius_pkpc / args.map_bins
    axis = -args.plot_radius_pkpc + (
        np.arange(args.map_bins) + 0.5
    ) * pixel_size
    xx, yy = np.meshgrid(axis, axis)
    projected_radius = np.hypot(xx, yy)
    in_circle = projected_radius < args.plot_radius_pkpc
    ebv = maps["tauB"] - maps["tauV"]
    positive_ebv = in_circle & np.isfinite(ebv) & (ebv > 0)
    ebv_reference = (
        float(np.nanmax(ebv[positive_ebv]))
        if np.any(positive_ebv) else np.nan
    )
    ebv_floor = args.min_ebv_fraction * ebv_reference
    ebv_pass = positive_ebv & (ebv >= ebv_floor)
    finite_rv = in_circle & np.isfinite(maps["rv"])
    valid = (
        finite_rv & ebv_pass & (maps["count_raw"] >= args.min_count)
    )
    rv = maps["rv"][valid]
    if len(rv) < 3:
        raise RuntimeError(
            f"{label}: only {len(rv)} R(V) pixels pass the common cuts"
        )

    summary = {
        "label": label,
        "output": str(output.resolve()),
        "snapshot": snap,
        "redshift": float(1.0/a - 1.0),
        "group_index": group_index,
        "m200_msun": float(halo.so_m200_code * MSUN_PER_CODE / h),
        "r200_pkpc": float(halo.so_r200_ckpch * a / h),
        "mstar_30pkpc_msun": census["mstar_msun"],
        "mgas_30pkpc_msun": mgas,
        "mdust_30pkpc_msun": mdust_msun,
        "sfr_30pkpc_msun_yr": census["sfr_msun_yr"],
        "dust_to_gas_30pkpc": dgr,
        "dust_to_total_metals_30pkpc": dztot,
        "nstar_30pkpc": census["n_star"],
        "ngas_30pkpc": census["n_gas"],
        "ndust_30pkpc": int(np.count_nonzero(in_sphere)),
        "small_grain_mass_fraction": float(
            np.sum(mass[radius < args.small_grain_nm]) / dust_mass_code
        ),
        "mass_weighted_median_radius_nm": weighted_median(radius, mass),
        "dust_mass_weighted_carbon_fraction": float(
            np.sum(mass * carbon) / dust_mass_code
        ),
        "dust_mass_fraction_8_12nm": float(
            np.sum(mass[(radius >= 8.0) & (radius <= 12.0)]) / dust_mass_code
        ),
        "dust_mass_fraction_95_105nm": float(
            np.sum(mass[(radius >= 95.0) & (radius <= 105.0)]) / dust_mass_code
        ),
        "rv_finite_pixels_inside_R20": int(np.count_nonzero(finite_rv)),
        "rv_valid_pixels": int(len(rv)),
        "rv_median": float(np.median(rv)),
        "rv_p16": float(np.percentile(rv, 16)),
        "rv_p84": float(np.percentile(rv, 84)),
        "rv_min": float(np.min(rv)),
        "rv_max": float(np.max(rv)),
        "rv_coverage_fraction": float(np.count_nonzero(valid)/np.count_nonzero(in_circle)),
        "median_dust_count_valid_pixel": float(np.median(maps["count_raw"][valid])),
        "dust_in_projection_slab": int(np.count_nonzero(slab)),
        "ebv_floor": float(ebv_floor),
    }
    summary.update(sfr_info)
    for threshold in (1, 3, 5, 10):
        summary[f"rv_pixels_count_ge_{threshold}"] = int(np.count_nonzero(
            finite_rv & (maps["count_raw"] >= threshold)
        ))
    print(
        f"{label}: M200={summary['m200_msun']:.4e} Msun; "
        f"Mstar30={summary['mstar_30pkpc_msun']:.4e}; "
        f"Mgas30={mgas:.4e}; Mdust30={mdust_msun:.4e}; "
        f"SFR30={summary['sfr_30pkpc_msun_yr']:.4g}"
    )
    print(
        f"{label}: R(V) N={len(rv):,}; median={np.median(rv):.3f}; "
        f"16--84%={np.percentile(rv,16):.3f}--{np.percentile(rv,84):.3f}; "
        f"coverage={summary['rv_coverage_fraction']:.3f}"
    )
    print(f"{label}: SFR diagnostics")
    for aperture in (10.0, 20.0, args.aperture_pkpc):
        tag = f"{aperture:g}pkpc"
        print(
            f"  r<{aperture:g} pkpc: instantaneous="
            f"{summary[f'sfr_instantaneous_{tag}_msun_yr']:.4g}; "
            f"N_SFgas={summary[f'n_star_forming_gas_{tag}']:,}; "
            f"<SFR>50={summary[f'sfr_stars_50myr_{tag}_msun_yr']:.4g}; "
            f"<SFR>100={summary[f'sfr_stars_100myr_{tag}_msun_yr']:.4g}"
        )
    print(
        f"  SFR-weighted median gas radius="
        f"{summary['sfr_weighted_median_radius_pkpc']:.3f} pkpc; "
        f"stellar mass field={summary['recent_sfr_stellar_mass_field']}"
    )
    return {
        "summary": summary,
        "gsd_edges": gsd_edges,
        "gsd_centers": gsd_centers,
        "gsd_total": gsd_total,
        "gsd_carbon": gsd_carbon,
        "gsd_silicate": gsd_silicate,
        "radius": radius,
        "carbon": carbon,
        "dust_mass": mass,
        "rv": rv,
        "valid_count": maps["count_raw"][valid],
        "maps": maps,
        "valid": valid,
        "in_circle": in_circle,
        "ebv": ebv,
        "xx": xx,
        "yy": yy,
    }


def write_summary(results: list[dict], prefix: Path) -> Path:
    path = prefix.with_name(prefix.name + "_summary.csv")
    fields = list(results[0]["summary"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result["summary"])
    return path


def write_gsd(results: list[dict], prefix: Path) -> Path:
    path = prefix.with_name(prefix.name + "_gsd.csv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run", "radius_lo_nm", "radius_hi_nm", "radius_center_nm",
            "dMtotal_over_Mdust_dlog10a", "dMcarbon_over_Mdust_dlog10a",
            "dMsilicate_over_Mdust_dlog10a",
        ])
        for result in results:
            edges = result["gsd_edges"]
            for i, center in enumerate(result["gsd_centers"]):
                writer.writerow([
                    result["summary"]["label"], edges[i], edges[i+1], center,
                    result["gsd_total"][i], result["gsd_carbon"][i],
                    result["gsd_silicate"][i],
                ])
    return path


def write_cdfs(results: list[dict], prefix: Path) -> Path:
    path = prefix.with_name(prefix.name + "_cdfs.csv")
    radius_grid = np.logspace(0.0, np.log10(200.0), 240)
    carbon_grid = np.linspace(0.0, 1.0, 201)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "variable", "value", "dust_mass_cdf"])
        for result in results:
            label = result["summary"]["label"]
            for variable, grid, values in (
                ("grain_radius_nm", radius_grid, result["radius"]),
                ("carbon_mass_fraction", carbon_grid, result["carbon"]),
            ):
                cdf = weighted_cdf(values, result["dust_mass"], grid)
                writer.writerows(
                    [label, variable, value, cumulative]
                    for value, cumulative in zip(grid, cdf)
                )
    return path


def write_pixels(results: list[dict], prefix: Path) -> Path:
    path = prefix.with_name(prefix.name + "_rv_pixels.csv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run", "x_pkpc", "y_pkpc", "inside_R20", "passes_common_cuts",
            "Rv", "dust_count", "dust_mass_code", "tauB_proxy", "tauV_proxy",
            "EBV_proxy",
        ])
        for result in results:
            maps = result["maps"]
            for py, px in np.ndindex(maps["rv"].shape):
                writer.writerow([
                    result["summary"]["label"], result["xx"][py,px],
                    result["yy"][py,px], int(result["in_circle"][py,px]),
                    int(result["valid"][py,px]), maps["rv"][py,px],
                    maps["count_raw"][py,px], maps["mass_raw"][py,px],
                    maps["tauB_raw"][py,px], maps["tauV_raw"][py,px],
                    result["ebv"][py,px],
                ])
    return path


def make_figure(results: list[dict], prefix: Path, comparison: dict) -> tuple[Path, Path]:
    colors = {"1024": "#276FBF", "2048": "#D1495B"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)

    metrics = [
        ("m200_msun", r"$M_{200c}$"),
        ("mstar_30pkpc_msun", r"$M_\star$"),
        ("mgas_30pkpc_msun", r"$M_{\rm gas}$"),
        ("mdust_30pkpc_msun", r"$M_{\rm dust}$"),
        ("sfr_30pkpc_msun_yr", r"SFR$_{\rm inst}$"),
        ("sfr_stars_50myr_30pkpc_msun_yr", r"$\langle$SFR$\rangle_{50}$"),
        ("sfr_stars_100myr_30pkpc_msun_yr", r"$\langle$SFR$\rangle_{100}$"),
        ("dust_to_gas_30pkpc", r"$D/G$"),
        ("dust_to_total_metals_30pkpc", r"$D/Z$"),
    ]
    base, high = results[0]["summary"], results[1]["summary"]
    ratios = [high[key]/base[key] if base[key] != 0 else np.nan for key,_ in metrics]
    ax = axes[0,0]
    ax.axhline(1.0, color="0.4", ls="--", lw=1.2)
    ax.scatter(range(len(metrics)), ratios, s=65, color=colors["2048"], zorder=3)
    for i, ratio in enumerate(ratios):
        if np.isfinite(ratio) and ratio > 0:
            ax.annotate(f"{ratio:.2g}", (i, ratio), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(metrics)), [label for _,label in metrics], rotation=35, ha="right")
    ax.set_yscale("log")
    finite = np.asarray(ratios)[np.isfinite(ratios) & (np.asarray(ratios) > 0)]
    if len(finite):
        ax.set_ylim(max(0.05, finite.min()/1.4), max(2.0, finite.max()*1.4))
    ax.set_ylabel("2048 / 1024")
    ax.set_title("(a) Integrated convergence, $r<30$ pkpc")

    ax = axes[0,1]
    for result in results:
        label = result["summary"]["label"]
        ax.plot(result["gsd_centers"], result["gsd_total"],
                color=colors[label], lw=2.2, label=label)
    ax.set(xscale="log", yscale="log", xlabel="Grain radius $a$ (nm)",
           ylabel=r"$d(M_{\rm dust}/M_{\rm dust,tot})/d\log_{10}a$",
           title="(b) Total grain-size distribution")
    ax.legend(frameon=False)

    ax = axes[0,2]
    for result in results:
        label = result["summary"]["label"]
        ax.plot(result["gsd_centers"], result["gsd_carbon"],
                color=colors[label], lw=2.2, ls="-", label=f"{label}: carbon")
        ax.plot(result["gsd_centers"], result["gsd_silicate"],
                color=colors[label], lw=2.0, ls="--", label=f"{label}: silicate")
    ax.set(xscale="log", yscale="log", xlabel="Grain radius $a$ (nm)",
           ylabel=r"$d(M_{\rm component}/M_{\rm dust,tot})/d\log_{10}a$",
           title="(c) Composition-resolved GSD")
    ax.legend(frameon=False, fontsize="small")

    ax = axes[1,0]
    grid = np.linspace(0.0, 1.0, 201)
    for result in results:
        label = result["summary"]["label"]
        ax.plot(grid, weighted_cdf(result["carbon"], result["dust_mass"], grid),
                color=colors[label], lw=2.2, label=label)
    ax.set(xlabel="Carbonaceous mass fraction", ylabel="Cumulative dust mass fraction",
           title="(d) Grain composition")
    ax.legend(frameon=False)

    ax = axes[1,1]
    all_rv = np.concatenate([result["rv"] for result in results])
    positive = all_rv[np.isfinite(all_rv) & (all_rv > 0)]
    lo, hi = positive.min(), positive.max()
    rv_edges = np.logspace(np.log10(lo), np.log10(hi), 32)
    for result in results:
        label = result["summary"]["label"]
        ax.hist(result["rv"], bins=rv_edges, density=True, histtype="step",
                lw=2.2, color=colors[label],
                label=f"{label} (N={len(result['rv']):,})")
        ax.axvline(np.median(result["rv"]), color=colors[label], lw=1.2, ls=":")
    ax.set(xscale="log", xlabel=r"Projected $R(V)$", ylabel="Probability density",
           title="(e) Extinction-curve distribution")
    ax.legend(frameon=False)

    ax = axes[1,2]
    thresholds = np.array([1, 3, 5, 10])
    x = np.arange(len(thresholds))
    width = 0.36
    for offset, result in zip((-width/2, width/2), results):
        finite_map = result["in_circle"] & np.isfinite(result["maps"]["rv"])
        fractions = [
            np.count_nonzero(finite_map & (result["maps"]["count_raw"] >= threshold))
            / np.count_nonzero(result["in_circle"])
            for threshold in thresholds
        ]
        label = result["summary"]["label"]
        ax.bar(x+offset, fractions, width, color=colors[label], alpha=.85, label=label)
    ax.set_xticks(x, [rf"$N_{{\rm dust}}\geq{n}$" for n in thresholds])
    ax.set(xlabel="Sightline cut", ylabel="Fraction of $R<20$ pkpc pixels",
           title="(f) Projected sampling")
    ax.legend(frameon=False)

    fig.suptitle(
        "CG10 convergence at the closest available matched epoch: "
        rf"1024 $z={base['redshift']:.3f}$; "
        rf"2048 $z={high['redshift']:.3f}$"
    )
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-1024", default="halo441_output_1024_mingashsml0.0")
    parser.add_argument("--output-2048", default="halo441_output_2048_mingashsml0.0")
    parser.add_argument("--snapshot", type=int, default=19)
    parser.add_argument("--group-index", type=int, default=1)
    parser.add_argument(
        "--max-redshift-offset", type=float, default=0.02,
        help=("Maximum permitted |z1024-z2048| for the closest available "
              "matched outputs (default: 0.02)"),
    )
    parser.add_argument("--aperture-pkpc", type=float, default=30.0)
    parser.add_argument("--orientation-radius-pkpc", type=float, default=20.0)
    parser.add_argument("--plot-radius-pkpc", type=float, default=20.0)
    parser.add_argument("--zmax-pkpc", type=float, default=20.0)
    parser.add_argument("--pixel-size-pkpc", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-signal-fraction", type=float, default=1.0e-3)
    parser.add_argument("--min-ebv-fraction", type=float, default=1.0e-3)
    parser.add_argument("--small-grain-nm", type=float, default=50.0)
    parser.add_argument("--gsd-min-nm", type=float, default=1.0)
    parser.add_argument("--gsd-max-nm", type=float, default=200.0)
    parser.add_argument("--gsd-bins", type=int, default=30)
    parser.add_argument("--carbon-table", default=str(
        SCRIPT_DIR / "dust_optics/callqcomp.out_D16MGemt"))
    parser.add_argument("--silicate-table", default=str(
        SCRIPT_DIR / "dust_optics/callqcomp.out_DH16asil"))
    parser.add_argument("--style-file")
    parser.add_argument("--prefix", default=str(
        PROJECT_ROOT / "CG10_convergence_snap019"))
    args = parser.parse_args()

    if args.pixel_size_pkpc <= 0 or args.plot_radius_pkpc <= 0:
        parser.error("pixel size and projection radius must be positive")
    bins_float = 2.0 * args.plot_radius_pkpc / args.pixel_size_pkpc
    if not np.isclose(bins_float, round(bins_float)):
        parser.error("2*plot-radius must be an integer multiple of pixel size")
    args.map_bins = int(round(bins_float))
    if args.min_count < 1 or args.gsd_bins < 2:
        parser.error("min-count must be >=1 and gsd-bins >=2")
    if not (0 <= args.min_ebv_fraction < 1):
        parser.error("min-ebv-fraction must be in [0,1)")

    style = apply_style(args.style_file)
    root = args.root.resolve()
    carbon_table = core.read_draine_qext_table(args.carbon_table)
    silicate_table = core.read_draine_qext_table(args.silicate_table)
    specs = [
        ("1024", root / args.output_1024),
        ("2048", root / args.output_2048),
    ]
    for label, path in specs:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} output directory not found: {path}")

    results = [
        analyze_run(label, output, args.snapshot, args.group_index,
                    carbon_table, silicate_table, args)
        for label, output in specs
    ]
    z1024 = results[0]["summary"]["redshift"]
    z2048 = results[1]["summary"]["redshift"]
    redshift_offset = abs(z1024-z2048)
    if redshift_offset > args.max_redshift_offset:
        raise RuntimeError(
            f"The closest outputs differ by |delta z|={redshift_offset:.6g}, "
            f"exceeding --max-redshift-offset={args.max_redshift_offset:g}"
        )
    if redshift_offset > 1.0e-4:
        print(
            f"\nNOTE: closest available outputs are approximately, not "
            f"exactly, redshift matched: z1024={z1024:.6f}, "
            f"z2048={z2048:.6f}, |delta z|={redshift_offset:.6f}."
        )

    rv1024, rv2048 = results[0]["rv"], results[1]["rv"]
    comparison = {
        "z_1024": float(z1024),
        "z_2048": float(z2048),
        "absolute_redshift_offset": float(redshift_offset),
        "ks_statistic_rv": float(ks_2samp(rv1024, rv2048).statistic),
        "ks_pvalue_rv": float(ks_2samp(rv1024, rv2048).pvalue),
        "wasserstein_rv": float(wasserstein_distance(rv1024, rv2048)),
    }
    prefix = Path(args.prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = write_summary(results, prefix)
    gsd_path = write_gsd(results, prefix)
    cdf_path = write_cdfs(results, prefix)
    pixels_path = write_pixels(results, prefix)
    png, pdf = make_figure(results, prefix, comparison)

    json_path = prefix.with_name(prefix.name + "_analysis.json")
    contract = {
        "analysis": "CG10 matched-redshift convergence",
        "style_file": style,
        "aperture_pkpc_3d": args.aperture_pkpc,
        "orientation_radius_pkpc": args.orientation_radius_pkpc,
        "projection_radius_pkpc": args.plot_radius_pkpc,
        "projection_half_depth_pkpc": args.zmax_pkpc,
        "pixel_size_pkpc": args.pixel_size_pkpc,
        "map_bins": args.map_bins,
        "minimum_dust_count": args.min_count,
        "minimum_tauV_signal_fraction": args.min_signal_fraction,
        "minimum_EBV_fraction_of_aperture_max": args.min_ebv_fraction,
        "maximum_permitted_redshift_offset": args.max_redshift_offset,
        "small_grain_threshold_nm": args.small_grain_nm,
        "gsd_normalization": "dust mass fraction per dex in grain radius",
        "orientation": "independent stellar angular momentum within fixed radius",
        "carbon_optical_table": str(Path(args.carbon_table).resolve()),
        "silicate_optical_table": str(Path(args.silicate_table).resolve()),
        "runs": [result["summary"] for result in results],
        "rv_two_sample_statistics": comparison,
    }
    json_path.write_text(json.dumps(contract, indent=2) + "\n")

    print("\n" + "=" * 80)
    print("MATCHED-REDSHIFT COMPARISON")
    print("=" * 80)
    for key, value in comparison.items():
        print(f"{key}: {value:.6g}")
    for path in (png, pdf, summary_path, gsd_path, cdf_path,
                 pixels_path, json_path):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
