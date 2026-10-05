#!/usr/bin/env python3
"""Compare the three controlled CosmicGrain enrichment runs.

Default run order:
  1. legacy        : instantaneous + delayed enrichment
  2. delayed_only  : delayed enrichment, before stellar-metallicity fix
  3. starmet_fix   : delayed enrichment, after stellar-metallicity fix

The script reads every HDF5 piece of one matched snapshot, prints absolute
metrics for all runs, and reports the two controlled steps and total change.
It requires only numpy and h5py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np


DEFAULT_RUNS = (
    ("pre_oxygen", "halo441_output_1024_pre_oxygen"),
    ("before_starmetfix", "halo441_output_1024_before_starmetfix"),
    ("after_starmetfix", "halo441_output_1024"),
)

ELEMENT_FIELDS = {
    "C": "GasCarbonMassFraction",
    "N": "GasNitrogenMassFraction",
    "O": "GasOxygenMassFraction",
    "Ne": "GasNeonMassFraction",
    "Mg": "GasMagnesiumMassFraction",
    "Si": "GasSiliconMassFraction",
    "Fe": "GasIronMassFraction",
}

SILICATE_OXYGEN_MASS_FRACTION = 0.69876708


@dataclass(frozen=True)
class SnapshotInfo:
    snap: int
    redshift: float
    files: Tuple[str, ...]
    git_commit: str


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def discover_complete_snapshots(root: str) -> Dict[int, SnapshotInfo]:
    found: Dict[int, SnapshotInfo] = {}
    pattern = os.path.join(root, "snapdir_*", "snapshot_*.0.hdf5")
    for first in sorted(glob.glob(pattern)):
        base = os.path.basename(first)
        try:
            snap = int(base.split(".")[0].split("_")[-1])
        except (IndexError, ValueError):
            continue
        with h5py.File(first, "r") as handle:
            header = handle["Header"].attrs
            expected = int(header["NumFilesPerSnapshot"])
            redshift = float(header["Redshift"])
            commit = decode_attr(header.get("Git_commit", "not recorded"))
        pieces = tuple(sorted(glob.glob(os.path.join(
            root, f"snapdir_{snap:03d}", f"snapshot_{snap:03d}.*.hdf5"
        ))))
        if len(pieces) == expected:
            found[snap] = SnapshotInfo(snap, redshift, pieces, commit)
    return found


def read_dataset(files: Sequence[str], ptype: str, field: str) -> np.ndarray:
    arrays = []
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if ptype in handle and field in handle[ptype]:
                arrays.append(handle[ptype][field][...])
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays)


def mass_unit_msun(first_file: str, override: float | None) -> float:
    if override is not None:
        return override
    with h5py.File(first_file, "r") as handle:
        attrs = handle["Header"].attrs
        # CosmicGrain/GADGET default: 1 code mass = 1e10 Msun/h.  Prefer a
        # recorded HubbleParam if present, otherwise use the suite cosmology.
        hubble = float(attrs.get("HubbleParam", 0.6732))
    return 1.0e10 / hubble


def safe_sum(values: np.ndarray) -> float:
    return float(np.sum(values[np.isfinite(values)], dtype=np.float64))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    denom = safe_sum(weights[good])
    return float(np.sum(values[good] * weights[good], dtype=np.float64) / denom) if denom > 0 else math.nan


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return math.nan
    v, w = values[good], weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w, dtype=np.float64) - 0.5 * w
    cumulative /= np.sum(w, dtype=np.float64)
    return float(np.interp(q, cumulative, v))


def abundance_oh(x_o: np.ndarray, x_h: np.ndarray) -> np.ndarray:
    result = np.full(x_o.shape, np.nan, dtype=np.float64)
    good = np.isfinite(x_o) & np.isfinite(x_h) & (x_o > 0) & (x_h > 0)
    result[good] = 12.0 + np.log10((x_o[good] / 16.0) / x_h[good])
    return result


def weighted_abundance_oh(x_o: np.ndarray, x_h: np.ndarray, weights: np.ndarray) -> float:
    """Return 12+log10 of the weighted mean *linear* O/H number ratio."""
    good = (np.isfinite(x_o) & np.isfinite(x_h) & np.isfinite(weights)
            & (x_o >= 0) & (x_h > 0) & (weights > 0))
    if not np.any(good):
        return math.nan
    denom = safe_sum(weights[good])
    linear_oh = np.sum(weights[good] * (x_o[good] / 16.0) / x_h[good], dtype=np.float64) / denom
    return 12.0 + math.log10(linear_oh) if linear_oh > 0 else math.nan


def summarize_run(info: SnapshotInfo, xh_mode: str, unit_override: float | None) -> Dict[str, float]:
    files = info.files
    unit = mass_unit_msun(files[0], unit_override)

    gas_mass_code = read_dataset(files, "PartType0", "Masses").astype(np.float64)
    gas_mass = gas_mass_code * unit
    gas_z = read_dataset(files, "PartType0", "Metallicity").astype(np.float64)
    sfr = read_dataset(files, "PartType0", "StarFormationRate").astype(np.float64)
    elements = {
        symbol: read_dataset(files, "PartType0", field).astype(np.float64)
        for symbol, field in ELEMENT_FIELDS.items()
    }

    star_mass = read_dataset(files, "PartType4", "Masses").astype(np.float64) * unit
    star_z = read_dataset(files, "PartType4", "Metallicity").astype(np.float64)

    dust_mass = read_dataset(files, "PartType6", "Masses").astype(np.float64) * unit
    dust_carbon = read_dataset(files, "PartType6", "CarbonMassFraction").astype(np.float64)

    if not (gas_mass.size == gas_z.size == sfr.size):
        raise RuntimeError("Gas Masses, Metallicity, and StarFormationRate lengths differ")
    if any(arr.size != gas_mass.size for arr in elements.values()):
        raise RuntimeError("One or more gas element arrays do not match the gas-particle count")
    if star_mass.size != star_z.size:
        raise RuntimeError("Stellar Masses and Metallicity lengths differ")
    if dust_mass.size != dust_carbon.size:
        raise RuntimeError("Dust Masses and CarbonMassFraction lengths differ")

    if xh_mode == "metal_corrected":
        x_h = 0.76 * np.clip(1.0 - gas_z, 0.0, None)
    else:
        x_h = np.full(gas_z.shape, 0.76)

    x_o = elements["O"]
    particle_oh = abundance_oh(x_o, x_h)
    sf = np.isfinite(sfr) & (sfr > 0)

    total_gas = safe_sum(gas_mass)
    total_star = safe_sum(star_mass)
    total_dust = safe_sum(dust_mass)
    gas_metal_mass = safe_sum(gas_mass * gas_z)
    element_masses = {k: safe_sum(gas_mass * v) for k, v in elements.items()}
    dust_o_mass = safe_sum(dust_mass * (1.0 - dust_carbon) * SILICATE_OXYGEN_MASS_FRACTION)
    total_o = element_masses["O"] + dust_o_mass

    total_h_mass = safe_sum(gas_mass * x_h)
    global_oh = (
        12.0 + math.log10((element_masses["O"] / 16.0) / total_h_mass)
        if element_masses["O"] > 0 and total_h_mass > 0 else math.nan
    )

    tracked_sum = np.zeros(gas_z.shape, dtype=np.float64)
    for arr in elements.values():
        tracked_sum += arr
    tolerance = np.maximum(1.0e-10, 1.0e-5 * np.abs(gas_z))

    out: Dict[str, float] = {
        "redshift": info.redshift,
        "gas_particles": float(gas_mass.size),
        "star_particles": float(star_mass.size),
        "dust_particles": float(dust_mass.size),
        "gas_mass_msun": total_gas,
        "stellar_mass_msun": total_star,
        "dust_mass_msun": total_dust,
        "sfr_msun_yr": safe_sum(sfr),
        "gas_metal_mass_msun": gas_metal_mass,
        "gas_oxygen_mass_msun": element_masses["O"],
        "dust_oxygen_mass_msun": dust_o_mass,
        "gas_plus_dust_oxygen_msun": total_o,
        "all_gas_12logOH": global_oh,
        "sf_gas_mass_weighted_12logOH": weighted_abundance_oh(x_o[sf], x_h[sf], gas_mass[sf]),
        "sf_gas_sfr_weighted_12logOH": weighted_abundance_oh(x_o[sf], x_h[sf], sfr[sf]),
        "sf_gas_median_12logOH": float(np.nanmedian(particle_oh[sf])) if np.any(sf) else math.nan,
        "dust_to_gas": total_dust / total_gas if total_gas > 0 else math.nan,
        "dust_to_gas_phase_metals": total_dust / gas_metal_mass if gas_metal_mass > 0 else math.nan,
        "dust_fraction_total_metals": total_dust / (total_dust + gas_metal_mass) if total_dust + gas_metal_mass > 0 else math.nan,
        "oxygen_fraction_in_dust": dust_o_mass / total_o if total_o > 0 else math.nan,
        "stellar_Z_mass_weighted": weighted_mean(star_z, star_mass),
        "stellar_Z_median_mass_weighted": weighted_quantile(star_z, star_mass, 0.50),
        "stellar_Z_p16_mass_weighted": weighted_quantile(star_z, star_mass, 0.16),
        "stellar_Z_p84_mass_weighted": weighted_quantile(star_z, star_mass, 0.84),
        "stellar_Z_zero_fraction_number": float(np.mean(star_z <= 0)) if star_z.size else math.nan,
        "tracked_elements_gt_Z_count": float(np.count_nonzero(tracked_sum > gas_z + tolerance)),
        "negative_element_count": float(sum(np.count_nonzero(arr < 0) for arr in elements.values())),
        "nonfinite_element_count": float(sum(np.count_nonzero(~np.isfinite(arr)) for arr in elements.values())),
        "negative_Z_count": float(np.count_nonzero(gas_z < 0) + np.count_nonzero(star_z < 0)),
        "nonfinite_Z_count": float(np.count_nonzero(~np.isfinite(gas_z)) + np.count_nonzero(~np.isfinite(star_z))),
    }
    for symbol, mass in element_masses.items():
        out[f"gas_{symbol}_mass_msun"] = mass
    return out


METRIC_LABELS = (
    ("gas_mass_msun", "Gas mass [Msun]", "ratio"),
    ("stellar_mass_msun", "Stellar mass [Msun]", "ratio"),
    ("dust_mass_msun", "Dust mass [Msun]", "ratio"),
    ("sfr_msun_yr", "SFR [Msun/yr]", "ratio"),
    ("gas_metal_mass_msun", "Gas metal mass [Msun]", "ratio"),
    ("gas_oxygen_mass_msun", "Gas oxygen mass [Msun]", "ratio"),
    ("dust_oxygen_mass_msun", "Dust oxygen mass [Msun]", "ratio"),
    ("gas_plus_dust_oxygen_msun", "Gas+dust oxygen [Msun]", "ratio"),
    ("all_gas_12logOH", "All-gas 12+log(O/H)", "dex"),
    ("sf_gas_mass_weighted_12logOH", "SF-gas mass-weighted O/H", "dex"),
    ("sf_gas_sfr_weighted_12logOH", "SF-gas SFR-weighted O/H", "dex"),
    ("sf_gas_median_12logOH", "SF-gas median O/H", "dex"),
    ("dust_to_gas", "Dust/gas", "ratio"),
    ("dust_to_gas_phase_metals", "Dust/gas-phase metals", "ratio"),
    ("dust_fraction_total_metals", "Dust fraction total metals", "absolute"),
    ("oxygen_fraction_in_dust", "Oxygen fraction in dust", "absolute"),
    ("stellar_Z_mass_weighted", "Stellar Z, mass-weighted", "ratio"),
    ("stellar_Z_median_mass_weighted", "Stellar Z, weighted median", "ratio"),
    ("stellar_Z_p16_mass_weighted", "Stellar Z, weighted p16", "ratio"),
    ("stellar_Z_p84_mass_weighted", "Stellar Z, weighted p84", "ratio"),
)


def change_text(old: float, new: float, kind: str) -> str:
    if not (math.isfinite(old) and math.isfinite(new)):
        return "n/a"
    if kind == "dex":
        return f"{new - old:+.4f} dex"
    if kind == "absolute":
        return f"{new - old:+.6e}"
    if old == 0:
        return "n/a (old=0)"
    ratio = new / old
    return f"{ratio:.6f}x ({(ratio - 1.0) * 100:+.2f}%)"


def print_table(labels: Sequence[str], summaries: Mapping[str, Mapping[str, float]]) -> None:
    width = 31
    print("\nABSOLUTE METRICS")
    print("=" * (width + 24 * len(labels)))
    print(f"{'Metric':<{width}}" + "".join(f"{label:>24}" for label in labels))
    print("-" * (width + 24 * len(labels)))
    for key, display, _ in METRIC_LABELS:
        print(f"{display:<{width}}" + "".join(f"{summaries[label][key]:>24.8e}" for label in labels))

    pairs = (
        (labels[0], labels[1], "remove instantaneous enrichment"),
        (labels[1], labels[2], "stellar-metallicity fix"),
        (labels[0], labels[2], "total correction"),
    )
    for old_label, new_label, title in pairs:
        print(f"\nCHANGE: {old_label} -> {new_label} ({title})")
        print("-" * 105)
        print(f"{'Metric':<39}{'old':>20}{'new':>20}{'change':>26}")
        for key, display, kind in METRIC_LABELS:
            old, new = summaries[old_label][key], summaries[new_label][key]
            print(f"{display:<39}{old:>20.8e}{new:>20.8e}{change_text(old, new, kind):>26}")


def write_csv(path: str, labels: Sequence[str], summaries: Mapping[str, Mapping[str, float]]) -> None:
    all_keys = sorted({key for summary in summaries.values() for key in summary})
    metric_meta = {key: (display, kind) for key, display, kind in METRIC_LABELS}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "display_name", "change_type", *labels,
                         f"{labels[0]}_to_{labels[1]}",
                         f"{labels[1]}_to_{labels[2]}",
                         f"{labels[0]}_to_{labels[2]}"])
        for key in all_keys:
            display, kind = metric_meta.get(key, (key, "absolute"))
            values = [summaries[label].get(key, math.nan) for label in labels]
            writer.writerow([key, display, kind, *values,
                             change_text(values[0], values[1], kind),
                             change_text(values[1], values[2], kind),
                             change_text(values[0], values[2], kind)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pre", default=DEFAULT_RUNS[0][1], help="legacy/pre-oxygen output directory")
    parser.add_argument("--before", default=DEFAULT_RUNS[1][1], help="delayed-only directory before stellar-metallicity fix")
    parser.add_argument("--after", default=DEFAULT_RUNS[2][1], help="current directory after stellar-metallicity fix")
    parser.add_argument("--snap", type=int, help="snapshot number; default is latest complete snapshot common to all runs")
    parser.add_argument("--csv", default="halo441_three_version_chemistry_comparison.csv")
    parser.add_argument("--xh-mode", choices=("fixed", "metal_corrected"), default="fixed",
                        help="hydrogen fraction used only for O/H reporting (default: fixed X_H=0.76)")
    parser.add_argument("--mass-unit-msun", type=float,
                        help="Msun per code-mass unit; default is 1e10/HubbleParam (or h=0.6732)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_paths = (
        ("pre_oxygen", args.pre),
        ("before_starmetfix", args.before),
        ("after_starmetfix", args.after),
    )
    available = {label: discover_complete_snapshots(path) for label, path in run_paths}
    for label, path in run_paths:
        if not available[label]:
            print(f"ERROR: no complete snapshots found for {label}: {path}", file=sys.stderr)
            return 2

    common = sorted(set.intersection(*(set(available[label]) for label, _ in run_paths)))
    if not common:
        print("ERROR: the three runs have no complete common snapshots", file=sys.stderr)
        return 2
    snap = args.snap if args.snap is not None else common[-1]
    if snap not in common:
        print(f"ERROR: snapshot {snap:03d} is not complete in all three runs. Common: {common}", file=sys.stderr)
        return 2

    labels = [label for label, _ in run_paths]
    infos = {label: available[label][snap] for label in labels}
    redshifts = [infos[label].redshift for label in labels]
    if max(redshifts) - min(redshifts) > 1.0e-7:
        print(f"ERROR: matched snapshot numbers have different redshifts: {redshifts}", file=sys.stderr)
        return 2

    print("=" * 103)
    print(f"COSMICGRAIN THREE-VERSION COMPARISON  snap={snap:03d}  z={redshifts[0]:.8f}  X_H={args.xh_mode}")
    print("=" * 103)
    for label, path in run_paths:
        print(f"{label:20s} {path}")
        print(f"{'':20s} Git commit: {infos[label].git_commit}")
    print(f"Complete common snapshots: {common}")

    summaries = {
        label: summarize_run(infos[label], args.xh_mode, args.mass_unit_msun)
        for label in labels
    }
    print_table(labels, summaries)

    print("\nGAS-PHASE ELEMENT MASSES: CONTROLLED STEP RATIOS")
    print("-" * 81)
    print(f"{'Element':<9}{'remove instant.':>18}{'stellar-Z fix':>18}{'total':>18}")
    for symbol in ELEMENT_FIELDS:
        key = f"gas_{symbol}_mass_msun"
        a, b, c = (summaries[label][key] for label in labels)
        fmt = lambda numerator, denominator: f"{numerator / denominator:.6f}x" if denominator != 0 else "n/a"
        print(f"{symbol:<9}{fmt(b, a):>18}{fmt(c, b):>18}{fmt(c, a):>18}")

    print("\nINTEGRITY CHECKS")
    print("-" * 81)
    for key in ("tracked_elements_gt_Z_count", "negative_element_count", "nonfinite_element_count",
                "negative_Z_count", "nonfinite_Z_count"):
        values = "  ".join(f"{label}={summaries[label][key]:.0f}" for label in labels)
        print(f"{key:35s} {values}")

    write_csv(args.csv, labels, summaries)
    print(f"\nWrote {args.csv}")
    print("NOTE: whole-volume metrics isolate code changes; paper comparisons should use the target-galaxy aperture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
