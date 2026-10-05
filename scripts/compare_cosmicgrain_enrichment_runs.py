#!/usr/bin/env python3
"""Compare elemental and dust budgets in matched CosmicGrain runs.

The default paths compare the legacy halo-441 1024 run with the
delayed-enrichment-only rerun.  Metrics are whole-zoom-volume quantities;
they are intended as a chemistry/bookkeeping diagnostic before a separate
target-galaxy aperture comparison.
"""

import argparse
import csv
import glob
import math
import os

import h5py
import numpy as np


ELEMENT_FIELDS = {
    "C": "GasCarbonMassFraction",
    "N": "GasNitrogenMassFraction",
    "O": "GasOxygenMassFraction",
    "Ne": "GasNeonMassFraction",
    "Mg": "GasMagnesiumMassFraction",
    "Si": "GasSiliconMassFraction",
    "Fe": "GasIronMassFraction",
}

SILICATE_O_MASS_FRACTION = 0.69876708


def snapshot_files(root, snap):
    pattern = os.path.join(
        root, f"snapdir_{snap:03d}", f"snapshot_{snap:03d}.*.hdf5"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    with h5py.File(files[0], "r") as f:
        expected = int(f["Header"].attrs["NumFilesPerSnapshot"])
    if len(files) != expected:
        raise RuntimeError(
            f"Incomplete snapshot {snap:03d} in {root}: "
            f"found {len(files)} files, expected {expected}"
        )
    return files


def complete_snapshots(root):
    result = {}
    pattern = os.path.join(root, "snapdir_*", "snapshot_*.0.hdf5")
    for first in sorted(glob.glob(pattern)):
        base = os.path.basename(first)
        snap = int(base.split(".")[0].split("_")[-1])
        try:
            files = snapshot_files(root, snap)
        except RuntimeError:
            continue
        with h5py.File(files[0], "r") as f:
            result[snap] = float(f["Header"].attrs["Redshift"])
    return result


def safe_log12(number_ratio):
    if number_ratio > 0.0 and math.isfinite(number_ratio):
        return 12.0 + math.log10(number_ratio)
    return float("nan")


def audit_snapshot(root, snap, h, x_h):
    totals = {
        "snap": snap,
        "redshift": float("nan"),
        "n_gas": 0,
        "n_sf_gas": 0,
        "n_star": 0,
        "n_dust": 0,
        "gas_mass_code": 0.0,
        "star_mass_code": 0.0,
        "dust_mass_code": 0.0,
        "gas_metal_mass_code": 0.0,
        "sfr_msun_yr": 0.0,
        "dust_carbon_mass_code": 0.0,
        "dust_silicate_mass_code": 0.0,
        "dust_oxygen_mass_code": 0.0,
        "sf_gas_mass_code": 0.0,
        "sf_oxygen_mass_code": 0.0,
        "sf_sfr_weight": 0.0,
        "sf_sfr_oh_numerator": 0.0,
        "tracked_gt_z_count": 0,
        "negative_element_count": 0,
        "nonfinite_element_count": 0,
    }
    for symbol in ELEMENT_FIELDS:
        totals[f"gas_{symbol}_mass_code"] = 0.0

    sf_log_oh = []
    sf_log_oh_weights = []

    files = snapshot_files(root, snap)
    for filename in files:
        with h5py.File(filename, "r") as f:
            totals["redshift"] = float(f["Header"].attrs["Redshift"])

            if "PartType0" in f:
                g = f["PartType0"]
                mass = np.asarray(g["Masses"], dtype=np.float64)
                zgas = np.asarray(g["Metallicity"], dtype=np.float64)
                sfr = np.asarray(g["StarFormationRate"], dtype=np.float64)
                totals["n_gas"] += mass.size
                totals["gas_mass_code"] += float(mass.sum())
                totals["gas_metal_mass_code"] += float(np.sum(mass * zgas))
                totals["sfr_msun_yr"] += float(np.sum(sfr))

                element_sum = np.zeros_like(mass)
                arrays = {}
                for symbol, field in ELEMENT_FIELDS.items():
                    if field not in g:
                        raise KeyError(f"{filename}: missing {field}")
                    values = np.asarray(g[field], dtype=np.float64)
                    arrays[symbol] = values
                    element_sum += np.where(np.isfinite(values), values, 0.0)
                    totals[f"gas_{symbol}_mass_code"] += float(
                        np.sum(mass * np.where(np.isfinite(values), values, 0.0))
                    )
                    totals["negative_element_count"] += int(np.count_nonzero(values < 0.0))
                    totals["nonfinite_element_count"] += int(
                        np.count_nonzero(~np.isfinite(values))
                    )

                valid_check = np.isfinite(zgas) & np.isfinite(element_sum)
                totals["tracked_gt_z_count"] += int(
                    np.count_nonzero(
                        valid_check
                        & (element_sum > zgas * (1.0 + 1.0e-5) + 1.0e-12)
                    )
                )

                xo = arrays["O"]
                sf = (
                    (sfr > 0.0)
                    & np.isfinite(sfr)
                    & (mass > 0.0)
                    & np.isfinite(mass)
                    & (xo > 0.0)
                    & np.isfinite(xo)
                )
                if np.any(sf):
                    ratio = xo[sf] / (16.0 * x_h)
                    logoh = 12.0 + np.log10(ratio)
                    totals["n_sf_gas"] += int(np.count_nonzero(sf))
                    totals["sf_gas_mass_code"] += float(np.sum(mass[sf]))
                    totals["sf_oxygen_mass_code"] += float(np.sum(mass[sf] * xo[sf]))
                    totals["sf_sfr_weight"] += float(np.sum(sfr[sf]))
                    totals["sf_sfr_oh_numerator"] += float(np.sum(sfr[sf] * ratio))
                    sf_log_oh.append(logoh)
                    sf_log_oh_weights.append(sfr[sf])

            if "PartType4" in f:
                mass = np.asarray(f["PartType4/Masses"], dtype=np.float64)
                totals["n_star"] += mass.size
                totals["star_mass_code"] += float(mass.sum())

            if "PartType6" in f:
                d = f["PartType6"]
                mass = np.asarray(d["Masses"], dtype=np.float64)
                cf = np.clip(
                    np.asarray(d["CarbonMassFraction"], dtype=np.float64), 0.0, 1.0
                )
                carbon = mass * cf
                silicate = mass * (1.0 - cf)
                totals["n_dust"] += mass.size
                totals["dust_mass_code"] += float(mass.sum())
                totals["dust_carbon_mass_code"] += float(carbon.sum())
                totals["dust_silicate_mass_code"] += float(silicate.sum())
                totals["dust_oxygen_mass_code"] += float(
                    silicate.sum() * SILICATE_O_MASS_FRACTION
                )

    mass_to_msun = 1.0e10 / h
    mass_keys = [k for k in totals if k.endswith("_mass_code")]
    for key in mass_keys:
        totals[key.replace("_code", "_msun")] = totals[key] * mass_to_msun

    mgas = totals["gas_mass_code"]
    mdust = totals["dust_mass_code"]
    mzgas = totals["gas_metal_mass_code"]
    mogas = totals["gas_O_mass_code"]
    totals["oh12_all_integrated"] = safe_log12(
        mogas / (16.0 * x_h * mgas) if mgas > 0.0 else float("nan")
    )
    totals["oh12_sf_mass_integrated"] = safe_log12(
        totals["sf_oxygen_mass_code"]
        / (16.0 * x_h * totals["sf_gas_mass_code"])
        if totals["sf_gas_mass_code"] > 0.0
        else float("nan")
    )
    totals["oh12_sf_sfr_weighted"] = safe_log12(
        totals["sf_sfr_oh_numerator"] / totals["sf_sfr_weight"]
        if totals["sf_sfr_weight"] > 0.0
        else float("nan")
    )
    totals["dust_to_gas"] = mdust / mgas if mgas > 0.0 else float("nan")
    totals["dust_to_gasphase_metals"] = (
        mdust / mzgas if mzgas > 0.0 else float("nan")
    )
    totals["dust_fraction_of_total_metals"] = (
        mdust / (mdust + mzgas) if mdust + mzgas > 0.0 else float("nan")
    )
    totals["oxygen_dust_fraction"] = (
        totals["dust_oxygen_mass_code"]
        / (totals["dust_oxygen_mass_code"] + mogas)
        if totals["dust_oxygen_mass_code"] + mogas > 0.0
        else float("nan")
    )

    if sf_log_oh:
        values = np.concatenate(sf_log_oh)
        totals["oh12_sf_p16"] = float(np.percentile(values, 16.0))
        totals["oh12_sf_median"] = float(np.percentile(values, 50.0))
        totals["oh12_sf_p84"] = float(np.percentile(values, 84.0))
    else:
        totals["oh12_sf_p16"] = float("nan")
        totals["oh12_sf_median"] = float("nan")
        totals["oh12_sf_p84"] = float("nan")
    return totals


def fmt(value):
    if isinstance(value, (int, np.integer)):
        return str(value)
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6e}"


def print_comparison(old, new):
    print("=" * 105)
    print(
        f"WHOLE-ZOOM CHEMISTRY COMPARISON  snap={old['snap']:03d}  "
        f"z={old['redshift']:.8f}"
    )
    print("=" * 105)
    metrics = [
        ("Gas mass [Msun]", "gas_mass_msun", "ratio"),
        ("Stellar mass [Msun]", "star_mass_msun", "ratio"),
        ("Dust mass [Msun]", "dust_mass_msun", "ratio"),
        ("SFR [Msun/yr]", "sfr_msun_yr", "ratio"),
        ("Gas metal mass [Msun]", "gas_metal_mass_msun", "ratio"),
        ("Gas oxygen mass [Msun]", "gas_O_mass_msun", "ratio"),
        ("Dust oxygen mass [Msun]", "dust_oxygen_mass_msun", "ratio"),
        ("All-gas 12+log(O/H)", "oh12_all_integrated", "dex"),
        ("SF-gas mass-weighted O/H", "oh12_sf_mass_integrated", "dex"),
        ("SF-gas SFR-weighted O/H", "oh12_sf_sfr_weighted", "dex"),
        ("SF-gas median O/H", "oh12_sf_median", "dex"),
        ("Dust/gas", "dust_to_gas", "ratio"),
        ("Dust/gas-phase metals", "dust_to_gasphase_metals", "ratio"),
        ("Dust fraction total metals", "dust_fraction_of_total_metals", "difference"),
        ("Oxygen fraction in dust", "oxygen_dust_fraction", "difference"),
    ]
    print(f"{'Metric':32s} {'legacy':>18s} {'delayed only':>18s} {'change new-old':>22s}")
    print("-" * 105)
    for label, key, mode in metrics:
        a = float(old[key])
        b = float(new[key])
        if mode == "dex":
            change = f"{b-a:+.4f} dex"
        elif mode == "ratio":
            change = f"{b/a:.6f}x ({(b/a-1.0)*100:+.2f}%)" if a != 0.0 else "nan"
        else:
            change = f"{b-a:+.6e}"
        print(f"{label:32s} {fmt(a):>18s} {fmt(b):>18s} {change:>22s}")

    print("\nGas-phase element masses: delayed-only / legacy")
    for symbol in ELEMENT_FIELDS:
        key = f"gas_{symbol}_mass_msun"
        a, b = float(old[key]), float(new[key])
        ratio = b / a if a else float("nan")
        print(
            f"  {symbol:>2s}: legacy={a:.6e}  delayed={b:.6e}  "
            f"ratio={ratio:.6f}  change={(ratio-1.0)*100:+.2f}%"
        )

    print("\nIntegrity checks")
    for label, key in [
        ("tracked elements > Z", "tracked_gt_z_count"),
        ("negative element values", "negative_element_count"),
        ("nonfinite element values", "nonfinite_element_count"),
    ]:
        print(f"  {label:28s}: legacy={old[key]} delayed={new[key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", default="halo441_output_1024_pre_oxygen")
    parser.add_argument("--delayed", default="halo441_output_1024")
    parser.add_argument("--snap", type=int, default=None,
                        help="Snapshot to print in detail; default latest complete common snapshot")
    parser.add_argument("--h", type=float, default=0.6732)
    parser.add_argument("--x-h", type=float, default=0.76,
                        help="Hydrogen mass fraction used for provisional O/H conversion")
    parser.add_argument("--csv", default="halo441_oxygen_enrichment_comparison.csv")
    args = parser.parse_args()

    old_available = complete_snapshots(args.legacy)
    new_available = complete_snapshots(args.delayed)
    common = sorted(set(old_available) & set(new_available))
    if not common:
        raise RuntimeError("No complete common snapshots")
    snap = common[-1] if args.snap is None else args.snap
    if snap not in common:
        raise RuntimeError(f"Snapshot {snap:03d} is not complete in both runs")

    all_rows = []
    detailed = None
    for current in common:
        old = audit_snapshot(args.legacy, current, args.h, args.x_h)
        new = audit_snapshot(args.delayed, current, args.h, args.x_h)
        if abs(old["redshift"] - new["redshift"]) > 1.0e-10:
            raise RuntimeError(f"Redshift mismatch at snapshot {current:03d}")
        row = {"snap": current, "redshift": old["redshift"]}
        for key in old:
            if key in ("snap", "redshift"):
                continue
            row[f"legacy_{key}"] = old[key]
            row[f"delayed_{key}"] = new[key]
            if isinstance(old[key], (int, float, np.integer, np.floating)):
                row[f"delta_{key}"] = new[key] - old[key]
        all_rows.append(row)
        if current == snap:
            detailed = (old, new)

    fieldnames = list(all_rows[0].keys())
    with open(args.csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print_comparison(*detailed)
    print("\nO/H evolution (delayed-only minus legacy)")
    print(f"{'snap':>4s} {'z':>9s} {'all gas':>11s} {'SF mass':>11s} {'SF SFR':>11s}")
    for row in all_rows:
        print(
            f"{row['snap']:4d} {row['redshift']:9.4f} "
            f"{row['delta_oh12_all_integrated']:+11.4f} "
            f"{row['delta_oh12_sf_mass_integrated']:+11.4f} "
            f"{row['delta_oh12_sf_sfr_weighted']:+11.4f}"
        )
    print(f"\nWrote {args.csv}")
    print("NOTE: whole-volume results diagnose chemistry; observational comparisons need the target-galaxy aperture.")


if __name__ == "__main__":
    main()
