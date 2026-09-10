#!/usr/bin/env python3
"""Diagnose why a CosmicGrain zoom halo underproduces stars.

The final halo is anchored by an explicit FOF group index. Earlier progenitors
are followed with halo_utils.get_zoom_halo_series(), which uses conserved HRDM
particle IDs rather than position or catalog rank.

Example
-------
python3 cosmicgrain_star_formation_diagnostic.py \
    ../halo441_output_1024 --group-index 1 \
    --sf-threshold 0.3 --min-gas-hsml-ckpch 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from halo_utils import (
    find_last_snap_num,
    find_snapshots,
    get_zoom_halo_series,
    glob_snap_chunks,
    read_snap_header,
)

MSUN_G = 1.98847e33
M_PROTON_G = 1.67262192369e-24
K_BOLTZ = 1.380649e-16
X_H = 0.76
GAMMA_MINUS_ONE = 2.0 / 3.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", help="GADGET-4 zoom output directory")
    p.add_argument("--group-index", type=int, required=True,
                   help="Target FOF group index at the final snapshot")
    p.add_argument("--redshifts", nargs="+", type=float,
                   default=[3.0, 2.0, 1.0, 0.5, 0.0])
    p.add_argument("--sf-threshold", type=float, default=0.3,
                   help="Physical SF threshold nH in cm^-3")
    p.add_argument("--min-gas-hsml-ckpch", type=float, default=1.0,
                   help="Configured MinGasHsml in ckpc/h (default: 1.0)")
    p.add_argument("--apertures", nargs="+", type=float,
                   default=[10, 20, 30, 50, 100],
                   help="Fixed physical aperture radii in pkpc")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--prefix", default=None)
    return p.parse_args()


def snapshot_metadata(output_dir):
    rows = []
    for snap, snapdir, _ in find_snapshots(output_dir):
        hdr = read_snap_header(snapdir)
        rows.append((snap, hdr["z"], snapdir, hdr))
    return rows


def select_snapshots(rows, targets):
    selected = {}
    for target in targets:
        best = min(rows, key=lambda row: abs(row[1] - target))
        selected[best[0]] = (target, best[1])
    return dict(sorted(selected.items()))


def periodic_radius(coords, center, box):
    delta = np.asarray(coords, float) - np.asarray(center, float)[None, :]
    delta -= box * np.round(delta / box)
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def units_from_file(path):
    with h5py.File(path, "r") as f:
        params = f["Parameters"].attrs
        header = f["Header"].attrs
        h = float(params.get("HubbleParam", 1.0))
        return {
            "h": h,
            "a": float(header["Time"]),
            "z": float(header["Redshift"]),
            "box": float(header["BoxSize"]),
            "umass_g": float(params.get("UnitMass_in_g", 1.989e43)),
            "ulength_cm": float(params.get("UnitLength_in_cm", 3.085678e21)),
            "uvel_cms": float(params.get("UnitVelocity_in_cm_per_s", 1.0e5)),
        }


def code_mass_to_msun(mass, units):
    return np.asarray(mass, float) * units["umass_g"] / units["h"] / MSUN_G


def density_to_nh(rho, units):
    rho_cgs = (np.asarray(rho, float) * units["umass_g"] /
               units["ulength_cm"]**3 * units["h"]**2 / units["a"]**3)
    return X_H * rho_cgs / M_PROTON_G


def internal_energy_to_temperature(u, electron_abundance, units):
    u = np.asarray(u, float)
    if electron_abundance is None:
        mu = np.full_like(u, 0.59)
    else:
        xe = np.clip(np.asarray(electron_abundance, float), 0.0, 2.0)
        mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    return (GAMMA_MINUS_ONE * u * units["uvel_cms"]**2 *
            mu * M_PROTON_G / K_BOLTZ)


def read_type_within(snapshot, ptype, center, rmax_ckpch, fields):
    chunks = glob_snap_chunks(snapshot)
    units = units_from_file(chunks[0])
    accum = {field: [] for field in fields}
    accum["r_ckpch"] = []
    key = f"PartType{ptype}"
    missing = set()
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if key not in f or "Coordinates" not in f[key]:
                continue
            group = f[key]
            r = periodic_radius(group["Coordinates"][:], center, units["box"])
            keep = r <= rmax_ckpch
            if not np.any(keep):
                continue
            accum["r_ckpch"].append(r[keep])
            for field in fields:
                if field in group:
                    accum[field].append(group[field][:][keep])
                else:
                    missing.add(field)
    result = {}
    for field, pieces in accum.items():
        if pieces:
            result[field] = np.concatenate(pieces, axis=0)
    result["missing_fields"] = sorted(missing)
    result["units"] = units
    return result


def mass_sum(mass, mask):
    return float(np.sum(mass[mask])) if np.any(mask) else 0.0


def analyze_snapshot(output_dir, snap, halo, apertures, sf_threshold,
                     min_gas_hsml_ckpch):
    snapdir = Path(output_dir) / f"snapdir_{snap:03d}"
    center = halo["center"]
    r200_ckpch = float(halo["r200_ckpch"])
    gas = read_type_within(
        snapdir, 0, center, r200_ckpch,
        ["Masses", "Density", "InternalEnergy", "ElectronAbundance",
         "SmoothingLength", "StarFormationRate", "Metallicity"],
    )
    stars = read_type_within(
        snapdir, 4, center, r200_ckpch,
        ["Masses", "StellarBirthMass", "GFM_InitialMass", "FormationTime"],
    )
    u = gas["units"]
    to_pkpc = u["a"] / u["h"]
    r200_pkpc = r200_ckpch * to_pkpc

    gm = code_mass_to_msun(gas.get("Masses", np.array([])), u)
    gr = gas.get("r_ckpch", np.array([])) * to_pkpc
    nH = density_to_nh(gas.get("Density", np.full_like(gm, np.nan)), u)
    temp = internal_energy_to_temperature(
        gas.get("InternalEnergy", np.full_like(gm, np.nan)),
        gas.get("ElectronAbundance"), u,
    )
    sfr = np.asarray(gas.get("StarFormationRate", np.zeros_like(gm)), float)
    hsml_ckpch = np.asarray(gas.get("SmoothingLength", np.full_like(gm, np.nan)), float)
    hsml_pkpc = hsml_ckpch * to_pkpc

    sm = code_mass_to_msun(stars.get("Masses", np.array([])), u)
    sr = stars.get("r_ckpch", np.array([])) * to_pkpc
    birth_field = None
    for candidate in ("StellarBirthMass", "GFM_InitialMass"):
        if candidate in stars:
            birth_field = candidate
            break
    birth = (code_mass_to_msun(stars[birth_field], u)
             if birth_field else np.full_like(sm, np.nan))

    row = {
        "snapshot": snap, "z": u["z"], "a": u["a"],
        "r200_pkpc": r200_pkpc,
        "m200_msun": float(halo["m200_msun"]),
        "gas_R200_msun": float(np.sum(gm)),
        "star_R200_msun": float(np.sum(sm)),
        "birthmass_R200_msun": float(np.nansum(birth)) if birth_field else np.nan,
        "birthmass_field": birth_field or "missing",
        "sfr_R200_msunyr": float(np.sum(sfr)),
        "gas_missing": ";".join(gas["missing_fields"]),
        "star_missing": ";".join(stars["missing_fields"]),
    }
    for aperture in apertures:
        tag = f"{aperture:g}pkpc"
        gsel = gr < aperture
        ssel = sr < aperture
        row[f"gas_{tag}_msun"] = mass_sum(gm, gsel)
        row[f"star_{tag}_msun"] = mass_sum(sm, ssel)
        row[f"sfr_{tag}_msunyr"] = float(np.sum(sfr[gsel]))

    central = gr < 30.0
    central_mass = mass_sum(gm, central)
    density_bins = {
        "nh_lt_0p01": nH < 0.01,
        "nh_0p01_0p1": (nH >= 0.01) & (nH < 0.1),
        "nh_0p1_sf": (nH >= 0.1) & (nH < sf_threshold),
        "nh_ge_sf": nH >= sf_threshold,
        "nh_ge_1": nH >= 1.0,
        "nh_ge_10": nH >= 10.0,
        "nh_ge_100": nH >= 100.0,
    }
    temp_bins = {
        "T_lt_1e4": temp < 1e4,
        "T_1e4_1e5": (temp >= 1e4) & (temp < 1e5),
        "T_1e5_1e6": (temp >= 1e5) & (temp < 1e6),
        "T_ge_1e6": temp >= 1e6,
    }
    for name, mask in density_bins.items():
        row[f"gas30_{name}_msun"] = mass_sum(gm, central & mask)
        row[f"gas30_{name}_frac"] = (row[f"gas30_{name}_msun"] / central_mass
                                      if central_mass > 0 else np.nan)
    # InternalEnergy for actively star-forming SH03 particles represents an
    # effective multiphase equation of state, not a single physical gas
    # temperature. Keep those particles out of the thermal phase bins.
    sf_eos = central & (sfr > 0)
    non_sf = central & ~sf_eos
    non_sf_mass = mass_sum(gm, non_sf)
    row["gas30_sf_effective_eos_msun"] = mass_sum(gm, sf_eos)
    row["gas30_sf_effective_eos_frac"] = (
        row["gas30_sf_effective_eos_msun"] / central_mass
        if central_mass > 0 else np.nan
    )
    for name, mask in temp_bins.items():
        row[f"gas30_nonSF_{name}_msun"] = mass_sum(gm, non_sf & mask)
        row[f"gas30_nonSF_{name}_frac"] = (
            row[f"gas30_nonSF_{name}_msun"] / central_mass
            if central_mass > 0 else np.nan
        )
        row[f"gas30_nonSF_{name}_frac_of_nonSF"] = (
            row[f"gas30_nonSF_{name}_msun"] / non_sf_mass
            if non_sf_mass > 0 else np.nan
        )
        # Backward-compatible alias, now explicitly excluding SF effective-EOS gas.
        row[f"gas30_{name}_msun"] = row[f"gas30_nonSF_{name}_msun"]
        row[f"gas30_{name}_frac"] = (row[f"gas30_{name}_msun"] / central_mass
                                      if central_mass > 0 else np.nan)

    eligible = central & (nH >= sf_threshold)
    eligible_mass = mass_sum(gm, eligible)
    sfr30 = float(np.sum(sfr[central]))
    row["depletion_time_sf_Gyr"] = (eligible_mass / sfr30 / 1e9
                                     if eligible_mass > 0 and sfr30 > 0 else np.nan)
    row["max_nH_cm3"] = float(np.nanmax(nH)) if len(nH) else np.nan
    row["median_hsml30_pkpc"] = (float(np.nanmedian(hsml_pkpc[central]))
                                  if np.any(central) else np.nan)
    valid_h = central & np.isfinite(hsml_ckpch)
    row["min_hsml_fraction30"] = (
        float(np.mean(hsml_ckpch[valid_h] <= 1.05 * min_gas_hsml_ckpch))
        if np.any(valid_h) else np.nan
    )
    arrays = {"nH30": nH[central], "temp30": temp[central],
              "gm30": gm[central], "hsml30_pkpc": hsml_pkpc[central]}
    return row, arrays


def recommendation(row):
    near = row.get("gas30_nh_0p1_sf_frac", np.nan)
    eligible = row.get("gas30_nh_ge_sf_frac", np.nan)
    hot_non_sf = row.get("gas30_nonSF_T_ge_1e6_frac_of_nonSF", np.nan)
    pinned = row.get("min_hsml_fraction30", np.nan)
    tdep = row.get("depletion_time_sf_Gyr", np.nan)
    if np.isfinite(pinned) and pinned > 0.30 and (not np.isfinite(eligible) or eligible < 0.05):
        return ("RESOLUTION FIRST", "More than 30% of central gas is at the configured hydro floor while little gas is SF-eligible. Test smaller hydro/force floors before lowering the SF threshold.")
    if np.isfinite(near) and near > 0.20 and (not np.isfinite(eligible) or eligible < 0.10):
        return ("CRITPHYSDENSITY FIRST", "A substantial central gas reservoir sits at 0.1<nH<SF threshold. Test CritPhysDensity 0.3->0.1 with all feedback parameters fixed.")
    if np.isfinite(eligible) and eligible >= 0.05 and np.isfinite(tdep) and tdep > 5.0:
        return ("SF CONVERSION FIRST", "Eligible dense gas exists but its depletion time is long. Inspect the spawning law and then test MaxSfrTimescale or its implementation.")
    if np.isfinite(hot_non_sf) and hot_non_sf > 0.50:
        return ("HOT NON-SF GAS: AUDIT FEEDBACK/COOLING",
                "Most non-star-forming central gas is hotter than 1e6 K. This is a diagnostic flag, not by itself evidence that feedback is excessive; compare inflow, outflow and cooling histories before selecting a parameter.")
    return ("HISTORY/FEEDBACK FIRST", "No single z=0 bottleneck dominates. Use the multi-epoch panels and feedback logs to determine whether burst heating or insufficient inflow suppresses assembly.")


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path, rows, latest_arrays, apertures, sf_threshold,
                min_gas_hsml_ckpch, run_name):
    rows = sorted(rows, key=lambda x: x["z"], reverse=True)
    z = np.array([r["z"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    for aperture in apertures:
        key = f"gas_{aperture:g}pkpc_msun"
        ax.plot(z, [r[key] for r in rows], marker="o", label=f"r<{aperture:g} pkpc")
    ax.plot(z, [r["gas_R200_msun"] for r in rows], marker="s", lw=2.5, label="r<R200")
    ax.set_yscale("log"); ax.set_xlabel("Redshift"); ax.set_ylabel("Gas mass [Msun]")
    ax.set_title("Where the gas resides"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[0, 1]
    ax.plot(z, [r["gas30_sf_effective_eos_frac"] for r in rows],
            marker="o", label="SF effective EOS")
    for name, label in [("T_lt_1e4", "non-SF cold"), ("T_1e4_1e5", "non-SF warm"),
                        ("T_1e5_1e6", "warm-hot"), ("T_ge_1e6", "hot")]:
        ax.plot(z, [r[f"gas30_nonSF_{name}_frac"] for r in rows], marker="o", label=label)
    ax.set_ylim(0, 1); ax.set_xlabel("Redshift"); ax.set_ylabel("Gas-mass fraction")
    ax.set_title("Gas phases inside 30 pkpc\n(SF gas shown separately)"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[0, 2]
    for name, label in [("nh_lt_0p01", "nH<0.01"), ("nh_0p01_0p1", "0.01-0.1"),
                        ("nh_0p1_sf", f"0.1-{sf_threshold:g}"),
                        ("nh_ge_sf", f">={sf_threshold:g}")]:
        ax.plot(z, [r[f"gas30_{name}_frac"] for r in rows], marker="o", label=label)
    ax.set_ylim(0, 1); ax.set_xlabel("Redshift"); ax.set_ylabel("Gas-mass fraction")
    ax.set_title("Density budget inside 30 pkpc"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1, 0]
    ax.plot(z, [r["star_R200_msun"] for r in rows], marker="o", label="surviving Mstar")
    birth = np.array([r["birthmass_R200_msun"] for r in rows])
    if np.any(np.isfinite(birth)):
        ax.plot(z, birth, marker="s", label="summed birth mass")
    ax.set_yscale("log"); ax.set_xlabel("Redshift"); ax.set_ylabel("Stellar mass [Msun]")
    ax2 = ax.twinx(); ax2.plot(z, [r["sfr_30pkpc_msunyr"] for r in rows], color="tab:red", marker="^", label="SFR")
    ax2.set_ylabel("SFR [Msun/yr]", color="tab:red")
    ax.set_title("Stellar assembly and current SFR"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)

    nH = latest_arrays["nH30"]; mass = latest_arrays["gm30"]
    ax = axes[1, 1]
    good = np.isfinite(nH) & (nH > 0) & np.isfinite(mass) & (mass > 0)
    bins = np.logspace(-8, max(1.0, math.ceil(np.log10(np.max(nH[good])))), 70) if np.any(good) else np.logspace(-8, 1, 70)
    ax.hist(nH[good], bins=bins, weights=mass[good], histtype="step", lw=2)
    ax.axvspan(0.1, sf_threshold, color="orange", alpha=.2, label="nearly eligible")
    ax.axvline(sf_threshold, color="red", ls="--", label="SF threshold")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("nH [cm^-3]"); ax.set_ylabel("Gas mass per bin [Msun]")
    ax.set_title("z=0 central density distribution"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1, 2]
    hsml = latest_arrays["hsml30_pkpc"]
    good = good & np.isfinite(hsml) & (hsml > 0)
    if np.any(good):
        ax.hexbin(nH[good], hsml[good], C=np.log10(mass[good]), reduce_C_function=np.mean,
                  gridsize=35, xscale="log", yscale="log", mincnt=1, cmap="viridis")
    ax.set_xlabel("nH [cm^-3]"); ax.set_ylabel("Smoothing length [pkpc]")
    ax.set_title(f"z=0 smoothing vs density\nMinGasHsml={min_gas_hsml_ckpch:g} ckpc/h")
    ax.grid(alpha=.25)

    fig.suptitle(f"CosmicGrain star-formation diagnostic: {run_name}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(path, rows, selected, decision):
    final = min(rows, key=lambda r: abs(r["z"]))
    with open(path, "w") as f:
        f.write("COSMICGRAIN STAR-FORMATION BOTTLENECK DIAGNOSTIC\n")
        f.write("=" * 72 + "\n")
        f.write(f"Final snapshot: {final['snapshot']:03d}, z={final['z']:.4f}\n")
        f.write(f"M200c={final['m200_msun']:.6e} Msun  R200c={final['r200_pkpc']:.3f} pkpc\n")
        f.write(f"Mstar(<R200)={final['star_R200_msun']:.6e} Msun\n")
        f.write(f"Mgas(<R200)={final['gas_R200_msun']:.6e} Msun\n")
        f.write(f"Mgas(<30 pkpc)={final.get('gas_30pkpc_msun', np.nan):.6e} Msun\n")
        f.write(f"SFR(<30 pkpc)={final.get('sfr_30pkpc_msunyr', np.nan):.6e} Msun/yr\n")
        f.write(f"max nH={final['max_nH_cm3']:.6e} cm^-3\n")
        f.write(f"near-threshold gas fraction={final['gas30_nh_0p1_sf_frac']:.6f}\n")
        f.write(f"SF-eligible gas fraction={final['gas30_nh_ge_sf_frac']:.6f}\n")
        f.write(f"SF effective-EOS gas fraction={final['gas30_sf_effective_eos_frac']:.6f}\n")
        f.write(f"hot non-SF fraction of non-SF gas={final['gas30_nonSF_T_ge_1e6_frac_of_nonSF']:.6f}\n")
        f.write(f"MinGasHsml-pinned fraction={final['min_hsml_fraction30']:.6f}\n")
        f.write(f"eligible-gas depletion time={final['depletion_time_sf_Gyr']:.6f} Gyr\n")
        f.write("\nSELECTED EPOCHS\n")
        for snap, (target, actual) in selected.items():
            f.write(f"  target z={target:g}: snap {snap:03d}, actual z={actual:.5f}\n")
        f.write("\nRECOMMENDATION\n")
        f.write(f"  {decision[0]}\n  {decision[1]}\n")
        f.write("\nThis recommendation is a diagnostic heuristic, not a fit. Change one parameter per controlled rerun.\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows_meta = snapshot_metadata(output_dir)
    if not rows_meta:
        raise SystemExit(f"No paired snapshots/catalogs in {output_dir}")
    selected = select_snapshots(rows_meta, args.redshifts)
    final_snap = find_last_snap_num(output_dir)
    requested = sorted(selected)
    print(f"Tracking group {args.group_index} from final snapshot {final_snap:03d}")
    halos = get_zoom_halo_series(
        output_dir, requested, group_index=args.group_index,
        reference_snap=final_snap, verbose=True,
    )

    results = []
    latest_arrays = None
    for snap in requested:
        halo = halos.get(snap)
        if halo is None:
            print(f"WARNING: no trusted halo at snapshot {snap:03d}; skipping")
            continue
        snapshot_header = read_snap_header(
            output_dir / f"snapdir_{snap:03d}"
        )
        print(f"Analyzing snapshot {snap:03d}, z={snapshot_header['z']:.4f}")
        row, arrays = analyze_snapshot(
            output_dir, snap, halo, args.apertures,
            args.sf_threshold, args.min_gas_hsml_ckpch,
        )
        results.append(row)
        if latest_arrays is None or row["z"] < min(r["z"] for r in results[:-1] or [row]):
            latest_arrays = arrays
    if not results:
        raise SystemExit("No trusted snapshots could be analyzed")
    final = min(results, key=lambda r: abs(r["z"]))
    # Recover arrays for the actual latest result deterministically.
    _, latest_arrays = analyze_snapshot(
        output_dir, final["snapshot"], halos[final["snapshot"]], args.apertures,
        args.sf_threshold, args.min_gas_hsml_ckpch,
    )
    decision = recommendation(final)

    product_dir = Path(args.output_dir) if args.output_dir else output_dir / "star_formation_diagnostic"
    product_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or output_dir.name
    csv_path = product_dir / f"{prefix}_sf_diagnostic.csv"
    report_path = product_dir / f"{prefix}_sf_diagnostic.txt"
    figure_path = product_dir / f"{prefix}_sf_diagnostic.png"
    write_csv(csv_path, results)
    write_report(report_path, results, selected, decision)
    make_figure(figure_path, results, latest_arrays, args.apertures,
                args.sf_threshold, args.min_gas_hsml_ckpch, output_dir.name)
    print("\n" + "=" * 72)
    print(f"RECOMMENDATION: {decision[0]}")
    print(decision[1])
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
