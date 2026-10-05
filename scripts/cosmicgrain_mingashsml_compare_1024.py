#!/usr/bin/env python3
"""Compare the CG10 1024^3 MinGasHsml=1.0, 0.25, and 0.0 runs.

Run from the gadget4 directory. Requires numpy, h5py and matplotlib.
The group index is explicit: check the printed M200c/position before interpreting
galaxy quantities. The default 30 pkpc aperture is spherical.
"""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


VALUES = ("1.0", "0.25", "0.0")
ROOTS = {
    "1.0": Path("halo441_output_1024"),
    "0.25": Path("halo441_output_1024_mingashsml0.25"),
    "0.0": Path("halo441_output_1024_mingashsml0.0"),
}
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
PROTON_G = 1.67262192369e-24


def read_attr(group, key, default=None):
    return float(group.attrs[key]) if key in group.attrs else default


def catalog_center(root, snap, index):
    path = root / f"groups_{snap:03d}" / f"fof_subhalo_tab_{snap:03d}.0.hdf5"
    with h5py.File(path) as f:
        group = f["Group"]
        if index >= len(group["GroupPos"]):
            raise ValueError(f"{path}: group index {index} is out of range")
        return (np.asarray(group["GroupPos"][index], dtype=float),
                float(group["Group_M_Crit200"][index]),
                float(group["Group_R_Crit200"][index]))


def snapshot(root, snap):
    paths = sorted((root / f"snapdir_{snap:03d}").glob(f"snapshot_{snap:03d}.*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No snapshot pieces in {root}/snapdir_{snap:03d}")
    fields = {
        "PartType0": ("Coordinates", "Masses", "Density", "SmoothingLength", "StarFormationRate"),
        "PartType4": ("Coordinates", "Masses"),
        "PartType6": ("Coordinates", "Masses"),
    }
    chunks = {part: {field: [] for field in names} for part, names in fields.items()}
    with h5py.File(paths[0]) as f:
        a = read_attr(f["Header"], "Time")
        hubble = read_attr(f["Parameters"], "HubbleParam")
        box = read_attr(f["Header"], "BoxSize")
        units = (read_attr(f["Parameters"], "UnitLength_in_cm", KPC_CM),
                 read_attr(f["Parameters"], "UnitMass_in_g", 1e10 * MSUN_G))
        if a is None or hubble is None or box is None:
            raise ValueError(f"Missing Time, HubbleParam, or BoxSize in {paths[0]}")
    for path in paths:
        with h5py.File(path) as f:
            for part, names in fields.items():
                if part not in f:
                    continue
                for name in names:
                    chunks[part][name].append(f[part][name][:])
    data = {part: {name: np.concatenate(arr) if arr else np.empty(0)
                   for name, arr in names.items()} for part, names in chunks.items()}
    return data, a, hubble, box, units, len(paths)


def mask_at_radius(pos, center, box, a, hubble, aperture):
    delta = pos - center
    delta -= box * np.rint(delta / box)
    radius_pkpc = np.linalg.norm(delta, axis=1) * a / hubble
    return radius_pkpc < aperture


def percentile(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=int, default=29)
    p.add_argument("--group-index", type=int, default=1,
                   help="FoF group index for halo 441; verify in output")
    p.add_argument("--aperture", type=float, default=30, help="spherical physical kpc")
    p.add_argument("--out-prefix", default="cg10_1024_mingashsml_1.0_0.25_0.0_snap029")
    args = p.parse_args()
    rows, gas_profiles = [], []
    for label in VALUES:
        root = ROOTS[label]
        center, m200_code, r200_code = catalog_center(root, args.snapshot, args.group_index)
        data, a, hubble, box, (unit_l, unit_m), pieces = snapshot(root, args.snapshot)
        if pieces != 4:
            print(f"WARNING: {label} has {pieces} snapshot pieces (expected 4)")
        floor = float(label)
        gas = data["PartType0"]
        sel = mask_at_radius(gas["Coordinates"], center, box, a, hubble, args.aperture)
        hcode = gas["SmoothingLength"][sel]
        rho = gas["Density"][sel]
        # Snapshot masses are commonly float32.  Cast before multiplying by
        # the cgs unit (~1e43 g), otherwise the intermediate overflows.
        mass = np.asarray(gas["Masses"][sel], dtype=np.float64)
        mass *= unit_m / hubble / MSUN_G
        sfr = gas["StarFormationRate"][sel]
        # Gadget comoving density unit: mass / length^3, multiplied by h^2/a^3.
        nh = rho * (unit_m / unit_l**3) * hubble**2 / a**3 * 0.76 / PROTON_G
        at_floor = np.isclose(hcode, floor, rtol=1e-4, atol=1e-6) if floor > 0 else np.zeros(len(hcode), bool)
        sf = sfr > 0
        row = {
            "MinGasHsml_code": label, "a": a, "z": 1 / a - 1,
            "group_index": args.group_index, "center_code_xyz": np.array2string(center, precision=4),
            "M200c_Msun": m200_code * unit_m / hubble / MSUN_G,
            "R200c_pkpc": r200_code * a / hubble,
            "aperture_pkpc": args.aperture, "N_gas": len(hcode),
            "Mgas_Msun": mass.sum(), "SFR_Msun_per_yr": sfr.sum(),
            "N_starforming_gas": int(sf.sum()),
            "gas_at_floor_number_frac": at_floor.mean() if len(at_floor) else np.nan,
            "gas_at_floor_mass_frac": mass[at_floor].sum() / mass.sum() if mass.sum() else np.nan,
            "SF_gas_at_floor_number_frac": at_floor[sf].mean() if sf.sum() else np.nan,
            "hsml_code_p01": percentile(hcode, 1),
            "hsml_code_p50": percentile(hcode, 50),
            "nH_cm3_p50": percentile(nh, 50),
            "nH_cm3_p95": percentile(nh, 95),
            "nH_cm3_p99": percentile(nh, 99),
            "nH_cm3_max": float(nh.max()) if len(nh) else np.nan,
            "Mgas_nH_gt_0p3_Msun": mass[nh > 0.3].sum(),
            "Mgas_nH_gt_1_Msun": mass[nh > 1].sum(),
        }
        for part, key in (("PartType4", "Mstar_Msun"), ("PartType6", "Mdust_Msun")):
            obj = data[part]
            subset = mask_at_radius(obj["Coordinates"], center, box, a, hubble, args.aperture)
            mass_code = np.asarray(obj["Masses"][subset], dtype=np.float64).sum()
            row[key] = mass_code * unit_m / hubble / MSUN_G
        row["D_over_G"] = row["Mdust_Msun"] / row["Mgas_Msun"] if row["Mgas_Msun"] else np.nan
        rows.append(row)
        gas_profiles.append((label, nh, hcode * a / hubble))
        print(f"{label}: M200c={row['M200c_Msun']:.3e} Msun, R200c={row['R200c_pkpc']:.2f} pkpc, "
              f"N_gas={row['N_gas']}, floor={row['gas_at_floor_number_frac']:.2%}, "
              f"SF floor={row['SF_gas_at_floor_number_frac']:.2%}, "
              f"nH_p99={row['nH_cm3_p99']:.3g}, SFR={row['SFR_Msun_per_yr']:.3g}, "
              f"Mstar={row['Mstar_Msun']:.3e}, Mdust={row['Mdust_Msun']:.3e}")

    csv_path = Path(args.out_prefix + ".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for label, nh, hp in gas_profiles:
        ax[0].hist(nh[nh > 0], bins=np.logspace(-6, 3, 100), density=True,
                   histtype="step", linewidth=1.8, label=label)
        ax[1].hist(hp[hp > 0], bins=np.logspace(-3, 3, 100), density=True,
                   histtype="step", linewidth=1.8, label=label)
    for axis in ax:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("Probability density")
        axis.legend(title="MinGasHsml")
    ax[0].set_xlabel(r"$n_{\rm H}$ (cm$^{-3}$)")
    ax[1].set_xlabel("Stored gas smoothing length (physical kpc)")
    fig.suptitle(f"CG10 1024³, snapshot {args.snapshot}; spherical r < {args.aperture:g} pkpc")
    png_path = Path(args.out_prefix + ".png")
    fig.savefig(png_path, dpi=180)
    print(f"Wrote {csv_path} and {png_path}")
    print("Check M200c and group center across rows: group index can change between runs.")


if __name__ == "__main__":
    main()
