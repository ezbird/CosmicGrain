#!/usr/bin/env python3
"""Snapshot census and target-halo evolutionary history for CosmicGrain.

Two complementary scopes are available:

``volume``
    Seven-type particle census plus whole-volume stellar and dust masses.
    This is primarily an integrity/bookkeeping view.

``target``
    Main-progenitor history for one explicitly selected final-snapshot halo.
    The final group is anchored with ``--group-index`` and earlier snapshots
    are followed through conserved PartType1 HRDM core IDs using halo_utils.

Mass conventions
----------------
Gadget code masses are 1e10 Msun/h.  Physical masses are therefore
``mass_code * 1e10 / h``.  The former version omitted the division by h and
silently labelled Msun/h values as Msun.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict, namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import h5py
import numpy as np

from halo_utils import (
    MSUN_PER_CODE,
    get_zoom_halo_series,
    load_particles_within_radius,
    periodic_delta,
)


SINGLE_RE = re.compile(r"(.*?/)?(snapshot_(\d{3}))\.hdf5$")
MULTI_RE = re.compile(
    r"(.*?/)?(snapdir_(\d{3}))/snapshot_\3\.\d+\.hdf5$"
)

PARTICLE_LABELS = {
    0: "P0(gas)",
    1: "P1(HRDM)",
    2: "P2(LRDM)",
    3: "P3",
    4: "P4(stars)",
    5: "P5",
    6: "P6(dust)",
}


def is_backup_or_temp(filename: str) -> bool:
    name = os.path.basename(filename).lower()
    return (
        name.startswith("bak-")
        or name.endswith(".bak.hdf5")
        or "bak_snapshot" in name
        or name.startswith("tmp-")
        or name.endswith(".tmp.hdf5")
        or ".partial." in name
        or name.endswith(".part")
        or ".old" in name
        or ("backup" in name and not name.startswith("snapshot_"))
    )


def discover_groups(root: str | Path):
    """Discover single- and multi-file snapshots, preferring newest duplicates."""
    Series = namedtuple("Series", ["key", "files"])
    by_key = defaultdict(list)
    index_of_key = {}

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".hdf5") or is_backup_or_temp(filename):
                continue
            full = os.path.join(dirpath, filename)
            multi = MULTI_RE.search(full)
            single = SINGLE_RE.search(full)
            if multi:
                base = os.path.join(multi.group(1) or "", multi.group(2))
                index = int(multi.group(3))
            elif single:
                base = os.path.join(single.group(1) or "", single.group(2))
                index = int(single.group(3))
            else:
                continue
            by_key[base].append(full)
            index_of_key[base] = index

    by_index = defaultdict(list)
    for key, files in by_key.items():
        by_index[index_of_key[key]].append(
            Series(key=key, files=sorted(files, key=_piece_key))
        )
    return by_index


def _piece_key(filename: str):
    try:
        return int(os.path.basename(filename).split(".")[-2])
    except Exception:
        return filename


def pick_newest_series(by_index):
    selected = []
    for index, candidates in by_index.items():
        best = max(
            candidates,
            key=lambda series: max(
                os.path.getmtime(filename) for filename in series.files
            ),
        )
        selected.append((index, best))
    return sorted(selected)


def _scalar_attr(attrs, name: str, default=None):
    if name not in attrs:
        if default is not None:
            return default
        raise KeyError(name)
    return float(np.asarray(attrs[name]).squeeze())


def read_header_and_counts(files: Sequence[str]):
    counts = {ptype: 0 for ptype in range(7)}
    reference = None
    newest_mtime = max(os.path.getmtime(filename) for filename in files)

    for filename in files:
        with h5py.File(filename, "r") as handle:
            header = handle["Header"].attrs
            params = handle["Parameters"].attrs if "Parameters" in handle else {}
            a = _scalar_attr(header, "Time", 1.0)
            z = _scalar_attr(header, "Redshift", 1.0 / a - 1.0)

            def cosmology_value(name: str, default: float) -> float:
                if name in params:
                    return _scalar_attr(params, name)
                return _scalar_attr(header, name, default)

            current = {
                "a": a,
                "z": z,
                "omega_m": cosmology_value("Omega0", 0.3),
                "omega_lambda": cosmology_value("OmegaLambda", 0.7),
                "h": cosmology_value("HubbleParam", 0.7),
                "box": _scalar_attr(header, "BoxSize", np.nan),
            }
            if reference is None:
                reference = current
            else:
                for key in current:
                    if not np.isclose(
                        current[key], reference[key], rtol=1.0e-10, atol=1.0e-12
                    ):
                        raise RuntimeError(
                            f"Inconsistent {key} among chunks of snapshot: "
                            f"{filename}"
                        )

            num_this = header.get("NumPart_ThisFile")
            if num_this is not None:
                for ptype in range(min(len(num_this), 7)):
                    counts[ptype] += int(num_this[ptype])
            else:
                for ptype in range(7):
                    group = f"PartType{ptype}"
                    if group in handle and "Coordinates" in handle[group]:
                        counts[ptype] += len(handle[group]["Coordinates"])

    assert reference is not None
    return reference, counts, newest_mtime


def _sum_dataset(dataset, block_size: int = 1_000_000) -> float:
    total = 0.0
    for start in range(0, len(dataset), block_size):
        total += float(
            np.sum(
                dataset[start:start + block_size],
                dtype=np.float64,
            )
        )
    return total


def compute_type_mass(files: Sequence[str], ptype: int) -> float:
    """Return whole-volume particle mass in Gadget code units."""
    total = 0.0
    for filename in files:
        with h5py.File(filename, "r") as handle:
            name = f"PartType{ptype}"
            if name not in handle:
                continue
            group = handle[name]
            if "Masses" in group:
                total += _sum_dataset(group["Masses"])
                continue
            mass_table = np.asarray(
                handle["Header"].attrs.get("MassTable", np.zeros(7)),
                dtype=float,
            )
            if ptype < len(mass_table) and mass_table[ptype] > 0:
                if "Coordinates" in group:
                    count = len(group["Coordinates"])
                else:
                    count = len(next(iter(group.values())))
                total += count * mass_table[ptype]
    return total


def age_of_universe_gyr(
    z: float,
    omega_m: float,
    omega_lambda: float,
    h: float,
) -> float:
    """Age for the matter+Lambda(+curvature) cosmology stored in the snapshot."""
    hubble_s = (100.0 * h) / 3.085677581e19
    a_now = 1.0 / (1.0 + max(z, 0.0))
    omega_k = 1.0 - omega_m - omega_lambda
    nstep = 4000
    log_a0, log_a1 = math.log(1.0e-8), math.log(a_now)
    total = 0.0
    for index in range(nstep):
        a = math.exp(log_a0 + (index + 0.5) * (log_a1-log_a0) / nstep)
        e2 = omega_m/a**3 + omega_k/a**2 + omega_lambda
        total += 1.0 / math.sqrt(e2)
    seconds = total * (log_a1-log_a0) / nstep / hubble_s
    return seconds / (3600.0 * 24.0 * 365.25 * 1.0e9)


def convert_mass(value: float, h: float, unit: str) -> float:
    if unit == "code":
        return value
    if unit == "msun-h":
        return value * MSUN_PER_CODE
    if unit == "msun":
        return value * MSUN_PER_CODE / h
    raise ValueError(unit)


def mass_unit_label(unit: str) -> str:
    return {
        "code": "code",
        "msun-h": "Msun/h",
        "msun": "Msun",
    }[unit]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        return "  ".join(
            cell.ljust(widths[index]) if index == 0
            else cell.rjust(widths[index])
            for index, cell in enumerate(row)
        )

    line = "-" * (sum(widths) + 2 * (len(widths)-1))
    return "\n".join(
        [format_row(headers), line]
        + [format_row(row) for row in rows]
    )


def timestamp_string(mtime: float, timezone_choice: str) -> str:
    timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
    if timezone_choice == "local":
        timestamp = timestamp.astimezone()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def build_volume_rows(series_list, mass_unit: str, timezone_choice: str):
    display_rows = []
    csv_rows = []
    unit_label = mass_unit_label(mass_unit)

    for snap, series in series_list:
        header, counts, mtime = read_header_and_counts(series.files)
        mstar = convert_mass(compute_type_mass(series.files, 4), header["h"], mass_unit)
        mdust = convert_mass(compute_type_mass(series.files, 6), header["h"], mass_unit)
        age = age_of_universe_gyr(
            header["z"],
            header["omega_m"],
            header["omega_lambda"],
            header["h"],
        )
        label = os.path.basename(series.key).replace("snapdir_", "snapshot_")
        display = [label, f"{header['z']:.3f}", f"{age:.3f}"]
        display.extend(f"{counts[ptype]:,}" for ptype in range(7))
        display.extend([
            f"{mstar:.3e}",
            f"{mdust:.3e}",
            timestamp_string(mtime, timezone_choice),
        ])
        display_rows.append(display)
        csv_rows.append({
            "snapshot": snap,
            "z": header["z"],
            "age_gyr": age,
            **{f"n_parttype{ptype}": counts[ptype] for ptype in range(7)},
            f"mstar_{unit_label}": mstar,
            f"mdust_{unit_label}": mdust,
            "last_modified": timestamp_string(mtime, timezone_choice),
        })
    return display_rows, csv_rows


def _halo_value(halo: dict, key: str, default=np.nan):
    return halo.get(key, default) if halo is not None else default


def _aperture_mask(
    particles: dict,
    center: np.ndarray,
    box: float,
    radius: float,
) -> np.ndarray:
    if not particles or "Coordinates" not in particles:
        return np.zeros(0, dtype=bool)
    delta = periodic_delta(particles["Coordinates"], center, box)
    return np.einsum("ij,ij->i", delta, delta) <= radius**2


def _mass_sum(particles: dict, mask: np.ndarray) -> float:
    if not particles or "Masses" not in particles or len(mask) == 0:
        return 0.0
    return float(np.sum(particles["Masses"][mask], dtype=np.float64))


def build_target_rows(
    output_dir: str | Path,
    series_list,
    group_index: int,
    reference_snap: Optional[int],
    aperture_r200: float,
    aperture_pkpc: Optional[float],
    mass_unit: str,
    tracking_verbose: bool,
):
    snapshots = [snap for snap, _ in series_list]
    histories = get_zoom_halo_series(
        output_dir,
        snapshots,
        group_index=group_index,
        reference_snap=reference_snap,
        verbose=tracking_verbose,
    )
    display_rows = []
    csv_rows = []

    fields = {
        0: ["Coordinates", "Masses", "Metallicity", "StarFormationRate"],
        1: ["Coordinates", "Masses"],
        2: ["Coordinates", "Masses"],
        4: ["Coordinates", "Masses"],
        6: ["Coordinates", "Masses"],
    }

    for snap, series in series_list:
        header, _, _ = read_header_and_counts(series.files)
        age = age_of_universe_gyr(
            header["z"],
            header["omega_m"],
            header["omega_lambda"],
            header["h"],
        )
        halo = histories.get(snap)
        if halo is None:
            display_rows.append(
                [f"snapshot_{snap:03d}", f"{header['z']:.3f}", f"{age:.3f}"]
                + ["--"] * 14
            )
            csv_rows.append({
                "snapshot": snap,
                "z": header["z"],
                "age_gyr": age,
                "tracking_status": "gap",
            })
            continue

        center = np.asarray(halo["center"], dtype=float)
        r200 = float(halo["r200_ckpch"])
        r200_pkpc = float(halo["r200_pkpc"])
        h = float(halo.get("h", header["h"]))
        a = float(halo.get("a", header["a"]))
        box = float(halo.get("box_ckpch", header["box"]))
        galaxy_radius = (
            float(aperture_pkpc) * h / a
            if aperture_pkpc is not None
            else float(aperture_r200) * r200
        )
        galaxy_radius_pkpc = galaxy_radius * a / h

        particles = load_particles_within_radius(
            series.key,
            center,
            r200,
            part_types=(0, 1, 2, 4, 6),
            fields_by_type=fields,
        )
        masks = {
            ptype: _aperture_mask(
                particles.get(ptype, {}), center, box, galaxy_radius
            )
            for ptype in (0, 1, 2, 4, 6)
        }

        gas = particles.get(0, {})
        stars = particles.get(4, {})
        dust = particles.get(6, {})
        gas_mass_code = _mass_sum(gas, masks[0])
        star_mass_code = _mass_sum(stars, masks[4])
        dust_mass_code = _mass_sum(dust, masks[6])
        gas_mass = convert_mass(gas_mass_code, h, mass_unit)
        star_mass = convert_mass(star_mass_code, h, mass_unit)
        dust_mass = convert_mass(dust_mass_code, h, mass_unit)
        m200 = convert_mass(float(halo["m200_code"]), h, mass_unit)

        sfr = np.nan
        if (
            "StarFormationRate" in gas
            and len(masks[0])
            and len(gas["StarFormationRate"]) == len(masks[0])
        ):
            sfr = float(np.sum(gas["StarFormationRate"][masks[0]]))

        gas_metal_code = np.nan
        if "Metallicity" in gas and len(masks[0]):
            metallicity = np.asarray(gas["Metallicity"])
            if metallicity.ndim == 1 and len(metallicity) == len(masks[0]):
                gas_metal_code = float(
                    np.sum(
                        gas["Masses"][masks[0]] * metallicity[masks[0]],
                        dtype=np.float64,
                    )
                )

        dgr = dust_mass_code/gas_mass_code if gas_mass_code > 0 else np.nan
        dz_gas = (
            dust_mass_code/gas_metal_code
            if np.isfinite(gas_metal_code) and gas_metal_code > 0 else np.nan
        )
        dz_total = (
            dust_mass_code/(gas_metal_code+dust_mass_code)
            if np.isfinite(gas_metal_code)
            and gas_metal_code+dust_mass_code > 0 else np.nan
        )

        nstar = int(np.count_nonzero(masks[4]))
        ngas = int(np.count_nonzero(masks[0]))
        ndust = int(np.count_nonzero(masks[6]))
        nlr_r200 = len(particles.get(2, {}).get("Coordinates", []))
        group = int(_halo_value(halo, "group_idx", -1))

        display_rows.append([
            f"snapshot_{snap:03d}",
            f"{header['z']:.3f}",
            f"{age:.3f}",
            str(group),
            f"{m200:.3e}",
            f"{r200_pkpc:.2f}",
            f"{galaxy_radius_pkpc:.2f}",
            str(nstar),
            str(ngas),
            str(ndust),
            str(nlr_r200),
            f"{star_mass:.3e}",
            f"{gas_mass:.3e}",
            f"{dust_mass:.3e}",
            f"{sfr:.3e}" if np.isfinite(sfr) else "--",
            f"{dgr:.3e}" if np.isfinite(dgr) else "--",
            f"{dz_total:.3e}" if np.isfinite(dz_total) else "--",
        ])
        csv_rows.append({
            "snapshot": snap,
            "z": header["z"],
            "age_gyr": age,
            "group_index": group,
            f"m200c_{mass_unit_label(mass_unit)}": m200,
            "r200c_pkpc": r200_pkpc,
            "galaxy_aperture_pkpc": galaxy_radius_pkpc,
            "nstar_galaxy": nstar,
            "ngas_galaxy": ngas,
            "ndust_galaxy": ndust,
            "nlrdm_r200": nlr_r200,
            f"mstar_{mass_unit_label(mass_unit)}": star_mass,
            f"mgas_{mass_unit_label(mass_unit)}": gas_mass,
            f"mdust_{mass_unit_label(mass_unit)}": dust_mass,
            "sfr_msun_per_yr": sfr,
            "dust_to_gas": dgr,
            "dust_to_gas_phase_metals": dz_gas,
            "dust_to_total_metals": dz_total,
            "tracking_status": "ok",
        })
    return display_rows, csv_rows


def write_csv(path: Path, rows: Sequence[dict]):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Whole-volume census and tracked target-halo history"
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--scope", choices=("volume", "target", "both"), default="volume"
    )
    parser.add_argument(
        "--group-index", type=int,
        help="Final/reference-snapshot FOF group index; required for target scope",
    )
    parser.add_argument(
        "--reference-snap", type=int,
        help="Reference snapshot for --group-index (default: latest complete snapshot)",
    )
    parser.add_argument(
        "--galaxy-aperture-r200", type=float, default=0.1,
        help="Galaxy aperture as a fraction of evolving R200c (default: 0.1)",
    )
    parser.add_argument(
        "--galaxy-aperture-pkpc", type=float,
        help="Fixed physical-kpc aperture; overrides --galaxy-aperture-r200",
    )
    parser.add_argument(
        "--mass-unit", choices=("code", "msun-h", "msun"), default="msun",
        help="Mass display unit (default: physical Msun)",
    )
    parser.add_argument("--tz", choices=("local", "utc"), default="local")
    parser.add_argument("--csv", type=Path, help="Optional machine-readable output")
    parser.add_argument(
        "--tracking-verbose", action="store_true",
        help="Print detailed HRDM-core progenitor diagnostics",
    )
    args = parser.parse_args()

    if args.scope in ("target", "both") and args.group_index is None:
        parser.error("--group-index is required for --scope target/both")
    if args.galaxy_aperture_r200 <= 0:
        parser.error("--galaxy-aperture-r200 must be positive")
    if args.galaxy_aperture_pkpc is not None and args.galaxy_aperture_pkpc <= 0:
        parser.error("--galaxy-aperture-pkpc must be positive")

    discovered = discover_groups(args.output_dir)
    if not discovered:
        parser.error(f"No snapshots found under {args.output_dir}")
    series_list = pick_newest_series(discovered)
    unit = mass_unit_label(args.mass_unit)

    if args.scope in ("volume", "both"):
        headers = ["SnapshotBase", "z", "Age(Gyr)"]
        headers.extend(PARTICLE_LABELS[ptype] for ptype in range(7))
        headers.extend([f"M*({unit})", f"Mdust({unit})", "LastModified"])
        rows, csv_rows = build_volume_rows(series_list, args.mass_unit, args.tz)
        print("\nWHOLE-VOLUME CENSUS")
        print(format_table(headers, rows))
        if args.csv:
            path = args.csv if args.scope == "volume" else args.csv.with_name(
                args.csv.stem + "_volume" + args.csv.suffix
            )
            write_csv(path, csv_rows)
            print(f"\nCSV: {path}")

    if args.scope in ("target", "both"):
        headers = [
            "SnapshotBase", "z", "Age(Gyr)", "Group", f"M200({unit})",
            "R200(pkpc)", "Rap(pkpc)", "Nstar", "Ngas", "Ndust",
            "NLR<R200", f"Mstar({unit})", f"Mgas({unit})",
            f"Mdust({unit})", "SFR", "D/G", "D/Ztot",
        ]
        rows, csv_rows = build_target_rows(
            args.output_dir,
            series_list,
            args.group_index,
            args.reference_snap,
            args.galaxy_aperture_r200,
            args.galaxy_aperture_pkpc,
            args.mass_unit,
            args.tracking_verbose,
        )
        print("\nTRACKED TARGET-HALO HISTORY")
        print(format_table(headers, rows))
        print(
            "\nD/Ztot = Mdust / (MZ,gas + Mdust); CSV also records "
            "Mdust/MZ,gas explicitly."
        )
        if args.csv:
            path = args.csv if args.scope == "target" else args.csv.with_name(
                args.csv.stem + "_target" + args.csv.suffix
            )
            write_csv(path, csv_rows)
            print(f"CSV: {path}")


if __name__ == "__main__":
    main()
