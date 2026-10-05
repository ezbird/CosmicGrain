#!/usr/bin/env python3
"""Snapshot census and target-halo evolutionary history for CosmicGrain.

Two complementary scopes are printed by default:

``volume``
    Seven-type particle census plus whole-volume stellar and dust masses.
    This is primarily an integrity/bookkeeping view.

``target``
    Main-progenitor history for the suite target.  Its final group is inferred
    from the selected parent-halo catalog and MUSIC2 domain shift; earlier
    snapshots are followed through conserved PartType1 HRDM core IDs using
    halo_utils.  ``--group-index`` remains available as an explicit override.

Mass conventions
----------------
Gadget code masses are 1e10 Msun/h.  Physical masses are therefore
``mass_code * 1e10 / h``.  The former version omitted the division by h and
silently labelled Msun/h values as Msun.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
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
    find_last_snap_num,
    get_zoom_halo_series,
    infer_zoom_target_group,
    infer_zoom_target_group_from_reference,
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

# Parent-box z=0 masses for the named CosmicGrain zoom targets.  These are
# used only as a sanity prior when the position-based matcher cannot identify
# an incomplete run; reported halo properties still come from the zoom.
SUITE_PARENT_M200_MSUN = {
    9235: 5.601e10, 7723: 6.499e10, 5834: 8.766e10,
    3886: 1.332e11, 3879: 1.438e11, 3352: 1.593e11,
    1653: 3.7562e11,
    1481: 4.241e11, 859: 6.808e11,
    441: 1.061e12, 308: 2.3951e12, 295: 2.6368e12,
}


def release_process_memory() -> None:
    """Return unused NumPy/SciPy allocations to the OS when possible.

    ``gc.collect()`` destroys unreachable Python objects, but glibc can retain
    their large allocation arenas.  That resident-memory watermark was enough
    to make a 30-snapshot target history unsafe on the largest zooms.
    ``malloc_trim`` is Linux/glibc-specific, so failure is intentionally benign.
    """
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


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
        # HRDM is used by get_zoom_halo_series for tracking, but is not needed
        # again for the aperture census below.  Avoid loading it twice.
        2: ["Coordinates"],
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
            part_types=(0, 2, 4, 6),
            fields_by_type=fields,
        )
        masks = {
            ptype: _aperture_mask(
                particles.get(ptype, {}), center, box, galaxy_radius
            )
            for ptype in (0, 2, 4, 6)
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
        # h5py/NumPy temporaries can otherwise leave a high resident-memory
        # watermark during a long all-snapshot history on large zooms.
        del particles, masks, gas, stars, dust
        release_process_memory()
    return display_rows, csv_rows


def build_target_rows_memory_safe(
    output_dir: str | Path,
    series_list,
    group_index: int,
    reference_snap: Optional[int],
    aperture_r200: float,
    aperture_pkpc: Optional[float],
    mass_unit: str,
    tracking_verbose: bool,
    batch_size: int,
):
    """Build a full target history without one monolithic particle pass.

    The original ``--target-mode all`` sent every snapshot through one call to
    ``build_target_rows``.  On large zooms, repeated SciPy trees and HDF5/NumPy
    particle arrays left a high resident-memory watermark even after objects
    went out of scope.  Here snapshots are requested in small newest-to-oldest
    batches.  Every batch completes its tracking and aperture census, then all
    large allocations are destroyed and trimmed before the next batch begins.

    A batch size of one is deliberately the default.  It repeats some tracking
    work, making it slower, but bounds the aperture-analysis state to a single
    requested epoch.  Users with ample memory may increase the batch size.
    """
    if batch_size < 1:
        raise ValueError("history batch size must be at least one")
    if len(series_list) <= batch_size:
        return build_target_rows(
            output_dir, series_list, group_index, reference_snap,
            aperture_r200, aperture_pkpc, mass_unit, tracking_verbose,
        )

    rows_by_snap = {}
    csv_by_snap = {}
    total = len(series_list)
    completed = 0

    # Work backward so the inexpensive late-time batches complete first and
    # so progress remains visible during a long progenitor reconstruction.
    for end in range(total, 0, -batch_size):
        start = max(0, end - batch_size)
        batch = series_list[start:end]
        batch_snaps = [snap for snap, _ in batch]
        print(
            f"\n[history] snapshots {batch_snaps[0]:03d}--"
            f"{batch_snaps[-1]:03d} ({completed}/{total} rows complete)",
            flush=True,
        )
        batch_rows, batch_csv = build_target_rows(
            output_dir,
            batch,
            group_index,
            reference_snap,
            aperture_r200,
            aperture_pkpc,
            mass_unit,
            tracking_verbose,
        )
        for (snap, _), display, csv_row in zip(batch, batch_rows, batch_csv):
            rows_by_snap[int(snap)] = display
            csv_by_snap[int(snap)] = csv_row
        completed += len(batch)
        del batch_rows, batch_csv, batch
        release_process_memory()

    ordered_snaps = [int(snap) for snap, _ in series_list]
    return (
        [rows_by_snap[snap] for snap in ordered_snaps],
        [csv_by_snap[snap] for snap in ordered_snaps],
    )


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


def find_same_ic_reference_output(
    output_dir: str | Path,
    required_snap: int,
) -> Optional[Path]:
    """Find the best completed sibling run for same-halo epoch matching.

    Cross-resolution references are allowed.  The reference halo is tracked
    to the target epoch using IDs *within the reference run*, after which its
    center is transferred through the MUSIC coordinate frames.  IDs are never
    compared across mass resolutions.
    """
    current = Path(output_dir).resolve()
    match = re.fullmatch(
        r"halo(?P<halo>[0-9]+)_output_(?P<resolution>[0-9]+)(?:_.*)?",
        current.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    halo_id = int(match.group("halo"))
    resolution = int(match.group("resolution"))
    current_suffix = re.sub(
        rf"^halo{halo_id}_output_{resolution}", "", current.name,
        flags=re.IGNORECASE,
    )
    candidates = []
    for path in current.parent.glob(f"halo{halo_id}_output_*"):
        if not path.is_dir() or path.resolve() == current:
            continue
        candidate_match = re.fullmatch(
            rf"halo{halo_id}_output_(?P<resolution>[0-9]+)(?P<suffix>_.*)?",
            path.name,
            flags=re.IGNORECASE,
        )
        if candidate_match is None:
            continue
        last = find_last_snap_num(path)
        if last is None or last < required_snap:
            continue
        candidate_resolution = int(candidate_match.group("resolution"))
        candidate_suffix = candidate_match.group("suffix") or ""
        same_physics_suffix = int(candidate_suffix == current_suffix)
        same_resolution = int(candidate_resolution == resolution)
        # Prefer the run extending furthest in time, then the closest physics
        # label, then the same resolution.  A completed 1024 run is therefore
        # preferred to an incomplete 2048 sibling when monitoring 2048.
        candidates.append((
            last,
            same_physics_suffix,
            same_resolution,
            path.stat().st_mtime,
            path,
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:4])[4]


def _natural_catalog_key(path: Path):
    """Sort split group catalogs by their numeric file suffix."""
    match = re.search(r"\.(\d+)\.hdf5$", path.name)
    return int(match.group(1)) if match else -1


def infer_latest_pure_zoom_group(output_dir: str | Path, snap: int):
    """Fallback target selection from mass and zoom-region purity.

    This is intentionally conservative.  It is used only after the standard
    parent-position matcher rejects its candidate.  A valid fallback must be
    close to the known parent-halo mass scale and dominated by HRDM rather
    than low-resolution dark matter.
    """
    output_dir = Path(output_dir)
    name_match = re.search(r"halo(\d+)_output_\d+", output_dir.name)
    if not name_match:
        raise ValueError(
            "output directory does not encode a suite halo ID as "
            "halo<ID>_output_<resolution>"
        )
    halo_id = int(name_match.group(1))
    parent_mass = SUITE_PARENT_M200_MSUN.get(halo_id)
    if parent_mass is None:
        raise ValueError(f"no parent-mass prior is known for halo {halo_id}")

    group_dir = output_dir / f"groups_{snap:03d}"
    patterns = (
        f"fof_subhalo_tab_{snap:03d}.*.hdf5",
        f"fof_subhalo_tab_{snap:03d}.hdf5",
        f"group_tab_{snap:03d}.*.hdf5",
        f"group_tab_{snap:03d}.hdf5",
    )
    files = []
    for pattern in patterns:
        files.extend(group_dir.glob(pattern))
    files = sorted(set(files), key=_natural_catalog_key)
    if not files:
        raise OSError(f"no group catalog found for snapshot {snap:03d}")

    m200_parts = []
    r200_parts = []
    length_parts = []
    mass_type_parts = []
    h = None
    a = None
    for filename in files:
        with h5py.File(filename, "r") as handle:
            if "Group" not in handle or "Group_M_Crit200" not in handle["Group"]:
                continue
            group = handle["Group"]
            m200_parts.append(np.asarray(group["Group_M_Crit200"], dtype=float))
            r200_parts.append(np.asarray(group["Group_R_Crit200"], dtype=float))
            length_parts.append(np.asarray(group["GroupLenType"], dtype=np.int64))
            if "GroupMassType" in group:
                mass_type_parts.append(np.asarray(group["GroupMassType"], dtype=float))
            header = handle["Header"].attrs
            parameters = (
                handle["Parameters"].attrs if "Parameters" in handle else {}
            )
            if h is None:
                h = float(np.asarray(
                    header.get("HubbleParam", parameters.get("HubbleParam", 1.0))
                ).squeeze())
                a = float(np.asarray(header.get("Time", 1.0)).squeeze())
    if not m200_parts:
        raise KeyError("group catalogs contain no Group_M_Crit200 data")

    m200 = np.concatenate(m200_parts) * 1.0e10 / h
    r200 = np.concatenate(r200_parts) * a / h
    lengths = np.concatenate(length_parts, axis=0)
    mass_types = (
        np.concatenate(mass_type_parts, axis=0)
        if len(mass_type_parts) == len(m200_parts) else None
    )
    if lengths.ndim != 2 or lengths.shape[1] < 3:
        raise ValueError("GroupLenType does not contain HRDM/LRDM columns")

    n_hr = lengths[:, 1].astype(float)
    n_lr = lengths[:, 2].astype(float)
    if mass_types is not None and mass_types.shape[1] >= 3:
        dm_total = mass_types[:, 1] + mass_types[:, 2]
        contamination = np.divide(
            mass_types[:, 2], dm_total,
            out=np.ones_like(dm_total), where=dm_total > 0,
        )
    else:
        dm_total = n_hr + n_lr
        contamination = np.divide(
            n_lr, dm_total,
            out=np.ones_like(dm_total), where=dm_total > 0,
        )

    ratio = m200 / parent_mass
    candidate = (
        np.isfinite(m200) & (m200 > 0)
        & (ratio >= 0.05) & (ratio <= 2.0)
        & (n_hr >= 100)
        & (contamination <= 0.05)
    )
    indices = np.flatnonzero(candidate)
    if len(indices) == 0:
        raise RuntimeError(
            "no group has 0.05--2.0 times the parent mass, at least 100 "
            "HRDM particles, and <=5% low-resolution DM contamination"
        )

    # Prefer the expected mass scale, with a strong penalty for contamination.
    score = np.abs(np.log10(ratio[indices])) + 30.0 * contamination[indices]
    ranked = indices[np.argsort(score)]
    best = int(ranked[0])

    evidence = {
        "halo_id": halo_id,
        "m200_msun": float(m200[best]),
        "mass_ratio": float(ratio[best]),
        "r200_pkpc": float(r200[best]),
        "n_hrdm": int(n_hr[best]),
        "n_lrdm": int(n_lr[best]),
        "contamination": float(contamination[best]),
    }
    return best, evidence


def resolve_target_group(args, series_list, parser):
    """Resolve the target only when the target table is about to be built."""
    group_index = args.group_index
    reference_snap = args.reference_snap
    if group_index is not None:
        return group_index, reference_snap

    try:
        group_index, reference_snap, _ = infer_zoom_target_group(
            args.output_dir,
            reference_snap=reference_snap,
            verbose=True,
        )
        return group_index, reference_snap
    except (OSError, ValueError, RuntimeError, KeyError) as direct_exc:
        direct_error = str(direct_exc)

    latest_current = max(snap for snap, _ in series_list)
    reference_output = args.reference_output
    if reference_output is None:
        reference_output = find_same_ic_reference_output(
            args.output_dir, latest_current
        )
        if reference_output is not None:
            print(
                "\nDirect z=0 parent-position matching was rejected; "
                "automatically trying a completed same-halo reference run: "
                f"{reference_output}"
            )
    reference_error = "no compatible completed same-halo reference run found"
    if reference_output is not None:
        try:
            group_index, reference_snap, _ = (
                infer_zoom_target_group_from_reference(
                    args.output_dir,
                    reference_output,
                    target_snap=latest_current,
                    verbose=True,
                )
            )
            return group_index, reference_snap
        except (OSError, ValueError, RuntimeError, KeyError) as ref_exc:
            reference_error = str(ref_exc)

    # Mass and nominal FOF purity can identify an attractive interloper, as
    # happened for CG10/2048 at snapshot 019.  Keep this heuristic available
    # only for explicit investigation; never silently promote it to identity.
    if args.allow_purity_fallback:
        try:
            group_index, evidence = infer_latest_pure_zoom_group(
                args.output_dir, latest_current
            )
            print(
                "\nWARNING: using explicitly enabled mass/purity heuristic; "
                "this candidate has no ancestry evidence:"
            )
            print(
                f"  group={group_index}  "
                f"M200={evidence['m200_msun']:.3e} Msun  "
                f"M/Mparent={evidence['mass_ratio']:.3f}  "
                f"R200={evidence['r200_pkpc']:.2f} pkpc  "
                f"N_HRDM={evidence['n_hrdm']:,}  "
                f"N_LRDM={evidence['n_lrdm']:,}  "
                f"f_contam={evidence['contamination']:.3e}"
            )
            return group_index, latest_current
        except (OSError, ValueError, RuntimeError, KeyError) as purity_exc:
            purity_error = str(purity_exc)
    else:
        purity_error = "disabled (enable only with --allow-purity-fallback)"

    parser.error(
        "automatic target selection failed after the whole-volume census: "
        f"direct matcher: {direct_error}; reference matcher: "
        f"{reference_error}; purity heuristic: {purity_error}. "
        "Pass --group-index and --reference-snap for a validated explicit "
        "anchor"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Whole-volume census and tracked target-halo history"
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--scope", choices=("volume", "target", "both"), default="both"
    )
    parser.add_argument(
        "--target-mode", choices=("latest", "all"), default="latest",
        help=("Target-halo rows to calculate. 'latest' is the memory-safe "
              "default for monitoring an active run; 'all' reconstructs the "
              "full target history in bounded-memory batches and is "
              "substantially slower."),
    )
    parser.add_argument(
        "--history-batch-size", type=int, default=1,
        help=("Requested target-history rows per memory-release batch when "
              "--target-mode all is used (default: 1, safest; increase only "
              "on machines with ample RAM)."),
    )
    parser.add_argument(
        "--group-index", type=int,
        help=("Final/reference-snapshot FOF group index override; normally "
              "inferred automatically from suite metadata"),
    )
    parser.add_argument(
        "--reference-snap", type=int,
        help="Reference snapshot for --group-index (default: latest complete snapshot)",
    )
    parser.add_argument(
        "--reference-output", type=Path,
        help=("Trusted completed same-halo run. Its target is ID-tracked "
              "internally to the current epoch, then position/mass/radius "
              "are matched across MUSIC coordinate frames."),
    )
    parser.add_argument(
        "--allow-purity-fallback", action="store_true",
        help=("Explicitly allow the unsafe mass/FOF-purity heuristic when no "
              "ancestry-backed anchor exists. Intended only for diagnostics; "
              "the selected group is not automatically trustworthy."),
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

    if args.galaxy_aperture_r200 <= 0:
        parser.error("--galaxy-aperture-r200 must be positive")
    if args.galaxy_aperture_pkpc is not None and args.galaxy_aperture_pkpc <= 0:
        parser.error("--galaxy-aperture-pkpc must be positive")
    if args.history_batch_size < 1:
        parser.error("--history-batch-size must be at least 1")

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
        group_index, reference_snap = resolve_target_group(
            args, series_list, parser
        )
        headers = [
            "SnapshotBase", "z", "Age(Gyr)", "Group", f"M200({unit})",
            "R200(pkpc)", "Rap(pkpc)", "Nstar", "Ngas", "Ndust",
            "NLR<R200", f"Mstar({unit})", f"Mgas({unit})",
            f"Mdust({unit})", "SFR", "D/G", "D/Ztot",
        ]
        target_series = (
            series_list[-1:] if args.target_mode == "latest" else series_list
        )
        if args.target_mode == "latest":
            print(
                "\nTarget mode: latest snapshot only (memory-safe). "
                "Use --target-mode all for the full history."
            )
        if args.target_mode == "all":
            print(
                "\nTarget mode: full history with bounded-memory batches "
                f"(batch size={args.history_batch_size}).",
                flush=True,
            )
            rows, csv_rows = build_target_rows_memory_safe(
                args.output_dir,
                target_series,
                group_index,
                reference_snap,
                args.galaxy_aperture_r200,
                args.galaxy_aperture_pkpc,
                args.mass_unit,
                args.tracking_verbose,
                args.history_batch_size,
            )
        else:
            rows, csv_rows = build_target_rows(
                args.output_dir,
                target_series,
                group_index,
                reference_snap,
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
