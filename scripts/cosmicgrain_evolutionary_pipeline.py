#!/usr/bin/env python3
"""
CosmicGrain suite-wide evolutionary extractor.

Purpose
-------
Turn the CG01--CG12 zoom suite into evolutionary tracks rather than twelve z=0
endpoints.  The script can be run while simulations are still in progress.

For each selected halo it:
  1. finds every paired snapshot + group catalog currently present;
  2. uses the latest paired output as the reference snapshot;
  3. infers the target group there with halo_utils.infer_zoom_target_group()
     unless --group-map supplies it;
  4. calls halo_utils.get_zoom_halo_series() ONCE to track the same physical
     target backward through all earlier outputs using the conserved HRDM core;
  5. extracts one tidy global row per snapshot;
  6. saves radial profiles and grain-size distributions in one compressed NPZ;
  7. writes per-halo and suite CSVs that can be plotted without reopening the
     full Gadget snapshots.

This is deliberately an EXTRACTOR/CACHE layer.  Publication plotting should be
kept in a separate script.

Important conventions
---------------------
* Galaxy aperture defaults to 30 physical kpc.
* GrainRadius in current CosmicGrain HDF5 snapshots is stored in nm, therefore
  the default conversion to micron is 1e-3.
* Grain-size summary fractions are:
      < 0.01 micron  (<10 nm)
      0.01--0.10 micron (10--100 nm)
      > 0.10 micron (>100 nm)
      < 0.03 micron  (legacy/current comparison diagnostic)
* D/Z = Mdust / sum(Mgas * Zgas), matching the post-run audit.
* Radial profiles in this first extractor use 3-D spherical radius.  They are
  intended as robust evolutionary diagnostics; a later face-on/cylindrical
  pipeline should be used for observational surface-density profiles.
* Repeated snapshots of one halo are correlated evolutionary states, not
  independent galaxies.

Example
-------
# Start immediately with the currently running CG07 / halo1534:
python3 scripts/cosmicgrain_evolutionary_pipeline.py \
    --root . --resolution 1024 --halos CG07 \
    --outdir cosmicgrain_evolution

# Run/re-run all available suite members:
python3 scripts/cosmicgrain_evolutionary_pipeline.py \
    --root . --resolution 1024 \
    --outdir cosmicgrain_evolution
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
import numpy as np

from halo_utils import (
    find_snapshot_and_group_files,
    get_zoom_halo_series,
    infer_zoom_target_group,
    periodic_delta,
)

CG_HALOS = {
    "CG01": 9235, "CG02": 7723, "CG03": 5834, "CG04": 3886,
    "CG05": 3879, "CG06": 3352, "CG07": 1534, "CG08": 1481,
    "CG09": 859, "CG10": 441, "CG11": 308, "CG12": 295,
}

PARENT_M200 = {
    "CG01": 5.601e10, "CG02": 6.499e10, "CG03": 8.766e10,
    "CG04": 1.332e11, "CG05": 1.438e11, "CG06": 1.593e11,
    "CG07": 4.139e11, "CG08": 4.241e11, "CG09": 6.808e11,
    "CG10": 1.061e12, "CG11": 1.353e12, "CG12": 1.954e12,
}

Z_FIELDS = ("Metallicity", "GFM_Metallicity", "Z")
SFR_FIELDS = ("StarFormationRate", "SFR")
DUST_RADIUS_FIELDS = ("GrainRadius", "DustGrainRadius", "GRAD")
DUST_CF_FIELDS = ("CarbonMassFraction", "CarbonFraction", "DustCarbonFraction", "CFRC")
DUST_SOURCE_FIELDS = ("DustSource", "Source", "DustSourceType", "DSRC")

SOURCE_MAP = {0: "SNII", 1: "AGB", 2: "LRN"}


def first_present(fields: dict, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in fields:
            return name
    return None


def discover_indices(output_dir: str) -> Tuple[List[int], List[int], List[int]]:
    snaps, groups = set(), set()
    for p in glob.glob(os.path.join(output_dir, "snapdir_*")):
        m = re.search(r"snapdir_(\d+)$", p)
        if m:
            snaps.add(int(m.group(1)))
    for p in glob.glob(os.path.join(output_dir, "groups_*")):
        m = re.search(r"groups_(\d+)$", p)
        if m:
            groups.add(int(m.group(1)))
    paired = sorted(snaps & groups)
    return sorted(snaps), sorted(groups), paired


def candidate_output_dirs(root: str, cg: str, resolution: int) -> List[str]:
    hid = CG_HALOS[cg]
    patterns = [
        os.path.join(root, f"halo{hid}_output_{resolution}"),
        os.path.join(root, f"halo{hid}_output_{resolution}_*"),
    ]
    found = []
    for pat in patterns:
        for d in glob.glob(pat):
            if os.path.isdir(d) and d not in found:
                found.append(d)
    # Prefer canonical name, then newest modified candidate.
    canonical = os.path.abspath(os.path.join(root, f"halo{hid}_output_{resolution}"))
    return sorted(
        found,
        key=lambda d: (
            0 if os.path.abspath(d) == canonical else 1,
            -os.path.getmtime(d),
        ),
    )


def choose_best_output_dir(root: str, cg: str, resolution: int) -> Optional[str]:
    best = None
    best_key = None
    for d in candidate_output_dirs(root, cg, resolution):
        _, _, paired = discover_indices(d)
        if not paired:
            continue
        # Prefer most advanced paired snapshot; canonical name breaks ties.
        canonical = os.path.abspath(d) == os.path.abspath(
            os.path.join(root, f"halo{CG_HALOS[cg]}_output_{resolution}")
        )
        key = (paired[-1], len(paired), int(canonical), os.path.getmtime(d))
        if best_key is None or key > best_key:
            best_key, best = key, d
    return best


def parse_group_map(text: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        cg, idx = item.split(":", 1)
        cg = cg.strip().upper()
        if cg not in CG_HALOS:
            raise ValueError(f"Unknown CG label in --group-map: {cg}")
        out[cg] = int(idx)
    return out


def snapshot_header(snapshot_files: Sequence[str]) -> dict:
    with h5py.File(snapshot_files[0], "r") as f:
        h = f["Header"].attrs
        a = float(h["Time"])
        return {
            "a": a,
            "z": 1.0 / a - 1.0,
            "h": float(f["Parameters"].attrs["HubbleParam"]),
            "box": float(h["BoxSize"]),
        }


def read_type(snapshot_files: Sequence[str], ptype: int, wanted_fields: Sequence[str]):
    group = f"PartType{ptype}"
    coords, masses = [], []
    fields = {name: [] for name in wanted_fields}
    present = {name: False for name in wanted_fields}

    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            if group not in f:
                continue
            g = f[group]
            if "Coordinates" not in g:
                continue
            n = len(g["Coordinates"])
            if n == 0:
                continue
            coords.append(np.asarray(g["Coordinates"], dtype=np.float64))
            if "Masses" in g:
                masses.append(np.asarray(g["Masses"], dtype=np.float64))
            else:
                mt = float(f["Header"].attrs["MassTable"][ptype])
                masses.append(np.full(n, mt, dtype=np.float64))
            for name in wanted_fields:
                if name in g:
                    fields[name].append(np.asarray(g[name]))
                    present[name] = True

    if not coords:
        return None

    out_fields = {}
    n_total = sum(len(x) for x in coords)
    for name in wanted_fields:
        if not present[name]:
            continue
        chunks = fields[name]
        # A field should exist in every file that contains this particle type.
        # If not, fail loudly instead of silently misaligning particle arrays.
        if sum(len(x) for x in chunks) != n_total:
            raise RuntimeError(
                f"{group}/{name} is present only in part of the snapshot; "
                "cannot align it safely."
            )
        out_fields[name] = np.concatenate(chunks)

    return {
        "coords": np.concatenate(coords),
        "masses": np.concatenate(masses),
        "fields": out_fields,
    }


def radii_pkpc(coords, center, box, a, h):
    delta = periodic_delta(np.asarray(coords), np.asarray(center), box)
    return np.sqrt(np.sum(delta * delta, axis=1)) * a / h


def weighted_percentile(values, weights, percentile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    v, w = values[good], weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(np.interp(percentile / 100.0 * c[-1], c, v))


def enclosed_radius(r, mass, fraction):
    r, mass = np.asarray(r), np.asarray(mass)
    good = np.isfinite(r) & np.isfinite(mass) & (mass > 0)
    if not np.any(good):
        return np.nan
    rr, mm = r[good], mass[good]
    order = np.argsort(rr)
    rr, mm = rr[order], mm[order]
    c = np.cumsum(mm)
    return float(np.interp(fraction * c[-1], c, rr))


def shell_mass(r, mass, edges):
    return np.histogram(r, bins=edges, weights=mass)[0].astype(float)


def shell_weighted_mean(r, values, weights, edges):
    num = np.histogram(r, bins=edges, weights=values * weights)[0].astype(float)
    den = np.histogram(r, bins=edges, weights=weights)[0].astype(float)
    out = np.full(len(edges) - 1, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def safe_ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out


def jsonable(x):
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def analyze_snapshot(
    cg: str,
    output_dir: str,
    snap: int,
    halo: dict,
    aperture_pkpc: float,
    radial_max_pkpc: float,
    radial_bins: int,
    grain_radius_to_micron: float,
    grain_edges_micron: np.ndarray,
):
    snapshot_files, _ = find_snapshot_and_group_files(output_dir, snap)
    hdr = snapshot_header(snapshot_files)

    # get_zoom_halo_series returns tracking dictionaries.
    center = np.asarray(halo["center_ckpch"] if "center_ckpch" in halo else halo["center"])
    a = float(halo.get("a", hdr["a"]))
    h = float(halo.get("h", hdr["h"]))
    box = float(halo.get("box_ckpch", hdr["box"]))
    z = float(halo.get("z", 1.0 / a - 1.0))
    r200_pkpc = float(halo.get("r200_pkpc", halo["r200_ckpch"] * a / h))
    m200_msun = float(halo.get("m200_msun", halo["m200_code"] * 1.0e10 / h))
    group_idx = int(halo.get("group_idx", halo.get("group_index", -1)))
    mass_to_msun = 1.0e10 / h

    gas = read_type(snapshot_files, 0, Z_FIELDS + SFR_FIELDS)
    stars = read_type(snapshot_files, 4, Z_FIELDS)
    dust = read_type(snapshot_files, 6, DUST_RADIUS_FIELDS + DUST_CF_FIELDS + DUST_SOURCE_FIELDS)

    if gas is None:
        raise RuntimeError("PartType0 gas is absent")
    if stars is None:
        # Early snapshots can legitimately predate star formation.
        stars = {"coords": np.empty((0, 3)), "masses": np.empty(0), "fields": {}}
    if dust is None:
        # Likewise dust can legitimately be absent at early times.
        dust = {"coords": np.empty((0, 3)), "masses": np.empty(0), "fields": {}}

    rg = radii_pkpc(gas["coords"], center, box, a, h)
    rs = radii_pkpc(stars["coords"], center, box, a, h) if len(stars["coords"]) else np.empty(0)
    rd = radii_pkpc(dust["coords"], center, box, a, h) if len(dust["coords"]) else np.empty(0)

    mg = gas["masses"] * mass_to_msun
    ms = stars["masses"] * mass_to_msun
    md = dust["masses"] * mass_to_msun

    qg, qs, qd = rg < aperture_pkpc, rs < aperture_pkpc, rd < aperture_pkpc

    zfield = first_present(gas["fields"], Z_FIELDS)
    sfrfield = first_present(gas["fields"], SFR_FIELDS)
    radius_field = first_present(dust["fields"], DUST_RADIUS_FIELDS)
    cf_field = first_present(dust["fields"], DUST_CF_FIELDS)
    source_field = first_present(dust["fields"], DUST_SOURCE_FIELDS)

    Mgas = float(mg[qg].sum())
    Mstar = float(ms[qs].sum())
    Mdust = float(md[qd].sum())

    if zfield:
        zg = np.asarray(gas["fields"][zfield], dtype=float).reshape(-1)
        MZgas = float(np.sum(mg[qg] * zg[qg]))
        Zgas = MZgas / Mgas if Mgas > 0 else np.nan
    else:
        zg = np.full(len(mg), np.nan)
        MZgas = Zgas = np.nan

    if sfrfield:
        sfr_arr = np.asarray(gas["fields"][sfrfield], dtype=float).reshape(-1)
        SFR = float(np.sum(sfr_arr[qg]))
    else:
        SFR = np.nan

    sSFR = SFR / Mstar if Mstar > 0 else np.nan
    DG = Mdust / Mgas if Mgas > 0 else np.nan
    DZ = Mdust / MZgas if MZgas > 0 else np.nan
    dust_star = Mdust / Mstar if Mstar > 0 else np.nan

    # Extents inside fixed galaxy aperture.
    r50d = enclosed_radius(rd[qd], md[qd], 0.50)
    r90d = enclosed_radius(rd[qd], md[qd], 0.90)
    r50s = enclosed_radius(rs[qs], ms[qs], 0.50)
    r90s = enclosed_radius(rs[qs], ms[qs], 0.90)
    r50g = enclosed_radius(rg[qg], mg[qg], 0.50)
    r90g = enclosed_radius(rg[qg], mg[qg], 0.90)

    # Grain-size summaries.
    grain_p16 = grain_p50 = grain_p84 = np.nan
    f_lt10 = f_10_100 = f_gt100 = f_lt30 = np.nan
    grain_hist_mass = np.zeros(len(grain_edges_micron) - 1)
    grain_hist_number = np.zeros(len(grain_edges_micron) - 1)

    if radius_field and len(md):
        grain = np.asarray(dust["fields"][radius_field], dtype=float).reshape(-1) * grain_radius_to_micron
        good = qd & np.isfinite(grain) & (grain > 0)
        total_good = md[good].sum()
        if total_good > 0:
            grain_p16 = weighted_percentile(grain[good], md[good], 16)
            grain_p50 = weighted_percentile(grain[good], md[good], 50)
            grain_p84 = weighted_percentile(grain[good], md[good], 84)
            f_lt10 = float(md[good & (grain < 0.01)].sum() / total_good)
            f_10_100 = float(md[good & (grain >= 0.01) & (grain < 0.10)].sum() / total_good)
            f_gt100 = float(md[good & (grain >= 0.10)].sum() / total_good)
            f_lt30 = float(md[good & (grain < 0.03)].sum() / total_good)
            grain_hist_mass, _ = np.histogram(grain[good], bins=grain_edges_micron, weights=md[good])
            grain_hist_number, _ = np.histogram(grain[good], bins=grain_edges_micron)
            dlog = np.diff(np.log10(grain_edges_micron))
            grain_hist_mass = grain_hist_mass / dlog
            grain_hist_number = grain_hist_number / dlog

    # Composition.
    carbon_fraction = np.nan
    carbon_mass = silicate_mass = np.nan
    if cf_field and len(md):
        cf = np.asarray(dust["fields"][cf_field], dtype=float).reshape(-1)
        good = qd & np.isfinite(cf)
        if md[good].sum() > 0:
            carbon_mass = float(np.sum(md[good] * cf[good]))
            silicate_mass = float(np.sum(md[good] * (1.0 - cf[good])))
            carbon_fraction = carbon_mass / md[good].sum()

    # Source provenance.
    source_fractions = {label: np.nan for label in SOURCE_MAP.values()}
    source_masses = {label: np.nan for label in SOURCE_MAP.values()}
    if source_field and len(md):
        src = np.asarray(dust["fields"][source_field]).reshape(-1)
        total = md[qd].sum()
        for code, label in SOURCE_MAP.items():
            m = float(md[qd & (src == code)].sum())
            source_masses[label] = m
            source_fractions[label] = m / total if total > 0 else np.nan

    # Spherical radial diagnostics. Store shell masses rather than calling them
    # projected surface densities; plotting can normalize appropriately later.
    rmax = min(radial_max_pkpc, r200_pkpc)
    edges = np.linspace(0.0, rmax, radial_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_g = shell_mass(rg, mg, edges)
    shell_s = shell_mass(rs, ms, edges)
    shell_d = shell_mass(rd, md, edges)
    shell_z = shell_mass(rg, mg * np.nan_to_num(zg, nan=0.0), edges) if zfield else np.full(radial_bins, np.nan)
    dg_r = safe_ratio(shell_d, shell_g)
    dz_r = safe_ratio(shell_d, shell_z)

    carbon_r = np.full(radial_bins, np.nan)
    if cf_field and len(md):
        cf = np.asarray(dust["fields"][cf_field], dtype=float).reshape(-1)
        good = np.isfinite(cf)
        carbon_r = shell_weighted_mean(rd[good], cf[good], md[good], edges)

    # Radial source fractions.
    source_r = {}
    if source_field and len(md):
        src = np.asarray(dust["fields"][source_field]).reshape(-1)
        total_shell = shell_mass(rd, md, edges)
        for code, label in SOURCE_MAP.items():
            m_shell = shell_mass(rd[src == code], md[src == code], edges)
            source_r[label] = safe_ratio(m_shell, total_shell)
    else:
        for label in SOURCE_MAP.values():
            source_r[label] = np.full(radial_bins, np.nan)

    row = {
        "CG": cg,
        "halo_id": CG_HALOS[cg],
        "output_dir": output_dir,
        "snapshot": snap,
        "group_index": group_idx,
        "a": a,
        "z": z,
        "parent_M200c_Msun": PARENT_M200[cg],
        "zoom_M200c_Msun": m200_msun,
        "R200c_pkpc": r200_pkpc,
        "aperture_pkpc": aperture_pkpc,
        "N_gas_ap": int(qg.sum()),
        "N_star_ap": int(qs.sum()),
        "N_dust_ap": int(qd.sum()),
        "Mstar_Msun": Mstar,
        "Mgas_Msun": Mgas,
        "Mdust_Msun": Mdust,
        "MZgas_Msun": MZgas,
        "SFR_Msun_yr": SFR,
        "sSFR_yr_inv": sSFR,
        "Zgas_mass_fraction": Zgas,
        "dust_to_stellar": dust_star,
        "dust_to_gas": DG,
        "dust_to_metals": DZ,
        "R50_dust_pkpc": r50d,
        "R90_dust_pkpc": r90d,
        "R50_star_pkpc": r50s,
        "R90_star_pkpc": r90s,
        "R50_gas_pkpc": r50g,
        "R90_gas_pkpc": r90g,
        "R50dust_over_R50star": r50d / r50s if r50s > 0 else np.nan,
        "R50dust_over_R50gas": r50d / r50g if r50g > 0 else np.nan,
        "grain_p16_micron": grain_p16,
        "grain_mass_median_micron": grain_p50,
        "grain_p84_micron": grain_p84,
        "grain_mass_fraction_lt_0p01um": f_lt10,
        "grain_mass_fraction_0p01_0p10um": f_10_100,
        "grain_mass_fraction_ge_0p10um": f_gt100,
        "small_grain_mass_fraction_lt_0p03um": f_lt30,
        "carbon_mass_fraction": carbon_fraction,
        "carbon_dust_mass_Msun": carbon_mass,
        "silicate_dust_mass_Msun": silicate_mass,
        **{f"source_{k}_mass_Msun": source_masses[k] for k in SOURCE_MAP.values()},
        **{f"source_{k}_mass_fraction": source_fractions[k] for k in SOURCE_MAP.values()},
    }

    profile = {
        "r_edges_pkpc": edges,
        "r_pkpc": centers,
        "r_over_r200": centers / r200_pkpc,
        "Mgas_shell_Msun": shell_g,
        "Mstar_shell_Msun": shell_s,
        "Mdust_shell_Msun": shell_d,
        "MZgas_shell_Msun": shell_z,
        "dust_to_gas": dg_r,
        "dust_to_metals": dz_r,
        "carbon_mass_fraction": carbon_r,
        **{f"source_{k}_mass_fraction": source_r[k] for k in SOURCE_MAP.values()},
        "grain_edges_micron": grain_edges_micron,
        "grain_centers_micron": np.sqrt(grain_edges_micron[:-1] * grain_edges_micron[1:]),
        "grain_dM_dloga": grain_hist_mass,
        "grain_dN_dloga": grain_hist_number,
    }
    return row, profile


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_profile_archive(path: Path, cg: str, profiles: Dict[int, dict]):
    arrays = {}
    for snap, prof in sorted(profiles.items()):
        prefix = f"snap{snap:03d}"
        for key, value in prof.items():
            arrays[f"{prefix}__{key}"] = np.asarray(value)
    arrays["snapshots"] = np.asarray(sorted(profiles), dtype=int)
    arrays["CG"] = np.asarray(cg)
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser(
        description="Extract all-snapshot CosmicGrain evolutionary diagnostics."
    )
    ap.add_argument("--root", default=".", help="Directory containing halo*_output_* runs.")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--halos", nargs="*", default=None,
                    help="Subset, e.g. --halos CG07 CG10. Default: all CG01..CG12.")
    ap.add_argument("--outdir", default="cosmicgrain_evolution")
    ap.add_argument("--aperture-pkpc", type=float, default=30.0)
    ap.add_argument("--radial-max-pkpc", type=float, default=30.0)
    ap.add_argument("--radial-bins", type=int, default=30)
    ap.add_argument("--grain-radius-to-micron", type=float, default=1.0e-3,
                    help="Current CosmicGrain snapshots store GrainRadius in nm.")
    ap.add_argument("--grain-min-micron", type=float, default=1.0e-4)
    ap.add_argument("--grain-max-micron", type=float, default=20.0)
    ap.add_argument("--grain-bins", type=int, default=60)
    ap.add_argument("--group-map", default=None,
                    help='Manual reference groups, e.g. "CG07:4,CG10:1". '
                         "Normally unnecessary.")
    ap.add_argument("--verbose-tracking", action="store_true")
    args = ap.parse_args()

    selected = [x.upper() for x in args.halos] if args.halos else list(CG_HALOS)
    bad = [x for x in selected if x not in CG_HALOS]
    if bad:
        ap.error(f"Unknown halo labels: {bad}")

    group_map = parse_group_map(args.group_map)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grain_edges = np.logspace(
        np.log10(args.grain_min_micron),
        np.log10(args.grain_max_micron),
        args.grain_bins + 1,
    )

    suite_rows: List[dict] = []
    manifest = {
        "resolution": args.resolution,
        "aperture_pkpc": args.aperture_pkpc,
        "grain_radius_to_micron": args.grain_radius_to_micron,
        "halos": {},
    }

    for cg in selected:
        d = choose_best_output_dir(args.root, cg, args.resolution)
        if d is None:
            print(f"[{cg}] no paired snapshot/catalog found; skipping")
            continue

        _, _, paired = discover_indices(d)
        ref_snap = paired[-1]
        print(f"\n[{cg}] halo{CG_HALOS[cg]}: {len(paired)} paired outputs "
              f"{paired[0]:03d}..{ref_snap:03d} in {d}")

        if cg in group_map:
            ref_group = group_map[cg]
            group_source = "--group-map"
        else:
            try:
                ref_group, inferred_ref, _ = infer_zoom_target_group(
                    d, reference_snap=ref_snap, verbose=True
                )
                ref_group = int(ref_group)
                group_source = "infer_zoom_target_group"
                if inferred_ref is not None and int(inferred_ref) != ref_snap:
                    raise RuntimeError(
                        f"inference returned reference {inferred_ref}, expected {ref_snap}"
                    )
            except Exception as exc:
                print(f"[{cg}] target inference FAILED: {exc}", file=sys.stderr)
                continue

        print(f"[{cg}] reference snap={ref_snap:03d}, group={ref_group} [{group_source}]")
        print(f"[{cg}] tracking target backward through {len(paired)} snapshots ...")

        try:
            halo_series = get_zoom_halo_series(
                d,
                paired,
                group_index=ref_group,
                reference_snap=ref_snap,
                verbose=args.verbose_tracking,
                refine_reference_center=True,
                refine_historical_centers=False,
            )
        except Exception as exc:
            print(f"[{cg}] halo tracking FAILED: {exc}", file=sys.stderr)
            continue

        rows, profiles = [], {}
        failed = {}
        for j, snap in enumerate(paired, 1):
            halo = halo_series.get(snap)
            if halo is None:
                failed[str(snap)] = "target tracking returned None"
                print(f"[{cg}] {snap:03d}: no tracked target")
                continue
            try:
                row, prof = analyze_snapshot(
                    cg, d, snap, halo,
                    args.aperture_pkpc,
                    args.radial_max_pkpc,
                    args.radial_bins,
                    args.grain_radius_to_micron,
                    grain_edges,
                )
                rows.append(row)
                profiles[snap] = prof
                print(
                    f"[{cg}] {snap:03d} ({j:2d}/{len(paired)}) "
                    f"z={row['z']:6.3f}  "
                    f"M*={row['Mstar_Msun']:.3e}  "
                    f"Md={row['Mdust_Msun']:.3e}  "
                    f"D/Z={row['dust_to_metals']:.3g}  "
                    f"f<10nm={row['grain_mass_fraction_lt_0p01um']:.3g}"
                )
            except Exception as exc:
                failed[str(snap)] = f"{type(exc).__name__}: {exc}"
                print(f"[{cg}] {snap:03d}: FAILED: {exc}", file=sys.stderr)

        rows.sort(key=lambda x: x["snapshot"])
        suite_rows.extend(rows)

        write_csv(outdir / f"{cg}_evolution.csv", rows)
        save_profile_archive(outdir / f"{cg}_profiles.npz", cg, profiles)
        with (outdir / f"{cg}_evolution.json").open("w") as f:
            json.dump(jsonable(rows), f, indent=2, allow_nan=False)

        manifest["halos"][cg] = {
            "halo_id": CG_HALOS[cg],
            "output_dir": d,
            "reference_snapshot": ref_snap,
            "reference_group": ref_group,
            "group_source": group_source,
            "paired_snapshots": paired,
            "successful_snapshots": [r["snapshot"] for r in rows],
            "failed_snapshots": failed,
        }

        print(f"[{cg}] wrote {len(rows)} evolutionary states")

    suite_rows.sort(key=lambda x: (x["CG"], x["snapshot"]))
    if suite_rows:
        write_csv(outdir / "cosmicgrain_evolution_all.csv", suite_rows)
        with (outdir / "cosmicgrain_evolution_all.json").open("w") as f:
            json.dump(jsonable(suite_rows), f, indent=2, allow_nan=False)

    with (outdir / "manifest.json").open("w") as f:
        json.dump(jsonable(manifest), f, indent=2, allow_nan=False)

    print("\n" + "=" * 72)
    print(f"Evolution extraction complete: {len(suite_rows)} successful halo-snapshots")
    print(f"Output: {outdir}")
    print("Primary table: cosmicgrain_evolution_all.csv")
    print("Per-halo radial/size caches: CGxx_profiles.npz")
    print("=" * 72)


if __name__ == "__main__":
    main()
