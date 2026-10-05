#!/usr/bin/env python3
"""
CosmicGrain multi-halo suite comparison.

Auto-discovers whichever of CG01..CG12 have completed 1024^3 zoom outputs,
loads their final paired snapshot/catalog, and produces a standardized set of
suite-level publication diagnostics.

Designed to use the same halo definition as cosmicgrain_zoom_postrun_audit_v3.py:
    halo_utils.find_snapshot_and_group_files
    halo_utils.get_zoom_halo
    halo_utils.periodic_delta

Best workflow
-------------
Run cosmicgrain_zoom_postrun_audit_v3.py once for each completed halo. This
script then reads the resulting *_postrun_audit.json to recover the accepted
group index automatically. If no audit JSON exists, supply --group-map.

Default CosmicGrain mapping:
    CG01 halo9235
    CG02 halo7723
    CG03 halo5834
    CG04 halo3886
    CG05 halo3879
    CG06 halo3352
    CG07 halo1653
    CG08 halo1481
    CG09 halo859
    CG10 halo441
    CG11 halo308
    CG12 halo295

Outputs
-------
<outdir>/
    cosmicgrain_multihalo_summary.csv
    cosmicgrain_multihalo_summary.json
    table_halo_sample.tex
    fig01_global_masses.png
    fig02_dust_scaling_relations.png
    fig03_radial_surface_density.png
    fig04_radial_dust_ratios.png
    fig05_dust_extent.png
    fig06_grain_size_distributions.png
    fig07_grain_size_summary.png
    fig08_composition_sources.png
    fig09_composition_radial.png
    fig10_dust_lifecycle_plane.png

Notes
-----
* Global galaxy quantities use r < 0.1 R200c by default.  --aperture-pkpc
  remains available as a fixed-radius override.  The summary also records the
  CGM (central boundary < r < R200c) and the entire halo (r < R200c).
* Radial profiles are spherical-shell projected surface-density profiles:
  mass in a 3-D radial shell divided by the corresponding projected annular
  area. This is orientation-independent and therefore robust for the first
  suite comparison. A later face-on disk analysis can add cylindrical profiles.
* Both common D/Z conventions are retained in the summary.  The LaTeX sample
  table uses the conventional fraction of all metals locked in dust:
      Mdust / (sum(Mgas * Zgas) + Mdust)
  while dust_to_gas_phase_metals preserves the audit-style ratio
  Mdust / sum(Mgas * Zgas).
* CosmicGrain snapshot PartType6/GrainRadius is stored in nm (the snapshot
  I/O applies the internal cm -> nm conversion on write).  This script therefore
  converts nm -> micron with a factor of 1e-3 before all grain-size diagnostics.
  If it is already microns, leave the default 1.
* This suite script owns current paper Figs. 1-3. Dedicated grain/R(V) and
  Milky Way scripts own paper Figs. 4-6; this script does not calculate R(V).
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
import warnings
from urllib.parse import urlencode
from urllib.request import urlopen
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from halo_utils import (
    find_snapshot_and_group_files,
    get_zoom_halo,
    infer_zoom_target_group,
    periodic_delta,
)


def apply_cosmicgrain_style(style_file=None):
    """Load the project Matplotlib style, failing loudly if it is absent."""
    candidates = []
    if style_file:
        candidates.append(os.path.abspath(style_file))
    candidates.extend([
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cosmicgrain.mplstyle"),
        os.path.join(os.getcwd(), "cosmicgrain.mplstyle"),
    ])
    for path in candidates:
        if os.path.isfile(path):
            plt.style.use(path)
            print(f"Using Matplotlib style: {path}")
            return path
    raise FileNotFoundError(
        "cosmicgrain.mplstyle was not found beside this script or in the current "
        "directory. Put it in scripts/ or pass --style-file PATH."
    )


CG_HALOS = {
    "CG01": 9235,
    "CG02": 7723,
    "CG03": 5834,
    "CG04": 3886,
    "CG05": 3879,
    "CG06": 3352,
    "CG07": 1653,
    "CG08": 1481,
    "CG09": 859,
    "CG10": 441,
    "CG11": 308,
    "CG12": 295,
}

# Parent DM-only M200c values supplied for the selected 12-halo suite.
PARENT_M200 = {
    "CG01": 5.601e10,
    "CG02": 6.499e10,
    "CG03": 8.766e10,
    "CG04": 1.332e11,
    "CG05": 1.438e11,
    "CG06": 1.593e11,
    "CG07": 4.139e11,
    "CG08": 4.241e11,
    "CG09": 6.808e11,
    "CG10": 1.061e12,
    "CG11": 1.353e12,
    "CG12": 1.954e12,
}

Z_FIELDS = ("Metallicity", "GFM_Metallicity", "Z")
OXYGEN_FIELDS = ("GasOxygenMassFraction", "OxygenMassFraction")
SFR_FIELDS = ("StarFormationRate", "SFR")
DUST_RADIUS_FIELDS = ("GrainRadius", "DustGrainRadius", "GRAD")
DUST_CF_FIELDS = ("CarbonMassFraction", "CarbonFraction", "DustCarbonFraction", "CFRC")
DUST_SOURCE_FIELDS = ("DustSource", "Source", "DustSourceType", "DSRC")
SOURCE_MAP_DEFAULT = {0: "SNII", 1: "AGB", 2: "LRN"}
SILICATE_OXYGEN_MASS_FRACTION = 0.69876708
Z_SOLAR = 0.0134

HALO_GROUPS = (
    ("Low-mass dwarf", ("CG01", "CG02", "CG03")),
    ("Magellanic-mass dwarf", ("CG04", "CG05", "CG06")),
    ("Intermediate and sub-Milky Way", ("CG07", "CG08", "CG09")),
    ("Milky-Way-mass and higher", ("CG10", "CG11", "CG12")),
)

# Peer-reviewed catalog data used for paper Fig. 2. De Vis et al. (2019),
# A&A 623 A5, J/A+A/623/A5/dp-mg: characteristic O/H and TOTAL metal mass.
# Nersesian et al. (2019), A&A 624 A80, J/A+A/624/A80/themis: dust mass.
# De Vis et al. explicitly used these Nersesian THEMIS masses in their study;
# joining the two released catalogs recreates their published D/G and D/Z
# definitions rather than mixing unrelated dust-mass prescriptions.
DEVIS_TABLES = {
    'metal': 'J/A+A/623/A5/dp-mg',
    'dust': 'J/A+A/624/A80/themis',
}


def _vizier_catalog(table, cache_path):
    """Fetch one public VizieR TSV once; retain raw catalog for reproducibility."""
    if not os.path.isfile(cache_path):
        query = urlencode({
            '-source': table, '-out.all': '1', '-out.max': 'unlimited',
        })
        body = None
        for host in ('https://vizier.cds.unistra.fr', 'https://vizier.cfa.harvard.edu'):
            try:
                with urlopen(f'{host}/viz-bin/asu-tsv?{query}', timeout=25) as response:
                    body = response.read(5_000_000)
                break
            except OSError:
                if host == 'https://vizier.cfa.harvard.edu':
                    raise
        if not body or len(body) >= 5_000_000:
            raise ValueError(f'empty or truncated VizieR response for {table}')
        with open(cache_path + '.tmp', 'wb') as handle:
            handle.write(body)
        os.replace(cache_path + '.tmp', cache_path)
    # VizieR ASU TSV marks the *column-name row* with a leading '#'. A normal
    # DictReader after filtering comments discards that row and misreads the
    # unit or separator row as field names, yielding zero valid galaxies.
    required = (('Name', 'MzPG16S', '12+logOHPG16S')
                if '/dp-mg' in table else ('Galaxy', 'Mdust'))
    with open(cache_path, encoding='utf-8-sig') as handle:
        lines = handle.readlines()
    header = None
    first_data = None
    for index, line in enumerate(lines):
        candidate = line.lstrip('#').strip()
        if '\t' not in candidate:
            continue
        fields = [field.strip().strip('"') for field in next(csv.reader([candidate], delimiter='\t'))]
        if all(name in fields for name in required):
            header, first_data = fields, index + 1
            break
    if header is None:
        raise ValueError(f'VizieR table {table} lacks columns {required}; '
                         f'check cached {cache_path}')
    result = []
    for line in lines[first_data:]:
        if not line.strip() or line.startswith('#'):
            continue
        values = next(csv.reader([line], delimiter='\t'))
        if len(values) == len(header):
            result.append(dict(zip(header, values)))
    return result


def _column(row, name):
    """VizieR TSV sometimes quotes or pads column names."""
    for key, value in row.items():
        if key and key.strip().strip('"') == name:
            return value.strip() if value else ''
    return ''


def load_devis_galaxies(outdir):
    """Return measured (O/H, D/G, D/Z) for joined DustPedia galaxies.

    MzPG16S is stored in the VizieR table as log10(MZ/Msun), despite its
    terse catalog description as a metal mass; MZ includes dust metals
    (De Vis et al. 2019, Section 4.2).
    Their adopted gas metal mass fraction is 27.36*(O/H), so their implied
    gas mass is (MzPG16S-Mdust)/fZ and D/G is Mdust/Mgas. De Vis et al.
    used the Nersesian THEMIS masses in their original D/G relation.
    """
    metal = _vizier_catalog(DEVIS_TABLES['metal'], os.path.join(outdir, 'devis2019_dp-mg_vizier.tsv'))
    dust = _vizier_catalog(DEVIS_TABLES['dust'], os.path.join(outdir, 'nersesian2019_themis_vizier.tsv'))

    def norm(name):
        clean = re.sub(r'[^A-Z0-9]', '', name.upper())
        match = re.match(r'^(NGC|IC|UGC|PGC)(0*)([1-9][0-9]*)(.*)$', clean)
        return (match.group(1) + match.group(3) + match.group(4)) if match else clean

    dust_by_name = {}
    dust_example = []
    for row in dust:
        name = norm(_column(row, 'Galaxy'))
        if name and len(dust_example) < 3 and not name.startswith('-'):
            dust_example.append((_column(row, 'Galaxy'), _column(row, 'Mdust')))
        try:
            md = float(_column(row, 'Mdust'))
        except ValueError:
            continue
        if name and np.isfinite(md) and md > 0:
            dust_by_name[name] = md
    result = []
    catalog_rows = []
    shared_names = 0
    matched_example = []
    for row in metal:
        name = norm(_column(row, 'Name'))
        if name not in dust_by_name:
            continue
        shared_names += 1
        if len(matched_example) < 3:
            matched_example.append((_column(row, 'Name'),
                                    _column(row, '12+logOHPG16S'),
                                    _column(row, 'MzPG16S'), dust_by_name[name]))
        try:
            oh = float(_column(row, '12+logOHPG16S'))
            log_mz = float(_column(row, 'MzPG16S'))
        except ValueError:
            continue
        md = dust_by_name[name]
        if not (np.isfinite(oh) and np.isfinite(log_mz) and
                6 < oh < 10 and 0 < log_mz < 15):
            continue
        mz = 10.0 ** log_mz
        if mz <= md:
            continue
        zgas = 27.36 * 10 ** (oh - 12)
        mgas = (mz - md) / zgas
        dg = md / mgas
        dz = md / mz
        result.append((oh, dg, dz))
        catalog_rows.append((_column(row, 'Name'), oh, md, mgas, mz, dg, dz))
    if len(result) < 25:
        metal_names = [_column(row, 'Name') for row in metal if _column(row, 'Name')][:3]
        dust_names = [_column(row, 'Galaxy') for row in dust if _column(row, 'Galaxy')][:3]
        raise ValueError(f'Only {len(result)} joined galaxies; metal rows={len(metal)} '
                         f'dust rows={len(dust)}, valid dust masses={len(dust_by_name)}, '
                         f'shared names={shared_names}; example names metal={metal_names}, '
                         f'dust={dust_names}; dust name/mass examples={dust_example}; '
                         f'matched name/OH/MZ/Mdust={matched_example}; '
                         'check cached TSV columns and units')
    with open(os.path.join(outdir, 'devis2019_joined_galaxy_points.csv'),
              'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('galaxy', '12+logOH_PG16S_at_0p4R25',
                         'Mdust_Nersesian_THEMIS_Msun',
                         'Mgas_reconstructed_from_DeVis_MZ_Msun',
                         'MZtotal_DeVis_Msun', 'dust_to_gas',
                         'dust_to_total_metals'))
        writer.writerows(catalog_rows)
    return np.asarray(result)


@dataclass
class RunInfo:
    cg: str
    halo_id: int
    output_dir: str
    snap: int
    group_index: int
    audit_json: Optional[str]
    group_source: str


def first_present(names: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    names = set(names)
    return next((x for x in candidates if x in names), None)


def finite_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def discover_indices(output_dir: str, kind: str) -> List[int]:
    if kind == "snapshot":
        patterns = ("snapdir_*", "snapshot_*.hdf5", "snapshot_.*.hdf5")
        regex = re.compile(r"(?:snapdir|snapshot)_(\d+)")
    else:
        patterns = ("groups_*", "fof_subhalo_tab_*.hdf5", "fof_subhalo_tab_.*.hdf5")
        regex = re.compile(r"(?:groups|fof_subhalo_tab)_(\d+)")
    found = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            m = regex.search(os.path.basename(path))
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def nearest_paired_snapshot(output_dir: str, target_redshift: float) -> Optional[Tuple[int, float]]:
    """Return the complete snapshot/catalog pair nearest a requested redshift."""
    paired = sorted(
        set(discover_indices(output_dir, "snapshot"))
        & set(discover_indices(output_dir, "catalog"))
    )
    candidates = []
    for snap in paired:
        try:
            snapshot_files, _ = find_snapshot_and_group_files(output_dir, snap)
            z = snapshot_header(snapshot_files)["z"]
            candidates.append((abs(z - target_redshift), snap, z))
        except (OSError, KeyError, IndexError, ValueError):
            continue
    if not candidates:
        return None
    _, snap, z = min(candidates)
    return int(snap), float(z)


def find_audit_json(output_dir: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(output_dir, "*_postrun_audit.json")))
    if not candidates:
        return None
    # Prefer newest in case the audit was rerun.
    return max(candidates, key=os.path.getmtime)


def load_audit_group(audit_json: str) -> Optional[int]:
    try:
        with open(audit_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data["halo"]["group_index"])
    except Exception:
        return None


def parse_group_map(text: Optional[str]) -> Dict[str, int]:
    out = {}
    if not text:
        return out
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[key.strip().upper()] = int(value)
    return out


def candidate_output_dirs(root: str, cg: str, halo_id: int, resolution: int) -> List[str]:
    # Exact canonical names first, then common rerun suffixes.
    exact = [
        os.path.join(root, f"halo{halo_id}_output_{resolution}"),
        os.path.join(root, f"{cg}_output_{resolution}"),
    ]
    patterns = [
        os.path.join(root, f"halo{halo_id}_output_{resolution}*"),
        os.path.join(root, f"{cg}_output_{resolution}*"),
    ]
    found = []
    for p in exact:
        if os.path.isdir(p):
            found.append(p)
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            if os.path.isdir(p) and p not in found:
                found.append(p)
    return found


def choose_completed_run(
    root: str,
    cg: str,
    halo_id: int,
    resolution: int,
    expected_snapshots: int,
    group_map: Dict[str, int],
    require_complete: bool,
) -> Optional[RunInfo]:
    candidates = candidate_output_dirs(root, cg, halo_id, resolution)
    valid = []
    for d in candidates:
        snaps = discover_indices(d, "snapshot")
        cats = discover_indices(d, "catalog")
        paired = sorted(set(snaps) & set(cats))
        if not paired:
            continue
        if require_complete and (len(snaps) < expected_snapshots or len(cats) < expected_snapshots):
            continue

        audit = find_audit_json(d)
        group = load_audit_group(audit) if audit else None
        group_source = os.path.basename(audit) if group is not None and audit else None

        # Explicit --group-map remains available as a manual fallback/override
        # when no valid audit group exists.
        if group is None and cg in group_map:
            group = int(group_map[cg])
            group_source = "--group-map"

        # Normal suite workflow: infer the final/reference target automatically
        # from the selected parent halo + MUSIC2 domain shift.  This is the same
        # machinery used by list_stars_by_snapshot.py.  Never substitute
        # "largest FOF group" here: the zoom target need not be catalog group 0.
        if group is None:
            try:
                inferred_group, inferred_ref_snap, _ = infer_zoom_target_group(
                    d,
                    reference_snap=paired[-1],
                    verbose=False,
                )
                group = int(inferred_group)
                group_source = "auto:infer_zoom_target_group"
                # We explicitly requested paired[-1] as the reference.  Keep the
                # returned value only as a sanity check; analysis is performed
                # on the latest paired snapshot selected for this run.
                if inferred_ref_snap is not None and int(inferred_ref_snap) != paired[-1]:
                    group_source += f"@ref{int(inferred_ref_snap):03d}"
            except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                # Candidate is simply not usable yet.  A later candidate/run may
                # still succeed, and the caller will report skipped directories.
                continue

        valid.append(
            RunInfo(cg, halo_id, d, paired[-1], group, audit, group_source or "unknown")
        )

    if not valid:
        return None

    # Prefer completed run with highest paired snapshot; audit-backed run breaks ties.
    valid.sort(key=lambda r: (r.snap, r.audit_json is not None, os.path.getmtime(r.output_dir)))
    return valid[-1]


def snapshot_header(snapshot_files: Sequence[str]) -> Dict[str, Any]:
    with h5py.File(snapshot_files[0], "r") as f:
        attrs = f["Header"].attrs
        params = f["Parameters"].attrs
        return {
            "a": float(attrs["Time"]),
            "z": float(attrs.get("Redshift", 1.0 / float(attrs["Time"]) - 1.0)),
            "h": float(params["HubbleParam"]),
            "box": float(attrs["BoxSize"]),
            "mass_table": np.asarray(attrs["MassTable"], dtype=float),
        }


def read_type(snapshot_files: Sequence[str], pt: int, extra_fields: Sequence[str] = ()):
    coords, masses = [], []
    extras: Dict[str, List[np.ndarray]] = {x: [] for x in extra_fields}
    available = set()
    mt = None

    for filename in snapshot_files:
        with h5py.File(filename, "r") as f:
            if mt is None:
                mt = np.asarray(f["Header"].attrs["MassTable"], dtype=float)
            name = f"PartType{pt}"
            if name not in f:
                continue
            g = f[name]
            available.update(g.keys())
            c = np.asarray(g["Coordinates"], dtype=float)
            coords.append(c)
            masses.append(
                np.asarray(g["Masses"], dtype=float)
                if "Masses" in g
                else np.full(len(c), mt[pt], dtype=float)
            )
            for field in extra_fields:
                if field in g:
                    extras[field].append(np.asarray(g[field]))

    if not coords:
        return None

    return {
        "coords": np.concatenate(coords),
        "masses": np.concatenate(masses),
        "fields": {k: np.concatenate(v) for k, v in extras.items() if v},
        "available": sorted(available),
    }


def radii_pkpc(coords, center, box, a, h):
    delta = periodic_delta(coords, center, box)
    r_ckpch = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return r_ckpch * a / h


def weighted_percentile(values, weights, percentile=50.0):
    q = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(q):
        return float("nan")
    v, w = values[q], weights[q]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(percentile / 100.0, cdf, v))


def enclosed_radius(r, mass, fraction):
    q = np.isfinite(r) & np.isfinite(mass) & (mass > 0)
    if not np.any(q):
        return float("nan")
    rr, mm = r[q], mass[q]
    order = np.argsort(rr)
    rr, mm = rr[order], mm[order]
    c = np.cumsum(mm)
    if c[-1] <= 0:
        return float("nan")
    return float(np.interp(fraction * c[-1], c, rr))


def shell_profile(r, mass_msun, edges):
    sums, _ = np.histogram(r, bins=edges, weights=mass_msun)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return sums / area


def shell_weighted_mean(r, values, weights, edges):
    num, _ = np.histogram(r, bins=edges, weights=weights * values)
    den, _ = np.histogram(r, bins=edges, weights=weights)
    out = np.full(len(edges) - 1, np.nan)
    q = den > 0
    out[q] = num[q] / den[q]
    return out


def safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out


def remy_ruyer_2014_dust_to_gas(oh12):
    """Rémy-Ruyer+14 Table 1 broken G/D fit, metallicity-dependent X_CO.

    x=12+log(O/H), x_solar=8.69; high-x log10(G/D)=2.21+(8.69-x),
    low-x log10(G/D)=0.96+3.10*(8.69-x), with transition x=8.10.
    This is an empirical fit to the DGS+KINGFISH+G11 sample, not
    additional independent galaxies or an estimate of its uncertainty.
    """
    x = np.asarray(oh12, dtype=float)
    log_gd = np.where(x > 8.10, 2.21 + 8.69 - x,
                      0.96 + 3.10 * (8.69 - x))
    return 10.0 ** (-log_gd)


def analyze_run(
    run: RunInfo,
    aperture_pkpc: Optional[float],
    central_aperture_r200: float,
    radial_max_pkpc: float,
    radial_bins: int,
    grain_radius_to_micron: float,
    grain_edges_micron: np.ndarray,
    source_map: Dict[int, str],
    xh_mode: str,
    dataset_label: str,
    r25_pkpc: Optional[float] = None,
    r25_annulus_fraction: float = 0.1,
):
    snapshot_files, _ = find_snapshot_and_group_files(run.output_dir, run.snap)
    hdr = snapshot_header(snapshot_files)
    halo = get_zoom_halo(
        run.output_dir,
        run.snap,
        group_index=run.group_index,
        refine_center=True,
        verbose=False,
    )

    a, h = halo.a, halo.h
    center, box = halo.chosen_center_ckpch, halo.boxsize_ckpch
    mass_to_msun = 1.0e10 / h
    r200_pkpc = halo.so_r200_ckpch * a / h
    m200_msun = halo.so_m200_code * mass_to_msun
    central_radius_pkpc = (
        float(aperture_pkpc)
        if aperture_pkpc is not None
        else float(central_aperture_r200) * r200_pkpc
    )

    gas = read_type(snapshot_files, 0, Z_FIELDS + OXYGEN_FIELDS + SFR_FIELDS)
    stars = read_type(snapshot_files, 4, Z_FIELDS)
    dust = read_type(
        snapshot_files, 6,
        DUST_RADIUS_FIELDS + DUST_CF_FIELDS + DUST_SOURCE_FIELDS
    )
    if gas is None or stars is None or dust is None:
        raise RuntimeError("gas, stars, and dust are all required")

    rg = radii_pkpc(gas["coords"], center, box, a, h)
    rs = radii_pkpc(stars["coords"], center, box, a, h)
    rd = radii_pkpc(dust["coords"], center, box, a, h)

    mg = gas["masses"] * mass_to_msun
    ms = stars["masses"] * mass_to_msun
    md = dust["masses"] * mass_to_msun

    qg = rg < central_radius_pkpc
    qs = rs < central_radius_pkpc
    qd = rd < central_radius_pkpc
    qg_halo, qs_halo, qd_halo = rg < r200_pkpc, rs < r200_pkpc, rd < r200_pkpc
    qg_cgm, qs_cgm, qd_cgm = qg_halo & ~qg, qs_halo & ~qs, qd_halo & ~qd

    zfield = first_present(gas["fields"], Z_FIELDS)
    oxygen_field = first_present(gas["fields"], OXYGEN_FIELDS)
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
        Zgas = float(np.sum(mg[qg] * zg[qg]) / Mgas) if Mgas > 0 else np.nan
    else:
        zg = np.full(len(mg), np.nan)
        MZgas = Zgas = np.nan

    if sfrfield:
        sfr_arr = np.asarray(gas["fields"][sfrfield], dtype=float).reshape(-1)
        SFR = float(np.sum(sfr_arr[qg]))
    else:
        SFR = np.nan

    # Integrated gas-phase oxygen abundance inside the galaxy aperture.
    # Average the linear number ratio first, then take the logarithm.
    oxygen_mass = np.nan
    oh12 = np.nan
    oh12_sfr = np.nan
    n_sfr_gas = 0
    oh12_r25 = np.nan
    n_r25_gas = 0
    if oxygen_field:
        xo = np.asarray(gas["fields"][oxygen_field], dtype=float).reshape(-1)
        xh = (
            0.76 * np.clip(1.0 - zg, 0.0, None)
            if xh_mode == "metal_corrected" and zfield
            else np.full(len(mg), 0.76)
        )
        good = qg & np.isfinite(xo) & np.isfinite(xh) & (xo >= 0) & (xh > 0)
        oxygen_mass = float(np.sum(mg[good] * xo[good]))
        hydrogen_mass = float(np.sum(mg[good] * xh[good]))
        if oxygen_mass > 0 and hydrogen_mass > 0:
            oh12 = 12.0 + np.log10((oxygen_mass / 16.0) / hydrogen_mass)
        if sfrfield:
            good_sfr = good & np.isfinite(sfr_arr) & (sfr_arr > 0)
            n_sfr_gas = int(np.count_nonzero(good_sfr))
            if np.any(good_sfr):
                linear_oh = np.average(
                    (xo[good_sfr] / 16.0) / xh[good_sfr],
                    weights=sfr_arr[good_sfr],
                )
                if linear_oh > 0:
                    oh12_sfr = 12.0 + np.log10(linear_oh)
        if r25_pkpc is not None:
            target = 0.4 * r25_pkpc
            # De Vis characteristic abundances are based on H II-region
            # emission lines near this radius; select star-forming gas only.
            # This is a spatial analogue, not synthetic strong-line spectroscopy.
            projected = np.sqrt(np.sum(periodic_delta(gas["coords"], center, box)[:, :2] ** 2, axis=1)) * a / h
            ring = good & (projected >= target * (1.0 - r25_annulus_fraction)) & (projected <= target * (1.0 + r25_annulus_fraction))
            if sfrfield:
                ring &= np.isfinite(sfr_arr) & (sfr_arr > 0)
            n_r25_gas = int(np.count_nonzero(ring))
            if n_r25_gas >= 5:
                w = sfr_arr[ring] if sfrfield else mg[ring]
                ratio = np.average((xo[ring] / 16.0) / xh[ring], weights=w)
                if ratio > 0:
                    oh12_r25 = 12.0 + np.log10(ratio)

    sSFR = SFR / Mstar if Mstar > 0 else np.nan
    DG = Mdust / Mgas if Mgas > 0 else np.nan
    DZ = Mdust / MZgas if MZgas > 0 else np.nan
    DZtotal = Mdust / (MZgas + Mdust) if MZgas + Mdust > 0 else np.nan
    dust_star = Mdust / Mstar if Mstar > 0 else np.nan

    # Extent measures inside the same galaxy aperture.
    r50d = enclosed_radius(rd[qd], md[qd], 0.50)
    r90d = enclosed_radius(rd[qd], md[qd], 0.90)
    r50s = enclosed_radius(rs[qs], ms[qs], 0.50)
    r90s = enclosed_radius(rs[qs], ms[qs], 0.90)
    r50g = enclosed_radius(rg[qg], mg[qg], 0.50)
    r90g = enclosed_radius(rg[qg], mg[qg], 0.90)

    # Radial profiles.
    rmax = min(radial_max_pkpc, r200_pkpc)
    edges = np.linspace(0.0, rmax, radial_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sig_g = shell_profile(rg, mg, edges)
    sig_s = shell_profile(rs, ms, edges)
    sig_d = shell_profile(rd, md, edges)
    sig_z = shell_profile(rg, mg * np.nan_to_num(zg, nan=0.0), edges) if zfield else np.full(radial_bins, np.nan)
    dg_r = safe_ratio(sig_d, sig_g)
    dz_r = safe_ratio(sig_d, sig_z)

    # Grain size.
    grain = None
    cf = (
        np.asarray(dust["fields"][cf_field], dtype=float).reshape(-1)
        if cf_field else np.full(len(md), np.nan)
    )
    grain_hist_mass = np.full(len(grain_edges_micron) - 1, np.nan)
    grain_hist_carbon_mass = np.full(len(grain_edges_micron) - 1, np.nan)
    grain_hist_silicate_mass = np.full(len(grain_edges_micron) - 1, np.nan)
    grain_hist_number = np.full(len(grain_edges_micron) - 1, np.nan)
    grain_p50 = np.nan
    grain_sigma_log10 = np.nan
    grain_particle_count = 0
    grain_unique_radius_count = 0
    grain_min_micron = np.nan
    grain_max_micron = np.nan
    grain_min_boundary_number_fraction = np.nan
    grain_max_boundary_number_fraction = np.nan
    grain_min_boundary_mass_fraction = np.nan
    grain_max_boundary_mass_fraction = np.nan
    f_small = np.nan
    f_small_carbon_total = np.nan
    f_small_within_carbon = np.nan
    if radius_field:
        grain = np.asarray(dust["fields"][radius_field], dtype=float).reshape(-1) * grain_radius_to_micron
        good = qd & np.isfinite(grain) & (grain > 0)
        if np.any(good):
            grain_values = grain[good]
            grain_masses = md[good]
            grain_particle_count = int(np.count_nonzero(good))
            grain_unique_radius_count = int(np.unique(grain_values).size)
            grain_min_micron = float(np.min(grain_values))
            grain_max_micron = float(np.max(grain_values))
            # Exact/repeated extrema diagnose grains accumulating at numerical
            # radius limits without smoothing those populations away.
            at_min = np.isclose(grain_values, grain_min_micron, rtol=1e-6, atol=0.0)
            at_max = np.isclose(grain_values, grain_max_micron, rtol=1e-6, atol=0.0)
            grain_min_boundary_number_fraction = float(np.mean(at_min))
            grain_max_boundary_number_fraction = float(np.mean(at_max))
            total_grain_mass = float(np.sum(grain_masses))
            if total_grain_mass > 0:
                grain_min_boundary_mass_fraction = float(np.sum(grain_masses[at_min]) / total_grain_mass)
                grain_max_boundary_mass_fraction = float(np.sum(grain_masses[at_max]) / total_grain_mass)
            grain_hist_mass, _ = np.histogram(grain[good], bins=grain_edges_micron, weights=md[good])
            good_cf = good & np.isfinite(cf) & (cf >= 0) & (cf <= 1)
            if np.any(good_cf):
                grain_hist_carbon_mass, _ = np.histogram(
                    grain[good_cf], bins=grain_edges_micron,
                    weights=md[good_cf] * cf[good_cf],
                )
                grain_hist_silicate_mass, _ = np.histogram(
                    grain[good_cf], bins=grain_edges_micron,
                    weights=md[good_cf] * (1.0 - cf[good_cf]),
                )
            grain_hist_number, _ = np.histogram(grain[good], bins=grain_edges_micron)
            dlog = np.diff(np.log10(grain_edges_micron))
            grain_hist_mass = grain_hist_mass / dlog
            grain_hist_carbon_mass = grain_hist_carbon_mass / dlog
            grain_hist_silicate_mass = grain_hist_silicate_mass / dlog
            grain_hist_number = grain_hist_number / dlog
            grain_p50 = weighted_percentile(grain[good], md[good], 50.0)
            f_small = float(md[good & (grain < 0.03)].sum() / md[good].sum()) if md[good].sum() > 0 else np.nan
            loga = np.log10(grain[good])
            mean_loga = np.average(loga, weights=md[good])
            grain_sigma_log10 = float(np.sqrt(np.average(
                (loga - mean_loga) ** 2, weights=md[good]
            )))
            if np.any(good_cf):
                total_dust = float(np.sum(md[good_cf]))
                carbon_dust = float(np.sum(md[good_cf] * cf[good_cf]))
                small_cf = good_cf & (grain < 0.03)
                small_carbon = float(np.sum(md[small_cf] * cf[small_cf]))
                f_small_carbon_total = small_carbon / total_dust if total_dust > 0 else np.nan
                f_small_within_carbon = small_carbon / carbon_dust if carbon_dust > 0 else np.nan

    # Carbon/silicate composition.
    carbon_fraction = np.nan
    dust_oxygen_mass = np.nan
    carbon_r = np.full(radial_bins, np.nan)
    if cf_field:
        good = qd & np.isfinite(cf)
        if np.any(good) and md[good].sum() > 0:
            carbon_fraction = float(np.sum(md[good] * cf[good]) / np.sum(md[good]))
            dust_oxygen_mass = float(np.sum(
                md[good] * (1.0 - cf[good]) * SILICATE_OXYGEN_MASS_FRACTION
            ))
        good_all = np.isfinite(cf)
        carbon_r = shell_weighted_mean(rd[good_all], cf[good_all], md[good_all], edges)

    # Source fractions.
    source_fractions = {label: np.nan for label in source_map.values()}
    if source_field:
        src = np.asarray(dust["fields"][source_field]).reshape(-1)
        total = md[qd].sum()
        if total > 0:
            for code, label in source_map.items():
                source_fractions[label] = float(md[qd & (src == code)].sum() / total)

    summary = {
        "CG": run.cg,
        "halo_id": run.halo_id,
        "output_dir": run.output_dir,
        "snapshot": run.snap,
        "group_index": run.group_index,
        "z": float(halo.z),
        "parent_M200c_Msun": PARENT_M200.get(run.cg, np.nan),
        "zoom_M200c_Msun": m200_msun,
        "R200c_pkpc": r200_pkpc,
        "aperture_pkpc": central_radius_pkpc,
        "central_aperture_r200": central_radius_pkpc / r200_pkpc,
        "dataset_status": dataset_label,
        "xh_mode": xh_mode,
        "Nstar_aperture": int(np.count_nonzero(qs)),
        "Ngas_aperture": int(np.count_nonzero(qg)),
        "Ndust_aperture": int(np.count_nonzero(qd)),
        "Ndust_R200c": int(np.count_nonzero(qd_halo)),
        "Mdust_R200c_Msun": float(md[qd_halo].sum()),
        "Mdust_outside_aperture_inside_R200c_Msun": float(md[qd_cgm].sum()),
        "Mstar_Msun": Mstar,
        "Mgas_Msun": Mgas,
        "Mdust_Msun": Mdust,
        "central_Mstar_Msun": Mstar,
        "central_Mgas_Msun": Mgas,
        "central_Mdust_Msun": Mdust,
        "cgm_Mstar_Msun": float(ms[qs_cgm].sum()),
        "cgm_Mgas_Msun": float(mg[qg_cgm].sum()),
        "cgm_Mdust_Msun": float(md[qd_cgm].sum()),
        "halo_Mstar_Msun": float(ms[qs_halo].sum()),
        "halo_Mgas_Msun": float(mg[qg_halo].sum()),
        "halo_Mdust_Msun": float(md[qd_halo].sum()),
        "MZgas_Msun": MZgas,
        "SFR_Msun_yr": SFR,
        "sSFR_yr_inv": sSFR,
        "Zgas_mass_fraction": Zgas,
        "gas_oxygen_mass_Msun": oxygen_mass,
        "dust_oxygen_mass_Msun": dust_oxygen_mass,
        "gas_12_plus_logOH": oh12,
        "sf_gas_sfr_weighted_12_plus_logOH": oh12_sfr,
        "Nstarforming_gas_aperture": n_sfr_gas,
        "characteristic_0p4R25_12_plus_logOH": oh12_r25,
        "R25_pkpc": r25_pkpc if r25_pkpc is not None else np.nan,
        "Nstarforming_gas_0p4R25": n_r25_gas,
        "dust_to_stellar": dust_star,
        "dust_to_gas": DG,
        "dust_to_gas_phase_metals": DZ,
        "dust_fraction_total_metals": DZtotal,
        # Backward-compatible alias used by the radial/profile figures.
        "dust_to_metals": DZ,
        "R50_dust_pkpc": r50d,
        "R90_dust_pkpc": r90d,
        "R50_star_pkpc": r50s,
        "R90_star_pkpc": r90s,
        "R50_gas_pkpc": r50g,
        "R90_gas_pkpc": r90g,
        "R50dust_over_R50star": r50d / r50s if r50s > 0 else np.nan,
        "R50dust_over_R50gas": r50d / r50g if r50g > 0 else np.nan,
        "grain_mass_median_micron": grain_p50,
        "grain_sigma_log10_radius": grain_sigma_log10,
        "grain_particle_count": grain_particle_count,
        "grain_unique_radius_count": grain_unique_radius_count,
        "grain_min_micron": grain_min_micron,
        "grain_max_micron": grain_max_micron,
        "grain_min_boundary_number_fraction": grain_min_boundary_number_fraction,
        "grain_max_boundary_number_fraction": grain_max_boundary_number_fraction,
        "grain_min_boundary_mass_fraction": grain_min_boundary_mass_fraction,
        "grain_max_boundary_mass_fraction": grain_max_boundary_mass_fraction,
        "small_grain_mass_fraction_lt_0p03um": f_small,
        "small_carbon_grain_fraction_total_dust": f_small_carbon_total,
        "small_grain_fraction_within_carbon_dust": f_small_within_carbon,
        "carbon_mass_fraction": carbon_fraction,
        **{f"source_{k}_mass_fraction": v for k, v in source_fractions.items()},
    }

    profile = {
        "r_pkpc": centers,
        "r_over_r200": centers / r200_pkpc,
        "Sigma_gas_Msun_pkpc2": sig_g,
        "Sigma_star_Msun_pkpc2": sig_s,
        "Sigma_dust_Msun_pkpc2": sig_d,
        "Sigma_gas_metals_Msun_pkpc2": sig_z,
        "dust_to_gas": dg_r,
        "dust_to_metals": dz_r,
        "carbon_mass_fraction": carbon_r,
        "grain_edges_micron": grain_edges_micron,
        "grain_centers_micron": np.sqrt(grain_edges_micron[:-1] * grain_edges_micron[1:]),
        "grain_dM_dloga": grain_hist_mass,
        "grain_carbon_dM_dloga": grain_hist_carbon_mass,
        "grain_silicate_dM_dloga": grain_hist_silicate_mass,
        "grain_dN_dloga": grain_hist_number,
    }
    return summary, profile


def setup_axes(ax, xlabel, ylabel, logx=False, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def label_points(ax, x, y, labels):
    for xx, yy, lab in zip(x, y, labels):
        if np.isfinite(xx) and np.isfinite(yy):
            ax.annotate(lab, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)


def savefig(fig, path, dataset_label):
    if dataset_label:
        fig.text(
            0.5, 0.985, dataset_label,
            ha="center", va="top", fontsize=9, color="crimson",
            fontweight="bold",
        )
    if not getattr(fig, "_cosmicgrain_manual_layout", False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    root, _ = os.path.splitext(path)
    png_path = root + ".png"
    pdf_path = root + ".pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def plot_all(summaries, profiles, outdir, dataset_label):
    labels = [s["CG"] for s in summaries]
    M200 = np.array([s["zoom_M200c_Msun"] for s in summaries])
    Mstar = np.array([s["Mstar_Msun"] for s in summaries])
    Mgas = np.array([s["Mgas_Msun"] for s in summaries])
    Mdust = np.array([s["Mdust_Msun"] for s in summaries])

    # Fig 1: global masses.
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(M200, Mstar, "o-", label=r"$M_\star$")
    ax.plot(M200, Mgas, "s-", label=r"$M_{\rm gas}$")
    ax.plot(M200, Mdust, "^-", label=r"$M_{\rm dust}$")
    setup_axes(ax, r"$M_{200c}\ [{\rm M_\odot}]$", r"Mass within galaxy aperture $[{\rm M_\odot}]$", True, True)
    ax.legend()
    for x, y, lab in zip(M200, Mstar, labels):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    savefig(fig, os.path.join(outdir, "fig01_global_masses.png"), dataset_label)

    # Fig 2: dust scaling relations.
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.5))
    quantities = [
        ("Mdust_Msun", r"$M_{\rm dust}\ [{\rm M_\odot}]$", r"$M_\star\ [{\rm M_\odot}]$", "Mstar", True, True),
        ("dust_to_stellar", r"$M_{\rm dust}/M_\star$", r"$M_\star\ [{\rm M_\odot}]$", "Mstar", True, True),
        ("dust_to_gas", r"$D/G$", r"$12+\log_{10}({\rm O/H})$", "OH", False, True),
        ("dust_fraction_total_metals", r"$M_{\rm dust}/(M_{Z,\rm gas}+M_{\rm dust})$", r"$12+\log_{10}({\rm O/H})$", "OH", False, False),
    ]
    oxygen = np.array([s["gas_12_plus_logOH"] for s in summaries])
    for ax, (key, ylabel, xlabel, xkind, logx, logy) in zip(axs.flat, quantities):
        y = np.array([s[key] for s in summaries])
        xvals = Mstar if xkind == "Mstar" else oxygen
        ax.scatter(xvals, y, s=55)
        label_points(ax, xvals, y, labels)
        setup_axes(ax, xlabel, ylabel, logx, logy)
    savefig(fig, os.path.join(outdir, "fig02_dust_scaling_relations.png"), dataset_label)

    # Fig 3: radial surface densities.
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for s, p in zip(summaries, profiles):
        r = p["r_pkpc"]
        axs[0].plot(r, p["Sigma_dust_Msun_pkpc2"], label=s["CG"])
        axs[1].plot(r, p["Sigma_gas_Msun_pkpc2"], label=s["CG"])
        axs[2].plot(r, p["Sigma_star_Msun_pkpc2"], label=s["CG"])
    for ax, title, ylabel in zip(
        axs,
        ["Dust", "Gas", "Stars"],
        [r"$\Sigma_{\rm dust}$", r"$\Sigma_{\rm gas}$", r"$\Sigma_\star$"],
    ):
        ax.set_title(title)
        setup_axes(ax, "Radius [pkpc]", ylabel + r" $[{\rm M_\odot\,pkpc^{-2}}]$", False, True)
    axs[0].legend()
    savefig(fig, os.path.join(outdir, "fig03_radial_surface_density.png"), dataset_label)

    # Fig 4: radial D/G and D/Z.
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for s, p in zip(summaries, profiles):
        axs[0].plot(p["r_pkpc"], p["dust_to_gas"], label=s["CG"])
        axs[1].plot(p["r_pkpc"], p["dust_to_metals"], label=s["CG"])
    setup_axes(axs[0], "Radius [pkpc]", r"$D/G(R)$", False, True)
    setup_axes(axs[1], "Radius [pkpc]", r"$D/Z(R)$", False, True)
    axs[0].legend()
    savefig(fig, os.path.join(outdir, "fig04_radial_dust_ratios.png"), dataset_label)

    # Fig 5: dust extent.
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    y1 = np.array([s["R50dust_over_R50star"] for s in summaries])
    y2 = np.array([s["R50dust_over_R50gas"] for s in summaries])
    for ax, y, ylabel in [
        (axs[0], y1, r"$R_{50,\rm dust}/R_{50,\star}$"),
        (axs[1], y2, r"$R_{50,\rm dust}/R_{50,\rm gas}$"),
    ]:
        ax.scatter(Mstar, y, s=55)
        label_points(ax, Mstar, y, labels)
        setup_axes(ax, r"$M_\star\ [{\rm M_\odot}]$", ylabel, True, False)
        ax.axhline(1.0, lw=1, ls="--")
    savefig(fig, os.path.join(outdir, "fig05_dust_extent.png"), dataset_label)

    # Fig 6: grain size distributions.
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for s, p in zip(summaries, profiles):
        y = p["grain_dM_dloga"]
        if np.any(np.isfinite(y)):
            ax.plot(p["grain_centers_micron"], y, label=s["CG"])
    setup_axes(ax, r"Grain radius $a\ [\mu{\rm m}]$", r"$dM_{\rm dust}/d\log_{10}a\ [{\rm M_\odot}]$", True, True)
    ax.legend()
    savefig(fig, os.path.join(outdir, "fig06_grain_size_distributions.png"), dataset_label)

    # Fig 7: grain size summary.
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    med = np.array([s["grain_mass_median_micron"] for s in summaries])
    small = np.array([s["small_grain_mass_fraction_lt_0p03um"] for s in summaries])
    axs[0].scatter(Mstar, med, s=55)
    label_points(axs[0], Mstar, med, labels)
    setup_axes(axs[0], r"$M_\star\ [{\rm M_\odot}]$", r"Mass-weighted median $a\ [\mu{\rm m}]$", True, True)
    axs[1].scatter(Mstar, small, s=55)
    label_points(axs[1], Mstar, small, labels)
    setup_axes(axs[1], r"$M_\star\ [{\rm M_\odot}]$", r"$f(a<0.03\,\mu{\rm m})$", True, False)
    savefig(fig, os.path.join(outdir, "fig07_grain_size_summary.png"), dataset_label)

    # Fig 8: carbon + source fractions.
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
    carbon = np.array([s["carbon_mass_fraction"] for s in summaries])
    axs[0].scatter(Mstar, carbon, s=55)
    label_points(axs[0], Mstar, carbon, labels)
    setup_axes(axs[0], r"$M_\star\ [{\rm M_\odot}]$", "Carbon dust mass fraction", True, False)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for src in ("SNII", "AGB", "LRN"):
        vals = np.array([finite_or_nan(s.get(f"source_{src}_mass_fraction")) for s in summaries])
        vals_plot = np.nan_to_num(vals, nan=0.0)
        axs[1].bar(x, vals_plot, bottom=bottom, label=src)
        bottom += vals_plot
    axs[1].set_xticks(x, labels)
    axs[1].set_ylabel("Dust mass fraction")
    axs[1].set_ylim(0, 1)
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].legend()
    savefig(fig, os.path.join(outdir, "fig08_composition_sources.png"), dataset_label)

    # Fig 9: radial carbon fraction.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for s, p in zip(summaries, profiles):
        if np.any(np.isfinite(p["carbon_mass_fraction"])):
            ax.plot(p["r_pkpc"], p["carbon_mass_fraction"], label=s["CG"])
    setup_axes(ax, "Radius [pkpc]", "Carbon dust mass fraction", False, False)
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, os.path.join(outdir, "fig09_composition_radial.png"), dataset_label)

    # Fig 10: lifecycle plane.
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    xsmall = np.array([s["small_grain_mass_fraction_lt_0p03um"] for s in summaries])
    ydz = np.array([s["dust_to_metals"] for s in summaries])
    sizes = 55 + 90 * (np.log10(Mstar) - np.nanmin(np.log10(Mstar))) / max(
        np.nanmax(np.log10(Mstar)) - np.nanmin(np.log10(Mstar)), 1e-12
    )
    ax.scatter(xsmall, ydz, s=sizes)
    label_points(ax, xsmall, ydz, labels)
    setup_axes(ax, r"Small-grain mass fraction $f(a<0.03\,\mu{\rm m})$", r"$D/Z$", False, True)
    savefig(fig, os.path.join(outdir, "fig10_dust_lifecycle_plane.png"), dataset_label)


def plot_paper_figure_set(summaries, profiles, outdir, dataset_label, earlier_epochs=None,
                          devis_galaxies=None,
                          grain_representatives=("CG01", "CG10", "CG12")):
    """Create the seven-panel-sequence draft discussed for the paper.

    These are real measurements from the selected snapshots.  Figure 1 shows
    endpoints until the multi-epoch tracker is connected; Figure 7 is only a
    labeled layout placeholder and contains no synthetic SKIRT measurements.
    """
    labels = [s["CG"] for s in summaries]
    m200 = np.array([s["zoom_M200c_Msun"] for s in summaries])
    mstar = np.array([s["Mstar_Msun"] for s in summaries])
    mgas = np.array([s["Mgas_Msun"] for s in summaries])
    mdust = np.array([s["Mdust_Msun"] for s in summaries])
    ssfr = np.array([s["sSFR_yr_inv"] for s in summaries])
    gas_fraction = safe_ratio(mgas, mgas + mstar)
    dstar = np.array([s["dust_to_stellar"] for s in summaries])
    dg = np.array([s["dust_to_gas"] for s in summaries])
    dz = np.array([s["dust_fraction_total_metals"] for s in summaries])
    oxygen_sfr = np.array([
        finite_or_nan(s.get("sf_gas_sfr_weighted_12_plus_logOH"))
        for s in summaries
    ])
    oxygen_all_gas = np.array([
        finite_or_nan(s.get("gas_12_plus_logOH")) for s in summaries
    ])
    n_sfr_gas = np.array([
        int(s.get("Nstarforming_gas_aperture", 0)) for s in summaries
    ])
    # Match Figure 1: use SFR-weighted O/H when it is adequately sampled;
    # otherwise retain the halo with an explicitly open all-gas fallback.
    well_sampled_sfr = np.isfinite(oxygen_sfr) & (n_sfr_gas >= 20)
    oxygen = np.where(well_sampled_sfr, oxygen_sfr, oxygen_all_gas)
    small = np.array([s["small_grain_mass_fraction_lt_0p03um"] for s in summaries])
    small_carbon = np.array([s["small_carbon_grain_fraction_total_dust"] for s in summaries])
    median_a = np.array([s["grain_mass_median_micron"] for s in summaries])
    width_a = np.array([s["grain_sigma_log10_radius"] for s in summaries])
    carbon = np.array([s["carbon_mass_fraction"] for s in summaries])

    order = np.argsort(m200)
    color_value = np.log10(m200)
    norm = plt.Normalize(np.nanmin(color_value), np.nanmax(color_value))
    cmap = plt.get_cmap("viridis")

    def point(ax, x, y, index, **kwargs):
        marker = "*" if labels[index] == "CG10" else "o"
        size = 145 if labels[index] == "CG10" else 62
        ax.scatter(x, y, c=[cmap(norm(color_value[index]))], marker=marker,
                   s=size, edgecolor="black", linewidth=0.45, zorder=3, **kwargs)

    def annotate_all(ax, x, y):
        label_points(ax, x, y, labels)
        ax.margins(x=0.09, y=0.14)

    # Paper Fig. 1: z=0 endpoints.  Tracks are intentionally not fabricated.
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for i in order:
        point(ax, m200[i], mstar[i], i)
    annotate_all(ax, m200, mstar)
    setup_axes(ax, r"$M_{200c}\ [{\rm M_\odot}]$", r"$M_\star\ [{\rm M_\odot}]$", True, True)
    ax.set_title("Suite coverage: current endpoints (evolutionary tracks pending)")
    savefig(fig, os.path.join(outdir, "paper_fig01_suite_coverage.png"), dataset_label)

    # Paper Fig. 2: a single D/G comparison. The published Rémy-Ruyer+14
    # broken-law curve is distinct from the De Vis+19 DustPedia dots.
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    if devis_galaxies is not None:
        ax.scatter(devis_galaxies[:, 0], devis_galaxies[:, 1], s=24,
                   color='0.55', alpha=.42, lw=0, zorder=1,
                   label='DustPedia (De Vis et al. 2019)')
    rr_oh = np.linspace(7.1, 9.1, 300)  # approximate range of the observed sample
    ax.plot(rr_oh, remy_ruyer_2014_dust_to_gas(rr_oh), '--', color='#bf482e',
            lw=2.7, zorder=2,
            label=r'Rémy-Ruyer et al. (2014) fit ($X_{\rm CO,Z}$)')
    # Hand-tuned offsets keep the compact CG03--CG06 group legible without
    # moving the data markers themselves. Values are in display points.
    figure2_label_offsets = {
        "CG01": (8, 8, "left"),
        "CG02": (8, 8, "left"),
        "CG03": (-8, 22, "center"),
        "CG04": (14, 7, "left"),
        "CG05": (-15, -26, "center"),
        "CG06": (8, 11, "left"),
        "CG07": (8, 9, "left"),
        "CG08": (8, 8, "left"),
        "CG09": (8, 8, "left"),
        "CG10": (9, 8, "left"),
        "CG11": (8, 8, "left"),
        "CG12": (9, 10, "left"),
    }
    for i in range(len(labels)):
        marker = "*" if labels[i] == "CG10" else "o"
        size = 230 if labels[i] == "CG10" else 105
        if well_sampled_sfr[i]:
            ax.scatter(oxygen[i], dg[i], c=[cmap(norm(color_value[i]))],
                       marker=marker, s=size, edgecolor="black", linewidth=0.8,
                       zorder=4)
        else:
            ax.scatter(oxygen[i], dg[i], facecolor="white",
                       edgecolor=cmap(norm(color_value[i])), marker=marker,
                       s=size, linewidth=2.2, zorder=4)
        if np.isfinite(oxygen[i]) and np.isfinite(dg[i]):
            dx, dy, horizontal_alignment = figure2_label_offsets.get(
                labels[i], (7, 6, "left")
            )
            moved = labels[i] in {"CG03", "CG05"}
            ax.annotate(labels[i], (oxygen[i], dg[i]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=12,
                        fontweight="bold", ha=horizontal_alignment, zorder=5,
                        arrowprops=(dict(arrowstyle="-", color="0.35", lw=0.8,
                                         shrinkA=2, shrinkB=5)
                                    if moved else None))
    ax.margins(x=0.09, y=0.14)
    setup_axes(ax, r"Gas-phase $12+\log_{10}({\rm O/H})$",
               r"$M_{\rm dust}/M_{\rm gas}$", False, True)
    ax.set_xlabel(r"Gas-phase $12+\log_{10}({\rm O/H})$", fontsize=17)
    ax.set_ylabel(r"$M_{\rm dust}/M_{\rm gas}$", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=14, length=7, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=12, length=4)
    ax.set_xlim(7.3, max(9.2, np.nanmax(oxygen) + 0.12))
    from matplotlib.lines import Line2D
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='0.25',
               markeredgecolor='black', markersize=9,
               label='CosmicGrain: SFR-weighted O/H'),
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='white',
               markeredgecolor='0.3', markeredgewidth=1.8, markersize=9,
               label='CosmicGrain: all-gas O/H fallback'),
    ])
    ax.legend(handles=handles, loc='upper left', fontsize=12.5,
              frameon=False, handlelength=2.3)
    fig.subplots_adjust(left=.12, right=.84, bottom=.14, top=.96)
    # Match the colorbar height exactly to the manually positioned plot axes.
    cax = fig.add_axes([.87, .14, .022, .82])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r'$\log_{10}(M_{200c}/M_\odot)$', fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    fig._cosmicgrain_manual_layout = True
    savefig(fig, os.path.join(outdir, "paper_fig02_dust_enrichment.png"), dataset_label)

    # Paper Fig. 3: representative GSD evolution above, suite-wide grain
    # transformation below. No interpolation between snapshots is performed.
    # Use a scientifically motivated sequence rather than whichever objects
    # happen to occupy the minimum, median, and maximum halo-mass ranks.  CG10
    # is the Milky Way analogue used by the later R(V) figures, so introducing
    # it here makes the narrative continuous from grain populations to
    # extinction.  Fall back to mass-ranked systems only when a requested
    # object is genuinely unavailable.
    requested = [str(cg).strip().upper() for cg in grain_representatives]
    if len(requested) != 3 or len(set(requested)) != 3:
        raise ValueError("grain_representatives must contain three distinct CG labels")
    label_to_index = {cg: i for i, cg in enumerate(labels)}
    if all(cg in label_to_index for cg in requested):
        representative = [label_to_index[cg] for cg in requested]
    else:
        missing = [cg for cg in requested if cg not in label_to_index]
        sorted_idx = np.argsort(m200)
        representative = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]
        print("Figure 3: requested representatives unavailable "
              f"({', '.join(missing)}); using mass-ranked fallback "
              f"{[labels[i] for i in representative]}")
    have_any_earlier = bool(earlier_epochs)
    representative_have_earlier = any(
        labels[i] in earlier_epochs for i in representative
    ) if earlier_epochs else False
    mass_class = ["Low mass", "Milky Way analogue", "High mass"]
    carbon_color = "#c45a2d"
    silicate_color = "#148f8b"
    fig, axs = plt.subplots(2, 3, figsize=(15.4, 9.0))

    def normalized_gsd(p):
        radius_nm = 1.0e3 * np.asarray(p["grain_centers_micron"], float)
        dlog = np.diff(np.log10(np.asarray(p["grain_edges_micron"], float)))
        total_curve = np.asarray(p["grain_dM_dloga"], float)
        # Convert the density per dex back to an actual mass fraction in each
        # (uniform logarithmic) plotting bin. The returned total sums to one.
        total_bin_mass = total_curve * dlog
        carbon_bin_mass = np.asarray(p["grain_carbon_dM_dloga"], float) * dlog
        silicate_bin_mass = np.asarray(p["grain_silicate_dM_dloga"], float) * dlog
        total_mass = np.nansum(total_bin_mass)
        if total_mass <= 0:
            empty = np.full_like(total_curve, np.nan)
            return radius_nm, empty, empty, empty
        return (radius_nm, total_bin_mass / total_mass,
                carbon_bin_mass / total_mass, silicate_bin_mass / total_mass)

    def draw_gsd_evolution(ax, now_summary, now_profile, earlier=None):
        radius_nm, total, carbon_curve, silicate_curve = normalized_gsd(now_profile)
        floor = 1.0e-5
        # Present-day composition is a stacked mass distribution: the two
        # colored areas sum to the black total outline.
        ax.fill_between(radius_nm, floor, np.maximum(silicate_curve, floor),
                        color=silicate_color, alpha=.58, linewidth=0)
        ax.fill_between(radius_nm, np.maximum(silicate_curve, floor),
                        np.maximum(total, floor), color=carbon_color,
                        alpha=.58, linewidth=0)
        ax.plot(radius_nm, total, color="black", lw=2.5, zorder=4)
        ax.plot(radius_nm, silicate_curve, color=silicate_color, lw=1.4, zorder=4)

        if earlier is not None:
            _, earlier_profile = earlier
            eradius, etotal, ecarbon, esilicate = normalized_gsd(earlier_profile)
            ax.plot(eradius, etotal, color="black", lw=2.0, ls="--", zorder=5)
            ax.plot(eradius, ecarbon, color=carbon_color, lw=1.8, ls="--", zorder=5)
            ax.plot(eradius, esilicate, color=silicate_color, lw=1.8, ls="--", zorder=5)

        # Approximate stellar injection scales used by the model.
        ax.axvspan(7.5, 13.5, color=silicate_color, alpha=.09, zorder=0)
        ax.axvspan(80.0, 120.0, color=carbon_color, alpha=.09, zorder=0)
        setup_axes(ax, r"Grain radius $a$ (nm)",
                   "Dust mass fraction per size bin", True, True)
        ax.set_xlim(1.0, 200.0)
        ax.set_ylim(bottom=floor)
        ax.tick_params(labelsize=12)

    for col, i in enumerate(representative):
        cg = labels[i]
        draw_gsd_evolution(axs[0, col], summaries[i], profiles[i],
                           earlier_epochs.get(cg) if earlier_epochs else None)
        exponent = int(np.floor(np.log10(m200[i])))
        mantissa = m200[i] / 10.0 ** exponent
        axs[0, col].set_title(
            rf"{mass_class[col]}: {cg}" + "\n" +
            rf"$M_{{200c}}={mantissa:.1f}\times10^{{{exponent}}}\,M_\odot$",
            fontsize=14.5,
        )
        if col:
            axs[0, col].set_ylabel("")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    gsd_handles = [
        Patch(facecolor=silicate_color, alpha=.58, label=r"Silicate, $z=0$"),
        Patch(facecolor=carbon_color, alpha=.58, label=r"Carbonaceous, $z=0$"),
        Line2D([], [], color="black", lw=2.5, label=r"Total, $z=0$"),
    ]
    if representative_have_earlier:
        gsd_handles.append(
            Line2D([], [], color="black", lw=2.0, ls="--",
                   label=r"Earlier epoch ($z\sim2$)")
        )
    axs[0, 0].legend(handles=gsd_handles, frameon=False, fontsize=11,
                     loc="upper left")
    axs[0, 0].text(10.2, .965, "SNII", color=silicate_color,
                   fontsize=9, ha="center", va="top",
                   transform=axs[0, 0].get_xaxis_transform())
    axs[0, 0].text(98.0, .965, "AGB", color=carbon_color,
                   fontsize=9, ha="center", va="top",
                   transform=axs[0, 0].get_xaxis_transform())

    # Suite-wide evolution panels. Halo mass is already encoded on the x axis,
    # so color is reserved for the present/earlier epoch distinction.
    present_color = "#2878a6"
    earlier_color = "#2878a6"
    summary_metrics = [
        ("small_grain_mass_fraction_lt_0p03um",
         r"Small-grain mass fraction ($a<30$ nm)", 1.0),
        ("grain_mass_median_micron", r"Mass-weighted median radius (nm)", 1.0e3),
        ("carbon_mass_fraction", "Carbonaceous dust mass fraction", 1.0),
    ]
    bottom_label_offsets = {
        (1, "CG11"): (-3, -16),
        (1, "CG12"): (7, 8),
        (2, "CG11"): (7, -16),
        (2, "CG12"): (7, 8),
        (2, "CG10"): (7, 8),
    }
    for col, (key, ylabel, scale) in enumerate(summary_metrics):
        ax = axs[1, col]
        for i, s in enumerate(summaries):
            x_now = s["zoom_M200c_Msun"]
            y_now = scale * finite_or_nan(s.get(key))
            cg = s["CG"]
            if cg in earlier_epochs:
                es, _ = earlier_epochs[cg]
                x_old = finite_or_nan(es.get("zoom_M200c_Msun"))
                y_old = scale * finite_or_nan(es.get(key))
                if np.all(np.isfinite([x_old, y_old, x_now, y_now])):
                    ax.plot([x_old, x_now], [y_old, y_now], color="0.68",
                            lw=1.15, zorder=1)
                    ax.scatter(x_old, y_old, s=46, facecolor="white",
                               edgecolor=earlier_color, linewidth=1.8,
                               zorder=2)
            marker = "*" if cg == "CG10" else "o"
            size = 155 if cg == "CG10" else 70
            ax.scatter(x_now, y_now, s=size, marker=marker,
                       color=present_color, edgecolor="black",
                       linewidth=.55, zorder=3)
            # Keep the ensemble readable: label the same three anchor systems
            # in every panel.  The underlying CSV retains every value.
            label_this = cg in {labels[j] for j in representative}
            if label_this:
                offset = bottom_label_offsets.get((col, cg), (5, 5))
                ax.annotate(cg, (x_now, y_now), xytext=offset,
                            textcoords="offset points", fontsize=9,
                            fontweight="bold")
        setup_axes(ax, r"$M_{200c}$ ($M_\odot$)", ylabel, True, False)
        ax.tick_params(labelsize=12)
        ax.set_title(("(d)" if col == 0 else "(e)" if col == 1 else "(f)"),
                     loc="left", fontsize=12)
        if key.endswith("fraction_lt_0p03um") or key == "carbon_mass_fraction":
            ax.set_ylim(-.03, 1.03)

    axs[0, 0].set_title("(a) " + axs[0, 0].get_title(), loc="center")
    axs[0, 1].set_title("(b) " + axs[0, 1].get_title(), loc="center")
    axs[0, 2].set_title("(c) " + axs[0, 2].get_title(), loc="center")
    if have_any_earlier:
        epoch_handle = [
            Line2D([], [], marker="o", ls="none", color="black",
                   markerfacecolor=present_color, label=r"$z=0$"),
            Line2D([], [], marker="o", ls="none", color=earlier_color,
                   markerfacecolor="white", label=r"earlier epoch ($z\sim2$)"),
        ]
        axs[1, 0].legend(handles=epoch_handle, frameon=False, fontsize=11,
                         loc="best")
    fig.subplots_adjust(hspace=.36, wspace=.27, left=.075, right=.985,
                        bottom=.08, top=.93)
    fig._cosmicgrain_manual_layout = True
    savefig(fig, os.path.join(outdir, "paper_fig03_grain_populations.png"), dataset_label)

    # Paper Figs. 4-6 are produced by the dedicated R(V) and MW comparison
    # scripts with matched optical tables, map resolution, and sightline cuts.
    print('Paper Figs. 4-6: run the dedicated R(V) and MW scripts; no placeholders generated.')


def jsonable(x):
    """Recursively convert NumPy objects and non-finite floats to valid JSON."""
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


def latex_number(value, decimals=3):
    """Format a scalar for a LaTeX table, using an ellipsis when unavailable."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return r"\ldots"
    if not math.isfinite(value):
        return r"\ldots"
    return f"{value:.{decimals}f}"


def latex_scientific(value, significant_figures=3):
    """Format a scalar as, e.g., ``$1.23\\times10^{8}$``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return r"\ldots"
    if not math.isfinite(value):
        return r"\ldots"
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    decimals = max(0, significant_figures - 1)
    return rf"${mantissa:.{decimals}f}\times10^{{{exponent}}}$"


def latex_log10(value, decimals=3):
    """Format log10(value), using an ellipsis for missing or non-positive data."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return r"\ldots"
    if not math.isfinite(value) or value <= 0:
        return r"\ldots"
    return f"{math.log10(value):.{decimals}f}"


def write_latex_halo_table(summaries, path, aperture_pkpc, aperture_r200):
    """Write the manuscript halo-sample table, including not-yet-finished runs."""
    by_cg = {row["CG"]: row for row in summaries}
    if aperture_pkpc is None:
        aperture_text = rf"$r<{aperture_r200:g}R_{{200{{\rm c}}}}$"
    else:
        aperture_text = rf"$r<{aperture_pkpc:g}\,{{\rm pkpc}}$"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (r"\caption{Properties of the CosmicGrain halo sample at the $z=0$ "
         r"snapshot.  Halo masses and radii are measured from the "
         r"zoom simulations; stellar, gas, star-formation, metallicity, and dust "
         rf"quantities are measured within {aperture_text}.  Gas metallicity is "
         rf"reported in solar units using $Z_\odot={Z_SOLAR:g}$.  We define "
         r"${\rm D/G}=M_{\rm dust}/M_{\rm gas}$ and "
         r"${\rm D/Z}=M_{\rm dust}/(M_{Z,{\rm gas}}+M_{\rm dust})$, i.e. the "
         r"fraction of the total metal mass locked in dust.  Ellipses denote "
         r"runs that were not yet available.}"),
        r"\label{tab:halo_sample}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l c c c c c c c c c c}",
        r"\hline",
        r"\textbf{Halo} & \textbf{Halo ID} & \textbf{$\log_{10} M_{200{\rm c}}$} & \textbf{$R_{200{\rm c}}$} & \textbf{$\log_{10} M_\star$} & \textbf{$\log_{10} M_{\rm gas}$} & \textbf{SFR} & \textbf{$Z_{\rm gas}/Z_\odot$} & \textbf{$\log_{10} M_{\rm dust}$} & \textbf{D/G} & \textbf{D/Z} \\",
        r"& & \textbf{($M_\odot$)} & \textbf{(kpc)} & \textbf{($M_\odot$)} & \textbf{($M_\odot$)} & \textbf{($M_\odot\,{\rm yr}^{-1}$)} & & \textbf{($M_\odot$)} & & \\",
        r"\hline",
    ]

    for group_name, members in HALO_GROUPS:
        lines.append(rf"\multicolumn{{11}}{{l}}{{\textbf{{{group_name}}}}} \\")
        for cg in members:
            row = by_cg.get(cg)
            if row is None:
                cells = [r"\ldots"] * 9
            else:
                z_solar = float(row.get("Zgas_mass_fraction", np.nan)) / Z_SOLAR
                cells = [
                    latex_log10(row.get("zoom_M200c_Msun")),
                    latex_number(row.get("R200c_pkpc"), 1),
                    latex_log10(row.get("Mstar_Msun")),
                    latex_log10(row.get("Mgas_Msun")),
                    latex_number(row.get("SFR_Msun_yr"), 3),
                    latex_number(z_solar, 3),
                    latex_log10(row.get("Mdust_Msun")),
                    latex_number(row.get("dust_to_gas"), 5),
                    latex_number(row.get("dust_fraction_total_metals"), 3),
                ]
            lines.append(f"{cg} & {CG_HALOS[cg]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend([r"\end{tabular}", r"\end{table*}"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Compare all currently completed CosmicGrain zoom halos.")
    ap.add_argument("--root", default=".", help="Directory containing halo<ID>_output_<resolution> runs")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--outdir", default="cosmicgrain_multihalo_comparison")
    ap.add_argument("--style-file", default=None,
                    help="Path to cosmicgrain.mplstyle (default: beside this script, then cwd)")
    ap.add_argument("--skip-devis", action="store_true",
                    help="Do not fetch/cache De Vis+19 and THEMIS galaxy catalogs for paper Fig. 2")
    ap.add_argument("--catalog-check", action="store_true",
                    help="Check the observational catalogs and exit without reading simulation snapshots")
    ap.add_argument("--r25-csv", default=None,
                    help="CSV with CG,R25_pkpc, measured from B-band 25 mag/arcsec² "
                         "profiles in the snapshot XY projection; required for matched Fig. 2 x values")
    ap.add_argument("--r25-annulus-fraction", type=float, default=0.1,
                    help="Half-width of the 0.4 R25 annulus as a fraction of its radius (default 0.1)")
    ap.add_argument("--aperture-pkpc", type=float, default=None,
                    help="Fixed central-galaxy radius; overrides --central-aperture-r200")
    ap.add_argument("--central-aperture-r200", type=float, default=0.1,
                    help="Central-galaxy radius in R200c when no fixed aperture is given (default: 0.1)")
    ap.add_argument("--radial-max-pkpc", type=float, default=30.0)
    ap.add_argument("--radial-bins", type=int, default=24)
    ap.add_argument("--expected-snapshots", type=int, default=30)
    ap.add_argument("--include-incomplete", action="store_true",
                    help="Use latest paired snapshot even if fewer than expected snapshots exist")
    ap.add_argument("--group-map", default=None,
                    help='Manual fallback zoom FOF groups, e.g. "CG01:7,CG02:3,CG10:1". '
                         "Normally unnecessary: audit JSON is preferred, then the "
                         "suite target is inferred automatically.")
    ap.add_argument("--source-map", default="0:SNII,1:AGB,2:LRN")
    ap.add_argument("--grain-radius-to-micron", type=float, default=1.0e-3,
                    help="Multiply stored GrainRadius (nm) by this to obtain microns (default: 1e-3)")
    ap.add_argument(
        "--xh-mode", choices=("fixed", "metal_corrected"),
        default="metal_corrected",
        help=("Hydrogen fraction for O/H: fixed uses X_H=0.76; "
              "metal_corrected uses X_H=0.76(1-Zgas) (default)"),
    )
    ap.add_argument(
        "--dataset-label",
        default="PROVISIONAL — LEGACY CHEMISTRY — NOT FOR FINAL QUANTITATIVE USE",
        help="Status text printed on every figure; pass an empty string to suppress",
    )
    ap.add_argument("--grain-min-micron", type=float, default=1.0e-3)
    ap.add_argument("--grain-max-micron", type=float, default=0.2,
                    help="Upper GSD edge in microns (default: 0.2 = 200 nm)")
    ap.add_argument("--grain-bins", type=int, default=24,
                    help="Common logarithmic GSD bins (default: 24; deliberately coarse for suite plots)")
    ap.add_argument(
        "--grain-representatives", default="CG01,CG10,CG12",
        help=("Three comma-separated systems for the Figure 3 GSD panels "
              "(default: CG01,CG10,CG12)"),
    )
    ap.add_argument("--grain-evolution-redshift", type=float, default=2.0,
                    help="Earlier epoch for paper Fig. 3 evolution overlays (default: 2.0)")
    ap.add_argument(
        "--include-grain-evolution", action="store_true",
        help=("Attempt earlier-epoch overlays. Off by default until validated "
              "main-progenitor matching is available."),
    )
    ap.add_argument("--skip-grain-evolution-row", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    apply_cosmicgrain_style(args.style_file)
    if not 0 < args.r25_annulus_fraction < 1:
        ap.error("--r25-annulus-fraction must lie strictly between zero and one")
    grain_representatives = tuple(
        item.strip().upper() for item in args.grain_representatives.split(",")
        if item.strip()
    )
    if len(grain_representatives) != 3 or len(set(grain_representatives)) != 3:
        ap.error("--grain-representatives requires three distinct CG labels")
    invalid_representatives = [
        cg for cg in grain_representatives if not re.fullmatch(r"CG(?:0[1-9]|1[0-2])", cg)
    ]
    if invalid_representatives:
        ap.error("invalid --grain-representatives: " + ", ".join(invalid_representatives))
    r25_by_cg = {}
    if args.r25_csv:
        with open(args.r25_csv, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                cg = row["CG"].strip().upper()
                radius = float(row["R25_pkpc"])
                if cg in r25_by_cg or not np.isfinite(radius) or radius <= 0:
                    ap.error(f"invalid or duplicate R25 for {cg}")
                r25_by_cg[cg] = radius

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if args.catalog_check:
        points = load_devis_galaxies(outdir)
        print(f"De Vis+19 / Nersesian+19: {len(points)} joined galaxies; "
              f"O/H range {np.min(points[:, 0]):.2f}–{np.max(points[:, 0]):.2f}")
        return 0

    group_map = parse_group_map(args.group_map)
    source_map = {}
    for item in args.source_map.split(","):
        code, label = item.split(":", 1)
        source_map[int(code.strip())] = label.strip()

    runs = []
    skipped_no_group = []
    for cg, halo_id in CG_HALOS.items():
        run = choose_completed_run(
            root, cg, halo_id, args.resolution, args.expected_snapshots,
            group_map, require_complete=not args.include_incomplete
        )
        if run:
            runs.append(run)
        else:
            # Distinguish "exists but no group mapping/audit" in console message.
            candidates = candidate_output_dirs(root, cg, halo_id, args.resolution)
            if candidates:
                skipped_no_group.append((cg, candidates))

    if not runs:
        print("No usable CosmicGrain runs found.", file=sys.stderr)
        print("Expected directories like halo441_output_1024.", file=sys.stderr)
        print("Target groups are inferred automatically when no audit/group-map is available.", file=sys.stderr)
        return 2

    print("=" * 88)
    print("COSMICGRAIN MULTI-HALO SUITE COMPARISON")
    print("=" * 88)
    print(f"Root: {root}")
    print(f"Resolution: {args.resolution}")
    if args.aperture_pkpc is None:
        print(f"Central galaxy: r < {args.central_aperture_r200:g} R200c")
    else:
        print(f"Central galaxy: r < {args.aperture_pkpc:g} pkpc (fixed override)")
    print("CGM: central aperture < r < R200c; entire halo: r < R200c")
    print(f"O/H hydrogen prescription: {args.xh_mode}")
    print(f"Dataset label: {args.dataset_label or '(none)'}")
    print(f"Found {len(runs)}/12 usable halos:")
    for r in runs:
        print(f"  {r.cg}: halo{r.halo_id}, snap {r.snap:03d}, group {r.group_index}, "
              f"{os.path.basename(r.output_dir)} [{r.group_source}]")
    if skipped_no_group:
        print("\nDirectories were found for the following CG halos but no usable run was selected")
        print("(incomplete when completion was required, missing paired data, or target inference failed):")
        for cg, dirs in skipped_no_group:
            print(f"  {cg}: " + ", ".join(os.path.basename(x) for x in dirs))

    grain_edges = np.logspace(
        np.log10(args.grain_min_micron),
        np.log10(args.grain_max_micron),
        args.grain_bins + 1,
    )

    summaries, profiles = [], []
    failures = []
    for r in runs:
        print(f"\nAnalyzing {r.cg} ({os.path.basename(r.output_dir)})...")
        try:
            summary, profile = analyze_run(
                r, args.aperture_pkpc, args.central_aperture_r200,
                args.radial_max_pkpc, args.radial_bins,
                args.grain_radius_to_micron, grain_edges, source_map,
                args.xh_mode, args.dataset_label,
                r25_pkpc=r25_by_cg.get(r.cg),
                r25_annulus_fraction=args.r25_annulus_fraction,
            )
            summaries.append(summary)
            profiles.append(profile)
            print(
                f"  M200={summary['zoom_M200c_Msun']:.3e}  "
                f"Mstar={summary['Mstar_Msun']:.3e}  "
                f"Mgas={summary['Mgas_Msun']:.3e}  "
                f"Mdust={summary['Mdust_Msun']:.3e} Msun"
            )
            print(
                f"  Md/M*={summary['dust_to_stellar']:.3e}  "
                f"D/G={summary['dust_to_gas']:.3e}  "
                f"D/Ztot={summary['dust_fraction_total_metals']:.3e}  "
                f"12+log(O/H)={summary['gas_12_plus_logOH']:.3f}"
            )
            print(
                f"  grains: N={summary['grain_particle_count']:,}  "
                f"unique radii={summary['grain_unique_radius_count']:,}  "
                f"range={summary['grain_min_micron']:.3e}--{summary['grain_max_micron']:.3e} um  "
                f"edge-mass={summary['grain_min_boundary_mass_fraction']:.2%}/"
                f"{summary['grain_max_boundary_mass_fraction']:.2%}"
            )
            print(
                f"  aperture r<{summary['aperture_pkpc']:.2f} pkpc: "
                f"Nstar={summary['Nstar_aperture']:,} "
                f"Ngas={summary['Ngas_aperture']:,} "
                f"Ndust={summary['Ndust_aperture']:,}; "
                f"R200c: Ndust={summary['Ndust_R200c']:,}, "
                f"Mdust={summary['Mdust_R200c_Msun']:.3e} Msun; "
                f"outside aperture: Mdust={summary['Mdust_outside_aperture_inside_R200c_Msun']:.3e} Msun"
            )
            if summary['dust_fraction_total_metals'] < 0.01:
                print("  CHECK: very low D/Ztot; compare central and R200c dust "
                      "censuses above, then verify the selected target group "
                      "against the postrun audit.")
        except Exception as exc:
            failures.append((r.cg, str(exc)))
            print(f"  FAILED: {exc}", file=sys.stderr)

    if not summaries:
        print("All selected halos failed analysis.", file=sys.stderr)
        return 2

    # Preserve canonical CG ordering.
    order = np.argsort([int(s["CG"][2:]) for s in summaries])
    summaries = [summaries[i] for i in order]
    profiles = [profiles[i] for i in order]

    # Analyze every system at the paired snapshot nearest the requested earlier
    # redshift. Re-identify the target group independently at that snapshot
    # because catalog group indices are not time-invariant. Figure 3 uses all
    # validated matches in its suite-wide evolution row and never substitutes
    # an unrelated group when a match fails.
    earlier_epochs = {}
    earlier_failed = []
    if (args.include_grain_evolution and
            not args.skip_grain_evolution_row and len(summaries) >= 1):
        run_by_cg = {r.cg: r for r in runs}
        print(f"\nEarlier-epoch grain populations (target z={args.grain_evolution_redshift:g}):")
        for i, current_summary in enumerate(summaries):
            cg = current_summary["CG"]
            base = run_by_cg[cg]
            nearest = nearest_paired_snapshot(base.output_dir, args.grain_evolution_redshift)
            if nearest is None:
                print(f"  {cg}: no paired earlier snapshot")
                continue
            snap, z = nearest
            try:
                group, matched_snap, _ = infer_zoom_target_group(
                    base.output_dir, reference_snap=snap, verbose=False
                )
                if matched_snap is not None and int(matched_snap) != snap:
                    raise RuntimeError(f"target matcher returned snap {matched_snap}, requested {snap}")
                epoch_run = RunInfo(
                    cg=base.cg, halo_id=base.halo_id, output_dir=base.output_dir,
                    snap=snap, group_index=int(group), audit_json=None,
                    group_source=f"auto:infer_zoom_target_group@{snap:03d}",
                )
                es, ep = analyze_run(
                    epoch_run, args.aperture_pkpc, args.central_aperture_r200,
                    args.radial_max_pkpc, args.radial_bins,
                    args.grain_radius_to_micron, grain_edges, source_map,
                    args.xh_mode, args.dataset_label,
                )
                earlier_epochs[cg] = (es, ep)
                print(f"  {cg}: snap={snap:03d} z={z:.3f} group={int(group)} "
                      f"N_dust={es['grain_particle_count']:,}")
            except Exception as exc:
                earlier_failed.append(cg)
                print(f"  {cg}: earlier-epoch analysis failed: {exc}")
        if earlier_failed:
            print("  Failed earlier-epoch systems are omitted from the evolution "
                  "overlay; no unrelated group is substituted.")

    # Summary CSV.
    csv_path = os.path.join(outdir, "cosmicgrain_multihalo_summary.csv")
    keys = list(summaries[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summaries:
            w.writerow({k: jsonable(row.get(k)) for k in keys})
    print(f"\nWrote {csv_path}")

    # Full JSON includes radial/profile arrays.
    json_path = os.path.join(outdir, "cosmicgrain_multihalo_summary.json")
    payload = {
        "settings": vars(args),
        "halos": [
            {"summary": s, "profiles": p}
            for s, p in zip(summaries, profiles)
        ],
        "failures": failures,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Wrote {json_path}")

    tex_path = os.path.join(outdir, "table_halo_sample.tex")
    write_latex_halo_table(
        summaries, tex_path, args.aperture_pkpc, args.central_aperture_r200
    )
    print(f"Wrote {tex_path}")

    devis_galaxies = None
    if not args.skip_devis:
        try:
            devis_galaxies = load_devis_galaxies(outdir)
            print(f"De Vis+19 / Nersesian+19: {len(devis_galaxies)} "
                  "matched galaxies with O/H, dust and total metal masses")
        except (OSError, ValueError, KeyError, TimeoutError) as exc:
            print(f"Observational catalog unavailable ({exc}); "
                  "paper Fig. 2 shows simulations only. No fabricated points are plotted.")

    plot_all(summaries, profiles, outdir, args.dataset_label)
    plot_paper_figure_set(
        summaries, profiles, outdir, args.dataset_label,
        earlier_epochs=earlier_epochs, devis_galaxies=devis_galaxies,
        grain_representatives=grain_representatives,
    )

    note_path = os.path.join(outdir, "paper_figure_status.txt")
    with open(note_path, "w", encoding="utf-8") as handle:
        handle.write("CURRENT SIX-FIGURE PAPER PLAN\n"
                     "1 suite context: paper_fig01_suite_coverage\n"
                     "2 dust buildup: paper_fig02_dust_enrichment\n"
                     "3 grain populations: paper_fig03_grain_populations\n"
                     "4 grains behind R(V): dedicated cosmicgrain_figure4_rv_grains.py\n"
                     "5 suite R(V): dedicated cosmicgrain_figure5_direct.py\n"
                     "6 MW benchmark: dedicated cosmicgrain_figure6_mw_comparison.py; "
                     "Zhang/Green and power spectrum still require matched comparison\n\n"
                     "Fig. 2: simulation x follows Figure 1 halo by halo: SFR-weighted "
                     "gas O/H within the central aperture is shown with a filled symbol "
                     "when at least 20 star-forming gas particles contribute; otherwise "
                     "the aperture-integrated all-gas O/H is shown with an open symbol. "
                     "De Vis x = characteristic strong-line PG16S O/H at 0.4 R25. "
                     "De Vis+19 used the Nersesian+19 THEMIS dust masses; "
                     "DustPedia gas mass is reconstructed as (MZtotal-Mdust)/"
                     "[27.36 x (O/H)] from their published total metal masses. "
                     "The two x definitions and spatial apertures differ.\n"
                     "Rémy-Ruyer+14 line: their Table 1 broken G/D relation "
                     "for metallicity-dependent X_CO, inverted to D/G, "
                     "plotted only for 7.1 <= 12+log(O/H) <= 9.1. "
                     "This fit summarizes a different observed sample and "
                     "has an adopted solar G/D normalization of 162; "
                     "the curve is not an error bound.\n"
                     f"Fig. 3 earlier-epoch coverage: {len(earlier_epochs)}/"
                     f"{len(summaries)} validated halos; unavailable matches are omitted.\n" +
                     "Old paper_fig04-07 files from prior runs may remain in this folder; "
                     "they are not current paper outputs.\n")
    print(f"Wrote {note_path}")

    print("\n" + "=" * 88)
    print(f"DONE: analyzed {len(summaries)}/12 halos")
    if failures:
        print("Failures:")
        for cg, msg in failures:
            print(f"  {cg}: {msg}")
    print(f"Products: {outdir}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
