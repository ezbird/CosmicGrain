#!/usr/bin/env python3
"""
cosmicgrain_z0_z6_gsd_rv.py
===========================

Make a real-data CosmicGrain z=0 versus z~6 comparison:

  (A) suite-integrated dust-mass grain-size distribution, z=0 vs z~6
  (B) ratio of the normalized z=0 / z~6 GSD
  (C) CG10 (halo 441) R(V) PDF, z=0 vs its tracked z~6 main progenitor
  (D) CG10-only dust-mass GSD, z=0 vs its tracked z~6 main progenitor

IMPORTANT
---------
* snapshot 007 is treated as the z~6 snapshot; snapshot 029 is z=0.
* Earlier halos are identified with halo_utils.get_zoom_halo_series(), which
  tracks the final target backward using conserved high-resolution DM IDs.
* "Suite-integrated" means the dust mass inside R200c of the 12 selected zoom
  halos is summed. It is NOT a formal volume-weighted cosmic mean.
* GrainRadius is assumed to be stored in nm.
* HubbleParam is read from Parameters, not Header.
* R(V) uses the same D16MGemt carbon + DH16asil silicate optical constants
  as the existing CosmicGrain R(V) analysis.
* R(V) maps use a fixed physical pixel size (default 0.5 pkpc).
* Each epoch is oriented face-on independently using stellar angular momentum;
  gas is used as a fallback if too few stars are available.

Run from ~/gadget4:
    python3 scripts/cosmicgrain_z0_z6_gsd_rv.py

Useful options:
    python3 scripts/cosmicgrain_z0_z6_gsd_rv.py \
        --root . \
        --snap-z6 7 --snap-z0 29 \
        --pixel-pkpc 0.5 \
        --output cosmicgrain_z0_z6_gsd_rv.png

Outputs
-------
* PNG and PDF figure
* NPZ containing histogram arrays and R(V) samples
* CSV summary of tracked halos and dust counts/masses
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
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from halo_utils import get_zoom_halo_series, periodic_delta


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

RHO_C = 2.24
RHO_SIL = 3.50
B_MICRON = 0.44
V_MICRON = 0.55

_float_re = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--root", default=".",
                   help="GADGET-4 root containing halo*_output_1024 directories")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--cg10-output-dir", type=Path, default=None,
                   help="Override CG10 output directory for both epochs; "
                        "e.g. halo441_output_1024_mingashsml0.0")
    p.add_argument("--snap-z6", type=int, default=7)
    p.add_argument("--snap-z0", type=int, default=29)
    p.add_argument("--halos", nargs="*", default=list(CG_HALOS),
                   choices=list(CG_HALOS),
                   help="Suite members used for the integrated GSD")
    p.add_argument("--cg10-group", type=int, default=None,
                   help="Override CG10 final group index; otherwise read audit JSON")
    p.add_argument("--n-gsd-bins", type=int, default=45)
    p.add_argument("--amin-nm", type=float, default=1.0)
    p.add_argument("--amax-nm", type=float, default=160.0)
    p.add_argument("--pixel-pkpc", type=float, default=0.5)
    p.add_argument("--rv-radius-pkpc", type=float, default=20.0,
                   help="Maximum projected half-width; capped at R200c")
    p.add_argument("--rv-zmax-pkpc", type=float, default=20.0,
                   help="Maximum LOS half-depth; capped at R200c")
    p.add_argument("--min-dust-count", type=int, default=1)
    p.add_argument("--min-signal-fraction", type=float, default=1e-3)
    p.add_argument("--orient-radius-pkpc", type=float, default=20.0)
    p.add_argument("--carbon-table",
                   default="dust_optics/callqcomp.out_D16MGemt")
    p.add_argument("--silicate-table",
                   default="dust_optics/callqcomp.out_DH16asil")
    p.add_argument("--style-file", default="scripts/cosmicgrain.mplstyle")
    p.add_argument("--output", default="cosmicgrain_z0_z6_gsd_rv.png")
    return p.parse_args()


def run_dir(root: str, cg: str, resolution: int) -> Path:
    return Path(root) / f"halo{CG_HALOS[cg]}_output_{resolution}"


def snapshot_pattern(outdir: Path, snap: int) -> str:
    s = f"{snap:03d}"
    return str(outdir / f"snapdir_{s}" / f"snapshot_{s}.*.hdf5")


def expand_snapshot(outdir: Path, snap: int) -> List[str]:
    s = f"{snap:03d}"
    pats = [
        str(outdir / f"snapdir_{s}" / f"snapshot_{s}.*.hdf5"),
        str(outdir / f"snapdir_{s}" / f"snapshot_{s}.hdf5"),
        str(outdir / f"snapshot_{s}.*.hdf5"),
        str(outdir / f"snapshot_{s}.hdf5"),
    ]
    files = []
    for pat in pats:
        files = sorted(glob.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No snapshot {s} found in {outdir}")
    return files


def read_header(outdir: Path, snap: int):
    fn = expand_snapshot(outdir, snap)[0]
    with h5py.File(fn, "r") as f:
        a = float(f["Header"].attrs["Time"])
        box = float(f["Header"].attrs["BoxSize"])
        h = float(f["Parameters"].attrs["HubbleParam"])
    return a, h, box


def read_ptype(outdir: Path, snap: int, ptype: int,
               fields: Sequence[str]) -> Dict[str, np.ndarray]:
    chunks = {field: [] for field in fields}
    found_any = False
    for fn in expand_snapshot(outdir, snap):
        with h5py.File(fn, "r") as f:
            key = f"PartType{ptype}"
            if key not in f:
                continue
            found_any = True
            g = f[key]
            for field in fields:
                if field not in g:
                    raise KeyError(f"{fn}: {key}/{field} missing")
                chunks[field].append(np.asarray(g[field][:]))
    if not found_any:
        return {field: np.empty((0, 3) if field in ("Coordinates", "Velocities")
                                else (0,)) for field in fields}
    return {field: np.concatenate(chunks[field], axis=0) for field in fields}


def read_dust(outdir: Path, snap: int):
    files = expand_snapshot(outdir, snap)
    coords, mass, radius, carbon = [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType6" not in f:
                continue
            g = f["PartType6"]
            cf = ("CarbonMassFraction" if "CarbonMassFraction" in g
                  else "CarbonFraction" if "CarbonFraction" in g else None)
            if cf is None:
                raise KeyError(f"{fn}: no dust carbon-fraction field")
            coords.append(np.asarray(g["Coordinates"][:], dtype=np.float64))
            mass.append(np.asarray(g["Masses"][:], dtype=np.float64))
            radius.append(np.asarray(g["GrainRadius"][:], dtype=np.float64))
            carbon.append(np.asarray(g[cf][:], dtype=np.float64))
    if not mass:
        return {
            "coords": np.empty((0, 3)),
            "mass": np.empty(0),
            "radius_nm": np.empty(0),
            "carbon": np.empty(0),
        }
    return {
        "coords": np.concatenate(coords),
        "mass": np.concatenate(mass),
        "radius_nm": np.concatenate(radius),
        "carbon": np.concatenate(carbon),
    }


def find_audit_group(outdir: Path) -> int:
    candidates = sorted(outdir.glob("*postrun_audit.json"))
    if not candidates:
        candidates = sorted(outdir.glob("**/*postrun_audit.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No postrun audit JSON in {outdir}; cannot safely anchor final halo."
        )

    with open(candidates[-1]) as f:
        d = json.load(f)

    keys = ("group_index", "group_idx", "target_group", "accepted_group_index")
    for key in keys:
        if key in d and d[key] is not None:
            return int(d[key])

    # Search one level down because audit layouts have changed over time.
    for value in d.values():
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] is not None:
                    return int(value[key])

    raise KeyError(
        f"Could not find final group index in {candidates[-1]}"
    )


def halo_mask(coords, halo, a, h, box):
    dr = periodic_delta(coords, np.asarray(halo["center"]), box)
    r_pkpc = np.linalg.norm(dr, axis=1) * a / h
    return r_pkpc < float(halo["r200_pkpc"]), r_pkpc


def normalized_mass_gsd(radius_nm, mass_msun, edges):
    good = (
        np.isfinite(radius_nm) & np.isfinite(mass_msun)
        & (radius_nm > 0) & (mass_msun > 0)
    )
    hist, _ = np.histogram(radius_nm[good], bins=edges,
                           weights=mass_msun[good])
    dlog = np.diff(np.log10(edges))
    total = hist.sum()
    y = hist / total / dlog if total > 0 else np.zeros_like(hist)
    return y, hist, total


def numbers_in_line(line):
    vals = []
    for x in _float_re.findall(line):
        try:
            vals.append(float(x.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return vals


def read_draine_qext_table(filename):
    blocks = {}
    current_radius = None
    wave, qext = [], []

    def finish():
        nonlocal wave, qext, current_radius
        if current_radius is not None and len(wave) >= 10:
            w = np.asarray(wave, float)
            q = np.asarray(qext, float)
            good = np.isfinite(w) & np.isfinite(q) & (w > 0) & (q >= 0)
            w, q = w[good], q[good]
            if len(w):
                order = np.argsort(w)
                blocks[current_radius] = (w[order], q[order])
        wave, qext = [], []

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if ("radius" in low or "rad" in low) and "micron" in low:
                nums = numbers_in_line(line)
                if nums:
                    finish()
                    current_radius = nums[0]
                continue
            if current_radius is None:
                continue
            vals = numbers_in_line(line)
            if len(vals) >= 4 and vals[0] > 0 and vals[1] >= 0:
                wave.append(vals[0])
                qext.append(vals[1])
    finish()
    if not blocks:
        raise RuntimeError(f"Could not parse optical table {filename}")
    return blocks


def qext_at_wavelength(blocks, particle_radius_um, wavelength_um):
    table_radii = np.array(sorted(blocks), float)
    qlam = np.empty(len(table_radii))
    for i, rad in enumerate(table_radii):
        wave, q = blocks[rad]
        if not (wave.min() <= wavelength_um <= wave.max()):
            raise RuntimeError(f"{wavelength_um} micron outside optical table")
        qlam[i] = np.interp(np.log(wavelength_um), np.log(wave), q)

    pr = np.asarray(particle_radius_um, float)
    if np.any(pr <= 0):
        raise RuntimeError("Non-positive grain radius")
    if pr.min() < table_radii.min() or pr.max() > table_radii.max():
        raise RuntimeError(
            f"Grain radii {pr.min():.4g}-{pr.max():.4g} micron outside "
            f"optics range {table_radii.min():.4g}-{table_radii.max():.4g}"
        )
    return np.interp(np.log(pr), np.log(table_radii), qlam)


def extinction_weights(mass, radius_nm, carbon, carbon_blocks, sil_blocks):
    radius_um = radius_nm * 1e-3
    fC = np.clip(carbon, 0, 1)
    qCB = qext_at_wavelength(carbon_blocks, radius_um, B_MICRON)
    qCV = qext_at_wavelength(carbon_blocks, radius_um, V_MICRON)
    qSB = qext_at_wavelength(sil_blocks, radius_um, B_MICRON)
    qSV = qext_at_wavelength(sil_blocks, radius_um, V_MICRON)
    pref = mass / radius_um
    WB = pref * (fC * qCB / RHO_C + (1-fC) * qSB / RHO_SIL)
    WV = pref * (fC * qCV / RHO_C + (1-fC) * qSV / RHO_SIL)
    return WB, WV


def rotation_from_jhat(jhat):
    """Return matrix R such that row-vector positions use pos @ R.T."""
    z = np.asarray(jhat, float)
    z /= np.linalg.norm(z)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    x = np.cross(trial, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.vstack([x, y, z])


def orientation_from_component(outdir, snap, halo, ptype, a, h, box,
                               radius_pkpc, min_particles):
    fields = ["Coordinates", "Velocities", "Masses"]
    p = read_ptype(outdir, snap, ptype, fields)
    if len(p["Masses"]) == 0:
        return None, 0

    dr_code = periodic_delta(p["Coordinates"], np.asarray(halo["center"]), box)
    dr = dr_code * a / h
    r = np.linalg.norm(dr, axis=1)
    rcut = min(radius_pkpc, float(halo["r200_pkpc"]))
    sel = r < rcut
    n = int(sel.sum())
    if n < min_particles:
        return None, n

    mass = np.asarray(p["Masses"][sel], float)
    vel = np.asarray(p["Velocities"][sel], float)
    pos = np.asarray(dr[sel], float)

    # For an orientation vector only, subtracting the mass-weighted bulk
    # velocity is the important operation; any global velocity unit factor
    # cancels out of the normalized angular-momentum direction.
    vbulk = np.average(vel, axis=0, weights=mass)
    L = np.sum(mass[:, None] * np.cross(pos, vel - vbulk), axis=0)
    norm = np.linalg.norm(L)
    if not np.isfinite(norm) or norm <= 0:
        return None, n
    return rotation_from_jhat(L / norm), n


def derive_orientation(outdir, snap, halo, a, h, box, radius_pkpc):
    R, n = orientation_from_component(
        outdir, snap, halo, 4, a, h, box, radius_pkpc, min_particles=20
    )
    if R is not None:
        return R, "stars", n
    R, ng = orientation_from_component(
        outdir, snap, halo, 0, a, h, box, radius_pkpc, min_particles=50
    )
    if R is not None:
        return R, "gas", ng
    raise RuntimeError(
        f"Could not determine face-on orientation at snapshot {snap}: "
        f"only {n} usable stars and {ng} usable gas particles."
    )


def rv_samples(outdir, snap, halo, carbon_blocks, sil_blocks,
               pixel_pkpc, max_radius_pkpc, max_z_pkpc,
               min_count, min_signal_fraction, orient_radius_pkpc):
    a, h, box = read_header(outdir, snap)
    dust = read_dust(outdir, snap)
    if len(dust["mass"]) == 0:
        raise RuntimeError(f"No dust at snapshot {snap}")

    R, orient_source, norient = derive_orientation(
        outdir, snap, halo, a, h, box, orient_radius_pkpc
    )

    dr_code = periodic_delta(dust["coords"], np.asarray(halo["center"]), box)
    pos = (dr_code * a / h) @ R.T
    r3d = np.linalg.norm(pos, axis=1)

    # Restrict to the tracked halo itself, then to a face-on cylinder.
    r200 = float(halo["r200_pkpc"])
    radius = min(max_radius_pkpc, r200)
    zmax = min(max_z_pkpc, r200)
    sel = (
        (r3d < r200)
        & (np.abs(pos[:, 0]) < radius)
        & (np.abs(pos[:, 1]) < radius)
        & (np.abs(pos[:, 2]) < zmax)
    )
    if sel.sum() < 10:
        raise RuntimeError(
            f"Only {sel.sum()} dust particles in CG10 R(V) volume at snap {snap}"
        )

    WB, WV = extinction_weights(
        dust["mass"][sel], dust["radius_nm"][sel], dust["carbon"][sel],
        carbon_blocks, sil_blocks
    )

    # Keep approximately the requested physical pixel size.
    bins = max(1, int(np.ceil(2 * radius / pixel_pkpc)))
    radius_eff = 0.5 * bins * pixel_pkpc
    xy = pos[sel, :2]
    rng = [[-radius_eff, radius_eff], [-radius_eff, radius_eff]]

    tauB = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                          range=rng, weights=WB)[0].T
    tauV = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                          range=rng, weights=WV)[0].T
    count = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                           range=rng)[0].T

    denom = tauB - tauV
    cut = min_signal_fraction * np.nanmax(tauV) if np.nanmax(tauV) > 0 else np.inf
    good = (
        (count >= min_count)
        & (tauV > cut)
        & np.isfinite(tauV)
        & np.isfinite(denom)
        & (denom > 0)
    )
    rv = tauV[good] / denom[good]
    rv = rv[np.isfinite(rv) & (rv > 0)]

    info = {
        "a": a,
        "z": 1.0/a - 1.0,
        "r200_pkpc": r200,
        "map_radius_pkpc": radius_eff,
        "zmax_pkpc": zmax,
        "bins": bins,
        "pixel_pkpc": pixel_pkpc,
        "n_dust_volume": int(sel.sum()),
        "n_valid_pixels": int(len(rv)),
        "orientation_source": orient_source,
        "n_orientation_particles": int(norient),
        "rotation": R,
    }
    return rv, info


def pdf_hist(values, edges):
    h, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    total = h.sum()
    return (h / total / widths if total > 0 else np.zeros_like(h, float)), h


def step_xy(edges, y):
    return edges, np.r_[y, y[-1] if len(y) else 0]


def main():
    args = parse_args()

    style = Path(args.style_file)
    if style.exists():
        plt.style.use(str(style))

    root = Path(args.root).resolve()
    cg10_override = None
    if args.cg10_output_dir is not None:
        cg10_override = args.cg10_output_dir
        if not cg10_override.is_absolute():
            cg10_override = root / cg10_override
        cg10_override = cg10_override.resolve()
        if not cg10_override.is_dir():
            raise FileNotFoundError(f"CG10 output directory not found: {cg10_override}")
    def output_dir(cg):
        return cg10_override if cg == "CG10" and cg10_override is not None \
            else run_dir(root, cg, args.resolution)
    snap0, snap6 = args.snap_z0, args.snap_z6

    # ------------------------------------------------------------
    # Track every requested halo from z=0 back to z~6.
    # ------------------------------------------------------------
    tracked = {}
    summary_rows = []
    suite_r = {snap0: [], snap6: []}
    suite_m = {snap0: [], snap6: []}

    print("=" * 78)
    print("TRACKING COSMICGRAIN HALOS")
    print("=" * 78)

    for cg in args.halos:
        outdir = output_dir(cg)
        if not outdir.is_dir():
            print(f"{cg}: SKIP -- missing {outdir}")
            continue

        try:
            group = (args.cg10_group if cg == "CG10" and args.cg10_group is not None
                     else find_audit_group(outdir))
            print(f"\n{cg}: {outdir.name}; final group={group}")
            series = get_zoom_halo_series(
                outdir,
                [snap6, snap0],
                group_index=group,
                reference_snap=snap0,
                verbose=True,
                refine_reference_center=True,
                refine_historical_centers=False,
            )
            tracked[cg] = series
        except Exception as exc:
            print(f"{cg}: TRACKING FAILED: {exc}")
            continue

        for snap in (snap6, snap0):
            halo = series.get(snap)
            if halo is None:
                print(f"{cg} snap {snap:03d}: no tracked halo")
                continue

            try:
                a, h, box = read_header(outdir, snap)
                dust = read_dust(outdir, snap)
                mask, _ = halo_mask(dust["coords"], halo, a, h, box)
                mass_msun = dust["mass"][mask] * 1e10 / h
                rad = dust["radius_nm"][mask]

                good = (
                    np.isfinite(rad) & np.isfinite(mass_msun)
                    & (rad > 0) & (mass_msun > 0)
                )
                suite_r[snap].append(rad[good])
                suite_m[snap].append(mass_msun[good])

                summary_rows.append({
                    "cg": cg,
                    "halo_id": CG_HALOS[cg],
                    "snap": snap,
                    "z": 1/a - 1,
                    "group_idx": halo.get("group_idx", ""),
                    "m200_msun": halo.get("m200_msun", np.nan),
                    "r200_pkpc": halo.get("r200_pkpc", np.nan),
                    "n_dust_r200": int(good.sum()),
                    "mdust_r200_msun": float(mass_msun[good].sum()),
                })
                print(
                    f"{cg} snap {snap:03d}: z={1/a-1:.3f}, "
                    f"R200={halo['r200_pkpc']:.2f} pkpc, "
                    f"Ndust={good.sum():,}, Mdust={mass_msun[good].sum():.3e} Msun"
                )
            except Exception as exc:
                print(f"{cg} snap {snap:03d}: DUST LOAD FAILED: {exc}")

    if not suite_r[snap0] or not suite_r[snap6]:
        raise RuntimeError(
            "Need at least one successfully tracked/dust-loaded halo at both epochs."
        )

    r0 = np.concatenate(suite_r[snap0])
    m0 = np.concatenate(suite_m[snap0])
    r6 = np.concatenate(suite_r[snap6])
    m6 = np.concatenate(suite_m[snap6])

    edges = np.logspace(
        np.log10(args.amin_nm), np.log10(args.amax_nm), args.n_gsd_bins + 1
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    gsd0, gsd0_mass, md0 = normalized_mass_gsd(r0, m0, edges)
    gsd6, gsd6_mass, md6 = normalized_mass_gsd(r6, m6, edges)

    # ------------------------------------------------------------
    # CG10-only GSD and R(V)
    # ------------------------------------------------------------
    if "CG10" not in tracked:
        raise RuntimeError("CG10 did not track successfully; cannot make R(V) panels.")

    cg10_dir = output_dir("CG10")
    cg10_series = tracked["CG10"]

    cg10_gsd = {}
    for snap in (snap6, snap0):
        halo = cg10_series[snap]
        a, h, box = read_header(cg10_dir, snap)
        dust = read_dust(cg10_dir, snap)
        mask, _ = halo_mask(dust["coords"], halo, a, h, box)
        mass = dust["mass"][mask] * 1e10 / h
        rad = dust["radius_nm"][mask]
        cg10_gsd[snap] = normalized_mass_gsd(rad, mass, edges)[0]

    carbon_blocks = read_draine_qext_table(args.carbon_table)
    sil_blocks = read_draine_qext_table(args.silicate_table)

    rv0, rvinfo0 = rv_samples(
        cg10_dir, snap0, cg10_series[snap0],
        carbon_blocks, sil_blocks,
        args.pixel_pkpc, args.rv_radius_pkpc, args.rv_zmax_pkpc,
        args.min_dust_count, args.min_signal_fraction,
        args.orient_radius_pkpc
    )
    rv6, rvinfo6 = rv_samples(
        cg10_dir, snap6, cg10_series[snap6],
        carbon_blocks, sil_blocks,
        args.pixel_pkpc, args.rv_radius_pkpc, args.rv_zmax_pkpc,
        args.min_dust_count, args.min_signal_fraction,
        args.orient_radius_pkpc
    )

    if len(rv0) == 0 or len(rv6) == 0:
        raise RuntimeError(
            f"No valid R(V) pixels: z0={len(rv0)}, z6={len(rv6)}. "
            "Try --min-dust-count 1 or a lower --min-signal-fraction."
        )

    rv_hi = max(
        8.0,
        float(np.nanpercentile(np.r_[rv0, rv6], 99.0))
    )
    rv_hi = min(rv_hi, 20.0)  # keep extreme numerical tails from owning plot
    rv_edges = np.linspace(1.5, rv_hi, 36)
    rvpdf0, _ = pdf_hist(rv0, rv_edges)
    rvpdf6, _ = pdf_hist(rv6, rv_edges)
    rvcent = 0.5 * (rv_edges[:-1] + rv_edges[1:])

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    axA, axB, axC, axD = axes.flat

    # A: suite GSD
    axA.plot(centers, gsd0, lw=2.2, label=f"z=0 (snap {snap0:03d})")
    axA.plot(centers, gsd6, lw=2.2, label=f"z≈6 (snap {snap6:03d})")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("Grain radius (nm)")
    axA.set_ylabel(r"$(1/M_{\rm dust})\,dM_{\rm dust}/d\log_{10}a$")
    axA.set_title("A. Suite-integrated grain-size distribution")
    axA.legend()

    # Reference injection scales.
    axA.axvline(10.0, ls=":", lw=1.0)
    axA.axvline(100.0, ls=":", lw=1.0)
    ymax = axA.get_ylim()[1]
    axA.text(10.0, ymax/1.6, "SNII\n10 nm", ha="center", va="top", fontsize=8)
    axA.text(100.0, ymax/1.6, "AGB\n100 nm", ha="center", va="top", fontsize=8)

    # B: ratio
    goodratio = (gsd0 > 0) & (gsd6 > 0)
    ratio = np.full_like(gsd0, np.nan)
    ratio[goodratio] = gsd0[goodratio] / gsd6[goodratio]
    axB.plot(centers, ratio, lw=2.2)
    axB.axhline(1.0, ls="--", lw=1.0)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("Grain radius (nm)")
    axB.set_ylabel("Normalized GSD ratio, z=0 / z≈6")
    axB.set_title("B. Evolution of the suite-integrated GSD")

    # C: RV PDFs
    axC.plot(rvcent, rvpdf0, lw=2.2,
             label=f"z=0, N={len(rv0):,} pixels")
    axC.plot(rvcent, rvpdf6, lw=2.2,
             label=f"z≈6, N={len(rv6):,} pixels")
    axC.set_xlabel(r"$R(V)$")
    axC.set_ylabel("Probability density")
    axC.set_title("C. CG10 extinction-law distribution")
    axC.legend()

    # D: CG10 GSD
    axD.plot(centers, cg10_gsd[snap0], lw=2.2, label="z=0")
    axD.plot(centers, cg10_gsd[snap6], lw=2.2, label="z≈6 progenitor")
    axD.set_xscale("log")
    axD.set_yscale("log")
    axD.set_xlabel("Grain radius (nm)")
    axD.set_ylabel(r"$(1/M_{\rm dust})\,dM_{\rm dust}/d\log_{10}a$")
    axD.set_title("D. CG10 grain-size distribution")
    axD.legend()

    fig.suptitle(
        "CosmicGrain: dust grain-size and extinction-law evolution",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    out = Path(args.output)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # Save machine-readable products
    # ------------------------------------------------------------
    npz = out.with_suffix(".npz")
    np.savez_compressed(
        npz,
        gsd_edges_nm=edges,
        gsd_centers_nm=centers,
        suite_gsd_z0=gsd0,
        suite_gsd_z6=gsd6,
        suite_gsd_mass_z0_msun=gsd0_mass,
        suite_gsd_mass_z6_msun=gsd6_mass,
        suite_total_dust_z0_msun=md0,
        suite_total_dust_z6_msun=md6,
        cg10_gsd_z0=cg10_gsd[snap0],
        cg10_gsd_z6=cg10_gsd[snap6],
        cg10_rv_z0=rv0,
        cg10_rv_z6=rv6,
        cg10_rv_edges=rv_edges,
        cg10_rv_pdf_z0=rvpdf0,
        cg10_rv_pdf_z6=rvpdf6,
        cg10_rotation_z0=rvinfo0["rotation"],
        cg10_rotation_z6=rvinfo6["rotation"],
    )

    csv_path = out.with_name(out.stem + "_halo_summary.csv")
    with open(csv_path, "w", newline="") as f:
        fields = [
            "cg", "halo_id", "snap", "z", "group_idx",
            "m200_msun", "r200_pkpc", "n_dust_r200", "mdust_r200_msun"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"Suite halos contributing at z=0 : {len(suite_r[snap0])}")
    print(f"Suite halos contributing at z~6 : {len(suite_r[snap6])}")
    print(f"Suite dust mass z=0             : {md0:.6e} Msun")
    print(f"Suite dust mass z~6             : {md6:.6e} Msun")
    print()
    print(
        f"CG10 z=0:  z={rvinfo0['z']:.4f}, R200={rvinfo0['r200_pkpc']:.2f} pkpc, "
        f"orientation={rvinfo0['orientation_source']} "
        f"(N={rvinfo0['n_orientation_particles']}), "
        f"RV pixels={len(rv0)}, median={np.median(rv0):.3f}, "
        f"p16-p84={np.percentile(rv0,16):.3f}-{np.percentile(rv0,84):.3f}"
    )
    print(
        f"CG10 z~6:  z={rvinfo6['z']:.4f}, R200={rvinfo6['r200_pkpc']:.2f} pkpc, "
        f"orientation={rvinfo6['orientation_source']} "
        f"(N={rvinfo6['n_orientation_particles']}), "
        f"RV pixels={len(rv6)}, median={np.median(rv6):.3f}, "
        f"p16-p84={np.percentile(rv6,16):.3f}-{np.percentile(rv6,84):.3f}"
    )
    print()
    print(f"Wrote {out}")
    print(f"Wrote {pdf}")
    print(f"Wrote {npz}")
    print(f"Wrote {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
