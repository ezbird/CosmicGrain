#!/usr/bin/env python3
"""Audit MinGasHsml behavior in CosmicGrain/GADGET-4 snapshots.

Reconstruct the smoothing length associated with the target effective
neighbour number,

    N_ngb = (4*pi/3) h^3 rho / m,

and compare it with the configured/stored MinGasHsml floor.

The script supports either:
  1) explicit --center-file and --rotation-file, or
  2) automatic centering with halo_utils.get_zoom_halo plus a stellar
     angular-momentum orientation.

HubbleParam is read from Parameters, matching CosmicGrain snapshots.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from halo_utils import get_zoom_halo, periodic_delta

NORM_COEFF = 4.188790204786
DEFAULT_DESNGB = 32.0
DEFAULT_DEV = 8.0
DEFAULT_MIN_HSML = 1.0
PCTS = (16, 50, 84)


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit the effect of MinGasHsml in a CosmicGrain snapshot."
    )
    p.add_argument("snapshot", help="Snapshot glob, e.g. '../out/snapdir_029/snapshot_029.*.hdf5'")
    p.add_argument("--center-file")
    p.add_argument("--rotation-file")
    p.add_argument("--output-dir", help="Required for automatic halo centering")
    p.add_argument("--group-index", type=int, help="FOF group index for automatic centering")
    p.add_argument("--orientation-radius", type=float, default=20.0, help="pkpc")
    p.add_argument("--param-file")
    p.add_argument("--label", default="CosmicGrain")
    p.add_argument("--m200", type=float, default=np.nan, help="Optional M200c [Msun]")
    p.add_argument("--prefix", default="mingashsml_audit")
    p.add_argument("--galaxy-radius", type=float, default=30.0)
    p.add_argument("--galaxy-zmax", type=float, default=5.0)
    p.add_argument("--disk-radius", type=float, default=20.0)
    p.add_argument("--disk-zmax", type=float, default=2.0)
    p.add_argument("--inner-radius", type=float, default=5.0)
    p.add_argument("--inner-zmax", type=float, default=2.0)
    p.add_argument("--candidate-floors", nargs="*", type=float, default=[1.0, 0.5, 0.25],
                   help="Candidate MinGasHsml values in code length units")
    return p.parse_args()


def read_param_file(path):
    vals = {}
    if not path:
        return vals
    with open(path) as f:
        for raw in f:
            line = raw.split("%", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            try:
                vals[key] = float(val)
            except ValueError:
                pass
    return vals


def snapshot_number(files):
    for fn in files:
        m = re.search(r"snapshot_(\d+)", os.path.basename(fn))
        if m:
            return int(m.group(1))
    raise RuntimeError("Could not determine snapshot number from snapshot filenames")


def load_snapshot(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files matched: {pattern}")

    coords, masses, density, hsml, sfr = [], [], [], [], []
    star_coords, star_masses, star_vel = [], [], []

    with h5py.File(files[0], "r") as f:
        a = float(f["Header"].attrs["Time"])
        box = float(f["Header"].attrs["BoxSize"])
        h = float(f["Parameters"].attrs["HubbleParam"])

    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType0" in f:
                g = f["PartType0"]
                coords.append(np.asarray(g["Coordinates"], dtype=float))
                masses.append(np.asarray(g["Masses"], dtype=float))
                density.append(np.asarray(g["Density"], dtype=float))
                hsml.append(np.asarray(g["SmoothingLength"], dtype=float))
                if "StarFormationRate" in g:
                    sfr.append(np.asarray(g["StarFormationRate"], dtype=float))
                elif "SFR" in g:
                    sfr.append(np.asarray(g["SFR"], dtype=float))
                else:
                    sfr.append(np.zeros(len(g["Coordinates"]), dtype=float))

            if "PartType4" in f:
                s = f["PartType4"]
                star_coords.append(np.asarray(s["Coordinates"], dtype=float))
                star_masses.append(np.asarray(s["Masses"], dtype=float))
                star_vel.append(np.asarray(s["Velocities"], dtype=float))

    if not coords:
        raise RuntimeError("No PartType0 gas particles found")

    def cat(x):
        return np.concatenate(x, axis=0) if x else None

    return {
        "files": files,
        "a": a,
        "box": box,
        "hubble": h,
        "coords": cat(coords),
        "mass": cat(masses),
        "density": cat(density),
        "hsml": cat(hsml),
        "sfr": cat(sfr),
        "star_coords": cat(star_coords),
        "star_mass": cat(star_masses),
        "star_vel": cat(star_vel),
    }


def make_rotation_matrix(j):
    j = np.asarray(j, dtype=float)
    norm = np.linalg.norm(j)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid stellar angular momentum vector")
    zhat = j / norm
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ zhat) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    xhat = np.cross(ref, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    return np.vstack((xhat, yhat, zhat))


def automatic_orientation(data, center, radius):
    if data["star_coords"] is None:
        raise RuntimeError("No PartType4 stars available for automatic orientation")

    pos = periodic_delta(data["star_coords"], center, data["box"])
    pos = pos * data["a"] / data["hubble"]
    mass = data["star_mass"]
    vel = data["star_vel"]
    r = np.linalg.norm(pos, axis=1)

    q = ((r < radius) & np.isfinite(r) & np.isfinite(mass) & (mass > 0)
         & np.all(np.isfinite(vel), axis=1))
    n = int(q.sum())
    if n < 3:
        raise RuntimeError(f"Only {n} valid stars inside {radius:g} pkpc")

    bulk = np.average(vel[q], axis=0, weights=mass[q])
    J = np.sum(np.cross(pos[q], vel[q] - bulk) * mass[q, None], axis=0)
    jnorm = np.linalg.norm(J)
    if not np.isfinite(jnorm) or jnorm <= 0:
        raise RuntimeError("Invalid stellar angular momentum")
    return make_rotation_matrix(J), J / jnorm, n


def get_geometry(args, data):
    if args.center_file:
        center = np.loadtxt(args.center_file, dtype=float).reshape(3)
        center_source = f"file: {args.center_file}"
    else:
        if args.output_dir is None or args.group_index is None:
            raise ValueError(
                "Provide --center-file, or both --output-dir and --group-index for automatic centering"
            )
        snap = snapshot_number(data["files"])
        halo = get_zoom_halo(
            args.output_dir, snap, group_index=args.group_index,
            refine_center=True, verbose=False
        )
        center = np.asarray(halo.chosen_center_ckpch, dtype=float)
        center_source = (
            f"halo_utils: group={halo.group_index}, "
            f"refinement_accepted={halo.refinement_accepted}"
        )

    if args.rotation_file:
        rotation = np.loadtxt(args.rotation_file, dtype=float).reshape(3, 3)
        jhat = rotation[2].copy()
        nstars = -1
        orientation_source = f"file: {args.rotation_file}"
    else:
        rotation, jhat, nstars = automatic_orientation(
            data, center, args.orientation_radius
        )
        orientation_source = (
            f"stellar angular momentum within {args.orientation_radius:g} pkpc"
        )
    return center, rotation, jhat, nstars, center_source, orientation_source


def percentile3(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(x, PCTS))


def fmt(x, nd=4):
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def safe_frac(q):
    return float(np.mean(q)) if len(q) else np.nan


def selection_stats(name, q, data, htarget, hlo, hhi, floor_phys, floor_hit,
                    candidate_floors, code_to_pkpc):
    idx = np.asarray(q, dtype=bool)
    ht = htarget[idx]
    stored_floor = floor_hit[idx]
    ratio = ht / floor_phys if floor_phys > 0 else np.full(len(ht), np.nan)

    out = {
        "selection": name,
        "n_gas": int(idx.sum()),
        "n_floor": int(stored_floor.sum()) if floor_phys > 0 else 0,
        "f_floor": safe_frac(stored_floor) if floor_phys > 0 else 0.0,
    }

    p16, p50, p84 = percentile3(ht)
    out.update(h_target_p16=p16, h_target_p50=p50, h_target_p84=p84)

    if floor_phys > 0:
        r16, r50, r84 = percentile3(ratio)
        out.update(ratio_p16=r16, ratio_p50=r50, ratio_p84=r84,
                   f_lt_075=safe_frac(ht < 0.75 * floor_phys),
                   f_lt_050=safe_frac(ht < 0.50 * floor_phys),
                   f_lt_025=safe_frac(ht < 0.25 * floor_phys))
    else:
        out.update(ratio_p16=np.nan, ratio_p50=np.nan, ratio_p84=np.nan,
                   f_lt_075=np.nan, f_lt_050=np.nan, f_lt_025=np.nan)

    if floor_phys > 0 and stored_floor.any():
        htf = ht[stored_floor]
        rf = htf / floor_phys
        fp16, fp50, fp84 = percentile3(htf)
        fr16, fr50, fr84 = percentile3(rf)
        lo50 = percentile3(hlo[idx][stored_floor])[1]
        hi50 = percentile3(hhi[idx][stored_floor])[1]
    else:
        fp16 = fp50 = fp84 = fr16 = fr50 = fr84 = lo50 = hi50 = np.nan

    out.update(
        floor_target_p16=fp16, floor_target_p50=fp50, floor_target_p84=fp84,
        floor_ratio_p16=fr16, floor_ratio_p50=fr50, floor_ratio_p84=fr84,
        allowed_lo_p50=lo50, allowed_hi_p50=hi50
    )

    print(f"\n{name}")
    print("-" * 78)
    print(f"N gas                         = {idx.sum():,}")
    print("Target h [pkpc]:")
    print(f"  p16={fmt(p16)}  p50={fmt(p50)}  p84={fmt(p84)}")

    if floor_phys > 0:
        print(f"N stored at floor             = {stored_floor.sum():,}")
        print(f"fraction stored at floor      = {fmt(safe_frac(stored_floor))}")
        print("h_target / physical floor:")
        print(f"  p16={fmt(out['ratio_p16'])}  p50={fmt(out['ratio_p50'])}  p84={fmt(out['ratio_p84'])}")
        print("Fraction whose target h is:")
        print(f"  < 0.75 h_floor : {fmt(out['f_lt_075'])}")
        print(f"  < 0.50 h_floor : {fmt(out['f_lt_050'])}")
        print(f"  < 0.25 h_floor : {fmt(out['f_lt_025'])}")
        if stored_floor.any():
            print("Particles STORED at MinGasHsml:")
            print(f"  target h [pkpc]: p16={fmt(fp16)}  p50={fmt(fp50)}  p84={fmt(fp84)}")
            print(f"  h_target / floor: p16={fmt(fr16)}  p50={fmt(fr50)}  p84={fmt(fr84)}")
            print(f"  median allowed-h range: {fmt(lo50)} -- {fmt(hi50)} pkpc")
    else:
        print("MinGasHsml = 0: no configured floor; stored-floor statistics are N/A.")

    print("Candidate-floor test from reconstructed target h:")
    for cf in candidate_floors:
        if cf <= 0:
            continue
        pf = cf * code_to_pkpc
        f1 = safe_frac(ht < pf)
        fhalf = safe_frac(ht < 0.5 * pf)
        print(
            f"  code {cf:g} ({pf:.4f} pkpc): "
            f"f(h_target < floor)={fmt(f1)}, "
            f"f(h_target < 0.5 floor)={fmt(fhalf)}"
        )
        out[f"candidate_{cf:g}_f_below"] = f1
        out[f"candidate_{cf:g}_f_below_half"] = fhalf

    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_figure(path, label, posrot, radius, density, ratio, floor_hit,
                floor_phys, htarget):
    if plt is None:
        print("WARNING: matplotlib unavailable; diagnostic figure skipped.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    x, y, z = posrot.T

    ax = axes[0, 0]
    q = (radius < 30) & (np.abs(z) < 5)
    ax.scatter(x[q], y[q], s=2, alpha=0.45)
    ax.set(xlabel="x (pkpc)", ylabel="y (pkpc)", title="Gas: R < 30 pkpc, |z| < 5 pkpc")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    qf = q & floor_hit
    if np.any(qf):
        c = htarget[qf] / floor_phys
        sc = ax.scatter(x[qf], y[qf], c=c, s=4)
        fig.colorbar(sc, ax=ax, label=r"$h_{\rm target}/h_{\rm floor}$")
    ax.set(xlabel="x (pkpc)", ylabel="y (pkpc)", title="Particles stored at floor")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    finite = q & np.isfinite(ratio)
    ax.scatter(radius[finite], ratio[finite], s=2, alpha=0.35)
    ax.axhline(1.0, lw=1)
    ax.set(xlabel="R (pkpc)", ylabel=r"$h_{\rm target}/h_{\rm floor}$",
           title="Floor severity versus radius")

    ax = axes[1, 1]
    finite = q & np.isfinite(ratio) & (density > 0)
    ax.scatter(density[finite], ratio[finite], s=2, alpha=0.35)
    ax.set_xscale("log")
    ax.axhline(1.0, lw=1)
    ax.set(xlabel="Density (code units)", ylabel=r"$h_{\rm target}/h_{\rm floor}$",
           title="Floor severity versus density")

    fig.suptitle(f"{label}: MinGasHsml audit", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    data = load_snapshot(args.snapshot)
    pars = read_param_file(args.param_file)

    desngb = float(pars.get("DesNumNgb", DEFAULT_DESNGB))
    dev = float(pars.get("MaxNumNgbDeviation", DEFAULT_DEV))
    min_hsml = float(pars.get("MinGasHsml", DEFAULT_MIN_HSML))

    if desngb <= 0 or dev < 0 or desngb <= dev:
        raise ValueError("Invalid DesNumNgb/MaxNumNgbDeviation")
    if min_hsml < 0:
        raise ValueError("MinGasHsml must be >= 0")

    center, rotation, jhat, nstars, center_source, orientation_source = get_geometry(
        args, data
    )

    code_to_pkpc = data["a"] / data["hubble"]
    floor_phys = min_hsml * code_to_pkpc

    d = periodic_delta(data["coords"], center, data["box"])
    pos = d * code_to_pkpc
    posrot = pos @ rotation.T
    R = np.hypot(posrot[:, 0], posrot[:, 1])
    z = posrot[:, 2]

    m = data["mass"]
    rho = data["density"]
    good = np.isfinite(m) & np.isfinite(rho) & (m > 0) & (rho > 0)

    htarget_code = np.full(len(m), np.nan)
    hlo_code = np.full(len(m), np.nan)
    hhi_code = np.full(len(m), np.nan)

    htarget_code[good] = (desngb * m[good] / (NORM_COEFF * rho[good])) ** (1.0 / 3.0)
    hlo_code[good] = ((desngb - dev) * m[good] / (NORM_COEFF * rho[good])) ** (1.0 / 3.0)
    hhi_code[good] = ((desngb + dev) * m[good] / (NORM_COEFF * rho[good])) ** (1.0 / 3.0)

    htarget = htarget_code * code_to_pkpc
    hlo = hlo_code * code_to_pkpc
    hhi = hhi_code * code_to_pkpc
    stored_phys = data["hsml"] * code_to_pkpc

    if min_hsml > 0:
        floor_hit = np.isclose(
            data["hsml"], min_hsml, rtol=1e-6,
            atol=max(1e-10, abs(min_hsml) * 1e-8)
        )
        ratio = htarget / floor_phys
    else:
        floor_hit = np.zeros(len(m), dtype=bool)
        ratio = np.full(len(m), np.nan)

    galaxy = good & (R < args.galaxy_radius) & (np.abs(z) < args.galaxy_zmax)
    disk = good & (R < args.disk_radius) & (np.abs(z) < args.disk_zmax)
    inner = good & (R < args.inner_radius) & (np.abs(z) < args.inner_zmax)
    sf = galaxy & np.isfinite(data["sfr"]) & (data["sfr"] > 0)

    print("=" * 78)
    print(f"{args.label} MinGasHsml audit")
    print("=" * 78)
    print(f"Snapshot files            = {len(data['files'])}")
    print(f"a                         = {data['a']:.8f}")
    print(f"HubbleParam               = {data['hubble']:.8f}")
    print(f"Kernel normalization      = {NORM_COEFF:.12f}")
    print(f"DesNumNgb                 = {desngb:g}")
    print(f"MaxNumNgbDeviation        = {dev:g}")
    print(f"allowed effective Ngb     = {desngb-dev:g} -- {desngb+dev:g}")
    print(f"MinGasHsml                = {min_hsml:g} code length")
    if min_hsml > 0:
        print(f"physical floor            = {floor_phys:.6f} pkpc")
    else:
        print("physical floor            = none")
    print(f"Center                    = {center}")
    print(f"Center source             = {center_source}")
    print(f"Orientation source        = {orientation_source}")
    print(f"j_hat                     = {jhat}")
    if nstars >= 0:
        print(f"Orientation stars         = {nstars:,}")

    rows = []
    rows.append(selection_stats(
        f"Galaxy: R < {args.galaxy_radius:g} pkpc, |z| < {args.galaxy_zmax:g} pkpc",
        galaxy, data, htarget, hlo, hhi, floor_phys, floor_hit,
        args.candidate_floors, code_to_pkpc
    ))
    rows.append(selection_stats(
        f"Disk: R < {args.disk_radius:g} pkpc, |z| < {args.disk_zmax:g} pkpc",
        disk, data, htarget, hlo, hhi, floor_phys, floor_hit,
        args.candidate_floors, code_to_pkpc
    ))
    rows.append(selection_stats(
        f"Inner disk: R < {args.inner_radius:g} pkpc, |z| < {args.inner_zmax:g} pkpc",
        inner, data, htarget, hlo, hhi, floor_phys, floor_hit,
        args.candidate_floors, code_to_pkpc
    ))
    rows.append(selection_stats(
        f"Star-forming gas: R < {args.galaxy_radius:g} pkpc, |z| < {args.galaxy_zmax:g} pkpc",
        sf, data, htarget, hlo, hhi, floor_phys, floor_hit,
        args.candidate_floors, code_to_pkpc
    ))

    print("\nRadial audit: |z| < 5 pkpc")
    print("-" * 78)
    print(f"{'R [pkpc]':>13} {'N':>8} {'f_floor':>10} {'ratio50':>10} {'N_SF':>8} {'f_floor,SF':>12}")
    radial_rows = []
    edges = [0, 2.5, 5, 10, 20, 30]
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = good & (R >= lo) & (R < hi) & (np.abs(z) < 5)
        qsf = q & (data["sfr"] > 0)
        ff = safe_frac(floor_hit[q]) if min_hsml > 0 and q.any() else (0.0 if q.any() else np.nan)
        rs = ratio[q]
        r50 = percentile3(rs)[1] if min_hsml > 0 else np.nan
        ffsf = safe_frac(floor_hit[qsf]) if min_hsml > 0 and qsf.any() else (0.0 if qsf.any() else np.nan)
        print(f"{lo:5.1f}-{hi:<5.1f} {q.sum():8d} {fmt(ff):>10} {fmt(r50):>10} {qsf.sum():8d} {fmt(ffsf):>12}")
        radial_rows.append({
            "r_lo_pkpc": lo, "r_hi_pkpc": hi, "n_gas": int(q.sum()),
            "f_floor": ff, "ratio50": r50, "n_sf": int(qsf.sum()),
            "f_floor_sf": ffsf
        })

    selection_csv = f"{args.prefix}_selections.csv"
    radial_csv = f"{args.prefix}_radial.csv"
    summary_csv = f"{args.prefix}_suite_row.csv"
    figure = f"{args.prefix}_diagnostic.png"

    write_csv(selection_csv, rows)
    write_csv(radial_csv, radial_rows)

    def row_named(prefix):
        for row in rows:
            if row["selection"].startswith(prefix):
                return row
        return {}

    rg = row_named("Galaxy:")
    rd = row_named("Disk:")
    ri = row_named("Inner disk:")
    rsf = row_named("Star-forming gas:")

    summary = {
        "label": args.label,
        "snapshot": snapshot_number(data["files"]),
        "a": data["a"],
        "z": 1.0 / data["a"] - 1.0,
        "hubble": data["hubble"],
        "m200_msun": args.m200,
        "desnumngb": desngb,
        "maxnumngbdeviation": dev,
        "mingashsml_code": min_hsml,
        "floor_pkpc": floor_phys if min_hsml > 0 else np.nan,
        "galaxy_n": rg.get("n_gas", 0),
        "galaxy_f_floor": rg.get("f_floor", np.nan),
        "galaxy_ratio50": rg.get("ratio_p50", np.nan),
        "disk_n": rd.get("n_gas", 0),
        "disk_f_floor": rd.get("f_floor", np.nan),
        "disk_ratio50": rd.get("ratio_p50", np.nan),
        "inner_n": ri.get("n_gas", 0),
        "inner_f_floor": ri.get("f_floor", np.nan),
        "inner_ratio50": ri.get("ratio_p50", np.nan),
        "sf_n": rsf.get("n_gas", 0),
        "sf_f_floor": rsf.get("f_floor", np.nan),
        "sf_ratio50": rsf.get("ratio_p50", np.nan),
    }
    for cf in args.candidate_floors:
        key = f"candidate_{cf:g}_f_below"
        summary[f"sf_f_target_below_code_{cf:g}"] = rsf.get(key, np.nan)
    write_csv(summary_csv, [summary])

    if min_hsml > 0:
        make_figure(
            figure, args.label, posrot, R, data["density"], ratio,
            floor_hit, floor_phys, htarget
        )
    else:
        print("\nDiagnostic ratio figure skipped for MinGasHsml=0 (no floor denominator).")
        figure = None

    print("\n" + "=" * 78)
    print("Outputs")
    print("=" * 78)
    print(f"Selection CSV : {selection_csv}")
    print(f"Radial CSV    : {radial_csv}")
    print(f"Suite row CSV : {summary_csv}")
    if figure:
        print(f"Figure        : {figure}")


if __name__ == "__main__":
    main()
