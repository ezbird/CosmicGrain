#!/usr/bin/env python3
"""
Audit the effective spatial sampling/resolution of the CG10 extinction field.

Measures, inside the same face-on comparison volume used for the ZG comparison:
  1) Dust 3-D neighbour distances d1, d8, d32.
  2) Gas SmoothingLength (if present).
  3) Unweighted and approximate A_V-weighted percentiles of those scales.

The A_V weight for each dust particle uses the same mixture convention as the
CosmicGrain R(V) work:
    weight_V ∝ M_dust/a * [f_C/rho_C Qext_C(V)
                         +(1-f_C)/rho_S Qext_S(V)]
Only relative weights matter for the weighted percentiles.

Coordinates:
  snapshot Coordinates: comoving kpc/h
  physical pkpc = delta_coord * a / h
HubbleParam is read from Parameters (not Header).

Example
-------
python3 scripts/cg10_extinction_resolution_audit.py \
    --output-dir halo441_output_1024_mingashsml0.0 \
    --snapshot 29 \
    --group 1 \
    --carbon-table scripts/dust_optics/callqcomp.out_D16MGemt \
    --silicate-table scripts/dust_optics/callqcomp.out_DH16asil \
    --prefix cg10_extinction_resolution

Requires scipy for cKDTree.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit("This script requires scipy (scipy.spatial.cKDTree).") from exc

try:
    from halo_utils import get_zoom_halo
except ImportError:
    get_zoom_halo = None

RHO_C = 2.24
RHO_SIL = 3.50
V_MICRON = 0.55


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="halo441_output_1024_mingashsml0.0")
    p.add_argument("--snapshot", type=int, default=29)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--radius", type=float, default=2.5,
                   help="Cylindrical radius in physical kpc.")
    p.add_argument("--zmax", type=float, default=0.4,
                   help="Absolute face-on height in physical kpc.")
    p.add_argument("--orientation-radius", type=float, default=20.0,
                   help="Radius for stellar angular-momentum orientation [pkpc].")
    p.add_argument("--pixel", type=float, default=0.25,
                   help="Comparison pixel size [kpc], diagnostic only.")
    p.add_argument("--assumed-sigma", type=float, default=0.5,
                   help="Previously assumed Gaussian sigma [kpc], diagnostic only.")
    p.add_argument("--carbon-table",
                   default="scripts/dust_optics/callqcomp.out_D16MGemt")
    p.add_argument("--silicate-table",
                   default="scripts/dust_optics/callqcomp.out_DH16asil")
    p.add_argument("--prefix", default="cg10_extinction_resolution")
    return p.parse_args()


def snapshot_files(output_dir, snap):
    od = Path(output_dir)
    pats = [
        od / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.*.hdf5",
        od / f"snapshot_{snap:03d}.*.hdf5",
        od / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.hdf5",
        od / f"snapshot_{snap:03d}.hdf5",
    ]
    for pat in pats:
        fs = sorted(glob.glob(str(pat)))
        if fs:
            return fs
    raise FileNotFoundError(f"Could not find snapshot {snap:03d} under {output_dir}")


def header_info(files):
    with h5py.File(files[0], "r") as f:
        a = float(f["Header"].attrs["Time"])
        box = float(f["Header"].attrs["BoxSize"])
        h = float(f["Parameters"].attrs["HubbleParam"])
    return a, h, box


def read_ptype(files, ptype, fields):
    out = {x: [] for x in fields}
    for fn in files:
        with h5py.File(fn, "r") as f:
            gname = f"PartType{ptype}"
            if gname not in f:
                continue
            g = f[gname]
            n = len(g["Coordinates"])
            for fld in fields:
                if fld in g:
                    out[fld].append(np.asarray(g[fld]))
                else:
                    out[fld].append(None)
    result = {}
    for fld, chunks in out.items():
        good = [x for x in chunks if x is not None]
        result[fld] = np.concatenate(good, axis=0) if good else None
    return result


def periodic_delta(x, center, box):
    d = np.asarray(x, float) - np.asarray(center, float)
    d -= box * np.rint(d / box)
    return d


def rotation_from_angular_momentum(pos, vel, mass):
    mt = np.sum(mass)
    v0 = np.sum(vel * mass[:, None], axis=0) / mt
    L = np.sum(np.cross(pos, vel - v0) * mass[:, None], axis=0)
    Lnorm = np.linalg.norm(L)
    if not np.isfinite(Lnorm) or Lnorm <= 0:
        raise RuntimeError("Could not determine angular-momentum orientation.")
    zhat = L / Lnorm
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, zhat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    xhat = trial - np.dot(trial, zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    # Rows are new basis vectors, so p_rot = p @ R.T
    return np.vstack([xhat, yhat, zhat])


def get_center(output_dir, snap, group):
    """Return the halo centre in snapshot/code coordinates.

    Current halo_utils.get_zoom_halo() returns a HaloReference object, while
    older versions returned dict-like objects. Support both APIs.
    """
    if get_zoom_halo is None:
        raise RuntimeError(
            "Could not import halo_utils.get_zoom_halo. Run from ~/gadget4 "
            "with this script in scripts/, or make halo_utils importable."
        )

    halo = get_zoom_halo(output_dir, snap, group_index=group)

    # Current HaloReference API first; retain compatibility with older names.
    for key in ("chosen_center_ckpch", "refined_center_ckpch", "catalog_center_ckpch", "center", "center_code", "centre", "pos"):
        if hasattr(halo, key):
            value = getattr(halo, key)
            if value is not None:
                return np.asarray(value, dtype=float), halo

    # Backward compatibility with older dict-returning halo_utils versions.
    if isinstance(halo, dict):
        for key in ("chosen_center_ckpch", "refined_center_ckpch", "catalog_center_ckpch", "center", "center_code", "centre", "pos"):
            if key in halo and halo[key] is not None:
                return np.asarray(halo[key], dtype=float), halo
        available = list(halo.keys())
    else:
        available = [
            name for name in dir(halo)
            if not name.startswith("_") and not callable(getattr(halo, name, None))
        ]

    raise KeyError(
        "Could not find a halo-centre field on the object returned by "
        f"get_zoom_halo(). Available fields/attributes: {available}"
    )


def get_mass_array(data, n):
    m = data.get("Masses")
    if m is None:
        return np.ones(n, dtype=float)
    return np.asarray(m, float)


def determine_orientation(files, center, a, h, box, radius):
    # Prefer stars, matching the existing comparison pipeline.
    for ptype, label, min_n in ((4, "stars", 20), (0, "gas", 50)):
        d = read_ptype(files, ptype, ["Coordinates", "Velocities", "Masses"])
        if d["Coordinates"] is None or d["Velocities"] is None:
            continue
        pos = periodic_delta(d["Coordinates"], center, box) * a / h
        rr = np.linalg.norm(pos, axis=1)
        q = rr <= radius
        if np.count_nonzero(q) < min_n:
            continue
        mass = get_mass_array(d, len(pos))
        R = rotation_from_angular_momentum(pos[q], d["Velocities"][q], mass[q])
        return R, label, int(np.count_nonzero(q))
    raise RuntimeError("Insufficient stars/gas to determine face-on orientation.")


def parse_optics_table(path):
    """
    Parse Draine-style callqcomp output into radius -> wavelength/Qext rows.
    Returns sorted unique radii and, for each radius, wavelength and Qext arrays.
    """
    radii = []
    waves = []
    qexts = []
    current_r = None
    cur_w, cur_q = [], []

    def flush():
        nonlocal cur_w, cur_q, current_r
        if current_r is not None and cur_w:
            radii.append(float(current_r))
            waves.append(np.asarray(cur_w, float))
            qexts.append(np.asarray(cur_q, float))
        cur_w, cur_q = [], []

    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if "= radius(micron)" in s:
                flush()
                try:
                    current_r = float(s.split()[0])
                except Exception:
                    current_r = None
                continue
            if current_r is None:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                w = float(parts[0])
                qe = float(parts[1])
            except ValueError:
                continue
            cur_w.append(w)
            cur_q.append(qe)
    flush()

    if not radii:
        raise RuntimeError(f"Could not parse optical table: {path}")
    order = np.argsort(radii)
    return np.asarray(radii)[order], [waves[i] for i in order], [qexts[i] for i in order]


def qext_at(table, radius_um, wavelength_um):
    radii, waves, qexts = table
    # First interpolate Qext in wavelength for every tabulated radius,
    # then interpolate in log radius.
    q_r = np.empty(len(radii), float)
    for i, (w, q) in enumerate(zip(waves, qexts)):
        o = np.argsort(w)
        q_r[i] = np.interp(wavelength_um, w[o], q[o])
    lr = np.log(radii)
    x = np.log(np.clip(radius_um, radii.min(), radii.max()))
    return np.interp(x, lr, q_r)


def weighted_percentile(x, w, percentiles=(16, 50, 84)):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    q = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[q], w[q]
    if len(x) == 0:
        return np.full(len(percentiles), np.nan)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    c = (c - 0.5*w) / np.sum(w)
    return np.interp(np.asarray(percentiles)/100.0, c, x)


def plain_percentile(x, ps=(16, 50, 84)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, ps) if len(x) else np.full(len(ps), np.nan)


def fmt(v):
    return " ".join(f"{x:.4f}" for x in v)


def main():
    aargs = parse_args()
    files = snapshot_files(aargs.output_dir, aargs.snapshot)
    aa, h, box = header_info(files)
    zred = 1.0/aa - 1.0

    center, halo = get_center(aargs.output_dir, aargs.snapshot, aargs.group)
    R, orient_label, orient_n = determine_orientation(
        files, center, aa, h, box, aargs.orientation_radius
    )

    dust = read_ptype(
        files, 6,
        ["Coordinates", "Masses", "GrainRadius", "CarbonMassFraction"]
    )
    if dust["Coordinates"] is None:
        raise RuntimeError("No PartType6 dust found.")
    if dust["GrainRadius"] is None:
        raise RuntimeError("PartType6/GrainRadius is required.")
    if dust["CarbonMassFraction"] is None:
        raise RuntimeError("PartType6/CarbonMassFraction is required.")

    dpos = periodic_delta(dust["Coordinates"], center, box) * aa / h
    dpos = dpos @ R.T
    dcyl = np.hypot(dpos[:, 0], dpos[:, 1])
    dq = (dcyl <= aargs.radius) & (np.abs(dpos[:, 2]) <= aargs.zmax)
    dp = dpos[dq]
    dm = get_mass_array(dust, len(dpos))[dq] * 1.0e10 / h
    dr_nm = np.asarray(dust["GrainRadius"], float)[dq]
    fc = np.clip(np.asarray(dust["CarbonMassFraction"], float)[dq], 0.0, 1.0)

    if len(dp) < 33:
        raise RuntimeError(f"Only {len(dp)} dust particles in audit volume; need >=33.")

    print(f"Snapshot {aargs.snapshot:03d}: a={aa:.8f} z={zred:.6g} h={h:.6g}")
    print(f"Orientation: {orient_label}, N={orient_n:,}")
    print(f"Audit volume: R<{aargs.radius:g} pkpc, |z|<{aargs.zmax:g} pkpc")
    print(f"Dust particles in audit volume: {len(dp):,}")

    # Dust neighbour scales. Query k=33 because result column 0 is the particle itself.
    tree = cKDTree(dp)
    dist, _ = tree.query(dp, k=33, workers=-1)
    d1 = dist[:, 1]
    d8 = dist[:, 8]
    d32 = dist[:, 32]

    # Approximate V-band extinction contribution of each dust particle.
    ct = parse_optics_table(aargs.carbon_table)
    st = parse_optics_table(aargs.silicate_table)
    r_um = dr_nm * 1.0e-3
    qC = qext_at(ct, r_um, V_MICRON)
    qS = qext_at(st, r_um, V_MICRON)
    # Constants common to every particle are intentionally omitted.
    avw = (dm / np.maximum(r_um, 1e-30)) * (
        fc / RHO_C * qC + (1.0 - fc) / RHO_SIL * qS
    )
    avw[~np.isfinite(avw) | (avw < 0)] = 0.0

    gas = read_ptype(files, 0, ["Coordinates", "SmoothingLength", "Masses"])
    gh = None
    ghw = None
    ngas = 0
    if gas["Coordinates"] is not None and gas["SmoothingLength"] is not None:
        gpos = periodic_delta(gas["Coordinates"], center, box) * aa / h
        gpos = gpos @ R.T
        gcyl = np.hypot(gpos[:, 0], gpos[:, 1])
        gq = (gcyl <= aargs.radius) & (np.abs(gpos[:, 2]) <= aargs.zmax)
        # SmoothingLength has the same comoving kpc/h length conversion.
        gh = np.asarray(gas["SmoothingLength"], float)[gq] * aa / h
        ngas = len(gh)

        # For an extinction-relevance weighting of gas h, assign each dust particle
        # to its nearest gas particle, then sum dust A_V weights onto that gas particle.
        gp = gpos[gq]
        if len(gp) and len(dp):
            gt = cKDTree(gp)
            _, gi = gt.query(dp, k=1, workers=-1)
            ghw = np.bincount(gi, weights=avw, minlength=len(gp))

    ps = (16, 50, 84)
    results = {}

    print("\nSpatial-scale percentiles [pkpc]: P16 P50 P84")
    for name, arr in (("dust d1", d1), ("dust d8", d8), ("dust d32", d32)):
        pu = plain_percentile(arr, ps)
        pw = weighted_percentile(arr, avw, ps)
        results[name] = {"unweighted": pu.tolist(), "Av_weighted": pw.tolist()}
        print(f"{name:10s} unweighted: {fmt(pu)}")
        print(f"{'':10s} A_V-weighted: {fmt(pw)}")

    if gh is not None and len(gh):
        pu = plain_percentile(gh, ps)
        results["gas h"] = {"unweighted": pu.tolist()}
        print(f"{'gas h':10s} unweighted: {fmt(pu)}")
        if ghw is not None and np.any(ghw > 0):
            pw = weighted_percentile(gh, ghw, ps)
            results["gas h"]["Av_weighted_nearest_dust"] = pw.tolist()
            print(f"{'':10s} A_V-weighted: {fmt(pw)}")
    else:
        print("gas h: PartType0/SmoothingLength unavailable in this snapshot.")

    print(f"\nReference pixel size = {aargs.pixel:.3f} kpc")
    print(f"Previously assumed Gaussian sigma = {aargs.assumed_sigma:.3f} kpc")
    print("Do not equate dN or SPH h directly to Gaussian sigma; compare them as sampling scales.")

    # Plot cumulative distributions; weighted CDFs are normalized A_V contribution.
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, arr in (("dust d1", d1), ("dust d8", d8), ("dust d32", d32)):
        s = np.sort(arr[np.isfinite(arr)])
        y = np.arange(1, len(s)+1)/len(s)
        ax.plot(s, y, label=label)
    if gh is not None and len(gh):
        s = np.sort(gh[np.isfinite(gh)])
        y = np.arange(1, len(s)+1)/len(s)
        ax.plot(s, y, label="gas h")
    ax.axvline(aargs.pixel, ls="--", lw=1.5, label=f"{aargs.pixel:g} kpc pixel")
    ax.axvline(aargs.assumed_sigma, ls=":", lw=1.8, label=f"{aargs.assumed_sigma:g} kpc assumed σ")
    ax.set_xscale("log")
    ax.set_xlabel("Physical scale (kpc)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CG10 spatial sampling in ZG comparison volume")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{aargs.prefix}_cdf.png", dpi=220)
    fig.savefig(f"{aargs.prefix}_cdf.pdf")
    plt.close(fig)

    # Extinction-weighted dust-scale CDF.
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, arr in (("dust d1", d1), ("dust d8", d8), ("dust d32", d32)):
        q = np.isfinite(arr) & np.isfinite(avw) & (avw > 0)
        x, w = arr[q], avw[q]
        o = np.argsort(x)
        x, w = x[o], w[o]
        y = np.cumsum(w)/np.sum(w)
        ax.plot(x, y, label=label)
    if gh is not None and ghw is not None and np.any(ghw > 0):
        q = np.isfinite(gh) & np.isfinite(ghw) & (ghw > 0)
        x, w = gh[q], ghw[q]
        o = np.argsort(x)
        x, w = x[o], w[o]
        y = np.cumsum(w)/np.sum(w)
        ax.plot(x, y, label="gas h (nearest-dust weighted)")
    ax.axvline(aargs.pixel, ls="--", lw=1.5, label=f"{aargs.pixel:g} kpc pixel")
    ax.axvline(aargs.assumed_sigma, ls=":", lw=1.8, label=f"{aargs.assumed_sigma:g} kpc assumed σ")
    ax.set_xscale("log")
    ax.set_xlabel("Physical scale (kpc)")
    ax.set_ylabel("Cumulative fraction of V-band extinction weight")
    ax.set_title("CG10 extinction-weighted spatial sampling")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{aargs.prefix}_Av_weighted_cdf.png", dpi=220)
    fig.savefig(f"{aargs.prefix}_Av_weighted_cdf.pdf")
    plt.close(fig)

    summary = {
        "snapshot": aargs.snapshot,
        "redshift": zred,
        "group": aargs.group,
        "radius_pkpc": aargs.radius,
        "zmax_pkpc": aargs.zmax,
        "n_dust": int(len(dp)),
        "n_gas_with_h": int(ngas),
        "orientation": orient_label,
        "orientation_particles": orient_n,
        "reference_pixel_kpc": aargs.pixel,
        "previous_assumed_sigma_kpc": aargs.assumed_sigma,
        "percentiles": list(ps),
        "scales_pkpc": results,
        "note": (
            "Dust extinction weights are relative V-band extinction contributions. "
            "Gas A_V weighting assigns each dust particle's weight to its nearest gas "
            "particle. Particle neighbour distances and SPH smoothing lengths are not "
            "Gaussian sigma values."
        ),
    }
    with open(f"{aargs.prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(
        f"{aargs.prefix}_products.npz",
        dust_d1_pkpc=d1,
        dust_d8_pkpc=d8,
        dust_d32_pkpc=d32,
        dust_Av_weight=avw,
        gas_h_pkpc=np.asarray([] if gh is None else gh),
        gas_Av_weight=np.asarray([] if ghw is None else ghw),
        pixel_kpc=aargs.pixel,
        assumed_sigma_kpc=aargs.assumed_sigma,
    )

    print(f"\nWrote {aargs.prefix}_summary.json")
    print(f"Wrote {aargs.prefix}_cdf.png/.pdf")
    print(f"Wrote {aargs.prefix}_Av_weighted_cdf.png/.pdf")
    print(f"Wrote {aargs.prefix}_products.npz")


if __name__ == "__main__":
    main()
