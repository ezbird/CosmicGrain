#!/usr/bin/env python3
"""
cosmicgrain_rv_maps.py

Make standardized face-on R_V, dust-property, and recent-SF maps for CosmicGrain zoom halos:
  1) R_V map
  2) recent SFR surface density from stars younger than 30 Myr
  3) recent SFR surface density from stars younger than 100 Myr

The script:
  * reads a split GADGET-4 snapshot
  * uses a supplied halo center
  * computes the stellar bulk velocity inside 20 pkpc
  * computes the stellar angular-momentum vector inside 20 pkpc
  * constructs a face-on rotation matrix from that angular momentum
  * applies the SAME center/orientation/grid to dust and stellar maps
  * computes R_V from Draine graphite + amorphous-silicate Q_ext tables
  * uses 500 pc pixels by default
  * saves PNG, PDF, NPZ, CSV, center, and rotation matrix

Recent SFR proxy:
    Sigma_SFR = Mstar,current(age < Delta_t) / (Delta_t * pixel_area)

Because stellar birth masses are not stored, this is an approximate recent-SF
surface-density proxy based on current young-star particle masses.

Example
-------
python3 cosmicgrain_rv_maps.py \
  '../simulation_runs_used_for_paper/S10_output_2048/snapdir_030/snapshot_030.*.hdf5' \
  --center 23150.3448030708 24545.5694831303 23840.7750336792 \
  --carbon-table dust_optics/callqcomp.out_D16MGemt \
  --silicate-table dust_optics/callqcomp.out_DH16asil \
  --plot-radius 20 --zmax 20 \
  --pixel-size 0.5 \
  --min-count 5 --signal-frac 1e-3 \
  --orientation-radius 20 \
  --prefix halo569_snap030_rv_sfr_500pc
"""

import argparse
import glob
import math
import re
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def snapshot_files(pattern):
    files = sorted(glob.glob(pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return files


def periodic_delta(coords, center, box):
    d = coords - center[None, :]
    d -= box * np.rint(d / box)
    return d


def snapshot_metadata(first_file):
    with h5py.File(first_file, "r") as f:
        hdr = f["Header"].attrs
        pars = f["Parameters"].attrs if "Parameters" in f else {}

        a = float(hdr.get("Time", 1.0))
        z = float(hdr.get("Redshift", 1.0 / a - 1.0))
        box = float(hdr["BoxSize"])

        def getp(name):
            if name in pars:
                return float(pars[name])
            if name in hdr:
                return float(hdr[name])
            return None

        h = getp("HubbleParam")
        om = getp("Omega0")
        ol = getp("OmegaLambda")

    if h is None or om is None or ol is None:
        raise RuntimeError("Missing HubbleParam/Omega0/OmegaLambda")

    return a, z, box, h, om, ol


def cosmic_time_flat_lcdm_gyr(a, h, om, ol):
    aa = np.asarray(a, dtype=float)
    out = np.zeros_like(aa, dtype=float)
    good = np.isfinite(aa) & (aa > 0)
    hubble_time_gyr = 9.778 / h

    if np.any(good):
        ag = aa[good]
        if ol > 0:
            out[good] = (
                2.0 / (3.0 * math.sqrt(ol))
                * np.arcsinh(math.sqrt(ol / om) * ag**1.5)
                * hubble_time_gyr
            )
        else:
            out[good] = (
                2.0 / (3.0 * math.sqrt(om))
                * ag**1.5
                * hubble_time_gyr
            )

    if np.ndim(a) == 0:
        return float(out)
    return out


def read_stars(files):
    coords, vel, mass, form = [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType4" not in f:
                continue
            g = f["PartType4"]
            coords.append(g["Coordinates"][:].astype(np.float64))
            vel.append(g["Velocities"][:].astype(np.float64))
            mass.append(g["Masses"][:].astype(np.float64))
            form.append(g["StellarFormationTime"][:].astype(np.float64))

    if not coords:
        raise RuntimeError("No PartType4 stars found")

    return (
        np.concatenate(coords),
        np.concatenate(vel),
        np.concatenate(mass),
        np.concatenate(form),
    )


def read_dust(files):
    coords, mass, radius, cf = [], [], [], []
    cf_field = None

    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType6" not in f:
                continue
            g = f["PartType6"]

            if "CarbonMassFraction" in g:
                name = "CarbonMassFraction"
            elif "CarbonFraction" in g:
                name = "CarbonFraction"
            else:
                raise RuntimeError("No carbon-fraction field in PartType6")

            if cf_field is None:
                cf_field = name

            coords.append(g["Coordinates"][:].astype(np.float64))
            mass.append(g["Masses"][:].astype(np.float64))
            radius.append(g["GrainRadius"][:].astype(np.float64))
            cf.append(g[name][:].astype(np.float64))

    if not coords:
        raise RuntimeError("No PartType6 dust found")

    print(f"Dust carbon field: {cf_field}")

    return (
        np.concatenate(coords),
        np.concatenate(mass),
        np.concatenate(radius),
        np.concatenate(cf),
    )


def stellar_orientation(coords, vel, mass, center, box, a, h, radius_pkpc):
    """
    Compute mass-weighted stellar bulk velocity and angular momentum.
    Coordinates are converted to physical kpc.

    For orientation, only the direction of L matters; the exact overall
    velocity-unit normalization cancels.
    """
    d = periodic_delta(coords, center, box) * a / h
    r = np.linalg.norm(d, axis=1)
    sel = r < radius_pkpc

    if sel.sum() < 10:
        raise RuntimeError(
            f"Only {sel.sum()} stars inside orientation radius"
        )

    w = mass[sel]
    bulk = np.average(vel[sel], axis=0, weights=w)
    dv = vel[sel] - bulk

    L = np.sum(
        w[:, None] * np.cross(d[sel], dv),
        axis=0
    )
    norm = np.linalg.norm(L)
    if norm <= 0:
        raise RuntimeError("Stellar angular momentum is zero")

    zhat = L / norm

    # Build a stable right-handed basis.
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(zhat, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    xhat = np.cross(trial, zhat)
    xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    yhat /= np.linalg.norm(yhat)

    R = np.vstack([xhat, yhat, zhat])

    return bulk, zhat, R, int(sel.sum())


def faceon_positions(coords, center, rotation, box, a, h):
    d = periodic_delta(coords, center, box) * a / h
    return d @ rotation.T


def read_qext_table(path):
    radii = []
    blocks = []
    current_r = None
    current = []

    radius_re = re.compile(
        r"^\s*([0-9.Ee+\-]+)\s*=\s*radius",
        re.IGNORECASE
    )

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = radius_re.search(line)
            if m:
                if current_r is not None and current:
                    radii.append(current_r)
                    blocks.append(np.asarray(current, dtype=float))
                current_r = float(m.group(1))
                current = []
                continue

            if current_r is None:
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    lam = float(parts[0])
                    qext = float(parts[1])
                    current.append((lam, qext))
                except ValueError:
                    pass

    if current_r is not None and current:
        radii.append(current_r)
        blocks.append(np.asarray(current, dtype=float))

    if not radii:
        raise RuntimeError(f"Could not parse optical table: {path}")

    wave = blocks[0][:, 0]
    q = np.empty((len(blocks), len(wave)), dtype=float)

    for i, b in enumerate(blocks):
        if (
            len(b) == len(wave)
            and np.allclose(b[:, 0], wave, rtol=1e-10, atol=0)
        ):
            q[i] = b[:, 1]
        else:
            order = np.argsort(b[:, 0])
            q[i] = np.interp(wave, b[order, 0], b[order, 1])

    radii = np.asarray(radii)
    ir = np.argsort(radii)
    iw = np.argsort(wave)

    return radii[ir], wave[iw], q[ir][:, iw]


def q_at_lambda(radii, wave, q, wavelength):
    return np.array(
        [np.interp(wavelength, wave, row) for row in q]
    )


def interp_q_radius(grain_um, radii_um, qvals):
    out = np.full(grain_um.shape, np.nan, dtype=float)
    good = (
        np.isfinite(grain_um)
        & (grain_um >= radii_um.min())
        & (grain_um <= radii_um.max())
    )
    out[good] = np.interp(
        np.log10(grain_um[good]),
        np.log10(radii_um),
        qvals
    )
    return out


def build_rv_map(
    coords, masses, radius_nm, cf,
    center, rotation, box, a, h,
    plot_radius, zmax, pixel_size,
    carbon_table, silicate_table,
    rho_c=2.24, rho_s=3.50
):
    rc, wc, qc = read_qext_table(carbon_table)
    rs, ws, qs = read_qext_table(silicate_table)

    qcb_t = q_at_lambda(rc, wc, qc, 0.44)
    qcv_t = q_at_lambda(rc, wc, qc, 0.55)
    qsb_t = q_at_lambda(rs, ws, qs, 0.44)
    qsv_t = q_at_lambda(rs, ws, qs, 0.55)

    xyz = faceon_positions(
        coords, center, rotation, box, a, h
    )
    x, y, z = xyz.T

    sel = (
        (np.abs(x) < plot_radius)
        & (np.abs(y) < plot_radius)
        & (np.abs(z) < zmax)
    )

    x = x[sel]
    y = y[sel]
    m = masses[sel]
    r_um = radius_nm[sel] * 1e-3
    f = np.clip(cf[sel], 0, 1)

    print(f"Dust in face-on volume: {len(x):,}")

    qcb = interp_q_radius(r_um, rc, qcb_t)
    qcv = interp_q_radius(r_um, rc, qcv_t)
    qsb = interp_q_radius(r_um, rs, qsb_t)
    qsv = interp_q_radius(r_um, rs, qsv_t)

    good = (
        np.isfinite(qcb) & np.isfinite(qcv)
        & np.isfinite(qsb) & np.isfinite(qsv)
        & np.isfinite(m) & (m > 0)
        & np.isfinite(r_um) & (r_um > 0)
    )

    x, y, m, r_um, f = (
        x[good], y[good], m[good], r_um[good], f[good]
    )
    qcb, qcv, qsb, qsv = (
        qcb[good], qcv[good], qsb[good], qsv[good]
    )

    wb = (m / r_um) * (
        f * qcb / rho_c
        + (1.0 - f) * qsb / rho_s
    )
    wv = (m / r_um) * (
        f * qcv / rho_c
        + (1.0 - f) * qsv / rho_s
    )

    nbins = int(round(2 * plot_radius / pixel_size))
    actual_pixel = 2 * plot_radius / nbins
    edges = np.linspace(-plot_radius, plot_radius, nbins + 1)

    tau_b, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=wb
    )
    tau_v, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=wv
    )
    counts, _, _ = np.histogram2d(
        y, x, bins=(edges, edges)
    )
    counts = counts.astype(np.int64)

    # Dust-property maps on exactly the same particles/grid as R_V.
    # Masses are in code units here; ratios/weighted means are independent
    # of the overall mass-unit conversion.
    dust_mass, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=m
    )
    radius_mass_sum, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=m * (r_um * 1.0e3)
    )
    carbon_mass_sum, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=m * f
    )

    # Define "small" as a < 30 nm by default for this diagnostic.
    # This threshold is stored in the NPZ so it is explicit/reproducible.
    small_grain_threshold_nm = 30.0
    small = (r_um * 1.0e3) < small_grain_threshold_nm
    small_mass, _, _ = np.histogram2d(
        y, x, bins=(edges, edges), weights=m * small
    )

    mean_radius_mw_nm = np.full_like(dust_mass, np.nan, dtype=float)
    carbon_fraction_mw = np.full_like(dust_mass, np.nan, dtype=float)
    small_mass_fraction = np.full_like(dust_mass, np.nan, dtype=float)
    has_mass = dust_mass > 0
    mean_radius_mw_nm[has_mass] = (
        radius_mass_sum[has_mass] / dust_mass[has_mass]
    )
    carbon_fraction_mw[has_mass] = (
        carbon_mass_sum[has_mass] / dust_mass[has_mass]
    )
    small_mass_fraction[has_mass] = (
        small_mass[has_mass] / dust_mass[has_mass]
    )

    denom = tau_b - tau_v
    rv = np.full_like(tau_v, np.nan, dtype=float)

    goodrv = (
        np.isfinite(tau_b) & np.isfinite(tau_v)
        & (tau_v > 0) & (denom > 0)
    )
    rv[goodrv] = tau_v[goodrv] / denom[goodrv]

    centers = 0.5 * (edges[:-1] + edges[1:])

    return (
        rv, tau_b, tau_v, counts, centers, actual_pixel,
        dust_mass, mean_radius_mw_nm, small_mass_fraction,
        carbon_fraction_mw, small_grain_threshold_nm
    )


def star_ages_myr(formation_a, a_snap, h, om, ol):
    age = np.full(formation_a.shape, np.nan)
    good = (
        np.isfinite(formation_a)
        & (formation_a > 0)
        & (formation_a <= a_snap)
    )

    t_snap = cosmic_time_flat_lcdm_gyr(
        a_snap, h, om, ol
    )
    if np.any(good):
        tform = cosmic_time_flat_lcdm_gyr(
            formation_a[good], h, om, ol
        )
        age[good] = (t_snap - tform) * 1000.0

    return age, t_snap


def sfr_map(
    coords, masses, ages_myr,
    center, rotation, box, a, h,
    plot_radius, zmax, pixel_size,
    age_limit_myr
):
    xyz = faceon_positions(
        coords, center, rotation, box, a, h
    )
    x, y, z = xyz.T

    sel = (
        np.isfinite(ages_myr)
        & (ages_myr >= 0)
        & (ages_myr < age_limit_myr)
        & (np.abs(x) < plot_radius)
        & (np.abs(y) < plot_radius)
        & (np.abs(z) < zmax)
    )

    nbins = int(round(2 * plot_radius / pixel_size))
    actual_pixel = 2 * plot_radius / nbins
    edges = np.linspace(-plot_radius, plot_radius, nbins + 1)

    # Standard GADGET mass unit here: 1e10 Msun/h.
    mass_msun = masses * 1e10 / h

    massmap, _, _ = np.histogram2d(
        y[sel], x[sel],
        bins=(edges, edges),
        weights=mass_msun[sel]
    )
    countmap, _, _ = np.histogram2d(
        y[sel], x[sel],
        bins=(edges, edges)
    )
    countmap = countmap.astype(np.int64)

    sigma = (
        massmap
        / (age_limit_myr * 1e6)
        / (actual_pixel**2)
    )

    return sigma, massmap, countmap, sel


def make_triptych(
    rv, rv_mask,
    sfr30, sfr100,
    plot_radius,
    pixel_size,
    z_snap,
    prefix,
    rv_vmin=None,
    rv_vmax=None
):
    extent = [
        -plot_radius, plot_radius,
        -plot_radius, plot_radius
    ]

    if rv_vmin is None:
        rv_vmin = np.nanpercentile(rv[rv_mask], 2)
    if rv_vmax is None:
        rv_vmax = np.nanpercentile(rv[rv_mask], 98)

    fig, axes = plt.subplots(
        1, 3, figsize=(16.2, 5.4),
        constrained_layout=True
    )

    rvplot = np.ma.masked_where(~rv_mask, rv)
    im0 = axes[0].imshow(
        rvplot,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="RdYlBu_r",
        vmin=rv_vmin,
        vmax=rv_vmax,
        aspect="equal"
    )
    cb0 = fig.colorbar(
        im0, ax=axes[0], fraction=0.046, pad=0.03
    )
    cb0.set_label(r"$R_V$")
    axes[0].set_title(r"$R_V$")

    # Use one common SFR log normalization for both age windows.
    positive = np.concatenate([
        sfr30[np.isfinite(sfr30) & (sfr30 > 0)],
        sfr100[np.isfinite(sfr100) & (sfr100 > 0)]
    ])

    if positive.size:
        vmin = positive.min()
        vmax = positive.max()
        if vmax <= vmin:
            vmax = 1.01 * vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for ax, arr, title in zip(
        axes[1:],
        [sfr30, sfr100],
        [
            r"$\Sigma_{\rm SFR}$: stars $<30$ Myr",
            r"$\Sigma_{\rm SFR}$: stars $<100$ Myr"
        ]
    ):
        masked = np.ma.masked_where(
            ~(np.isfinite(arr) & (arr > 0)),
            arr
        )
        im = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="magma",
            norm=norm,
            aspect="equal"
        )
        ax.set_title(title)

    if positive.size:
        cb = fig.colorbar(
            im, ax=axes[1:].tolist(),
            fraction=0.046, pad=0.025
        )
        cb.set_label(
            r"$\Sigma_{\rm SFR}$ "
            r"[$M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2}$]"
        )

    for ax in axes:
        ax.set_xlabel(r"$X_{\rm face-on}$ [pkpc]")
        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)
        ax.axhline(0, ls="--", lw=0.7, alpha=0.5)
        ax.axvline(0, ls="--", lw=0.7, alpha=0.5)

        x0 = -plot_radius + 2
        y0 = -plot_radius + 2
        ax.plot([x0, x0 + 5], [y0, y0], lw=3)
        ax.text(
            x0 + 2.5, y0 + 0.7,
            "5 kpc",
            ha="center", va="bottom", fontsize=9
        )

    axes[0].set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")
    axes[1].set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")
    axes[2].set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")

    fig.suptitle(
        rf"CosmicGrain — $z={z_snap:.3f}$ — "
        rf"{pixel_size*1000:.0f} pc pixels",
        fontsize=15
    )

    png = prefix + "_triptych.png"
    pdf = prefix + "_triptych.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument(
        "--center", nargs=3, type=float, required=True,
        metavar=("X", "Y", "Z")
    )
    ap.add_argument("--carbon-table", required=True)
    ap.add_argument("--silicate-table", required=True)
    ap.add_argument("--plot-radius", type=float, default=25.0)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--pixel-size", type=float, default=0.25)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--signal-frac", type=float, default=1e-3)
    ap.add_argument("--orientation-radius", type=float, default=20.0)
    ap.add_argument("--rv-vmin", type=float, default=None)
    ap.add_argument("--rv-vmax", type=float, default=None)
    ap.add_argument(
        "--prefix",
        default="cosmicgrain_rv_master"
    )

    args = ap.parse_args()

    files = snapshot_files(args.snapshot)
    center = np.asarray(args.center, dtype=float)

    a, z, box, h, om, ol = snapshot_metadata(files[0])

    print(
        f"Snapshot: a={a:.8f}, z={z:.8f}, "
        f"BoxSize={box:.6f}, h={h:.6f}"
    )
    print("Input center:", center)

    sxyz, svel, smass, sform = read_stars(files)
    print(f"Star particles: {len(smass):,}")

    bulk, jhat, rotation, norient = stellar_orientation(
        sxyz, svel, smass,
        center, box, a, h,
        args.orientation_radius
    )

    print(
        f"\nStellar orientation inside "
        f"{args.orientation_radius:g} pkpc:"
    )
    print(f"  Nstars = {norient:,}")
    print("  bulk velocity =", bulk)
    print("  j_hat =", jhat)
    print("  rotation matrix:")
    print(rotation)

    dxyz, dmass, drad, dcf = read_dust(files)
    print(f"Dust particles: {len(dmass):,}")
    print(
        f"Dust radius range: "
        f"{np.nanmin(drad):.4f}-{np.nanmax(drad):.4f} nm"
    )
    print(
        f"Dust CF range: "
        f"{np.nanmin(dcf):.4f}-{np.nanmax(dcf):.4f}"
    )

    (
        rv, tau_b, tau_v, dcount, centers, actual_pixel,
        dust_mass_map, mean_grain_radius_mw_nm,
        small_grain_mass_fraction, carbon_mass_fraction_mw,
        small_grain_threshold_nm
    ) = build_rv_map(
        dxyz, dmass, drad, dcf,
        center, rotation, box, a, h,
        args.plot_radius, args.zmax, args.pixel_size,
        args.carbon_table, args.silicate_table
    )

    finite = np.isfinite(rv)
    peak_tauv = np.nanmax(tau_v[np.isfinite(tau_v)])
    sig_thresh = args.signal_frac * peak_tauv

    rv_mask = (
        finite
        & (dcount >= args.min_count)
        & (tau_v >= sig_thresh)
    )

    print("\nR(V) map:")
    print(f"  finite R(V) pixels = {finite.sum():,}")
    print(
        f"  N_dust >= {args.min_count} pixels = "
        f"{(dcount >= args.min_count).sum():,}"
    )
    print(f"  peak tau_V = {peak_tauv:.8e}")
    print(f"  signal threshold = {sig_thresh:.8e}")
    print(f"  final valid pixels = {rv_mask.sum():,}")

    if rv_mask.sum() == 0:
        raise RuntimeError("No valid R(V) pixels survive mask")

    p16, med, p84 = np.nanpercentile(
        rv[rv_mask], [16, 50, 84]
    )
    print(
        f"  R(V) p16/median/p84 = "
        f"{p16:.4f} / {med:.4f} / {p84:.4f}"
    )

    print("\nDust-property maps (valid R(V) pixels):")
    print(
        f"  small-grain definition = a < "
        f"{small_grain_threshold_nm:g} nm"
    )
    for label, arr in [
        ("mass-weighted grain radius [nm]", mean_grain_radius_mw_nm),
        ("small-grain dust-mass fraction", small_grain_mass_fraction),
        ("mass-weighted CarbonMassFraction", carbon_mass_fraction_mw),
    ]:
        goodprop = rv_mask & np.isfinite(arr)
        if np.any(goodprop):
            q16, q50, q84 = np.nanpercentile(arr[goodprop], [16, 50, 84])
            print(
                f"  {label}: p16/median/p84 = "
                f"{q16:.5g} / {q50:.5g} / {q84:.5g}"
            )

    ages, cosmic_age = star_ages_myr(
        sform, a, h, om, ol
    )

    sfr30, m30, n30map, sel30 = sfr_map(
        sxyz, smass, ages,
        center, rotation, box, a, h,
        args.plot_radius, args.zmax, args.pixel_size,
        30.0
    )
    sfr100, m100, n100map, sel100 = sfr_map(
        sxyz, smass, ages,
        center, rotation, box, a, h,
        args.plot_radius, args.zmax, args.pixel_size,
        100.0
    )

    print("\nRecent star formation:")
    print(f"  cosmic age = {cosmic_age:.4f} Gyr")
    print(f"  N(<30 Myr) in map volume = {sel30.sum():,}")
    print(f"  N(<100 Myr) in map volume = {sel100.sum():,}")
    print(
        f"  SFR30 proxy = "
        f"{m30.sum()/(30e6):.6f} Msun/yr"
    )
    print(
        f"  SFR100 proxy = "
        f"{m100.sum()/(100e6):.6f} Msun/yr"
    )
    print(
        f"  occupied SF pixels: "
        f"30 Myr={np.count_nonzero(sfr30>0)}, "
        f"100 Myr={np.count_nonzero(sfr100>0)}"
    )

    # Save center + orientation for reuse.
    np.savetxt(
        args.prefix + "_center.txt",
        center.reshape(1, 3),
        fmt="%.12f"
    )
    np.savetxt(
        args.prefix + "_rotation_matrix.txt",
        rotation,
        fmt="%.12f"
    )
    np.savetxt(
        args.prefix + "_stellar_jhat.txt",
        jhat.reshape(1, 3),
        fmt="%.12f"
    )

    npz = args.prefix + "_maps.npz"
    np.savez_compressed(
        npz,
        rv=rv,
        rv_mask=rv_mask,
        tau_b=tau_b,
        tau_v=tau_v,
        dust_count=dcount,
        dust_mass_code=dust_mass_map,
        mean_grain_radius_massweighted_nm=mean_grain_radius_mw_nm,
        small_grain_mass_fraction=small_grain_mass_fraction,
        carbon_mass_fraction_massweighted=carbon_mass_fraction_mw,
        small_grain_threshold_nm=small_grain_threshold_nm,
        sigma_sfr_30=sfr30,
        sigma_sfr_100=sfr100,
        young_mass_30=m30,
        young_mass_100=m100,
        young_count_30=n30map,
        young_count_100=n100map,
        centers_pkpc=centers,
        pixel_size_pkpc=actual_pixel,
        plot_radius_pkpc=args.plot_radius,
        zmax_pkpc=args.zmax,
        center=center,
        rotation=rotation,
        stellar_jhat=jhat,
        snapshot_scale_factor=a,
        snapshot_redshift=z,
        signal_fraction=args.signal_frac,
        signal_threshold_tau_v=sig_thresh,
        min_dust_count=args.min_count,
    )

    csv = args.prefix + "_pixel_table.csv"
    xgrid, ygrid = np.meshgrid(centers, centers)
    with open(csv, "w") as f:
        f.write(
            "iy,ix,x_pkpc,y_pkpc,Rv,Rv_valid,tauB,tauV,"
            "dust_count,DustMassCode,MeanGrainRadiusMassWeighted_nm,"
            "SmallGrainMassFraction,CarbonMassFractionMassWeighted,"
            "SigmaSFR30,SigmaSFR100,Nstar30,Nstar100\n"
        )
        for iy in range(rv.shape[0]):
            for ix in range(rv.shape[1]):
                f.write(
                    f"{iy},{ix},"
                    f"{xgrid[iy,ix]:.8f},{ygrid[iy,ix]:.8f},"
                    f"{rv[iy,ix]:.10e},{int(rv_mask[iy,ix])},"
                    f"{tau_b[iy,ix]:.10e},{tau_v[iy,ix]:.10e},"
                    f"{dcount[iy,ix]},"
                    f"{dust_mass_map[iy,ix]:.10e},"
                    f"{mean_grain_radius_mw_nm[iy,ix]:.10e},"
                    f"{small_grain_mass_fraction[iy,ix]:.10e},"
                    f"{carbon_mass_fraction_mw[iy,ix]:.10e},"
                    f"{sfr30[iy,ix]:.10e},"
                    f"{sfr100[iy,ix]:.10e},"
                    f"{n30map[iy,ix]},{n100map[iy,ix]}\n"
                )

    png, pdf = make_triptych(
        rv, rv_mask,
        sfr30, sfr100,
        args.plot_radius,
        actual_pixel,
        z,
        args.prefix,
        rv_vmin=args.rv_vmin,
        rv_vmax=args.rv_vmax
    )

    print("\nSaved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", npz)
    print(" ", csv)
    print(" ", args.prefix + "_center.txt")
    print(" ", args.prefix + "_rotation_matrix.txt")
    print(" ", args.prefix + "_stellar_jhat.txt")


if __name__ == "__main__":
    main()
