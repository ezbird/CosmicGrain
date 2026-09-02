#!/usr/bin/env python3
"""
plot_gas_and_instantaneous_sf_faceon.py

Make a direct face-on gas diagnostic for Halo 569 using the exact center,
rotation matrix, footprint, and pixel scale stored in an existing R_V map NPZ.

Outputs:
  1) Gas surface density Sigma_gas
  2) Instantaneous star-forming gas surface density Sigma_gas,SF
  3) Instantaneous SFR surface density Sigma_SFR
  4) Binary map of pixels containing gas with instantaneous SFR > 0

This diagnostic is intentionally independent of the R_V validity mask, so it
tests whether the compact star-forming footprint is intrinsic to the gas/SF
distribution rather than imposed by the dust map selection.

Example
-------
python3 plot_gas_and_instantaneous_sf_faceon.py \
  '../simulation_runs_used_for_paper/S10_output_2048/snapdir_020/snapshot_020.*.hdf5' \
  halo569_snap020_rv_sfr_500pc_maps.npz \
  --prefix halo569_snap020_gas_sf_diagnostic
"""

import argparse
import glob
import re
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


SFR_FIELD_CANDIDATES = (
    "StarFormationRate",
    "StarFormationRates",
    "SFR",
)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def snapshot_files(pattern):
    files = sorted(glob.glob(pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No snapshot files match: {pattern}")
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

        if "HubbleParam" in pars:
            h = float(pars["HubbleParam"])
        elif "HubbleParam" in hdr:
            h = float(hdr["HubbleParam"])
        else:
            raise RuntimeError("HubbleParam not found")

    return a, z, box, h


def discover_sfr_field(files):
    fields = set()
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType0" in f:
                fields.update(f["PartType0"].keys())

    for name in SFR_FIELD_CANDIDATES:
        if name in fields:
            return name, sorted(fields)

    return None, sorted(fields)


def read_gas(files, sfr_field):
    coords, mass, sfr = [], [], []

    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType0" not in f:
                continue

            g = f["PartType0"]
            for field in ("Coordinates", "Masses", sfr_field):
                if field not in g:
                    raise RuntimeError(
                        f"{fn}: missing PartType0/{field}"
                    )

            coords.append(g["Coordinates"][:].astype(np.float64))
            mass.append(g["Masses"][:].astype(np.float64))
            sfr.append(g[sfr_field][:].astype(np.float64))

    if not coords:
        raise RuntimeError("No PartType0 gas found")

    return (
        np.concatenate(coords),
        np.concatenate(mass),
        np.concatenate(sfr),
    )


def faceon_positions(coords, center, rotation, box, a, h):
    d = periodic_delta(coords, center, box) * a / h
    return d @ rotation.T


def safe_lognorm(arrays):
    vals = []
    for arr in arrays:
        x = arr[np.isfinite(arr) & (arr > 0)]
        if x.size:
            vals.append(x)

    if not vals:
        return None

    v = np.concatenate(vals)
    vmin = np.nanpercentile(v, 5)
    vmax = np.nanpercentile(v, 99)

    if not np.isfinite(vmin) or vmin <= 0:
        vmin = np.nanmin(v[v > 0])
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.nanmax(v)

    if vmax <= vmin:
        vmax = vmin * 1.01

    return LogNorm(vmin=vmin, vmax=vmax)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("snapshot")
    ap.add_argument(
        "maps_npz",
        help="NPZ from halo569_rv_sfr_triptych.py"
    )
    ap.add_argument(
        "--sfr-field",
        default=None,
        help="Override PartType0 instantaneous SFR field name"
    )
    ap.add_argument(
        "--prefix",
        default="gas_sf_faceon_diagnostic"
    )

    args = ap.parse_args()

    files = snapshot_files(args.snapshot)
    a, z, box, h = snapshot_metadata(files[0])

    d = np.load(args.maps_npz)
    center = d["center"]
    rotation = d["rotation"]
    centers = d["centers_pkpc"]

    pixel_size = float(np.asarray(d["pixel_size_pkpc"]))
    plot_radius = float(np.asarray(d["plot_radius_pkpc"]))
    zmax = float(np.asarray(d["zmax_pkpc"]))

    if args.sfr_field is None:
        sfr_field, fields = discover_sfr_field(files)
        if sfr_field is None:
            print("No recognized instantaneous gas SFR field found.")
            print("Available PartType0 fields:")
            for name in fields:
                print(" ", name)
            sys.exit(2)
    else:
        sfr_field = args.sfr_field

    print(f"Snapshot: a={a:.8f}, z={z:.8f}")
    print(f"Using SFR field: PartType0/{sfr_field}")
    print(f"Map: +/-{plot_radius:g} pkpc, |z'|<{zmax:g} pkpc")
    print(f"Pixel size: {pixel_size:g} pkpc")

    coords, masses, sfr = read_gas(files, sfr_field)

    xyz = faceon_positions(
        coords, center, rotation, box, a, h
    )
    x, y, zz = xyz.T

    finite = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(zz)
        & np.isfinite(masses) & np.isfinite(sfr)
    )

    inmap = (
        finite
        & (np.abs(x) < plot_radius)
        & (np.abs(y) < plot_radius)
        & (np.abs(zz) < zmax)
    )

    sf = inmap & (sfr > 0)

    print(f"Gas particles total: {len(masses):,}")
    print(f"Gas particles in map volume: {inmap.sum():,}")
    print(f"Gas particles with SFR > 0 in map volume: {sf.sum():,}")

    # Current gas masses in Msun.
    mass_msun = masses * 1.0e10 / h

    nbins = len(centers)
    edges = np.linspace(
        -plot_radius, plot_radius, nbins + 1
    )

    gas_mass_map, _, _ = np.histogram2d(
        y[inmap], x[inmap],
        bins=(edges, edges),
        weights=mass_msun[inmap]
    )

    sf_gas_mass_map, _, _ = np.histogram2d(
        y[sf], x[sf],
        bins=(edges, edges),
        weights=mass_msun[sf]
    )

    sfr_map, _, _ = np.histogram2d(
        y[sf], x[sf],
        bins=(edges, edges),
        weights=sfr[sf]
    )

    sf_count_map, _, _ = np.histogram2d(
        y[sf], x[sf],
        bins=(edges, edges)
    )
    sf_count_map = sf_count_map.astype(np.int64)

    gas_count_map, _, _ = np.histogram2d(
        y[inmap], x[inmap],
        bins=(edges, edges)
    )
    gas_count_map = gas_count_map.astype(np.int64)

    pixel_area = pixel_size**2

    sigma_gas = gas_mass_map / pixel_area
    sigma_sf_gas = sf_gas_mass_map / pixel_area
    sigma_sfr = sfr_map / pixel_area
    sf_present = sf_count_map > 0

    print(f"Pixels containing any gas: {(gas_count_map > 0).sum():,}")
    print(f"Pixels containing SFR > 0 gas: {sf_present.sum():,}")
    print(f"Total gas mass in map: {gas_mass_map.sum():.6e} Msun")
    print(f"Star-forming gas mass in map: {sf_gas_mass_map.sum():.6e} Msun")
    print(f"Instantaneous SFR in map: {sfr_map.sum():.6f} Msun/yr")

    if gas_mass_map.sum() > 0:
        frac = sf_gas_mass_map.sum() / gas_mass_map.sum()
        print(f"Fraction of gas mass that is star-forming: {frac:.6f}")

    # Radial extent diagnostics.
    rproj = np.sqrt(x**2 + y**2)

    if np.any(sf):
        print(
            "Projected radius of SFR>0 gas: "
            f"min={rproj[sf].min():.3f}, "
            f"median={np.median(rproj[sf]):.3f}, "
            f"p90={np.percentile(rproj[sf],90):.3f}, "
            f"max={rproj[sf].max():.3f} pkpc"
        )

    gas_nonzero = inmap & (mass_msun > 0)
    if np.any(gas_nonzero):
        print(
            "Projected radius of all gas in map: "
            f"median={np.median(rproj[gas_nonzero]):.3f}, "
            f"p90={np.percentile(rproj[gas_nonzero],90):.3f}, "
            f"max={rproj[gas_nonzero].max():.3f} pkpc"
        )

    extent = [
        -plot_radius, plot_radius,
        -plot_radius, plot_radius
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.2, 10.2),
        constrained_layout=True
    )

    # Panel 1: total gas surface density.
    norm_gas = safe_lognorm([sigma_gas])
    im0 = axes[0,0].imshow(
        np.ma.masked_where(sigma_gas <= 0, sigma_gas),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        norm=norm_gas,
        aspect="equal"
    )
    cb0 = fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.03)
    cb0.set_label(
        r"$\Sigma_{\rm gas}$ [$M_\odot\,{\rm kpc}^{-2}$]"
    )
    axes[0,0].set_title("All gas")

    # Panel 2: star-forming gas surface density.
    norm_sfg = safe_lognorm([sigma_sf_gas])
    im1 = axes[0,1].imshow(
        np.ma.masked_where(sigma_sf_gas <= 0, sigma_sf_gas),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        norm=norm_sfg,
        aspect="equal"
    )
    cb1 = fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.03)
    cb1.set_label(
        r"$\Sigma_{\rm gas,SF}$ [$M_\odot\,{\rm kpc}^{-2}$]"
    )
    axes[0,1].set_title(r"Gas with instantaneous SFR $>0$")

    # Panel 3: instantaneous SFR surface density.
    norm_sfr = safe_lognorm([sigma_sfr])
    im2 = axes[1,0].imshow(
        np.ma.masked_where(sigma_sfr <= 0, sigma_sfr),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        norm=norm_sfr,
        aspect="equal"
    )
    cb2 = fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.03)
    cb2.set_label(
        r"$\Sigma_{\rm SFR}$ "
        r"[$M_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2}$]"
    )
    axes[1,0].set_title("Instantaneous SFR surface density")

    # Panel 4: binary footprint + all-gas contours.
    axes[1,1].imshow(
        np.ma.masked_where(~sf_present, sf_present.astype(float)),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    positive_gas = sigma_gas[np.isfinite(sigma_gas) & (sigma_gas > 0)]
    if positive_gas.size:
        levels = np.unique(
            np.nanpercentile(positive_gas, [50, 75, 90])
        )
        if len(levels):
            axes[1,1].contour(
                0.5*(edges[:-1]+edges[1:]),
                0.5*(edges[:-1]+edges[1:]),
                sigma_gas,
                levels=levels,
                linewidths=1.0
            )

    axes[1,1].set_title(
        r"Binary SFR $>0$ footprint + gas contours"
    )

    for ax in axes.flat:
        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)
        ax.set_xlabel(r"$X_{\rm face-on}$ [pkpc]")
        ax.set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")
        ax.axhline(0, ls="--", lw=0.6, alpha=0.35)
        ax.axvline(0, ls="--", lw=0.6, alpha=0.35)

    fig.suptitle(
        rf"Halo 569 — $z={z:.3f}$ — "
        rf"{pixel_size*1000:.0f} pc pixels",
        fontsize=15
    )

    png = args.prefix + ".png"
    pdf = args.prefix + ".pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    npz_out = args.prefix + "_maps.npz"
    np.savez_compressed(
        npz_out,
        sigma_gas=sigma_gas,
        sigma_sf_gas=sigma_sf_gas,
        sigma_sfr=sigma_sfr,
        sf_present=sf_present,
        gas_count=gas_count_map,
        sf_gas_count=sf_count_map,
        centers_pkpc=centers,
        pixel_size_pkpc=pixel_size,
        plot_radius_pkpc=plot_radius,
        zmax_pkpc=zmax,
        snapshot_redshift=z,
        center=center,
        rotation=rotation,
        sfr_field=sfr_field,
    )

    txt = args.prefix + "_summary.txt"
    with open(txt, "w") as f:
        f.write(f"snapshot_redshift = {z:.8f}\n")
        f.write(f"sfr_field = {sfr_field}\n")
        f.write(f"gas_particles_total = {len(masses)}\n")
        f.write(f"gas_particles_in_map = {int(inmap.sum())}\n")
        f.write(f"sf_gas_particles_in_map = {int(sf.sum())}\n")
        f.write(f"gas_pixels = {int((gas_count_map>0).sum())}\n")
        f.write(f"sf_pixels = {int(sf_present.sum())}\n")
        f.write(f"gas_mass_map_Msun = {gas_mass_map.sum():.10e}\n")
        f.write(f"sf_gas_mass_map_Msun = {sf_gas_mass_map.sum():.10e}\n")
        f.write(f"instantaneous_sfr_Msun_per_yr = {sfr_map.sum():.10e}\n")
        if gas_mass_map.sum() > 0:
            f.write(
                "sf_gas_mass_fraction = "
                f"{sf_gas_mass_map.sum()/gas_mass_map.sum():.10e}\n"
            )
        if np.any(sf):
            f.write(
                "sf_gas_projected_radius_pkpc "
                f"min={rproj[sf].min():.8f} "
                f"median={np.median(rproj[sf]):.8f} "
                f"p90={np.percentile(rproj[sf],90):.8f} "
                f"max={rproj[sf].max():.8f}\n"
            )

    print("Saved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", npz_out)
    print(" ", txt)


if __name__ == "__main__":
    main()
