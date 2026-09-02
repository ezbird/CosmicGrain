#!/usr/bin/env python3
"""
plot_rv_instantaneous_sfr_gas_overlay.py

Overlay gas particles with instantaneous SFR > 0 on an existing CosmicGrain
face-on R_V map.

This script reuses the exact center, face-on rotation matrix, map footprint,
and pixel size stored in the NPZ produced by halo569_rv_sfr_triptych.py.

It reads PartType0 and looks for a gas instantaneous-SFR field using common
names:
    StarFormationRate
    SFR
    StarFormationRates

If none is present, the script prints all available PartType0 fields and exits
rather than guessing from density or temperature.

Two outputs are made:
  1) Binary overlay: pixels containing at least one gas particle with SFR > 0
  2) SFR-weighted overlay: same R_V map with instantaneous Sigma_SFR contours

Example
-------
python3 plot_rv_instantaneous_sfr_gas_overlay.py \
  '../simulation_runs_used_for_paper/S10_output_2048/snapdir_020/snapshot_020.*.hdf5' \
  halo569_snap020_rv_sfr_500pc_maps.npz \
  --prefix halo569_snap020_rv_instantaneous_sf_gas
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
            if "PartType0" not in f:
                continue
            fields.update(f["PartType0"].keys())

    for name in SFR_FIELD_CANDIDATES:
        if name in fields:
            return name, sorted(fields)

    return None, sorted(fields)


def read_gas(files, sfr_field):
    coords = []
    sfr = []
    mass = []

    for fn in files:
        with h5py.File(fn, "r") as f:
            if "PartType0" not in f:
                continue

            g = f["PartType0"]
            if sfr_field not in g:
                raise RuntimeError(
                    f"{fn}: PartType0/{sfr_field} missing in this file"
                )

            coords.append(g["Coordinates"][:].astype(np.float64))
            sfr.append(g[sfr_field][:].astype(np.float64))

            if "Masses" in g:
                mass.append(g["Masses"][:].astype(np.float64))

    if not coords:
        raise RuntimeError("No PartType0 gas particles found")

    coords = np.concatenate(coords)
    sfr = np.concatenate(sfr)
    masses = np.concatenate(mass) if mass else None

    return coords, sfr, masses


def faceon_positions(coords, center, rotation, box, a, h):
    d = periodic_delta(coords, center, box) * a / h
    return d @ rotation.T


def make_overlay(
    rv,
    rv_mask,
    xgrid,
    ygrid,
    sf_present,
    sigma_sfr,
    plot_radius,
    pixel_size,
    z,
    prefix,
    rv_vmin=None,
    rv_vmax=None,
):
    extent = [
        -plot_radius, plot_radius,
        -plot_radius, plot_radius
    ]

    if rv_vmin is None:
        rv_vmin = float(np.nanpercentile(rv[rv_mask], 2))
    if rv_vmax is None:
        rv_vmax = float(np.nanpercentile(rv[rv_mask], 98))

    # ------------------------------------------------------------------
    # Figure 1: binary instantaneous-SF footprint
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)

    rvplot = np.ma.masked_where(~rv_mask, rv)
    im = ax.imshow(
        rvplot,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="RdYlBu_r",
        vmin=rv_vmin,
        vmax=rv_vmax,
        aspect="equal"
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$R_V$")

    if np.any(sf_present):
        ax.scatter(
            xgrid[sf_present],
            ygrid[sf_present],
            s=105,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            label=r"Gas with instantaneous SFR $>0$"
        )

    ax.axhline(0, ls="--", lw=0.7, alpha=0.45)
    ax.axvline(0, ls="--", lw=0.7, alpha=0.45)

    x0 = -plot_radius + 1.1
    y0 = -plot_radius + 1.1
    ax.plot([x0, x0 + 5], [y0, y0], lw=3)
    ax.text(
        x0 + 2.5, y0 + 0.45, "5 kpc",
        ha="center", va="bottom", fontsize=9
    )

    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel(r"$X_{\rm face-on}$ [pkpc]")
    ax.set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")
    ax.set_title(
        rf"Halo 569 — $z={z:.3f}$ — "
        rf"{pixel_size*1000:.0f} pc pixels"
    )

    if np.any(sf_present):
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    binary_png = prefix + "_binary_overlay.png"
    binary_pdf = prefix + "_binary_overlay.pdf"
    fig.savefig(binary_png, dpi=260, bbox_inches="tight")
    fig.savefig(binary_pdf, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Sigma_SFR contours on R_V
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)

    im = ax.imshow(
        rvplot,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="RdYlBu_r",
        vmin=rv_vmin,
        vmax=rv_vmax,
        aspect="equal"
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$R_V$")

    positive = sigma_sfr[np.isfinite(sigma_sfr) & (sigma_sfr > 0)]

    if positive.size:
        # Use quantile-based contours so sparse, quantized star-forming gas
        # is still visible without implying an arbitrary physical threshold.
        levels = np.unique(
            np.nanpercentile(positive, [25, 50, 75])
        )

        if len(levels) > 0:
            cs = ax.contour(
                xgrid,
                ygrid,
                sigma_sfr,
                levels=levels,
                linewidths=1.5
            )
            ax.clabel(
                cs,
                inline=True,
                fontsize=8,
                fmt=lambda v: f"{v:.2g}"
            )

    ax.axhline(0, ls="--", lw=0.7, alpha=0.45)
    ax.axvline(0, ls="--", lw=0.7, alpha=0.45)

    ax.plot([x0, x0 + 5], [y0, y0], lw=3)
    ax.text(
        x0 + 2.5, y0 + 0.45, "5 kpc",
        ha="center", va="bottom", fontsize=9
    )

    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel(r"$X_{\rm face-on}$ [pkpc]")
    ax.set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")
    ax.set_title(
        rf"Halo 569 — instantaneous gas $\Sigma_{{\rm SFR}}$ contours"
    )

    contour_png = prefix + "_sfr_contours.png"
    contour_pdf = prefix + "_sfr_contours.pdf"
    fig.savefig(contour_png, dpi=260, bbox_inches="tight")
    fig.savefig(contour_pdf, bbox_inches="tight")
    plt.close(fig)

    return binary_png, binary_pdf, contour_png, contour_pdf


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
        help="Override PartType0 instantaneous-SFR field name"
    )
    ap.add_argument(
        "--prefix",
        default="rv_instantaneous_sf_gas"
    )
    ap.add_argument("--rv-vmin", type=float, default=None)
    ap.add_argument("--rv-vmax", type=float, default=None)

    args = ap.parse_args()

    files = snapshot_files(args.snapshot)
    a, z, box, h = snapshot_metadata(files[0])

    d = np.load(args.maps_npz)

    rv = d["rv"]
    rv_mask = d["rv_mask"].astype(bool)
    centers = d["centers_pkpc"]
    center = d["center"]
    rotation = d["rotation"]
    pixel_size = float(np.asarray(d["pixel_size_pkpc"]))
    plot_radius = float(np.asarray(d["plot_radius_pkpc"]))
    zmax = float(np.asarray(d["zmax_pkpc"]))

    if "snapshot_redshift" in d:
        z_npz = float(np.asarray(d["snapshot_redshift"]))
        if abs(z_npz - z) > 1e-4:
            raise RuntimeError(
                f"Snapshot z={z} does not match NPZ z={z_npz}"
            )

    if args.sfr_field is None:
        sfr_field, fields = discover_sfr_field(files)
        if sfr_field is None:
            print(
                "\nNo recognized instantaneous gas-SFR field was found."
            )
            print("Available PartType0 fields:")
            for name in fields:
                print(" ", name)
            print(
                "\nI will not infer instantaneous SFR from density or "
                "temperature automatically."
            )
            sys.exit(2)
    else:
        sfr_field = args.sfr_field

    print(f"Snapshot: a={a:.8f}, z={z:.8f}")
    print(f"Using PartType0 SFR field: {sfr_field}")

    gxyz, gsfr, gmass = read_gas(files, sfr_field)

    finite_sfr = np.isfinite(gsfr)
    sf = finite_sfr & (gsfr > 0)

    print(f"Gas particles total: {len(gsfr):,}")
    print(f"Gas particles with finite SFR: {finite_sfr.sum():,}")
    print(f"Gas particles with SFR > 0: {sf.sum():,}")

    if np.any(sf):
        print(
            f"Instantaneous SFR range among SF gas: "
            f"{gsfr[sf].min():.6e} - {gsfr[sf].max():.6e}"
        )
        print(
            f"Total instantaneous SFR in full snapshot: "
            f"{gsfr[sf].sum():.6f} Msun/yr"
        )

    xyz = faceon_positions(
        gxyz, center, rotation, box, a, h
    )
    x, y, zz = xyz.T

    inmap = (
        sf
        & (np.abs(x) < plot_radius)
        & (np.abs(y) < plot_radius)
        & (np.abs(zz) < zmax)
    )

    print(
        f"SF gas particles in face-on map volume: "
        f"{inmap.sum():,}"
    )
    if np.any(inmap):
        print(
            f"Instantaneous SFR in map volume: "
            f"{gsfr[inmap].sum():.6f} Msun/yr"
        )

    nbins = len(centers)
    edges = np.linspace(
        -plot_radius, plot_radius, nbins + 1
    )

    sfr_map, _, _ = np.histogram2d(
        y[inmap],
        x[inmap],
        bins=(edges, edges),
        weights=gsfr[inmap]
    )

    sf_count, _, _ = np.histogram2d(
        y[inmap],
        x[inmap],
        bins=(edges, edges)
    )
    sf_count = sf_count.astype(np.int64)

    pixel_area = pixel_size**2
    sigma_sfr = sfr_map / pixel_area
    sf_present = sf_count > 0

    xgrid, ygrid = np.meshgrid(centers, centers)

    overlap = sf_present & rv_mask

    print(f"Pixels containing SFR > 0 gas: {sf_present.sum():,}")
    print(
        f"Those pixels overlapping valid R(V): "
        f"{overlap.sum():,}"
    )

    if np.any(overlap):
        vals = rv[overlap]
        print(
            f"R(V) at active-SF gas pixels: "
            f"median={np.nanmedian(vals):.4f}, "
            f"p16={np.nanpercentile(vals,16):.4f}, "
            f"p84={np.nanpercentile(vals,84):.4f}"
        )

    nonsf = rv_mask & (~sf_present)
    if np.any(nonsf):
        vals = rv[nonsf]
        print(
            f"R(V) at other valid pixels: "
            f"median={np.nanmedian(vals):.4f}, "
            f"p16={np.nanpercentile(vals,16):.4f}, "
            f"p84={np.nanpercentile(vals,84):.4f}"
        )

    binary_png, binary_pdf, contour_png, contour_pdf = make_overlay(
        rv,
        rv_mask,
        xgrid,
        ygrid,
        sf_present,
        sigma_sfr,
        plot_radius,
        pixel_size,
        z,
        args.prefix,
        rv_vmin=args.rv_vmin,
        rv_vmax=args.rv_vmax,
    )

    npz_out = args.prefix + "_maps.npz"
    np.savez_compressed(
        npz_out,
        rv=rv,
        rv_mask=rv_mask,
        gas_sfr_map=sfr_map,
        sigma_sfr_gas=sigma_sfr,
        sf_gas_count=sf_count,
        sf_present=sf_present,
        centers_pkpc=centers,
        pixel_size_pkpc=pixel_size,
        plot_radius_pkpc=plot_radius,
        zmax_pkpc=zmax,
        center=center,
        rotation=rotation,
        snapshot_redshift=z,
        sfr_field=sfr_field,
    )

    txt = args.prefix + "_summary.txt"
    with open(txt, "w") as f:
        f.write(f"snapshot_redshift = {z:.8f}\n")
        f.write(f"sfr_field = {sfr_field}\n")
        f.write(f"gas_particles_total = {len(gsfr)}\n")
        f.write(f"gas_particles_sfr_gt_0 = {int(sf.sum())}\n")
        f.write(
            f"sf_gas_particles_in_map = "
            f"{int(inmap.sum())}\n"
        )
        f.write(
            f"instantaneous_sfr_map_Msun_per_yr = "
            f"{gsfr[inmap].sum():.10e}\n"
        )
        f.write(
            f"sf_pixels = {int(sf_present.sum())}\n"
        )
        f.write(
            f"sf_pixels_with_valid_Rv = "
            f"{int(overlap.sum())}\n"
        )

        if np.any(overlap):
            vals = rv[overlap]
            f.write(
                "Rv_active_sf_pixels "
                f"median={np.nanmedian(vals):.8f} "
                f"p16={np.nanpercentile(vals,16):.8f} "
                f"p84={np.nanpercentile(vals,84):.8f}\n"
            )

        if np.any(nonsf):
            vals = rv[nonsf]
            f.write(
                "Rv_other_valid_pixels "
                f"median={np.nanmedian(vals):.8f} "
                f"p16={np.nanpercentile(vals,16):.8f} "
                f"p84={np.nanpercentile(vals,84):.8f}\n"
            )

    print("Saved:")
    print(" ", binary_png)
    print(" ", binary_pdf)
    print(" ", contour_png)
    print(" ", contour_pdf)
    print(" ", npz_out)
    print(" ", txt)


if __name__ == "__main__":
    main()
