#!/usr/bin/env python3
"""
plot_rv_recent_sf_overlay.py

Overlay recent-star-formation locations directly on an existing CosmicGrain
R_V map produced by halo569_rv_sfr_triptych.py.

The SF information is treated primarily as a binary spatial tracer:
  * 100 Myr recent-SF pixels: outlined squares
  * 30 Myr recent-SF pixels: filled circles

This avoids over-interpreting pixel-to-pixel SFR amplitudes when only a small
number of young star particles populate the map.

Input NPZ must contain:
  rv, rv_mask, sigma_sfr_30, sigma_sfr_100,
  centers_pkpc, pixel_size_pkpc, plot_radius_pkpc,
  snapshot_redshift

Example
-------
python3 plot_rv_recent_sf_overlay.py \
  halo569_snap020_rv_sfr_500pc_maps.npz \
  --prefix halo569_snap020_rv_recent_sf_overlay
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument(
        "--prefix",
        default="rv_recent_sf_overlay"
    )
    ap.add_argument(
        "--rv-vmin",
        type=float,
        default=None,
        help="Optional lower R_V display limit"
    )
    ap.add_argument(
        "--rv-vmax",
        type=float,
        default=None,
        help="Optional upper R_V display limit"
    )
    ap.add_argument(
        "--marker-scale",
        type=float,
        default=1.0,
        help="Scale factor for overlay marker sizes"
    )
    args = ap.parse_args()

    d = np.load(args.npz)

    rv = d["rv"]
    rv_mask = d["rv_mask"].astype(bool)
    sfr30 = d["sigma_sfr_30"]
    sfr100 = d["sigma_sfr_100"]
    centers = d["centers_pkpc"]

    pixel_size = float(np.asarray(d["pixel_size_pkpc"]))
    plot_radius = float(np.asarray(d["plot_radius_pkpc"]))
    z = float(np.asarray(d["snapshot_redshift"]))

    if rv.shape != sfr30.shape or rv.shape != sfr100.shape:
        raise RuntimeError("R_V and SFR maps do not have matching shapes")

    xgrid, ygrid = np.meshgrid(centers, centers)

    sf30 = np.isfinite(sfr30) & (sfr30 > 0)
    sf100 = np.isfinite(sfr100) & (sfr100 > 0)

    # 100 Myr-only footprint, so that 30 Myr markers are not duplicated.
    sf100_only = sf100 & (~sf30)

    if args.rv_vmin is None:
        rv_vmin = float(np.nanpercentile(rv[rv_mask], 2))
    else:
        rv_vmin = args.rv_vmin

    if args.rv_vmax is None:
        rv_vmax = float(np.nanpercentile(rv[rv_mask], 98))
    else:
        rv_vmax = args.rv_vmax

    extent = [
        -plot_radius, plot_radius,
        -plot_radius, plot_radius
    ]

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

    # Marker areas are chosen to be easy to see but still close to the
    # physical 500-pc pixel footprint.
    s100 = 90.0 * args.marker_scale
    s30 = 36.0 * args.marker_scale

    if np.any(sf100_only):
        ax.scatter(
            xgrid[sf100_only],
            ygrid[sf100_only],
            s=s100,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label=r"Recent SF: $30{-}100$ Myr"
        )

    if np.any(sf30):
        ax.scatter(
            xgrid[sf30],
            ygrid[sf30],
            s=s30,
            marker="o",
            facecolors="black",
            edgecolors="white",
            linewidths=0.8,
            label=r"Recent SF: $<30$ Myr",
            zorder=5
        )

    ax.axhline(0, ls="--", lw=0.7, alpha=0.45)
    ax.axvline(0, ls="--", lw=0.7, alpha=0.45)

    x0 = -plot_radius + 1.1
    y0 = -plot_radius + 1.1
    bar = 5.0
    ax.plot([x0, x0 + bar], [y0, y0], lw=3)
    ax.text(
        x0 + bar / 2,
        y0 + 0.45,
        "5 kpc",
        ha="center",
        va="bottom",
        fontsize=9
    )

    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel(r"$X_{\rm face-on}$ [pkpc]")
    ax.set_ylabel(r"$Y_{\rm face-on}$ [pkpc]")

    ax.set_title(
        rf"Halo 569 — $z={z:.3f}$ — "
        rf"{pixel_size*1000:.0f} pc pixels"
    )

    if np.any(sf30) or np.any(sf100_only):
        ax.legend(
            loc="upper left",
            frameon=True,
            fontsize=9
        )

    png = args.prefix + ".png"
    pdf = args.prefix + ".pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Also write a small summary text file.
    txt = args.prefix + "_summary.txt"
    with open(txt, "w") as f:
        f.write(f"Input NPZ: {args.npz}\n")
        f.write(f"z = {z:.8f}\n")
        f.write(f"pixel_size_pkpc = {pixel_size:.8f}\n")
        f.write(f"valid_Rv_pixels = {int(rv_mask.sum())}\n")
        f.write(f"SF_pixels_lt30Myr = {int(sf30.sum())}\n")
        f.write(f"SF_pixels_lt100Myr = {int(sf100.sum())}\n")
        f.write(
            f"SF_pixels_30to100Myr_only = "
            f"{int(sf100_only.sum())}\n"
        )
        overlap30 = sf30 & rv_mask
        overlap100 = sf100 & rv_mask
        f.write(
            f"lt30Myr_SF_pixels_with_valid_Rv = "
            f"{int(overlap30.sum())}\n"
        )
        f.write(
            f"lt100Myr_SF_pixels_with_valid_Rv = "
            f"{int(overlap100.sum())}\n"
        )

        if np.any(overlap30):
            vals = rv[overlap30]
            f.write(
                "Rv_at_lt30Myr_SF_pixels "
                f"median={np.nanmedian(vals):.6f} "
                f"p16={np.nanpercentile(vals,16):.6f} "
                f"p84={np.nanpercentile(vals,84):.6f}\n"
            )

        if np.any(overlap100):
            vals = rv[overlap100]
            f.write(
                "Rv_at_lt100Myr_SF_pixels "
                f"median={np.nanmedian(vals):.6f} "
                f"p16={np.nanpercentile(vals,16):.6f} "
                f"p84={np.nanpercentile(vals,84):.6f}\n"
            )

        vals = rv[rv_mask]
        f.write(
            "Rv_all_valid_pixels "
            f"median={np.nanmedian(vals):.6f} "
            f"p16={np.nanpercentile(vals,16):.6f} "
            f"p84={np.nanpercentile(vals,84):.6f}\n"
        )

    print(f"Input: {args.npz}")
    print(f"Valid R(V) pixels: {rv_mask.sum():,}")
    print(f"Recent-SF pixels <30 Myr: {sf30.sum():,}")
    print(f"Recent-SF pixels <100 Myr: {sf100.sum():,}")
    print(
        f"Recent-SF pixels 30-100 Myr only: "
        f"{sf100_only.sum():,}"
    )
    print(
        f"<30 Myr pixels overlapping valid R(V): "
        f"{(sf30 & rv_mask).sum():,}"
    )
    print(
        f"<100 Myr pixels overlapping valid R(V): "
        f"{(sf100 & rv_mask).sum():,}"
    )
    print("Saved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", txt)


if __name__ == "__main__":
    main()
