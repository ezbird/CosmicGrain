#!/usr/bin/env python3
"""Controlled CosmicGrain grain-versus-geometry attenuation pilot.

Assigns equal intrinsic B/V light to each selected star particle and uses dust
Q_ext as removal from each stellar sightline. Fixed-grain cross sections are
rescaled to match the evolved model's total V-band opacity; this artificial
normalization isolates the B/V opacity shape more cleanly. This is a
diagnostic, NOT full radiative transfer:
it omits stellar-population spectra, scattering into the beam, and dust emission.
All inclinations use the same star and dust particles in a spherical aperture.

Run from the GADGET-4 root::

    python3 scripts/cosmicgrain_geometry_pilot.py \
        --root . --output-dir halo441_output_1024_mingashsml0.0 \
        --group 1 --output cosmicgrain_geometry_pilot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import cosmicgrain_z0_z6_gsd_rv as cg

MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21


def args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path,
                   default=Path("halo441_output_1024_mingashsml0.0"))
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--snap", type=int, default=29)
    p.add_argument("--radius-pkpc", type=float, default=15.)
    p.add_argument("--pixel-pkpc", type=float, default=.5)
    p.add_argument("--depth-pkpc", type=float, default=.5)
    p.add_argument("--smooth-pkpc", type=float, default=.5)
    p.add_argument("--fixed-radius-nm", type=float, default=100.)
    p.add_argument("--unmatched-opacity", action="store_true",
                   help="retain raw fixed-grain opacity instead of matching total V-band opacity")
    p.add_argument("--inclinations", type=float, nargs="+", default=[0., 30., 60., 75.])
    p.add_argument("--carbon-table", type=Path,
                   default=Path("scripts/dust_optics/callqcomp.out_D16MGemt"))
    p.add_argument("--silicate-table", type=Path,
                   default=Path("scripts/dust_optics/callqcomp.out_DH16asil"))
    p.add_argument("--style-file", type=Path, default=Path("scripts/cosmicgrain.mplstyle"))
    p.add_argument("--output", type=Path, default=Path("cosmicgrain_geometry_pilot.png"))
    a = p.parse_args()
    a.root = a.root.resolve()
    for key in ("output_dir", "carbon_table", "silicate_table", "style_file"):
        value = getattr(a, key)
        setattr(a, key, (a.root / value).resolve() if not value.is_absolute() else value)
    a.output = a.output.resolve()
    if (a.radius_pkpc <= 0 or a.pixel_pkpc <= 0 or a.depth_pkpc <= 0
            or a.smooth_pkpc < 0 or a.fixed_radius_nm <= 0):
        p.error("radii, pixel size, and depth must be positive; smoothing may be zero")
    if any(not 0 <= angle < 90 for angle in a.inclinations):
        p.error("inclinations must be between 0 and 90 degrees")
    return a


def rotate_inclination(pos, angle):
    theta = np.deg2rad(angle)
    out = pos.copy()
    out[:, 1] = np.cos(theta) * pos[:, 1] - np.sin(theta) * pos[:, 2]
    out[:, 2] = np.sin(theta) * pos[:, 1] + np.cos(theta) * pos[:, 2]
    return out


def attenuation(stars, dust, weights_b, weights_v, a, h, angle):
    """Return integrated B,V attenuation for one unchanged 3D particle set."""
    stars = rotate_inclination(stars, angle)
    dust = rotate_inclination(dust, angle)
    nxy = int(np.ceil(2 * a.radius_pkpc / a.pixel_pkpc))
    nz = int(np.ceil(2 * a.radius_pkpc / a.depth_pkpc))
    # Use exact cell dimensions so the full sphere fits the mapped cube.
    xy_edge = .5 * nxy * a.pixel_pkpc
    z_edge = .5 * nz * a.depth_pkpc
    bounds = [[-xy_edge, xy_edge], [-xy_edge, xy_edge], [-z_edge, z_edge]]
    dust_b = np.histogramdd(dust, bins=(nxy, nxy, nz), range=bounds,
                            weights=weights_b)[0]
    dust_v = np.histogramdd(dust, bins=(nxy, nxy, nz), range=bounds,
                            weights=weights_v)[0]
    if a.smooth_pkpc:
        sigma = a.smooth_pkpc / a.pixel_pkpc
        dust_b = gaussian_filter(dust_b, (sigma, sigma, 0), mode="constant")
        dust_v = gaussian_filter(dust_v, (sigma, sigma, 0), mode="constant")
    # weights are in (simulation mass unit / micron) times Q/rho. The
    # cross section is (3/4) M Q / (a rho); 1 micron = 1e-4 cm.
    area = (a.pixel_pkpc * KPC_CM) ** 2
    tau_factor = .75 * (1e10 / h) * MSUN_G / (1e-4 * area)
    tau_b = dust_b * tau_factor
    tau_v = dust_v * tau_factor
    # Observer looks from +z; half of the dust in the star's own slab is in
    # front on average. Same convention for both grain prescriptions.
    front_b = np.cumsum(tau_b[:, :, ::-1], axis=2)[:, :, ::-1] - .5 * tau_b
    front_v = np.cumsum(tau_v[:, :, ::-1], axis=2)[:, :, ::-1] - .5 * tau_v
    ijk = np.floor((stars + [xy_edge, xy_edge, z_edge]) /
                   [a.pixel_pkpc, a.pixel_pkpc, a.depth_pkpc]).astype(int)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nxy - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, nxy - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
    tb = front_b[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    tv = front_v[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    ab = -2.5 * np.log10(np.mean(np.exp(-tb)))
    av = -2.5 * np.log10(np.mean(np.exp(-tv)))
    color_excess = ab - av
    return {"inclination_deg": angle, "A_B_mag": float(ab),
            "A_V_mag": float(av), "E_B_minus_V_mag": float(color_excess),
            "R_V_effective": float(av / color_excess) if color_excess > 0 else None}


def main():
    a = args()
    out = a.output_dir
    if not out.is_dir():
        raise FileNotFoundError(out)
    series = cg.get_zoom_halo_series(out, [a.snap], group_index=a.group,
                                    reference_snap=a.snap, verbose=True,
                                    refine_reference_center=True,
                                    refine_historical_centers=False)
    halo = series.get(a.snap)
    if halo is None:
        raise RuntimeError("Target halo not found")
    scale, h, box = cg.read_header(out, a.snap)
    rotation, source, _ = cg.derive_orientation(out, a.snap, halo, scale, h,
                                                box, a.radius_pkpc)
    dust = cg.read_dust(out, a.snap)
    stars = cg.read_ptype(out, a.snap, 4, ["Coordinates", "Masses"])
    if not len(dust["mass"]) or not len(stars["Masses"]):
        raise RuntimeError("Both dust and stellar particles are required")
    dpos = (cg.periodic_delta(dust["coords"], halo["center"], box)
            * scale / h) @ rotation.T
    spos = (cg.periodic_delta(stars["Coordinates"], halo["center"], box)
            * scale / h) @ rotation.T
    radius = min(a.radius_pkpc, float(halo["r200_pkpc"]))
    dm = (np.linalg.norm(dpos, axis=1) < radius) & (dust["mass"] > 0)
    sm = (np.linalg.norm(spos, axis=1) < radius) & (stars["Masses"] > 0)
    if dm.sum() < 20 or sm.sum() < 20:
        raise RuntimeError(f"Too few particles: dust={dm.sum()}, stars={sm.sum()}")
    dpos, spos = dpos[dm], spos[sm]
    mass = dust["mass"][dm]
    carbon = dust["carbon"][dm]
    grain = dust["radius_nm"][dm]
    cb = cg.read_draine_qext_table(a.carbon_table)
    sil = cg.read_draine_qext_table(a.silicate_table)
    fixed_carbon = float(np.average(np.clip(carbon, 0, 1), weights=mass))
    prescriptions = {
        "Evolved grains": (grain, carbon),
        "Fixed grains": (np.full_like(grain, a.fixed_radius_nm),
                         np.full_like(carbon, fixed_carbon)),
    }
    results = {}
    intrinsic = {}
    optics = {name: cg.extinction_weights(mass, radii, fractions, cb, sil)
              for name, (radii, fractions) in prescriptions.items()}
    matched_scale = 1.0
    if not a.unmatched_opacity:
        evolved_v = optics["Evolved grains"][1].sum()
        fixed_v = optics["Fixed grains"][1].sum()
        if evolved_v <= 0 or fixed_v <= 0:
            raise RuntimeError("Cannot match V opacity with non-positive total cross section")
        matched_scale = float(evolved_v / fixed_v)
        fixed_b, fixed_v_weights = optics["Fixed grains"]
        optics["Fixed grains"] = (fixed_b * matched_scale,
                                  fixed_v_weights * matched_scale)
    for name, (wb, wv) in optics.items():
        total_b, total_v = wb.sum(), wv.sum()
        intrinsic[name] = (float(total_v / (total_b - total_v))
                           if total_b > total_v else None)
        results[name] = [attenuation(spos, dpos, wb, wv, a, h, angle)
                         for angle in a.inclinations]
    if a.style_file.is_file():
        plt.style.use(str(a.style_file))
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    colors = {"Evolved grains": "#7030A0", "Fixed grains": "#16834B"}
    for name, rows in results.items():
        angles = [row["inclination_deg"] for row in rows]
        rv = [row["R_V_effective"] for row in rows]
        display_name = ("Fixed grains (100 nm, mean carbon fraction)"
                        if name == "Fixed grains" else name)
        ax.plot(angles, rv, "o-", lw=2, color=colors[name], label=display_name)
    ax.set_xlabel("Inclination (degrees from face-on)")
    ax.set_ylabel(r"$R_V$")
    ax.set_xlim(-3, max(a.inclinations) + 3)
    ax.grid(alpha=.16)
    ax.legend(loc="best")
    mantissa, exponent = f"{halo['m200_msun']:.2e}".split("e")
    ax.set_title(f"Halo 441 attenuation ($M_{{200c}}={mantissa}\\times"
                 f"10^{{{int(exponent)}}}\\,M_\\odot$)")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=220)
    fig.savefig(a.output.with_suffix(".pdf"))
    plt.close(fig)
    metadata = {"snapshot": a.snap, "redshift": 1 / scale - 1,
                "group": a.group, "radius_pkpc": radius,
                "pixel_pkpc": a.pixel_pkpc, "depth_pkpc": a.depth_pkpc,
                "smooth_pkpc": a.smooth_pkpc, "n_dust": int(dm.sum()),
                "n_stars": int(sm.sum()), "orientation": source,
                "fixed_radius_nm": a.fixed_radius_nm,
                "fixed_carbon_mass_fraction": fixed_carbon,
                "fixed_opacity_V_scale": matched_scale,
                "total_V_opacity_matched": not a.unmatched_opacity,
                "attenuation_definition": "For X=B,V, A_X=-2.5 log10[mean_i exp(-tau_X,i)] over selected star particles; plotted R_V=A_V/(A_B-A_V).",
                "intrinsic_screen_R_V": intrinsic, "measurements": results,
                "caution": "B/V extinction-only diagnostic; equal star-particle light is not a stellar luminosity model; no SSP spectra or scattering."}
    a.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {a.output}, PDF, and JSON")


if __name__ == "__main__":
    main()
