#!/usr/bin/env python3
"""Face-on gas, dust, and resolved dust-to-gas maps for CosmicGrain."""

import argparse
import glob

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

MSUN_G = 1.98847e33


def snapshot_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return files


def read_particles(pattern, particle_type):
    coordinates, masses = [], []
    a = h = box = unit_mass_g = None

    for filename in snapshot_files(pattern):
        with h5py.File(filename, "r") as handle:
            if a is None:
                header = handle["Header"].attrs
                params = handle["Parameters"].attrs
                a = float(header["Time"])
                box = float(header["BoxSize"])
                h = float(params["HubbleParam"])
                unit_mass_g = float(params.get("UnitMass_in_g", 1.0e10 * MSUN_G))

            group_name = f"PartType{particle_type}"
            if group_name not in handle:
                continue
            group = handle[group_name]
            coordinates.append(np.asarray(group["Coordinates"][:], dtype=float))
            masses.append(np.asarray(group["Masses"][:], dtype=float))

    if not coordinates:
        raise RuntimeError(f"No PartType{particle_type} particles found")

    return (
        np.concatenate(coordinates),
        np.concatenate(masses),
        a, h, box, unit_mass_g,
    )


def rotate_positions(coordinates, center, box, a, h, rotation):
    delta = coordinates - center[None, :]
    delta -= box * np.round(delta / box)
    return (delta * a / h) @ rotation.T


def projected_maps(position, mass, radius, zmax, bins):
    inside = (
        (np.abs(position[:, 0]) < radius)
        & (np.abs(position[:, 1]) < radius)
        & (np.abs(position[:, 2]) < zmax)
        & np.isfinite(mass)
        & (mass > 0)
    )
    edges = np.linspace(-radius, radius, bins + 1)
    mass_map, _, _ = np.histogram2d(
        position[inside, 1], position[inside, 0],
        bins=(edges, edges), weights=mass[inside],
    )
    count_map, _, _ = np.histogram2d(
        position[inside, 1], position[inside, 0], bins=(edges, edges),
    )
    return mass_map, count_map, inside


def positive_limits(image, lower=1, upper=99):
    values = image[np.isfinite(image) & (image > 0)]
    return np.percentile(values, [lower, upper])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", help="Snapshot path or quoted wildcard")
    parser.add_argument("--center-file", required=True)
    parser.add_argument("--rotation-file", required=True)
    parser.add_argument("--plot-radius", type=float, default=20.0)
    parser.add_argument("--zmax", type=float, default=5.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--min-dust-count", type=int, default=5)
    parser.add_argument("--min-gas-count", type=int, default=1)
    parser.add_argument("--min-gas-signal-fraction", type=float, default=1e-3)
    parser.add_argument("--prefix", default="halo441_dust_to_gas_500pc")
    args = parser.parse_args()

    center = np.asarray(np.loadtxt(args.center_file), dtype=float).reshape(3)
    rotation = np.asarray(
        np.loadtxt(args.rotation_file), dtype=float
    ).reshape(3, 3)

    gas_xyz, gas_mass, a, h, box, unit_mass_g = read_particles(args.snapshot, 0)
    dust_xyz, dust_mass, da, dh, dbox, dust_unit_mass_g = read_particles(
        args.snapshot, 6
    )
    if not np.allclose(
        [a, h, box, unit_mass_g], [da, dh, dbox, dust_unit_mass_g]
    ):
        raise ValueError("Gas and dust snapshot metadata do not match")

    gas_position = rotate_positions(gas_xyz, center, box, a, h, rotation)
    dust_position = rotate_positions(dust_xyz, center, box, a, h, rotation)

    gas_map, gas_count, gas_inside = projected_maps(
        gas_position, gas_mass, args.plot_radius, args.zmax, args.bins
    )
    dust_map, dust_count, dust_inside = projected_maps(
        dust_position, dust_mass, args.plot_radius, args.zmax, args.bins
    )

    pixel_size = 2.0 * args.plot_radius / args.bins
    pixel_area = pixel_size**2
    mass_unit_msun = unit_mass_g / MSUN_G / h
    gas_sigma = gas_map * mass_unit_msun / pixel_area
    dust_sigma = dust_map * mass_unit_msun / pixel_area

    gas_cut = args.min_gas_signal_fraction * np.nanmax(gas_map)
    valid = (
        (gas_count >= args.min_gas_count)
        & (dust_count >= args.min_dust_count)
        & (gas_map >= gas_cut)
        & (gas_map > 0)
        & (dust_map > 0)
    )

    dust_to_gas = np.full_like(gas_map, np.nan)
    dust_to_gas[valid] = dust_map[valid] / gas_map[valid]
    log_dg = np.log10(dust_to_gas)

    if not np.any(valid):
        raise RuntimeError(
            "No valid D/G pixels; reduce the particle-count or gas-signal cuts"
        )

    integrated_dg = dust_map.sum() / gas_map.sum()
    median_dg = np.nanmedian(dust_to_gas)
    p16, p84 = np.nanpercentile(dust_to_gas, [16, 84])

    print(f"Gas particles in map  = {np.count_nonzero(gas_inside):,}")
    print(f"Dust particles in map = {np.count_nonzero(dust_inside):,}")
    print(f"Pixel size             = {pixel_size:.3f} pkpc")
    print(f"Valid D/G pixels       = {np.count_nonzero(valid):,}")
    print(f"Integrated D/G         = {integrated_dg:.6e}")
    print(f"Median resolved D/G    = {median_dg:.6e}")
    print(f"Resolved 16--84%       = {p16:.6e} -- {p84:.6e}")

    extent = [-args.plot_radius, args.plot_radius,
              -args.plot_radius, args.plot_radius]
    cmap_mass = plt.get_cmap("magma").copy()
    cmap_ratio = plt.get_cmap("viridis").copy()
    cmap_mass.set_bad("white")
    cmap_ratio.set_bad("white")

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    gas_show = gas_sigma.copy()
    gas_show[gas_show <= 0] = np.nan
    gas_lo, gas_hi = positive_limits(gas_show)
    image = axes[0, 0].imshow(
        gas_show, origin="lower", extent=extent, cmap=cmap_mass,
        norm=LogNorm(gas_lo, gas_hi), interpolation="nearest",
    )
    axes[0, 0].set_title(r"Gas surface density")
    fig.colorbar(image, ax=axes[0, 0], label=r"$\Sigma_{\rm gas}$ ($M_\odot\,{\rm kpc}^{-2}$)")

    dust_show = dust_sigma.copy()
    dust_show[dust_show <= 0] = np.nan
    dust_lo, dust_hi = positive_limits(dust_show)
    image = axes[0, 1].imshow(
        dust_show, origin="lower", extent=extent, cmap=cmap_mass,
        norm=LogNorm(dust_lo, dust_hi), interpolation="nearest",
    )
    axes[0, 1].set_title(r"Dust surface density")
    fig.colorbar(image, ax=axes[0, 1], label=r"$\Sigma_{\rm dust}$ ($M_\odot\,{\rm kpc}^{-2}$)")

    ratio_values = log_dg[valid]
    ratio_lo, ratio_hi = np.percentile(ratio_values, [1, 99])
    image = axes[1, 0].imshow(
        log_dg, origin="lower", extent=extent, cmap=cmap_ratio,
        vmin=ratio_lo, vmax=ratio_hi, interpolation="nearest",
    )
    axes[1, 0].set_title(r"Dust-to-gas ratio")
    fig.colorbar(image, ax=axes[1, 0], label=r"$\log_{10}(\mathrm{D/G})$")

    hist_edges = np.linspace(ratio_lo, ratio_hi, 30)
    axes[1, 1].hist(
        ratio_values, bins=hist_edges, density=True, histtype="stepfilled",
        alpha=0.35, color="#2b6f9f", label="Pixel weighted",
    )
    axes[1, 1].hist(
        ratio_values, bins=hist_edges, weights=gas_map[valid], density=True,
        histtype="step", linewidth=2.2, color="#b43c2f",
        label="Gas-mass weighted",
    )
    axes[1, 1].axvline(
        np.log10(integrated_dg), color="black", linestyle="--", linewidth=1.8,
        label="Integrated D/G",
    )
    axes[1, 1].set_xlabel(r"$\log_{10}(\mathrm{D/G})$")
    axes[1, 1].set_ylabel("Normalized density")
    axes[1, 1].set_title("D/G distribution")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.18)

    for axis in axes.flat[:3]:
        axis.set_xlabel("x (pkpc)")
        axis.set_ylabel("y (pkpc)")
        axis.set_aspect("equal")

    fig.suptitle(
        rf"Halo 441: face-on dust-to-gas structure "
        rf"({1000 * pixel_size:g} pc pixels, $|z|<{args.zmax:g}$ pkpc)",
        fontsize=16,
    )
    fig.tight_layout()

    figure_name = args.prefix + "_fourpanel.png"
    fig.savefig(figure_name, dpi=240, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        args.prefix + "_maps.npz",
        gas_surface_density_msun_kpc2=gas_sigma,
        dust_surface_density_msun_kpc2=dust_sigma,
        dust_to_gas=dust_to_gas,
        gas_particle_count=gas_count,
        dust_particle_count=dust_count,
        valid_mask=valid,
        pixel_size_pkpc=pixel_size,
        extent_pkpc=np.asarray(extent),
        integrated_dust_to_gas=integrated_dg,
        center=center,
        rotation_matrix=rotation,
    )
    print(f"Saved {figure_name}")
    print(f"Saved {args.prefix}_maps.npz")


if __name__ == "__main__":
    main()
