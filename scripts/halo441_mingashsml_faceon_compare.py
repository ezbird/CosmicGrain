#!/usr/bin/env python3
"""Matched face-on gas and dust maps for the CG10 MinGasHsml experiment.

The default comparison is snapshot 029 of the existing 1024^3 runs:
  MinGasHsml=1.0: ../halo441_output_1024
  MinGasHsml=0.0: ../halo441_output_1024_mingashsml0.0

Gas particles are rendered with Gaussian kernels tied to their stored SPH
smoothing lengths. Dust particles have no stored smoothing length, so they are
rendered with a common fixed Gaussian FWHM. By default, each run is independently
centered and face-on oriented; a shared baseline frame remains available.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np

MSUN_G = 1.98847e33


def snapshot_files(root: Path, snap: int) -> list[Path]:
    files = sorted((root / f"snapdir_{snap:03d}").glob(f"snapshot_{snap:03d}.*.hdf5"))
    if not files:
        single = root / f"snapshot_{snap:03d}.hdf5"
        if single.exists():
            files = [single]
    if not files:
        raise FileNotFoundError(f"No snapshot {snap:03d} beneath {root}")
    return files


def catalog_files(root: Path, snap: int) -> list[Path]:
    files = sorted((root / f"groups_{snap:03d}").glob(f"fof_subhalo_tab_{snap:03d}.*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No group catalog {snap:03d} beneath {root}")
    return files


def attr_any(h5: h5py.File, name: str, default=None):
    for group in ("Parameters", "Header", "Config"):
        if group in h5 and name in h5[group].attrs:
            return h5[group].attrs[name]
    return default


def run_metadata(root: Path, snap: int) -> dict:
    first = snapshot_files(root, snap)[0]
    with h5py.File(first, "r") as f:
        a = float(f["Header"].attrs["Time"])
        hubble = float(attr_any(f, "HubbleParam"))
        box = float(f["Header"].attrs["BoxSize"])
        unit_mass = float(attr_any(f, "UnitMass_in_g", 1.0e10 * MSUN_G))
    return {"a": a, "hubble": hubble, "box": box, "unit_mass": unit_mass}


def group_center(root: Path, snap: int, group_index: int) -> np.ndarray:
    chunks = []
    for path in catalog_files(root, snap):
        with h5py.File(path, "r") as f:
            if "Group" in f and "GroupPos" in f["Group"]:
                chunks.append(np.asarray(f["Group/GroupPos"], dtype=np.float64))
    pos = np.concatenate(chunks, axis=0)
    if not 0 <= group_index < len(pos):
        raise IndexError(f"group {group_index} outside catalog range 0..{len(pos)-1}")
    return pos[group_index]


def periodic_delta(coords: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    delta = coords.astype(np.float64, copy=False) - center
    delta -= box * np.rint(delta / box)
    return delta


def load_type(root: Path, snap: int, ptype: int, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {field: [] for field in fields}
    group = f"PartType{ptype}"
    for path in snapshot_files(root, snap):
        with h5py.File(path, "r") as f:
            if group not in f:
                continue
            for field in fields:
                if field not in f[group]:
                    raise KeyError(f"{path}: missing {group}/{field}")
                out[field].append(np.asarray(f[f"{group}/{field}"]))
    return {field: np.concatenate(parts, axis=0) for field, parts in out.items()}


def stellar_frame(root: Path, snap: int, center: np.ndarray, meta: dict, radius: float) -> np.ndarray:
    stars = load_type(root, snap, 4, ("Coordinates", "Velocities", "Masses"))
    xyz = periodic_delta(stars["Coordinates"], center, meta["box"]) * meta["a"] / meta["hubble"]
    vel = stars["Velocities"].astype(np.float64)
    mass = stars["Masses"].astype(np.float64)
    select = np.linalg.norm(xyz, axis=1) < radius
    if select.sum() < 100:
        raise RuntimeError(f"Only {select.sum()} stars inside {radius:g} pkpc; cannot orient disk")
    xyz, vel, mass = xyz[select], vel[select], mass[select]
    vbulk = np.average(vel, axis=0, weights=mass)
    angmom = np.sum(mass[:, None] * np.cross(xyz, vel - vbulk), axis=0)
    zhat = angmom / np.linalg.norm(angmom)

    # Align the projected stellar major axis with +x to fix the face-on roll.
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, zhat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, zhat) * zhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(zhat, e1)
    uv = np.column_stack((xyz @ e1, xyz @ e2))
    cov = np.cov(uv.T, aweights=mass)
    major2 = np.linalg.eigh(cov)[1][:, -1]
    xhat = major2[0] * e1 + major2[1] * e2
    xhat /= np.linalg.norm(xhat)
    if np.dot(xhat, trial) < 0:
        xhat *= -1
    yhat = np.cross(zhat, xhat)
    return np.vstack((xhat, yhat, zhat))


def central_particles(root: Path, snap: int, ptype: int, center: np.ndarray, meta: dict,
                      rotation: np.ndarray, radius: float, zmax: float) -> dict:
    fields = ["Coordinates", "Masses"]
    if ptype == 0:
        fields.extend(("SmoothingLength", "Metallicity"))
    data = load_type(root, snap, ptype, tuple(fields))
    xyz = periodic_delta(data["Coordinates"], center, meta["box"]) * meta["a"] / meta["hubble"]
    xyz = xyz @ rotation.T
    keep = ((np.abs(xyz[:, 0]) < radius) & (np.abs(xyz[:, 1]) < radius) &
            (np.abs(xyz[:, 2]) < zmax))
    result = {
        "x": xyz[keep, 0],
        "y": xyz[keep, 1],
        "mass": data["Masses"].astype(np.float64)[keep] * meta["unit_mass"] / meta["hubble"] / MSUN_G,
    }
    if ptype == 0:
        result["hsml"] = data["SmoothingLength"].astype(np.float64)[keep] * meta["a"] / meta["hubble"]
        result["metallicity"] = data["Metallicity"].astype(np.float64)[keep]
    return result


def gaussian_splat(x, y, mass, fwhm, radius: float, pixels: int) -> np.ndarray:
    """Deposit normalized 2-D Gaussian kernels; result is Msun/pc^2."""
    image = np.zeros((pixels, pixels), dtype=np.float64)
    pixel = 2.0 * radius / pixels
    fwhm = np.broadcast_to(np.asarray(fwhm, dtype=np.float64), mass.shape)
    sigma_pix = np.maximum(fwhm / 2.354820045 / pixel, 0.60)
    xp = (x + radius) / pixel - 0.5
    yp = (y + radius) / pixel - 0.5
    for cx, cy, m, sig in zip(xp, yp, mass, sigma_pix):
        reach = max(2, int(np.ceil(4.0 * sig)))
        ix0, ix1 = max(0, int(np.floor(cx)) - reach), min(pixels, int(np.floor(cx)) + reach + 1)
        iy0, iy1 = max(0, int(np.floor(cy)) - reach), min(pixels, int(np.floor(cy)) + reach + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx = np.arange(ix0, ix1) - cx
        gy = np.arange(iy0, iy1) - cy
        kernel = np.exp(-0.5 * (gy[:, None] ** 2 + gx[None, :] ** 2) / sig ** 2)
        kernel /= kernel.sum()
        image[iy0:iy1, ix0:ix1] += m * kernel
    return image / (pixel * 1000.0) ** 2


def robust_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    positive = np.concatenate((a[a > 0], b[b > 0]))
    if not len(positive):
        return 1.0, 10.0
    return float(np.percentile(positive, 2)), float(np.percentile(positive, 99.7))


def ratio_map(new: np.ndarray, old: np.ndarray, signal_fraction: float) -> np.ndarray:
    threshold_new = signal_fraction * np.nanmax(new)
    threshold_old = signal_fraction * np.nanmax(old)
    good = (new > threshold_new) & (old > threshold_old)
    out = np.full(new.shape, np.nan)
    out[good] = np.log10(new[good] / old[good])
    return out


def radial_profile(particles: dict, edges: np.ndarray) -> dict[str, np.ndarray]:
    radius = np.hypot(particles["x"], particles["y"])
    shell_mass = np.histogram(radius, bins=edges, weights=particles["mass"])[0]
    area_pc2 = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * 1.0e6
    result = {
        "shell_mass": shell_mass,
        "surface_density": shell_mass / area_pc2,
        "enclosed_mass": np.cumsum(shell_mass),
    }
    if "metallicity" in particles:
        metal_shell_mass = np.histogram(
            radius, bins=edges, weights=particles["mass"] * particles["metallicity"])[0]
        result["metal_shell_mass"] = metal_shell_mass
        result["gas_metallicity"] = np.divide(
            metal_shell_mass, shell_mass, out=np.full_like(shell_mass, np.nan), where=shell_mass > 0)
    return result


def write_profiles(path: Path, edges: np.ndarray, profiles: dict) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "MinGasHsml", "r_inner_pkpc", "r_outer_pkpc", "r_mid_pkpc",
            "gas_shell_mass_Msun", "dust_shell_mass_Msun",
            "gas_surface_density_Msun_pc2", "dust_surface_density_Msun_pc2",
            "gas_phase_metal_mass_Msun", "gas_metallicity_mass_fraction",
            "D_over_G", "D_over_Z_total", "gas_enclosed_mass_Msun", "dust_enclosed_mass_Msun",
        ])
        for label in ("1.0", "0.0"):
            gas = profiles[(label, "gas")]
            dust = profiles[(label, "dust")]
            ratio = np.divide(
                dust["shell_mass"], gas["shell_mass"],
                out=np.full_like(dust["shell_mass"], np.nan), where=gas["shell_mass"] > 0)
            dz = np.divide(
                dust["shell_mass"], gas["metal_shell_mass"] + dust["shell_mass"],
                out=np.full_like(dust["shell_mass"], np.nan),
                where=(gas["metal_shell_mass"] + dust["shell_mass"]) > 0)
            for i, rmid in enumerate(centers):
                writer.writerow([
                    label, edges[i], edges[i + 1], rmid,
                    gas["shell_mass"][i], dust["shell_mass"][i],
                    gas["surface_density"][i], dust["surface_density"][i],
                    gas["metal_shell_mass"][i], gas["gas_metallicity"][i],
                    ratio[i], dz[i],
                    gas["enclosed_mass"][i], dust["enclosed_mass"][i],
                ])


def plot_profiles(path: Path, edges: np.ndarray, profiles: dict, snapshot: int, zmax: float,
                  solar_metallicity: float) -> None:
    radius = 0.5 * (edges[:-1] + edges[1:])
    colors = {"1.0": "#3569b8", "0.0": "#d94a3a"}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for label in ("1.0", "0.0"):
        gas = profiles[(label, "gas")]
        dust = profiles[(label, "dust")]
        axes[0, 0].plot(radius, gas["surface_density"], color=colors[label], label=label)
        axes[0, 1].plot(radius, dust["surface_density"], color=colors[label], label=label)
        dg = np.divide(
            dust["shell_mass"], gas["shell_mass"],
            out=np.full_like(dust["shell_mass"], np.nan), where=gas["shell_mass"] > 0)
        axes[1, 0].plot(radius, dg, color=colors[label], label=label)
        axes[0, 2].plot(radius, gas["gas_metallicity"] / solar_metallicity,
                        color=colors[label], label=label)
        dz = np.divide(
            dust["shell_mass"], gas["metal_shell_mass"] + dust["shell_mass"],
            out=np.full_like(dust["shell_mass"], np.nan),
            where=(gas["metal_shell_mass"] + dust["shell_mass"]) > 0)
        axes[1, 1].plot(radius, dz, color=colors[label], label=label)
        axes[1, 2].plot(radius, gas["enclosed_mass"], color=colors[label], label=f"gas, {label}")
        axes[1, 2].plot(radius, dust["enclosed_mass"], color=colors[label], linestyle="--",
                        label=f"dust, {label}")
    axes[0, 0].set_ylabel(r"$\Sigma_{\rm gas}$ ($M_\odot\,\mathrm{pc}^{-2}$)")
    axes[0, 1].set_ylabel(r"$\Sigma_{\rm dust}$ ($M_\odot\,\mathrm{pc}^{-2}$)")
    axes[1, 0].set_ylabel("D/G in annulus")
    axes[0, 2].set_ylabel(r"gas-phase $Z/Z_\odot$")
    axes[1, 1].set_ylabel(r"$D/Z_{\rm tot}$ in annulus")
    axes[1, 2].set_ylabel(r"$M(<R)$ ($M_\odot$)")
    for ax in axes.flat:
        ax.set_xlabel("cylindrical radius (physical kpc)")
        ax.grid(alpha=0.22)
        ax.legend(title="MinGasHsml", fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
        ax.set_yscale("log")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].axhline(0.5, color="0.5", linewidth=0.8, linestyle=":")
    axes[1, 2].legend(fontsize=8, ncol=2)
    fig.suptitle(f"CG10 1024³ radial structure at snapshot {snapshot:03d}; |z|<{zmax:g} pkpc")
    fig.savefig(path, dpi=220, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("../halo441_output_1024"))
    parser.add_argument("--zero", type=Path, default=Path("../halo441_output_1024_mingashsml0.0"))
    parser.add_argument("--snapshot", type=int, default=29)
    parser.add_argument("--group-index", type=int, default=1)
    parser.add_argument("--radius", type=float, default=20.0, help="face-on half-width (physical kpc)")
    parser.add_argument("--zmax", type=float, default=5.0, help="line-of-sight half-depth (physical kpc)")
    parser.add_argument("--orientation-radius", type=float, default=20.0)
    parser.add_argument("--orientation-mode", choices=("independent", "baseline"), default="independent",
                        help="independent makes each galaxy face-on; baseline preserves a shared 3-D frame")
    parser.add_argument("--pixels", type=int, default=320)
    parser.add_argument("--gas-hsml-fwhm", type=float, default=0.5,
                        help="gas Gaussian FWHM as a multiple of stored Hsml")
    parser.add_argument("--dust-fwhm", type=float, default=0.20, help="fixed dust FWHM (physical kpc)")
    parser.add_argument("--ratio-signal-fraction", type=float, default=1e-3)
    parser.add_argument("--radial-bins", type=int, default=40)
    parser.add_argument("--solar-metallicity", type=float, default=0.0134,
                        help="solar metal mass fraction used only for plotting Z/Zsun")
    parser.add_argument("--output", type=Path, default=Path("halo441_mingashsml1_vs0_faceon.png"))
    parser.add_argument("--profile-output", type=Path,
                        default=Path("halo441_mingashsml1_vs0_radial_profiles.png"))
    parser.add_argument("--profile-csv", type=Path,
                        default=Path("halo441_mingashsml1_vs0_radial_profiles.csv"))
    args = parser.parse_args()

    roots = {"1.0": args.baseline, "0.0": args.zero}
    meta = {label: run_metadata(root, args.snapshot) for label, root in roots.items()}
    centers = {label: group_center(root, args.snapshot, args.group_index) for label, root in roots.items()}
    baseline_rotation = stellar_frame(
        args.baseline, args.snapshot, centers["1.0"], meta["1.0"], args.orientation_radius)
    if args.orientation_mode == "independent":
        rotations = {
            "1.0": baseline_rotation,
            "0.0": stellar_frame(args.zero, args.snapshot, centers["0.0"], meta["0.0"], args.orientation_radius),
        }
    else:
        rotations = {"1.0": baseline_rotation, "0.0": baseline_rotation}

    maps = {}
    counts = {}
    particles = {}
    for label, root in roots.items():
        gas = central_particles(
            root, args.snapshot, 0, centers[label], meta[label], rotations[label], args.radius, args.zmax)
        dust = central_particles(
            root, args.snapshot, 6, centers[label], meta[label], rotations[label], args.radius, args.zmax)
        particles[(label, "gas")] = gas
        particles[(label, "dust")] = dust
        maps[(label, "gas")] = gaussian_splat(
            gas["x"], gas["y"], gas["mass"], args.gas_hsml_fwhm * gas["hsml"], args.radius, args.pixels)
        maps[(label, "dust")] = gaussian_splat(
            dust["x"], dust["y"], dust["mass"], args.dust_fwhm, args.radius, args.pixels)
        counts[label] = (len(gas["mass"]), len(dust["mass"]))

    extent = (-args.radius, args.radius, -args.radius, args.radius)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.0), constrained_layout=True)
    labels = {"gas": r"Gas surface density ($M_\odot\,\mathrm{pc}^{-2}$)",
              "dust": r"Dust surface density ($M_\odot\,\mathrm{pc}^{-2}$)"}
    cmaps = {"gas": "magma", "dust": "cividis"}
    for row, component in enumerate(("gas", "dust")):
        old, new = maps[("1.0", component)], maps[("0.0", component)]
        vmin, vmax = robust_limits(old, new)
        norm = LogNorm(vmin=max(vmin, vmax * 1e-5), vmax=vmax)
        for col, (title, image) in enumerate((("MinGasHsml = 1.0", old), ("MinGasHsml = 0.0", new))):
            artist = axes[row, col].imshow(image, origin="lower", extent=extent, cmap=cmaps[component], norm=norm)
            axes[row, col].set_title(title)
            fig.colorbar(artist, ax=axes[row, col], label=labels[component], shrink=0.86)
        ratio = ratio_map(new, old, args.ratio_signal_fraction)
        artist = axes[row, 2].imshow(ratio, origin="lower", extent=extent, cmap="RdBu_r",
                                     norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
        axes[row, 2].set_title(r"$\log_{10}(\Sigma_{0.0}/\Sigma_{1.0})$")
        fig.colorbar(artist, ax=axes[row, 2], label="dex", shrink=0.86)
        axes[row, 0].set_ylabel(f"{component.capitalize()}\ny (physical kpc)")
        for ax in axes[row]:
            ax.set_aspect("equal")
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)
    for ax in axes[-1]:
        ax.set_xlabel("x (physical kpc)")

    z = 1.0 / meta["1.0"]["a"] - 1.0
    fig.suptitle(
        f"CG10 1024³, snapshot {args.snapshot:03d}, z={z:.3f}; |z|<{args.zmax:g} pkpc\n"
        f"{args.orientation_mode} stellar frame; gas/dust counts: "
        f"1.0={counts['1.0'][0]:,}/{counts['1.0'][1]:,}, "
        f"0.0={counts['0.0'][0]:,}/{counts['0.0'][1]:,}", fontsize=14)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    edges = np.linspace(0.0, args.radius, args.radial_bins + 1)
    profiles = {
        key: radial_profile(value, edges)
        for key, value in particles.items()
    }
    write_profiles(args.profile_csv, edges, profiles)
    plot_profiles(
        args.profile_output, edges, profiles, args.snapshot, args.zmax, args.solar_metallicity)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.profile_output}")
    print(f"Wrote {args.profile_csv}")
    print("Caution: ratio panels include genuine evolution plus phase shifts and, in independent mode, frame alignment.")


if __name__ == "__main__":
    main()
