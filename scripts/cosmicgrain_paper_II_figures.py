#!/usr/bin/env python3
"""Build the Paper II figures for a Halo 441 CosmicGrain run.

Run from the GADGET-4 repository root. The script derives a fresh center and
orientation from the selected snapshot, calls the existing Halo 441 map
analysis, and builds the mock-Sun comparison from the shared CosmicGrain
analysis module. It writes figures and numerical diagnostics into
``paper_II_figures``. Structure functions use independent native validity
masks and pair support for ZG and CG10.

The default snapshot is snapshot 029 in the named run. Override --snapshot
when the run is still progressing or uses a different snapshot layout.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.ndimage import convolve, gaussian_filter
from scipy.special import ndtr



# Radius of each mock Sun's circular cutout, matching the Zhang & Green map footprint (kpc).
MOCK_SUN_CUTOUT_RADIUS_KPC = 2.5

def get_perceptual_colormaps():
    """Prefer Cmasher, then Colorcet, with perceptual Matplotlib fallbacks."""
    try:
        import cmasher as cmr
        return {
            "dust": cmr.cosmic,
            "av": cmr.rainforest,
            "rv": cmr.guppy,
            "threshold": cmr.rainforest,
            "support": cmr.guppy,
        }, "Cmasher"
    except (ImportError, AttributeError):
        pass
    try:
        import colorcet as cc
        return {
            "dust": cc.cm.fire,
            "av": getattr(cc.cm, "CET_L3", cc.cm.fire),
            "rv": getattr(cc.cm, "CET_L4", cc.cm.fire),
            "threshold": getattr(cc.cm, "CET_L3", cc.cm.fire),
            "support": getattr(cc.cm, "CET_L4", cc.cm.fire),
        }, "Colorcet"
    except (ImportError, AttributeError):
        return {
            "dust": "magma",
            "av": "Greens",
            "rv": "Purples",
            "threshold": plt.get_cmap("Greens"),
            "support": plt.get_cmap("Purples"),
        }, "Matplotlib fallback (Cmasher/Colorcet unavailable)"

MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
AV_PER_TAU = 2.5 / np.log(10.0)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path,
                   default=Path("halo441_output_1024_paper"))
    p.add_argument("--snapshot", type=int, default=29)
    p.add_argument("--analysis-script", type=Path,
                   default=Path("scripts/halo441_rv_grain_physics_beta.py"))
    p.add_argument("--geometry-helper", type=Path,
                   default=Path("scripts/cosmicgrain_zg_cg10_2048_unified_v2.py"),
                   help="Shared halo-centering/orientation and Zhang--Green comparison module")
    p.add_argument("--zg-map", type=Path, default=Path("data/dust_observations/Rv_map_new.h5"),
                   help="Zhang & Green Rv_map_new.h5 input")
    p.add_argument("--group", type=int, default=1,
                   help="Target halo group index at the selected snapshot")
    p.add_argument("--carbon-table", type=Path,
                   default=Path("data/dust_optics/callqcomp.out_D16MGemt"))
    p.add_argument("--silicate-table", type=Path,
                   default=Path("data/dust_optics/callqcomp.out_DH16asil"))
    p.add_argument("--hd23-fits", type=Path,
                   default=Path("data/dust_observations/astrodust+PAH_MW_RV3.1.fits"),
                   help="Hensley & Draine (2023) MW R(V)=3.1 size-distribution FITS")
    p.add_argument("--dust-map-zoom", type=float, default=15.0,
                   help="Half-width (kpc, physical) shown in the Figure 1 dust map")
    p.add_argument("--map-zoom", type=float, default=7.5,
                   help="Half-width (kpc, physical) shown in the Figure 2 A_V/R_V maps")
    p.add_argument("--gsd-shells", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0],
                   help="Shell edges (kpc, physical) for figure 8b: consecutive pairs form the shells")
    p.add_argument("--gsd-radial-shells", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0],
                   help="Edges (kpc, physical) of consecutive shells for the radial-evolution GSD figure")
    p.add_argument("--gsd-outer-shells", type=float, nargs="+", default=[5.0, 10.0, 20.0, 30.0],
                   help="Figure 8c: first value is the inner radius excluded (kpc, physical); "
                        "panels show inner <= r < each later value")
    p.add_argument("--gsd-apertures", type=float, nargs="+", default=[20.0, 30.0, 40.0],
                   help="Spherical apertures (kpc, physical) for the central-galaxy GSD panels")
    p.add_argument("--outdir", type=Path, default=Path("paper_II_figures"))
    p.add_argument("--plot-radius", type=float, default=20.0,
                   help="Face-on half-width in physical kpc")
    p.add_argument("--observer-radius", type=float, default=None,
                   help="Galactocentric radius of the eight mock observers in kpc; "
                        "overrides --observer-radius-factor")
    p.add_argument("--observer-radius-factor", type=float, default=1.0,
                   help="Without --observer-radius, place the mock observers at this "
                        "multiple of the face-on dust half-mass radius of the map")
    p.add_argument("--radial-bin-size", type=float, default=1.0,
                   help="Annular bin width for the radial A_V/R_V profiles in kpc")
    p.add_argument("--zmax", type=float, default=5.0,
                   help="Line-of-sight half-depth in physical kpc")
    p.add_argument("--pixel", type=float, default=0.5,
                   help="Face-on pixel size in physical kpc; default 500 pc")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="Gaussian map smoothing sigma in physical kpc")
    p.add_argument("--map-min-ebv", type=float, default=0.01,
                   help="E(B-V) floor (mag) for the Figure 2 maps only")
    p.add_argument("--map-min-count", type=int, default=4,
                   help="Minimum dust particles per pixel shown in the Figure 2 R_V map")
    p.add_argument("--min-count", type=int, default=5,
                   help="Minimum dust particles per pixel for Rv and beta")
    p.add_argument("--min-ebv", type=float, default=0.03,
                   help="Minimum E(B-V) in mag for full-galaxy Rv map/histogram and Rv-beta")
    p.add_argument("--local-ebv-min", type=float, default=0.03,
                   help="Optional extra projected E(B-V) cut for mock-Sun structure functions")
    p.add_argument("--mass-unit-1e10-msun", type=float, default=1.0,
                   help="GADGET mass unit in 1e10 Msun")
    p.add_argument("--style", type=Path, default=Path("scripts/cosmicgrain.mplstyle"))
    p.add_argument("--sf-min-pairs", type=int, default=100,
                   help="Minimum pixel pairs per orientation for displaying structure-function bins")
    p.add_argument("--sf-min-orientations", type=int, default=4,
                   help="Minimum CG mock-Sun orientations supporting a separation bin")
    p.add_argument("--zg-smoothing", type=float, default=0.5,
                   help="Gaussian smoothing sigma for the ZG 3-D cube (kpc); 0.5 passes support cut")
    p.add_argument("--mock-sun-smoothing", type=float, default=0.5,
                   help="Gaussian smoothing sigma for CG mock-Sun cubes (kpc); default matches ZG")
    p.add_argument("--zg-min-support", type=float, default=0.30,
                   help="Minimum fraction of valid ZG voxels in the smoothing kernel")
    p.add_argument("--force-carbon", action="store_true",
                   help="Post-processing experiment: treat all mock-Sun dust as 100%% carbon")
    p.add_argument("--force-silicate", action="store_true",
                   help="Post-processing experiment: treat all mock-Sun dust as 100%% silicate")
    p.add_argument("--fixed-gsd", default="none",
                   help="Post-processing experiment: give every mock-Sun dust particle the same "
                        "size distribution (none | galaxy | mrn); composition unchanged")
    p.add_argument("--size-mixing", type=float, default=0.0,
                   help="Post-processing experiment: Gaussian sigma (kpc) over which each mock-Sun "
                        "dust particle takes the mass-weighted size mix of its neighbours; "
                        "composition and mass unchanged; 0 = off")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Use existing intermediate NPZ files without rerunning analysis")
    return p


def find_snapshot(run: Path, snap: int) -> str:
    name = f"snapshot_{snap:03d}"
    candidates = [
        str(run / f"{name}.*.hdf5"),
        str(run / f"{name}.hdf5"),
        str(run / f"snapdir_{snap:03d}" / f"{name}.*.hdf5"),
        str(run / f"snapdir_{snap:03d}" / f"{name}.hdf5"),
    ]
    for pattern in candidates:
        if glob.glob(pattern):
            return pattern
    raise FileNotFoundError(f"No snapshot {snap:03d} found under {run}")



def load_geometry_helper(helper_path: Path):
    """Load shared CosmicGrain snapshot, projection, and ZG-map routines."""
    if not helper_path.is_file():
        raise FileNotFoundError(
            f"Geometry helper not found: {helper_path}. It should be "
            "scripts/cosmicgrain_zg_cg10_2048_unified_v2.py or supplied with --geometry-helper.")
    helper_path = helper_path.resolve()
    sys.path.insert(0, str(helper_path.parent))
    spec = importlib.util.spec_from_file_location("cosmicgrain_paper_ii_geometry", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load geometry helper {helper_path}")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def derive_current_geometry(helper_path: Path, run: Path, snap: int, group: int, outdir: Path):
    """Derive center and face-on rotation from this run's selected snapshot."""
    base = load_geometry_helper(helper_path)
    series = base.get_zoom_halo_series(
        run, [snap], group_index=group, reference_snap=snap, verbose=True,
        refine_reference_center=True, refine_historical_centers=False)
    halo = series[snap]
    scale, h, box = base.cg.read_header(run, snap)
    rotation, source, nrot = base.cg.derive_orientation(
        run, snap, halo, scale, h, box, 20.0)
    center = np.asarray(halo["center"], dtype=float)
    outdir.mkdir(parents=True, exist_ok=True)
    center_file = outdir / f"halo441_snap{snap:03d}_current_center.txt"
    rotation_file = outdir / f"halo441_snap{snap:03d}_current_rotation_matrix.txt"
    np.savetxt(center_file, center[None, :], fmt="%.12g")
    np.savetxt(rotation_file, np.asarray(rotation), fmt="%.12g")
    print(f"Current-snapshot center: {center_file} ({center})")
    print(f"Current-snapshot orientation: {rotation_file} ({source}; N={nrot})")
    return center_file, rotation_file


def locate_zg_map(path: Path) -> Path:
    """Find the Zhang & Green HDF5 input in common repository locations."""
    if path.is_file():
        return path
    candidates = [
        Path("data/dust_observations/Rv_map_new.h5"),
        Path("data/observations/Rv_map_new.h5"),
        Path("scripts/Rv_map_new.h5"),
        Path("Rv_map_new.h5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for root in (Path("data"), Path("scripts")):
        if root.is_dir():
            matches = list(root.rglob("Rv_map_new.h5"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Zhang & Green input not found: {path}. Supply --zg-map with Rv_map_new.h5.")



def mix_size_opacities(base, xyz, mass, radius, carbon_table_data,
                       silicate_table_data, sigma_kpc, half_xy=12.0, half_z=4.0):
    """Size-mixing experiment: per-particle carbon and silicate opacities (B, V)
    replaced by their mass-weighted Gaussian-kernel mean over neighbouring dust.

    Equivalent to each particle carrying the size distribution of the dust within
    ~sigma_kpc of it. Done on a 3-D grid (cell = sigma/2, at least 25 pc) inside
    |x|,|y| < half_xy and |z| < half_z kpc, which covers the mock-Sun cubes plus
    their smoothing padding; particles outside keep their own sizes.
    """
    from scipy.ndimage import map_coordinates
    out = {}
    inside = ((np.abs(xyz[:, 0]) < half_xy) & (np.abs(xyz[:, 1]) < half_xy) &
              (np.abs(xyz[:, 2]) < half_z))
    cell = max(sigma_kpc / 2.0, 0.025)
    pad = 4.0 * sigma_kpc
    lo = np.array([-half_xy - pad, -half_xy - pad, -half_z - pad])
    shape = np.ceil((-2.0 * lo) / cell).astype(int) + 1
    if np.prod(shape) > 2.5e8:
        raise MemoryError(f"size-mixing grid {shape} too large; increase sigma")
    idx = np.floor((xyz[inside] - lo) / cell).astype(int)
    m_in = mass[inside]
    msum = np.zeros(shape)
    np.add.at(msum, tuple(idx.T), m_in)
    msum = gaussian_filter(msum, sigma_kpc / cell, mode="constant")
    coords = ((xyz[inside] - lo) / cell - 0.5).T
    mden = map_coordinates(msum, coords, order=1)
    for wave in (base.cg.B_MICRON, base.cg.V_MICRON):
        pair = []
        for table, rho in ((carbon_table_data, base.cg.RHO_C),
                           (silicate_table_data, base.cg.RHO_SIL)):
            k = 3.0 / (4.0 * radius * 1.0e-4) * base.cg.qext_at_wavelength(
                table, radius, wave) / rho
            grid = np.zeros(shape)
            np.add.at(grid, tuple(idx.T), m_in * k[inside])
            grid = gaussian_filter(grid, sigma_kpc / cell, mode="constant")
            kmix = k.copy()
            kmix[inside] = map_coordinates(grid, coords, order=1) / mden
            pair.append(kmix)
        out[wave] = tuple(pair)
    print(f"[experiment] size mixing sigma={sigma_kpc:g} kpc over {inside.sum():,} dust "
          f"particles (grid {tuple(shape)}, cell {cell*1e3:.0f} pc)")
    return out


def prepare_mock_sun_sim(base, run: Path, snap: int, group: int,
                         carbon_table: Path, silicate_table: Path,
                         pixel: float, orientation_radius: float = 20.0,
                         force_carbon: bool = False, force_silicate: bool = False,
                         fixed_gsd: str = "none",
                         size_mixing: float = 0.0):
    """Load and rotate the dust snapshot once for all mock-Sun positions."""
    series = base.get_zoom_halo_series(
        run, [snap], group_index=group, reference_snap=snap, verbose=True,
        refine_reference_center=True, refine_historical_centers=False)
    halo = series[snap]
    scale, h, box = base.cg.read_header(run, snap)
    dust = base.cg.read_dust(run, snap)
    rotation, source, nrot = base.cg.derive_orientation(
        run, snap, halo, scale, h, box, orientation_radius)
    xyz = base.periodic_delta(
        dust["coords"], np.asarray(halo["center"]), box) * scale / h @ rotation.T

    radius = dust["radius_nm"] * 1.0e-3
    carbon = np.clip(dust["carbon"], 0.0, 1.0)
    mass = dust["mass"] * 1.0e10 / h * base.MSUN
    carbon_table_data = base.cg.read_draine_qext_table(carbon_table)
    silicate_table_data = base.cg.read_draine_qext_table(silicate_table)

    # Post-processing experiments (both off by default).
    central = np.linalg.norm(xyz, axis=1) < 30.0
    if force_carbon:
        carbon = np.ones_like(carbon)
        print("[experiment] mock-Sun dust composition forced to 100% carbon")
    if force_silicate:
        if force_carbon:
            raise ValueError("--force-carbon and --force-silicate are mutually exclusive")
        carbon = np.zeros_like(carbon)
        print("[experiment] mock-Sun dust composition forced to 100% silicate")
    fixed = None
    if fixed_gsd != "none":
        table_radii = np.array(sorted(carbon_table_data), float)
        lo, hi = table_radii.min() * 1.0001, table_radii.max() * 0.9999
        if fixed_gsd == "galaxy":
            # The central galaxy's own mass-weighted size distribution (r < 30 kpc).
            gal_edges = np.logspace(np.log10(max(radius[central].min(), lo)),
                                    np.log10(min(radius[central].max(), hi)), 121)
            size_weight, _ = np.histogram(radius[central], bins=gal_edges,
                                          weights=mass[central])
            size_grid = np.sqrt(gal_edges[1:] * gal_edges[:-1])
        elif fixed_gsd == "mrn":
            # MRN dn/da ~ a^-3.5 over 5-250 nm: mass per log-radius bin ~ a^0.5.
            size_grid = np.logspace(np.log10(0.005), np.log10(0.25), 200)
            size_weight = size_grid ** 0.5
        else:
            raise ValueError(f"Unknown --fixed-gsd {fixed_gsd}")
        size_grid = np.clip(size_grid, lo, hi)
        size_weight = size_weight / size_weight.sum()
        median_nm = 1e3 * size_grid[np.searchsorted(np.cumsum(size_weight), 0.5)]
        print(f"[experiment] fixed GSD '{fixed_gsd}' for all mock-Sun dust: "
              f"mass-weighted median a = {median_nm:.1f} nm")
        fixed = (size_grid, size_weight)

    mixed = None
    if size_mixing > 0:
        if fixed is not None:
            raise ValueError("--size-mixing and --fixed-gsd are mutually exclusive")
        mixed = mix_size_opacities(base, xyz, mass, radius, carbon_table_data,
                                   silicate_table_data, size_mixing)

    def tau_weight(wavelength):
        if mixed is not None:
            kappa_c, kappa_s = mixed[wavelength]
            opacity = carbon * kappa_c + (1.0 - carbon) * kappa_s
            return mass * opacity / (pixel * base.KPC_CM) ** 2
        if fixed is not None:
            size_grid, size_weight = fixed
            kappa_c = np.sum(size_weight * 3.0 / (4.0 * size_grid * 1.0e-4) *
                             base.cg.qext_at_wavelength(carbon_table_data, size_grid, wavelength)
                             / base.cg.RHO_C)
            kappa_s = np.sum(size_weight * 3.0 / (4.0 * size_grid * 1.0e-4) *
                             base.cg.qext_at_wavelength(silicate_table_data, size_grid, wavelength)
                             / base.cg.RHO_SIL)
            opacity = carbon * kappa_c + (1.0 - carbon) * kappa_s
            return mass * opacity / (pixel * base.KPC_CM) ** 2
        q_carbon = base.cg.qext_at_wavelength(carbon_table_data, radius, wavelength)
        q_silicate = base.cg.qext_at_wavelength(silicate_table_data, radius, wavelength)
        opacity = 3.0 / (4.0 * radius * 1.0e-4) * (
            carbon * q_carbon / base.cg.RHO_C +
            (1.0 - carbon) * q_silicate / base.cg.RHO_SIL)
        return mass * opacity / (pixel * base.KPC_CM) ** 2

    weight_b = tau_weight(base.cg.B_MICRON)
    weight_v = tau_weight(base.cg.V_MICRON)
    metadata = {"orientation": source, "orientation_particles": int(nrot),
                "group": group, "snapshot": snap, "h": float(h),
                "redshift": float(1.0 / scale - 1.0)}
    return xyz, weight_b, weight_v, metadata, dust["radius_nm"], mass


def make_mock_sun_cube(base, edges, zedges, xyz, weight_b, weight_v,
                       phi_deg: float, observer_radius: float, half_size: float,
                       zmax: float, pixel: float, dz: float, smooth_sigma: float):
    """Build a local mock-Sun A_V/R_V cube with weighted Gaussian smoothing."""
    phi = np.deg2rad(phi_deg)
    er = np.array([np.cos(phi), np.sin(phi), 0.0])
    et = np.array([-np.sin(phi), np.cos(phi), 0.0])
    observer = observer_radius * er
    delta = xyz - observer
    lx = delta @ (-er)
    ly = delta @ et
    lz = delta[:, 2]

    if smooth_sigma > 0:
        padx = int(np.ceil(4.0 * smooth_sigma / pixel)) + 1
        padz = int(np.ceil(4.0 * smooth_sigma / dz)) + 1
    else:
        padx = padz = 0
    xe = edges[0] + pixel * np.arange(-padx, len(edges) + padx) if padx else edges
    ze = zedges[0] + dz * np.arange(-padz, len(zedges) + padz) if padz else zedges
    inside = ((np.abs(lx) < half_size + padx * pixel) &
              (np.abs(ly) < half_size + padx * pixel) &
              (np.abs(lz) < zmax + padz * dz))
    points = np.column_stack((lz[inside], ly[inside], lx[inside]))
    bins = (ze, xe, xe)
    tau_b = np.histogramdd(points, bins=bins, weights=weight_b[inside])[0]
    tau_v = np.histogramdd(points, bins=bins, weights=weight_v[inside])[0]
    if smooth_sigma > 0:
        sigma_pixels = (smooth_sigma / dz, smooth_sigma / pixel, smooth_sigma / pixel)
        tau_b = gaussian_filter(tau_b, sigma_pixels, mode="constant")
        tau_v = gaussian_filter(tau_v, sigma_pixels, mode="constant")
        crop = (slice(padz, -padz), slice(padx, -padx), slice(padx, -padx))
        tau_b, tau_v = tau_b[crop], tau_v[crop]

    av = 1.086 * tau_v
    r55 = np.divide(tau_v, tau_b - tau_v, out=np.full_like(tau_v, np.nan),
                    where=tau_b > tau_v)
    rv = 1.1 * r55 + 0.07
    return av, rv


def make_mock_sun_neff(edges, xyz, weight_v, phi_deg: float,
                       observer_radius: float, half_size: float, zmax: float,
                       pixel: float, smooth_sigma: float):
    """Estimate effective V-band dust-particle support per smoothed sky pixel.

    For particle contributions w_i, N_eff=(sum w_i)^2/sum(w_i^2). The z
    Gaussian's fraction inside the projected slab is included before the
    matched x-y smoothing. This is a sampling diagnostic, not a new RV mask.
    """
    phi = np.deg2rad(phi_deg)
    er = np.array([np.cos(phi), np.sin(phi), 0.0])
    et = np.array([-np.sin(phi), np.cos(phi), 0.0])
    delta = xyz - observer_radius * er
    lx = delta @ (-er)
    ly = delta @ et
    lz = delta[:, 2]

    if smooth_sigma > 0:
        padx = int(np.ceil(4.0 * smooth_sigma / pixel)) + 1
        xe = edges[0] + pixel * np.arange(-padx, len(edges) + padx)
        slab_fraction = (ndtr((zmax - lz) / smooth_sigma) -
                         ndtr((-zmax - lz) / smooth_sigma))
    else:
        padx = 0
        xe = edges
        slab_fraction = (np.abs(lz) <= zmax).astype(float)

    inside = ((np.abs(lx) < half_size + padx * pixel) &
              (np.abs(ly) < half_size + padx * pixel) &
              (slab_fraction > 0))
    # Arrays follow the projected cube convention: y is the first map axis,
    # x the second. The denominator keeps contributions from individual
    # particles separate when the field is smoothed.
    bins = (xe, xe)
    w = weight_v[inside] * slab_fraction[inside]
    sum_w = np.histogram2d(ly[inside], lx[inside], bins=bins, weights=w)[0]
    sum_w2 = np.histogram2d(ly[inside], lx[inside], bins=bins, weights=w * w)[0]

    if smooth_sigma > 0:
        sigma_pix = smooth_sigma / pixel
        sum_w = gaussian_filter(sum_w, sigma_pix, mode="constant", truncate=4.0)
        radius = int(4.0 * sigma_pix + 0.5)
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel_1d = np.exp(-0.5 * (offsets / sigma_pix) ** 2)
        kernel_1d /= kernel_1d.sum()
        kernel_2d_squared = np.outer(kernel_1d, kernel_1d) ** 2
        sum_w2 = convolve(sum_w2, kernel_2d_squared, mode="constant", cval=0.0)
        crop = (slice(padx, -padx), slice(padx, -padx))
        sum_w, sum_w2 = sum_w[crop], sum_w2[crop]

    neff = np.divide(sum_w * sum_w, sum_w2,
                     out=np.zeros_like(sum_w), where=sum_w2 > 0)
    return neff


def project_mock_sun_pair(base, zg_av, zg_rv, cg_av, cg_rv,
                          zg_selection, footprint):
    """Project ZG and CG with their independent native valid-cell masks."""
    zg_valid = (np.isfinite(zg_av) & np.isfinite(zg_rv) & (zg_av > 0) &
                (zg_rv > 0) & footprint[None])
    cg_valid = (np.isfinite(cg_av) & np.isfinite(cg_rv) & (cg_av > 0) &
                (cg_rv > 0) & footprint[None])
    projected = {}
    for label, av, rv, valid in (("ZG", zg_av, zg_rv, zg_valid),
                                  ("CG10", cg_av, cg_rv, cg_valid)):
        # The ZG reliability selection belongs only to the observed ZG cube.
        # Transferring its voxel-by-voxel pattern onto CG would condition the
        # simulation on the Milky Way's particular 3-D extinction structure.
        selected = valid & footprint[None]
        if label == "ZG":
            selected &= zg_selection
        av_map, rv_map, have_rv = base.project(av, rv, selected)
        coverage = np.any(selected, axis=0)
        rv_mask = footprint & coverage & have_rv & np.isfinite(rv_map)
        projected[label] = (av_map, rv_map, rv_mask)
    return projected, {label: projected[label][2] for label in projected}

def snapshot_hubble(snapshot_pattern: str) -> float:
    files = sorted(glob.glob(snapshot_pattern))
    if not files:
        files = [snapshot_pattern]
    with h5py.File(files[0], "r") as f:
        if "Parameters" in f and "HubbleParam" in f["Parameters"].attrs:
            return float(f["Parameters"].attrs["HubbleParam"])
        if "Header" in f and "HubbleParam" in f["Header"].attrs:
            return float(f["Header"].attrs["HubbleParam"])
    raise KeyError(f"HubbleParam is not recorded in {files[0]}")


def save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved", path)
    plt.close(fig)


def read_hd23_gsd(path: Path):
    """Read the HD23 MW R(V)=3.1 astrodust size distribution (PAH column dropped).

    CosmicGrain has no PAHs, so the comparison uses HD23's astrodust component
    alone, renormalised; the PAH column is what makes their bump near 0.6 nm.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"HD23 GSD FITS file not found: {path}. Supply --hd23-fits.")
    with fits.open(path) as hdus:
        data = np.asarray(hdus["SIZE DISTRIBUTION"].data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(f"Unexpected HD23 size-distribution shape: {data.shape}")
    radius_nm = data[:, 0] * 1000.0
    if np.any(radius_nm <= 0) or np.any(np.diff(radius_nm) <= 0):
        raise ValueError("HD23 grain radii must be positive and increasing")
    # FITS values are grains per H in each size bin. Convert to relative mass
    # with rho*a^3; no extra bin-width factor belongs in the mass weights.
    mass_weight = data[:, 1] * 2.74 * radius_nm**3
    if not np.all(np.isfinite(mass_weight)) or mass_weight.sum() <= 0:
        raise ValueError("HD23 GSD contains invalid or zero total mass")
    return radius_nm, mass_weight


def plot_gsd_comparison(radius_nm, mass_weight, xyz, hd23_fits, outdir,
                        apertures_pkpc, shells=False, stem="figure8_grain_size_distributions"):
    """Differential mass-weighted GSD of the central galaxy inside each 3D aperture."""
    good = (np.isfinite(radius_nm) & (radius_nm > 0) &
            np.isfinite(mass_weight) & (mass_weight > 0) &
            np.all(np.isfinite(xyz), axis=1))
    radius_nm = np.asarray(radius_nm)[good]
    mass_weight = np.asarray(mass_weight)[good]
    xyz = np.asarray(xyz)[good]
    central = np.linalg.norm(xyz, axis=1) < max(apertures_pkpc)
    if not np.any(good) or not np.any(central):
        raise ValueError("No valid dust particles for the requested GSD samples")

    hd_radius, hd_mass = read_hd23_gsd(hd23_fits)
    hd_log = np.log10(hd_radius)
    hd_mid = 0.5 * (hd_log[1:] + hd_log[:-1])
    hd_edges = 10.0 ** np.r_[hd_log[0] - (hd_mid[0] - hd_log[0]), hd_mid,
                              hd_log[-1] + (hd_log[-1] - hd_mid[-1])]
    hd_width = np.diff(np.log10(hd_edges))
    hd_density = hd_mass / hd_mass.sum() / hd_width

    cg_edges = np.logspace(np.log10(0.3), np.log10(6000.0), 110)
    radius3d = np.linalg.norm(xyz, axis=1)
    if shells == "inner":  # [e0,e1), [e0,e2), ...: everything outside e0 out to each radius
        shell_lo = [apertures_pkpc[0]] * (len(apertures_pkpc) - 1)
        apertures_pkpc = list(apertures_pkpc[1:])
    elif shells:  # edges -> consecutive shells [e0,e1), [e1,e2), ...
        shell_lo, apertures_pkpc = list(apertures_pkpc[:-1]), list(apertures_pkpc[1:])
    fig, axes = plt.subplots(1, len(apertures_pkpc), figsize=(4.2 * len(apertures_pkpc), 3.9),
                             sharex=True, sharey=True, constrained_layout=True)
    saved = {"cg_edges_nm": cg_edges, "hd_radius_nm": hd_radius, "hd_density": hd_density}
    for k, (ax, rcut) in enumerate(zip(np.atleast_1d(axes), apertures_pkpc)):
        lo_cut = shell_lo[k] if shells else 0.0
        sel = (radius3d >= lo_cut) & (radius3d < rcut)
        radii, weights = radius_nm[sel], mass_weight[sel]
        hist = np.histogram(radii, bins=cg_edges, weights=weights)[0]
        cg_density = hist / np.sum(weights) / np.diff(np.log10(cg_edges))
        saved[f"cg_density_{lo_cut:g}_{rcut:g}"] = cg_density
        saved[f"cg_mass_{lo_cut:g}_{rcut:g}"] = np.sum(weights)
        ax.stairs(cg_density, cg_edges, color="black", lw=2.0, label="CosmicGrain")
        ax.plot(hd_radius, hd_density, color="tab:blue", lw=2.5,
                label="Hensley & Draine (2023), astrodust")
        for target in (10.0, 100.0):
            ax.axvline(target, color="0.3", ls=(":" if target == 10.0 else "--"), lw=1.3, alpha=.8,
                       label=(("Initial SNII grains: 10 nm" if target == 10.0
                               else "Initial AGB grains: 100 nm") if k == 0 else None))
        tag = (rf"${lo_cut:g} \leq r < {rcut:g}$ kpc" if shells and lo_cut > 0
               else rf"$r < {rcut:g}$ kpc")
        ax.text(0.04, 0.96, tag, transform=ax.transAxes,
                ha="left", va="top")
        ax.set_xscale("log")
        ax.set_xlim(1.0, 3000.0)
        ax.set_xlabel("Grain radius (nm)")
        ax.grid(alpha=.15, which="both")
    np.atleast_1d(axes)[0].set_ylabel("Dust mass fraction per dex")
    np.atleast_1d(axes)[0].set_ylim(0.0, 1.3 * np.atleast_1d(axes)[0].get_ylim()[1])  # room for the legend
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper right")
    save_figure(fig, outdir / stem)
    np.savez_compressed(outdir / f"{stem}.npz", **saved)


def plot_mocksun_example(store, native_results, native_pairs, sep, sf_args, a, stem):
    """ZG vs one CG mock-Sun cutout: R_V maps, R_V distributions and S2 (abs and normalised).

    The displayed observer is the one whose R_V S2 is closest to ZG (mean |log ratio| over
    0.3-2 kpc); the S2 panels also show the median and 16-84% range over all observers.
    """
    if not store:
        return
    n = len(store)
    zg_s2 = np.asarray([native_results["zg_rv"][i][0] for i in range(n)], dtype=float)
    cg_s2 = np.asarray([native_results["cg_rv"][i][0] for i in range(n)], dtype=float)
    zg_nm = np.asarray([native_results["zg_rv"][i][1] for i in range(n)], dtype=float)
    cg_nm = np.asarray([native_results["cg_rv"][i][1] for i in range(n)], dtype=float)
    ok_pairs = ((native_pairs["cg_rv"] >= a.sf_min_pairs) &
                (native_pairs["zg_rv"] >= a.sf_min_pairs))
    cg_s2 = np.where(ok_pairs, cg_s2, np.nan); cg_nm = np.where(ok_pairs, cg_nm, np.nan)
    zg_s2 = np.where(ok_pairs, zg_s2, np.nan); zg_nm = np.where(ok_pairs, zg_nm, np.nan)
    use = (sep >= 0.3) & (sep <= 2.0)
    with np.errstate(all="ignore"):
        score = np.nanmean(np.abs(np.log10(cg_s2[:, use] / zg_s2[:, use])), axis=1)
    score = np.where(np.isfinite(score), score, np.inf)
    best = int(np.argmin(score))
    ex = store[best]
    half = sf_args.half_size
    extent = (-half, half, -half, half)
    norm = Normalize(2.5, 5.0)

    fig = plt.figure(figsize=(11.0, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    top = gs[0, :].subgridspec(1, 2)
    for k, (key, mkey, tag) in enumerate((("zg_rv", "ZG", "Milky Way (Zhang & Green)"),
                                          ("cg_rv", "CG10", rf"CG10, $R={sf_args.observer_radius:.1f}$ kpc, "
                                                            rf"$\phi={ex['phi']:.0f}^\circ$"))):
        ax = fig.add_subplot(top[0, k])
        img = np.where(ex["masks"][mkey], ex["fields"][key], np.nan)
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap="viridis", norm=norm,
                       interpolation="nearest", aspect="equal")
        ax.grid(False)
        ax.set_xlabel("x (kpc)")
        if k == 0:
            ax.set_ylabel("y (kpc)")
        else:
            ax.tick_params(labelleft=False)
        tag = f"({'ab'[k]}) " + tag
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        if k == 1:
            cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
            cb = fig.colorbar(im, cax=cax, extend="both")
            cb.set_label(r"$R_V$")

    # R_V distributions
    axc = fig.add_subplot(gs[1, 0])
    bins = np.linspace(1.5, 7.0, 56)
    zg_vals = ex["fields"]["zg_rv"][ex["masks"]["ZG"]]
    for i, item in enumerate(store):
        if i == best:
            continue
        v = item["fields"]["cg_rv"][item["masks"]["CG10"]]
        if v.size:
            axc.hist(v, bins=bins, density=True, histtype="step", color="#397da1", alpha=0.25, lw=0.8)
    cg_vals = ex["fields"]["cg_rv"][ex["masks"]["CG10"]]
    axc.hist(zg_vals, bins=bins, density=True, histtype="step", color="#222222", lw=2.2,
             label="Zhang & Green")
    axc.hist(cg_vals, bins=bins, density=True, histtype="step", color="#397da1", lw=2.2,
             label=rf"CG10, $\phi={ex['phi']:.0f}^\circ$")
    axc.plot([], [], color="#397da1", alpha=0.3, lw=0.8, label="other CG10 cutouts")
    for vals, col in ((zg_vals, "#222222"), (cg_vals, "#397da1")):
        if vals.size:
            axc.axvline(np.median(vals), color=col, ls="--", lw=1.2)
    axc.set(xlabel=r"$R_V$", ylabel="Probability density", xlim=(2.0, 6.0))
    axc.text(0.03, 0.97, "(c)", transform=axc.transAxes, ha="left", va="top")
    axc.legend(frameon=False, fontsize=9)

    # S2 panels
    right = gs[1, 1].subgridspec(2, 1, hspace=0.0)
    for row, (zc, cc, ylab) in enumerate(((zg_s2, cg_s2, r"$S_2(R_V)$"),
                                          (zg_nm, cg_nm, r"$S_2/(2\sigma^2_{R_V})$"))):
        ax = fig.add_subplot(right[row, 0])
        with np.errstate(all="ignore"):
            zmed = np.nanmedian(zc, axis=0)
            cmed = np.nanmedian(cc, axis=0)
            clo = np.nanpercentile(cc, 16, axis=0)
            chi = np.nanpercentile(cc, 84, axis=0)
        vis = np.isfinite(zmed)
        ax.plot(sep[vis], zmed[vis], "o-", color="#222222", ms=3, lw=1.8, label="Zhang & Green")
        vis = np.isfinite(cmed)
        ax.fill_between(sep[vis], clo[vis], chi[vis], color="#397da1", alpha=0.2, lw=0,
                        label="CG10 16–84% over cutouts")
        ax.plot(sep[vis], cmed[vis], "o-", color="#397da1", ms=3, lw=1.6, label="CG10 median")
        vb = np.isfinite(cc[best])
        ax.plot(sep[vb], cc[best][vb], "--", color="#1d4e6b", lw=1.6,
                label=rf"CG10 $\phi={ex['phi']:.0f}^\circ$")
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.text(0.03, 0.95, ("(d1)", "(d2)")[row], transform=ax.transAxes, ha="left", va="top")
        ax.grid(alpha=.15, which="both")
        if row == 0:
            ax.tick_params(labelbottom=False)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        else:
            ax.set_xlabel("Separation (kpc)")
    save_figure(fig, stem)
    print(f"Mock-Sun example: displayed phi={ex['phi']:.0f} deg (S2 |log ratio| {score[best]:.2f} dex; "
          f"all observers {np.round(score, 2).tolist()})")


def axis_map(ax, image, radius, cmap, norm=None):
    return ax.imshow(image, origin="lower", extent=(-radius, radius, -radius, radius),
                     interpolation="nearest", cmap=cmap, norm=norm, aspect="equal")


def main():
    a = args_parser().parse_args()
    if a.min_ebv < 0 or a.local_ebv_min < 0:
        raise ValueError("--min-ebv and --local-ebv-min must be non-negative")
    if ((a.observer_radius is not None and a.observer_radius <= 0)
            or a.observer_radius_factor <= 0 or a.radial_bin_size <= 0):
        raise ValueError("--observer-radius, --observer-radius-factor and "
                         "--radial-bin-size must be positive")
    if a.sf_min_pairs < 1 or a.sf_min_orientations < 1 or a.sf_min_orientations > 8:
        raise ValueError("Structure-function support requires positive pair counts and 1–8 orientations")
    if a.style.is_file():
        plt.style.use(str(a.style))
    cmaps, cmap_source = get_perceptual_colormaps()
    print(f"Using {cmap_source} perceptual colormaps")
    a.outdir.mkdir(parents=True, exist_ok=True)
    # Remove outputs from earlier revisions that compared both datasets on a
    # shared pixel mask; leaving them beside Figure 5 could imply they remain
    # part of the current analysis.
    for obsolete in (
            "figure5b_common_mask_diagnostic.png",
            "figure5b_common_mask_diagnostic.pdf",
            "structure_function_pair_support_zg_common.csv",
            "structure_function_pair_support_cg10_common.csv"):
        stale_path = a.outdir / obsolete
        if stale_path.is_file():
            stale_path.unlink()
    intermediates = a.outdir / "_analysis"
    intermediates.mkdir(parents=True, exist_ok=True)
    prefix = intermediates / f"halo441_snap{a.snapshot:03d}"
    map_path = Path(str(prefix) + "_heatmap_data.npz")
    diag_path = Path(str(prefix) + "_grain_rv_diagnostics.npz")
    snapshot = find_snapshot(a.run, a.snapshot)

    if not a.skip_analysis:
        required = [a.analysis_script, a.geometry_helper, a.carbon_table, a.silicate_table]
        missing = [str(x) for x in required if not x.is_file()]
        if missing:
            raise FileNotFoundError("Required analysis inputs not found: " + ", ".join(missing))
        center_file, rotation_file = derive_current_geometry(
            a.geometry_helper, a.run, a.snapshot, a.group, intermediates)
        bins_float = 2 * a.plot_radius / a.pixel
        bins = int(round(bins_float))
        if not np.isclose(bins_float, bins):
            raise ValueError("2*plot-radius/pixel must be an integer")
        cmd = [sys.executable, str(a.analysis_script), snapshot,
               "--center-file", str(center_file),
               "--rotation-file", str(rotation_file),
               "--carbon-table", str(a.carbon_table),
               "--silicate-table", str(a.silicate_table),
               "--plot-radius", str(a.plot_radius), "--zmax", str(a.zmax),
               "--bins", str(bins), "--smooth", str(a.smooth),
               "--diagnostic-min-count", str(a.min_count),
               "--prefix", str(prefix)]
        print("Running face-on dust and extinction analysis:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    if not map_path.is_file() or not diag_path.is_file():
        raise FileNotFoundError(f"Expected analysis outputs {map_path} and {diag_path}")

    h = snapshot_hubble(snapshot)
    with np.load(map_path, allow_pickle=False) as d:
        tauv_raw = np.asarray(d["tauV"], float)
        taub_raw = np.asarray(d["tauB"], float)
        rv_map = np.asarray(d["Rv"], float)
        dustmass_code = np.asarray(d["dustmass_raw"], float)
        count = np.asarray(d["dust_particle_count_raw"], float)
        radius = float(d["plot_radius_pkpc"])
        pixel = float(d["pixel_size_pkpc"])
    with np.load(diag_path, allow_pickle=False) as d:
        diag_valid = np.asarray(d["valid_mask"], bool)
        beta = np.asarray(d["beta_kappa_250_850"], float)
        beta_valid = np.asarray(d["beta_valid_mask"], bool)
        beta_wave = np.asarray(d["beta_kappa_wavelengths_um"], float)
        carbon_fraction = np.asarray(d["carbon_mass_fraction"], float)
        small_grain_fraction = np.asarray(d["small_grain_mass_fraction"], float)
        median_grain_radius_nm = np.asarray(d["median_grain_radius_nm"], float)

    # The analysis map stores the proportional opacity sum (mass/radius).
    # Convert it to tau using C_ext=3 M Q/(4 rho a), then Av=1.086 tau_V.
    mass_unit_g = a.mass_unit_1e10_msun * 1.0e10 * MSUN_G / h
    pixel_area_cm2 = (pixel * KPC_CM) ** 2
    tau_factor = 0.75 * mass_unit_g / (1.0e-4 * pixel_area_cm2)
    tauv = tauv_raw * tau_factor
    taub = taub_raw * tau_factor
    av = AV_PER_TAU * tauv
    dustmass_msun = dustmass_code * a.mass_unit_1e10_msun * 1.0e10 / h
    ebv = AV_PER_TAU * (taub - tauv)
    valid_rv = ((count >= a.min_count) & diag_valid & np.isfinite(rv_map) &
                (rv_map > 0) & np.isfinite(ebv) & (ebv >= a.min_ebv))
    valid_beta = (valid_rv & beta_valid & np.isfinite(beta) &
                  np.isfinite(carbon_fraction))

    # Place the mock observers relative to the dust disk's own size, so runs
    # whose disks differ chaotically in extent are sampled at the same place.
    centres = np.arange(-radius + 0.5 * pixel, radius, pixel)
    pixel_r = np.hypot(*np.meshgrid(centres, centres, indexing="ij")).ravel()
    order = np.argsort(pixel_r)
    cumulative = np.cumsum(np.nan_to_num(dustmass_code.ravel()[order]))
    dust_half_mass_radius = float(pixel_r[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
    if a.observer_radius is None:
        a.observer_radius = a.observer_radius_factor * dust_half_mass_radius
        print(f"Mock observers at {a.observer_radius_factor:g} x face-on dust half-mass "
              f"radius = {a.observer_radius:.2f} kpc")
    else:
        print(f"Mock observers at fixed R = {a.observer_radius:g} kpc "
              f"(face-on dust half-mass radius {dust_half_mass_radius:.2f} kpc)")

    # 1. Face-on dust distribution: physical dust mass per projected pixel.
    logmass = np.where(dustmass_msun > 0, np.log10(np.maximum(dustmass_msun, 1e-300)), np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    im = axis_map(ax, logmass, radius, cmaps["dust"], Normalize(-1, 5))
    ax.set(xlabel="Face-on x (kpc)", ylabel="Face-on y (kpc)",
           xlim=(-a.dust_map_zoom, a.dust_map_zoom), ylim=(-a.dust_map_zoom, a.dust_map_zoom))
    pixel_edges = np.arange(-radius, radius + 0.5 * pixel, pixel)
    ax.set_xticks(pixel_edges, minor=True)
    ax.set_yticks(pixel_edges, minor=True)
    ax.grid(which="minor", color="white", alpha=.20, linewidth=.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-a.dust_map_zoom, a.dust_map_zoom)  # after set_xticks, which widens the limits
    ax.set_ylim(-a.dust_map_zoom, a.dust_map_zoom)
    cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])  # exactly the height of the map
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\log_{10}(M_{\rm dust}/M_\odot)$ per pixel")
    save_figure(fig, a.outdir / "figure1_faceon_dust_distribution")

    # 2. A_V and R_V maps, with physically converted A_V.
    ebv_floor_mask = np.isfinite(ebv) & (ebv >= a.min_ebv)
    # Figure 2 shows coverage with its own (looser) E(B-V) floor, --map-min-ebv;
    # all R_V statistics keep --min-ebv.
    map_ebv_mask = np.isfinite(ebv) & (ebv >= a.map_min_ebv)
    av_show = np.where(map_ebv_mask & np.isfinite(av) & (av > 0), av, np.nan)
    # Figure 2 may use a looser particle-count cut than the quantitative R_V
    # selections (--min-count), purely to show the map's spatial coverage.
    valid_rv_map = ((count >= a.map_min_count) & diag_valid & np.isfinite(rv_map) &
                    (rv_map > 0) & map_ebv_mask)
    rv_show = np.where(valid_rv_map, rv_map, np.nan)
    print(f"Figure 2 R_V map: N_dust>={a.map_min_count}, E(B-V)>={a.map_min_ebv:g}: {int(valid_rv_map.sum())} pixels "
          f"(N_dust>={a.min_count}: {int(valid_rv.sum())})")
    av_top = float(np.nanpercentile(av_show, 99)) if np.any(np.isfinite(av_show)) else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.6), sharey=True, constrained_layout=True)
    ima = axis_map(axes[0], av_show, radius, cmaps["av"], Normalize(0, av_top))
    imr = axis_map(axes[1], rv_show, radius, cmaps["rv"], Normalize(2, 6))
    for ax, tag in zip(axes, (r"$A_V$", r"$R_V$")):
        ax.set(xlabel="Face-on x (kpc)",
               xlim=(-a.map_zoom, a.map_zoom), ylim=(-a.map_zoom, a.map_zoom))
        ax.text(0.04, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontsize=14, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
    axes[0].set_ylabel("Face-on y (kpc)")
    axes[1].tick_params(axis="y", labelleft=False)
    # Mark the eight mock-Sun observer positions used in Figure 5. The ring
    # indicates their galactocentric radius; these marks are locations, not
    # claims that the two figures use identical pixel masks.
    observer_phi = np.deg2rad(np.arange(8) * 45.0)
    observer_x = a.observer_radius * np.cos(observer_phi)
    observer_y = a.observer_radius * np.sin(observer_phi)
    for ax in axes:
        ax.add_patch(Circle((0, 0), a.observer_radius, fill=False, linestyle="--",
                            linewidth=1.0, edgecolor="white", alpha=.8))
        for ox, oy in zip(observer_x, observer_y):  # each mock Sun's Zhang & Green-sized cutout
            ax.add_patch(Circle((ox, oy), MOCK_SUN_CUTOUT_RADIUS_KPC, fill=False, linestyle="-",
                                linewidth=0.7, edgecolor="white", alpha=.6))
        ax.scatter(observer_x, observer_y, marker="+", s=45, linewidths=1.2,
                   color="white", zorder=5)
    # Horizontal colorbars on top; their labels serve as the panel titles.
    for im, ax in ((ima, axes[0]), (imr, axes[1])):
        fig.colorbar(im, ax=ax, location="top", extend="both", shrink=.92,
                     aspect=30, pad=.02)
    save_figure(fig, a.outdir / "figure2_av_rv_colormaps")

    # Show the R_V map's sensitivity to the raw dust-particle-count cut.
    # This is a diagnostic alongside the adopted map, not a replacement for it.
    rv_base = (np.isfinite(rv_map) & (rv_map > 0) & np.isfinite(ebv) &
               (ebv >= a.min_ebv))
    count_cuts = (1, 5, 20)
    fig, axes = plt.subplots(1, len(count_cuts), figsize=(5.2 * len(count_cuts), 5.1),
                             squeeze=False, constrained_layout=True)
    map_count_pixels = {}
    for ax, ncut in zip(axes[0], count_cuts):
        count_mask = rv_base & (count >= ncut)
        map_count_pixels[ncut] = int(count_mask.sum())
        rv_cut = np.where(count_mask, rv_map, np.nan)
        im = axis_map(ax, rv_cut, radius, cmaps["rv"], Normalize(2, 10))
        ax.set(title=rf"$N_{{dust}}\geq{ncut}$ ({map_count_pixels[ncut]:,} pixels)",
               xlabel="Face-on x (kpc)", ylabel="Face-on y (kpc)")
        ax.add_patch(Circle((0, 0), a.observer_radius, fill=False, linestyle="--",
                            linewidth=1.0, edgecolor="white", alpha=.8))
        ax.scatter(observer_x, observer_y, marker="+", s=45, linewidths=1.2,
                   color="white", zorder=5)
    fig.colorbar(im, ax=axes[0].tolist(), label=r"$R_V=A_V/E(B-V)$", shrink=.86)
    fig.suptitle(rf"$R_V$ count-cut sensitivity ($E(B-V)\geq{a.min_ebv:g}$ mag; {pixel*1000:g} pc pixels)"
                 + "\n"
                 + rf"White ring and crosses mark mock-observer positions at $R={a.observer_radius:.2f}$ kpc")
    save_figure(fig, a.outdir / "figure2b_rv_count_cut_sensitivity")

    # 3. A_V and R_V distributions for the exact same selected map pixels.
    # Drawn after the mock-Sun comparison, which supplies the Zhang & Green
    # medians marked in each panel.
    av_vals = av[valid_rv]
    rv_vals = rv_map[valid_rv]

    def plot_figure3(zg_av_median, zg_rv_median, zg_av_vals=None, zg_rv_vals=None):
      # Corner plot: A_V histogram (top), R_V histogram (right, rotated) and
      # the joint A_V-R_V distribution as density contours (bottom left).
      from scipy.stats import gaussian_kde
      fig = plt.figure(figsize=(7.6, 7.4), constrained_layout=True)
      gs = fig.add_gridspec(2, 2)
      ax_j = fig.add_subplot(gs[1, 0])
      ax_a = fig.add_subplot(gs[0, 0], sharex=ax_j)
      ax_r = fig.add_subplot(gs[1, 1], sharey=ax_j)
      av_color = plt.get_cmap(cmaps["av"])(0.68)
      rv_color = plt.get_cmap(cmaps["rv"])(0.62)
      av_hi = 3.0
      rv_lo, rv_hi = 1.0, max(float(np.nanpercentile(rv_vals, 99)) * 1.05, 6.0) if len(rv_vals) else 8.0
      ax_a.hist(av_vals, bins=np.linspace(0, av_hi, 30), color=av_color, lw=0)
      ax_r.hist(rv_vals, bins=np.linspace(rv_lo, rv_hi, 30), color=rv_color, lw=0,
                orientation="horizontal")
      med_av, med_rv = np.median(av_vals), np.median(rv_vals)
      ax_a.axvline(med_av, color="#1f5fbf", lw=2, label=f"CG median {med_av:.2f}")
      ax_r.axhline(med_rv, color="#e67e22", lw=2, label=f"CG median {med_rv:.2f}")
      if np.isfinite(zg_av_median):
          ax_a.axvline(zg_av_median, color="black", lw=2, ls="--", label=f"ZG median {zg_av_median:.2f}")
      if np.isfinite(zg_rv_median):
          ax_r.axhline(zg_rv_median, color="black", lw=2, ls="--", label=f"ZG median {zg_rv_median:.2f}")
      ax_a.legend(frameon=False)
      ax_r.legend(frameon=False, loc="upper right")

      def kde_contours(ax, xv, yv, color, ls, fill):
          ok = np.isfinite(xv) & np.isfinite(yv)
          xv, yv = xv[ok], yv[ok]
          if len(xv) < 10:
              return
          kde = gaussian_kde(np.vstack((xv, yv)))
          kde.set_bandwidth(max(kde.factor, 0.25))  # keep large-N contours smooth
          gx, gy = np.meshgrid(np.linspace(0, av_hi, 160), np.linspace(rv_lo, rv_hi, 160))
          z = kde(np.vstack((gx.ravel(), gy.ravel()))).reshape(gx.shape)
          zs = np.sort(z.ravel())[::-1]
          cum = np.cumsum(zs) / zs.sum()
          # levels enclosing 86%, 68% and 39% of the probability (2D 2-, 1.5-, 1-sigma)
          levels = sorted(zs[np.searchsorted(cum, q)] for q in (0.86, 0.68, 0.39))
          if fill:
              ax.contourf(gx, gy, z, levels=levels + [z.max()], colors=[color], alpha=.18)
          ax.contour(gx, gy, z, levels=levels, colors=[color], linestyles=ls, linewidths=1.4)

      ax_j.scatter(av_vals, rv_vals, s=6, color=rv_color, alpha=.35, lw=0, rasterized=True)
      kde_contours(ax_j, np.asarray(av_vals), np.asarray(rv_vals), rv_color, "solid", True)
      handles = [plt.Line2D([], [], color=rv_color, lw=1.6, label="CosmicGrain")]
      if zg_av_vals is not None and len(zg_av_vals) >= 10:
          kde_contours(ax_j, np.asarray(zg_av_vals), np.asarray(zg_rv_vals), "black", "dashed", False)
          handles.append(plt.Line2D([], [], color="black", lw=1.4, ls="--", label="Zhang & Green (2025)"))
      ax_j.legend(handles=handles, frameon=False, loc="upper right")
      ax_j.set(xlabel=r"$A_V$ (mag)", ylabel=r"$R_V$", xlim=(0, av_hi), ylim=(rv_lo, rv_hi))
      ax_a.set_ylabel("Pixels")
      ax_r.set_xlabel("Pixels")
      ax_a.tick_params(axis="x", labelbottom=False)
      ax_r.tick_params(axis="y", labelleft=False)
      for ax in (ax_j, ax_a, ax_r):
          ax.grid(alpha=.18)
          ax.set_box_aspect(1)
      save_figure(fig, a.outdir / "figure3_rv_histogram")

    # 4. Intrinsic opacity slope beta (250--850 micron) versus optical R_V.
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    x, y = rv_map[valid_beta], beta[valid_beta]
    if len(x):
        scatter = ax.scatter(x, y, c=carbon_fraction[valid_beta], s=18, alpha=.48,
                             cmap="viridis", vmin=0.0, vmax=1.0,
                             edgecolors="none", rasterized=True,
                             label=f"CosmicGrain pixels (N={len(x):,})")
        fig.colorbar(scatter, ax=ax, label="Dust carbon mass fraction")
        order = np.argsort(x)
        groups = np.array_split(order, min(10, max(1, len(order) // 10)))
        n_groups = len(groups)
        gx, gy, lo, hi = [], [], [], []
        for group in groups:
            if len(group) < 3:
                continue
            gx.append(np.median(x[group])); gy.append(np.median(y[group]))
            lo.append(np.percentile(y[group], 16)); hi.append(np.percentile(y[group], 84))
        if gx:
            gy = np.asarray(gy); lo = np.asarray(lo); hi = np.asarray(hi)
            ax.errorbar(gx, gy, yerr=np.vstack((gy-lo, hi-gy)), fmt="o-", ms=5,
                        lw=1.8, color="#b43c2f", capsize=2.5,
                        label=rf"CosmicGrain binned medians (16–84%)")
    # Milky Way: Schlafly et al. (2016) R(V)-beta(Planck) anticorrelation, as
    # fitted by Zelko & Finkbeiner (2020, eq. 24): beta = 2.82 - 0.36 R_V, over
    # the R_V range their fit samples (2.94-3.67; S16 R'(V) scale, ~0.2 above
    # standard R(V)).  The main panel is scaled to the CG data; the inset puts
    # CG and the Milky Way relation on a common beta axis.
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set(xlabel=r"$R_V$", ylabel=rf"$β$ ({beta_wave.min():g}–{beta_wave.max():g} $\mu$m)")
    ins = ax.inset_axes([0.60, 0.58, 0.36, 0.38])
    rv_s16 = np.linspace(2.94, 3.67, 50)
    if len(x):
        ins.scatter(x, y, s=4, color="#5e4fa2", alpha=.5, lw=0, rasterized=True)
    ins.plot(rv_s16, 2.82 - 0.36 * rv_s16, color="black", lw=2.0)
    ins.text(0.97, 0.06, "Milky Way\n(Schlafly+16)", transform=ins.transAxes,
             ha="right", va="bottom", fontsize=7)
    ins.text(0.97, 0.93, "CG", transform=ins.transAxes, ha="right", va="top",
             fontsize=7, color="#5e4fa2")
    ins.set_xlim(1.5, 6.0)
    ins.set_ylim(1.4, 2.0)
    ins.tick_params(labelsize=7)
    ins.set_xlabel(r"$R_V$", fontsize=8, labelpad=1)
    ins.set_ylabel(r"$β$", fontsize=8, labelpad=1)
    ins.grid(alpha=.15)
    ax.grid(alpha=.18)
    save_figure(fig, a.outdir / "figure4_rv_beta")

    # 4b. Per-pixel dust properties versus R_V for this CG10 run.
    rv_property_titles = (
        "Small-grain fraction",
        "Carbon Fraction",
    )
    rv_property_ylabels = (
        r"$M(a<50\,\mathrm{nm})/M_{\rm dust}$",
        r"$M_{\rm C}/M_{\rm dust}$",
    )
    rv_property_arrays = (small_grain_fraction, carbon_fraction)
    rv_property_valid = (valid_rv & np.isfinite(small_grain_fraction) &
                         np.isfinite(carbon_fraction))
    rv_property_rv = rv_map[rv_property_valid]
    n_rv_property_pixels = int(rv_property_valid.sum())
    rv_property_xmax = max(10.0, float(np.ceil(np.nanmax(rv_property_rv)))
                           if n_rv_property_pixels else 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), constrained_layout=True)
    property_color = "#25845b"
    for ax, values, title, ylabel in zip(
            axes, rv_property_arrays, rv_property_titles, rv_property_ylabels):
        ax.scatter(rv_property_rv, values[rv_property_valid], s=18, alpha=.75,
                   color=property_color, edgecolors="white", linewidths=0.3)
        ax.set(xlim=(1.0, rv_property_xmax), title=title,
               xlabel=r"Face-on sightline $R_V$", ylabel=ylabel)
        ax.grid(alpha=.18)
    save_figure(fig, a.outdir / "figure7_cg10_rv_grain_properties")
    rv_property_indices = np.argwhere(rv_property_valid)
    np.savetxt(
        a.outdir / "figure7_cg10_rv_grain_properties.csv",
        np.column_stack((rv_property_indices, rv_property_rv, ebv[rv_property_valid],
                         count[rv_property_valid], small_grain_fraction[rv_property_valid],
                         carbon_fraction[rv_property_valid])),
        delimiter=",", fmt="%.8g",
        header=("pixel_i,pixel_j,RV,E_BV_mag,N_dust,small_grain_mass_fraction_a_lt_50nm,"
                "carbon_mass_fraction; "
                f"selected pixels require N_dust>={a.min_count} and "
                f"E(B-V)>={a.min_ebv:g} mag"), comments="")
    print(f"CG10 grain-property scatter points = {n_rv_property_pixels:,} selected sightlines")

    # 5. Zhang & Green / mock-Sun A_V and R_V structure functions, using
    #    each dataset's own valid pixels and pair support.
    zg_map = locate_zg_map(a.zg_map)
    base = load_geometry_helper(a.geometry_helper)
    sf_args = argparse.Namespace(
        zg_h5=zg_map, root=Path("."), output_dir=a.run, snapshot=a.snapshot,
        group=a.group, carbon_table=a.carbon_table, silicate_table=a.silicate_table,
        nside="combined", half_size=MOCK_SUN_CUTOUT_RADIUS_KPC, radius=MOCK_SUN_CUTOUT_RADIUS_KPC, zmax=0.4, pixel=0.25,
        dz=0.2, sim_sigma=a.mock_sun_smoothing, match_smoothing=True,
        native_no_smoothing=False, zg_min_support=a.zg_min_support,
        zg_smoothing=a.zg_smoothing, cut_quantile=0.10, completeness_bins=18,
        observer_radius=a.observer_radius, n_observers=8, observer_phase=0.0)
    sf_args.root = sf_args.root.resolve()
    if not sf_args.zg_h5.is_absolute():
        sf_args.zg_h5 = sf_args.root / sf_args.zg_h5
    if not sf_args.output_dir.is_absolute():
        sf_args.output_dir = sf_args.root / sf_args.output_dir
    for key in ("carbon_table", "silicate_table"):
        value = getattr(sf_args, key)
        if not value.is_absolute():
            value = sf_args.root / value
        if not value.is_file():
            raise FileNotFoundError(f"Optical constants not found: {value}")
        setattr(sf_args, key, value)
    edges, zedges, xx, yy, zcenters = base.grid(sf_args)
    # Use the same smoothing for both datasets by default. The ZG support
    # filtering is applied before measuring its P10 reliability threshold.
    cg_sigma = sf_args.sim_sigma
    sf_args.sim_sigma = sf_args.zg_smoothing
    zg_av_3d, zg_rv_3d = base.zg_cube(sf_args, xx, yy, zcenters)
    sf_args.sim_sigma = cg_sigma
    footprint = np.hypot(xx, yy) <= sf_args.radius
    zg_valid = (np.isfinite(zg_av_3d) & np.isfinite(zg_rv_3d) &
                (zg_av_3d > 0) & (zg_rv_3d > 0) & footprint[None])
    if np.count_nonzero(zg_valid) < 20:
        raise RuntimeError(
            f"Only {np.count_nonzero(zg_valid)} supported positive ZG cells remain "
            f"(sigma_ZG={sf_args.zg_smoothing:g} kpc, min_support={sf_args.zg_min_support:g}). "
            "Try the validated --zg-smoothing 0.5 setting or inspect the ZG support mask.")
    zg_cut = float(np.quantile(zg_av_3d[zg_valid], sf_args.cut_quantile))
    zg_select = zg_valid & (zg_av_3d >= zg_cut)
    xyz, weight_b, weight_v, _sf_meta, particle_radius_nm, particle_mass_weight = prepare_mock_sun_sim(
        base, sf_args.output_dir, a.snapshot, a.group,
        sf_args.carbon_table, sf_args.silicate_table, sf_args.pixel,
        force_carbon=a.force_carbon, force_silicate=a.force_silicate, fixed_gsd=a.fixed_gsd, size_mixing=a.size_mixing)
    plot_gsd_comparison(particle_radius_nm, particle_mass_weight, xyz,
                        a.hd23_fits, a.outdir, a.gsd_apertures)
    plot_gsd_comparison(particle_radius_nm, particle_mass_weight, xyz,
                        a.hd23_fits, a.outdir, a.gsd_shells, shells=True,
                        stem="figure8b_grain_size_distributions_shells")
    plot_gsd_comparison(particle_radius_nm, particle_mass_weight, xyz,
                        a.hd23_fits, a.outdir, a.gsd_outer_shells, shells="inner",
                        stem="figure8c_grain_size_distributions_outside_inner")
    plot_gsd_comparison(particle_radius_nm, particle_mass_weight, xyz,
                        a.hd23_fits, a.outdir, a.gsd_radial_shells, shells=True,
                        stem="figure8d_grain_size_distributions_radial")
    phi_deg = (sf_args.observer_phase +
               np.arange(sf_args.n_observers) * 360.0 / sf_args.n_observers) % 360.0
    # The selected footprint is circular, so its largest possible pair
    # separation is its diameter (the square-corner distance is not physical).
    max_sep = 2.0 * sf_args.radius
    sf_edges = np.arange(0.0, max_sep + sf_args.pixel, sf_args.pixel)
    sf_sep = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    keys = ("zg_av", "cg_av", "zg_rv", "cg_rv")
    native_results = {key: [] for key in keys}
    native_pairs = {key: [] for key in keys}
    native_pixels = {"ZG": [], "CG10": []}
    local_ebv_counts = {
        threshold: {"ZG": [], "CG10": []}
        for threshold in (a.min_ebv,)}
    local_rv_sensitivity = {
        threshold: {"zg_s2": [], "cg_s2": [], "zg_norm": [], "cg_norm": [],
                   "zg_pairs": [], "cg_pairs": [], "zg_sigma": [], "cg_sigma": []}
        for threshold in local_ebv_counts}
    neff_thresholds = (1.0, 2.0, 5.0, 10.0)
    cg_reliability = {
        ncut: {
            ebvcut: {"s2": [], "norm": [], "sigma": [], "pixels": [], "pairs": []}
            for ebvcut in local_ebv_counts}
        for ncut in neff_thresholds}
    sigma = {key: [] for key in keys}
    zg_hist_medians = None
    zg_hist_values = (None, None)

    def safe_structure_function(field, mask):
        valid = mask & np.isfinite(field)
        if np.count_nonzero(valid) < 2:
            nbin = len(sf_edges) - 1
            return (np.full(nbin, np.nan), np.zeros(nbin, dtype=np.int64),
                    np.full(nbin, np.nan), np.nan, int(np.count_nonzero(valid)))
        return base.structure_function(field, mask, sf_args.pixel, sf_edges)

    example_store = []  # per-observer projected maps for the example-cutout figure
    for phi in phi_deg:
        sim_av, sim_rv = make_mock_sun_cube(
            base, edges, zedges, xyz, weight_b, weight_v, phi,
            sf_args.observer_radius, sf_args.half_size, sf_args.zmax,
            sf_args.pixel, sf_args.dz, sf_args.sim_sigma)
        cg_neff = make_mock_sun_neff(
            edges, xyz, weight_v, phi, sf_args.observer_radius,
            sf_args.half_size, sf_args.zmax, sf_args.pixel, sf_args.sim_sigma)
        projected, native_masks = project_mock_sun_pair(
            base, zg_av_3d, zg_rv_3d, sim_av, sim_rv, zg_select, footprint)
        fields = {"zg_av": projected["ZG"][0], "zg_rv": projected["ZG"][1],
                  "cg_av": projected["CG10"][0], "cg_rv": projected["CG10"][1]}
        zg_ebv = np.divide(fields["zg_av"], fields["zg_rv"],
                           out=np.full_like(fields["zg_av"], np.nan),
                           where=np.isfinite(fields["zg_rv"]) & (fields["zg_rv"] > 0))
        cg_ebv = np.divide(fields["cg_av"], fields["cg_rv"],
                           out=np.full_like(fields["cg_av"], np.nan),
                           where=np.isfinite(fields["cg_rv"]) & (fields["cg_rv"] > 0))
        ebv_by_dataset = {"ZG": zg_ebv, "CG10": cg_ebv}
        native_before_ebv = {dataset: mask.copy()
                             for dataset, mask in native_masks.items()}
        cg_native_before_ebv = native_before_ebv["CG10"]
        for dataset in native_masks:
            native_masks[dataset] &= (np.isfinite(ebv_by_dataset[dataset]) &
                                      (ebv_by_dataset[dataset] >= a.local_ebv_min))
            native_pixels[dataset].append(int(native_masks[dataset].sum()))
        if zg_hist_medians is None and np.any(native_masks["ZG"]):
            zg_hist_values = (fields["zg_av"][native_masks["ZG"]], fields["zg_rv"][native_masks["ZG"]])
            zg_hist_medians = (float(np.median(fields["zg_av"][native_masks["ZG"]])),
                               float(np.median(fields["zg_rv"][native_masks["ZG"]])))
        for threshold in local_ebv_counts:
            threshold_masks = {
                "ZG": native_before_ebv["ZG"] & (zg_ebv >= threshold),
                "CG10": native_before_ebv["CG10"] & (cg_ebv >= threshold),
            }
            for dataset in threshold_masks:
                local_ebv_counts[threshold][dataset].append(
                    int(threshold_masks[dataset].sum()))
            zsf, zpairs, znorm, zstd, _zn = safe_structure_function(
                fields["zg_rv"], threshold_masks["ZG"])
            csf, cpairs, cnorm, cstd, _cn = safe_structure_function(
                fields["cg_rv"], threshold_masks["CG10"])
            local_rv_sensitivity[threshold]["zg_s2"].append(zsf)
            local_rv_sensitivity[threshold]["cg_s2"].append(csf)
            local_rv_sensitivity[threshold]["zg_norm"].append(znorm)
            local_rv_sensitivity[threshold]["cg_norm"].append(cnorm)
            local_rv_sensitivity[threshold]["zg_pairs"].append(zpairs)
            local_rv_sensitivity[threshold]["cg_pairs"].append(cpairs)
            local_rv_sensitivity[threshold]["zg_sigma"].append(zstd)
            local_rv_sensitivity[threshold]["cg_sigma"].append(cstd)

            cg_test_mask = (cg_native_before_ebv & (cg_neff >= 0) &
                            (cg_ebv >= threshold))
            for ncut in neff_thresholds:
                reliability_mask = cg_test_mask & (cg_neff >= ncut)
                rsf, rpairs, rnorm, rsigma, _rn = safe_structure_function(
                    fields["cg_rv"], reliability_mask)
                record = cg_reliability[ncut][threshold]
                record["s2"].append(rsf)
                record["norm"].append(rnorm)
                record["sigma"].append(rsigma)
                record["pixels"].append(int(reliability_mask.sum()))
                record["pairs"].append(rpairs)

        example_store.append({"phi": float(phi),
                              "fields": {k: v.copy() for k, v in fields.items()},
                              "masks": {k: v.copy() for k, v in native_masks.items()}})
        for key, field in fields.items():
            dataset = "ZG" if key.startswith("zg_") else "CG10"
            native_sf, native_npair, native_norm, std, _n = safe_structure_function(
                field, native_masks[dataset])
            native_results[key].append((native_sf, native_norm))
            native_pairs[key].append(native_npair)
            sigma[key].append(std)

    native_pairs = {key: np.asarray(value, dtype=float)
                    for key, value in native_pairs.items()}
    plot_mocksun_example(example_store, native_results, native_pairs, sf_sep,
                         sf_args, a, a.outdir / "figure5f_mocksun_example")
    plot_figure3(*(zg_hist_medians or (np.nan, np.nan)), *zg_hist_values)
    if zg_hist_medians:
        print(f"Zhang & Green medians marked in figure 3 (local E(B-V)>={a.local_ebv_min:g}): "
              f"A_V={zg_hist_medians[0]:.3f} mag, R_V={zg_hist_medians[1]:.3f}")

    def ensemble(results_set, key, component):
        return np.asarray([item[component] for item in results_set[key]], dtype=float)

    def finite_median(values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else np.nan

    def curve_percentile(curves, percentile):
        curves = np.asarray(curves, dtype=float)
        output = np.full(curves.shape[1], np.nan)
        for j in range(curves.shape[1]):
            finite = curves[:, j][np.isfinite(curves[:, j])]
            if finite.size:
                output[j] = np.percentile(finite, percentile)
        return output

    def summarize_support(pair_arrays, min_views):
        nvalid = np.sum(pair_arrays >= a.sf_min_pairs, axis=0)
        med = np.nanmedian(pair_arrays, axis=0)
        p16 = np.nanpercentile(pair_arrays, 16, axis=0)
        p84 = np.nanpercentile(pair_arrays, 84, axis=0)
        shown = ((nvalid >= min_views) & np.isfinite(sf_sep) &
                 (sf_sep > 0) & (sf_sep <= max_sep))
        return med, p16, p84, nvalid, shown

    native_support = {
        "ZG": summarize_support(native_pairs["zg_rv"], 1),
        "CG10": summarize_support(native_pairs["cg_rv"], a.sf_min_orientations)}

    def support_span(shown):
        selected = sf_sep[shown]
        return (f"{selected.min():.2f}–{selected.max():.2f} kpc"
                if selected.size else "no supported bins")

    def plot_structure_functions(results_set, pair_set, support_set, stem,
                                 mask_label):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharex=True,
                                 constrained_layout=True)
        for col, field, label, ylabel in (
                (0, "av", r"$A_V$", r"$S_2(r)$ (mag$^2$)"),
                (1, "rv", r"$R_V$", r"$S_2(r)$")):
            for row, component, ylab, title in (
                    (0, 0, ylabel, label),
                    (1, 1, r"$S_2/(2\sigma^2)$",
                     label + " normalized by variance")):
                ax = axes[row, col]
                for dataset, name, color in (("zg", "Zhang & Green", "#222222"),
                                             ("cg", "CG10", "#397da1")):
                    key = dataset + "_" + field
                    curves = ensemble(results_set, key, component)
                    counts = pair_set[key]
                    qualified = np.where(counts >= a.sf_min_pairs, curves, np.nan)
                    median = curve_percentile(qualified, 50)
                    shown = support_set["ZG" if dataset == "zg" else "CG10"][4]
                    visible = shown & np.isfinite(median) & (median > 0)
                    if dataset == "zg":
                        ax.loglog(sf_sep[visible], median[visible], "o-", color=color,
                                  ms=3.5, lw=2, label=name)
                    else:
                        lo = curve_percentile(qualified, 16)
                        hi = curve_percentile(qualified, 84)
                        ax.loglog(sf_sep[visible], median[visible], "o-", color=color,
                                  ms=3.5, lw=1.8, label=f"{name} median over cutouts")
                        band = shown & np.isfinite(lo) & np.isfinite(hi)
                        ax.fill_between(sf_sep[band], lo[band], hi[band], color=color,
                                        alpha=.18, label="CG10 16–84% over cutouts")
                ax.set_ylabel(ylab)
                ax.set_title(title)
                ax.grid(alpha=.2, which="both")
        axes[0, 0].legend(frameon=False, loc="lower right")
        for ax in axes[1]:
            ax.set_xlabel(r"Separation $r$ (kpc)")
        if "EXPERIMENT" in mask_label:
            fig.suptitle(mask_label)
        support_rule = (f"ZG: ≥{a.sf_min_pairs} pairs; CG: ≥{a.sf_min_pairs} pairs "
                        f"in ≥{a.sf_min_orientations}/8 orientations")
        fig.text(.5, -.015,
                 f"Shown where {support_rule}; "
                 f"supported range: ZG {support_span(support_set['ZG'][4])}, "
                 f"CG {support_span(support_set['CG10'][4])}.",
                 ha="center", va="top", fontsize=8)
        save_figure(fig, stem)

    plot_structure_functions(native_results, native_pairs, native_support,
                             a.outdir / "figure5_structure_functions",
                             "Independent validity-mask comparison" +
                             (" [EXPERIMENT: all dust carbon]" if a.force_carbon else "") +
                             (" [EXPERIMENT: all dust silicate]" if a.force_silicate else "") +
                             (f" [EXPERIMENT: fixed GSD = {a.fixed_gsd}]" if a.fixed_gsd != "none" else "") +
                             (f" [EXPERIMENT: size mixing {a.size_mixing:g} kpc]" if a.size_mixing > 0 else ""))

    # 6. Radial profiles through the face-on map, using the adopted E(B-V) floor.
    map_x = -radius + (np.arange(av.shape[1]) + 0.5) * pixel
    map_y = -radius + (np.arange(av.shape[0]) + 0.5) * pixel
    xx_map, yy_map = np.meshgrid(map_x, map_y)
    radial_distance = np.hypot(xx_map, yy_map)
    n_radial_float = a.plot_radius / a.radial_bin_size
    n_radial = int(round(n_radial_float))
    if not np.isclose(n_radial_float, n_radial):
        raise ValueError("--plot-radius must be an integer multiple of --radial-bin-size")
    radial_edges = np.linspace(0.0, a.plot_radius, n_radial + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    av_profile_map = np.where(ebv_floor_mask & np.isfinite(av) & (av > 0), av, np.nan)
    av_profile_mask = (ebv_floor_mask & (radial_distance < a.plot_radius))

    def radial_percentiles(values, mask):
        med = np.full(n_radial, np.nan)
        p16 = np.full(n_radial, np.nan)
        p84 = np.full(n_radial, np.nan)
        npx = np.zeros(n_radial, dtype=int)
        for ibin, (rlo, rhi) in enumerate(zip(radial_edges[:-1], radial_edges[1:])):
            selected = (mask & (radial_distance >= rlo) &
                        (radial_distance < rhi) & np.isfinite(values))
            vals = values[selected]
            npx[ibin] = len(vals)
            if len(vals):
                med[ibin] = np.median(vals)
                p16[ibin], p84[ibin] = np.percentile(vals, [16, 84])
        return med, p16, p84, npx

    av_med, av_p16, av_p84, av_n = radial_percentiles(
        av_profile_map, av_profile_mask)
    # Both radial profiles use the same E(B-V) floor as the other selected maps.
    rv_radial_selection = (
        (count >= a.min_count) & diag_valid & np.isfinite(rv_map) & (rv_map > 0) &
        np.isfinite(ebv) & (ebv >= a.min_ebv) & (radial_distance < a.plot_radius))
    rv_med, rv_p16, rv_p84, rv_n = radial_percentiles(rv_map, rv_radial_selection)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.0), sharex=True,
                             constrained_layout=True)
    axes[0].plot(radial_centers, av_med, color="#26734d", marker="o", ms=3.5,
                 lw=1.8, label="Median")
    axes[0].fill_between(radial_centers, av_p16, av_p84, color="#26734d",
                         alpha=.18, label="16–84% of sightlines")
    axes[0].set_ylabel(r"$A_V$ (mag)")
    axes[1].plot(radial_centers, rv_med, color="#5e35b1", marker="o", ms=3.5,
                 lw=1.8, label="Median")
    axes[1].fill_between(radial_centers, rv_p16, rv_p84, color="#5e35b1",
                         alpha=.18, label="16–84% of selected pixels")
    axes[1].set_ylabel(r"$R_V$")
    axes[1].set_xlabel("Galactocentric radius (kpc)")
    for ax in axes:
        ax.set_xlim(0, min(10.0, a.plot_radius))  # bins beyond 10 kpc hold isolated pixels
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save_figure(fig, a.outdir / "figure6_radial_av_rv")
    np.savetxt(a.outdir / "radial_av_rv_profiles.csv",
               np.column_stack((radial_edges[:-1], radial_edges[1:], radial_centers,
                                av_n, av_med, av_p16, av_p84,
                                rv_n, rv_med, rv_p16, rv_p84)),
               delimiter=",", fmt="%.7g",
               header="r_inner_pkpc,r_outer_pkpc,r_center_pkpc,N_Av_pixels,"
                      "Av_median_mag,Av_P16_mag,Av_P84_mag,N_Rv_pixels,"
                      "Rv_median,Rv_P16,Rv_P84",
               comments="")


    # Reliability diagnostic: use the adopted E(B-V) floor and compare the
    # CG10 structure function with the ZG reference.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True,
                             constrained_layout=True)
    colors = plt.get_cmap(cmaps["threshold"])(
        np.linspace(0.12, 0.88, len(local_ebv_counts)))
    for color, threshold in zip(colors, local_ebv_counts):
        record = cg_reliability[2.0][threshold]
        pair_support = np.asarray(record["pairs"], dtype=float)
        supported = (np.sum(pair_support >= a.sf_min_pairs, axis=0) >=
                     a.sf_min_orientations)
        for ax, key, ylabel, component in (
                (axes[0], "s2", r"$S_2[R_V](r)$", "raw"),
                (axes[1], "norm", r"$S_2/(2\sigma_{R_V}^2)$", "normalized")):
            curves = np.asarray(record[key], float)
            median = curve_percentile(curves, 50)
            lo = curve_percentile(curves, 16)
            hi = curve_percentile(curves, 84)
            valid = np.isfinite(median) & (median > 0) & supported
            ax.loglog(sf_sep[valid], median[valid], color=color, lw=1.7,
                      label=rf"$E(B-V)\geq{threshold:g}$ mag")
            band = valid & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
            ax.fill_between(sf_sep[band], lo[band], hi[band], color=color, alpha=.08)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2, which="both")
            if component == "raw":
                ax.legend(frameon=False, fontsize=7)
    zg_floor = a.min_ebv
    zg_record = local_rv_sensitivity[zg_floor]
    for ax, value_key, pair_key in (
            (axes[0], "zg_s2", "zg_pairs"),
            (axes[1], "zg_norm", "zg_pairs")):
        zg_values = np.asarray(zg_record[value_key], float)
        zg_pair_counts = np.asarray(zg_record[pair_key], float)
        zg_supported = np.any(zg_pair_counts >= a.sf_min_pairs, axis=0)
        zg_median = curve_percentile(
            np.where(zg_pair_counts >= a.sf_min_pairs, zg_values, np.nan), 50)
        zg_visible = zg_supported & np.isfinite(zg_median) & (zg_median > 0)
        ax.loglog(sf_sep[zg_visible], zg_median[zg_visible], color="black",
                  ls="--", lw=2.2,
                  label=rf"ZG reference, E(B-V)≥{a.min_ebv:g}")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xlabel(r"Separation $r$ (kpc)")
    fig.suptitle(rf"CG10 $R_V$ reliability comparison: $N_{{\rm eff,V}}\geq2$; "
                 rf"250 pc pixels; $\sigma_{{\rm CG}}={sf_args.sim_sigma:g}$ kpc; "
                 rf"shown CG bins meet support in {a.sf_min_orientations}/8 orientations; "
                 rf"$E(B-V)\geq{a.min_ebv:g}$ mag; dashed line is ZG")
    save_figure(fig, a.outdir / "figure5c_cg_rv_reliability")

    # Compare both datasets and their CG10/ZG ratio at the adopted E(B-V) floor,
    # retaining each dataset's native mask and requiring pair support in both.
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.2), sharex="col",
                             constrained_layout=True)
    floor_colors = plt.get_cmap(cmaps["threshold"])(
        np.linspace(0.10, 0.90, len(local_ebv_counts)))
    floor_rows = []
    for color, threshold in zip(floor_colors, local_ebv_counts):
        for dataset, title, s2_key, norm_key, pair_key, column in (
                ("CG10", "CG10 mock-Sun ensemble", "cg_s2", "cg_norm",
                 "cg_pairs", 0),
                ("ZG", "Zhang & Green map", "zg_s2", "zg_norm",
                 "zg_pairs", 1)):
            curves = np.asarray(local_rv_sensitivity[threshold][s2_key], float)
            normalized = np.asarray(local_rv_sensitivity[threshold][norm_key], float)
            pair_arrays = np.asarray(local_rv_sensitivity[threshold][pair_key], float)
            qualified = np.where(pair_arrays >= a.sf_min_pairs, curves, np.nan)
            qualified_norm = np.where(pair_arrays >= a.sf_min_pairs, normalized, np.nan)
            med = curve_percentile(qualified, 50)
            med_norm = curve_percentile(qualified_norm, 50)
            if dataset == "CG10":
                n_supported = np.sum(pair_arrays >= a.sf_min_pairs, axis=0)
                shown = n_supported >= a.sf_min_orientations
                lo = curve_percentile(qualified, 16)
                hi = curve_percentile(qualified, 84)
                lo_norm = curve_percentile(qualified_norm, 16)
                hi_norm = curve_percentile(qualified_norm, 84)
            else:
                shown = np.any(pair_arrays >= a.sf_min_pairs, axis=0)
                lo = hi = lo_norm = hi_norm = None
            for row, values, lower, upper, ylabel in (
                    (0, med, lo, hi, r"$S_2[R_V](r)$"),
                    (1, med_norm, lo_norm, hi_norm, r"$S_2/(2\sigma_{R_V}^2)$")):
                ax = axes[row, column]
                visible = shown & np.isfinite(values) & (values > 0)
                ax.loglog(sf_sep[visible], values[visible], color=color, lw=1.7,
                          label=rf"$E(B-V)\geq{threshold:g}$ mag")
                if lower is not None:
                    band = visible & np.isfinite(lower) & np.isfinite(upper) & (lower > 0)
                    ax.fill_between(sf_sep[band], lower[band], upper[band],
                                    color=color, alpha=.10)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2, which="both")
            if dataset == "CG10":
                pixel_counts = np.asarray(local_ebv_counts[threshold][dataset], int)
                phi_for_csv = phi_deg
            else:
                pixel_counts = np.asarray(local_ebv_counts[threshold][dataset], int)[:1]
                phi_for_csv = np.asarray([-1.0])
            s2_csv = curves[:len(phi_for_csv)]
            norm_csv = normalized[:len(phi_for_csv)]
            pairs_csv = pair_arrays[:len(phi_for_csv)]
            for view_index, view_phi in enumerate(phi_for_csv):
                for sep_index, separation in enumerate(sf_sep):
                    floor_rows.append((threshold, 1 if dataset == "CG10" else 0,
                                       view_phi, separation, pixel_counts[view_index],
                                       pairs_csv[view_index, sep_index],
                                       s2_csv[view_index, sep_index],
                                       norm_csv[view_index, sep_index]))
    ratio_rows = []
    for color, threshold in zip(floor_colors, local_ebv_counts):
        record = local_rv_sensitivity[threshold]
        cg_raw = np.asarray(record["cg_s2"], float)
        zg_raw = np.asarray(record["zg_s2"], float)
        cg_norm = np.asarray(record["cg_norm"], float)
        zg_norm = np.asarray(record["zg_norm"], float)
        cg_pairs = np.asarray(record["cg_pairs"], float)
        zg_pairs = np.asarray(record["zg_pairs"], float)
        paired_support = ((cg_pairs >= a.sf_min_pairs) &
                          (zg_pairs >= a.sf_min_pairs))
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_ratio = cg_raw / zg_raw
            norm_ratio = cg_norm / zg_norm
        raw_ratio[~paired_support] = np.nan
        norm_ratio[~paired_support] = np.nan
        n_orientation = np.sum(np.isfinite(raw_ratio) & (raw_ratio > 0), axis=0)
        shown = n_orientation >= a.sf_min_orientations
        for row, ratios, ylabel in (
                (0, raw_ratio, r"$S_{2,\rm CG10}[R_V]/S_{2,\rm ZG}[R_V]$"),
                (1, norm_ratio, r"$[S_2/(2\sigma_{R_V}^2)]_{\rm CG10}/"
                                r"[S_2/(2\sigma_{R_V}^2)]_{\rm ZG}$")):
            median = curve_percentile(ratios, 50)
            lower = curve_percentile(ratios, 16)
            upper = curve_percentile(ratios, 84)
            visible = shown & np.isfinite(median) & (median > 0)
            ax = axes[row, 2]
            ax.semilogx(sf_sep[visible], median[visible], color=color, lw=1.7,
                        label=rf"$E(B-V)\geq{threshold:g}$ mag")
            ax.set_ylabel(ylabel)
            ax.axhline(1.0, color="0.25", lw=1.0, ls="--", zorder=0)
            ax.grid(alpha=.2, which="both")
            for sep_index, separation in enumerate(sf_sep):
                ratio_rows.append((threshold, row, separation,
                                   n_orientation[sep_index], median[sep_index],
                                   lower[sep_index], upper[sep_index]))
    axes[0, 0].set_title("CG10 mock-Sun ensemble")
    axes[0, 1].set_title("Zhang & Green map")
    axes[0, 2].set_title("CG10 / ZG ratio")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 2].legend(frameon=False, fontsize=7)
    axes[1, 2].set_xlabel(r"Separation $r$ (kpc)")
    for ax in axes[1, :2]:
        ax.set_xlabel(r"Separation $r$ (kpc)")
    fig.suptitle(rf"$R_V$ structure functions with $E(B-V)\geq{a.min_ebv:g}$ mag"
                 f"\nIndependent native masks; floor applied separately; "
                 f"pair cut {a.sf_min_pairs}; ratio uses paired orientations "
                 f"supported in both datasets (≥{a.sf_min_orientations}/8)")
    save_figure(fig, a.outdir / "figure5e_rv_ebv003_comparison")
    np.savetxt(a.outdir / "figure5e_rv_ebv003_comparison.csv",
               np.asarray(floor_rows, float), delimiter=",", fmt="%.8g",
               header="E_BV_floor_mag,dataset_code_0_ZG_1_CG10,orientation_deg_(-1=ZG),"
                      "separation_kpc,N_selected_pixels,N_pairs,S2_RV,S2_RV_normalized",
               comments="")
    np.savetxt(a.outdir / "figure5e_rv_ebv003_ratio.csv",
               np.asarray(ratio_rows, float), delimiter=",", fmt="%.8g",
               header="E_BV_floor_mag,component_0_raw_1_normalized,separation_kpc,"
                      "N_paired_supported_orientations,ratio_median,ratio_P16,ratio_P84",
               comments="")

    # Complement the fixed E(B-V)-floor comparison with a test of
    # the effective V-band particle support. Each curve retains its own
    # supported separation range; unsupported tails are omitted.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True,
                             constrained_layout=True)
    neff_colors = plt.get_cmap(cmaps["support"])(
        np.linspace(0.12, 0.88, len(neff_thresholds)))
    for color, ncut in zip(neff_colors, neff_thresholds):
        record = cg_reliability[ncut][a.min_ebv]
        pair_support = np.asarray(record["pairs"], dtype=float)
        supported = (np.sum(pair_support >= a.sf_min_pairs, axis=0) >=
                     a.sf_min_orientations)
        median_pixels = finite_median(record["pixels"])
        for ax, key, ylabel in (
                (axes[0], "s2", r"$S_2[R_V](r)$"),
                (axes[1], "norm", r"$S_2/(2\sigma_{R_V}^2)$")):
            curves = np.asarray(record[key], float)
            median = curve_percentile(curves, 50)
            lo = curve_percentile(curves, 16)
            hi = curve_percentile(curves, 84)
            valid = np.isfinite(median) & (median > 0) & supported
            ax.loglog(sf_sep[valid], median[valid], color=color, lw=1.7,
                      label=rf"$N_{{\rm eff,V}}\geq{ncut:g}$ (median {median_pixels:.0f} pixels)")
            band = valid & np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
            ax.fill_between(sf_sep[band], lo[band], hi[band], color=color, alpha=.08)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xlabel(r"Separation $r$ (kpc)")
    fig.suptitle(r"CG10 $R_V$ sensitivity to effective dust-particle support "
                 rf"($E(B-V)\geq{a.min_ebv:g}$ mag; 250 pc pixels; "
                 rf"$\sigma_{{\rm CG}}={sf_args.sim_sigma:g}$ kpc)"
                 f"\nShown bins meet pair support in {a.sf_min_orientations}/8 orientations")
    save_figure(fig, a.outdir / "figure5d_cg_rv_neff_sensitivity")

    av_sigma_zg = np.asarray(sigma["zg_av"], float)
    av_sigma_cg = np.asarray(sigma["cg_av"], float)
    rv_sigma_zg = np.asarray(sigma["zg_rv"], float)
    rv_sigma_cg = np.asarray(sigma["cg_rv"], float)
    print(f"ZG selection: ΔA_V P10={zg_cut:.6g} mag per 0.2 kpc; "
          f"median native pixels after local E(B-V)>={a.local_ebv_min:g} cut: "
          f"ZG={finite_median(native_pixels['ZG']):.0f}, "
          f"CG10={finite_median(native_pixels['CG10']):.0f}; "
          f"median finite σ_RV: ZG={finite_median(rv_sigma_zg):.3f}, "
          f"CG10={finite_median(rv_sigma_cg):.3f}")
    print(f"Native structure-function support: ZG {support_span(native_support['ZG'][4])}; "
          f"CG {support_span(native_support['CG10'][4])}; "
          f"pair cuts: ZG ≥{a.sf_min_pairs}; CG ≥{a.sf_min_pairs} in "
          f"≥{a.sf_min_orientations}/8 orientations")

    # Reproducible numeric products for later manuscript revisions.
    np.savez_compressed(a.outdir / "paper_II_figure_data.npz",
                        A_V=av, E_BV=ebv, R_V=rv_map, dust_mass_msun=dustmass_msun,
                        carbon_mass_fraction=carbon_fraction,
                        small_grain_mass_fraction=small_grain_fraction,
                        median_grain_radius_nm=median_grain_radius_nm,
                        dust_count=count, beta_kappa_250_850=beta,
                        beta_wavelengths_um=beta_wave, pixel_pkpc=pixel,
                        h=h, tau_conversion_factor=tau_factor,
                        min_count=a.min_count, min_ebv_mag=a.min_ebv,
                        local_ebv_min_mag=a.local_ebv_min,
                        mock_sun_phi_deg=phi_deg,
                        mock_sun_radius_kpc=sf_args.observer_radius,
                        radial_profile_edges_kpc=radial_edges,
                        radial_profile_centers_kpc=radial_centers,
                        radial_av_n_pixels=av_n, radial_av_median=av_med,
                        radial_av_p16=av_p16, radial_av_p84=av_p84,
                        radial_rv_n_pixels=rv_n, radial_rv_median=rv_med,
                        radial_rv_p16=rv_p16, radial_rv_p84=rv_p84,
                        sf_separation_kpc=sf_sep,
                        sf_min_pairs=a.sf_min_pairs,
                        sf_min_orientations=a.sf_min_orientations,
                        sf_native_pairs_zg=native_pairs["zg_rv"],
                        sf_native_pairs_cg10=native_pairs["cg_rv"],
                        sf_native_pixels_zg=native_pixels["ZG"],
                        sf_native_pixels_cg10=native_pixels["CG10"],
                        sf_local_ebv_thresholds=np.asarray(list(local_ebv_counts), float),
                        sf_local_ebv_pixels_zg=np.asarray(
                            [local_ebv_counts[t]["ZG"] for t in local_ebv_counts]),
                        sf_local_ebv_pixels_cg10=np.asarray(
                            [local_ebv_counts[t]["CG10"] for t in local_ebv_counts]),
                        sf_local_rv_zg_s2=np.asarray([local_rv_sensitivity[t]["zg_s2"] for t in local_ebv_counts]),
                        sf_local_rv_cg_s2=np.asarray([local_rv_sensitivity[t]["cg_s2"] for t in local_ebv_counts]),
                        sf_local_rv_zg_s2_norm=np.asarray([local_rv_sensitivity[t]["zg_norm"] for t in local_ebv_counts]),
                        sf_local_rv_cg_s2_norm=np.asarray([local_rv_sensitivity[t]["cg_norm"] for t in local_ebv_counts]),
                        sf_local_rv_zg_pairs=np.asarray([local_rv_sensitivity[t]["zg_pairs"] for t in local_ebv_counts]),
                        sf_local_rv_cg_pairs=np.asarray([local_rv_sensitivity[t]["cg_pairs"] for t in local_ebv_counts]),
                        zg_cut_av_mag=zg_cut, zg_smoothing_kpc=sf_args.zg_smoothing,
                        cg_smoothing_kpc=sf_args.sim_sigma,
                        zg_min_support=sf_args.zg_min_support,
                        zg_av_s2=ensemble(native_results, "zg_av", 0),
                        cg_av_s2=ensemble(native_results, "cg_av", 0),
                        zg_rv_s2=ensemble(native_results, "zg_rv", 0),
                        cg_rv_s2=ensemble(native_results, "cg_rv", 0),
                        zg_av_s2_norm=ensemble(native_results, "zg_av", 1),
                        cg_av_s2_norm=ensemble(native_results, "cg_av", 1),
                        zg_rv_s2_norm=ensemble(native_results, "zg_rv", 1),
                        cg_rv_s2_norm=ensemble(native_results, "cg_rv", 1),
                        sf_sigma_av_zg=av_sigma_zg, sf_sigma_av_cg=av_sigma_cg,
                        sf_sigma_rv_zg=rv_sigma_zg, sf_sigma_rv_cg=rv_sigma_cg)
    for suffix, support in (("zg_native", native_support["ZG"]),
                            ("cg10_native", native_support["CG10"])):
        med, p16, p84, nvalid, shown = support
        pair_support = np.column_stack((sf_sep, med, p16, p84, nvalid, shown.astype(int)))
        np.savetxt(a.outdir / f"structure_function_pair_support_{suffix}.csv", pair_support,
                   delimiter=",", fmt="%.6g",
                   header="separation_kpc,pairs_median,pairs_P16,pairs_P84,"
                          "orientations_with_min_pairs,displayed", comments="")
    np.savetxt(a.outdir / "rv_histogram_values.txt", rv_vals,
               header=(f"Rv map pixel values; count >= {a.min_count}; E(B-V) >= {a.min_ebv:g} mag; "
                       f"snapshot {a.snapshot:03d}"))
    sensitivity = []
    for threshold in local_ebv_counts:
        selected = ((count >= a.min_count) & diag_valid & np.isfinite(rv_map) &
                    (rv_map > 0) & np.isfinite(ebv) & (ebv >= threshold))
        vals = rv_map[selected]
        sensitivity.append((threshold, len(vals), np.median(vals) if len(vals) else np.nan,
                            np.percentile(vals, 84) if len(vals) else np.nan,
                            np.percentile(vals, 95) if len(vals) else np.nan,
                            np.max(vals) if len(vals) else np.nan))
    np.savetxt(a.outdir / "rv_ebv_cut_sensitivity.csv", np.asarray(sensitivity),
               delimiter=",", fmt=["%.3f", "%d", "%.6g", "%.6g", "%.6g", "%.6g"],
               header="min_EBV_mag,N_pixels,RV_median,RV_P84,RV_P95,RV_max", comments="")
    local_rows = []
    # Select the first plotted bin at or above each requested reference scale.
    # With 250 pc pixels the bin centers are offset by 125 pc (e.g. 0.625 kpc
    # is the first center at or above 0.5 kpc).
    sf_at = [min(int(np.searchsorted(sf_sep, r, side="left")), len(sf_sep) - 1)
             for r in (0.5, 1.0)]
    for threshold in local_ebv_counts:
        zcurves = np.asarray(local_rv_sensitivity[threshold]["zg_s2"], float)
        ccurves = np.asarray(local_rv_sensitivity[threshold]["cg_s2"], float)
        zmed = curve_percentile(zcurves, 50)
        cmed = curve_percentile(ccurves, 50)
        zsig = np.asarray(local_rv_sensitivity[threshold]["zg_sigma"], float)
        csig = np.asarray(local_rv_sensitivity[threshold]["cg_sigma"], float)
        zcounts = np.asarray(local_ebv_counts[threshold]["ZG"], float)
        ccounts = np.asarray(local_ebv_counts[threshold]["CG10"], float)
        local_rows.append((
            threshold, np.median(zcounts), np.percentile(zcounts, 16),
            np.percentile(zcounts, 84), np.median(ccounts),
            np.percentile(ccounts, 16), np.percentile(ccounts, 84),
            finite_median(zsig), finite_median(csig),
            zmed[sf_at[0]], cmed[sf_at[0]], zmed[sf_at[1]], cmed[sf_at[1]]))
    np.savetxt(a.outdir / "mock_sun_ebv_cut_sensitivity.csv", np.asarray(local_rows),
               delimiter=",", fmt="%.6g",
               header=("min_EBV_mag,ZG_pixels_median,ZG_pixels_P16,ZG_pixels_P84,"
                       "CG10_pixels_median,CG10_pixels_P16,CG10_pixels_P84,"
                       "sigma_RV_ZG_median,sigma_RV_CG10_median,"
                       f"S2_ZG_r{sf_sep[sf_at[0]]:.3g}kpc,S2_CG10_r{sf_sep[sf_at[0]]:.3g}kpc,"
                       f"S2_ZG_r{sf_sep[sf_at[1]]:.3g}kpc,S2_CG10_r{sf_sep[sf_at[1]]:.3g}kpc"),
               comments="")
    support_rows = []
    for ncut in neff_thresholds:
        for ebvcut in local_ebv_counts:
            record = cg_reliability[ncut][ebvcut]
            raw = curve_percentile(np.asarray(record["s2"], float), 50)
            norm = curve_percentile(np.asarray(record["norm"], float), 50)
            pair_counts = np.asarray(record["pairs"], float)
            support_rows.append((
                ncut, ebvcut,
                np.median(record["pixels"]),
                np.percentile(record["pixels"], 16),
                np.percentile(record["pixels"], 84),
                finite_median(record["sigma"]),
                raw[sf_at[0]], raw[sf_at[1]],
                norm[sf_at[0]], norm[sf_at[1]],
                np.count_nonzero(pair_counts[:, sf_at[0]] >= a.sf_min_pairs),
                np.count_nonzero(pair_counts[:, sf_at[1]] >= a.sf_min_pairs)))
    np.savetxt(a.outdir / "cg_rv_particle_support_sensitivity.csv",
               np.asarray(support_rows), delimiter=",", fmt="%.6g",
               header=("N_eff_V_min,min_EBV_mag,N_pixels_median,N_pixels_P16,"
                       "N_pixels_P84,sigma_RV_median,"
                       f"S2_raw_r{sf_sep[sf_at[0]]:.3g}kpc,"
                       f"S2_raw_r{sf_sep[sf_at[1]]:.3g}kpc,"
                       f"S2_norm_r{sf_sep[sf_at[0]]:.3g}kpc,"
                       f"S2_norm_r{sf_sep[sf_at[1]]:.3g}kpc,"
                       f"orientations_with_min_pairs_r{sf_sep[sf_at[0]]:.3g}kpc,"
                       f"orientations_with_min_pairs_r{sf_sep[sf_at[1]]:.3g}kpc"), comments="")
    print(f"\nCompleted seven primary figures plus GSD, count-cut, and CG R_V reliability diagnostics "
          f"for {a.run}, "
          f"snapshot {a.snapshot:03d}.")
    print("Additional products: figure6_radial_av_rv, radial_av_rv_profiles.csv, "
          "figure7_cg10_rv_grain_properties, figure7_cg10_rv_grain_properties.csv, "
          "figure2b_rv_count_cut_sensitivity, "
          "figure5c_cg_rv_reliability, figure5d_cg_rv_neff_sensitivity, "
          "figure5e_rv_ebv003_comparison, figure8_grain_size_distributions, and "
          "cg_rv_particle_support_sensitivity.csv")
    print(f"Outputs: {a.outdir.resolve()}")
    print(f"A_V conversion: tau factor={tau_factor:.7g}; h={h:.5f}; pixel={pixel:g} pkpc")
    print(f"R_V pixels={valid_rv.sum():,}; beta pixels={valid_beta.sum():,}; "
          f"E(B-V) cut={a.min_ebv:g} mag")
    print("R_V count-cut sensitivity pixels: " + ", ".join(
          f"N_dust>={ncut}: {number:,}" for ncut, number in map_count_pixels.items()))


if __name__ == "__main__":
    main()
