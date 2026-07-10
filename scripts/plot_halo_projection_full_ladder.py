#!/usr/bin/env python3
"""
plot_halo_projection_full_ladder.py  –  CosmicGrain Physics Ladder Figure
==========================================================================
4-column × 3-row dust projection grid for S0–S10, plus a 12th summary panel.

    Panels 0–10 : Σ_dust maps for S0–S10 (shared colour scale)
    Panel 11    : summary panel, controlled by --bar-quantity:
                    mdust       – horizontal bar chart: M_dust ISM vs CGM
                    dz          – D/Z_ISM bar chart with MW reference line
                    dg          – D/G_ISM bar chart with MW reference line
                    gas_compare – Σ_gas map of the reference rung (purple cmap)
                                  requires --gas-compare-rung (default: last rung)

Usage
-----
# ISM view, D/G bar chart
python plot_halo_projection_full_ladder.py \\
    --snap-pattern  "../{rung}_output_1024/snapdir_{num}/snapshot_{num}.0.hdf5" \\
    --group-pattern "../{rung}_output_1024/groups_{num}/fof_subhalo_tab_{num}.0.hdf5" \\
    --snap-num 049 --rungs S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \\
    --axis z --view ism --depth-frac 0.5 \\
    --bar-quantity dg --vmin-dust 1e-5 --out ladder_ism_dg.png

# Gas-vs-dust comparison (12th panel = S10 gas in purple)
python plot_halo_projection_full_ladder.py \\
    --snap-pattern  "../{rung}_output_1024/snapdir_{num}/snapshot_{num}.0.hdf5" \\
    --group-pattern "../{rung}_output_1024/groups_{num}/fof_subhalo_tab_{num}.0.hdf5" \\
    --snap-num 049 --rungs S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \\
    --axis z --view ism --depth-frac 0.5 \\
    --bar-quantity gas_compare --gas-compare-rung S10 \\
    --vmin-dust 1e-5 --out ladder_ism_gascompare.png

Tips
----
* Use --npix 256 for a fast draft; bump to 512 for the final figure.
* --half-width defaults to Halo 569 R_200c from the updated halo_utils tracker.
* --gas-compare-rung defaults to the last entry in --rungs.
* Pass --no-circle to suppress the R_200c ring on every panel.
* Halo centers/R200 are taken from halo_utils with refine_center=False.
"""

import argparse
import os
import sys
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

from halo_utils import get_halo569_reference, get_halo569

# ── Constants ──────────────────────────────────────────────────────────────────
PROTONMASS = 1.6726e-24
BOLTZMANN  = 1.3806e-16
GAMMA      = 5.0 / 3.0
MU_FIXED   = 0.6
KPC_TO_PC  = 1.0e3
TEAL       = '#2a9d8f'

RUNG_DESCRIPTIONS = {
    'S0':  'Creation only',
    'S1':  '+ Cooling',
    'S2':  '+ Drag',
    'S3':  '+ Astration',
    'S4':  '+ Thermal sputtering',
    'S5':  '+ Grain growth',
    'S6':  '+ Subgrid clumping',
    'S7':  '+ SN shock destruction',
    'S8':  '+ Coagulation',
    'S9':  '+ Shattering',
    'S10': '+ Radiation pressure',
}

# ── Colormaps ──────────────────────────────────────────────────────────────────
def teal_cmap():
    return LinearSegmentedColormap.from_list(
        'dust_teal',
        ['#000000', '#0d4f47', TEAL, '#85d3cb', '#ffffff'],
    )

def purple_cmap():
    return LinearSegmentedColormap.from_list(
        'gas_purple',
        ['#000000', '#2d0a4e', '#7b2d8b', '#c76bcf', '#f0c4f4', '#ffffff'],
    )

# ── Unit helpers ───────────────────────────────────────────────────────────────
def to_phys_kpc(x, a, h):   return x * a / h
def to_msun(m, h):           return m * 1e10 / h

def u_to_K(u, xe=None):
    u_cgs = u * 1e10
    mu = (4.0 / (1.0 + 3*0.76 + 4*0.76*xe) if xe is not None else MU_FIXED)
    return u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN

# ── Multi-file snapshot helpers ────────────────────────────────────────────────
def snap_file_list(snap_arg):
    import glob
    base = snap_arg.rstrip('/')
    for pat in ('.0.hdf5', '.hdf5'):
        if base.endswith(pat):
            base = base[:-len(pat)]
            break
    if os.path.isdir(base) or os.path.isdir(snap_arg):
        d = snap_arg if os.path.isdir(snap_arg) else base
        candidates = sorted(glob.glob(os.path.join(d, '*.0.hdf5')))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(d, '*.hdf5')))
        if not candidates:
            return None
        first = candidates[0]
        base  = first[:-len('.0.hdf5')] if first.endswith('.0.hdf5') else first[:-len('.hdf5')]
    first_try = base + '.0.hdf5'
    single    = base + '.hdf5'
    if os.path.exists(first_try):
        first_file = first_try
    elif os.path.exists(single):
        return [single]
    else:
        return None
    with h5py.File(first_file, 'r') as f:
        n = int(f['Header'].attrs.get('NumFilesPerSnapshot', 1))
    paths = [f'{base}.{i}.hdf5' for i in range(n)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f'  WARNING: missing files: {missing}')
    return paths

def read_header(snap_files):
    with h5py.File(snap_files[0], 'r') as f:
        h = float(f['Parameters'].attrs['HubbleParam'])
        a = float(f['Header'].attrs['Time'])
        z = float(f['Header'].attrs['Redshift'])
        box = float(f['Header'].attrs.get('BoxSize', 0.0))
    return dict(h=h, a=a, z=z, box=box)


# ── Updated Halo 569 / periodic helpers ───────────────────────────────────────
_HALO_REF_CACHE = {}


def _infer_output_dir_from_group_file(group_file):
    p = Path(group_file).resolve()
    if p.parent.name.startswith("groups_"):
        return p.parent.parent
    for parent in p.parents:
        if any(parent.glob("groups_*")):
            return parent
    raise RuntimeError(f"Could not infer output_dir from group_file={group_file}")


def _infer_snap_num_from_group_file(group_file):
    p = Path(group_file)
    for part in p.parts:
        m = re.match(r"groups_(\d+)", part)
        if m:
            return int(m.group(1))
    m = re.search(r"fof_subhalo_tab_(\d+)", p.name)
    if m:
        return int(m.group(1))
    raise RuntimeError(f"Could not infer snapshot number from group_file={group_file}")


def _find_last_snap_num(output_dir):
    output_dir = Path(output_dir)
    last = None
    for groups_dir in sorted(output_dir.glob("groups_*")):
        m = re.search(r"groups_(\d+)", groups_dir.name)
        if not m:
            continue
        snap_num = int(m.group(1))
        snapdir = output_dir / f"snapdir_{snap_num:03d}"
        if snapdir.exists():
            last = snap_num
    return last


def _output_dir_for_rung(group_pattern, rung):
    """
    Derive a rung's output_dir (the parent of its groups_NNN/ and
    snapdir_NNN/ folders) directly from --group-pattern, by substituting
    {rung} and cutting the path off right before the groups_{num} segment.
    Relies on the same "groups_NNN" naming convention already assumed
    throughout this script's halo-tracking code
    (_infer_output_dir_from_group_file, _find_last_snap_num).
    """
    placeholder = "SNAPNUMPLACEHOLDER"
    filled = group_pattern.format(rung=rung, num=placeholder)
    p = Path(filled)
    for i, part in enumerate(p.parts):
        if part.startswith(f"groups_{placeholder}"):
            return Path(*p.parts[:i])
    raise RuntimeError(
        f"Could not derive output_dir for rung={rung!r} from --group-pattern "
        f"(expected a 'groups_{{num}}' path segment): {group_pattern}"
    )


def _find_common_max_snap_num(group_pattern, rungs, verbose=True):
    """
    Find the highest snapshot number available for EVERY requested rung
    (i.e. min of each rung's own latest snapshot), so the ladder figure
    compares all rungs at the same redshift rather than each rung's own
    most-recent (and possibly different) epoch.
    """
    per_rung_last = {}
    for rung in rungs:
        try:
            out_dir = _output_dir_for_rung(group_pattern, rung)
            last = _find_last_snap_num(out_dir)
        except Exception as e:
            last = None
            if verbose:
                print(f'  [{rung}] WARNING: could not scan for snapshots ({e})')
        per_rung_last[rung] = last
        if verbose:
            status = f'{last:03d}' if last is not None else 'NOT FOUND'
            print(f'  [{rung}] latest available snap: {status}')

    valid = {r: v for r, v in per_rung_last.items() if v is not None}
    if not valid:
        raise RuntimeError(
            'Could not find any available snapshot for any requested rung '
            '-- check --snap-pattern / --group-pattern.'
        )

    common_max = min(valid.values())
    bottleneck = [r for r, v in valid.items() if v == common_max]
    missing = [r for r in rungs if r not in valid]
    if verbose:
        print(f'  --> auto-selected snap {common_max:03d} '
              f'(latest common to all rungs; bottleneck: {", ".join(bottleneck)})')
        if missing:
            print(f'  WARNING: no snapshots found at all for: {", ".join(missing)} '
                  f'-- these rungs will be skipped entirely')
    return common_max


def _get_halo569_from_paths(group_file, verbose=False):
    output_dir = _infer_output_dir_from_group_file(group_file)
    snap_num = _infer_snap_num_from_group_file(group_file)
    groups_dir = output_dir / f"groups_{snap_num:03d}"

    key = str(output_dir.resolve())
    if key not in _HALO_REF_CACHE:
        last_snap = _find_last_snap_num(output_dir)
        if last_snap is None:
            raise RuntimeError(f"No valid reference snapshot found in {output_dir}")
        _HALO_REF_CACHE[key] = get_halo569_reference(
            output_dir,
            snap_num_z0=last_snap,
            refine_center=False,
            verbose=verbose,
        )

    ref = _HALO_REF_CACHE[key]
    halo = get_halo569(
        groups_dir,
        snap_num,
        ref,
        refine_center=False,
        verbose=verbose,
    )
    if halo is None:
        raise RuntimeError(f"halo_utils could not locate Halo 569 at snap {snap_num:03d}")

    center = np.asarray(halo["center"], dtype=float)
    r200_pkpc = float(halo["r200_pkpc"])
    r200_ckpch = float(halo["r200_ckpch"])
    used_fallback = bool(
        halo.get("used_catalog_fallback", halo.get("used_fallback", False))
    )
    return center, r200_pkpc, r200_ckpch, halo, used_fallback


def _periodic_delta(pos, center, box):
    dx = np.asarray(pos, dtype=float) - np.asarray(center, dtype=float)[None, :]
    if box is not None and np.isfinite(box) and box > 0:
        dx -= box * np.round(dx / box)
    return dx

def read_halo_center(group_file, halo_id, hdr):
    # Compatibility wrapper: halo_id ignored because this script tracks Halo 569.
    center, r200_pkpc, _, _, _ = _get_halo569_from_paths(group_file, verbose=False)
    return center, r200_pkpc

def _box_mask(pos, center, half_h, depth_h, axis, box=None):
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [i for i in range(3) if i != los]
    dx   = _periodic_delta(pos, center, box)
    return (np.all(np.abs(dx[:, perp]) < half_h, axis=1) &
            (np.abs(dx[:, los]) < depth_h))

def _read_fields(snap_files, ptype, fields):
    """Read fields across all files; optional fields return None if absent."""
    bufs  = {n: [] for n in fields}
    found = False
    for fp in snap_files:
        with h5py.File(fp, 'r') as f:
            if ptype not in f:
                continue
            found = True
            for name, optional in fields.items():
                if name in f[ptype]:
                    bufs[name].append(f[ptype][name][:])
                elif not optional:
                    sys.exit(f'ERROR: {ptype}/{name} missing in {fp}')
    if not found:
        return None
    return {n: (np.concatenate(bufs[n], axis=0) if bufs[n] else None)
            for n in fields}

# ── Projection helpers ─────────────────────────────────────────────────────────
def project_sph(pos2d, mass, hsml, npix, hw, quantity=None,
                smooth_fac=1.0, max_stamp=48):
    """Adaptive-kernel SPH projection.  Returns (mass_map, qty_map)."""
    pix = 2 * hw / npix
    px  = (pos2d[:, 0] + hw) / pix
    py  = (pos2d[:, 1] + hw) / pix
    ph  = np.maximum(0.5, hsml / pix)
    ok  = ((px >= -max_stamp) & (px < npix + max_stamp) &
           (py >= -max_stamp) & (py < npix + max_stamp))
    px, py, ph, mass = px[ok], py[ok], ph[ok], mass[ok]
    qty = quantity[ok] if quantity is not None else None

    mass_map = np.zeros((npix, npix), dtype=np.float64)
    qty_map  = np.zeros((npix, npix), dtype=np.float64) if qty is not None else None

    ph_int = np.clip(ph.astype(int), 1, max_stamp)
    order  = np.argsort(ph_int)
    px, py, ph_int, mass = px[order], py[order], ph_int[order], mass[order]
    if qty is not None:
        qty = qty[order]
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)

    for h_px in np.unique(ph_int):
        sel = ph_int == h_px
        _ix, _iy, _m = ix[sel], iy[sel], mass[sel]
        _q = qty[sel] if qty is not None else None
        r  = min(int(np.ceil(2.5 * h_px)), max_stamp)
        xx = np.arange(-r, r + 1, dtype=np.float64)
        K  = np.exp(-0.5 * (xx[:, None]**2 + xx[None, :]**2) / h_px**2)
        K /= K.sum()
        for i in range(len(_ix)):
            x0, y0 = _ix[i], _iy[i]
            xl = max(0, x0 - r);  xh = min(npix, x0 + r + 1)
            yl = max(0, y0 - r);  yh = min(npix, y0 + r + 1)
            if xl >= xh or yl >= yh:
                continue
            kxl = xl - (x0 - r);  kxh = kxl + (xh - xl)
            kyl = yl - (y0 - r);  kyh = kyl + (yh - yl)
            mass_map[xl:xh, yl:yh] += _m[i] * K[kxl:kxh, kyl:kyh]
            if qty_map is not None:
                qty_map[xl:xh, yl:yh] += _m[i] * _q[i] * K[kxl:kxh, kyl:kyh]

    if smooth_fac > 1.0:
        mass_map = gaussian_filter(mass_map, sigma=smooth_fac)
        if qty_map is not None:
            qty_map = gaussian_filter(qty_map, sigma=smooth_fac)
    return mass_map, qty_map


def project_points(pos2d, mass, npix, hw, quantity=None, smooth_pkpc=None):
    """Histogram projection for point particles (dust)."""
    edges = np.linspace(-hw, hw, npix + 1)
    mmap, _, _ = np.histogram2d(pos2d[:, 0], pos2d[:, 1],
                                bins=[edges, edges], weights=mass)
    qmap = None
    if quantity is not None:
        qmap, _, _ = np.histogram2d(pos2d[:, 0], pos2d[:, 1],
                                    bins=[edges, edges], weights=mass * quantity)
    if smooth_pkpc and smooth_pkpc > 0:
        sig  = smooth_pkpc / (2 * hw / npix)
        mmap = gaussian_filter(mmap, sigma=sig)
        if qmap is not None:
            qmap = gaussian_filter(qmap, sigma=sig)
    return mmap, qmap


def project_dust_adaptive(pos2d, mass, npix, hw,
                          k=16, min_smooth_pkpc=1.0, max_smooth_pkpc=60.0):
    """
    Adaptive-kernel dust projection.
    Smoothing length per particle = distance to k-th nearest neighbour,
    clamped to [min_smooth_pkpc, max_smooth_pkpc].
    Falls back to fixed histogram if too few particles.
    """
    from scipy.spatial import cKDTree
    if len(pos2d) < k + 1:
        pix  = 2 * hw / npix
        mmap, _, _ = np.histogram2d(
            pos2d[:, 0], pos2d[:, 1],
            bins=[np.linspace(-hw, hw, npix + 1)] * 2,
            weights=mass)
        sig = max(min_smooth_pkpc, 3.0) / pix
        return gaussian_filter(mmap, sigma=sig), None
    tree        = cKDTree(pos2d)
    dists, _    = tree.query(pos2d, k=k + 1)
    hsml        = np.clip(dists[:, -1], min_smooth_pkpc, max_smooth_pkpc)
    return project_sph(pos2d, mass, hsml, npix, hw,
                       smooth_fac=1.0,
                       max_stamp=int(max_smooth_pkpc / (2 * hw / npix)) + 2)


# ── Centering helper ───────────────────────────────────────────────────────────
def _find_center_offset(snap_files, center, hw_c, dep_c, axis, hw, hdr):
    """
    Return (dx_corr, dy_corr) in physical kpc using iterative stellar shrinking.
    Falls back to gas centroid if no stars found.
    """
    a, h = hdr['a'], hdr['h']
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [i for i in range(3) if i != los]

    dx_corr = dy_corr = 0.0

    sdata = _read_fields(snap_files, 'PartType4',
                         {'Coordinates': False, 'Masses': False})
    if sdata is not None:
        sp    = sdata['Coordinates']
        smask = _box_mask(sp, center, hw_c * 4, dep_c * 8, axis, hdr.get('box'))
        spos  = to_phys_kpc(_periodic_delta(sp[smask], center, hdr.get('box')), a, h)[:, perp]
        sm    = to_msun(sdata['Masses'][smask], h)
        if len(sm) >= 10:
            cx, cy = 0.0, 0.0
            for shrink_r in [hw * 0.5, hw * 0.3, hw * 0.15, hw * 0.08]:
                r2d_s = np.sqrt((spos[:, 0] - cx)**2 + (spos[:, 1] - cy)**2)
                mask  = r2d_s < shrink_r
                if mask.sum() >= 5:
                    cx = np.average(spos[mask, 0], weights=sm[mask])
                    cy = np.average(spos[mask, 1], weights=sm[mask])
            dx_corr, dy_corr = cx, cy
            print(f'    centering: stars={len(sm):,}  '
                  f'shift=({dx_corr:+.1f}, {dy_corr:+.1f}) pkpc')
            return dx_corr, dy_corr

    # Gas fallback
    gdata = _read_fields(snap_files, 'PartType0',
                         {'Coordinates': False, 'Masses': False})
    if gdata is not None:
        pos   = gdata['Coordinates']
        gmask = _box_mask(pos, center, hw_c, dep_c, axis, hdr.get('box'))
        gpos  = to_phys_kpc(_periodic_delta(pos[gmask], center, hdr.get('box')), a, h)[:, perp]
        if len(gpos) > 10:
            r2d   = np.sqrt(gpos[:, 0]**2 + gpos[:, 1]**2)
            inner = r2d < hw * 0.3
            if inner.sum() > 5:
                dx_corr = np.mean(gpos[inner, 0])
                dy_corr = np.mean(gpos[inner, 1])
            else:
                dx_corr = np.mean(gpos[:, 0])
                dy_corr = np.mean(gpos[:, 1])
            print(f'    centering: gas fallback  '
                  f'shift=({dx_corr:+.1f}, {dy_corr:+.1f}) pkpc')
    return dx_corr, dy_corr


# ── Per-rung projections ───────────────────────────────────────────────────────
def project_rung_dust(snap_files, group_file, halo_id, half_width, depth_frac,
                      axis, npix, dust_smooth, hdr=None,
                      dust_adaptive_k=16, dust_adaptive_min=1.0):
    """
    Dust-only projection.
    Returns dict with dust_sigma, ISM/CGM masses, D/Z, D/G, and metadata.
    gas_sigma is NOT included (use project_rung_gas for that).
    """
    if hdr is None:
        hdr = read_header(snap_files)
    center, r200, r200_ckpch, halo, used_fallback = _get_halo569_from_paths(group_file)
    hw    = half_width if half_width is not None else r200
    depth = hw * depth_frac
    a, h  = hdr['a'], hdr['h']
    hw_c  = hw    / a * h
    dep_c = depth / a * h

    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [i for i in range(3) if i != los]

    dx_corr, dy_corr = _find_center_offset(
        snap_files, center, hw_c, dep_c, axis, hw, hdr)

    # Gas: needed for D/G and D/Z only — no SPH projection here
    gdata = _read_fields(snap_files, 'PartType0',
                         {'Coordinates': False, 'Masses': False,
                          'Metallicity': True})
    pos   = gdata['Coordinates']
    gmask = _box_mask(pos, center, hw_c, dep_c, axis, hdr.get('box'))
    gpos_all = to_phys_kpc(_periodic_delta(pos[gmask], center, hdr.get('box')), a, h)[:, perp]
    gpos_all[:, 0] -= dx_corr
    gpos_all[:, 1] -= dy_corr
    gm = to_msun(gdata['Masses'][gmask], h)

    # Dust projection (padded to avoid edge artefacts)
    PAD       = 2.0
    hw_proj   = hw * PAD
    hw_proj_c = hw_proj / a * h
    npix_proj = int(npix * PAD) + 2
    pix_pc2   = (2 * hw / npix * KPC_TO_PC)**2

    ddata = _read_fields(snap_files, 'PartType6',
                         {'Coordinates': False, 'Masses': False})
    ism_r = 20.0   # ISM aperture (pkpc) — consistent with analysis scripts
    if ddata is not None:
        dp    = ddata['Coordinates']
        dmask = _box_mask(dp, center, hw_proj_c, dep_c, axis, hdr.get('box'))
        dpos  = to_phys_kpc(_periodic_delta(dp[dmask], center, hdr.get('box')), a, h)[:, perp]
        dm    = to_msun(ddata['Masses'][dmask], h)
        dpos[:, 0] -= dx_corr
        dpos[:, 1] -= dy_corr
        if dust_smooth > 0:
            d_full, _ = project_points(dpos, dm, npix_proj, hw_proj,
                                       smooth_pkpc=dust_smooth)
        else:
            d_full, _ = project_dust_adaptive(
                dpos, dm, npix_proj, hw_proj,
                k=dust_adaptive_k,
                min_smooth_pkpc=dust_adaptive_min)
        lo          = int((npix_proj - npix) / 2)
        hi          = lo + npix
        d_mass_map  = d_full[lo:hi, lo:hi]
        pix_proj_pc2 = (2 * hw_proj / npix_proj * KPC_TO_PC)**2
        dust_sigma   = d_mass_map / pix_proj_pc2

        ii, jj      = np.indices((npix, npix))
        r_map       = np.sqrt((ii - npix / 2)**2 + (jj - npix / 2)**2) * (2 * hw / npix)
        ism_px      = r_map < ism_r
        pix_area_pc2 = (2 * hw / npix * KPC_TO_PC)**2
        m_dust_ism   = float(np.nansum(dust_sigma[ism_px])  * pix_area_pc2)
        m_dust_cgm   = float(np.nansum(dust_sigma[~ism_px]) * pix_area_pc2)
        n_dust       = int(dmask.sum())
    else:
        dust_sigma = np.zeros((npix, npix))
        m_dust_ism = m_dust_cgm = 0.0
        n_dust     = 0

    m_dust_tot = m_dust_ism + m_dust_cgm

    # D/G and D/Z in ISM aperture
    r_gas    = np.sqrt(gpos_all[:, 0]**2 + gpos_all[:, 1]**2)
    ism_gas  = r_gas < ism_r
    gm_ism   = gm[ism_gas]
    m_gas_ism = float(np.sum(gm_ism))
    dg_ism = m_dust_ism / m_gas_ism if m_gas_ism > 0 and m_dust_ism > 0 else np.nan
    dz_ism = np.nan
    if gdata.get('Metallicity') is not None and m_dust_ism > 0:
        Z_field = gdata['Metallicity']
        if Z_field.ndim > 1:
            Z_field = Z_field[:, 0]
        gz_ism = Z_field[gmask][ism_gas]
        m_metals_ism = float(np.sum(gm_ism * gz_ism))
        if m_metals_ism > 0:
            dz_ism = m_dust_ism / m_metals_ism

    return dict(dust_sigma=dust_sigma, gas_sigma=None,
                z=hdr['z'], r200_pkpc=r200, hw=hw,
                n_gas=int(gmask.sum()), n_dust=n_dust,
                m_dust_ism=m_dust_ism, m_dust_cgm=m_dust_cgm,
                m_dust_tot=m_dust_tot, dg_ism=dg_ism, dz_ism=dz_ism,
                center_offset=(dx_corr, dy_corr),
                used_catalog_fallback=used_fallback)


def project_rung_gas(snap_files, group_file, halo_id, half_width, depth_frac,
                     axis, npix, smooth_fac=1.0, hdr=None,
                     center_offset=None, post_smooth_pkpc=1.5,
                     adaptive=False, adaptive_k=32,
                     adaptive_min_smooth=0.5, adaptive_max_smooth=20.0):
    """
    Gas surface-density projection for the gas_compare 12th panel.

    Two modes selected by `adaptive`:

    adaptive=False (SPH mode)
        Uses stored SPH smoothing lengths.  Faithful but produces uneven
        surfaces at ISM resolution due to the wide dynamic range in h.
        A post-projection Gaussian (post_smooth_pkpc) reduces blockiness.

    adaptive=True (KDTree mode, recommended)
        Treats gas as point masses; smoothing length = distance to k-th
        nearest neighbour — same algorithm as the dust panels.  Gives a
        visually consistent result that matches the teal dust maps in style.
        adaptive_k            : neighbour count  (default 32)
        adaptive_min_smooth   : min smoothing in pkpc (default 0.5)
        adaptive_max_smooth   : max smoothing in pkpc (default 20.0)

    center_offset    : (dx, dy) pkpc — reuse dust centering for alignment.
    post_smooth_pkpc : extra Gaussian after SPH projection (ignored in
                       adaptive mode).
    """
    if hdr is None:
        hdr = read_header(snap_files)
    center, r200, r200_ckpch, halo, used_fallback = _get_halo569_from_paths(group_file)
    hw    = half_width if half_width is not None else r200
    depth = hw * depth_frac
    a, h  = hdr['a'], hdr['h']
    hw_c  = hw    / a * h
    dep_c = depth / a * h

    axes_map = {'x': 0, 'y': 1, 'z': 2}
    los  = axes_map[axis]
    perp = [i for i in range(3) if i != los]

    if center_offset is not None:
        dx_corr, dy_corr = center_offset
        print(f'    centering: reusing dust offset '
              f'({dx_corr:+.1f}, {dy_corr:+.1f}) pkpc')
    else:
        dx_corr, dy_corr = _find_center_offset(
            snap_files, center, hw_c, dep_c, axis, hw, hdr)

    fields = {'Coordinates': False, 'Masses': False}
    if not adaptive:
        fields['SmoothingLength']   = False
        fields['InternalEnergy']    = False
        fields['ElectronAbundance'] = True

    gdata = _read_fields(snap_files, 'PartType0', fields)
    pos   = gdata['Coordinates']
    # Expand the box mask to account for the centering offset so that
    # particles which will be shifted into the field of view are not
    # pre-clipped.  Without this, a large stellar centroid offset causes
    # a hard black edge on one side of the gas panel.
    offset_frac = (abs(dx_corr) + abs(dy_corr)) / max(hw, 1.0)
    mask_pad    = 1.15 + offset_frac   # 15% baseline + offset correction
    gmask = _box_mask(pos, center, hw_c * mask_pad, dep_c, axis, hdr.get('box'))
    gpos  = to_phys_kpc(_periodic_delta(pos[gmask], center, hdr.get('box')), a, h)[:, perp]
    gpos[:, 0] -= dx_corr
    gpos[:, 1] -= dy_corr
    gm    = to_msun(gdata['Masses'][gmask], h)

    if adaptive:
        # KDTree adaptive projection — same algorithm as dust panels.
        # Padded box prevents boundary particles getting artificially large
        # kernels; crop back to display half-width afterwards.
        PAD       = 1.5
        hw_proj   = hw * PAD
        npix_proj = int(npix * PAD) + 2
        print(f'    gas adaptive KDTree: k={adaptive_k}, '
              f'h=[{adaptive_min_smooth}, {adaptive_max_smooth}] pkpc, '
              f'n_gas={gmask.sum():,}')
        mass_full, _ = project_dust_adaptive(
            gpos, gm, npix_proj, hw_proj,
            k=adaptive_k,
            min_smooth_pkpc=adaptive_min_smooth,
            max_smooth_pkpc=adaptive_max_smooth)
        lo           = int((npix_proj - npix) / 2)
        hi           = lo + npix
        mass_map     = mass_full[lo:hi, lo:hi]
        pix_proj_pc2 = (2 * hw_proj / npix_proj * KPC_TO_PC)**2
        gas_sigma    = mass_map / pix_proj_pc2
    else:
        # SPH projection using stored smoothing lengths
        gh = to_phys_kpc(gdata['SmoothingLength'][gmask], a, h)
        mass_map, _ = project_sph(gpos, gm, gh, npix, hw, smooth_fac=smooth_fac)
        if post_smooth_pkpc > 0:
            sigma_pix = post_smooth_pkpc / (2 * hw / npix)
            mass_map  = gaussian_filter(mass_map, sigma=sigma_pix)
        pix_pc2   = (2 * hw / npix * KPC_TO_PC)**2
        gas_sigma = mass_map / pix_pc2

    return dict(gas_sigma=gas_sigma, z=hdr['z'], r200_pkpc=r200, hw=hw,
                n_gas=int(gmask.sum()),
                used_catalog_fallback=used_fallback,
                center_offset=(dx_corr, dy_corr))


# ── Figure helpers ─────────────────────────────────────────────────────────────
def _safe_log_bounds(arr, plo=0.5, phi=99.9):
    pos = arr[arr > 0]
    if len(pos) == 0:
        return 1e-10, 1.0
    return np.nanpercentile(pos, plo), np.nanpercentile(pos, phi)


def _nice_scalebar(hw):
    for s in [500, 200, 100, 50, 20, 10, 5]:
        if s < hw * 0.9:
            return float(s)
    return max(1.0, round(hw * 0.2, -1))


def _perp_labels(axis):
    """Return (xlabel, ylabel) for the two projected axes."""
    return {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[axis]


# ── Main figure ────────────────────────────────────────────────────────────────
def make_ladder_figure(rung_maps, rungs, descriptions, axis,
                       show_circle, out_path,
                       vmin_dust=None, vmax_dust=None,
                       bar_quantity='mdust',
                       gas_map=None, gas_rung_label=None,
                       gas_vmin=None, gas_vmax=None):
    """
    4-column × 3-row dust ladder figure.

    Parameters
    ----------
    rung_maps       : list of dicts from project_rung_dust()
    rungs           : list of rung label strings
    descriptions    : dict rung → description string
    axis            : projection axis ('x', 'y', 'z')
    show_circle     : bool — draw R_200c dashed ring
    out_path        : output filename
    vmin_dust/vmax_dust : manual dust colour limits (optional)
    bar_quantity    : 'mdust' | 'dz' | 'dg' | 'gas_compare'
    gas_map         : dict from project_rung_gas() — required when
                      bar_quantity == 'gas_compare'
    gas_rung_label  : string label for the gas panel (e.g. 'S10')
    """
    from matplotlib.cm import ScalarMappable
    n_rungs = len(rungs)
    NC, NR  = 4, 3

    # ── Shared dust colour bounds ──────────────────────────────────────────────
    all_dust = np.concatenate([np.ravel(d['dust_sigma']) for d in rung_maps])
    vd_lo, vd_hi = _safe_log_bounds(all_dust)
    if vmin_dust is not None: vd_lo = vmin_dust
    if vmax_dust is not None: vd_hi = vmax_dust
    norm_dust = LogNorm(vmin=vd_lo, vmax=vd_hi)
    cmap_dust = teal_cmap()

    hw_ref = rung_maps[0]['hw']
    ax_label = _perp_labels(axis)

    # ── Layout ────────────────────────────────────────────────────────────────
    panel_in = 2.3
    cbar_in  = 0.22
    fig_w = NC * panel_in + cbar_in + 0.55
    fig_h = NR * panel_in + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        NR, NC + 1,
        figure=fig,
        width_ratios=[1] * NC + [cbar_in / panel_in],
        left=0.06, right=0.97,
        top=0.92,  bottom=0.06,
        hspace=0.04, wspace=0.04,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(NC)] for r in range(NR)]
    cax  = fig.add_subplot(gs[:, NC])

    # Rung colour ramp for bar-chart lines
    profile_cmap = LinearSegmentedColormap.from_list(
        'rung_seq',
        ['#1d6b5e', '#2a9d8f', '#85d3cb', '#ddf5f0', '#ffffff'], N=256)
    n_total = 11
    def rung_color(idx):
        return profile_cmap(idx / (n_total - 1))

    # ── Dust panels (positions 0–10) ───────────────────────────────────────────
    for idx, (rung, data) in enumerate(zip(rungs, rung_maps)):
        row, col = divmod(idx, NC)
        ax  = axes[row][col]
        hw  = data['hw']
        img = data['dust_sigma']

        ax.set_facecolor('black')
        if np.any(img > 0):
            cm_disp = teal_cmap()
            cm_disp.set_bad('black')
            cm_disp.set_under('black')
            disp = np.where((img > 0) & np.isfinite(img), img, np.nan)
            ax.imshow(disp.T, origin='lower',
                      extent=[-hw, hw, -hw, hw],
                      cmap=cm_disp, norm=norm_dust,
                      interpolation='lanczos', aspect='equal')
        else:
            ax.text(0.5, 0.5, 'no dust', transform=ax.transAxes,
                    ha='center', va='center', color=TEAL,
                    fontsize=8, style='italic')

        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hw, hw)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-10, 0, 10])

        if show_circle:
            th = np.linspace(0, 2 * np.pi, 360)
            ax.plot(data['r200_pkpc'] * np.cos(th),
                    data['r200_pkpc'] * np.sin(th),
                    '--', color='white', lw=0.5, alpha=0.45)

        bottom_row = (row == NR - 1) or (idx == n_rungs - 1)
        left_col   = (col == 0)
        ax.tick_params(colors='white', labelcolor='black', labelsize=8,
                       direction='in', top=True, right=True,
                       labelbottom=bottom_row, labelleft=left_col)
        for sp in ax.spines.values():
            sp.set_edgecolor('#aaaaaa')
            sp.set_linewidth(0.5)
        if bottom_row:
            ax.set_xlabel(f'{ax_label[0]} (kpc)', color='black', fontsize=9)
        if left_col:
            ax.set_ylabel(f'{ax_label[1]} (kpc)', color='black', fontsize=9,
                          labelpad=2)

        desc  = descriptions.get(rung, '')
        label = f'{rung}  {desc}' if desc else rung
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha='left', va='top', color='white',
                fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black',
                          alpha=0.55, lw=0))

        if idx == n_rungs - 1:
            for sp in ax.spines.values():
                sp.set_edgecolor(TEAL)
                sp.set_linewidth(1.2)

    # ── 12th panel ────────────────────────────────────────────────────────────
    ax_bar = axes[NR - 1][NC - 1]
    ax_bar.set_facecolor('black')
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('white')
        sp.set_linewidth(0.4)

    all_rung_names = ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
    ypos  = np.arange(n_rungs)
    m_ism = np.array([d['m_dust_ism'] for d in rung_maps])
    m_cgm = np.array([d['m_dust_cgm'] for d in rung_maps])
    ism_r = 20.0
    CGM_COL = '#777777'

    import matplotlib.ticker as _ticker
    from matplotlib.patches import Patch as _Patch

    def _ratio_bars(ax, vals, xlabel, ref_val, ref_lbl):
        for i, (rung, v) in enumerate(zip(rungs, vals)):
            try:    s_idx = all_rung_names.index(rung)
            except  ValueError: s_idx = i
            col = rung_color(s_idx)
            if np.isfinite(v) and v > 0:
                ax.barh(ypos[i], v, height=0.55, color=col, alpha=0.95, zorder=3)
        ax.axvline(ref_val, color='white', lw=1.0, ls='--', alpha=0.6)
        ax.text(ref_val * 1.03, -0.7, ref_lbl, color='white',
                fontsize=6.5, va='top')
        valid = vals[np.isfinite(vals)]
        x_hi  = (max(float(np.nanmax(valid)) * 1.2, ref_val * 1.5)
                 if len(valid) else ref_val * 2)
        ax.set_xlim(0, x_hi)
        ax.set_xscale('linear')
        ax.xaxis.set_major_formatter(_ticker.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel, color='black', fontsize=9)

    def _bar_yticks():
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels([])
        ax_bar.invert_yaxis()
        ax_bar.tick_params(colors='white', labelcolor='black', labelsize=8,
                           direction='in', top=True, right=True, which='both')
        ax_bar.grid(True, axis='x', which='major',
                    color='white', alpha=0.08, lw=0.4)
        for i, rung in enumerate(rungs):
            desc = descriptions.get(rung, '')
            lbl  = f'{rung}  {desc}' if desc else rung
            ax_bar.text(0.02, ypos[i], lbl,
                        transform=ax_bar.get_yaxis_transform(),
                        va='center', ha='left', color='white',
                        fontsize=6.0, zorder=10,
                        bbox=dict(boxstyle='round,pad=0.1', fc='black',
                                  alpha=0.5, lw=0))

    # ── gas_compare ────────────────────────────────────────────────────────────
    if bar_quantity == 'gas_compare':
        if gas_map is not None and gas_map.get('gas_sigma') is not None:
            gimg = gas_map['gas_sigma']
            gd_lo, gd_hi = _safe_log_bounds(gimg)
            if gas_vmin is not None: gd_lo = gas_vmin
            if gas_vmax is not None: gd_hi = gas_vmax
            norm_gas = LogNorm(vmin=gd_lo, vmax=gd_hi)
            cm_gas   = purple_cmap()
            cm_gas.set_bad('black')
            cm_gas.set_under('black')
            hw_g = gas_map['hw']
            disp = np.where((gimg > 0) & np.isfinite(gimg), gimg, np.nan)
            ax_bar.imshow(disp.T, origin='lower',
                          extent=[-hw_g, hw_g, -hw_g, hw_g],
                          cmap=cm_gas, norm=norm_gas,
                          interpolation='lanczos', aspect='equal')
            ax_bar.set_xlim(-hw_g, hw_g)
            ax_bar.set_ylim(-hw_g, hw_g)
            ax_bar.set_xticks([-10, 0, 10])
            ax_bar.set_yticks([-10, 0, 10])

            if show_circle:
                th = np.linspace(0, 2 * np.pi, 360)
                ax_bar.plot(gas_map['r200_pkpc'] * np.cos(th),
                            gas_map['r200_pkpc'] * np.sin(th),
                            '--', color='white', lw=0.5, alpha=0.45)

            lbl = (f'{gas_rung_label}  gas surface density'
                   if gas_rung_label else 'gas surface density')
            ax_bar.text(0.03, 0.97, lbl, transform=ax_bar.transAxes,
                        ha='left', va='top', color='white',
                        fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black',
                                  alpha=0.55, lw=0))

            xl, yl = ax_label
            ax_bar.set_xlabel(f'{xl} (kpc)', color='black', fontsize=9)
            ax_bar.tick_params(colors='white', labelcolor='black', labelsize=8,
                               direction='in', top=True, right=True,
                               labelbottom=True, labelleft=False)
            for sp in ax_bar.spines.values():
                sp.set_edgecolor('#aaaaaa')
                sp.set_linewidth(0.5)

            # Inset colorbar — slim vertical strip inside the lower-right
            # corner of the gas panel.  Placed well clear of the rung label
            # (top-left) and the R_200c circle.
            # [x0, y0, width, height] in axes-fraction coordinates.
            cax_gas = ax_bar.inset_axes([0.72, 0.04, 0.055, 0.38])
            sm_gas  = ScalarMappable(cmap=cm_gas, norm=norm_gas)
            sm_gas.set_array([])
            cb_gas  = fig.colorbar(sm_gas, cax=cax_gas)
            cb_gas.ax.set_title(
                r'$\Sigma_\mathrm{gas}$' + '\n' + r'(M$_\odot$ pc$^{-2}$)',
                color='white', fontsize=5.5, pad=3, loc='left')
            cb_gas.ax.yaxis.set_tick_params(color='white', labelsize=5.5,
                                            length=2, width=0.5)
            plt.setp(cb_gas.ax.yaxis.get_ticklabels(), color='white')
            cb_gas.outline.set_edgecolor('white')
            cb_gas.outline.set_linewidth(0.4)
        else:
            ax_bar.text(0.5, 0.5,
                        'gas_sigma not available\npass --bar-quantity gas_compare\n'
                        'with --gas-compare-rung',
                        transform=ax_bar.transAxes,
                        ha='center', va='center', color='white',
                        fontsize=7.5, style='italic')
            ax_bar.tick_params(colors='white', labelcolor='black', labelsize=8,
                               direction='in', top=True, right=True)

    # ── dz ─────────────────────────────────────────────────────────────────────
    elif bar_quantity == 'dz':
        dz_vals = np.array([d.get('dz_ism', np.nan) for d in rung_maps])
        _ratio_bars(ax_bar, dz_vals,
                    r'$\mathcal{D}/Z_\mathrm{ISM}$  ($r < 20$ kpc)',
                    ref_val=0.4, ref_lbl='MW')
        _bar_yticks()

    # ── dg ─────────────────────────────────────────────────────────────────────
    elif bar_quantity == 'dg':
        dg_vals = np.array([d.get('dg_ism', np.nan) for d in rung_maps])
        _ratio_bars(ax_bar, dg_vals,
                    r'$\mathcal{D}/G_\mathrm{ISM}$  ($r < 20$ kpc)',
                    ref_val=0.01, ref_lbl='MW')
        _bar_yticks()

    # ── mdust (default) ────────────────────────────────────────────────────────
    else:
        for i, (rung, mi, mc) in enumerate(zip(rungs, m_ism, m_cgm)):
            try:    s_idx = all_rung_names.index(rung)
            except  ValueError: s_idx = i
            col = rung_color(s_idx)
            if mc > 0:
                ax_bar.barh(ypos[i], mc, height=0.45,
                            color=CGM_COL, alpha=0.75, zorder=2, left=mi)
            ax_bar.barh(ypos[i], mi, height=0.55,
                        color=col, alpha=0.95, zorder=3)
        ax_bar.set_xscale('log')
        ax_bar.set_xlabel(r'$M_\mathrm{dust}$  (M$_\odot$)',
                          color='black', fontsize=9)
        _lc = rung_color(10)
        ax_bar.legend(
            [_Patch(facecolor=_lc, alpha=0.95),
             _Patch(facecolor=CGM_COL, alpha=0.75)],
            [rf'$r < {ism_r:.0f}$ kpc (ISM)',
             rf'$r > {ism_r:.0f}$ kpc (CGM)'],
            fontsize=6.5, framealpha=0.0, labelcolor='white',
            loc='upper right', handlelength=1.2,
            borderpad=0.4, labelspacing=0.4)
        _bar_yticks()

    # ── Shared dust colorbar ───────────────────────────────────────────────────
    sm_dust = ScalarMappable(cmap=cmap_dust, norm=norm_dust)
    sm_dust.set_array([])
    cb = fig.colorbar(sm_dust, cax=cax)
    cb.ax.set_ylabel(r'$\Sigma_\mathrm{dust}$  (M$_\odot$ pc$^{-2}$)',
                     rotation=-90, va='bottom', color='black',
                     fontsize=12, labelpad=12)
    cb.ax.yaxis.set_tick_params(color='black', labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(0.5)

    # ── Scale bar (first panel) ────────────────────────────────────────────────
    ax0 = axes[0][0]
    sb  = _nice_scalebar(hw_ref)
    x0  = -hw_ref + 0.07 * 2 * hw_ref
    y0  = -hw_ref + 0.06 * 2 * hw_ref
    ax0.plot([x0, x0 + sb], [y0, y0], '-', color='white', lw=1.2)
    ax0.text(x0 + sb / 2, y0 + 0.04 * 2 * hw_ref, f'{sb:.0f} kpc',
             ha='center', va='bottom', color='white', fontsize=8)

    # ── Redshift label (bottom-right of panel 0, near scale bar) ───────────────
    z_ref = rung_maps[-1]['z']
    ax0.text(0.97, 0.05, f'$z={z_ref:.2f}$',
             transform=ax0.transAxes,
             ha='right', va='bottom', color='white', fontsize=8)

    # ── Save ───────────────────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.94)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.08)
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snap-pattern', required=True,
        help='Path template with {rung} and {num}, e.g. '
             '"../{rung}_output_1024/snapdir_{num}/snapshot_{num}.0.hdf5"')
    p.add_argument('--group-pattern', required=True,
        help='Same for group catalog, e.g. '
             '"../{rung}_output_1024/groups_{num}/fof_subhalo_tab_{num}.0.hdf5"')
    p.add_argument('--snap-num', required=True,
        help='Snapshot number string as in filenames, e.g. "049". '
             'Pass "auto" to automatically use the latest snapshot number '
             'available for ALL requested --rungs (the minimum of each '
             "rung's own latest snapshot), so every panel stays at the "
             'same redshift even if some rungs are further along than '
             'others.')
    p.add_argument('--rungs', nargs='+',
        default=['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
        help='Rung labels in order (default S0 … S10)')
    p.add_argument('--halo-id', type=int, default=0)
    p.add_argument('--view', choices=['ism', 'halo'], default=None,
        help='ism = 20 kpc half-width; halo = R_200c (default)')
    p.add_argument('--half-width', type=float, default=None,
        help='Half-width in pkpc; overridden by --view if both given')
    p.add_argument('--depth-frac', type=float, default=0.5)
    p.add_argument('--axis', choices=['x', 'y', 'z'], default='z')
    p.add_argument('--npix', type=int, default=512,
        help='Pixels per panel (default 512)')
    p.add_argument('--dust-smooth', type=float, default=0.0,
        help='Gaussian smoothing for dust panels in pkpc '
             '(default 0 = adaptive SPH-style)')
    p.add_argument('--dust-adaptive-k', type=int, default=16,
        help='Neighbour count for adaptive dust projection (default 16); '
             'lower values give sharper structure — match to '
             '--gas-adaptive-k for a direct visual comparison')
    p.add_argument('--dust-adaptive-min', type=float, default=1.0,
        help='Min smoothing length in pkpc for adaptive dust (default 1.0)')
    p.add_argument('--gas-smooth-fac', type=float, default=1.0,
        help='SPH smooth_fac for gas projection (SPH mode only, default 1.0)')
    p.add_argument('--gas-post-smooth', type=float, default=1.5,
        help='Post-projection Gaussian sigma in pkpc (SPH mode only, '
             'default 1.5; set 0 to disable)')
    p.add_argument('--gas-adaptive', action='store_true',
        help='Use KDTree adaptive projection for gas (recommended; matches '
             'the style of the dust panels)')
    p.add_argument('--gas-adaptive-k', type=int, default=32,
        help='Neighbour count for adaptive gas projection (default 32)')
    p.add_argument('--gas-adaptive-min', type=float, default=0.5,
        help='Min smoothing length in pkpc for adaptive gas (default 0.5)')
    p.add_argument('--gas-adaptive-max', type=float, default=20.0,
        help='Max smoothing length in pkpc for adaptive gas (default 20.0)')
    p.add_argument('--gas-fixed-smooth', type=float, default=None,
        help='Fix gas adaptive smoothing to this physical scale in pkpc '
             '(sets both min and max to the same value, giving a uniform '
             'kernel size across all gas particles — useful for a fair '
             'visual comparison with the dust panels). '
             'Overrides --gas-adaptive-min and --gas-adaptive-max when set.')
    p.add_argument('--gas-vmin', type=float, default=None,
        help='Manual lower colour limit for gas panel (M_sun/pc^2)')
    p.add_argument('--gas-vmax', type=float, default=None,
        help='Manual upper colour limit for gas panel (M_sun/pc^2)')
    p.add_argument('--no-circle', action='store_true',
        help='Suppress R_200c circle on every panel')
    p.add_argument('--no-descriptions', action='store_true',
        help='Show rung labels only, no physics descriptions')
    p.add_argument('--bar-quantity',
        choices=['mdust', 'dz', 'dg', 'gas_compare'], default='mdust',
        help=(
            'Panel 12 content: '
            'mdust = M_dust ISM+CGM bars; '
            'dz = D/Z_ISM bars; '
            'dg = D/G_ISM bars; '
            'gas_compare = Σ_gas map of --gas-compare-rung (purple colourmap, '
            'for direct dust/gas visual comparison)'
        ))
    p.add_argument('--gas-compare-rung', default=None,
        help='Rung label to project for the gas_compare panel '
             '(default: last entry in --rungs). '
             'Must be a valid entry in --rungs.')
    p.add_argument('--vmin-dust', type=float, default=None)
    p.add_argument('--vmax-dust', type=float, default=None)
    p.add_argument('--out', default='ladder.pdf')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve "auto" once, up front, so every downstream args.snap_num usage
    # (snap_pattern/group_pattern .format() calls) automatically picks up the
    # resolved value without needing to touch each call site individually.
    if str(args.snap_num).strip().lower() == 'auto':
        print('--snap-num auto: scanning each rung for its latest available snapshot...')
        common_max = _find_common_max_snap_num(args.group_pattern, args.rungs, verbose=True)
        args.snap_num = f'{common_max:03d}'
        print(f'Using snap-num = {args.snap_num} for all rungs\n')

    descriptions = {} if args.no_descriptions else RUNG_DESCRIPTIONS

    # ── Determine shared half-width from last rung ─────────────────────────────
    ref_rung   = args.rungs[-1]
    ref_snap   = args.snap_pattern .format(rung=ref_rung, num=args.snap_num)
    ref_groups = args.group_pattern.format(rung=ref_rung, num=args.snap_num)
    ref_files  = snap_file_list(ref_snap)
    if ref_files is None:
        sys.exit(f'ERROR: cannot find snapshot for reference rung '
                 f'{ref_rung}: {ref_snap}')
    ref_hdr = read_header(ref_files)
    _, r200_ref, _, ref_halo, ref_used_fallback = _get_halo569_from_paths(ref_groups)

    if args.view == 'ism':
        half_width = 20.0
    elif args.view == 'halo':
        half_width = r200_ref
    elif args.half_width is not None:
        half_width = args.half_width
    else:
        half_width = r200_ref

    fb_note = " [catalog fallback]" if ref_used_fallback else ""
    print(f'Reference rung : {ref_rung},  R_200c = {r200_ref:.1f} pkpc{fb_note}')
    print(f'Projection     : half-width={half_width:.1f} pkpc, '
          f'depth={half_width * args.depth_frac:.1f} pkpc, '
          f'axis={args.axis}')

    # ── Project dust for all rungs ─────────────────────────────────────────────
    rung_maps   = []
    actual_rungs = []
    for rung in args.rungs:
        snap_path  = args.snap_pattern .format(rung=rung, num=args.snap_num)
        group_path = args.group_pattern.format(rung=rung, num=args.snap_num)
        snap_files = snap_file_list(snap_path)
        if snap_files is None:
            print(f'  [{rung}] WARNING: snapshot not found — skipping')
            continue
        n_f = len(snap_files)
        print(f'\n[{rung}]  {os.path.basename(snap_files[0])}'
              f'{f"  (+{n_f-1} more)" if n_f > 1 else ""}')
        hdr  = read_header(snap_files)
        data = project_rung_dust(
            snap_files, group_path, args.halo_id,
            half_width, args.depth_frac, args.axis,
            args.npix, args.dust_smooth, hdr=hdr,
            dust_adaptive_k=args.dust_adaptive_k,
            dust_adaptive_min=args.dust_adaptive_min)
        fb = " [catalog fallback]" if data.get("used_catalog_fallback", False) else ""
        print(f'  z={data["z"]:.3f}  R200={data["r200_pkpc"]:.1f} pkpc  '
              f'gas={data["n_gas"]:,}  dust={data["n_dust"]:,}{fb}')
        rung_maps.append(data)
        actual_rungs.append(rung)

    if not rung_maps:
        sys.exit('ERROR: no rungs projected successfully.')

    # ── Gas projection for gas_compare panel ──────────────────────────────────
    gas_map        = None
    gas_rung_label = None
    if args.bar_quantity == 'gas_compare':
        gc_rung = args.gas_compare_rung or actual_rungs[-1]
        if gc_rung not in actual_rungs:
            print(f'WARNING: --gas-compare-rung {gc_rung} not in projected '
                  f'rungs; falling back to {actual_rungs[-1]}')
            gc_rung = actual_rungs[-1]
        gc_snap   = args.snap_pattern .format(rung=gc_rung, num=args.snap_num)
        gc_groups = args.group_pattern.format(rung=gc_rung, num=args.snap_num)
        gc_files  = snap_file_list(gc_snap)
        if gc_files is None:
            print(f'WARNING: cannot find snapshot for gas_compare rung '
                  f'{gc_rung} — gas panel will be blank')
        else:
            # Re-use the center offset already computed for this rung's dust
            # projection so the gas map is pixel-aligned with the dust panels.
            gc_idx          = actual_rungs.index(gc_rung)
            stored_offset   = rung_maps[gc_idx].get('center_offset', None)

            n_f = len(gc_files)
            print(f'\n[{gc_rung} GAS]  {os.path.basename(gc_files[0])}'
                  f'{f"  (+{n_f-1} more)" if n_f > 1 else ""}')
            gc_hdr  = read_header(gc_files)
            # --gas-fixed-smooth overrides min/max to a single scale
            gas_amin = args.gas_adaptive_min
            gas_amax = args.gas_adaptive_max
            if args.gas_fixed_smooth is not None:
                gas_amin = args.gas_fixed_smooth
                gas_amax = args.gas_fixed_smooth
                print(f'    gas fixed smooth: {args.gas_fixed_smooth} pkpc '
                      f'(overrides adaptive min/max)')
            gas_map = project_rung_gas(
                gc_files, gc_groups, args.halo_id,
                half_width, args.depth_frac, args.axis,
                args.npix, smooth_fac=args.gas_smooth_fac, hdr=gc_hdr,
                center_offset=stored_offset,
                post_smooth_pkpc=args.gas_post_smooth,
                adaptive=args.gas_adaptive,
                adaptive_k=args.gas_adaptive_k,
                adaptive_min_smooth=gas_amin,
                adaptive_max_smooth=gas_amax)
            gas_rung_label = gc_rung
            fb = " [catalog fallback]" if gas_map.get("used_catalog_fallback", False) else ""
            print(f'  z={gas_map["z"]:.3f}  R200={gas_map["r200_pkpc"]:.1f} pkpc  '
                  f'gas={gas_map["n_gas"]:,}{fb}')

    # ── Render ─────────────────────────────────────────────────────────────────
    print(f'\nRendering 4×3 dust ladder → {args.out}')
    make_ladder_figure(
        rung_maps, actual_rungs, descriptions,
        axis=args.axis,
        show_circle=not args.no_circle,
        out_path=args.out,
        vmin_dust=args.vmin_dust,
        vmax_dust=args.vmax_dust,
        bar_quantity=args.bar_quantity,
        gas_map=gas_map,
        gas_rung_label=gas_rung_label,
        gas_vmin=args.gas_vmin,
        gas_vmax=args.gas_vmax,
    )


if __name__ == '__main__':
    main()
