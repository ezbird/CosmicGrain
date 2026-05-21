#!/usr/bin/env python3
"""
plot_physics_ladder.py  –  CosmicGrain Physics Ladder Figure
=============================================================
11-row × 3-column grid comparing S0–S10 physics rungs at the same snapshot.

    Columns : Σ_gas  |  ⟨T⟩_mw  |  Σ_dust
    Rows    : S0 → S10

Color scales are SHARED across all rows within each column so rung-to-rung
differences are directly visible.  Dust panels for rungs with no PartType6
are rendered as blank (black).

Usage
-----

LATEST:
python plot_halo_projection_full_ladder.py     --snap-pattern  "../{rung}_output_1024/snapdir_{num}/snapshot_{num}.0.hdf5"     --group-pattern "../{rung}_output_1024/groups_{num}/fof_subhalo_tab_{num}.0.hdf5"     --snap-num 049 --rungs S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10     --axis z --view ism --depth-frac 0.5     --bar-quantity dg     --vmin-dust 1e-5     --out ladder_ism_clipped.png

python plot_halo_projection_full_ladder.py \
    --snap-pattern  "../{rung}_output_1024/snapdir_{num}/snapshot_{num}.0.hdf5" \
    --group-pattern "../{rung}_output_1024/groups_{num}/fof_subhalo_tab_{num}.0.hdf5" \
    --snap-num 049 --rungs S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
    --axis z --view halo --depth-frac 0.5 \
    --bar-quantity dg --out ladder_dg.png

or bar quantity dz

Tips
----
* Use --npix 256 for a fast draft; bump to 512 for the final figure.
* --half-width defaults to R_200c of halo 0 (read from the S10 group catalog).
* If run directories use zero-padded numbers (S00, S01 …), pass those as --rungs.
* Pass --no-circle to suppress the R_200c ring on every panel.
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

# ── Constants ─────────────────────────────────────────────────────────────────
PROTONMASS = 1.6726e-24
BOLTZMANN  = 1.3806e-16
GAMMA      = 5.0 / 3.0
MU_FIXED   = 0.6
KPC_TO_PC  = 1.0e3
TEAL       = '#2a9d8f'

# Default physics-ladder descriptions (edit to match your actual ladder)
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

# ── Colormaps ─────────────────────────────────────────────────────────────────
def teal_cmap():
    return LinearSegmentedColormap.from_list(
        'dust_teal',
        ['#000000', '#0d4f47', TEAL, '#85d3cb', '#ffffff'],
    )

# ── Unit helpers ──────────────────────────────────────────────────────────────
def to_phys_kpc(x, a, h):   return x * a / h
def to_msun(m, h):           return m * 1e10 / h

def u_to_K(u, xe=None):
    u_cgs = u * 1e10
    mu = (4.0 / (1.0 + 3*0.76 + 4*0.76*xe) if xe is not None else MU_FIXED)
    return u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN

# ── Multi-file snapshot helpers ───────────────────────────────────────────────
def snap_file_list(snap_arg):
    import glob
    base = snap_arg.rstrip('/')
    for pat in ('.0.hdf5', '.hdf5'):      # longer pattern first
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
    return dict(h=h, a=a, z=z)

def read_halo_center(group_file, halo_id, hdr):
    with h5py.File(group_file, 'r') as gf:
        first_sub = int(gf['Group/GroupFirstSub'][halo_id])
        pos   = gf['Subhalo/SubhaloPos'][first_sub]
        r200c = gf['Group/Group_R_Crit200'][halo_id]
    return pos, to_phys_kpc(r200c, hdr['a'], hdr['h'])

def _box_mask(pos, center, half_h, depth_h, axis):
    axes_map = {'x':0,'y':1,'z':2}
    los  = axes_map[axis]
    perp = [i for i in range(3) if i != los]
    dx   = pos - center
    return (np.all(np.abs(dx[:, perp]) < half_h, axis=1) &
            (np.abs(dx[:, los]) < depth_h))

def _read_fields(snap_files, ptype, fields):
    """Read fields across all files; optional fields return None if absent."""
    bufs = {n: [] for n in fields}
    found = False
    for fp in snap_files:
        with h5py.File(fp, 'r') as f:
            if ptype not in f: continue
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

# ── Projection ────────────────────────────────────────────────────────────────
def project_sph(pos2d, mass, hsml, npix, hw, quantity=None,
                smooth_fac=1.0, max_stamp=48):
    """Adaptive-kernel SPH projection.  Returns (mass_map, qty_map)."""
    pix = 2*hw/npix
    px  = (pos2d[:,0]+hw)/pix;  py = (pos2d[:,1]+hw)/pix
    ph  = np.maximum(0.5, hsml/pix)
    ok  = (px>=-max_stamp)&(px<npix+max_stamp)&(py>=-max_stamp)&(py<npix+max_stamp)
    px,py,ph,mass = px[ok],py[ok],ph[ok],mass[ok]
    qty = quantity[ok] if quantity is not None else None

    mass_map = np.zeros((npix,npix),dtype=np.float64)
    qty_map  = np.zeros((npix,npix),dtype=np.float64) if qty is not None else None

    ph_int  = np.clip(ph.astype(int),1,max_stamp)
    order   = np.argsort(ph_int)
    px,py,ph_int,mass = px[order],py[order],ph_int[order],mass[order]
    if qty is not None: qty = qty[order]
    ix = np.round(px).astype(int);  iy = np.round(py).astype(int)

    for h_px in np.unique(ph_int):
        sel = ph_int==h_px
        _ix,_iy,_m = ix[sel],iy[sel],mass[sel]
        _q = qty[sel] if qty is not None else None
        r  = min(int(np.ceil(2.5*h_px)), max_stamp)
        xx = np.arange(-r,r+1,dtype=np.float64)
        K  = np.exp(-0.5*(xx[:,None]**2+xx[None,:]**2)/h_px**2)
        K /= K.sum()
        for i in range(len(_ix)):
            x0,y0 = _ix[i],_iy[i]
            xl=max(0,x0-r); xh=min(npix,x0+r+1)
            yl=max(0,y0-r); yh=min(npix,y0+r+1)
            if xl>=xh or yl>=yh: continue
            kxl=xl-(x0-r); kxh=kxl+(xh-xl)
            kyl=yl-(y0-r); kyh=kyl+(yh-yl)
            mass_map[xl:xh,yl:yh] += _m[i]*K[kxl:kxh,kyl:kyh]
            if qty_map is not None:
                qty_map[xl:xh,yl:yh] += _m[i]*_q[i]*K[kxl:kxh,kyl:kyh]

    if smooth_fac>1.0:
        mass_map = gaussian_filter(mass_map, sigma=smooth_fac)
        if qty_map is not None:
            qty_map  = gaussian_filter(qty_map, sigma=smooth_fac)
    return mass_map, qty_map

def project_points(pos2d, mass, npix, hw, quantity=None, smooth_pkpc=None):
    """Histogram projection for point particles (dust)."""
    edges = np.linspace(-hw,hw,npix+1)
    mmap,_,_ = np.histogram2d(pos2d[:,0],pos2d[:,1],bins=[edges,edges],weights=mass)
    qmap = None
    if quantity is not None:
        qmap,_,_ = np.histogram2d(pos2d[:,0],pos2d[:,1],bins=[edges,edges],
                                   weights=mass*quantity)
    if smooth_pkpc and smooth_pkpc>0:
        sig = smooth_pkpc/(2*hw/npix)
        mmap = gaussian_filter(mmap,sigma=sig)
        if qmap is not None: qmap = gaussian_filter(qmap,sigma=sig)
    return mmap, qmap

def project_dust_adaptive(pos2d, mass, npix, hw,
                           k=16, min_smooth_pkpc=1.0, max_smooth_pkpc=60.0):
    """
    Adaptive-kernel dust projection.
    Smoothing length per particle = distance to kth nearest neighbour,
    clamped to [min_smooth_pkpc, max_smooth_pkpc].
    Dense ISM regions get tight kernels; sparse CGM gets wide kernels.
    Falls back to fixed histogram if too few particles.
    """
    from scipy.spatial import cKDTree

    if len(pos2d) < k + 1:
        # too few particles — fall back to fixed Gaussian
        pix = 2 * hw / npix
        mmap, _, _ = np.histogram2d(
            pos2d[:, 0], pos2d[:, 1],
            bins=[np.linspace(-hw, hw, npix + 1)] * 2,
            weights=mass)
        sig = max(min_smooth_pkpc, 3.0) / pix
        return gaussian_filter(mmap, sigma=sig), None

    tree  = cKDTree(pos2d)
    dists, _ = tree.query(pos2d, k=k + 1)   # k+1 because first hit is self
    hsml  = np.clip(dists[:, -1], min_smooth_pkpc, max_smooth_pkpc)

    return project_sph(pos2d, mass, hsml, npix, hw,
                       smooth_fac=1.0, max_stamp=int(max_smooth_pkpc / (2*hw/npix)) + 2)


# ── Per-rung projection ───────────────────────────────────────────────────────
def project_rung(snap_files, group_file, halo_id, half_width, depth_frac,
                 axis, npix, smooth_fac, dust_smooth, hdr=None):
    """
    Project one rung.  Returns dict:
      gas_sigma, T_mw, dust_sigma  (all npix×npix arrays)
      z, r200_pkpc, n_gas, n_dust
    """
    if hdr is None:
        hdr = read_header(snap_files)

    center, r200 = read_halo_center(group_file, halo_id, hdr)
    hw    = half_width if half_width is not None else r200
    depth = hw * depth_frac
    a,h   = hdr['a'], hdr['h']
    hw_c  = hw   / a * h
    dep_c = depth/ a * h

    # ── Gas ──────────────────────────────────────────────────────────────────
    axes_map = {'x':0,'y':1,'z':2}
    los  = axes_map[axis]; perp = [i for i in range(3) if i!=los]

    gdata = _read_fields(snap_files,'PartType0',{
        'Coordinates':False,'Masses':False,'InternalEnergy':False,
        'SmoothingLength':False,'ElectronAbundance':True})
    assert gdata is not None

    pos  = gdata['Coordinates'];  gmask = _box_mask(pos,center,hw_c,dep_c,axis)
    gpos = to_phys_kpc(pos[gmask]-center,a,h)[:,perp]
    gm   = to_msun(gdata['Masses'][gmask],h)
    gt   = u_to_K(gdata['InternalEnergy'][gmask],
                  gdata['ElectronAbundance'][gmask] if gdata['ElectronAbundance'] is not None else None)
    gh   = to_phys_kpc(gdata['SmoothingLength'][gmask],a,h)

    # Recenter on the innermost gas — more robust than SubhaloPos for ISM zoom.
    # All SPH particles have equal mass so we use spatial proximity instead.
    if len(gpos) > 10:
        r2d    = np.sqrt(gpos[:,0]**2 + gpos[:,1]**2)
        inner  = r2d < hw * 0.4          # innermost 40% of half-width
        if inner.sum() > 5:
            dx_corr = np.mean(gpos[inner, 0])
            dy_corr = np.mean(gpos[inner, 1])
        else:                             # fallback: centroid of all particles
            dx_corr = np.mean(gpos[:, 0])
            dy_corr = np.mean(gpos[:, 1])
        gpos[:, 0] -= dx_corr
        gpos[:, 1] -= dy_corr

    mass_map, temp_map = project_sph(gpos,gm,gh,npix,hw,quantity=gt,
                                     smooth_fac=smooth_fac)
    pix_pc2 = (2*hw/npix * KPC_TO_PC)**2
    gas_sigma = mass_map / pix_pc2
    T_mw      = np.where(mass_map>0, temp_map/mass_map, np.nan)
    # Extra smoothing of T_mw to suppress large-kernel artefacts
    T_mw_s = gaussian_filter(np.nan_to_num(T_mw), sigma=3)
    T_mw   = np.where(mass_map>0, T_mw_s, np.nan)

    # ── Dust ─────────────────────────────────────────────────────────────────
    ddata = _read_fields(snap_files,'PartType6',
                         {'Coordinates':False,'Masses':False})
    if ddata is not None:
        dp   = ddata['Coordinates'];  dmask = _box_mask(dp,center,hw_c,dep_c,axis)
        dpos = to_phys_kpc(dp[dmask]-center,a,h)[:,perp]
        dm   = to_msun(ddata['Masses'][dmask],h)
        # Apply the same centering correction as gas
        if len(gpos) > 10:
            dpos[:, 0] -= dx_corr
            dpos[:, 1] -= dy_corr
        d_mass_map,_ = project_points(dpos,dm,npix,hw,
                                      smooth_pkpc=dust_smooth if dust_smooth>0 else None)
        dust_sigma = d_mass_map / pix_pc2
        n_dust = int(dmask.sum())
    else:
        dust_sigma = np.zeros((npix,npix))
        n_dust = 0

    return dict(gas_sigma=gas_sigma, T_mw=T_mw, dust_sigma=dust_sigma,
                z=hdr['z'], r200_pkpc=r200, hw=hw,
                n_gas=int(gmask.sum()), n_dust=n_dust)


def project_rung_dust(snap_files, group_file, halo_id, half_width, depth_frac,
                      axis, npix, dust_smooth, hdr=None):
    """Dust-only projection — reads gas for centering but skips SPH projection."""
    if hdr is None:
        hdr = read_header(snap_files)
    center, r200 = read_halo_center(group_file, halo_id, hdr)
    hw    = half_width if half_width is not None else r200
    depth = hw * depth_frac
    a, h  = hdr['a'], hdr['h']
    hw_c  = hw   / a * h
    dep_c = depth / a * h

    axes_map = {'x':0,'y':1,'z':2}
    los  = axes_map[axis]; perp = [i for i in range(3) if i != los]

    # ── Gas: positions + metallicity (for centering and D/Z) ──────────────────
    gdata = _read_fields(snap_files,'PartType0',
                         {'Coordinates':False,'Masses':False,'Metallicity':True})
    pos  = gdata['Coordinates']
    gmask = _box_mask(pos, center, hw_c, dep_c, axis)
    gpos  = to_phys_kpc(pos[gmask] - center, a, h)[:, perp]
    gm    = to_msun(gdata['Masses'][gmask], h)

    dx_corr = dy_corr = 0.0

    # ── Stars: primary centering (most reliable at z=0) ───────────────────────
    # Search in a wide sphere (4×R_200c comoving) to catch all disk stars,
    # then iteratively shrink to the densest 20 pkpc core.
    sdata = _read_fields(snap_files,'PartType4',
                         {'Coordinates':False,'Masses':False})
    center_found = False
    if sdata is not None:
        sp    = sdata['Coordinates']
        smask = _box_mask(sp, center, hw_c*4, dep_c*8, axis)
        spos  = to_phys_kpc(sp[smask] - center, a, h)[:, perp]
        sm    = to_msun(sdata['Masses'][smask], h)
        if len(sm) >= 10:
            # Iterative shrinking sphere: start at 0.5*hw, converge to core
            cx, cy = 0.0, 0.0
            for shrink_r in [hw*0.5, hw*0.3, hw*0.15, hw*0.08]:
                r2d_s = np.sqrt((spos[:,0]-cx)**2 + (spos[:,1]-cy)**2)
                mask  = r2d_s < shrink_r
                if mask.sum() >= 5:
                    cx = np.average(spos[mask,0], weights=sm[mask])
                    cy = np.average(spos[mask,1], weights=sm[mask])
            dx_corr, dy_corr = cx, cy
            center_found = True
            print(f'    centering: stars={len(sm):,}  '
                  f'shift=({dx_corr:+.1f}, {dy_corr:+.1f}) pkpc')

    # ── Gas fallback (if no stars found) ─────────────────────────────────────
    if not center_found and len(gpos) > 10:
        r2d   = np.sqrt(gpos[:,0]**2 + gpos[:,1]**2)
        inner = r2d < hw * 0.3
        if inner.sum() > 5:
            dx_corr = np.mean(gpos[inner, 0])
            dy_corr = np.mean(gpos[inner, 1])
        else:
            dx_corr = np.mean(gpos[:, 0])
            dy_corr = np.mean(gpos[:, 1])
        print(f'    centering: gas fallback  '
              f'shift=({dx_corr:+.1f}, {dy_corr:+.1f}) pkpc')

    # ── Dust projection ───────────────────────────────────────────────────────
    # Project into a padded box (15% larger) to avoid hard boundary artefacts,
    # then crop the central hw×hw region for display.
    PAD      = 2.0   # 2× padding: ensures adaptive kernels never hit boundary
    hw_proj  = hw * PAD
    hw_proj_c = hw_proj / a * h
    npix_proj = int(npix * PAD) + 2   # slightly larger pixel grid
    pix_pc2   = (2*hw/npix * KPC_TO_PC)**2   # display pixel area

    ddata = _read_fields(snap_files,'PartType6',
                         {'Coordinates':False,'Masses':False})
    if ddata is not None:
        dp    = ddata['Coordinates']
        dmask = _box_mask(dp, center, hw_proj_c, dep_c, axis)
        dpos  = to_phys_kpc(dp[dmask] - center, a, h)[:, perp]
        dm    = to_msun(ddata['Masses'][dmask], h)
        dpos[:,0] -= dx_corr;  dpos[:,1] -= dy_corr
        if dust_smooth > 0:
            d_full, _ = project_points(dpos, dm, npix_proj, hw_proj,
                                       smooth_pkpc=dust_smooth)
        else:
            d_full, _ = project_dust_adaptive(dpos, dm, npix_proj, hw_proj)
        # Crop central hw region
        lo = int((npix_proj - npix) / 2)
        hi = lo + npix
        d_mass_map = d_full[lo:hi, lo:hi]
        # Rescale surface density: padded pixel area → display pixel area
        pix_proj_pc2 = (2*hw_proj/npix_proj * KPC_TO_PC)**2
        dust_sigma   = d_mass_map / pix_proj_pc2
        n_dust = int(dmask.sum())
        # ── Projected dust masses ─────────────────────────────────────────────
        pix_area_pc2 = (2*hw/npix * KPC_TO_PC)**2
        ii, jj  = np.indices((npix, npix))
        r_map   = np.sqrt((ii - npix/2)**2 + (jj - npix/2)**2) * (2*hw/npix)
        ism_r   = 20.0   # ISM radius = 20 kpc (consistent with analysis scripts)
        ism_px  = r_map < ism_r
        m_dust_ism  = float(np.nansum(dust_sigma[ism_px])  * pix_area_pc2)
        m_dust_cgm  = float(np.nansum(dust_sigma[~ism_px]) * pix_area_pc2)
        m_dust_tot  = m_dust_ism + m_dust_cgm
    else:
        dust_sigma  = np.zeros((npix, npix))
        n_dust = 0
        m_dust_ism = m_dust_cgm = m_dust_tot = 0.0

    # ── D/Z and D/G in ISM ───────────────────────────────────────────────────────
    dz_ism = np.nan
    dg_ism = np.nan
    gpos_all = to_phys_kpc(gdata['Coordinates'][gmask] - center, a, h)[:, perp]
    gpos_all[:, 0] -= dx_corr;  gpos_all[:, 1] -= dy_corr
    r_gas    = np.sqrt(gpos_all[:, 0]**2 + gpos_all[:, 1]**2)
    ism_gas  = r_gas < ism_r
    gm_ism   = to_msun(gdata['Masses'][gmask][ism_gas], h)
    m_gas_ism = float(np.sum(gm_ism))
    if m_gas_ism > 0 and m_dust_ism > 0:
        dg_ism = m_dust_ism / m_gas_ism
    if gdata.get('Metallicity') is not None and m_dust_ism > 0:
        Z_field = gdata['Metallicity']
        if Z_field.ndim > 1:
            Z_field = Z_field[:, 0]
        gz_ism = Z_field[gmask][ism_gas]
        m_metals_ism = float(np.sum(gm_ism * gz_ism))
        if m_metals_ism > 0:
            dz_ism = m_dust_ism / m_metals_ism

    return dict(dust_sigma=dust_sigma, z=hdr['z'], r200_pkpc=r200,
                hw=hw, n_gas=int(gmask.sum()), n_dust=n_dust,
                m_dust_ism=m_dust_ism, m_dust_cgm=m_dust_cgm,
                m_dust_tot=m_dust_tot, dz_ism=dz_ism, dg_ism=dg_ism)

# ── Figure ────────────────────────────────────────────────────────────────────
def _safe_log_bounds(arr, plo=0.5, phi=99.9):
    pos = arr[arr>0]
    if len(pos)==0: return 1e-10, 1.0
    return np.nanpercentile(pos,plo), np.nanpercentile(pos,phi)


def radial_profile(dust_sigma, half_width_pkpc, n_bins=28):
    """Mean dust surface density in annular bins.  Returns (r_pkpc, sigma)."""
    npix = dust_sigma.shape[0]
    pix_kpc = 2*half_width_pkpc / npix
    cy, cx  = npix/2.0, npix/2.0
    iy, ix  = np.indices(dust_sigma.shape)
    r_pix   = np.sqrt((ix-cx)**2 + (iy-cy)**2)
    r_kpc   = r_pix * pix_kpc
    r_edges = np.linspace(0, half_width_pkpc, n_bins+1)
    r_cen   = 0.5*(r_edges[:-1]+r_edges[1:])
    prof    = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r_kpc >= r_edges[i]) & (r_kpc < r_edges[i+1])
        vals = dust_sigma[mask]
        pos  = vals[vals > 0]
        prof[i] = np.mean(pos) if len(pos) > 0 else np.nan
    return r_cen, prof


def make_ladder_figure(rung_maps, rungs, descriptions, axis,
                       show_circle, out_path,
                       vmin_dust=None, vmax_dust=None,
                       bar_quantity='mdust'):
    """
    4-column × 3-row dust-only ladder figure.
    Positions 0-10 : Σ_dust maps for S0–S10.
    Position 11    : radial Σ_dust(r) profiles, all rungs overlaid.
    One shared colorbar on the far right.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap

    n_rungs = len(rungs)     # ≤ 11
    NC, NR  = 4, 3           # 4 columns, 3 rows

    # ── Shared dust colour bounds ─────────────────────────────────────────────
    all_dust = np.concatenate([np.ravel(d['dust_sigma']) for d in rung_maps])
    vd_lo, vd_hi = _safe_log_bounds(all_dust)
    if vmin_dust is not None: vd_lo = vmin_dust
    if vmax_dust is not None: vd_hi = vmax_dust
    norm_dust = LogNorm(vmin=vd_lo, vmax=vd_hi)
    cmap_dust = teal_cmap()

    hw_ref = rung_maps[0]['hw']

    # ── Layout ───────────────────────────────────────────────────────────────
    panel_in = 2.3
    cbar_in  = 0.22
    fig_w = NC * panel_in + cbar_in + 0.55   # 4 panels + cbar + margins
    fig_h = NR * panel_in + 0.55             # 3 rows + top margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        NR, NC + 1,
        figure=fig,
        width_ratios=[1]*NC + [cbar_in/panel_in],
        left=0.06, right=0.97,
        top=0.92,  bottom=0.06,
        hspace=0.04, wspace=0.04,
    )

    # panel axes [row, col]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(NC)] for r in range(NR)]
    cax  = fig.add_subplot(gs[:, NC])   # colorbar spans all rows

    # ── Rung colour ramp for radial profile lines ─────────────────────────────
    profile_cmap = LinearSegmentedColormap.from_list(
        'rung_seq', ['#1d6b5e', '#2a9d8f', '#85d3cb', '#ddf5f0', '#ffffff'], N=256)
    n_total = 11   # always 11 colours so S-indices stay consistent
    def rung_color(idx):
        return profile_cmap(idx / (n_total - 1))

    ax_label = {'x':('y','z'),'y':('x','z'),'z':('x','y')}[axis]

    # ── Draw dust panels ──────────────────────────────────────────────────────
    for idx, (rung, data) in enumerate(zip(rungs, rung_maps)):
        row, col = divmod(idx, NC)
        ax  = axes[row][col]
        hw  = data['hw']
        img = data['dust_sigma']

        ax.set_facecolor('black')
        if np.any(img > 0):
            # NaN for empty pixels → rendered as cmap 'bad' colour (black)
            cmap_disp = teal_cmap()
            cmap_disp.set_bad('black')
            cmap_disp.set_under('black')
            disp = np.where((img > 0) & np.isfinite(img), img, np.nan)
            ax.imshow(disp.T, origin='lower',
                      extent=[-hw, hw, -hw, hw],
                      cmap=cmap_disp, norm=norm_dust,
                      interpolation='lanczos', aspect='equal')
        else:
            ax.text(0.5, 0.5, 'no dust', transform=ax.transAxes,
                    ha='center', va='center', color=TEAL, fontsize=8, style='italic')

        ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)

        # R_200c circle
        if show_circle:
            th = np.linspace(0, 2*np.pi, 360)
            ax.plot(data['r200_pkpc']*np.cos(th), data['r200_pkpc']*np.sin(th),
                    '--', color='white', lw=0.5, alpha=0.45)

        # Ticks
        bottom_row = (row == NR-1) or (idx == n_rungs-1)
        left_col   = (col == 0)
        ax.tick_params(colors='white', labelcolor='black', labelsize=8,
                       direction='in', top=True, right=True,
                       labelbottom=bottom_row, labelleft=left_col)
        for sp in ax.spines.values():
            sp.set_edgecolor('#aaaaaa'); sp.set_linewidth(0.5)

        if bottom_row:
            ax.set_xlabel(f'{ax_label[0]} (kpc)', color='black', fontsize=9)
        if left_col:
            ax.set_ylabel(f'{ax_label[1]} (kpc)', color='black', fontsize=9, labelpad=2)

        # Rung label (top-left corner)
        desc = descriptions.get(rung, '')
        label = (f'{rung}  {desc}' if desc else rung)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha='left', va='top', color='white',
                fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))

        # Outline the fiducial (last rung = S10)
        if idx == n_rungs - 1:
            for sp in ax.spines.values():
                sp.set_edgecolor(TEAL); sp.set_linewidth(1.2)

    # ── 12th panel: M_dust ISM vs CGM per rung ──────────────────────────────────
    ax_bar = axes[NR-1][NC-1]
    ax_bar.set_facecolor('black')
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('white'); sp.set_linewidth(0.4)

    all_rung_names = ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
    n_r  = len(rungs)
    ypos = np.arange(n_r)

    m_ism = np.array([d['m_dust_ism'] for d in rung_maps])
    m_cgm = np.array([d['m_dust_cgm'] for d in rung_maps])

    ism_r    = 20.0
    CGM_COL  = '#777777'
    from matplotlib.patches import Patch as _Patch
    import matplotlib.ticker as _ticker

    def _ratio_bars(ax, vals, xlabel, ref_val, ref_lbl):
        for i, (rung, v) in enumerate(zip(rungs, vals)):
            try:   s_idx = all_rung_names.index(rung)
            except ValueError: s_idx = i
            col = rung_color(s_idx)
            if np.isfinite(v) and v > 0:
                ax.barh(ypos[i], v, height=0.55, color=col, alpha=0.95, zorder=3)
        ax.axvline(ref_val, color='white', lw=1.0, ls='--', alpha=0.6)
        ax.text(ref_val * 1.03, -0.7, ref_lbl, color='white', fontsize=6.5, va='top')
        valid = vals[np.isfinite(vals)]
        x_hi = max(float(np.nanmax(valid))*1.2, ref_val*1.5) if len(valid) else ref_val*2
        ax.set_xlim(0, x_hi)
        ax.set_xscale('linear')
        ax.xaxis.set_major_formatter(_ticker.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel, color='black', fontsize=9)

    if bar_quantity == 'dz':
        dz_vals = np.array([d.get('dz_ism', np.nan) for d in rung_maps])
        _ratio_bars(ax_bar, dz_vals,
                    r'$\mathcal{D}/Z_\mathrm{ISM}$  ($r < 20$ kpc)',
                    ref_val=0.4, ref_lbl='MW')
    elif bar_quantity == 'dg':
        dg_vals = np.array([d.get('dg_ism', np.nan) for d in rung_maps])
        _ratio_bars(ax_bar, dg_vals,
                    r'$\mathcal{D}/G_\mathrm{ISM}$  ($r < 20$ kpc)',
                    ref_val=0.01, ref_lbl='MW')
    else:
        # ── M_dust ISM + CGM bar chart ────────────────────────────────────────
        for i, (rung, mi, mc) in enumerate(zip(rungs, m_ism, m_cgm)):
            try:   s_idx = all_rung_names.index(rung)
            except ValueError: s_idx = i
            col = rung_color(s_idx)
            if mc > 0:
                ax_bar.barh(ypos[i], mc, height=0.45,
                            color=CGM_COL, alpha=0.75, zorder=2, left=mi)
            ax_bar.barh(ypos[i], mi, height=0.55,
                        color=col, alpha=0.95, zorder=3)
        ax_bar.set_xscale('log')
        ax_bar.set_xlabel(r'$M_\mathrm{dust}$  (M$_\odot$)', color='black', fontsize=9)
        _lc = rung_color(10)
        ax_bar.legend(
            [_Patch(facecolor=_lc, alpha=0.95),
             _Patch(facecolor=CGM_COL, alpha=0.75)],
            [rf'$r < {ism_r:.0f}$ kpc (ISM)', rf'$r > {ism_r:.0f}$ kpc (CGM)'],
            fontsize=6.5, framealpha=0.0, labelcolor='white',
            loc='upper right', handlelength=1.2, borderpad=0.4, labelspacing=0.4)

    # Shared bar-chart formatting (both modes)
    for i, rung in enumerate(rungs):
        desc = descriptions.get(rung, '')
        lbl  = f'{rung}  {desc}' if desc else rung
        ax_bar.text(0.02, ypos[i], lbl,
                    transform=ax_bar.get_yaxis_transform(),
                    va='center', ha='left', color='white',
                    fontsize=6.0, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, lw=0))

    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.tick_params(colors='white', labelcolor='black', labelsize=8,
                       direction='in', top=True, right=True, which='both')
    ax_bar.grid(True, axis='x', which='major', color='white', alpha=0.08, lw=0.4)



    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap_dust, norm=norm_dust)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.set_ylabel(r'$\Sigma_\mathrm{dust}$  (M$_\odot$ pc$^{-2}$)',
                    rotation=-90, va='bottom', color='black', fontsize=9, labelpad=10)
    cb.ax.yaxis.set_tick_params(color='black', labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')
    cb.outline.set_edgecolor('black'); cb.outline.set_linewidth(0.5)

    # ── Scale bar (first panel) ───────────────────────────────────────────────
    ax0  = axes[0][0]
    sb   = _nice_scalebar(hw_ref)
    x0   = -hw_ref + 0.07*2*hw_ref
    y0   = -hw_ref + 0.06*2*hw_ref
    ax0.plot([x0, x0+sb], [y0, y0], '-', color='white', lw=1.2)
    ax0.text(x0+sb/2, y0+0.04*2*hw_ref, f'{sb:.0f} kpc',
             ha='center', va='bottom', color='white', fontsize=8)

    # ── z label ───────────────────────────────────────────────────────────────
    z_ref = rung_maps[-1]['z']
    fig.text(0.98, 0.97, f'$z={z_ref:.2f}$', ha='right', va='top',
             color='black', fontsize=10, transform=fig.transFigure)

    # ── Column title ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             r'CosmicGrain — Halo 569    $\Sigma_\mathrm{dust}$ physics ladder',
             ha='center', va='top', color='black', fontsize=11,
             transform=fig.transFigure)

    # ── Save ──────────────────────────────────────────────────────────────────
    # Extra right-margin padding so the profile panel's right-side y-label
    # isn't clipped by bbox_inches='tight'
    fig.subplots_adjust(right=0.94)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pad_inches=0.08)
    plt.close(fig)
    print(f'  Saved → {out_path}')

def _nice_scalebar(hw):
    for s in [500,200,100,50,20,10,5]:
        if s < hw*0.9: return float(s)
    return max(1.0, round(hw*0.2,-1))

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snap-pattern',  required=True,
        help='Path template with {rung} and {num} placeholders, e.g. '
             '"../S{rung}_output_2048/snapdir_{num}/snapshot_{num}.0.hdf5"')
    p.add_argument('--group-pattern', required=True,
        help='Same for group catalog, e.g. '
             '"../S{rung}_output_2048/groups_{num}/fof_subhalo_tab_{num}.0.hdf5"')
    p.add_argument('--snap-num',  required=True,
        help='Snapshot number string as it appears in filenames, e.g. "027"')
    p.add_argument('--rungs', nargs='+',
        default=['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
        help='Rung labels in order (default S0 … S10)')
    p.add_argument('--halo-id',   type=int, default=0)
    p.add_argument('--view', choices=['ism','halo'], default=None,
        help='ism = 20 kpc half-width; halo = R_200c (default)')
    p.add_argument('--half-width',type=float,default=None,
        help='Half-width in kpc; overridden by --view if both given')
    p.add_argument('--depth-frac',type=float,default=0.5)
    p.add_argument('--axis',      choices=['x','y','z'],default='z')
    p.add_argument('--npix',      type=int,  default=512,
        help='Pixels per panel (default 512)')
    p.add_argument('--dust-smooth',type=float,default=0.0,
        help='Gaussian smoothing for dust panels in pkpc (default 0 = adaptive SPH-style)')
    p.add_argument('--n-rbins',   type=int,  default=28,
        help='Radial bins in the profile panel (default 28)')
    p.add_argument('--no-circle', action='store_true',
        help='Suppress R_200c circle on every panel')
    p.add_argument('--no-descriptions', action='store_true',
        help='Show rung labels only, no physics descriptions')
    p.add_argument('--bar-quantity', choices=['mdust','dz','dg'], default='mdust',
        help='Panel 12: mdust=M_dust, dz=D/Z_ISM, dg=D/G_ISM')
    p.add_argument('--vmin-dust', type=float,default=None)
    p.add_argument('--vmax-dust', type=float,default=None)
    p.add_argument('--out',       default='ladder.pdf')
    return p.parse_args()


def main():
    args = parse_args()

    descriptions = {} if args.no_descriptions else RUNG_DESCRIPTIONS

    # ── Determine shared half-width from the last rung in the list ───────────
    ref_rung = args.rungs[-1]   # use the most complete rung for R_200c
    ref_snap   = args.snap_pattern .format(rung=ref_rung, num=args.snap_num)
    ref_groups = args.group_pattern.format(rung=ref_rung, num=args.snap_num)
    ref_files  = snap_file_list(ref_snap)
    if ref_files is None:
        sys.exit(f'ERROR: cannot find snapshot for reference rung {ref_rung}: {ref_snap}')
    ref_hdr = read_header(ref_files)
    _, r200_ref = read_halo_center(ref_groups, args.halo_id, ref_hdr)
    if getattr(args, 'view', None) == 'ism':
        half_width = 20.0
    elif getattr(args, 'view', None) == 'halo':
        half_width = r200_ref
    elif args.half_width is not None:
        half_width = args.half_width
    else:
        half_width = r200_ref
    print(f'Reference rung: {ref_rung},  R_200c = {r200_ref:.1f} pkpc')
    print(f'Projection half-width: {half_width:.1f} pkpc,  '
          f'depth: {half_width*args.depth_frac:.1f} pkpc')

    # ── Project all rungs ────────────────────────────────────────────────────
    rung_maps = []
    for rung in args.rungs:
        snap_path  = args.snap_pattern .format(rung=rung, num=args.snap_num)
        group_path = args.group_pattern.format(rung=rung, num=args.snap_num)

        snap_files = snap_file_list(snap_path)
        if snap_files is None:
            print(f'  [{rung}] WARNING: snapshot not found at {snap_path} — skipping')
            continue
        n_f = len(snap_files)
        print(f'\n[{rung}]  {os.path.basename(snap_files[0])}'
              f'{f"  (+{n_f-1} more)" if n_f>1 else ""}')

        hdr = read_header(snap_files)
        data = project_rung_dust(
            snap_files, group_path, args.halo_id,
            half_width, args.depth_frac, args.axis,
            args.npix, args.dust_smooth, hdr=hdr)

        print(f'  z={data["z"]:.3f}  R200={data["r200_pkpc"]:.1f} pkpc  '
              f'gas={data["n_gas"]:,}  dust={data["n_dust"]:,}')
        rung_maps.append(data)

    if not rung_maps:
        sys.exit('ERROR: no rungs projected successfully.')

    actual_rungs = args.rungs[:len(rung_maps)]  # in case some were skipped

    # ── Render figure ────────────────────────────────────────────────────────
    print(f'\nRendering 4×3 dust ladder → {args.out}')
    make_ladder_figure(
        rung_maps, actual_rungs, descriptions,
        axis=args.axis,
        show_circle=not args.no_circle,
        out_path=args.out,
        vmin_dust=args.vmin_dust, vmax_dust=args.vmax_dust,
        bar_quantity=args.bar_quantity,
    )


if __name__ == '__main__':
    main()
