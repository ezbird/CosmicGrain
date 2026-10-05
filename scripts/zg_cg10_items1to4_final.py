#!/usr/bin/env python3
"""Matched Zhang--Green / CosmicGrain differential-extinction comparison.

Requires numpy, scipy, h5py, matplotlib, astropy, astropy-healpix. Run from
~/gadget4 with this file in scripts/ and cosmicgrain_z0_z6_gsd_rv.py beside it.
The ZG input is the original Rv_map_new.h5; the simulation is a z=0 snapshot.

Example:
 python3 scripts/zg_cg10_items1to4_final.py Rv_map_new.h5 \
   --root . --output-prefix cg10_zg_baseline_250pc

To analyze the separate MinGasHsml=0.0 run from ~/gadget4 instead:
 python3 scripts/zg_cg10_items1to4_final.py Rv_map_new.h5 \
   --output-dir halo441_output_1024_mingashsml0.0 \
   --output-prefix cg10_zg_mingashsml0_250pc

To adopt a 0.01-mag lower layer-extinction floor while auditing nearby choices:
 python3 scripts/zg_cg10_items1to4_final.py Rv_map_new.h5 \
   --output-dir halo441_output_1024_mingashsml0.0 \
   --minimum-delta-av 0.01 \
   --av-floor-sweep 0,0.001,0.003,0.01,0.03,0.1 \
   --output-prefix cg10_zg_avfloor_250pc

The 3D A_V quantity is the differential extinction across a common dz layer,
not cumulative extinction from the Sun. FFT outputs are pseudo-spectra: masks
couple Fourier modes and no mask deconvolution or dust shot-noise subtraction
is attempted. Resolution-sensitive bins must be excluded from science claims.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, label as connected_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR/'scripts', SCRIPT_DIR.parent/'scripts'):
    if (candidate/'cosmicgrain_z0_z6_gsd_rv.py').is_file():
        ASSET_DIR = candidate
        sys.path.insert(0, str(candidate))
        break
else:
    raise SystemExit('Could not find cosmicgrain_z0_z6_gsd_rv.py beside this script or in gadget4/scripts')
import cosmicgrain_z0_z6_gsd_rv as cg
from halo_utils import get_zoom_halo_series, periodic_delta

MSUN = 1.98847e33
KPC_CM = 3.085677581491367e21
SCRIPT_VERSION = '2026-10-05-v20-native-r55-two-panel-structure-function'

def apply_cosmicgrain_style(style_file=None):
    candidates=[]
    if style_file is not None:
        candidates.append(Path(style_file))
    candidates.extend((SCRIPT_DIR/'cosmicgrain.mplstyle',
                       ASSET_DIR/'cosmicgrain.mplstyle'))
    for path in candidates:
        if path.is_file():
            plt.style.use(path)
            print(f'Using Matplotlib style: {path}')
            return
    raise FileNotFoundError('cosmicgrain.mplstyle not found; pass --style-file')

def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('zg_h5', type=Path)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--snapshot', type=int, default=29)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--group', type=int, default=None)
    p.add_argument('--carbon-table', type=Path, default=Path('dust_optics/callqcomp.out_D16MGemt'))
    p.add_argument('--silicate-table', type=Path, default=Path('dust_optics/callqcomp.out_DH16asil'))
    p.add_argument('--output-prefix', default='cg10_zg_250pc')
    p.add_argument('--nside', default='combined')
    p.add_argument('--half-size', type=float, default=2.5, help='Half-width in physical kpc')
    p.add_argument('--radius', type=float, default=2.5, help='Circular analysis radius in kpc')
    p.add_argument('--cg-patch-radius', type=float, default=8.0,
                   help='Galactocentric radius of the CG10 patch center in the face-on disk; use 0 for the nucleus')
    p.add_argument('--cg-patch-azimuth', type=float, default=0.0,
                   help='Azimuth of the CG10 patch center in face-on degrees')
    p.add_argument('--zmax', type=float, default=0.4)
    p.add_argument('--pixel', type=float, default=0.25)
    p.add_argument('--dz', type=float, default=0.2)
    p.add_argument('--high-cut-quantile', type=float, default=0.95,
                   help='ZG positive layer A_V upper quantile used to test loss of dense/high-extinction cells')
    p.add_argument('--minimum-delta-av', type=float, default=0.0,
                   help='Matched lower Delta A_V cut in mag per dz layer, applied to both ZG and CG10')
    p.add_argument('--av-floor-sweep', default='0,0.001,0.003,0.01,0.03,0.1',
                   help='Comma-separated matched Delta A_V floors for the sensitivity audit; empty disables it')
    p.add_argument('--apodize', type=float, default=2.0, help='Taper width in pixels')
    p.add_argument('--n-power-bins', type=int, default=14)
    p.add_argument('--style-file', type=Path, default=None)
    p.add_argument('--paper-rv-min', type=float, default=1.5)
    p.add_argument('--paper-rv-max', type=float, default=10.0)
    p.add_argument('--minimum-common-pixels-for-power', type=int, default=100)
    p.add_argument('--minimum-largest-component-pixels', type=int, default=50)
    p.add_argument('--minimum-largest-component-fraction', type=float, default=0.5)
    return p.parse_args()

def grid(a):
    n = round(2*a.half_size/a.pixel)
    nz = round(2*a.zmax/a.dz)
    if n < 8 or nz < 1 or abs(n*a.pixel-2*a.half_size)>1e-8 or abs(nz*a.dz-2*a.zmax)>1e-8:
        raise ValueError('Box width and slab depth must be integer multiples of pixel and dz')
    e = np.linspace(-a.half_size, a.half_size, n+1)
    ze = np.linspace(-a.zmax, a.zmax, nz+1)
    c = (e[1:]+e[:-1])/2
    zc = (ze[1:]+ze[:-1])/2
    xx, yy = np.meshgrid(c,c)
    return e, ze, xx, yy, zc

def zg_cube(a, xx, yy, zc):
    qx,qy,zp=xx,yy,zc
    with h5py.File(a.zg_h5,'r') as f:
        bins=np.asarray(f['distance_bins'],float)
        mg=f[f'map_{a.nside}']
        ns=int(mg['nside'][()]); order=mg['hp_order'][()]
        if isinstance(order,bytes): order=order.decode()
        hp=HEALPix(nside=ns,order=str(order),frame='galactic')
        av=np.full((len(zp),*qx.shape),np.nan)
        rv=np.full_like(av,np.nan)
        for j,z in enumerate(zp):
            coords=SkyCoord(u=qx.ravel()*u.kpc,v=qy.ravel()*u.kpc,
                            w=np.full(qx.size,z)*u.kpc,frame='galactic',
                            representation_type='cartesian')
            di=np.searchsorted(bins,coords.spherical.distance.to_value(u.kpc))-1
            hi=hp.skycoord_to_healpix(coords)
            aa=np.full(qx.size,np.nan); rr=np.full(qx.size,np.nan)
            for ib in np.unique(di[(di>=0)&(di<len(bins)-1)]):
                q=di==ib
                aa[q]=np.asarray(mg['AV_map_diff'][int(ib),:],float)[hi[q]]*a.dz
                # Compare in the map's native monochromatic convention.
                rr[q]=np.asarray(mg['R55_map_diff'][int(ib),:],float)[hi[q]]
            av[j]=aa.reshape(qx.shape); rv[j]=rr.reshape(qx.shape)
    assert av.shape==(len(zc),*xx.shape)
    return av,rv

def sim_cube(a, e, ze):
    out = a.output_dir or a.root/'halo441_output_1024'
    if a.group is not None:
        group = a.group
    else:
        try:
            group = cg.find_audit_group(out)
        except FileNotFoundError:
            # CG10 (parent halo 441) has audited final snapshot group index 1.
            # The audit file is not required if this output lacks a copy.
            group = 1
            print(f'No audit JSON found in {out}; using audited CG10 final group index 1')
    series = get_zoom_halo_series(out,[a.snapshot], group_index=group,
            reference_snap=a.snapshot,verbose=True,refine_reference_center=True,
            refine_historical_centers=False)
    halo = series[a.snapshot]
    scale, h, box = cg.read_header(out,a.snapshot)
    dust = cg.read_dust(out,a.snapshot)
    rot, source, nrot = cg.derive_orientation(out,a.snapshot,halo,scale,h,box,20.)
    xyz_gal = periodic_delta(dust['coords'],np.asarray(halo['center']),box)*scale/h @ rot.T
    phi=np.deg2rad(a.cg_patch_azimuth)
    patch_center=np.array([a.cg_patch_radius*np.cos(phi),
                           a.cg_patch_radius*np.sin(phi),0.0])
    xyz=xyz_gal-patch_center
    xe=e
    zz=ze
    inside = (np.linalg.norm(xyz_gal,axis=1)<halo['r200_pkpc']) & \
             (np.abs(xyz[:,0])<a.half_size) & \
             (np.abs(xyz[:,1])<a.half_size) & \
             (np.abs(xyz[:,2])<a.zmax)
    xyz=xyz[inside]
    radius=dust['radius_nm'][inside]*1e-3 # micron
    f=np.clip(dust['carbon'][inside],0,1)
    mass=dust['mass'][inside]*1e10/h*MSUN
    cb=cg.read_draine_qext_table(a.carbon_table)
    sb=cg.read_draine_qext_table(a.silicate_table)
    # tau = m * [3 Q_ext/(4 rho a)] / pixel area. Mixture is by mass.
    def opacity_and_tau_weight(wave):
        # C_ext/m = 3 Q_ext/(4 rho a) for a spherical grain.
        qc=cg.qext_at_wavelength(cb,radius,wave)
        qs=cg.qext_at_wavelength(sb,radius,wave)
        kap=3/(4*radius*1e-4)*(f*qc/cg.RHO_C+(1-f)*qs/cg.RHO_SIL)
        # mass*kappa is extinction cross-section [cm^2]; divide by the
        # physical x-y cell area to obtain dimensionless optical depth.
        tau_weight=mass*kap/(a.pixel*KPC_CM)**2
        return kap,tau_weight
    kb,wb=opacity_and_tau_weight(cg.B_MICRON)
    kv,wv=opacity_and_tau_weight(cg.V_MICRON)
    shape=(len(ze)-1,len(e)-1,len(e)-1)
    bins=(zz,xe,xe)
    pts=xyz[:,[2,1,0]]
    tb=np.histogramdd(pts,bins=bins,weights=wb)[0]
    tv=np.histogramdd(pts,bins=bins,weights=wv)[0]
    counts=np.histogramdd(pts,bins=bins)[0].astype(np.int32)

    in_ext=(pts[:,0]>=zz[0])&(pts[:,0]<zz[-1]) & \
           (pts[:,1]>=xe[0])&(pts[:,1]<xe[-1]) & \
           (pts[:,2]>=xe[0])&(pts[:,2]<xe[-1])
    direct_tv=float(np.sum(wv[in_ext]))
    hist_tv=float(np.sum(tv))
    print('\nCG ABSOLUTE A_V NORMALIZATION AUDIT')
    print('  Dust mass: Masses * 1e10/h * Msun -> g')
    print('  Grain radius: nm * 1e-3 -> micron; micron * 1e-4 -> cm')
    print('  kappa_V = 3/(4a) [f_C Q_C/rho_C + (1-f_C) Q_sil/rho_sil]')
    print('  tau_V particle = M_dust * kappa_V / physical pixel area')
    print('  Delta A_V layer = 1.086 * Delta tau_V [mag]')
    print(f'  Physical pixel area = ({a.pixel:g} kpc)^2 = {(a.pixel*KPC_CM)**2:.6e} cm^2')
    print(f'  dz = {a.dz:g} kpc: bin thickness only; CG tau is NOT multiplied by dz')
    print(f'  Dust particles in extended grid = {int(in_ext.sum()):,}')
    print(f'  Dust mass in extended grid = {np.sum(mass[in_ext])/MSUN:.6e} Msun')
    print('  radius [micron] P16/P50/P84 =',
          np.array2string(np.percentile(radius[in_ext],[16,50,84]),precision=6))
    print('  kappa_V [cm^2/g] P16/P50/P84 =',
          np.array2string(np.percentile(kv[in_ext],[16,50,84]),precision=6))
    print(f'  direct sum particle tau_V weights = {direct_tv:.12e}')
    print(f'  histogram sum tau_V             = {hist_tv:.12e}')
    print(f'  histogram/direct ratio          = {hist_tv/direct_tv:.12f}')
    print('  No additional Gaussian smoothing applied.')
    assert tb.shape==shape
    av=1.086*tv
    print(f'  cropped target sum tau_V        = {np.sum(tv):.12e}')
    print(f'  cropped target sum Delta A_V    = {np.sum(av):.12e} mag')
    finite_av=av[np.isfinite(av)&(av>0)]
    if finite_av.size:
        print('  positive target Delta A_V [mag/layer] P5/P10/P50/P90/P95 =',
              np.array2string(np.percentile(finite_av,[5,10,50,90,95]),precision=6))
    print('  CHECK: CG Delta A_V is absolute mag per '
          f'{a.dz:g}-kpc layer, directly comparable to ZG AV_map_diff*dz.\n')
    rv=np.divide(tv,tb-tv,out=np.full_like(tv,np.nan),where=tb>tv)
    return av,rv,counts,{'group':group,'redshift':1/scale-1,'dust_particles_near_box':int(inside.sum()),
                         'orientation':source,'orientation_particles':int(nrot),
                         'cg_patch_radius_pkpc':float(a.cg_patch_radius),
                         'cg_patch_azimuth_deg':float(a.cg_patch_azimuth)}

def zg_completeness_diagnostic(av, rv, footprint, nbins=18):
    """Measure ZG R(55)-valid fraction as a function of positive 3-D layer A_V.

    Denominator: queried cells with finite positive A_V inside the aperture.
    Numerator: those same cells with finite positive R(55).
    The plateau is estimated from the highest-A_V bins containing data.  The
    reported 80/90/95% thresholds are the first bins that reach the requested
    fraction of that plateau and remain consistent with it at higher A_V.
    """
    av_ok = np.isfinite(av) & (av > 0) & footprint[None]
    rv_ok = av_ok & np.isfinite(rv) & (rv > 0)
    vals = av[av_ok]
    if vals.size < 20:
        raise RuntimeError('Too few positive ZG A_V cells for completeness diagnostic')
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    edges = np.geomspace(lo, hi * (1 + 1e-10), nbins + 1)
    cen = np.sqrt(edges[:-1] * edges[1:])
    total = np.histogram(av[av_ok], bins=edges)[0]
    valid = np.histogram(av[rv_ok], bins=edges)[0]
    frac = np.divide(valid, total, out=np.full(nbins, np.nan), where=total > 0)

    occupied = np.flatnonzero(total > 0)
    # Use up to the five highest occupied bins for the high-A_V plateau.
    high = occupied[-min(5, occupied.size):]
    plateau = float(np.average(frac[high], weights=total[high]))

    thresholds = {}
    for target in (0.80, 0.90, 0.95):
        level = target * plateau
        threshold = np.nan
        # Require this and all subsequently occupied bins to be >= level.
        for i in occupied:
            later = occupied[occupied >= i]
            if frac[i] >= level and np.all(frac[later] >= level):
                threshold = float(edges[i])
                break
        thresholds[target] = threshold
    return dict(edges=edges, centers=cen, total=total, valid=valid, fraction=frac,
                plateau=plateau, thresholds=thresholds, av_ok=av_ok, rv_ok=rv_ok)

def project(av,rv,select):
    # Extinction adds along the sightline. For R(55), add monochromatic color
    # excess E(44-55)=A(55)/R(55), then form the projected ratio.
    good = select & np.isfinite(av) & np.isfinite(rv) & (av > 0) & (rv > 0)
    av_cells = np.where(good, av, 0.0)
    ebv_cells = np.where(good, av / rv, 0.0)
    avmap = av_cells.sum(axis=0)
    ebvmap = ebv_cells.sum(axis=0)
    rmap = np.divide(avmap, ebvmap, out=np.full_like(avmap, np.nan), where=ebvmap > 0)
    return avmap, rmap, avmap > 0

def mask_connectivity(mask):
    """Return 8-connected mask statistics used to gate spatial power."""
    n=int(mask.sum())
    labels,ncomp=connected_label(mask,np.ones((3,3),dtype=int))
    sizes=np.bincount(labels.ravel())[1:] if ncomp else np.array([],dtype=int)
    largest=int(sizes.max()) if sizes.size else 0
    return {'pixels':n,'components':int(ncomp),'largest_component_pixels':largest,
            'largest_component_fraction':float(largest/n) if n else 0.0}

def power_decision(a,stats):
    reasons=[]
    if stats['pixels'] < a.minimum_common_pixels_for_power:
        reasons.append(f"common pixels {stats['pixels']} < {a.minimum_common_pixels_for_power}")
    if stats['largest_component_pixels'] < a.minimum_largest_component_pixels:
        reasons.append(f"largest component {stats['largest_component_pixels']} < {a.minimum_largest_component_pixels}")
    if stats['largest_component_fraction'] < a.minimum_largest_component_fraction:
        reasons.append('largest-component fraction '
                       f"{stats['largest_component_fraction']:.3f} < {a.minimum_largest_component_fraction:.3f}")
    return not reasons,reasons

def parse_floor_sweep(text):
    """Parse, validate, sort and de-duplicate the requested A_V floors."""
    if text is None or not str(text).strip():
        return []
    try:
        values=[float(x.strip()) for x in str(text).split(',') if x.strip()]
    except ValueError as exc:
        raise ValueError('--av-floor-sweep must be a comma-separated list of numbers') from exc
    if any((not np.isfinite(x)) or x < 0 for x in values):
        raise ValueError('All A_V floors must be finite and non-negative')
    return sorted(set(values))

def structure_function(field,valid,dx,edges):
    """Second-order 2-D structure function on an irregular common mask.

    Every unordered valid-pixel pair contributes once.  Unlike an FFT
    pseudo-spectrum this estimator does not fill holes or multiply the field by
    a window, making it the primary spatial diagnostic for fragmented masks.
    """
    ny,nx=field.shape
    sums=np.zeros(len(edges)-1,float)
    counts=np.zeros(len(edges)-1,np.int64)
    for dy in range(ny):
        for dxpix in range(-(nx-1),nx):
            if dy == 0 and dxpix <= 0:
                continue
            if dxpix >= 0:
                y0=slice(0,ny-dy); y1=slice(dy,ny)
                x0=slice(0,nx-dxpix); x1=slice(dxpix,nx)
            else:
                shift=-dxpix
                y0=slice(0,ny-dy); y1=slice(dy,ny)
                x0=slice(shift,nx); x1=slice(0,nx-shift)
            pair=valid[y0,x0]&valid[y1,x1]
            n=int(pair.sum())
            if not n:
                continue
            separation=np.hypot(dy,dxpix)*dx
            ib=np.searchsorted(edges,separation,side='right')-1
            if 0 <= ib < len(sums):
                delta=field[y1,x1][pair]-field[y0,x0][pair]
                sums[ib]+=float(np.sum(delta*delta))
                counts[ib]+=n
    values=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    return values,counts

def build_floor_case(a,gz,gr,sz,sr,scount,gvalid,svalid,footprint,highcut,floor):
    """Build matched projected maps and diagnostics for one lower A_V floor."""
    result={}
    selectors={
        'ZG':gvalid&(gz>=floor)&(gz<=highcut),
        'CG10':svalid&(sz>=floor)&(sz<=highcut),
    }
    for label,av,rv in (('ZG',gz,gr),('CG10',sz,sr)):
        av2,rv2,have=project(av,rv,selectors[label])
        avmask=footprint&have&np.isfinite(av2)&(av2>0)
        rvmask=avmask&np.isfinite(rv2)&(rv2>=a.paper_rv_min)&(rv2<=a.paper_rv_max)
        result[f'{label}_av']=av2; result[f'{label}_rv']=rv2
        result[f'{label}_av_mask']=avmask; result[f'{label}_rv_mask']=rvmask
    result['common_av']=result['ZG_av_mask']&result['CG10_av_mask']
    result['common_rv']=result['ZG_rv_mask']&result['CG10_rv_mask']
    result['connectivity']=mask_connectivity(result['common_rv'])
    result['projected_counts']=np.sum(np.where(selectors['CG10'],scount,0),axis=0)
    return result

def percentile_summary(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    if not values.size:
        return {'p16':np.nan,'median':np.nan,'p84':np.nan}
    q=np.percentile(values,[16,50,84])
    return {'p16':float(q[0]),'median':float(q[1]),'p84':float(q[2])}

def spectrum(field,valid,dx,taper,edges):
    if valid.sum()<12: return np.full(len(edges)-1,np.nan),np.zeros(len(edges)-1,int),0
    dist=distance_transform_edt(np.pad(valid,1))[1:-1,1:-1]
    win=np.where(valid,np.sin(.5*np.pi*np.clip(dist/max(taper,1e-9),0,1))**2,0.)
    mu=np.sum(win*np.where(valid,field,0))/win.sum()
    ff=np.fft.fft2(win*np.where(valid,field-mu,0))*dx**2
    area=np.sum(win**2)*dx**2
    p=np.abs(ff)**2/area
    ny,nx=field.shape
    ky=2*np.pi*np.fft.fftfreq(ny,dx); kx=2*np.pi*np.fft.fftfreq(nx,dx)
    kk=np.hypot(*np.meshgrid(kx,ky))
    vals=[]; counts=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(kk>=lo)&(kk<hi)
        counts.append(int(q.sum()));vals.append(float(p[q].mean()) if q.any() else np.nan)
    return np.array(vals),np.array(counts),float(area)

def paper_figure(a, maps, common, rows, kc, highcut, prefix, power_ok, power_reasons, mask_stats):
    """Four-panel Milky Way benchmark using the common upper-cut sample."""
    mask=common[('uppercut','rv')]
    zg=maps['ZG_uppercut_rv'].copy()
    sim=maps['CG10_uppercut_rv'].copy()
    zg[~mask]=np.nan; sim[~mask]=np.nan
    extent=[-a.half_size,a.half_size,-a.half_size,a.half_size]

    fig=plt.figure(figsize=(12.8,9.4))
    gs=fig.add_gridspec(2,2,height_ratios=(1.05,0.9),hspace=0.29,wspace=0.24)
    ax_zg=fig.add_subplot(gs[0,0]); ax_cg=fig.add_subplot(gs[0,1])
    ax_pdf=fig.add_subplot(gs[1,0]); ax_ps=fig.add_subplot(gs[1,1])

    image_axes=[]
    for ax,arr,title,panel in ((ax_zg,zg,'Zhang & Green','(a)'),
                               (ax_cg,sim,'CosmicGrain CG10','(b)')):
        im=ax.imshow(arr,origin='lower',extent=extent,cmap='viridis',
                     vmin=a.paper_rv_min,vmax=a.paper_rv_max,
                     interpolation='nearest')
        ax.set_title(f'{panel} {title}')
        ax.set_xlabel(r'$x$ (pkpc)'); ax.set_ylabel(r'$y$ (pkpc)')
        ax.set_aspect('equal'); image_axes.append(ax)
    cb=fig.colorbar(im,ax=image_axes,orientation='vertical',fraction=0.038,pad=0.025)
    cb.set_label(r'Projected $R(55)$')

    bins=np.linspace(a.paper_rv_min,a.paper_rv_max,35)
    colors={'Zhang & Green':'#3569a8','CosmicGrain CG10':'#d3542f'}
    for label,arr in (('Zhang & Green',zg),('CosmicGrain CG10',sim)):
        values=arr[np.isfinite(arr)]
        ax_pdf.hist(values,bins=bins,density=True,histtype='step',lw=2.2,
                    color=colors[label],label=f'{label} (N={values.size})')
        ax_pdf.axvline(np.median(values),color=colors[label],lw=1.4,ls='--')
    ax_pdf.set_title(r'(c) Matched projected $R(55)$ distributions')
    ax_pdf.set_xlabel(r'Projected $R(55)$'); ax_pdf.set_ylabel('Probability density')
    ax_pdf.set_xlim(a.paper_rv_min,a.paper_rv_max)
    ax_pdf.legend(frameon=False); ax_pdf.grid(alpha=.2)

    if power_ok:
        for label,color in (('ZG',colors['Zhang & Green']),
                            ('CG10',colors['CosmicGrain CG10'])):
            selected=[r for r in rows if r[0]==label and r[1]=='uppercut' and r[2]=='RV']
            if len(selected)!=1:
                raise RuntimeError(f'Expected one upper-cut RV spectrum for {label}')
            _,_,_,power,nm,_,nvalid=selected[0]
            scale=2*np.pi/kc
            good=np.isfinite(power)&(power>0)&(nm>=8)
            order=np.argsort(scale[good])
            ax_ps.loglog(scale[good][order],power[good][order],marker='o',ms=4,
                         lw=2,color=color,
                         label=('Zhang & Green' if label=='ZG' else 'CosmicGrain CG10'))
        positive_scales=(2*np.pi/kc)[np.isfinite(kc)&(kc>0)]
        ax_ps.axvspan(max(np.min(positive_scales)*.8,1e-6),2*a.pixel,
                      color='0.85',alpha=.7,label='Below 2 pixels')
        ax_ps.set_title(r'(d) Matched $R(55)$ pseudo-power')
        ax_ps.set_xlabel('Spatial scale (pkpc)')
        ax_ps.set_ylabel(r'$P_{R(55)}$ (pkpc$^2$)')
        ax_ps.grid(alpha=.2); ax_ps.legend(frameon=False)
    else:
        ax_ps.imshow(mask,origin='lower',extent=extent,cmap='Greys',interpolation='nearest')
        ax_ps.set_title('(d) Common-pixel mask; power withheld')
        ax_ps.set_xlabel(r'$x$ (pkpc)'); ax_ps.set_ylabel(r'$y$ (pkpc)')
        ax_ps.set_aspect('equal')
        reason='; '.join(power_reasons)
        ax_ps.text(.5,.04,
                   f"N={mask_stats['pixels']}; components={mask_stats['components']}; "
                   f"largest={mask_stats['largest_component_pixels']}\n{reason}",
                   transform=ax_ps.transAxes,ha='center',va='bottom',fontsize=9,
                   bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))

    fig.text(.5,.012,
             rf'Common {1000*a.pixel:g} pc grid; $|z|<{a.zmax:g}$ pkpc; '
             rf'${a.minimum_delta_av:g}\leq\Delta A_V<{highcut:.3g}$ mag per {a.dz:g} pkpc '
             rf'(Zhang--Green P{100*a.high_cut_quantile:g})',
             ha='center',va='bottom',fontsize=10)
    fig.subplots_adjust(left=.08,right=.92,bottom=.09,top=.95)
    png=f'{prefix}_paper_figure6.png'; pdf=f'{prefix}_paper_figure6.pdf'
    fig.savefig(png,dpi=240,bbox_inches='tight')
    fig.savefig(pdf,bbox_inches='tight')
    plt.close(fig)
    return png,pdf

def main():
    a=arguments()
    apply_cosmicgrain_style(a.style_file)
    print(f'zhang_green_vs_cosmicgrain_3d_to_ps.py version: {SCRIPT_VERSION}')
    print(f'Script location: {Path(__file__).resolve()}')
    print('Extinction-law convention: native monochromatic R(55) in both datasets')
    if not a.zg_h5.is_file() and not a.zg_h5.is_absolute():
        for candidate in (ASSET_DIR/a.zg_h5, Path('scripts')/a.zg_h5,
                          SCRIPT_DIR/'scripts'/a.zg_h5,
                          SCRIPT_DIR.parent/'scripts'/a.zg_h5):
            if candidate.is_file():
                a.zg_h5 = candidate
                break
    if not a.zg_h5.is_file():
        raise FileNotFoundError(f'Zhang--Green input not found: {a.zg_h5}')
    # Both `cd ~/gadget4` and `cd ~/gadget4/scripts` are common entry points.
    # Resolve repository-relative paths without changing the ZG file argument.
    a.root = a.root.resolve()
    if a.output_dir is None and not (a.root/'halo441_output_1024').is_dir():
        for candidate in (a.root.parent, ASSET_DIR.parent, SCRIPT_DIR.parent):
            if (candidate/'halo441_output_1024').is_dir():
                a.root = candidate
                break
    for key in ('carbon_table','silicate_table'):
        path=getattr(a,key)
        if not path.is_absolute() and not path.exists():
            for candidate in (ASSET_DIR/path.name, ASSET_DIR/path, a.root/path):
                if candidate.is_file():
                    setattr(a,key,candidate)
                    break
        if not getattr(a,key).is_file():
            raise FileNotFoundError(f'Optical constants not found: {path}; expected under {ASSET_DIR}/dust_optics')
    if not 0<a.high_cut_quantile<=1:
        raise ValueError('high-cut-quantile must be in (0,1]')
    if not np.isfinite(a.minimum_delta_av) or a.minimum_delta_av < 0:
        raise ValueError('minimum-delta-av must be finite and non-negative')
    sweep_floors=parse_floor_sweep(a.av_floor_sweep)
    e,ze,xx,yy,zc=grid(a)
    gz,gr=zg_cube(a,xx,yy,zc)
    sz,sr,scount,meta=sim_cube(a,e,ze)
    footprint=np.hypot(xx,yy)<=a.radius
    gvalid=np.isfinite(gz)&np.isfinite(gr)&(gz>0)&(gr>0)&footprint[None]
    svalid=np.isfinite(sz)&np.isfinite(sr)&(sz>0)&(sr>0)&footprint[None]
    if gvalid.sum()<20: raise RuntimeError('Too few valid ZG cells for cut')
    if svalid.sum()<20: raise RuntimeError('Too few valid simulation cells; inspect halo, optics and smoothing')

    # Item 2: define one simple upper A_V cut from the ZG positive-cell
    # distribution. P95 is an operational proxy for the high-extinction edge,
    # not a measured Gaia-XP completeness boundary.
    zg_positive=gz[gvalid]
    highcut=float(np.quantile(zg_positive,a.high_cut_quantile))
    if a.minimum_delta_av >= highcut:
        raise ValueError(f'minimum-delta-av={a.minimum_delta_av:g} must be below the ZG upper cut {highcut:g}')
    sweep_floors=[x for x in sweep_floors if x < highcut]
    if a.minimum_delta_av not in sweep_floors:
        sweep_floors=sorted(set(sweep_floors+[a.minimum_delta_av]))
    print(f'ZG upper-A_V cut P{100*a.high_cut_quantile:g}: {highcut:.6g} mag per {a.dz:g} kpc')
    print(f'Matched lower Delta A_V floor: {a.minimum_delta_av:.6g} mag per {a.dz:g} kpc')
    print(f'ZG positive A_V / R(55)-valid cells = {gvalid.sum()}')
    print(f'3D valid cells: ZG={gvalid.sum()}, sim={svalid.sum()}')
    pre=Path(a.output_prefix); pre.parent.mkdir(parents=True,exist_ok=True)

    # Item 1: 3D histograms.
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    bins=np.geomspace(max(min(np.min(gz[gvalid]),np.min(sz[svalid]))*.8,1e-9),
                      max(np.max(gz[gvalid]),np.max(sz[svalid]))*1.2,36)
    for name,av,rv,valid,color in [('ZG',gz,gr,gvalid,'#276b9b'),('CG10',sz,sr,svalid,'#c35c39')]:
        axs[0].hist(av[valid],bins=bins,histtype='step',density=True,label=name,color=color)
        axs[1].hist(rv[valid],bins=np.linspace(1,10,46),histtype='step',density=True,label=name,color=color)
    axs[0].axvline(highcut,color='k',ls='--',
                   label=f'ZG P{100*a.high_cut_quantile:g} upper cut = {highcut:.3f} mag')
    if a.minimum_delta_av > 0:
        axs[0].axvline(a.minimum_delta_av,color='0.35',ls=':',
                       label=rf'Matched lower floor = {a.minimum_delta_av:g} mag')
    axs[0].set(xscale='log',yscale='log',xlabel=rf'3D layer $\Delta A_V$ (mag per {a.dz:g} kpc)',
               ylabel='Probability density')
    axs[1].set(xlabel=r'3D $R(55)$',ylabel='Probability density')
    for ax in axs: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(f'{pre}_01_3d_histograms.png',dpi=200); plt.close(fig)

    # Items 3-4: no cut and the selected matched A_V window.  The historical
    # key name "uppercut" is retained to preserve output compatibility.
    regimes={
        'all': lambda av,valid: valid,
        'uppercut': lambda av,valid: valid&(av>=a.minimum_delta_av)&(av<=highcut),
    }
    maps={}
    for label,av,rv,valid in [('ZG',gz,gr,gvalid),('CG10',sz,sr,svalid)]:
        for regime,selector in regimes.items():
            sel=selector(av,valid)
            av2,rv2,have=project(av,rv,sel)
            maps[f'{label}_{regime}_av']=av2
            maps[f'{label}_{regime}_rv']=rv2
            maps[f'{label}_{regime}_av_mask']=footprint&have&np.isfinite(av2)&(av2>0)
            maps[f'{label}_{regime}_rv_mask']=footprint&have&np.isfinite(rv2)

    # The paper comparison is explicitly limited to the displayed physical
    # R(55) interval. Counts printed below therefore match the plotted sample.
    for label in ('ZG','CG10'):
        key=f'{label}_uppercut_rv'
        maps[f'{label}_uppercut_rv_mask'] &= ((maps[key]>=a.paper_rv_min) &
                                              (maps[key]<=a.paper_rv_max))

    common={}
    for regime in regimes:
        for field in ('av','rv'):
            m=maps[f'ZG_{regime}_{field}_mask']&maps[f'CG10_{regime}_{field}_mask']
            common[(regime,field)]=m
            print(f'Common {regime} {field.upper()} footprint: {m.sum():,} pixels')

    attrition={
        'footprint_pixels':int(footprint.sum()),
        'ZG_3d_positive_Av_and_RV_cells':int(gvalid.sum()),
        'CG10_3d_positive_Av_and_RV_cells':int(svalid.sum()),
        'ZG_all_projected_RV_pixels':int(maps['ZG_all_rv_mask'].sum()),
        'CG10_all_projected_RV_pixels':int(maps['CG10_all_rv_mask'].sum()),
        'common_all_projected_RV_pixels':int(common[('all','rv')].sum()),
        'ZG_uppercut_projected_RV_pixels':int(maps['ZG_uppercut_rv_mask'].sum()),
        'CG10_uppercut_projected_RV_pixels':int(maps['CG10_uppercut_rv_mask'].sum()),
        'common_uppercut_paper_range_RV_pixels':int(common[('uppercut','rv')].sum()),
    }

    paper_mask=common[('uppercut','rv')]
    mask_stats=mask_connectivity(paper_mask)
    power_ok,power_reasons=power_decision(a,mask_stats)
    projected_counts=np.sum(np.where(svalid&(sz>=a.minimum_delta_av)&(sz<=highcut),scount,0),axis=0)
    count_values=projected_counts[paper_mask]
    count_summary={
        'minimum':int(np.min(count_values)) if count_values.size else 0,
        'p16':float(np.percentile(count_values,16)) if count_values.size else 0.0,
        'median':float(np.median(count_values)) if count_values.size else 0.0,
        'p84':float(np.percentile(count_values,84)) if count_values.size else 0.0,
        'maximum':int(np.max(count_values)) if count_values.size else 0,
        'pixels_below_3':int(np.sum(count_values<3)),
        'pixels_below_5':int(np.sum(count_values<5)),
    }
    print('\nPAPER R(55) MASK AUDIT')
    print(json.dumps({'attrition':attrition,'connectivity':mask_stats,'CG10_projected_dust_counts':count_summary,
                      'power_accepted':power_ok,'power_rejection_reasons':power_reasons},indent=2))

    # Save projected maps for visual inspection.
    for regime in regimes:
        fig,ax=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
        for i,label in enumerate(('ZG','CG10')):
            for j,(field,cmap) in enumerate((('av','magma'),('rv','viridis'))):
                arr=maps[f'{label}_{regime}_{field}'].copy()
                arr[~maps[f'{label}_{regime}_{field}_mask']]=np.nan
                im=ax[i,j].imshow(arr,origin='lower',
                    extent=[-a.half_size,a.half_size,-a.half_size,a.half_size],cmap=cmap)
                ax[i,j].set(title=f'{label} {field.upper()} ({regime})',
                            xlabel='x (kpc)',ylabel='y (kpc)')
                fig.colorbar(im,ax=ax[i,j],label='mag' if field=='av' else 'R(55)')
        fig.savefig(f'{pre}_02_{regime}_maps.png',dpi=200); plt.close(fig)

    # Identical k bins and common masks for ZG and CG10.
    kmin=2*np.pi/(2*a.half_size); kmax=np.pi/a.pixel
    edges=np.geomspace(kmin,kmax,a.n_power_bins+1)
    kc=np.sqrt(edges[:-1]*edges[1:])
    rows=[]
    for field in ('av','rv'):
        for regime in regimes:
            m=common[(regime,field)]
            for label in ('ZG','CG10'):
                p,nm,area=spectrum(maps[f'{label}_{regime}_{field}'],m,a.pixel,a.apodize,edges)
                rows.append((label,regime,field.upper(),p,nm,area,int(m.sum())))

    # Main answer to #4: one simple raw pseudo-power panel per field.
    # No overall title and no k^2 P(k) companion panel.
    selected_desc=('matched A_V window' if a.minimum_delta_av>0 else 'upper cut')
    styles={'all':('--','no cut'),'uppercut':('-',selected_desc)}
    for field in ('AV','RV'):
        fig,ax=plt.subplots(figsize=(7.0,5.2))
        for label,regime,name,p,nm,area,nvalid in rows:
            if name!=field: continue
            q=np.isfinite(p)&(p>0)&(nm>=8)
            ls,desc=styles[regime]
            ax.loglog(kc[q],p[q],ls,marker='o',ms=4,
                      label=f'{label} {desc} (N={nvalid})')
        ax.set(xlabel=r'$k$ (kpc$^{-1}$)',
               ylabel=rf'$P_{{{field}}}(k)$')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{pre}_03_{field.lower()}_power.png',
                    dpi=200,bbox_inches='tight')
        plt.close(fig)

    for label,regime,name,p,nm,area,nvalid in rows:
        np.savetxt(f'{pre}_{label.lower()}_{regime}_{name.lower()}_power.txt',
                   np.column_stack((kc,2*np.pi/kc,p,kc**2*p,nm)),
                   header='k_kpc-1 wavelength_kpc P_field2_kpc2 k2P n_modes')

    # Structure functions are the primary spatial diagnostic for the irregular
    # common mask.  They compare only actual pixel pairs and require no mask
    # filling or Fourier-window correction.
    sf_max=np.sqrt(2)*(2*a.half_size-a.pixel)
    sf_edges=np.geomspace(a.pixel,sf_max*(1+1e-10),a.n_power_bins+1)
    sf_sep=np.sqrt(sf_edges[:-1]*sf_edges[1:])
    sf_rows=[]
    for field in ('av','rv'):
        m=common[('uppercut',field)]
        for label in ('ZG','CG10'):
            values=maps[f'{label}_uppercut_{field}']
            sf,npairs=structure_function(values,m,a.pixel,sf_edges)
            variance=float(np.nanvar(values[m],ddof=1)) if m.sum()>1 else np.nan
            normalized=np.divide(sf,2*variance,
                                 out=np.full_like(sf,np.nan),
                                 where=np.isfinite(variance)&(variance>0))
            sf_rows.append((label,field.upper(),sf,normalized,npairs,
                            int(m.sum()),variance))
    # Keep the paper-facing plot deliberately simple. Variance-normalized
    # values remain in the CSV for the heavy-tail robustness diagnostic.
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.6),constrained_layout=True)
    colors={'ZG':'#3569a8','CG10':'#d3542f'}
    for col,field in enumerate(('AV','R55')):
        stored_field='RV' if field=='R55' else field
        for label,name,sf,normalized,npairs,nvalid,variance in sf_rows:
            if name != stored_field: continue
            q=np.isfinite(sf)&(sf>0)&(npairs>=10)
            axs[col].loglog(sf_sep[q],sf[q],marker='o',lw=2,
                            color=colors[label],label=f'{label} (N={nvalid})')
        symbol=r'A_V' if field=='AV' else r'R(55)'
        axs[col].set_title(rf'${symbol}$ structure function')
        axs[col].set_ylabel(r'$S_2(r)$')
        axs[col].set_xlabel('Separation (pkpc)')
        axs[col].grid(alpha=.2)
        axs[col].legend(frameon=False)
    fig.suptitle('Matched-mask second-order structure functions')
    fig.savefig(f'{pre}_04_structure_functions.png',dpi=220,bbox_inches='tight')
    fig.savefig(f'{pre}_04_structure_functions.pdf',bbox_inches='tight')
    plt.close(fig)
    with Path(f'{pre}_structure_functions.csv').open('w',newline='') as fh:
        writer=csv.writer(fh)
        writer.writerow(['sample','field','separation_pkpc','S2',
                         'S2_over_2variance','field_variance','n_pairs',
                         'common_pixels'])
        for label,field,sf,normalized,npairs,nvalid,variance in sf_rows:
            public_field='R55' if field=='RV' else field
            for sep,value,norm,npair in zip(sf_sep,sf,normalized,npairs):
                writer.writerow([label,public_field,sep,value,norm,variance,
                                 int(npair),nvalid])

    # Lower-floor sensitivity audit.  All floors use the same absolute units,
    # upper cut, R(55) interval, geometry and projection in both datasets.
    floor_rows=[]; floor_sf_rows=[]; floor_cases={}
    for floor in sweep_floors:
        case=build_floor_case(a,gz,gr,sz,sr,scount,gvalid,svalid,footprint,highcut,floor)
        floor_cases[floor]=case
        mask=case['common_rv']; stats=case['connectivity']
        counts=case['projected_counts'][mask]
        zgq=percentile_summary(case['ZG_rv'][mask])
        cgq=percentile_summary(case['CG10_rv'][mask])
        floor_rows.append({
            'minimum_delta_av_mag':floor,
            'common_av_pixels':int(case['common_av'].sum()),
            'common_rv_pixels':int(mask.sum()),
            'components':stats['components'],
            'largest_component_pixels':stats['largest_component_pixels'],
            'largest_component_fraction':stats['largest_component_fraction'],
            'cg_dust_count_p16':float(np.percentile(counts,16)) if counts.size else np.nan,
            'cg_dust_count_median':float(np.median(counts)) if counts.size else np.nan,
            'cg_dust_count_p84':float(np.percentile(counts,84)) if counts.size else np.nan,
            'zg_rv_p16':zgq['p16'],'zg_rv_median':zgq['median'],'zg_rv_p84':zgq['p84'],
            'cg_rv_p16':cgq['p16'],'cg_rv_median':cgq['median'],'cg_rv_p84':cgq['p84'],
            'power_mask_accepted':power_decision(a,stats)[0],
        })
        for field in ('av','rv'):
            common_mask=case[f'common_{field}']
            for label in ('ZG','CG10'):
                sf,npairs=structure_function(case[f'{label}_{field}'],common_mask,
                                             a.pixel,sf_edges)
                for sep,value,npair in zip(sf_sep,sf,npairs):
                    floor_sf_rows.append({
                        'minimum_delta_av_mag':floor,'sample':label,
                        'field':field.upper(),'separation_pkpc':sep,
                        'S2':value,'n_pairs':int(npair),
                        'common_pixels':int(common_mask.sum())})
    if floor_rows:
        with Path(f'{pre}_av_floor_sweep.csv').open('w',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(floor_rows[0]))
            writer.writeheader(); writer.writerows(floor_rows)
        with Path(f'{pre}_av_floor_sweep_structure_functions.csv').open('w',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(floor_sf_rows[0]))
            writer.writeheader(); writer.writerows(floor_sf_rows)
        x=np.array([row['minimum_delta_av_mag'] for row in floor_rows])
        # The floor sweep is a small set of analysis choices rather than a
        # continuously sampled function.  Categorical positions make the
        # no-floor case and each tested threshold unambiguous (and avoid the
        # visually awkward zero point on a symlog axis).
        xplot=np.arange(x.size,dtype=float)
        def floor_label(value):
            if np.isclose(value,0.0):
                return 'None'
            exponent=int(np.floor(np.log10(value)))
            coefficient=value/10**exponent
            if np.isclose(coefficient,1.0):
                return rf'$10^{{{exponent}}}$'
            return rf'${coefficient:g}\times10^{{{exponent}}}$'
        floor_labels=[floor_label(value) for value in x]
        fig,axs=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
        axs[0].plot(xplot,[r['common_rv_pixels'] for r in floor_rows],'o-',label='Common pixels')
        axs[0].plot(xplot,[r['largest_component_pixels'] for r in floor_rows],'s--',label='Largest component')
        axs[0].set(ylabel='Pixels',xlabel=r'$A_V$ floor, $\Delta A_{V,\min}$ (mag per 200 pc)',
                   title='Usable spatial footprint')
        axs[0].legend(frameon=False)
        median_shifts=[]
        for label,prefix,color in (('ZG','zg','#3569a8'),('CG10','cg','#d3542f')):
            med=np.array([r[f'{prefix}_rv_median'] for r in floor_rows])
            baseline=med[int(np.argmin(np.abs(x)))]
            shift=med-baseline
            median_shifts.extend(shift[np.isfinite(shift)])
            axs[1].plot(xplot,shift,'o-',color=color,label=label)
        axs[1].axhline(0,color='0.45',ls='--',lw=1)
        max_shift=max(np.abs(median_shifts)) if median_shifts else np.nan
        if np.isfinite(max_shift):
            axs[1].text(.04,.95,rf'maximum change $={max_shift:.2f}$',
                        transform=axs[1].transAxes,ha='left',va='top',color='0.3')
        axs[1].set(ylabel=r'Change in median $R(55)$ from no floor',
                   xlabel=r'$A_V$ floor, $\Delta A_{V,\min}$ (mag per 200 pc)',
                   title=r'Median $R(55)$ stability')
        axs[1].legend(frameon=False)
        med=np.array([r['cg_dust_count_median'] for r in floor_rows])
        lo=np.array([r['cg_dust_count_p16'] for r in floor_rows])
        hi=np.array([r['cg_dust_count_p84'] for r in floor_rows])
        axs[2].errorbar(xplot,med,yerr=np.vstack((med-lo,hi-med)),fmt='o-',
                        color='#6a3d9a',capsize=3,label='Median and 16th--84th')
        axs[2].axhline(3,color='0.4',ls='--',lw=1,label='3 particles')
        axs[2].set(ylabel='CG10 dust particles per pixel',
                   xlabel=r'$A_V$ floor, $\Delta A_{V,\min}$ (mag per 200 pc)',
                   title='Sampling of retained pixels')
        axs[2].legend(frameon=False)
        for ax in axs:
            ax.set_xticks(xplot,floor_labels)
            ax.grid(alpha=.2)
        fig.savefig(f'{pre}_05_av_floor_sensitivity.png',dpi=220,bbox_inches='tight')
        plt.close(fig)

        # Direct lower-floor test.  Evaluate the no-lower-floor and adopted-
        # floor projections on the *same* final common footprint so that a
        # changing mask cannot masquerade as a change in the R(55)
        # distribution.  The upper A_V cut is identical in both cases.
        if a.minimum_delta_av > 0:
            no_floor=floor_cases[min(floor_cases,key=lambda value:abs(value))]
            adopted=floor_cases[a.minimum_delta_av]
            fixed_mask=adopted['common_rv']
            fig,ax=plt.subplots(figsize=(7.2,5.2),constrained_layout=True)
            cdf_rows=[]
            for label,color in (('ZG','#3569a8'),('CG10','#d3542f')):
                for case,selection,linestyle in (
                        (no_floor,'No lower floor','--'),
                        (adopted,rf'$\Delta A_V\geq {a.minimum_delta_av:g}$ mag','-')):
                    values=np.asarray(case[f'{label}_rv'][fixed_mask],float)
                    values=np.sort(values[np.isfinite(values)])
                    if not values.size:
                        continue
                    cumulative=np.arange(1,values.size+1,dtype=float)/values.size
                    median=float(np.median(values))
                    ax.step(values,cumulative,where='post',color=color,ls=linestyle,
                            lw=2.2,label=f'{label}: {selection}')
                    ax.plot(median,.5,'o',ms=6,color=color,
                            markerfacecolor=('white' if linestyle=='--' else color),
                            markeredgewidth=1.4,zorder=5)
                    for rv_value,cdf_value in zip(values,cumulative):
                        cdf_rows.append({
                            'sample':label,'selection':selection,
                            'fixed_common_pixels':int(fixed_mask.sum()),
                            'median_RV':median,'RV':float(rv_value),
                            'empirical_CDF':float(cdf_value)})
            ax.set(xlabel=r'Projected $R(55)$',ylabel='Cumulative fraction',
                   xlim=(a.paper_rv_min,a.paper_rv_max),ylim=(0,1),
                   title=rf'Effect of a {a.minimum_delta_av:g}-mag $A_V$ floor')
            ax.text(.98,.04,
                    rf'Same {int(fixed_mask.sum())} pixels; '
                    rf'$\Delta A_V\leq {highcut:.3g}$ mag in both cases',
                    transform=ax.transAxes,ha='right',va='bottom',color='0.35')
            ax.grid(alpha=.2); ax.legend(frameon=False,loc='upper left')
            fig.savefig(f'{pre}_06_av_floor_rv_cdf.png',dpi=220,bbox_inches='tight')
            fig.savefig(f'{pre}_06_av_floor_rv_cdf.pdf',bbox_inches='tight')
            plt.close(fig)
            with Path(f'{pre}_av_floor_rv_cdf.csv').open('w',newline='') as fh:
                writer=csv.DictWriter(fh,fieldnames=list(cdf_rows[0]))
                writer.writeheader(); writer.writerows(cdf_rows)

    paper_png,paper_pdf=paper_figure(a,maps,common,rows,kc,highcut,pre,
                                     power_ok,power_reasons,mask_stats)

    np.savez_compressed(f'{pre}_products.npz',
        zg_av_3d=gz,zg_rv_3d=gr,sim_av_3d=sz,sim_rv_3d=sr,
        upper_cut_av_layer=highcut,pixel_kpc=a.pixel,dz_kpc=a.dz,
        k_kpc_inv=kc,sim_dust_count_3d=scount,
        sim_uppercut_projected_dust_count=projected_counts,
        paper_common_rv_mask=paper_mask,**maps)

    report={
        'script_version':SCRIPT_VERSION,
        'analysis':'native common Cartesian grid; no added Gaussian smoothing',
        'pixel_kpc':a.pixel,'dz_kpc':a.dz,
        'zg_upper_cut_quantile':a.high_cut_quantile,
        'zg_upper_cut_delta_Av_mag':highcut,
        'extinction_law_convention':{
            'quantity':'R(55)',
            'definition':'A(5500 Angstrom) / [A(4400 Angstrom) - A(5500 Angstrom)]',
            'zg_source':'R55_map_diff',
            'conversion_to_RV_applied':False},
        'matched_lower_delta_Av_mag':a.minimum_delta_av,
        'av_floor_sweep_mag':sweep_floors,
        'radius_kpc':a.radius,'zmax_kpc':a.zmax,
        'cg_patch_radius_pkpc':a.cg_patch_radius,
        'cg_patch_azimuth_deg':a.cg_patch_azimuth,
        'sim_metadata':meta,
        'mask_attrition':attrition,
        'paper_rv_mask':{
            'range':[a.paper_rv_min,a.paper_rv_max],
            'connectivity_8_neighbor':mask_stats,
            'CG10_projected_dust_particle_counts':count_summary,
            'power_accepted':power_ok,
            'power_rejection_reasons':power_reasons,
            'thresholds':{
                'minimum_common_pixels':a.minimum_common_pixels_for_power,
                'minimum_largest_component_pixels':a.minimum_largest_component_pixels,
                'minimum_largest_component_fraction':a.minimum_largest_component_fraction}},
        'selection':{
            f'{label}_{regime}_{name}':{
                'valid_pixels':n,'effective_area_kpc2':area}
            for label,regime,name,p,nm,area,n in rows},
        'caution':'Masked pseudo-spectra; no mask deconvolution, noise or particle shot-noise subtraction. Spatial power is a paper product only when the configured common-mask tests pass.'
    }
    Path(f'{pre}_summary.json').write_text(json.dumps(report,indent=2))
    with Path(f'{pre}_mask_attrition.csv').open('w') as fh:
        fh.write('stage,pixels\n')
        for stage,value in attrition.items():
            fh.write(f'{stage},{value}\n')
    print(json.dumps(report,indent=2))
    print('\\nDONE: items 1-4')
    print(f'  #1 histograms: {pre}_01_3d_histograms.png')
    print(f'  #2 ZG upper cut: {highcut:.6g} mag per {a.dz:g} kpc')
    print(f'  #3 maps: {pre}_02_all_maps.png and {pre}_02_uppercut_maps.png')
    print(f'  #4 PS: {pre}_03_av_power.png and {pre}_03_rv_power.png')
    print(f'  Structure functions: {pre}_04_structure_functions.png,')
    print(f'                       {pre}_04_structure_functions.pdf, and {pre}_structure_functions.csv')
    if floor_rows:
        print(f'  A_V floor audit: {pre}_05_av_floor_sensitivity.png, {pre}_av_floor_sweep.csv,')
        print(f'                   and {pre}_av_floor_sweep_structure_functions.csv')
        if a.minimum_delta_av > 0:
            print(f'  Fixed-mask floor CDF: {pre}_06_av_floor_rv_cdf.png,')
            print(f'                        {pre}_06_av_floor_rv_cdf.pdf, and {pre}_av_floor_rv_cdf.csv')
    print(f'  Paper Figure 6: {paper_png} and {paper_pdf}')
    print(f'  Mask audit: {pre}_mask_attrition.csv and {pre}_summary.json')
    if not power_ok:
        print('  PAPER POWER WITHHELD: '+'; '.join(power_reasons))

if __name__=='__main__':main()
