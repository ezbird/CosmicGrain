#!/usr/bin/env python3
"""Matched Zhang--Green / CosmicGrain differential-extinction comparison.

Requires numpy, scipy, h5py, matplotlib, astropy, astropy-healpix. Run from
~/gadget4 with this file in scripts/ and cosmicgrain_z0_z6_gsd_rv.py beside it.
The ZG input is the original Rv_map_new.h5; the simulation is a z=0 snapshot.

Example:
 python3 scripts/cosmicgrain_zg_av_rv_compare.py Rv_map_new.h5 \
   --root . --output-prefix cg10_zg_baseline_250pc

To analyze the separate MinGasHsml=0.0 run from ~/gadget4 instead:
 python3 scripts/cosmicgrain_zg_av_rv_compare.py Rv_map_new.h5 \
   --output-dir halo441_output_1024_mingashsml0.0 \
   --output-prefix cg10_zg_mingashsml0_250pc

The 3D A_V quantity is the differential extinction across a common dz layer,
not cumulative extinction from the Sun. FFT outputs are pseudo-spectra: masks
couple Fourier modes and no mask deconvolution or dust shot-noise subtraction
is attempted. Resolution-sensitive bins must be excluded from science claims.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
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
SCRIPT_VERSION = '2026-09-28-v10-same-mask-control-p5p10p20'

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
    p.add_argument('--zmax', type=float, default=0.4)
    p.add_argument('--pixel', type=float, default=0.25)
    p.add_argument('--dz', type=float, default=0.2)
    p.add_argument('--sim-sigma', type=float, default=0.5, help='Gaussian smoothing sigma in kpc, before selection')
    p.add_argument('--match-smoothing', action='store_true',
                   help='Apply the same 3D Gaussian width to ZG differential AV and AV-weighted RV')
    p.add_argument('--zg-min-support', type=float, default=0.3,
                   help='Minimum fraction of valid ZG voxels inside Gaussian kernel')
    p.add_argument('--cut-quantile', type=float, default=0.10, help='ZG positive layer A_V quantile, 0..1')
    p.add_argument('--cut-mode', choices=['quantile','completeness'], default='quantile',
                   help='How to choose the applied ZG A_V cut. Diagnostic thresholds are always measured.')
    p.add_argument('--completeness-target', type=float, default=0.90,
                   help='For --cut-mode completeness: fraction of the high-A_V R(V)-valid plateau')
    p.add_argument('--completeness-bins', type=int, default=18,
                   help='Number of logarithmic A_V bins for the ZG completeness diagnostic')
    p.add_argument('--apodize', type=float, default=2.0, help='Taper width in pixels')
    p.add_argument('--n-power-bins', type=int, default=14)
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
    if a.match_smoothing:
        padx=int(np.ceil(4*a.sim_sigma/a.pixel))+1
        padz=int(np.ceil(4*a.sim_sigma/a.dz))+1
        xc=xx[0,:]
        yc=yy[:,0]
        xp=xc[0]+a.pixel*np.arange(-padx,len(xc)+padx)
        yp=yc[0]+a.pixel*np.arange(-padx,len(yc)+padx)
        zp=zc[0]+a.dz*np.arange(-padz,len(zc)+padz)
        qx,qy=np.meshgrid(xp,yp)
    else:
        padx=padz=0
        qx,qy,zp=xx,yy,zc
    with h5py.File(a.zg_h5,'r') as f:
        bins = np.asarray(f['distance_bins'], float)
        mg = f[f'map_{a.nside}']
        ns = int(mg['nside'][()]); order = mg['hp_order'][()]
        if isinstance(order, bytes): order = order.decode()
        hp = HEALPix(nside=ns, order=str(order), frame='galactic')
        av = np.full((len(zp),*qx.shape), np.nan)
        rv = np.full_like(av,np.nan)
        for j,z in enumerate(zp):
            coords = SkyCoord(u=qx.ravel()*u.kpc,v=qy.ravel()*u.kpc,
                              w=np.full(qx.size,z)*u.kpc,frame='galactic',
                              representation_type='cartesian')
            # Galactic Cartesian SkyCoord has no direct .distance attribute in
            # some Astropy versions; its spherical representation does.
            di = np.searchsorted(bins,coords.spherical.distance.to_value(u.kpc))-1
            hi = hp.skycoord_to_healpix(coords)
            aa=np.full(qx.size,np.nan); rr=np.full(qx.size,np.nan)
            for b in np.unique(di[(di>=0)&(di<len(bins)-1)]):
                q=di==b
                aa[q]=np.asarray(mg['AV_map_diff'][int(b),:],float)[hi[q]]*a.dz
                rr[q]=1.1*np.asarray(mg['R55_map_diff'][int(b),:],float)[hi[q]]+0.07
            av[j]=aa.reshape(qx.shape); rv[j]=rr.reshape(qx.shape)
    if a.match_smoothing:
        good=np.isfinite(av)&np.isfinite(rv)&(av>0)&(rv>0)
        sig=(a.sim_sigma/a.dz,a.sim_sigma/a.pixel,a.sim_sigma/a.pixel)
        support=gaussian_filter(good.astype(float),sig,mode='constant')
        avsum=gaussian_filter(np.where(good,av,0),sig,mode='constant')
        ebvsum=gaussian_filter(np.where(good,av/rv,0),sig,mode='constant')
        sl=(slice(padz,-padz),slice(padx,-padx),slice(padx,-padx))
        support=support[sl];avsum=avsum[sl];ebvsum=ebvsum[sl]
        print('ZG Gaussian valid-support percentiles (5, 25, 50, 75, 95, max):',
              np.round(np.percentile(support,[5,25,50,75,95,100]),3))
        print(f'ZG cells above support {a.zg_min_support:.2f}: '
              f'{np.count_nonzero(support>=a.zg_min_support):,}/{support.size:,}')
        avg=np.divide(avsum,support,out=np.full_like(avsum,np.nan),where=support>0)
        rvg=np.divide(avsum,ebvsum,out=np.full_like(avsum,np.nan),where=ebvsum>0)
        avg[support<a.zg_min_support]=np.nan
        rvg[support<a.zg_min_support]=np.nan
        av,rv=avg,rvg
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
    xyz = periodic_delta(dust['coords'],np.asarray(halo['center']),box)*scale/h @ rot.T
    padx=int(np.ceil(4*a.sim_sigma/a.pixel))+1
    padz=int(np.ceil(4*a.sim_sigma/a.dz))+1
    xe=e[0]+a.pixel*np.arange(-padx,len(e)+padx)
    zz=ze[0]+a.dz*np.arange(-padz,len(ze)+padz)
    inside = (np.linalg.norm(xyz,axis=1)<halo['r200_pkpc']) & \
             (np.abs(xyz[:,0])<a.half_size+padx*a.pixel) & \
             (np.abs(xyz[:,1])<a.half_size+padx*a.pixel) & \
             (np.abs(xyz[:,2])<a.zmax+padz*a.dz)
    xyz=xyz[inside]
    radius=dust['radius_nm'][inside]*1e-3 # micron
    f=np.clip(dust['carbon'][inside],0,1)
    mass=dust['mass'][inside]*1e10/h*MSUN
    cb=cg.read_draine_qext_table(a.carbon_table)
    sb=cg.read_draine_qext_table(a.silicate_table)
    # tau = m * [3 Q_ext/(4 rho a)] / pixel area. Mixture is by mass.
    def weight(wave):
        qc=cg.qext_at_wavelength(cb,radius,wave)
        qs=cg.qext_at_wavelength(sb,radius,wave)
        kap=3/(4*radius*1e-4)*(f*qc/cg.RHO_C+(1-f)*qs/cg.RHO_SIL)
        return mass*kap/(a.pixel*KPC_CM)**2
    shape=(len(ze)-1,len(e)-1,len(e)-1)
    bins=(zz,xe,xe)
    pts=xyz[:,[2,1,0]]
    tb=np.histogramdd(pts,bins=bins,weights=weight(cg.B_MICRON))[0]
    tv=np.histogramdd(pts,bins=bins,weights=weight(cg.V_MICRON))[0]
    sig=(a.sim_sigma/a.dz,a.sim_sigma/a.pixel,a.sim_sigma/a.pixel)
    # Smooth on an extended cube; only then take the target slab. Otherwise
    # dust immediately outside |z|<=zmax disappears before convolution.
    tb=gaussian_filter(tb,sig,mode='constant')
    tv=gaussian_filter(tv,sig,mode='constant')
    sl=(slice(padz,-padz),slice(padx,-padx),slice(padx,-padx))
    tb=tb[sl];tv=tv[sl]
    assert tb.shape==shape
    av=1.086*tv
    rv=np.divide(tv,tb-tv,out=np.full_like(tv,np.nan),where=tb>tv)
    return av,rv,{'group':group,'redshift':1/scale-1,'dust_particles_near_box':int(inside.sum()),
                  'orientation':source,'orientation_particles':int(nrot)}

def zg_completeness_diagnostic(av, rv, footprint, nbins=18):
    """Measure ZG R(V)-valid fraction as a function of positive 3-D layer A_V.

    Denominator: queried cells with finite positive A_V inside the aperture.
    Numerator: those same cells with finite positive R(V).
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
    # Extinction adds along the sightline.  For R(V), add color excess
    # E(B-V) = A(V)/R(V), then form R(V)_2D = sum A(V) / sum E(B-V).
    good = select & np.isfinite(av) & np.isfinite(rv) & (av > 0) & (rv > 0)
    av_cells = np.where(good, av, 0.0)
    ebv_cells = np.where(good, av / rv, 0.0)
    avmap = av_cells.sum(axis=0)
    ebvmap = ebv_cells.sum(axis=0)
    rmap = np.divide(avmap, ebvmap, out=np.full_like(avmap, np.nan), where=ebvmap > 0)
    return avmap, rmap, avmap > 0

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

def main():
    a=arguments()
    print(f'zhang_green_vs_cosmicgrain_3d_to_ps.py version: {SCRIPT_VERSION}')
    print(f'Script location: {Path(__file__).resolve()}')
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
    if not 0<=a.cut_quantile<=1 or a.sim_sigma<=0 or not 0<a.zg_min_support<=1:
        raise ValueError('Invalid cut quantile, smoothing width or ZG support threshold')
    e,ze,xx,yy,zc=grid(a)
    gz,gr=zg_cube(a,xx,yy,zc)
    sz,sr,meta=sim_cube(a,e,ze)
    footprint=np.hypot(xx,yy)<=a.radius
    gvalid=np.isfinite(gz)&np.isfinite(gr)&(gz>0)&(gr>0)&footprint[None]
    svalid=np.isfinite(sz)&np.isfinite(sr)&(sz>0)&(sr>0)&footprint[None]
    if gvalid.sum()<20: raise RuntimeError('Too few valid ZG cells for cut')
    if svalid.sum()<20: raise RuntimeError('Too few valid simulation cells; inspect halo, optics and smoothing')

    comp=zg_completeness_diagnostic(gz,gr,footprint,a.completeness_bins)
    qcuts={q:float(np.quantile(gz[comp['rv_ok']],q)) for q in (0.05,0.10,0.20)}
    c80,c90,c95=(comp['thresholds'][q] for q in (0.80,0.90,0.95))
    qcut=float(np.quantile(gz[comp['rv_ok']],a.cut_quantile))
    if a.cut_mode=='completeness':
        # Interpolate among the explicitly measured standard targets only when
        # the requested target is one of them; otherwise measure it directly.
        level=a.completeness_target*comp['plateau']
        occupied=np.flatnonzero(comp['total']>0)
        cut=np.nan
        for i in occupied:
            later=occupied[occupied>=i]
            if comp['fraction'][i]>=level and np.all(comp['fraction'][later]>=level):
                cut=float(comp['edges'][i]); break
        if not np.isfinite(cut):
            raise RuntimeError('No stable ZG completeness threshold found; inspect diagnostic plot')
    else:
        cut=qcut

    print('ZG positive-cell A_V quantiles: '+', '.join(
          f'P{int(q*100)}={v:.6g}' for q,v in qcuts.items())+f' mag per {a.dz:g} kpc')
    print(f"ZG R(V)-valid high-A_V plateau: {comp['plateau']:.3f}")
    print('ZG completeness thresholds (fraction of plateau): '
          f'80%={c80:.6g}, 90%={c90:.6g}, 95%={c95:.6g} mag' if all(np.isfinite(x) for x in (c80,c90,c95))
          else f'80%={c80}, 90%={c90}, 95%={c95} mag')
    print(f'Applied ZG A_V cut [{a.cut_mode}]: {cut:.6g} mag per {a.dz:g} kpc')
    print(f"ZG positive A_V queried cells={comp['av_ok'].sum()}, R(V)-valid={comp['rv_ok'].sum()}")
    print(f'3D valid cells: ZG={gvalid.sum()}, sim={svalid.sum()}')
    pre=Path(a.output_prefix);pre.parent.mkdir(parents=True,exist_ok=True)

    # Completeness diagnostic: counts and R(V)-valid fraction versus layer A_V.
    fig,ax1=plt.subplots(figsize=(7.2,5.4))
    q=comp['total']>0
    ax1.semilogx(comp['centers'][q],comp['fraction'][q],marker='o',label='ZG R(V)-valid fraction')
    ax1.axhline(comp['plateau'],color='0.35',ls='--',label=f"high-$A_V$ plateau = {comp['plateau']:.2f}")
    for target,ls in ((0.80,':'),(0.90,'-.'),(0.95,(0,(3,1,1,1)))):
        th=comp['thresholds'][target]
        if np.isfinite(th): ax1.axvline(th,color='0.45',ls=ls,label=f'{int(target*100)}% plateau: {th:.3g} mag')
    for qv,v in qcuts.items(): ax1.axvline(v,alpha=.45,ls='--',label=f'P{int(qv*100)}: {v:.3g}')
    ax1.set(xlabel=r'3D layer $\Delta A_V$ (mag)',ylabel=r'fraction with valid $R(V)$',ylim=(-0.03,1.05))
    ax1.grid(alpha=.2); ax1.legend(fontsize=8,ncol=2)
    ax2=ax1.twinx(); ax2.step(comp['centers'][q],comp['total'][q],where='mid',color='0.6',alpha=.45)
    ax2.set_ylabel('ZG positive-$A_V$ cells per bin',color='0.4')
    fig.tight_layout(); fig.savefig(f'{pre}_zg_av_completeness.png',dpi=200); plt.close(fig)
    np.savetxt(f'{pre}_zg_av_completeness.txt',
               np.column_stack((comp['edges'][:-1],comp['edges'][1:],comp['centers'],comp['total'],comp['valid'],comp['fraction'])),
               header='Av_lo_mag Av_hi_mag Av_center_mag n_positive_Av n_valid_Rv valid_fraction')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    bins=np.geomspace(max(min(np.min(gz[gvalid]),np.min(sz[svalid]))*.8,1e-9),
                      max(np.max(gz[gvalid]),np.max(sz[svalid]))*1.2,36)
    for name,av,rv,valid,color in [('ZG',gz,gr,gvalid,'#276b9b'),('CG10',sz,sr,svalid,'#c35c39')]:
        axs[0].hist(av[valid],bins=bins,histtype='step',density=True,label=name,color=color)
        axs[1].hist(rv[valid],bins=np.linspace(1,10,46),histtype='step',density=True,label=name,color=color)
    axs[0].axvline(cut,color='k',ls='--',label='ZG threshold')
    axs[0].set(xscale='log',yscale='log',xlabel=r'3D layer $\Delta A_V$ (mag)',ylabel='Probability density')
    axs[1].set(xlabel=r'3D $R(V)$',ylabel='Probability density')
    for ax in axs: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(f'{pre}_3d_histograms.png',dpi=180);plt.close(fig)
    kmin=2*np.pi/(2*a.half_size);kmax=np.pi/a.pixel
    edges=np.geomspace(kmin,kmax,a.n_power_bins+1);kc=np.sqrt(edges[:-1]*edges[1:])
    if not np.any(kc<1/a.sim_sigma):
        print('WARNING: no plotted k bins below 1/sigma; enlarge --half-size or reduce --sim-sigma')
    rows=[]; maps={}
    # Build natural all/cut projections first.
    for label,av,rv,valid in [('ZG',gz,gr,gvalid),('CG10',sz,sr,svalid)]:
        for regime,mask in [('all',valid),('cut',valid&(av>=cut))]:
            vm,rm,have=project(av,rv,mask)
            coverage=np.any(valid,axis=0)
            amask=footprint&coverage
            rmask=footprint&have&np.isfinite(rm)
            maps[f'{label}_{regime}_av']=vm
            maps[f'{label}_{regime}_rv']=rm
            maps[f'{label}_{regime}_av_mask']=amask
            maps[f'{label}_{regime}_rv_mask']=rmask

    # Common natural masks for the two datasets.
    common_masks={}
    for regime in ('all','cut'):
        for name in ('av','rv'):
            common=maps[f'ZG_{regime}_{name}_mask'] & maps[f'CG10_{regime}_{name}_mask']
            common_masks[(regime,name)]=common
            print(f'Common {regime} {name.upper()} footprint: {common.sum():,} pixels')

    # Main spectra: all, cut, and the crucial mask control.  The mask-control
    # uses the uncut field but exactly the cut Fourier window, isolating the
    # effect of changing the projected footprint from the effect of removing
    # low-A_V 3-D material.
    for name in ('av','rv'):
        for regime in ('all','cut','all_on_cut_mask'):
            common = common_masks[('cut' if regime=='all_on_cut_mask' else regime,name)]
            source_regime = 'all' if regime=='all_on_cut_mask' else regime
            for label in ('ZG','CG10'):
                field=maps[f'{label}_{source_regime}_{name}']
                power,nm,area=spectrum(field,common,a.pixel,a.apodize,edges)
                rows.append((label,regime,name.upper(),power,nm,area,int(common.sum())))

    print('Usable plotted k bins below 1/sigma:',
          int(np.sum((kc<1/a.sim_sigma)&(rows[0][4]>=8))))

    # P5/P10/P20 selection sensitivity.  These are saved numerically and in a
    # compact diagnostic figure, but kept out of the main comparison figure.
    sensitivity=[]
    for qv,qcutv in qcuts.items():
        tmp={}
        for label,av,rv,valid in [('ZG',gz,gr,gvalid),('CG10',sz,sr,svalid)]:
            vm,rm,have=project(av,rv,valid&(av>=qcutv))
            coverage=np.any(valid,axis=0)
            tmp[f'{label}_av']=vm; tmp[f'{label}_rv']=rm
            tmp[f'{label}_av_mask']=footprint&coverage
            tmp[f'{label}_rv_mask']=footprint&have&np.isfinite(rm)
        for name in ('av','rv'):
            common=tmp[f'ZG_{name}_mask']&tmp[f'CG10_{name}_mask']
            for label in ('ZG','CG10'):
                power,nm,area=spectrum(tmp[f'{label}_{name}'],common,a.pixel,a.apodize,edges)
                sensitivity.append((qv,qcutv,label,name.upper(),power,nm,area,int(common.sum())))
        print(f'P{int(qv*100)} sensitivity cut={qcutv:.6g} mag: '
              f"AV common={int((tmp['ZG_av_mask']&tmp['CG10_av_mask']).sum())}, "
              f"RV common={int((tmp['ZG_rv_mask']&tmp['CG10_rv_mask']).sum())}")

    for regime in ('all','cut'):
        fig,ax=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
        for i,label in enumerate(('ZG','CG10')):
            for j,(name,cmap) in enumerate((('av','magma'),('rv','viridis'))):
                arr=maps[f'{label}_{regime}_{name}'].copy()
                arr[~maps[f'{label}_{regime}_{name}_mask']]=np.nan
                im=ax[i,j].imshow(arr,origin='lower',extent=[-a.half_size,a.half_size]*2,cmap=cmap)
                ax[i,j].set(title=f'{label} {name.upper()} ({regime})',xlabel='x (kpc)',ylabel='y (kpc)')
                fig.colorbar(im,ax=ax[i,j],label='mag' if name=='av' else 'R(V)')
        fig.savefig(f'{pre}_{regime}_maps.png',dpi=180);plt.close(fig)

    np.savez_compressed(f'{pre}_products.npz',zg_av_3d=gz,zg_rv_3d=gr,
       sim_av_3d=sz,sim_rv_3d=sr,cut_av_layer=cut,pixel_kpc=a.pixel,dz_kpc=a.dz,
       k_kpc_inv=kc,**maps)

    # Main PS figures. Solid=cut, dashed=all, dotted=uncut field evaluated on
    # the cut mask.  Thus cut-vs-dotted isolates the 3-D selection effect.
    styles={'all':('--','all'), 'cut':('-','P10 cut'),
            'all_on_cut_mask':(':','all on P10 mask')}
    for field in ('AV','RV'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
        for label,regime,name,p,nm,area,nvalid in rows:
            if field!=name:continue
            q=np.isfinite(p)&(p>0)&(nm>=8)
            ls,desc=styles[regime]
            for ax,y in zip(axes,(p,kc**2*p)):
                ax.loglog(kc[q],y[q],ls,marker='o',ms=3,
                          label=f'{label} {desc} (N={nvalid})')
        for ax,title,ylabel in zip(axes,('Raw pseudo-power','Power per log spatial scale'),
                                   (rf'$P_{{{field}}}(k)$',rf'$k^2P_{{{field}}}(k)$')):
            ax.set(xlabel=r'$k$ (kpc$^{-1}$)',ylabel=ylabel,title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{field}: ZG vs CG10; '+('matched 0.5-kpc smoothing' if a.match_smoothing
                                               else 'native ZG / smoothed CG10'))
        fig.tight_layout()
        fig.savefig(f'{pre}_{field.lower()}_power.png',dpi=200);plt.close(fig)

    # P5/P10/P20 sensitivity figure: cut spectra only, common mask at each cut.
    for field in ('AV','RV'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
        for qv,qcutv,label,name,p,nm,area,nvalid in sensitivity:
            if field!=name: continue
            good=np.isfinite(p)&(p>0)&(nm>=8)
            for ax,y in zip(axes,(p,kc**2*p)):
                ax.loglog(kc[good],y[good],marker='o',ms=3,
                          label=f'{label} P{int(qv*100)} (N={nvalid})')
        for ax,title,ylabel in zip(axes,('Raw pseudo-power','Power per log spatial scale'),
                                   (rf'$P_{{{field}}}(k)$',rf'$k^2P_{{{field}}}(k)$')):
            ax.set(xlabel=r'$k$ (kpc$^{-1}$)',ylabel=ylabel,title=title); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7,ncol=2)
        fig.suptitle(f'{field}: P5/P10/P20 ZG-cut sensitivity')
        fig.tight_layout(); fig.savefig(f'{pre}_{field.lower()}_cut_sensitivity.png',dpi=200); plt.close(fig)

    for label,regime,name,p,nm,area,nvalid in rows:
        np.savetxt(f'{pre}_{label.lower()}_{regime}_{name.lower()}_power.txt',
                   np.column_stack((kc,2*np.pi/kc,p,kc**2*p,nm)),
                   header='k_kpc-1 wavelength_kpc P_field2_kpc2 k2P n_modes')
    for qv,qcutv,label,name,p,nm,area,nvalid in sensitivity:
        np.savetxt(f'{pre}_p{int(qv*100):02d}_{label.lower()}_{name.lower()}_power.txt',
                   np.column_stack((kc,2*np.pi/kc,p,kc**2*p,nm)),
                   header=f'ZG_cut_quantile={qv} cut_Av_mag={qcutv} valid_pixels={nvalid}\n'
                          'k_kpc-1 wavelength_kpc P_field2_kpc2 k2P n_modes')
    report={'script_version':SCRIPT_VERSION,
            'zg_cut_mode':a.cut_mode,'zg_cut_quantile':a.cut_quantile,'cut_delta_Av_mag':cut,'dz_kpc':a.dz,
            'zg_av_quantiles_mag':{f'P{int(q*100)}':v for q,v in qcuts.items()},
            'zg_rv_valid_plateau':comp['plateau'],
            'zg_completeness_thresholds_mag':{f'{int(q*100)}pct_plateau':comp['thresholds'][q] for q in (0.80,0.90,0.95)},
            'pixel_kpc':a.pixel,'sigma_sim_kpc':a.sim_sigma,
            'matched_added_gaussian':a.match_smoothing,'zg_min_support':a.zg_min_support,
            'n_k_bins_below_1_over_sigma':int(np.sum(kc<1/a.sim_sigma)),
            'radius_kpc':a.radius,
            'zmax_kpc':a.zmax,'sim_metadata':meta,
            'selection':{f'{label}_{regime}_{name}':{'valid_pixels':n,'effective_area_kpc2':area}
                         for label,regime,name,p,nm,area,n in rows},
             'cut_sensitivity':{f'P{int(qv*100)}_{label}_{name}':{'cut_Av_mag':qcutv,'valid_pixels':n,'effective_area_kpc2':area}
                                for qv,qcutv,label,name,p,nm,area,n in sensitivity},
            'caution':'Masked pseudo-spectra; no mask deconvolution, noise or particle shot-noise subtraction.'}
    Path(f'{pre}_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(f'Wrote {pre}_zg_av_completeness.png, {pre}_3d_histograms.png, main/mask-control PS, P5/P10/P20 sensitivity PS, and numerical products')

if __name__=='__main__':main()
