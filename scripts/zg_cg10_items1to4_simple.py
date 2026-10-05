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
from scipy.ndimage import distance_transform_edt
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
SCRIPT_VERSION = '2026-09-29-v13-simple-items1-4-native-grid'

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
    p.add_argument('--high-cut-quantile', type=float, default=0.95,
                   help='ZG positive layer A_V upper quantile used to test loss of dense/high-extinction cells')
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
                rr[q]=1.1*np.asarray(mg['R55_map_diff'][int(ib),:],float)[hi[q]]+0.07
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
    xyz = periodic_delta(dust['coords'],np.asarray(halo['center']),box)*scale/h @ rot.T
    xe=e
    zz=ze
    inside = (np.linalg.norm(xyz,axis=1)<halo['r200_pkpc']) & \
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
    if not 0<a.high_cut_quantile<=1:
        raise ValueError('high-cut-quantile must be in (0,1]')
    e,ze,xx,yy,zc=grid(a)
    gz,gr=zg_cube(a,xx,yy,zc)
    sz,sr,meta=sim_cube(a,e,ze)
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
    print(f'ZG upper-A_V cut P{100*a.high_cut_quantile:g}: {highcut:.6g} mag per {a.dz:g} kpc')
    print(f'ZG positive A_V / R(V)-valid cells = {gvalid.sum()}')
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
    axs[0].set(xscale='log',yscale='log',xlabel=rf'3D layer $\Delta A_V$ (mag per {a.dz:g} kpc)',
               ylabel='Probability density')
    axs[1].set(xlabel=r'3D $R(V)$',ylabel='Probability density')
    for ax in axs: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(f'{pre}_01_3d_histograms.png',dpi=200); plt.close(fig)

    # Items 3-4: only two cases -- no cut and the same ZG-derived upper cut.
    regimes={
        'all': lambda av,valid: valid,
        'uppercut': lambda av,valid: valid&(av<=highcut),
    }
    maps={}
    for label,av,rv,valid in [('ZG',gz,gr,gvalid),('CG10',sz,sr,svalid)]:
        for regime,selector in regimes.items():
            sel=selector(av,valid)
            av2,rv2,have=project(av,rv,sel)
            coverage=np.any(valid,axis=0)
            maps[f'{label}_{regime}_av']=av2
            maps[f'{label}_{regime}_rv']=rv2
            maps[f'{label}_{regime}_av_mask']=footprint&coverage
            maps[f'{label}_{regime}_rv_mask']=footprint&have&np.isfinite(rv2)

    common={}
    for regime in regimes:
        for field in ('av','rv'):
            m=maps[f'ZG_{regime}_{field}_mask']&maps[f'CG10_{regime}_{field}_mask']
            common[(regime,field)]=m
            print(f'Common {regime} {field.upper()} footprint: {m.sum():,} pixels')

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
                fig.colorbar(im,ax=ax[i,j],label='mag' if field=='av' else 'R(V)')
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

    # Main answer to #4: ZG and CG10 together, with and without the upper cut.
    styles={'all':('--','no cut'),'uppercut':('-','upper cut')}
    for field in ('AV','RV'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
        for label,regime,name,p,nm,area,nvalid in rows:
            if name!=field: continue
            q=np.isfinite(p)&(p>0)&(nm>=8)
            ls,desc=styles[regime]
            for ax,y in zip(axes,(p,kc**2*p)):
                ax.loglog(kc[q],y[q],ls,marker='o',ms=3,
                          label=f'{label} {desc} (N={nvalid})')
        for ax,title,ylabel in zip(axes,('Raw pseudo-power','Power per log spatial scale'),
                                   (rf'$P_{{{field}}}(k)$',rf'$k^2P_{{{field}}}(k)$')):
            ax.set(xlabel=r'$k$ (kpc$^{-1}$)',ylabel=ylabel,title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{field}: ZG vs CG10; native 250-pc grid, no added Gaussian smoothing')
        fig.tight_layout()
        fig.savefig(f'{pre}_03_{field.lower()}_power.png',dpi=200); plt.close(fig)

    for label,regime,name,p,nm,area,nvalid in rows:
        np.savetxt(f'{pre}_{label.lower()}_{regime}_{name.lower()}_power.txt',
                   np.column_stack((kc,2*np.pi/kc,p,kc**2*p,nm)),
                   header='k_kpc-1 wavelength_kpc P_field2_kpc2 k2P n_modes')

    np.savez_compressed(f'{pre}_products.npz',
        zg_av_3d=gz,zg_rv_3d=gr,sim_av_3d=sz,sim_rv_3d=sr,
        upper_cut_av_layer=highcut,pixel_kpc=a.pixel,dz_kpc=a.dz,
        k_kpc_inv=kc,**maps)

    report={
        'script_version':SCRIPT_VERSION,
        'analysis':'native common Cartesian grid; no added Gaussian smoothing',
        'pixel_kpc':a.pixel,'dz_kpc':a.dz,
        'zg_upper_cut_quantile':a.high_cut_quantile,
        'zg_upper_cut_delta_Av_mag':highcut,
        'radius_kpc':a.radius,'zmax_kpc':a.zmax,
        'sim_metadata':meta,
        'selection':{
            f'{label}_{regime}_{name}':{
                'valid_pixels':n,'effective_area_kpc2':area}
            for label,regime,name,p,nm,area,n in rows},
        'caution':'Masked pseudo-spectra; no mask deconvolution, noise or particle shot-noise subtraction.'
    }
    Path(f'{pre}_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('\\nDONE: items 1-4')
    print(f'  #1 histograms: {pre}_01_3d_histograms.png')
    print(f'  #2 ZG upper cut: {highcut:.6g} mag per {a.dz:g} kpc')
    print(f'  #3 maps: {pre}_02_all_maps.png and {pre}_02_uppercut_maps.png')
    print(f'  #4 PS: {pre}_03_av_power.png and {pre}_03_rv_power.png')

if __name__=='__main__':main()
