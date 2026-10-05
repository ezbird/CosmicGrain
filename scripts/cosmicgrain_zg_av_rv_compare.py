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
SCRIPT_VERSION = '2026-09-27-v8-common-mask-support-diagnostics'

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
        rvsum=gaussian_filter(np.where(good,av*rv,0),sig,mode='constant')
        sl=(slice(padz,-padz),slice(padx,-padx),slice(padx,-padx))
        support=support[sl];avsum=avsum[sl];rvsum=rvsum[sl]
        print('ZG Gaussian valid-support percentiles (5, 25, 50, 75, 95, max):',
              np.round(np.percentile(support,[5,25,50,75,95,100]),3))
        print(f'ZG cells above support {a.zg_min_support:.2f}: '
              f'{np.count_nonzero(support>=a.zg_min_support):,}/{support.size:,}')
        avg=np.divide(avsum,support,out=np.full_like(avsum,np.nan),where=support>0)
        rvg=np.divide(rvsum,avsum,out=np.full_like(rvsum,np.nan),where=avsum>0)
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

def project(av,rv,select):
    w=np.where(select,av,0.)
    avmap=w.sum(axis=0)
    rmap=np.divide((w*np.where(np.isfinite(rv),rv,0.)).sum(axis=0),avmap,
                   out=np.full_like(avmap,np.nan),where=avmap>0)
    return avmap,rmap,avmap>0

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
    print(f'cosmicgrain_zg_av_rv_compare.py version: {SCRIPT_VERSION}')
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
    cut=float(np.quantile(gz[gvalid],a.cut_quantile))
    print(f'ZG {a.cut_quantile:.0%} positive-cell differential A_V cut: {cut:.6g} mag per {a.dz:g} kpc')
    print(f'3D valid cells: ZG={gvalid.sum()}, sim={svalid.sum()}')
    pre=Path(a.output_prefix);pre.parent.mkdir(parents=True,exist_ok=True)
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
    for label,av,rv,valid in [('ZG',gz,gr,gvalid),('CG10',sz,sr,svalid)]:
        for regime,mask in [('all',valid),('cut',valid&(av>=cut))]:
            vm,rm,have=project(av,rv,mask)
            # Do not interpret missing ZG voxels as measured A_V=0. Reuse
            # pre-cut coverage for both A_V spectra so their windows match.
            coverage=np.any(valid,axis=0)
            amask=footprint&coverage
            rmask=footprint&have&np.isfinite(rm)
            maps[f'{label}_{regime}_av']=vm
            maps[f'{label}_{regime}_rv']=rm
            maps[f'{label}_{regime}_av_mask']=amask
            maps[f'{label}_{regime}_rv_mask']=rmask
    # Identical Fourier windows within each field and selection. Otherwise
    # the two pseudo-spectra differ partly because their masks differ.
    for regime in ('all','cut'):
        for name in ('av','rv'):
            common=maps[f'ZG_{regime}_{name}_mask'] & maps[f'CG10_{regime}_{name}_mask']
            for label in ('ZG','CG10'):
                field=maps[f'{label}_{regime}_{name}']
                power,nm,area=spectrum(field,common,a.pixel,a.apodize,edges)
                rows.append((label,regime,name.upper(),power,nm,area,int(common.sum())))
            print(f'Common {regime} {name.upper()} footprint: {common.sum():,} pixels')
    print('Usable plotted k bins below 1/sigma:',
          int(np.sum((kc<1/a.sim_sigma)&(rows[0][4]>=8))))
    for regime in ('all','cut'):
        fig,ax=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
        for i,label in enumerate(('ZG','CG10')):
            for j,(name,cmap) in enumerate((('av','magma'),('rv','viridis'))):
                arr=maps[f'{label}_{regime}_{name}'].copy()
                if name=='rv':
                    arr[~maps[f'{label}_{regime}_rv_mask']]=np.nan
                else: arr[~maps[f'{label}_{regime}_av_mask']]=np.nan
                im=ax[i,j].imshow(arr,origin='lower',extent=[-a.half_size,a.half_size]*2,cmap=cmap)
                ax[i,j].set(title=f'{label} {name.upper()} ({regime})',xlabel='x (kpc)',ylabel='y (kpc)')
                fig.colorbar(im,ax=ax[i,j],label='mag' if name=='av' else 'R(V)')
        fig.savefig(f'{pre}_{regime}_maps.png',dpi=180);plt.close(fig)
    np.savez_compressed(f'{pre}_products.npz',zg_av_3d=gz,zg_rv_3d=gr,
       sim_av_3d=sz,sim_rv_3d=sr,cut_av_layer=cut,pixel_kpc=a.pixel,dz_kpc=a.dz,
       k_kpc_inv=kc,**maps)
    for field in ('AV','RV'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
        for label,regime,name,p,nm,area,nvalid in rows:
            if field!=name:continue
            q=np.isfinite(p)&(p>0)&(nm>=8)
            for ax,y in zip(axes,(p,kc**2*p)):
                ax.loglog(kc[q],y[q],'-' if regime=='cut' else '--',marker='o',
                          ms=3,label=f'{label} {regime} (N={nvalid})')
        for ax,title,ylabel in zip(axes,('Raw pseudo-power','Power per log spatial scale'),
                                   (rf'$P_{{{field}}}(k)$',rf'$k^2P_{{{field}}}(k)$')):
            ax.set(xlabel=r'$k$ (kpc$^{-1}$)',ylabel=ylabel,title=title)
            ax.axvline(1/a.sim_sigma,color='0.45',ls=':',lw=1,label=r'$1/\sigma$')
            ax.axvspan(1/a.sim_sigma,kmax,color='0.5',alpha=.08)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{field}: ZG vs CG10; '+('matched added Gaussian' if a.match_smoothing
                                               else 'unmatched smoothing'))
        fig.tight_layout()
        fig.savefig(f'{pre}_{field.lower()}_power.png',dpi=200);plt.close(fig)
    for label,regime,name,p,nm,area,nvalid in rows:
        np.savetxt(f'{pre}_{label.lower()}_{regime}_{name.lower()}_power.txt',
                   np.column_stack((kc,2*np.pi/kc,p,kc**2*p,nm)),
                   header='k_kpc-1 wavelength_kpc P_field2_kpc2 k2P n_modes')
    report={'script_version':SCRIPT_VERSION,
            'zg_cut_quantile':a.cut_quantile,'cut_delta_Av_mag':cut,'dz_kpc':a.dz,
            'pixel_kpc':a.pixel,'sigma_sim_kpc':a.sim_sigma,
            'matched_added_gaussian':a.match_smoothing,'zg_min_support':a.zg_min_support,
            'n_k_bins_below_1_over_sigma':int(np.sum(kc<1/a.sim_sigma)),
            'radius_kpc':a.radius,
            'zmax_kpc':a.zmax,'sim_metadata':meta,
            'selection':{f'{label}_{regime}_{name}':{'valid_pixels':n,'effective_area_kpc2':area}
                         for label,regime,name,p,nm,area,n in rows},
            'caution':'Masked pseudo-spectra; no mask deconvolution, noise or particle shot-noise subtraction.'}
    Path(f'{pre}_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(f'Wrote {pre}_3d_histograms.png, {pre}_av_power.png, {pre}_rv_power.png and numerical products')

if __name__=='__main__':main()
