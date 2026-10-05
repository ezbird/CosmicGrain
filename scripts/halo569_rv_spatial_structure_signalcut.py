#!/usr/bin/env python3
"""
halo569_rv_spatial_structure_signalcut.py

Standalone face-on CosmicGrain R(V) spatial-structure analysis.

This validated version adds the same low-extinction signal cut used by the
multi-resolution analysis:

    tau_V >= signal_frac * max(tau_V)

with signal_frac=1e-3 by default, combined with N_dust >= min_count.

It reports every stage of mask construction, then computes:
  * raw and radial-trend-subtracted 2-D structure functions;
  * 10, 20, 30, and 50 percent fractional-RMS threshold scales;
  * complementary apodized 2-D power spectra;
  * map, CSV, NPZ, and mask-summary outputs.

No other local Python scripts are imported.
"""
import argparse, glob, os, math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

RHO_C=2.24; RHO_SIL=3.50; B_MICRON=0.44; V_MICRON=0.55

def expand_files(pattern):
    fs=sorted(glob.glob(pattern))
    if not fs and os.path.exists(pattern): fs=[pattern]
    if not fs: raise FileNotFoundError(pattern)
    return fs

def periodic_delta(c, cen, box):
    d=c-cen[None,:]; d-=box*np.round(d/box); return d

def load_draine(path):
    radii=[]; wavs=[]; qext=[]; cr=None; cw=[]; cq=[]
    def finish():
        nonlocal cr,cw,cq
        if cr is not None and cw:
            w=np.array(cw,float); q=np.array(cq,float); o=np.argsort(w)
            radii.append(cr); wavs.append(w[o]); qext.append(q[o])
        cr=None; cw=[]; cq=[]
    with open(path,'r',errors='ignore') as f:
        for line in f:
            s=line.strip(); low=s.lower()
            if not s: continue
            if 'radius' in low and ('micron' in low or 'um' in low):
                finish(); vals=[]
                for t in s.replace('=',' ').replace(',',' ').split():
                    try: vals.append(float(t))
                    except: pass
                pos=[v for v in vals if v>0]
                if pos: cr=pos[0]
                continue
            if cr is not None:
                p=s.split()
                if len(p)>=2:
                    try: w=float(p[0]); q=float(p[1])
                    except: continue
                    if w>0 and q>0 and np.isfinite(w) and np.isfinite(q): cw.append(w); cq.append(q)
    finish()
    r=np.array(radii); o=np.argsort(r); r=r[o]; wavs=[wavs[i] for i in o]; qext=[qext[i] for i in o]
    bw=wavs[0]; grid=[]
    for w,q in zip(wavs,qext):
        if len(w)==len(bw) and np.allclose(w,bw,rtol=1e-8,atol=0): qi=q
        else: qi=10**np.interp(np.log10(bw),np.log10(w),np.log10(q),left=np.nan,right=np.nan)
        grid.append(qi)
    grid=np.array(grid); good=np.all(np.isfinite(grid)&(grid>0),axis=0); bw=bw[good]; grid=grid[:,good]
    interp=RegularGridInterpolator((np.log10(r),np.log10(bw)),np.log10(grid),bounds_error=False,fill_value=np.nan)
    print(f"Loaded {path}: {len(r)} radii, {r.min():.4g}-{r.max():.4g} micron")
    return interp

def qeval(interp,a,w):
    pts=np.column_stack((np.log10(a),np.full_like(a,np.log10(w))))
    return 10**interp(pts)

def read_dust(pattern):
    C=[]; M=[]; A=[]; F=[]; a=h=box=None; cfield=None
    print('\nReading PartType6')
    for fn in expand_files(pattern):
        print(' ',fn)
        with h5py.File(fn,'r') as f:
            if a is None:
                a=float(f['Header'].attrs['Time']); box=float(f['Header'].attrs['BoxSize'])
                h=float(f['Parameters'].attrs['HubbleParam']) if 'Parameters' in f and 'HubbleParam' in f['Parameters'].attrs else float(f['Header'].attrs['HubbleParam'])
            if 'PartType6' not in f: continue
            p=f['PartType6']; cf='CarbonMassFraction' if 'CarbonMassFraction' in p else 'CarbonFraction'
            print('   carbon field =',cf); cfield=cf
            C.append(np.asarray(p['Coordinates'][:],float)); M.append(np.asarray(p['Masses'][:],float)); A.append(np.asarray(p['GrainRadius'][:],float)); F.append(np.asarray(p[cf][:],float))
    C=np.concatenate(C); M=np.concatenate(M); A=np.concatenate(A); F=np.concatenate(F)
    print(f"Dust particles = {len(C):,}; radius {A.min():.4f}-{A.max():.4f} nm; CF {F.min():.4f}-{F.max():.4f}")
    return dict(coords=C,mass=M,rad=A,cf=F,a=a,h=h,box=box,cfield=cfield)

def weights(d, ci, si):
    a=d['rad']*1e-3; f=d['cf']; m=d['mass']; base=m/a
    cb=qeval(ci,a,B_MICRON); cv=qeval(ci,a,V_MICRON); sb=qeval(si,a,B_MICRON); sv=qeval(si,a,V_MICRON)
    wb=base*(f*cb/RHO_C+(1-f)*sb/RHO_SIL); wv=base*(f*cv/RHO_C+(1-f)*sv/RHO_SIL)
    good=np.isfinite(wb)&np.isfinite(wv)&(wb>0)&(wv>0)
    return wb,wv,good

def build_map(d,center,R,wb,wv,good,rad,zmax,pix):
    xyz=(periodic_delta(d['coords'],center,d['box'])*d['a']/d['h'])@R.T
    x,y,z=xyz.T; sel=good&(np.abs(x)<rad)&(np.abs(y)<rad)&(np.abs(z)<zmax)
    bins=max(2,int(round(2*rad/pix))); edges=np.linspace(-rad,rad,bins+1); actual=2*rad/bins
    tb=np.histogram2d(y[sel],x[sel],bins=[edges,edges],weights=wb[sel])[0]
    tv=np.histogram2d(y[sel],x[sel],bins=[edges,edges],weights=wv[sel])[0]
    n=np.histogram2d(y[sel],x[sel],bins=[edges,edges])[0].astype(int)
    rv=np.full_like(tv,np.nan); den=tb-tv; ok=(tv>0)&(den>0); rv[ok]=tv[ok]/den[ok]
    cen=.5*(edges[:-1]+edges[1:]); print(f"Dust in volume = {sel.sum():,}; grid={bins}x{bins}; pixel={actual:.4f} pkpc")
    return rv,n,cen,tb,tv,actual

def radial_detrend(rv,mask,cen,width):
    X,Y=np.meshgrid(cen,cen); rr=np.hypot(X,Y); rmax=np.nanmax(rr[mask]); edges=np.arange(0,rmax+width,width)
    trend=np.full_like(rv,np.nan); rows=[]
    for i in range(len(edges)-1):
        s=mask&(rr>=edges[i])&(rr<edges[i+1]); med=np.nanmedian(rv[s]) if np.any(s) else np.nan; trend[s]=med; rows.append((edges[i],edges[i+1],s.sum(),med))
    return trend,rv-trend,rows

def sf2(field,mask,pix,maxsep,binw):
    ny,nx=field.shape; nb=int(np.ceil(maxsep/binw)); ss=np.zeros(nb); sa=np.zeros(nb); nn=np.zeros(nb,dtype=np.int64); mp=int(np.ceil(maxsep/pix))
    for dy in range(mp+1):
        for dx in range(-mp,mp+1):
            if dy==0 and dx<=0: continue
            sep=pix*math.hypot(dx,dy)
            if sep<=0 or sep>maxsep: continue
            b=int(sep/binw)
            if b>=nb: continue
            y1a=max(0,-dy); y1b=min(ny,ny-dy); y2a=y1a+dy; y2b=y1b+dy
            x1a=max(0,-dx); x1b=min(nx,nx-dx); x2a=x1a+dx; x2b=x1b+dx
            if y1a>=y1b or x1a>=x1b: continue
            g=mask[y1a:y1b,x1a:x1b]&mask[y2a:y2b,x2a:x2b]
            if not np.any(g): continue
            diff=field[y2a:y2b,x2a:x2b][g]-field[y1a:y1b,x1a:x1b][g]
            ss[b]+=np.sum(diff*diff); sa[b]+=np.sum(np.abs(diff)); nn[b]+=diff.size
    rows=[]
    for b in range(nb):
        c=(b+.5)*binw
        if nn[b]: s2=ss[b]/nn[b]; rms=np.sqrt(s2); mad=sa[b]/nn[b]
        else: s2=rms=mad=np.nan
        rows.append((c,int(nn[b]),s2,rms,mad))
    return rows

def crossing(rows,norm,thr):
    x=np.array([r[0] for r in rows if r[1]>0 and np.isfinite(r[3])]); y=np.array([r[3]/norm for r in rows if r[1]>0 and np.isfinite(r[3])])
    hit=np.where(y>=thr)[0]
    if not len(hit): return np.nan
    i=hit[0]
    if i==0:return x[0]
    return x[i-1]+(thr-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1]) if y[i]!=y[i-1] else x[i]

def apod(mask,width):
    from scipy.ndimage import distance_transform_edt
    d=distance_transform_edt(mask); w=np.zeros_like(d,float)
    if width<=0: w[mask]=1; return w
    q=np.clip(d[mask]/width,0,1); w[mask]=.5-.5*np.cos(np.pi*q); return w

def power(field,mask,pix,aw):
    w=apod(mask,aw); work=np.zeros_like(field); work[mask]=field[mask]-np.nanmean(field[mask]); work*=w
    ft=np.fft.fftshift(np.fft.fft2(work)); p=np.abs(ft)**2; ny,nx=field.shape
    kx=np.fft.fftshift(np.fft.fftfreq(nx,d=pix)); ky=np.fft.fftshift(np.fft.fftfreq(ny,d=pix)); K=np.hypot(*np.meshgrid(kx,ky))
    dk=min(1/(nx*pix),1/(ny*pix)); edges=np.arange(0,.5/pix+dk,dk); rows=[]
    for i in range(1,len(edges)-1):
        s=(K>=edges[i])&(K<edges[i+1]);
        if not np.any(s): continue
        k=np.nanmean(K[s]); rows.append((k,1/k if k>0 else np.nan,int(s.sum()),np.nanmean(p[s])))
    return rows,w,p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('snapshot'); ap.add_argument('--center-file',required=True); ap.add_argument('--rotation-file',required=True); ap.add_argument('--carbon-table',required=True); ap.add_argument('--silicate-table',required=True)
    ap.add_argument('--plot-radius',type=float,default=20); ap.add_argument('--zmax',type=float,default=20); ap.add_argument('--pixel-size',type=float,default=.5); ap.add_argument('--min-count',type=int,default=5); ap.add_argument('--signal-frac',type=float,default=1e-3,help='Require tau_V >= signal_frac * peak(tau_V); default 1e-3 to match the multiresolution analysis.'); ap.add_argument('--radial-bin-width',type=float,default=1.0); ap.add_argument('--max-separation',type=float,default=10); ap.add_argument('--separation-bin-width',type=float,default=.5); ap.add_argument('--apodize-pixels',type=int,default=3); ap.add_argument('--prefix',default='halo569_rv_structure')
    a=ap.parse_args(); center=np.loadtxt(a.center_file).reshape(3); R=np.loadtxt(a.rotation_file).reshape(3,3)
    print('Galaxy center:',center); print('Rotation matrix:\n',R)
    ci=load_draine(a.carbon_table); si=load_draine(a.silicate_table); d=read_dust(a.snapshot); wb,wv,good=weights(d,ci,si)
    rv,n,cen,tb,tv,pix=build_map(d,center,R,wb,wv,good,a.plot_radius,a.zmax,a.pixel_size)

    finite_rv=np.isfinite(rv)
    tauv_pos=np.isfinite(tv)&(tv>0)
    tau_order=np.isfinite(tb)&np.isfinite(tv)&((tb-tv)>0)
    count_mask=(n>=a.min_count)

    peak_tauv=np.nanmax(tv[tauv_pos]) if np.any(tauv_pos) else np.nan
    if not np.isfinite(peak_tauv) or peak_tauv<=0:
        raise RuntimeError('Could not determine a positive peak tau_V for the signal mask.')

    signal_threshold=a.signal_frac*peak_tauv
    signal_mask=tauv_pos&(tv>=signal_threshold)

    mask=finite_rv&tau_order&count_mask&signal_mask

    print('\nMask construction:')
    print(f'  total pixels                  = {rv.size:,}')
    print(f'  finite mathematical Rv        = {finite_rv.sum():,}')
    print(f'  tau_V > 0                     = {tauv_pos.sum():,}')
    print(f'  tau_B - tau_V > 0             = {tau_order.sum():,}')
    print(f'  N_dust >= {a.min_count:<3d}                  = {count_mask.sum():,}')
    print(f'  peak tau_V                     = {peak_tauv:.8e}')
    print(f'  signal fraction                = {a.signal_frac:.8e}')
    print(f'  tau_V signal threshold         = {signal_threshold:.8e}')
    print(f'  pixels above signal threshold  = {signal_mask.sum():,}')
    print(f'  FINAL valid pixels             = {mask.sum():,}')

    if mask.sum()<10:
        raise RuntimeError('Too few pixels survive the combined count and signal mask.')

    med=np.nanmedian(rv[mask]); p16,p84=np.nanpercentile(rv[mask],[16,84]); print(f"Median Rv={med:.4f}; p16-p84={p16:.4f}-{p84:.4f}")
    trend,res,radrows=radial_detrend(rv,mask,cen,a.radial_bin_width); rmask=mask&np.isfinite(res)
    raw=sf2(rv,mask,pix,a.max_separation,a.separation_bin_width); det=sf2(res,rmask,pix,a.max_separation,a.separation_bin_width)
    print('\nThreshold scales:')
    for t in (.1,.2,.3,.5):
        r1=crossing(raw,med,t); r2=crossing(det,med,t); print(f"{int(t*100):2d}%: raw={r1 if np.isfinite(r1) else 'not reached'} pkpc; detrended={r2 if np.isfinite(r2) else 'not reached'} pkpc")
    rps,rw,rp=power(rv,mask,pix,a.apodize_pixels); dps,dw,dp=power(res,rmask,pix,a.apodize_pixels)
    with open(a.prefix+'_mask_summary.txt','w') as f:
        f.write(f'pixel_size_pkpc = {pix:.8f}\n')
        f.write(f'min_count = {a.min_count}\n')
        f.write(f'peak_tau_V = {peak_tauv:.12e}\n')
        f.write(f'signal_fraction = {a.signal_frac:.12e}\n')
        f.write(f'signal_threshold_tau_V = {signal_threshold:.12e}\n')
        f.write(f'finite_Rv_pixels = {finite_rv.sum()}\n')
        f.write(f'count_mask_pixels = {count_mask.sum()}\n')
        f.write(f'signal_mask_pixels = {signal_mask.sum()}\n')
        f.write(f'final_valid_pixels = {mask.sum()}\n')

    with open(a.prefix+'_structure_function.csv','w') as f:
        f.write('separation_pkpc,raw_pairs,raw_S2,raw_sqrtS2,raw_fractional_rms,detrended_pairs,detrended_S2,detrended_sqrtS2,detrended_fractional_rms\n')
        for rr,dd in zip(raw,det): f.write(f"{rr[0]:.8e},{rr[1]},{rr[2]:.8e},{rr[3]:.8e},{rr[3]/med:.8e},{dd[1]},{dd[2]:.8e},{dd[3]:.8e},{dd[3]/med:.8e}\n")
    with open(a.prefix+'_power_spectrum.csv','w') as f:
        f.write('raw_k_cycles_per_pkpc,raw_scale_pkpc,raw_modes,raw_power,detrended_k_cycles_per_pkpc,detrended_scale_pkpc,detrended_modes,detrended_power\n')
        L=max(len(rps),len(dps))
        for i in range(L):
            rr=rps[i] if i<len(rps) else (np.nan,np.nan,0,np.nan); dd=dps[i] if i<len(dps) else (np.nan,np.nan,0,np.nan)
            f.write(f"{rr[0]:.8e},{rr[1]:.8e},{rr[2]},{rr[3]:.8e},{dd[0]:.8e},{dd[1]:.8e},{dd[2]},{dd[3]:.8e}\n")
    with open(a.prefix+'_radial_trend.csv','w') as f:
        f.write('r_inner_pkpc,r_outer_pkpc,Npix,median_Rv\n'); [f.write(f"{x:.8e},{y:.8e},{nn},{m:.8e}\n") for x,y,nn,m in radrows]
    np.savez_compressed(a.prefix+'_spatial_structure.npz',rv=rv,counts=n,mask=mask,signal_mask=signal_mask,count_mask=count_mask,finite_rv_mask=finite_rv,radial_trend=trend,residual=res,centers_pkpc=cen,pixel_size_pkpc=pix,tau_b=tb,tau_v=tv,peak_tau_v=peak_tauv,signal_fraction=a.signal_frac,signal_threshold_tau_v=signal_threshold)
    ext=[cen[0],cen[-1],cen[0],cen[-1]]
    fig,axs=plt.subplots(1,3,figsize=(15,5)); im=axs[0].imshow(np.where(mask,rv,np.nan),origin='lower',extent=ext,interpolation='nearest'); axs[0].set_title(r'$R_V$'); fig.colorbar(im,ax=axs[0]); im=axs[1].imshow(np.where(rmask,res,np.nan),origin='lower',extent=ext,interpolation='nearest'); axs[1].set_title(r'$R_V-\widetilde{R_V}(R)$'); fig.colorbar(im,ax=axs[1]); im=axs[2].imshow(np.where(n>0,n,np.nan),origin='lower',extent=ext,interpolation='nearest'); axs[2].set_title('Dust count'); fig.colorbar(im,ax=axs[2]); [ax.set_xlabel("x' [pkpc]") for ax in axs]; [ax.set_ylabel("y' [pkpc]") for ax in axs]; fig.tight_layout(); fig.savefig(a.prefix+'_maps.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot([r[0] for r in raw],[r[3]/med for r in raw],marker='o',label='Raw Rv'); ax.plot([r[0] for r in det],[r[3]/med for r in det],marker='s',label='Radial-trend subtracted'); [ax.axhline(v,ls='--',lw=1) for v in (.1,.2,.3)]; ax.set_xlabel('Separation [pkpc]'); ax.set_ylabel(r'$\sqrt{S_2}/\widetilde{R_V}$'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.prefix+'_structure_function.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for rows,label,mark in [(rps,'Raw Rv','o'),(dps,'Radial-trend subtracted','s')]:
        sc=np.array([r[1] for r in rows]); pp=np.array([r[3] for r in rows]); g=np.isfinite(sc)&np.isfinite(pp)&(sc>0)&(pp>0); sc=sc[g]; pp=pp[g]; o=np.argsort(sc); ax.loglog(sc[o],(pp/pp.max())[o],marker=mark,label=label)
    ax.set_xlabel('Spatial scale [pkpc]'); ax.set_ylabel('Normalized 2-D power'); ax.legend(); ax.grid(alpha=.25,which='both'); fig.tight_layout(); fig.savefig(a.prefix+'_power_spectrum.png',dpi=180); plt.close(fig)
    print('\nSaved outputs with prefix',a.prefix); print('Mask summary:',a.prefix+'_mask_summary.txt')
if __name__=='__main__': main()
