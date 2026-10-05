#!/usr/bin/env python3
"""Zhang & Green face-on R(V) map, pseudo-power spectrum, and structure function.

Uses the released example_query.py conventions:
  R(V) = 1.1 * R55_diff + 0.07
and an AV_diff-weighted mean through the vertical slab.

Default geometry matches the existing ZG power-spectrum work:
  +/-2.5 kpc, 50 pc pixels, |z|<=0.4 kpc, dz=0.05 kpc,
  R<=2.5 kpc, 4-pixel cosine apodization, 35 log k bins.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

RV_SCALE, RV_OFFSET = 1.1, 0.07

def args():
    p=argparse.ArgumentParser()
    p.add_argument('mapfile', nargs='?', default='Rv_map_new.h5')
    p.add_argument('--nside', default='combined', choices=['combined','64','128','256'])
    p.add_argument('--half-size', type=float, default=2.5)
    p.add_argument('--pixel', type=float, default=0.05)
    p.add_argument('--zmax', type=float, default=0.4)
    p.add_argument('--dz', type=float, default=0.05)
    p.add_argument('--radius', type=float, default=2.5)
    p.add_argument('--apodize', type=int, default=4)
    p.add_argument('--nbins', type=int, default=35)
    p.add_argument('--sf-bins', type=int, default=30)
    p.add_argument('--sf-max', type=float, default=2.5)
    p.add_argument('--min-column-weight', type=float, default=0.0)
    p.add_argument('--min-valid-z-fraction', type=float, default=0.0)
    p.add_argument('--rv-min', type=float)
    p.add_argument('--rv-max', type=float)
    p.add_argument('--field', choices=['absolute','fractional'], default='absolute')
    p.add_argument('--prefix', default='zhang_green_rv')
    return p.parse_args()

def metadata(fn, ns):
    with h5py.File(fn,'r') as f:
        db=np.asarray(f['distance_bins'][:],float)
        n=int(f[f'map_{ns}/nside'][()])
        o=f[f'map_{ns}/hp_order'][()]
        if isinstance(o,bytes): o=o.decode()
        else: o=str(o)
    return db,n,o

def query_diff(fn, ns, coords):
    db,nside,order=metadata(fn,ns)
    xyz = coords.cartesian.xyz.to_value(u.kpc)
    d = np.sqrt(np.sum(xyz**2, axis=0))
    di=np.searchsorted(db,d)-1
    good=(di>=0)&(di<len(db)-1)
    hp=HEALPix(nside=nside,order=order,frame='galactic')
    hi=hp.skycoord_to_healpix(coords)
    r55=np.full(len(d),np.nan); av=np.full(len(d),np.nan)
    with h5py.File(fn,'r') as f:
        rd=f[f'map_{ns}/R55_map_diff']; ad=f[f'map_{ns}/AV_map_diff']
        for ib in np.unique(di[good]):
            q=good&(di==ib)
            rrow=np.asarray(rd[int(ib),:],float); arow=np.asarray(ad[int(ib),:],float)
            r55[q]=rrow[hi[q]]; av[q]=arow[hi[q]]
    return r55,av

def build(a):
    n=int(round(2*a.half_size/a.pixel)); dx=2*a.half_size/n
    e=np.linspace(-a.half_size,a.half_size,n+1); c=.5*(e[:-1]+e[1:])
    xx,yy=np.meshgrid(c,c,indexing='xy')
    nz=max(1,int(round(2*a.zmax/a.dz))); ze=np.linspace(-a.zmax,a.zmax,nz+1)
    zc=.5*(ze[:-1]+ze[1:]); dz=ze[1]-ze[0]
    sw=np.zeros_like(xx); srw=np.zeros_like(xx); nl=np.zeros_like(xx,dtype=int)
    fx,fy=xx.ravel(),yy.ravel()
    print(f'Building {n}x{n} map; pixel={dx:.4f} kpc; {nz} z layers')
    for iz,z in enumerate(zc):
        co=SkyCoord(u=fx*u.kpc,v=fy*u.kpc,w=np.full_like(fx,z)*u.kpc,
                    frame='galactic',representation_type='cartesian')
        r55,av=query_diff(a.mapfile,a.nside,co); rv=RV_SCALE*r55+RV_OFFSET
        q=np.isfinite(rv)&np.isfinite(av)&(av>0)
        if a.rv_min is not None: q &= rv>=a.rv_min
        if a.rv_max is not None: q &= rv<=a.rv_max
        w=np.where(q,av*dz,0.0)
        sw.ravel()[:] += w; srw.ravel()[:] += np.where(q,rv*w,0.0); nl.ravel()[:] += q
        print(f'  layer {iz+1:2d}/{nz}: {q.sum():,}/{q.size:,} valid')
    vf=nl/nz; rr=np.hypot(xx,yy)
    valid=(sw>a.min_column_weight)&(vf>=a.min_valid_z_fraction)&(rr<=a.radius)
    rv=np.full_like(sw,np.nan); rv[valid]=srw[valid]/sw[valid]
    return c,xx,yy,rr,rv,sw,vf,valid,dx

def window(rr,R,t):
    if t<=0:return (rr<=R).astype(float)
    w=np.zeros_like(rr); inner=max(0,R-t); w[rr<=inner]=1
    q=(rr>inner)&(rr<R); x=(rr[q]-inner)/t; w[q]=.5*(1+np.cos(np.pi*x))
    return w

def power(field,valid,rr,dx,R,ap,nb):
    win=valid.astype(float)*window(rr,R,ap*dx)
    mu=np.sum(field[valid]*win[valid])/np.sum(win[valid])
    f=np.zeros_like(field); f[valid]=field[valid]-mu
    F=np.fft.fftshift(np.fft.fft2(f*win))*dx**2
    A=np.sum(win**2)*dx**2; P2=np.abs(F)**2/A
    n=field.shape[0]; kv=2*np.pi*np.fft.fftshift(np.fft.fftfreq(n,d=dx))
    kx,ky=np.meshgrid(kv,kv,indexing='xy'); kk=np.hypot(kx,ky)
    kf=2*np.pi/(n*dx); kn=np.pi/dx; ed=np.geomspace(kf,kn,nb+1)
    K=[];P=[];N=[]
    for lo,hi in zip(ed[:-1],ed[1:]):
        q=(kk>=lo)&(kk<hi)
        if q.any(): K.append(math.sqrt(lo*hi));P.append(P2[q].mean());N.append(q.sum())
    return np.array(K),np.array(P),np.array(N),win,mu,kf,kn,A

def structure(field,valid,dx,rmax,nb):
    ed=np.geomspace(dx,rmax,nb+1); ss=np.zeros(nb); nn=np.zeros(nb,dtype=np.int64)
    ny,nx=field.shape; ms=int(np.floor(rmax/dx))
    for dy in range(ms+1):
        for sx in range(-ms,ms+1):
            if dy==0 and sx<=0: continue
            r=dx*math.hypot(sx,dy)
            if r<ed[0] or r>=ed[-1]: continue
            ib=np.searchsorted(ed,r,side='right')-1
            ya=max(0,-dy); yb=min(ny,ny-dy); xa=max(0,-sx); xb=min(nx,nx-sx)
            if yb<=ya or xb<=xa: continue
            A=field[ya:yb,xa:xb]; B=field[ya+dy:yb+dy,xa+sx:xb+sx]
            q=valid[ya:yb,xa:xb]&valid[ya+dy:yb+dy,xa+sx:xb+sx]&np.isfinite(A)&np.isfinite(B)
            if q.any(): d=A[q]-B[q]; ss[ib]+=np.sum(d*d); nn[ib]+=q.sum()
    rc=np.sqrt(ed[:-1]*ed[1:]); s=np.full(nb,np.nan); q=nn>0; s[q]=ss[q]/nn[q]
    return rc,s,nn

def main():
    a=args(); c,xx,yy,rr,rv,cw,vf,valid,dx=build(a)
    foot=rr<=a.radius; print('\nDiagnostics')
    print(f'R(V) = {RV_SCALE} R55_diff + {RV_OFFSET}')
    print(f'valid pixels = {valid.sum():,}/{foot.sum():,} ({valid.sum()/foot.sum():.4f})')
    print('R(V) mean/median/std =',np.nanmean(rv),np.nanmedian(rv),np.nanstd(rv))
    fld=rv.copy(); mean=np.nanmean(rv[valid])
    if a.field=='fractional': fld[valid]=rv[valid]/mean-1
    k,P,nm,win,mu,kf,kn,A=power(fld,valid,rr,dx,a.radius,a.apodize,a.nbins); k2=k*k*P
    rs,s2,npair=structure(fld,valid,dx,a.sf_max,a.sf_bins)
    print(f'kfund={kf:.6f}, kNyq={kn:.6f} kpc^-1, effective area={A:.4f} kpc^2')
    print('\nRadial valid fraction:')
    red=np.linspace(0,a.radius,11); rows=[]
    for lo,hi in zip(red[:-1],red[1:]):
        q=(rr>=lo)&(rr<hi); frac=valid[q].mean() if q.any() else np.nan
        rows.append((lo,hi,q.sum(),frac)); print(f'{lo:4.2f}-{hi:4.2f}: {frac:.4f}')
    pre=Path(a.prefix)
    np.savez(f'{pre}_products.npz',x_kpc=c,y_kpc=c,rv=rv,column_weight=cw,valid_z_fraction=vf,
             valid_mask=valid,fft_window=win,k_kpc_inv=k,power=P,k2power=k2,n_modes=nm,
             sf_r_kpc=rs,sf_s2=s2,sf_n_pairs=npair,pixel_kpc=dx,rv_mean=mean,
             rv_scale=RV_SCALE,rv_offset=RV_OFFSET,field_definition=a.field)
    np.savetxt(f'{pre}_power.txt',np.c_[k,P,k2,nm],header='k_kpc^-1 P_kpc2 k2P n_modes')
    np.savetxt(f'{pre}_structure.txt',np.c_[rs,s2,npair],header='r_kpc S2 n_pairs')
    np.savetxt(f'{pre}_valid_fraction_radial.txt',np.asarray(rows),header='r_lo r_hi n_pix valid_fraction')
    ext=[-a.half_size,a.half_size,-a.half_size,a.half_size]
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(rv,origin='lower',extent=ext,cmap='coolwarm_r')
    ax.set(xlabel='x (kpc)',ylabel='y (kpc)',title='Zhang & Green face-on R(V)');ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='R(V)');fig.tight_layout();fig.savefig(f'{pre}_faceon.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6)); show=np.where(foot,cw,np.nan);im=ax.imshow(show,origin='lower',extent=ext)
    ax.contour(xx,yy,valid.astype(float),levels=[.5],linewidths=1);ax.set(xlabel='x (kpc)',ylabel='y (kpc)',title='R(V) column weight + validity boundary');ax.set_aspect('equal');fig.colorbar(im,ax=ax,label=r'$\sum \Delta A_V\,dz$');fig.tight_layout();fig.savefig(f'{pre}_weight_mask.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.5));q=(k2>0)&np.isfinite(k2);ax.loglog(k[q],k2[q],marker='o',ms=4);ax.set_xlabel(r'$k\ (\mathrm{kpc}^{-1})$');ax.set_ylabel(r'$k^2P_{R(V)}(k)$');ax.set_title('Zhang & Green R(V) masked pseudo-power spectrum');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(f'{pre}_power_spectrum.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.5));q=(s2>0)&np.isfinite(s2)&(npair>0);ax.loglog(rs[q],s2[q],marker='o',ms=4);ax.set_xlabel(r'$r\ (\mathrm{kpc})$');ax.set_ylabel(r'$S_2^{R(V)}(r)$');ax.set_title('Zhang & Green R(V) structure function');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(f'{pre}_structure_function.png',dpi=200);plt.close(fig)
    print('\nOutputs written with prefix:',pre)
if __name__=='__main__': main()
