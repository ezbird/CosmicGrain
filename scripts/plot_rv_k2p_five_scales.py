#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f: raise ValueError(f"{a.shape} not divisible by {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def make_rv(tb,tv):
    d=tb-tv; r=np.full_like(tv,np.nan,dtype=float)
    g=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0); r[g]=tv[g]/d[g]
    return r

def detrend(a,m,dx,dr):
    ny,nx=a.shape
    x=(np.arange(nx)-(nx-1)/2)*dx; y=(np.arange(ny)-(ny-1)/2)*dx
    X,Y=np.meshgrid(x,y); ib=np.floor(np.hypot(X,Y)/dr).astype(int)
    o=np.full_like(a,np.nan)
    for b in np.unique(ib[m]):
        q=m&(ib==b)
        if q.any(): o[q]=a[q]-np.nanmean(a[q])
    return o

def window(m,t):
    if t<=0:return m.astype(float)
    q=np.clip(distance_transform_edt(m)/t,0,1)
    w=np.sin(np.pi*q/2)**2; w[~m]=0; return w

def spectrum(a,m,dx,L,t):
    v=m&np.isfinite(a); w=window(v,t)
    mu=np.sum(w*np.where(v,a,0))/w.sum()
    d=np.zeros_like(a); d[v]=a[v]-mu
    F=dx**2*np.fft.fftshift(np.fft.fft2(w*d))
    P=np.abs(F)**2/(dx**2*np.sum(w*w))
    ny,nx=a.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    X,Y=np.meshgrid(kx,ky); K=np.hypot(X,Y)
    kf=2*np.pi/L; kn=np.pi/dx
    edges=np.arange(.5*kf,kn+1.5*kf,kf)
    ks=[]; ps=[]; ns=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.any():ks.append(K[q].mean());ps.append(P[q].mean());ns.append(q.sum())
    k=np.array(ks);p=np.array(ps)
    return k,k*k*p,np.array(ns),kn

def main():
    p=argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1.,2.])
    p.add_argument("--min-count",type=int,default=1)
    p.add_argument("--signal-frac",type=float,default=1e-3)
    p.add_argument("--radial-bin-width",type=float,default=1.)
    p.add_argument("--taper-pixels",type=int,default=2)
    p.add_argument("--min-modes",type=int,default=8)
    p.add_argument("--raw",action="store_true")
    p.add_argument("--prefix",default="rv_k2p_five_scales")
    a=p.parse_args()
    z=np.load(a.npz);tb0=z["tau_b"].astype(float);tv0=z["tau_v"].astype(float)
    nc0=z["dust_count"].astype(int);dx0=float(z["pixel_size_pkpc"])
    R=float(z["plot_radius_pkpc"]);L=2*R
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    print(f"Master dx={dx0:g} pkpc; mode={'raw' if a.raw else 'detrended'}")
    for dx in a.pixel_sizes:
        f=int(round(dx/dx0))
        if f<1 or not np.isclose(dx/dx0,f):raise ValueError(f"{dx} not integer multiple of {dx0}")
        tb=block_sum(tb0,f);tv=block_sum(tv0,f);nc=block_sum(nc0,f).astype(int)
        rv=make_rv(tb,tv);th=a.signal_frac*np.nanmax(tv)
        m=np.isfinite(rv)&(nc>=a.min_count)&(tv>=th)
        field=rv if a.raw else detrend(rv,m,dx,a.radial_bin_width)
        k,k2p,n,kn=spectrum(field,m,dx,L,a.taper_pixels)
        g=(n>=a.min_modes)&(k>0)&(k2p>0)
        label=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"
        ax.loglog(k[g],k2p[g],marker="o",ms=3,lw=1.5,label=label)
        q=np.percentile(rv[m],[16,50,84])
        print(f"{label}: valid={m.sum()}/{m.size} ({100*m.mean():.2f}%), Rv={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}, kNyq={kn:.3f}")
    ax.set_xlabel(r"$k\ ({\rm kpc}^{-1})$")
    ax.set_ylabel(r"$k^2P(k)$" if a.raw else r"$k^2P_{\delta R_V}(k)$")
    #ax.set_title("Raw $R_V$ power spectrum" if a.raw else "Radial-detrended $R_V$ power spectrum")
    ax.set_title("Halo 441 - 1024^3")
    ax.grid(True,which="both",alpha=.22);ax.legend(title="Pixel size")
    out=a.prefix+("_raw" if a.raw else "_detrended")
    fig.savefig(out+".png",dpi=250,bbox_inches="tight");fig.savefig(out+".pdf",bbox_inches="tight")
    print("Saved",out+".png/pdf")
if __name__=="__main__":main()
