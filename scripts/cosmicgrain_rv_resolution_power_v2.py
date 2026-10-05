#!/usr/bin/env python3
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f: raise ValueError(f"{a.shape} not divisible by {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def expand_mask(m,f): return np.repeat(np.repeat(m,f,0),f,1)

def make_rv(tb,tv):
    d=tb-tv; r=np.full_like(tv,np.nan,dtype=float)
    g=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0); r[g]=tv[g]/d[g]
    return r

def detrend(a,m,dx,dr):
    ny,nx=a.shape; x=(np.arange(nx)-(nx-1)/2)*dx; y=(np.arange(ny)-(ny-1)/2)*dx
    X,Y=np.meshgrid(x,y); ib=np.floor(np.hypot(X,Y)/dr).astype(int); o=np.full_like(a,np.nan)
    for b in np.unique(ib[m]):
        q=m&(ib==b)
        if q.any(): o[q]=a[q]-np.nanmean(a[q])
    return o

def win(m,t):
    if t<=0:return m.astype(float)
    q=np.clip(distance_transform_edt(m)/t,0,1); w=np.sin(np.pi*q/2)**2; w[~m]=0; return w

def spec(a,m,dx,L,t):
    v=m&np.isfinite(a); w=win(v,t); mu=np.sum(w*np.where(v,a,0))/w.sum()
    d=np.zeros_like(a); d[v]=a[v]-mu
    F=dx**2*np.fft.fftshift(np.fft.fft2(w*d)); P=np.abs(F)**2/(dx**2*np.sum(w*w))
    ny,nx=a.shape; kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx)); ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    X,Y=np.meshgrid(kx,ky); K=np.hypot(X,Y); kf=2*np.pi/L; kn=np.pi/dx
    edges=np.arange(.5*kf,kn+1.5*kf,kf); ks=[]; ps=[]; ns=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.any():ks.append(K[q].mean());ps.append(P[q].mean());ns.append(q.sum())
    k=np.array(ks);p=np.array(ps);n=np.array(ns);return k,p,k*k*p,n,kf,kn

def tag(dx):return f"{dx*1000:.0f}pc"

def ratio(k0,y0,k,y):
    g0=(k0>0)&(y0>0)&np.isfinite(y0); g=(k>0)&(y>0)&np.isfinite(y)
    use=g&(k>=k0[g0].min())&(k<=k0[g0].max())
    if not use.any():return np.array([]),np.array([])
    ref=np.exp(np.interp(np.log(k[use]),np.log(k0[g0]),np.log(y0[g0])))
    return k[use],y[use]/ref

def figure(R,idx,title,ylabel,name,minm,cut):
    fig,(ax,ra)=plt.subplots(2,1,figsize=(8.5,8),sharex=True,gridspec_kw={"height_ratios":[3.2,1.1]},constrained_layout=True)
    ref=R[0][idx]; gr=(ref[3]>=minm)&(ref[0]>0)
    for r in R:
        q=r[idx]; g=(q[3]>=minm)&(q[0]>0)
        ax.loglog(q[0][g],q[2][g],marker="o",ms=3,label=tag(r[0]))
        ax.axvline(cut*q[5],ls=":",lw=.8,alpha=.3)
        if r is not R[0]:
            x,y=ratio(ref[0][gr],ref[2][gr],q[0][g],q[2][g])
            if len(x):ra.semilogx(x,y,marker="o",ms=2.5,label=tag(r[0]))
    ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(True,which="both",alpha=.25);ax.legend(title="Pixel")
    ra.axhline(1,lw=1);ra.axhspan(.8,1.2,alpha=.12);ra.set_ylabel("ratio to\nfinest");ra.set_xlabel(r"$k$ [pkpc$^{-1}$]");ra.grid(True,which="both",alpha=.25);ra.legend(fontsize=8,ncol=3)
    top=ax.secondary_xaxis("top",functions=(lambda k:2*np.pi/k,lambda l:2*np.pi/l));top.set_xlabel(r"Wavelength $\lambda=2\pi/k$ [pkpc]")
    fig.savefig(name+".png",dpi=220,bbox_inches="tight");fig.savefig(name+".pdf",bbox_inches="tight");plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument("npz");p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.25,.5,1,2]);p.add_argument("--min-count",type=int,default=1);p.add_argument("--signal-frac",type=float,default=1e-3);p.add_argument("--radial-bin-width",type=float,default=1);p.add_argument("--taper-pixels",type=int,default=2);p.add_argument("--min-modes",type=int,default=8);p.add_argument("--cutoff-fraction",type=float,default=.5);p.add_argument("--prefix",default="rv_resolution_power_v2");a=p.parse_args()
    z=np.load(a.npz);tb0=z["tau_b"].astype(float);tv0=z["tau_v"].astype(float);nc0=z["dust_count"].astype(int);dx0=float(z["pixel_size_pkpc"]);R=float(z["plot_radius_pkpc"]);L=2*R
    sizes=sorted(a.pixel_sizes);M=[]
    for dx in sizes:
        f=int(round(dx/dx0))
        if not np.isclose(dx/dx0,f):raise ValueError("Pixel sizes must be integer multiples of master")
        tb=block_sum(tb0,f);tv=block_sum(tv0,f);nc=block_sum(nc0,f).astype(int);rv=make_rv(tb,tv);th=a.signal_frac*np.nanmax(tv);m=np.isfinite(rv)&(nc>=a.min_count)&(tv>=th)
        M.append([dx,f,tb,tv,nc,rv,th,m])
    cm0=M[-1][7];fc=M[-1][1];IND=[];COM=[]
    print("="*76);print("COSMICGRAIN R_V POWER V2 — INDEPENDENT + COMMON FOOTPRINT");print("="*76)
    for d in M:
        dx,f,tb,tv,nc,rv,th,im=d; rr=fc//f
        if fc%f:raise ValueError("Requested pixel sizes must nest exactly")
        cm=expand_mask(cm0,rr)&np.isfinite(rv)
        for m,out in [(im,IND),(cm,COM)]:
            de=detrend(rv,m,dx,a.radial_bin_width); raw=spec(rv,m,dx,L,a.taper_pixels); ds=spec(de,m,dx,L,a.taper_pixels);qs=np.percentile(rv[m],[16,50,84]);out.append((dx,rv,m,tb,tv,nc,de,raw,ds,qs,th))
        print(f"{tag(dx)}: independent={im.sum()}/{im.size} ({100*im.mean():.2f}%), common={cm.sum()}/{cm.size} ({100*cm.mean():.2f}%)")
        print(f"  common Rv p16/median/p84={COM[-1][9][0]:.4f}/{COM[-1][9][1]:.4f}/{COM[-1][9][2]:.4f}; kNyq={COM[-1][7][5]:.4f}, cutoff={a.cutoff_fraction*COM[-1][7][5]:.4f}")
    figure(IND,7,"Raw $R_V$ power — independent masks",r"$k^2P_{R_V}(k)$",a.prefix+"_independent_raw",a.min_modes,a.cutoff_fraction)
    figure(IND,8,"Radial-detrended $R_V$ power — independent masks",r"$k^2P_{\delta R_V}(k)$",a.prefix+"_independent_detrended",a.min_modes,a.cutoff_fraction)
    figure(COM,7,"Raw $R_V$ power — common physical footprint",r"$k^2P_{R_V}(k)$",a.prefix+"_common_raw",a.min_modes,a.cutoff_fraction)
    figure(COM,8,"Radial-detrended $R_V$ power — common physical footprint",r"$k^2P_{\delta R_V}(k)$",a.prefix+"_common_detrended",a.min_modes,a.cutoff_fraction)
    with open(a.prefix+"_summary.csv","w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["pixel_pkpc","ind_valid","ind_fraction","common_valid","common_fraction","common_Rv_p16","common_Rv_median","common_Rv_p84","kNyq","diagnostic_cutoff"])
        for i,c in zip(IND,COM):w.writerow([i[0],i[2].sum(),i[2].mean(),c[2].sum(),c[2].mean(),*c[9],c[7][5],a.cutoff_fraction*c[7][5]])
    print("Saved independent/common raw+detrended convergence PNG/PDF and summary CSV.")

if __name__=="__main__":main()
