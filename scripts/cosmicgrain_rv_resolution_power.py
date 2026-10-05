#!/usr/bin/env python3
import argparse, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def block_sum(a, f):
    ny,nx=a.shape
    if ny%f or nx%f: raise ValueError(f"{a.shape} not divisible by factor {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def make_rv(tb,tv):
    d=tb-tv; r=np.full_like(tv,np.nan,dtype=float)
    g=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0); r[g]=tv[g]/d[g]
    return r

def radial_detrend(a,m,dx,dr):
    ny,nx=a.shape
    x=(np.arange(nx)-(nx-1)/2)*dx; y=(np.arange(ny)-(ny-1)/2)*dx
    rr=np.hypot(*np.meshgrid(x,y))
    out=np.full_like(a,np.nan)
    ib=np.floor(rr/dr).astype(int)
    for b in np.unique(ib[m]):
        s=m&(ib==b)
        if s.any(): out[s]=a[s]-np.nanmean(a[s])
    return out

def window(mask,taper):
    if taper<=0:return mask.astype(float)
    q=np.clip(distance_transform_edt(mask)/taper,0,1)
    w=np.sin(np.pi*q/2)**2; w[~mask]=0
    return w

def spectrum(a,m,dx,L,taper):
    v=m&np.isfinite(a); w=window(v,taper)
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
        s=(K>=lo)&(K<hi)
        if s.any(): ks.append(K[s].mean()); ps.append(P[s].mean()); ns.append(s.sum())
    k=np.array(ks); p=np.array(ps); n=np.array(ns)
    return k,p,k*k*p,n,kf,kn

def tag(dx): return f"{dx*1000:.0f}pc"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.25,.5,1.,2.])
    p.add_argument("--min-count",type=int,default=1)
    p.add_argument("--signal-frac",type=float,default=1e-3)
    p.add_argument("--radial-bin-width",type=float,default=1.)
    p.add_argument("--taper-pixels",type=int,default=2)
    p.add_argument("--min-modes",type=int,default=8)
    p.add_argument("--prefix",default="rv_resolution_power")
    a=p.parse_args()
    z=np.load(a.npz)
    tb0=z["tau_b"].astype(float); tv0=z["tau_v"].astype(float)
    nc0=z["dust_count"].astype(int)
    dx0=float(z["pixel_size_pkpc"]); R=float(z["plot_radius_pkpc"]); L=2*R
    print("="*76); print("COSMICGRAIN MATCHED R_V RESOLUTION / POWER ANALYSIS"); print("="*76)
    print(f"Master: {tb0.shape}, dx={dx0:g} pkpc, width={L:g} pkpc\n")
    res=[]
    for dx in a.pixel_sizes:
        q=dx/dx0; f=int(round(q))
        if f<1 or not np.isclose(q,f): raise ValueError(f"{dx} not integer multiple of {dx0}")
        tb=block_sum(tb0,f); tv=block_sum(tv0,f); nc=block_sum(nc0,f).astype(int)
        rv=make_rv(tb,tv); th=a.signal_frac*np.nanmax(tv)
        m=np.isfinite(rv)&(nc>=a.min_count)&(tv>=th)
        det=radial_detrend(rv,m,dx,a.radial_bin_width)
        raw=spectrum(rv,m,dx,L,a.taper_pixels); de=spectrum(det,m,dx,L,a.taper_pixels)
        qs=np.percentile(rv[m],[16,50,84])
        res.append((dx,rv,m,tb,tv,nc,det,raw,de,qs,th))
        print(f"{tag(dx)}: grid={rv.shape[0]}x{rv.shape[1]}, valid={m.sum()}/{m.size} ({100*m.mean():.2f}%)")
        print(f"  Rv p16/median/p84 = {qs[0]:.4f} / {qs[1]:.4f} / {qs[2]:.4f}")
        print(f"  kfund={raw[4]:.6f}, kNyq={raw[5]:.6f} pkpc^-1, lambdaNyq={2*np.pi/raw[5]:.3f} pkpc\n")
        np.savez_compressed(f"{a.prefix}_{tag(dx)}_maps.npz",rv=rv,rv_mask=m,tau_b=tb,tau_v=tv,
            dust_count=nc,rv_radial_detrended=det,pixel_size_pkpc=dx,plot_radius_pkpc=R,
            signal_fraction=a.signal_frac,signal_threshold_tau_v=th,min_dust_count=a.min_count)
        with open(f"{a.prefix}_{tag(dx)}_power.csv","w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["k_pkpc^-1","lambda_pkpc","P_raw","k2P_raw","Nmodes_raw",
                                         "P_detrended","k2P_detrended","Nmodes_detrended"])
            for j in range(min(len(raw[0]),len(de[0]))):
                w.writerow([raw[0][j],2*np.pi/raw[0][j],raw[1][j],raw[2][j],raw[3][j],
                            de[1][j],de[2][j],de[3][j]])
    # Rv ladder
    fig,axs=plt.subplots(1,len(res),figsize=(4.3*len(res),4.4),constrained_layout=True)
    axs=np.atleast_1d(axs); vals=np.concatenate([r[1][r[2]] for r in res]); lo,hi=np.percentile(vals,[2,98])
    for ax,r in zip(axs,res):
        im=ax.imshow(np.ma.masked_where(~r[2],r[1]),origin="lower",extent=(-R,R,-R,R),
                     interpolation="nearest",vmin=lo,vmax=hi)
        ax.set_title(f"{tag(r[0])}\\nvalid={100*r[2].mean():.1f}%, med={r[9][1]:.2f}")
        ax.set_xlabel("X face-on [pkpc]"); ax.set_ylabel("Y face-on [pkpc]")
    fig.colorbar(im,ax=list(axs),label=r"$R_V$",shrink=.8)
    fig.savefig(a.prefix+"_rv_resolution_ladder.png",dpi=220); fig.savefig(a.prefix+"_rv_resolution_ladder.pdf"); plt.close(fig)
    # convergence plots
    for which,title,suffix,ylabel in [(7,"Raw $R_V$ power","raw",r"$k^2P_{R_V}(k)$"),
                                      (8,"Radial-detrended $R_V$ power","detrended",r"$k^2P_{\delta R_V}(k)$")]:
        fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
        for r in res:
            s=r[which]; g=(s[3]>=a.min_modes)&(s[0]>0)
            ax.loglog(s[0][g],s[2][g],marker="o",ms=3,label=tag(r[0]))
        ax.set_xlabel(r"$k$ [pkpc$^{-1}$]"); ax.set_ylabel(ylabel); ax.set_title(title+" — matched resolution")
        ax.grid(True,which="both",alpha=.25); ax.legend(title="Pixel")
        fig.savefig(f"{a.prefix}_k2p_{suffix}_convergence.png",dpi=220); fig.savefig(f"{a.prefix}_k2p_{suffix}_convergence.pdf"); plt.close(fig)
    # raw vs detrended panels
    rows=math.ceil(len(res)/2); fig,axs=plt.subplots(rows,2,figsize=(11,4.5*rows),constrained_layout=True)
    axs=np.atleast_1d(axs).ravel()
    for ax,r in zip(axs,res):
        for s,label in [(r[7],"raw"),(r[8],"radial-detrended")]:
            g=(s[3]>=a.min_modes)&(s[0]>0); ax.loglog(s[0][g],s[2][g],marker="o",ms=3,label=label)
        ax.set_title(tag(r[0])); ax.set_xlabel(r"$k$ [pkpc$^{-1}$]"); ax.set_ylabel(r"$k^2P(k)$")
        ax.grid(True,which="both",alpha=.25); ax.legend()
    for ax in axs[len(res):]: ax.axis("off")
    fig.savefig(a.prefix+"_raw_vs_detrended.png",dpi=220); fig.savefig(a.prefix+"_raw_vs_detrended.pdf"); plt.close(fig)
    with open(a.prefix+"_summary.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["pixel_size_pkpc","valid","total","coverage","Rv_p16","Rv_median","Rv_p84","kfund","kNyq","lambdaNyq"])
        for r in res:w.writerow([r[0],r[2].sum(),r[2].size,r[2].mean(),*r[9],r[7][4],r[7][5],2*np.pi/r[7][5]])
    print("Saved resolution ladder, raw/detrended convergence figures, per-resolution NPZ/CSV, and summary CSV.")

if __name__=="__main__": main()
