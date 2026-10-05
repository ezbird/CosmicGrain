#!/usr/bin/env python3
"""
Diagnose why the R_V power spectrum changes with map pixel size.

Three experiments, all starting from the same finest tau_B/tau_V maps:

A natural:
    Sum tau_B and tau_V into coarse pixels, recompute R_V, and apply the
    natural resolution-dependent signal mask.

B footprint-controlled:
    Same physically correct optical-depth rebinning as A, but retain only
    coarse pixels for which ALL native subpixels satisfy the native R_V
    signal mask. This gives a deliberately strict, resolution-independent
    fine-pixel footprint and isolates window/mask effects.

C arithmetic-RV:
    On exactly the same strict footprint as B, arithmetic-average the native
    R_V values into coarse pixels instead of recomputing R_V from summed
    optical depths. B vs C therefore isolates nonlinear extinction mixing.

Run with --raw for the original global-mean-subtracted R_V PS.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def block_sum(x,f):
    ny,nx=x.shape
    if ny%f or nx%f: raise ValueError(f"{x.shape} not divisible by factor {f}")
    return x.reshape(ny//f,f,nx//f,f).sum((1,3))

def block_mean(x,f):
    ny,nx=x.shape
    return x.reshape(ny//f,f,nx//f,f).mean((1,3))

def block_all(m,f):
    ny,nx=m.shape
    return m.reshape(ny//f,f,nx//f,f).all((1,3))

def make_rv(tb,tv):
    d=tb-tv
    rv=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    rv[q]=tv[q]/d[q]
    return rv

def detrend(rv,m,dx,dr):
    ny,nx=rv.shape
    x=(np.arange(nx)-(nx-1)/2)*dx
    y=(np.arange(ny)-(ny-1)/2)*dx
    X,Y=np.meshgrid(x,y)
    ib=np.floor(np.hypot(X,Y)/dr).astype(int)
    out=np.full_like(rv,np.nan)
    for b in np.unique(ib[m]):
        q=m&(ib==b)
        if q.any(): out[q]=rv[q]-np.nanmean(rv[q])
    return out

def window(m,taper):
    if taper<=0:return m.astype(float)
    q=np.clip(distance_transform_edt(m)/taper,0,1)
    w=np.sin(np.pi*q/2)**2
    w[~m]=0
    return w

def spectrum(field,m,dx,L,taper):
    v=m&np.isfinite(field)
    w=window(v,taper)
    if w.sum()==0:return np.array([]),np.array([]),np.array([]),np.pi/dx
    mu=np.sum(w*np.where(v,field,0))/w.sum()
    d=np.zeros_like(field); d[v]=field[v]-mu
    F=dx**2*np.fft.fftshift(np.fft.fft2(w*d))
    P=np.abs(F)**2/(dx**2*np.sum(w*w))
    ny,nx=field.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    K=np.hypot(*np.meshgrid(kx,ky))
    kf=2*np.pi/L; kn=np.pi/dx
    edges=np.arange(.5*kf,kn+1.5*kf,kf)
    ks=[]; pp=[]; nn=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.any():
            ks.append(K[q].mean()); pp.append(P[q].mean()); nn.append(q.sum())
    k=np.asarray(ks); P=np.asarray(pp)
    return k,k*k*P,np.asarray(nn),kn

p=argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1.,2.])
p.add_argument("--min-count",type=int,default=1)
p.add_argument("--signal-frac",type=float,default=1e-3)
p.add_argument("--radial-bin-width",type=float,default=1.)
p.add_argument("--taper-pixels",type=int,default=2)
p.add_argument("--min-modes",type=int,default=8)
p.add_argument("--raw",action="store_true")
p.add_argument("--prefix",default="rv_ps_shift_diagnostic")
a=p.parse_args()

z=np.load(a.npz)
tb0=z["tau_b"].astype(float); tv0=z["tau_v"].astype(float)
nc0=z["dust_count"].astype(int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])
R=float(np.asarray(z["plot_radius_pkpc"]).ravel()[0]); L=2*R
rv0=make_rv(tb0,tv0)
th0=a.signal_frac*np.nanmax(tv0)
native_mask=np.isfinite(rv0)&(nc0>=a.min_count)&(tv0>=th0)

print("="*80)
print("R_V POWER-SPECTRUM SHIFT DIAGNOSTIC")
print("="*80)
print(f"native dx={dx0:g} pkpc, native valid={native_mask.sum()}/{native_mask.size} "
      f"({100*native_mask.mean():.2f}%)")
print(f"mode={'raw/global-mean-subtracted' if a.raw else 'fluctuations about radial average'}")
print("B/C use strict footprint: every native subpixel in a coarse cell must be valid.\n")

experiments={"A_natural":[],"B_strict_recomputed":[],"C_strict_arithmetic":[]}

for dx in a.pixel_sizes:
    f=int(round(dx/dx0))
    if f<1 or not np.isclose(dx/dx0,f): raise ValueError(f"{dx} not integer multiple of {dx0}")

    tb=block_sum(tb0,f); tv=block_sum(tv0,f); nc=block_sum(nc0,f).astype(int)
    rv_re=make_rv(tb,tv)

    # A: current analysis
    th=a.signal_frac*np.nanmax(tv)
    mA=np.isfinite(rv_re)&(nc>=a.min_count)&(tv>=th)

    # B/C: strict common native-pixel support
    mS=block_all(native_mask,f)
    mB=mS&np.isfinite(rv_re)

    # C arithmetic mean is safe because strict support guarantees all subpixels valid
    rv_ar=rv0 if f==1 else block_mean(rv0,f)
    mC=mS&np.isfinite(rv_ar)

    rows=[("A_natural",rv_re,mA),("B_strict_recomputed",rv_re,mB),
          ("C_strict_arithmetic",rv_ar,mC)]
    label=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"

    print(label)
    for key,rv,m in rows:
        field=rv if a.raw else detrend(rv,m,dx,a.radial_bin_width)
        k,k2p,n,kn=spectrum(field,m,dx,L,a.taper_pixels)
        experiments[key].append((dx,k,k2p,n,kn))
        med=np.nanmedian(rv[m]) if m.any() else np.nan
        print(f"  {key:22s}: valid={m.sum():6d}/{m.size:<6d} "
              f"({100*m.mean():6.2f}%)  median Rv={med:7.3f}")

# 3-panel direct comparison
titles={
"A_natural":"A. Recompute $R_V$ + natural mask",
"B_strict_recomputed":"B. Recompute $R_V$ + strict footprint",
"C_strict_arithmetic":"C. Arithmetic-average $R_V$ + strict footprint"}
fig,axs=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
for ax,(key,vals) in zip(axs,experiments.items()):
    for dx,k,k2p,n,kn in vals:
        g=(n>=a.min_modes)&(k>0)&np.isfinite(k2p)&(k2p>0)
        lab=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"
        ax.loglog(k[g],k2p[g],lw=1.5,label=lab)
    ax.set_title(titles[key])
    ax.set_xlabel(r"$k$ (kpc$^{-1}$)")
    ax.grid(True,which="both",alpha=.18)
axs[0].set_ylabel(r"$k^2P_{R_V}(k)$" if a.raw else r"$k^2P_{\delta R_V}(k)$")
axs[-1].legend(title="Map pixel size",fontsize=8)
fig.tight_layout()
tag="_raw" if a.raw else "_radial_fluctuations"
fig.savefig(a.prefix+tag+"_three_experiments.png",dpi=240,bbox_inches="tight")
fig.savefig(a.prefix+tag+"_three_experiments.pdf",bbox_inches="tight")
plt.close(fig)

# Quantify B/C difference at matched k: nonlinear mixing diagnostic
print("\nB/C power ratio (recomputed / arithmetic), k <= 0.5 kNyq:")
for B,C in zip(experiments["B_strict_recomputed"],experiments["C_strict_arithmetic"]):
    dx,kb,pb,nb,kn=B; _,kc,pc,nc,_=C
    use=(kb>0)&(kb<=.5*kn)&(pb>0)&(nb>=a.min_modes)
    pci=np.interp(kb[use],kc,pc,left=np.nan,right=np.nan)
    good=np.isfinite(pci)&(pci>0)
    ratio=pb[use][good]/pci[good]
    if ratio.size:
        q=np.percentile(ratio,[16,50,84])
        print(f"  {dx:g} kpc: p16/median/p84 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")

out={}
for key,vals in experiments.items():
    for dx,k,k2p,n,kn in vals:
        tag2=f"{round(dx*1000):.0f}pc"
        out[f"k_{key}_{tag2}"]=k; out[f"k2p_{key}_{tag2}"]=k2p
        out[f"Nmodes_{key}_{tag2}"]=n
np.savez_compressed(a.prefix+tag+"_spectra.npz",**out)
print("\nSaved three-experiment PNG/PDF and spectra NPZ.")
