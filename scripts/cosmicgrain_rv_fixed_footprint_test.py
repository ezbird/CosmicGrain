#!/usr/bin/env python3
"""
Fixed-footprint R_V power-spectrum test.

Purpose: isolate whether the resolution-dependent R_V mask/window is producing
the apparent vertical P(k) shift.

For each requested map scale:
  A) NORMAL: recompute R_V from rebinned tau_B,tau_V and use that scale's
     ordinary R_V signal mask.
  B) FIXED FOOTPRINT: use the SAME physical footprint at every scale, defined
     once on the native map. The native footprint is converted to a coverage
     fraction in each coarse pixel; pixels are retained if coverage >= threshold.
     No new regions become valid merely because the map was coarse-grained.

Crucially, this does NOT require every native subpixel to be valid (which made
the previous strict test collapse at coarse scales). Instead it holds the
physical footprint fixed and carries its fractional coverage forward.

R_V is always recomputed correctly from summed optical depths:
    R_V = sum(tau_V) / [sum(tau_B) - sum(tau_V)]
"""

import argparse, numpy as np, matplotlib.pyplot as plt

def block_sum(a,f):
    ny,nx=a.shape
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def rv(tb,tv):
    d=tb-tv
    r=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    r[q]=tv[q]/d[q]
    return r

def taper(mask,npix=2):
    # Same simple cosine-ish edge taper concept used in our R_V spectra.
    # Distance-to-invalid via repeated erosion without scipy dependency.
    m=mask.astype(bool)
    w=m.astype(float)
    if npix<=0:return w
    cur=m.copy()
    for j in range(npix):
        up=np.zeros_like(cur);up[1:]=cur[:-1]
        dn=np.zeros_like(cur);dn[:-1]=cur[1:]
        le=np.zeros_like(cur);le[:,1:]=cur[:,:-1]
        ri=np.zeros_like(cur);ri[:,:-1]=cur[:,1:]
        er=cur&up&dn&le&ri
        edge=cur&~er
        w[edge]=min(w[edge].min(initial=1), (j+1)/(npix+1))
        cur=er
    return w

def spectrum(vals,mask,dx,taper_px=2,min_modes=8):
    w=taper(mask,taper_px)
    if w.sum()==0:return np.array([]),np.array([]),np.array([])
    mu=np.sum(w[mask]*vals[mask])/np.sum(w[mask])
    g=np.zeros_like(vals,float);g[mask]=vals[mask]-mu;g*=w
    F=dx**2*np.fft.fftshift(np.fft.fft2(g))
    Aeff=dx**2*np.sum(w*w)
    P2=np.abs(F)**2/Aeff
    ny,nx=vals.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    X,Y=np.meshgrid(kx,ky);K=np.hypot(X,Y)
    kf=2*np.pi/min(nx*dx,ny*dx);kn=np.pi/dx
    ed=np.arange(.5*kf,kn+1.5*kf,kf)
    ks=[];ps=[];ns=[]
    for lo,hi in zip(ed[:-1],ed[1:]):
        q=(K>=lo)&(K<hi)
        if q.sum()>=min_modes:
            ks.append(K[q].mean());ps.append(P2[q].mean());ns.append(q.sum())
    return np.asarray(ks),np.asarray(ps),np.asarray(ns)

def label(dx):return f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"

p=argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1.,2.])
p.add_argument("--signal-frac",type=float,default=1e-3)
p.add_argument("--coverage",type=float,default=.25,
               help="minimum fraction of native-valid area required in a coarse fixed-footprint pixel")
p.add_argument("--taper-pixels",type=int,default=2)
p.add_argument("--prefix",default="halo441_rv_fixed_footprint")
a=p.parse_args()

z=np.load(a.npz)
tb0=np.asarray(z["tau_b"],float);tv0=np.asarray(z["tau_v"],float)
nc0=np.asarray(z["dust_count"],int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])
r0=rv(tb0,tv0)
native=np.isfinite(r0)&(nc0>=1)&(tv0>=a.signal_frac*np.nanmax(tv0))

print("="*88)
print("FIXED-PHYSICAL-FOOTPRINT R_V POWER TEST")
print("="*88)
print(f"native dx={dx0:g} kpc; native Rv footprint={native.sum()}/{native.size} ({100*native.mean():.2f}%)")
print(f"fixed-footprint coarse acceptance: native-valid coverage >= {a.coverage:.2f}")

normal=[];fixed=[]
for dx in a.pixel_sizes:
    f=int(round(dx/dx0))
    if not np.isclose(dx/dx0,f):raise ValueError("pixel sizes must be integer multiples of native")
    tb=block_sum(tb0,f);tv=block_sum(tv0,f);nc=block_sum(nc0,f);rr=rv(tb,tv)

    # ordinary scale-dependent mask
    mn=np.isfinite(rr)&(nc>=1)&(tv>=a.signal_frac*np.nanmax(tv))

    # SAME native physical footprint carried forward as area coverage
    cov=block_sum(native.astype(float),f)/(f*f)
    mf=np.isfinite(rr)&(cov>=a.coverage)

    kn,pn,nn=spectrum(rr,mn,dx,a.taper_pixels)
    kf,pf,nf=spectrum(rr,mf,dx,a.taper_pixels)
    normal.append((dx,kn,kn*kn*pn))
    fixed.append((dx,kf,kf*kf*pf))

    print(f"{label(dx):>6}: normal={mn.sum():5d}/{mn.size:<6d} ({100*mn.mean():5.1f}%)  "
          f"fixed={mf.sum():5d}/{mf.size:<6d} ({100*mf.mean():5.1f}%)  "
          f"coverage median={np.median(cov[mf]) if mf.any() else np.nan:.3f}")

fig,axs=plt.subplots(1,2,figsize=(11.4,4.7))
for ax,S,title in [(axs[0],normal,"Normal resolution-dependent mask"),
                   (axs[1],fixed,"Fixed physical footprint")]:
    for dx,k,y in S:
        q=(k>0)&np.isfinite(y)&(y>0)
        ax.loglog(k[q],y[q],lw=1.7,label=label(dx))
    ax.set_title(title);ax.set_xlabel(r"$k$ (kpc$^{-1}$)")
    ax.set_ylabel(r"$k^2P_{R_V}(k)$")
    ax.grid(True,which="both",alpha=.18);ax.tick_params(direction="in",which="both",top=True,right=True)
fig.suptitle("halo441 R(V) power spectrum",fontsize=15)
H,L=axs[0].get_legend_handles_labels()
fig.legend(H,L,loc="lower center",ncol=len(L),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+".pdf",bbox_inches="tight")
plt.close(fig)

# Quantify vertical shift relative to native on common k.
print("\nMedian coarse/native k^2P ratio over common k:")
for name,S in [("NORMAL",normal),("FIXED",fixed)]:
    print(name)
    _,kr,yr=S[0]
    for dx,k,y in S[1:]:
        q=(k>=kr.min())&(k<=kr.max())&(y>0)
        ref=np.interp(k[q],kr,yr,left=np.nan,right=np.nan)
        rat=y[q]/ref
        rat=rat[np.isfinite(rat)&(rat>0)]
        if len(rat):
            v=np.percentile(rat,[16,50,84])
            print(f"  {label(dx):>6}: {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}")

np.savez_compressed(a.prefix+"_spectra.npz",
    native_mask=native.astype(np.uint8),dx0=dx0,coverage_threshold=a.coverage)
print("\nSaved",a.prefix+".png/.pdf")
