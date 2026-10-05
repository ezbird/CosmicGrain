#!/usr/bin/env python3
"""
CosmicGrain: fixed-square + native-master-mask R_V power-spectrum test.

Within exactly the same central square (default 20x20 kpc), compare:

  A) NATURAL MASK
     At each resolution, R_V is recomputed from rebinned tau_B/tau_V and that
     resolution gets its ordinary signal/validity mask.

  B) NATIVE-MASK CONTROL
     Define the valid mask ONCE on the native 125-pc map. For every coarse
     pixel, carry forward the fraction of its native subpixels that belonged
     to that master mask. Keep the coarse pixel when coverage >= --coverage.

The outer aperture is identical in A and B. R_V is recomputed correctly from
summed optical depths in both cases. Thus the main difference is whether newly
measurable coarse regions are allowed to enter the FFT.

Also runs several coverage thresholds in the terminal (default 0.25,0.50,0.75)
so the conclusion does not hinge on one arbitrary threshold.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f:
        raise ValueError(f"shape {a.shape} not divisible by factor {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def make_rv(tb,tv):
    d=tb-tv
    r=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    r[q]=tv[q]/d[q]
    return r

def taper(mask,n=2):
    m=mask.astype(bool)
    w=m.astype(float)
    cur=m.copy()
    for j in range(n):
        up=np.zeros_like(cur); up[1:]=cur[:-1]
        dn=np.zeros_like(cur); dn[:-1]=cur[1:]
        le=np.zeros_like(cur); le[:,1:]=cur[:,:-1]
        ri=np.zeros_like(cur); ri[:,:-1]=cur[:,1:]
        er=cur&up&dn&le&ri
        edge=cur&~er
        w[edge]=(j+1)/(n+1)
        cur=er
    return w

def spectrum(vals,mask,dx,taper_pixels=2,min_modes=8):
    if mask.sum()<2:
        return np.array([]),np.array([]),np.array([])
    w=taper(mask,taper_pixels)
    sw=np.sum(w[mask])
    if sw<=0:
        return np.array([]),np.array([]),np.array([])
    mu=np.sum(w[mask]*vals[mask])/sw
    g=np.zeros_like(vals,float)
    g[mask]=vals[mask]-mu
    g*=w
    F=dx**2*np.fft.fftshift(np.fft.fft2(g))
    Aeff=dx**2*np.sum(w*w)
    P2=np.abs(F)**2/Aeff

    ny,nx=vals.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    KX,KY=np.meshgrid(kx,ky)
    K=np.hypot(KX,KY)
    kf=2*np.pi/min(nx*dx,ny*dx)
    kn=np.pi/dx
    edges=np.arange(0.5*kf,kn+1.5*kf,kf)
    ks=[]; ps=[]; nm=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.sum()>=min_modes:
            ks.append(K[q].mean())
            ps.append(P2[q].mean())
            nm.append(q.sum())
    return np.asarray(ks),np.asarray(ps),np.asarray(nm)

def label(dx):
    return f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"

def ratios(S):
    if not S: return []
    _,kr,yr,_,_=S[0]
    out=[]
    for dx,k,y,frac,covmed in S[1:]:
        q=(k>=kr.min())&(k<=kr.max())&np.isfinite(y)&(y>0)
        ref=np.interp(k[q],kr,yr,left=np.nan,right=np.nan)
        r=y[q]/ref
        r=r[np.isfinite(r)&(r>0)]
        if len(r):
            out.append((dx,*np.percentile(r,[16,50,84])))
    return out

ap=argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--square-size",type=float,default=20.0)
ap.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1.,2.])
ap.add_argument("--signal-frac",type=float,default=1e-3)
ap.add_argument("--coverage",type=float,default=.25,
                help="coverage threshold used in plotted native-mask control")
ap.add_argument("--coverage-scan",nargs="+",type=float,default=[.25,.50,.75],
                help="thresholds additionally quantified in terminal")
ap.add_argument("--taper-pixels",type=int,default=2)
ap.add_argument("--min-modes",type=int,default=8)
ap.add_argument("--prefix",default="halo441_rv_square_mastermask")
a=ap.parse_args()

z=np.load(a.npz)
tb0=np.asarray(z["tau_b"],float)
tv0=np.asarray(z["tau_v"],float)
nc0=np.asarray(z["dust_count"],int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

N=tb0.shape[0]
nn=int(round(a.square_size/dx0))
if not np.isclose(nn*dx0,a.square_size):
    raise ValueError("square size must align with native grid")
if nn>N: raise ValueError("square exceeds map")
lo=(N-nn)//2; hi=lo+nn
tb=tb0[lo:hi,lo:hi]
tv=tv0[lo:hi,lo:hi]
nc=nc0[lo:hi,lo:hi]

rv0=make_rv(tb,tv)
master=np.isfinite(rv0)&(nc>=1)&(tv>=a.signal_frac*np.nanmax(tv))

print("="*92)
print("FIXED SQUARE + NATIVE MASTER-MASK R_V POWER TEST")
print("="*92)
print(f"square = central {a.square_size:g} x {a.square_size:g} kpc")
print(f"native grid = {nn}x{nn}, dx={dx0:g} kpc")
print(f"native master mask = {master.sum()}/{master.size} ({100*master.mean():.2f}%)")
print(f"plotted native-mask coverage threshold = {a.coverage:.2f}")

natural=[]
controlled_by_cov={c:[] for c in sorted(set(a.coverage_scan+[a.coverage]))}

for dx in a.pixel_sizes:
    f=int(round(dx/dx0))
    if not np.isclose(f*dx0,dx) or nn%f:
        raise ValueError(f"{dx:g} kpc incompatible with square/native grid")
    tbb=block_sum(tb,f)
    tvv=block_sum(tv,f)
    ncc=block_sum(nc,f)
    rr=make_rv(tbb,tvv)

    natural_mask=np.isfinite(rr)&(ncc>=1)&(tvv>=a.signal_frac*np.nanmax(tvv))
    k,p,nm=spectrum(rr,natural_mask,dx,a.taper_pixels,a.min_modes)
    natural.append((dx,k,k*k*p,natural_mask.mean(),np.nan))

    coverage=block_sum(master.astype(float),f)/(f*f)

    print(f"\n{label(dx)}: natural valid {natural_mask.sum()}/{natural_mask.size} "
          f"({100*natural_mask.mean():.1f}%)")
    for c in controlled_by_cov:
        cmask=np.isfinite(rr)&(coverage>=c)
        kc,pc,nmc=spectrum(rr,cmask,dx,a.taper_pixels,a.min_modes)
        covmed=np.median(coverage[cmask]) if cmask.any() else np.nan
        controlled_by_cov[c].append((dx,kc,kc*kc*pc,cmask.mean(),covmed))
        print(f"   master coverage >= {c:.2f}: {cmask.sum():5d}/{cmask.size:<5d} "
              f"({100*cmask.mean():5.1f}%), median coverage={covmed:.3f}")

print("\nNATURAL coarse/native k^2P ratio (p16/median/p84):")
for row in ratios(natural):
    print(f"  {label(row[0]):>6}: {row[1]:.3f}/{row[2]:.3f}/{row[3]:.3f}")

for c,S in controlled_by_cov.items():
    print(f"\nMASTER-MASK coverage >= {c:.2f} coarse/native ratio:")
    for row in ratios(S):
        print(f"  {label(row[0]):>6}: {row[1]:.3f}/{row[2]:.3f}/{row[3]:.3f}")

controlled=controlled_by_cov[a.coverage]
fig,axs=plt.subplots(1,2,figsize=(11.4,4.8))
for ax,S,title in [
    (axs[0],natural,"Natural internal mask"),
    (axs[1],controlled,f"125 pc master mask (coverage >= {a.coverage:.2f})")
]:
    for dx,k,y,frac,covmed in S:
        q=(k>0)&np.isfinite(y)&(y>0)
        ax.loglog(k[q],y[q],lw=1.7,label=label(dx))
    ax.set_title(title)
    ax.set_xlabel(r"$k$ (kpc$^{-1}$)")
    ax.set_ylabel(r"$k^2P_{R_V}(k)$")
    ax.grid(True,which="both",alpha=.18)
    ax.tick_params(direction="in",which="both",top=True,right=True)

fig.suptitle(f"halo441 R(V) power spectrum - central {a.square_size:g} x {a.square_size:g} kpc",
             fontsize=15)
H,L=axs[0].get_legend_handles_labels()
fig.legend(H,L,loc="lower center",ncol=len(L),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+".pdf",bbox_inches="tight")
plt.close(fig)

np.savez_compressed(a.prefix+"_masks.npz",
                    master_mask=master.astype(np.uint8),
                    native_dx=dx0,
                    square_size=a.square_size,
                    plotted_coverage=a.coverage)

print("\nInterpretation:")
print("Left and right use exactly the same outer square and the same rebinned R_V values.")
print("Only the rule deciding which coarse pixels enter the FFT is changed.")
print("If the staircase largely collapses on the right, newly admitted coarse pixels")
print("from the resolution-dependent internal mask are the main source of the shift.")
print("\nSaved",a.prefix+".png/.pdf")
