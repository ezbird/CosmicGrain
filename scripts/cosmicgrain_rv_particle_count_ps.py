#!/usr/bin/env python3
"""
CosmicGrain: isolate the effect of minimum dust-particle count on the R(V)
power spectrum at ONE fixed map resolution.

Default experiment:
  - native 125 pc map
  - exact central 20 x 20 kpc square
  - identical R(V) values for every test
  - same optical-depth signal criterion
  - vary only N_dust,min = 1, 2, 5, 10

This cleanly tests how the particle-count validity cut changes P_RV(k).
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def make_rv(tb, tv):
    d = tb - tv
    rv = np.full_like(tv, np.nan, dtype=float)
    q = np.isfinite(tb) & np.isfinite(tv) & (tv > 0) & (d > 0)
    rv[q] = tv[q] / d[q]
    return rv

def taper(mask, n=2):
    m = mask.astype(bool)
    w = m.astype(float)
    cur = m.copy()
    for j in range(n):
        up=np.zeros_like(cur); up[1:]=cur[:-1]
        dn=np.zeros_like(cur); dn[:-1]=cur[1:]
        le=np.zeros_like(cur); le[:,1:]=cur[:,:-1]
        ri=np.zeros_like(cur); ri[:,:-1]=cur[:,1:]
        er = cur & up & dn & le & ri
        edge = cur & ~er
        w[edge] = (j+1)/(n+1)
        cur = er
    return w

def spectrum(rv, mask, dx, taper_pixels=2, min_modes=8):
    w = taper(mask, taper_pixels)
    sw = np.sum(w[mask])
    if mask.sum() < 2 or sw <= 0:
        return np.array([]), np.array([])

    mu = np.sum(w[mask] * rv[mask]) / sw
    g = np.zeros_like(rv, float)
    g[mask] = rv[mask] - mu
    g *= w

    F = dx**2 * np.fft.fftshift(np.fft.fft2(g))
    Aeff = dx**2 * np.sum(w*w)
    P2 = np.abs(F)**2 / Aeff

    ny,nx = rv.shape
    kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    KX,KY = np.meshgrid(kx,ky)
    K = np.hypot(KX,KY)

    kf = 2*np.pi/min(nx*dx,ny*dx)
    kn = np.pi/dx
    edges = np.arange(0.5*kf, kn+1.5*kf, kf)

    ks=[]; ps=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.sum() >= min_modes:
            ks.append(K[q].mean())
            ps.append(P2[q].mean())
    return np.asarray(ks), np.asarray(ps)

ap=argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--square-size",type=float,default=20.0)
ap.add_argument("--min-counts",nargs="+",type=int,default=[1,2,5,10])
ap.add_argument("--signal-frac",type=float,default=1e-3)
ap.add_argument("--taper-pixels",type=int,default=2)
ap.add_argument("--min-modes",type=int,default=8)
ap.add_argument("--prefix",default="halo441_rv_particle_count_ps")
a=ap.parse_args()

z=np.load(a.npz)
tb=np.asarray(z["tau_b"],float)
tv=np.asarray(z["tau_v"],float)
nc=np.asarray(z["dust_count"],int)
dx=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

N=tb.shape[0]
nn=int(round(a.square_size/dx))
if not np.isclose(nn*dx,a.square_size):
    raise ValueError("square size must align with native grid")
if nn>N:
    raise ValueError("square exceeds map")

lo=(N-nn)//2
hi=lo+nn
tb=tb[lo:hi,lo:hi]
tv=tv[lo:hi,lo:hi]
nc=nc[lo:hi,lo:hi]
rv=make_rv(tb,tv)

# IMPORTANT: signal threshold is fixed once for this map. Only N_dust changes.
signal = np.isfinite(rv) & (tv >= a.signal_frac*np.nanmax(tv))

print("="*92)
print("R(V) POWER SPECTRUM VS MINIMUM DUST-PARTICLE COUNT")
print("="*92)
print(f"region        : central {a.square_size:g} x {a.square_size:g} kpc")
print(f"pixel size    : {dx:g} kpc ({dx*1000:g} pc)")
print(f"grid          : {nn} x {nn}")
print(f"signal cut    : tauV >= {a.signal_frac:g} * peak tauV")
print("Only N_dust,min changes between curves.")

results=[]
rows=[]
for nmin in a.min_counts:
    mask=signal & (nc>=nmin)
    vals=rv[mask]
    k,P=spectrum(rv,mask,dx,a.taper_pixels,a.min_modes)
    y=k*k*P
    results.append((nmin,k,y,mask,vals))

    p=np.percentile(vals,[1,16,50,84,99])
    print(f"\nN_dust >= {nmin}")
    print(f"  valid       : {mask.sum():6d}/{mask.size} ({100*mask.mean():6.2f}%)")
    print(f"  Rv mean/std : {np.mean(vals):.4f} / {np.std(vals):.4f}")
    print(f"  p01/16/50/84/99: {p}")

# Ratios to N>=1 on common k.
nref,kref,yref,_,_=results[0]
print(f"\nPower ratios relative to N_dust >= {nref} (p16/median/p84 over common k):")
for nmin,k,y,mask,vals in results[1:]:
    q=(k>=kref.min())&(k<=kref.max())&np.isfinite(y)&(y>0)
    ref=np.interp(k[q],kref,yref,left=np.nan,right=np.nan)
    ratio=y[q]/ref
    ratio=ratio[np.isfinite(ratio)&(ratio>0)]
    if len(ratio):
        pp=np.percentile(ratio,[16,50,84])
        print(f"  N >= {nmin:2d}: {pp[0]:.3f}/{pp[1]:.3f}/{pp[2]:.3f}")

# Save spectral points.
csvname=a.prefix+".csv"
with open(csvname,"w",newline="") as fp:
    w=csv.writer(fp)
    w.writerow(["min_dust_count","k_kpc^-1","k2P","valid_pixels","valid_fraction",
                "Rv_mean","Rv_std"])
    for nmin,k,y,mask,vals in results:
        for kk,yy in zip(k,y):
            w.writerow([nmin,kk,yy,int(mask.sum()),mask.mean(),
                        np.mean(vals),np.std(vals)])

# Figure: spectrum + histogram so we can see the two effects together.
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8))

for nmin,k,y,mask,vals in results:
    q=(k>0)&np.isfinite(y)&(y>0)
    axs[0].loglog(k[q],y[q],lw=1.8,label=fr"$N_{{dust}}\geq {nmin}$")

# common histogram bins to make width/tail changes visually honest
allvals=results[0][4]
lohist=np.percentile(allvals,0.5)
hihist=np.percentile(allvals,99.5)
bins=np.linspace(lohist,hihist,45)
for nmin,k,y,mask,vals in results:
    axs[1].hist(vals,bins=bins,density=True,histtype="step",lw=1.8,
                label=fr"$N_{{dust}}\geq {nmin}$")

axs[0].set_title("Power spectrum")
axs[0].set_xlabel(r"$k$ (kpc$^{-1}$)")
axs[0].set_ylabel(r"$k^2P_{R_V}(k)$")
axs[1].set_title(r"$R(V)$ distribution")
axs[1].set_xlabel(r"$R(V)$")
axs[1].set_ylabel("Probability density")

for ax in axs:
    ax.grid(True,alpha=.18)
    ax.tick_params(direction="in",which="both",top=True,right=True)

fig.suptitle(f"halo441 - {dx*1000:g} pc pixels, central {a.square_size:g} x {a.square_size:g} kpc",
             fontsize=15)
H,L=axs[0].get_legend_handles_labels()
fig.legend(H,L,loc="lower center",ncol=len(L),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+".pdf",bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(" ",a.prefix+".png")
print(" ",a.prefix+".pdf")
print(" ",csvname)
