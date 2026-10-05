#!/usr/bin/env python3
"""
CosmicGrain: minimum dust-particle-count test at a chosen rebinned map scale.

Default: 500 pc pixels in the central 20 x 20 kpc.
R(V) is first recomputed correctly from rebinned tau_B/tau_V. Then ONLY the
minimum dust-particle count is varied. This isolates whether an N_dust cut is
safe/useful at the proposed fiducial 500-pc science resolution.
"""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f: raise ValueError(f"{a.shape} not divisible by {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def make_rv(tb,tv):
    d=tb-tv
    rv=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    rv[q]=tv[q]/d[q]
    return rv

def taper(mask,n=2):
    m=mask.astype(bool); w=m.astype(float); cur=m.copy()
    for j in range(n):
        up=np.zeros_like(cur); up[1:]=cur[:-1]
        dn=np.zeros_like(cur); dn[:-1]=cur[1:]
        le=np.zeros_like(cur); le[:,1:]=cur[:,:-1]
        ri=np.zeros_like(cur); ri[:,:-1]=cur[:,1:]
        er=cur&up&dn&le&ri
        w[cur&~er]=(j+1)/(n+1)
        cur=er
    return w

def spectrum(rv,mask,dx,taper_pixels=2,min_modes=8):
    if mask.sum()<2: return np.array([]),np.array([])
    w=taper(mask,taper_pixels); sw=w[mask].sum()
    if sw<=0: return np.array([]),np.array([])
    mu=np.sum(w[mask]*rv[mask])/sw
    g=np.zeros_like(rv); g[mask]=rv[mask]-mu; g*=w
    F=dx**2*np.fft.fftshift(np.fft.fft2(g))
    Aeff=dx**2*np.sum(w*w)
    P2=np.abs(F)**2/Aeff
    ny,nx=rv.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    KX,KY=np.meshgrid(kx,ky); K=np.hypot(KX,KY)
    kf=2*np.pi/min(nx*dx,ny*dx); kn=np.pi/dx
    edges=np.arange(.5*kf,kn+1.5*kf,kf)
    ks=[]; ps=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.sum()>=min_modes:
            ks.append(K[q].mean()); ps.append(P2[q].mean())
    return np.asarray(ks),np.asarray(ps)

ap=argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--pixel-size",type=float,default=.5)
ap.add_argument("--square-size",type=float,default=20.)
ap.add_argument("--min-counts",nargs="+",type=int,default=[1,2,5,10,20])
ap.add_argument("--signal-frac",type=float,default=1e-3)
ap.add_argument("--taper-pixels",type=int,default=2)
ap.add_argument("--min-modes",type=int,default=8)
ap.add_argument("--prefix",default="halo441_rv_500pc_particle_count")
a=ap.parse_args()

z=np.load(a.npz)
tb0=np.asarray(z["tau_b"],float); tv0=np.asarray(z["tau_v"],float)
nc0=np.asarray(z["dust_count"],int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

N=tb0.shape[0]; nn=int(round(a.square_size/dx0))
if not np.isclose(nn*dx0,a.square_size): raise ValueError("square not aligned")
lo=(N-nn)//2; hi=lo+nn
tb0=tb0[lo:hi,lo:hi]; tv0=tv0[lo:hi,lo:hi]; nc0=nc0[lo:hi,lo:hi]

f=int(round(a.pixel_size/dx0))
if not np.isclose(f*dx0,a.pixel_size) or nn%f:
    raise ValueError("pixel size incompatible with native grid")

tb=block_sum(tb0,f); tv=block_sum(tv0,f); nc=block_sum(nc0,f)
rv=make_rv(tb,tv)
signal=np.isfinite(rv)&(tv>=a.signal_frac*np.nanmax(tv))

print("="*92)
print("R(V) POWER SPECTRUM VS MINIMUM DUST COUNT AT FIXED REBINNED RESOLUTION")
print("="*92)
print(f"region     : central {a.square_size:g} x {a.square_size:g} kpc")
print(f"pixel size : {a.pixel_size:g} kpc ({a.pixel_size*1000:g} pc)")
print(f"grid       : {rv.shape[0]} x {rv.shape[1]}")
print(f"signal-only valid pixels before N_dust cut: {signal.sum()}/{signal.size} "
      f"({100*signal.mean():.2f}%)")
print("Only N_dust,min changes below.")

results=[]
for nmin in a.min_counts:
    mask=signal&(nc>=nmin)
    vals=rv[mask]
    k,P=spectrum(rv,mask,a.pixel_size,a.taper_pixels,a.min_modes)
    y=k*k*P
    results.append((nmin,k,y,mask,vals))
    pp=np.percentile(vals,[1,16,50,84,99])
    print(f"\nN_dust >= {nmin}")
    print(f"  valid       : {mask.sum():5d}/{mask.size} ({100*mask.mean():6.2f}%)")
    print(f"  Rv mean/std : {vals.mean():.4f} / {vals.std():.4f}")
    print(f"  p01/16/50/84/99: {pp}")

nref,kref,yref,_,_=results[0]
print(f"\nPower ratios relative to N_dust >= {nref} (p16/median/p84 over common k):")
for nmin,k,y,mask,vals in results[1:]:
    q=(k>=kref.min())&(k<=kref.max())&np.isfinite(y)&(y>0)
    ref=np.interp(k[q],kref,yref,left=np.nan,right=np.nan)
    rr=y[q]/ref
    rr=rr[np.isfinite(rr)&(rr>0)]
    if len(rr):
        p=np.percentile(rr,[16,50,84])
        print(f"  N >= {nmin:2d}: {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}")

fig,axs=plt.subplots(1,2,figsize=(11.5,4.8))
for nmin,k,y,mask,vals in results:
    q=(k>0)&np.isfinite(y)&(y>0)
    axs[0].loglog(k[q],y[q],lw=1.8,label=fr"$N_{{dust}}\geq {nmin}$")

base=results[0][4]
bins=np.linspace(np.percentile(base,.5),np.percentile(base,99.5),40)
for nmin,k,y,mask,vals in results:
    axs[1].hist(vals,bins=bins,density=True,histtype="step",lw=1.8,
                label=fr"$N_{{dust}}\geq {nmin}$")

axs[0].set_title("Power spectrum")
axs[0].set_xlabel(r"$k$ (kpc$^{-1}$)")
axs[0].set_ylabel(r"$k^2P_{R_V}(k)$")
axs[1].set_title(r"$R(V)$ distribution")
axs[1].set_xlabel(r"$R(V)$"); axs[1].set_ylabel("Probability density")
for ax in axs:
    ax.grid(True,alpha=.18)
    ax.tick_params(direction="in",which="both",top=True,right=True)

fig.suptitle(f"halo441 - {a.pixel_size*1000:g} pc pixels, central "
             f"{a.square_size:g} x {a.square_size:g} kpc",fontsize=15)
H,L=axs[0].get_legend_handles_labels()
fig.legend(H,L,loc="lower center",ncol=len(L),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+".pdf",bbox_inches="tight")
plt.close(fig)

with open(a.prefix+".csv","w",newline="") as fp:
    w=csv.writer(fp)
    w.writerow(["min_dust_count","k_kpc^-1","k2P","valid_pixels","valid_fraction",
                "Rv_mean","Rv_std"])
    for nmin,k,y,mask,vals in results:
        for kk,yy in zip(k,y):
            w.writerow([nmin,kk,yy,mask.sum(),mask.mean(),vals.mean(),vals.std()])

print("\nSaved",a.prefix+".png/.pdf/.csv")
