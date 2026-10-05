#!/usr/bin/env python3
"""
CosmicGrain fixed-square R_V power-spectrum test.

Uses the real native tau_B/tau_V maps. For each requested square centered on
the map center, extracts EXACTLY that physical square, rebins it to each map
pixel size, recomputes R_V from summed optical depths, applies the ordinary
R_V validity criterion within that square, and computes k^2 P_RV(k).

This tests whether simply holding the outer physical region fixed removes the
resolution-dependent P(k) shift. It does NOT force the internal validity mask
to be identical; the script prints that mask fraction explicitly.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def block_sum(x,f):
    ny,nx=x.shape
    if ny%f or nx%f: raise ValueError(f"{x.shape} not divisible by {f}")
    return x.reshape(ny//f,f,nx//f,f).sum((1,3))

def make_rv(tb,tv):
    d=tb-tv
    r=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    r[q]=tv[q]/d[q]
    return r

def taper(mask,n=2):
    # simple inward edge taper; zero outside mask
    m=mask.astype(bool); w=m.astype(float)
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

def power(rv,mask,dx,taper_pixels=2,min_modes=8):
    w=taper(mask,taper_pixels)
    if mask.sum()<2 or np.sum(w[mask])<=0:
        return np.array([]),np.array([]),np.array([])
    mu=np.sum(w[mask]*rv[mask])/np.sum(w[mask])
    g=np.zeros_like(rv,float)
    g[mask]=rv[mask]-mu
    g*=w
    F=dx**2*np.fft.fftshift(np.fft.fft2(g))
    Aeff=dx**2*np.sum(w*w)
    P2=np.abs(F)**2/Aeff

    ny,nx=rv.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    X,Y=np.meshgrid(kx,ky); K=np.hypot(X,Y)
    kf=2*np.pi/min(nx*dx,ny*dx)
    kn=np.pi/dx
    edges=np.arange(0.5*kf,kn+1.5*kf,kf)
    ks=[]; ps=[]; nm=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.sum()>=min_modes:
            ks.append(K[q].mean()); ps.append(P2[q].mean()); nm.append(q.sum())
    return np.asarray(ks),np.asarray(ps),np.asarray(nm)

def label(dx):
    return f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"

ap=argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--square-sizes",nargs="+",type=float,default=[10,20],
                help="full side lengths in kpc")
ap.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1,2])
ap.add_argument("--signal-frac",type=float,default=1e-3)
ap.add_argument("--taper-pixels",type=int,default=2)
ap.add_argument("--min-modes",type=int,default=8)
ap.add_argument("--prefix",default="halo441_rv_fixed_squares")
a=ap.parse_args()

z=np.load(a.npz)
tb0=np.asarray(z["tau_b"],float)
tv0=np.asarray(z["tau_v"],float)
nc0=np.asarray(z["dust_count"],int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

if tb0.shape[0]!=tb0.shape[1]: raise ValueError("expected square native map")
N=tb0.shape[0]
print("="*88)
print("FIXED CONTIGUOUS SQUARE R_V POWER TEST")
print("="*88)
print(f"native map {N}x{N}, dx={dx0:g} kpc, full width={N*dx0:g} kpc")

results={}
for L in a.square_sizes:
    n_native=int(round(L/dx0))
    if not np.isclose(n_native*dx0,L): raise ValueError(f"{L} kpc not aligned with native grid")
    if n_native>N: raise ValueError(f"{L} kpc square exceeds map")
    lo=(N-n_native)//2; hi=lo+n_native
    tb=tb0[lo:hi,lo:hi]; tv=tv0[lo:hi,lo:hi]; nc=nc0[lo:hi,lo:hi]
    print(f"\n--- CENTRAL {L:g} x {L:g} kpc ---")
    print(f"native indices [{lo}:{hi}], exactly {n_native}x{n_native} native pixels")
    S=[]
    for dx in a.pixel_sizes:
        f=int(round(dx/dx0))
        if not np.isclose(f*dx0,dx): raise ValueError(f"{dx} not integer multiple of native dx")
        if n_native%f: 
            print(f"{label(dx):>6}: skipped; square not divisible by factor {f}")
            continue
        tbb=block_sum(tb,f); tvv=block_sum(tv,f); ncc=block_sum(nc,f)
        rr=make_rv(tbb,tvv)
        # Ordinary Rv measurement rule, but outer square is identical physically.
        mask=np.isfinite(rr)&(ncc>=1)&(tvv>=a.signal_frac*np.nanmax(tvv))
        k,P,nm=power(rr,mask,dx,a.taper_pixels,a.min_modes)
        y=k*k*P
        vals=rr[mask]
        S.append((dx,k,y,mask.mean(),np.nanmedian(vals),np.nanstd(vals)))
        print(f"{label(dx):>6}: grid={rr.shape[0]:3d}x{rr.shape[1]:3d}, "
              f"valid={mask.sum():5d}/{mask.size:<5d} ({100*mask.mean():5.1f}%), "
              f"Rv median={np.median(vals):6.3f}, std={np.std(vals):6.3f}")
    results[L]=S

    # Quantify shifts relative to finest curve, only where both exist.
    if S:
        _,kr,yr,_,_,_=S[0]
        print("coarse/native k^2P ratio over common k (p16/median/p84):")
        for dx,k,y,_,_,_ in S[1:]:
            q=(k>=kr.min())&(k<=kr.max())&np.isfinite(y)&(y>0)
            ref=np.interp(k[q],kr,yr,left=np.nan,right=np.nan)
            r=y[q]/ref
            r=r[np.isfinite(r)&(r>0)]
            if len(r):
                v=np.percentile(r,[16,50,84])
                print(f"  {label(dx):>6}: {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}")

# One panel per requested square
fig,axs=plt.subplots(1,len(a.square_sizes),figsize=(5.7*len(a.square_sizes),4.8),squeeze=False)
for ax,L in zip(axs[0],a.square_sizes):
    for dx,k,y,frac,med,std in results[L]:
        q=(k>0)&np.isfinite(y)&(y>0)
        ax.loglog(k[q],y[q],lw=1.7,label=label(dx))
    ax.set_title(f"Central {L:g} x {L:g} kpc")
    ax.set_xlabel(r"$k$ (kpc$^{-1}$)")
    ax.set_ylabel(r"$k^2P_{R_V}(k)$")
    ax.grid(True,which="both",alpha=.18)
    ax.tick_params(direction="in",which="both",top=True,right=True)
fig.suptitle("halo441 R(V) power spectrum - fixed square region",fontsize=15)
H,Lb=axs[0,0].get_legend_handles_labels()
fig.legend(H,Lb,loc="lower center",ncol=len(Lb),frameon=False)
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight")
fig.savefig(a.prefix+".pdf",bbox_inches="tight")
plt.close(fig)

print("\nIMPORTANT: the outer square is identical at every resolution, but the")
print("internal R_V validity mask is still allowed to change. The printed valid")
print("fractions tell us whether a shift survives solely because of that mask.")
print("Saved",a.prefix+".png/.pdf")
