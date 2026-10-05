#!/usr/bin/env python3
import argparse,numpy as np,matplotlib.pyplot as plt

def scalar(d,names):
    for n in names:
        if n in d:return float(np.asarray(d[n]).ravel()[0])
def dx_native(d):
    x=scalar(d,["pixel_size_pkpc","pixel_size","dx_pkpc"])
    if x:return x
    for n in ["x_centers","x_centers_pkpc","x","xc"]:
        if n in d:
            a=np.asarray(d[n]).ravel()
            if len(a)>1:return float(np.median(np.diff(a)))
    raise RuntimeError("Cannot determine native pixel size")
def rebin_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f:raise ValueError("shape not divisible by rebin factor")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))
def ps(a,dx):
    n=a.shape[0]; L=n*dx; q=a-a.mean()
    F=dx**2*np.fft.fft2(q); P=np.abs(F)**2/L**2
    kx=2*np.pi*np.fft.fftfreq(n,dx)
    KX,KY=np.meshgrid(kx,kx); K=np.hypot(KX,KY)
    dk=2*np.pi/L; kn=np.pi/dx
    edges=np.arange(.5*dk,kn+1.5*dk,dk)
    ko=[];po=[];no=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(K>=lo)&(K<hi)
        if m.any():ko.append(K[m].mean());po.append(P[m].mean());no.append(m.sum())
    return np.array(ko),np.array(po),np.array(no),dk,kn

p=argparse.ArgumentParser(description="Linear dust-density PS null test")
p.add_argument("npz")
p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1,2])
p.add_argument("--prefix",default="dust_density_power_sanity")
a=p.parse_args()
d=np.load(a.npz,allow_pickle=True)
if "dust_mass_code" not in d:raise KeyError("NPZ lacks dust_mass_code")
m0=np.asarray(d["dust_mass_code"],float); dx0=dx_native(d)
print("="*76);print("DUST SURFACE-DENSITY POWER-SPECTRUM SANITY TEST")
print("="*76);print("native:",m0.shape,f"dx={dx0:g} pkpc")
print("No R_V signal mask; empty cells remain physical Sigma_dust=0.\n")
res=[]
for dx in a.pixel_sizes:
    rat=dx/dx0; f=int(round(rat))
    if not np.isclose(rat,f):raise ValueError(f"{dx} not integer multiple of {dx0}")
    mm=m0 if f==1 else rebin_sum(m0,f)
    sig=mm/dx**2
    k,P,n,dk,kn=ps(sig,dx);res.append((dx,k,P,n,kn))
    dm=mm.sum()/m0.sum()-1 if m0.sum() else np.nan
    print(f"{dx*1000:5.0f} pc  shape={str(mm.shape):12s} "
          f"dM/M={dm:+.3e} occupied={100*np.count_nonzero(mm)/mm.size:6.2f}% "
          f"kNyq={kn:.4f}")
fig,ax=plt.subplots(figsize=(7.2,5.2))
for dx,k,P,n,kn in res:
    m=(k>0)&(P>0)
    lab=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"
    ax.loglog(k[m],k[m]**2*P[m],label=lab)
ax.set_xlabel(r"$k$ (kpc$^{-1}$)");ax.set_ylabel(r"$k^2P_{\Sigma_{\rm dust}}(k)$")
ax.legend(title="Map pixel size");fig.tight_layout()
fig.savefig(a.prefix+"_k2p.png",dpi=240,bbox_inches="tight")
fig.savefig(a.prefix+"_k2p.pdf",bbox_inches="tight");plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,5.2))
for dx,k,P,n,kn in res:
    m=(k>0)&(P>0)
    lab=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"
    ax.loglog(k[m],P[m],label=lab)
ax.set_xlabel(r"$k$ (kpc$^{-1}$)");ax.set_ylabel(r"$P_{\Sigma_{\rm dust}}(k)$")
ax.legend(title="Map pixel size");fig.tight_layout()
fig.savefig(a.prefix+"_pk.png",dpi=240,bbox_inches="tight");plt.close(fig)
out={}
for dx,k,P,n,kn in res:
    t=f"{round(dx*1000):.0f}pc";out["k_"+t]=k;out["P_"+t]=P;out["Nmodes_"+t]=n
np.savez_compressed(a.prefix+"_spectra.npz",**out)
print("\nOverlap ratios versus finest map, restricted to k <= 0.5 kNyq(coarse):")
dxF,kF,PF,_,_=res[0]
for dx,k,P,_,kn in res[1:]:
    u=(kF>0)&(kF<=.5*kn)&(PF>0)
    pi=np.interp(kF[u],k,P,left=np.nan,right=np.nan)
    g=np.isfinite(pi)&(pi>0)
    rr=pi[g]/PF[u][g]
    if len(rr):
        q=np.percentile(rr,[16,50,84])
        print(f"{dx:g} kpc: P ratio p16/median/p84 = {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}")
print("\nSaved k2p PNG/PDF, P(k) PNG, and spectra NPZ.")
