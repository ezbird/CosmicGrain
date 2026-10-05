#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def bsum(x,f):
    n,m=x.shape
    return x.reshape(n//f,f,m//f,f).sum((1,3))
def rvfun(tb,tv):
    d=tb-tv; r=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0); r[q]=tv[q]/d[q]; return r
def win(mask,t):
    if t<=0:return mask.astype(float)
    u=np.clip(distance_transform_edt(mask)/t,0,1)
    w=np.sin(np.pi*u/2)**2; w[~mask]=0; return w
def ps(field,dx,mask,taper=0):
    mask=mask&np.isfinite(field); w=win(mask,taper)
    vals=np.where(mask,field,0.); mu=np.sum(w*vals)/np.sum(w)
    d=np.zeros_like(field); d[mask]=field[mask]-mu
    F=dx**2*np.fft.fftshift(np.fft.fft2(w*d))
    P=np.abs(F)**2/(dx**2*np.sum(w*w))
    ny,nx=field.shape
    kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx))
    K=np.hypot(*np.meshgrid(kx,ky)); kf=2*np.pi/min(nx*dx,ny*dx); kn=np.pi/dx
    edges=np.arange(.5*kf,kn+1.5*kf,kf); ks=[]; pp=[]; nn=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        q=(K>=lo)&(K<hi)
        if q.any():ks.append(K[q].mean());pp.append(P[q].mean());nn.append(q.sum())
    return np.array(ks),np.array(pp),np.array(nn),kn
def draw(ax,k,y,n,kn,label,minm):
    g=(n>=minm)&(k>0)&(y>0)&np.isfinite(y); s=g&(k<=.5*kn); d=g&(k>.5*kn)
    line,=ax.loglog(k[s],y[s],lw=1.8,label=label); c=line.get_color()
    ids=np.where(d)[0]
    if len(ids):
        use=ids
        if ids[0]>0 and g[ids[0]-1]:use=np.r_[ids[0]-1,ids]
        ax.loglog(k[use],y[use],ls="--",lw=1.35,color=c,alpha=.9)

ap=argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1,2])
ap.add_argument("--min-count",type=int,default=1); ap.add_argument("--signal-frac",type=float,default=1e-3)
ap.add_argument("--taper-pixels",type=int,default=2); ap.add_argument("--min-modes",type=int,default=8)
ap.add_argument("--prefix",default="halo441_best_snap029_rv_dust_ps_comparison")
a=ap.parse_args(); z=np.load(a.npz)
tb0=z["tau_b"].astype(float);tv0=z["tau_v"].astype(float);dm0=z["dust_mass_code"].astype(float);nc0=z["dust_count"].astype(int)
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0]); sigmean=np.mean(dm0/dx0**2)
R=[];D=[]
print("="*80);print("R_V + DUST-DENSITY POWER-SPECTRUM COMPARISON");print("="*80)
for dx in a.pixel_sizes:
    f=int(round(dx/dx0))
    if not np.isclose(dx/dx0,f):raise ValueError("pixel sizes must be integer multiples of native dx")
    tb,tv,dm,nc=bsum(tb0,f),bsum(tv0,f),bsum(dm0,f),bsum(nc0,f)
    rv=rvfun(tb,tv); mask=np.isfinite(rv)&(nc>=a.min_count)&(tv>=a.signal_frac*np.nanmax(tv))
    k,p,n,kn=ps(rv,dx,mask,a.taper_pixels);R.append((dx,k,k*k*p,n,kn))
    sig=dm/dx**2; dd=(sig-sigmean)/sigmean
    k,p,n,kn=ps(dd,dx,np.ones_like(dd,dtype=bool),0);D.append((dx,k,k*k*p,n,kn))
    print(f"{dx:g} kpc: Rv valid {mask.sum()}/{mask.size} ({100*mask.mean():.2f}%), dM/M={dm.sum()/dm0.sum()-1:+.2e}, kNyq={kn:.4f}")

print("\nDust P ratios vs native, k <= 0.5 coarse kNyq (p16/median/p84)")
kr,pr= D[0][1],D[0][2]
for dx,k,y,n,kn in D[1:]:
    q=(k<=.5*kn)&(n>=a.min_modes)&(y>0)
    ref=np.interp(k[q],kr,pr,left=np.nan,right=np.nan); rr=y[q]/ref;rr=rr[np.isfinite(rr)&(rr>0)]
    if len(rr):
        v=np.percentile(rr,[16,50,84]);print(f"{dx:g} kpc: {v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}")

fig,axs=plt.subplots(1,2,figsize=(11.6,4.7))
for ax,S in zip(axs,[R,D]):
    for dx,k,y,n,kn in S:
        lab=f"{dx*1000:.0f} pc" if dx<1 else f"{dx:g} kpc"
        draw(ax,k,y,n,kn,lab,a.min_modes)
    ax.set_xlabel(r"$k$ (kpc$^{-1}$)");ax.grid(True,which="both",alpha=.18);ax.tick_params(direction="in",which="both",top=True,right=True)
axs[0].set_title(r"$R_V$");axs[1].set_title("Dust surface density")
axs[0].set_ylabel(r"$k^2P_{R_V}(k)$");axs[1].set_ylabel(r"$k^2P_{\delta_{\rm dust}}(k)$")
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,title="Map pixel size",loc="upper center",bbox_to_anchor=(.5,1.015),ncol=len(l),frameon=False)
fig.text(.5,.012,r"Solid: $k\leq0.5\,k_{\rm Nyq}$     Dashed: $0.5\,k_{\rm Nyq}<k\leq k_{\rm Nyq}$",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.055,1,.91))
fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight");fig.savefig(a.prefix+".pdf",bbox_inches="tight");plt.close(fig)
out={}
for stem,S in [("rv",R),("dust",D)]:
    for dx,k,y,n,kn in S:
        tag=f"{round(dx*1000):.0f}pc";out[f"k_{stem}_{tag}"]=k;out[f"k2p_{stem}_{tag}"]=y;out[f"Nmodes_{stem}_{tag}"]=n;out[f"knyq_{stem}_{tag}"]=np.array(kn)
np.savez_compressed(a.prefix+"_spectra.npz",**out)
print("\nSaved "+a.prefix+".png/.pdf and spectra NPZ")
