#!/usr/bin/env python3
"""
cosmicgrain_rv_grain_correlations.py

Diagnose what drives spatial R_V variations in a CosmicGrain master map.

Measures pixel-by-pixel relationships between R_V and:
  * small-grain dust-mass fraction
  * mass-weighted mean grain radius
  * mass-weighted carbon fraction

For each relation it reports:
  * raw Spearman rank correlation
  * radius-controlled (radial-detrended) Spearman correlation

Radial detrending subtracts the median of each quantity in galactocentric
annuli, so the residual correlation asks whether pixels at similar radius
still show a relationship.

Example:
  python3 cosmicgrain_rv_grain_correlations.py \
    halo441_snap029_rv_master_250pc_maps.npz \
    --min-count 1 --signal-frac 1e-3 \
    --radial-bin-width 1.0 \
    --prefix halo441_snap029_rv_graincorr
"""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def radial_residual(a, mask, dx, dr):
    ny,nx=a.shape
    x=(np.arange(nx)-(nx-1)/2)*dx
    y=(np.arange(ny)-(ny-1)/2)*dx
    X,Y=np.meshgrid(x,y)
    rr=np.hypot(X,Y)
    ib=np.floor(rr/dr).astype(int)
    out=np.full_like(a,np.nan,dtype=float)
    radial=np.full_like(a,np.nan,dtype=float)
    for b in np.unique(ib[mask]):
        q=mask&(ib==b)&np.isfinite(a)
        if q.any():
            med=np.nanmedian(a[q])
            radial[q]=med
            out[q]=a[q]-med
    return out,radial,rr

def rho(x,y):
    g=np.isfinite(x)&np.isfinite(y)
    if g.sum()<3:return np.nan,np.nan,int(g.sum())
    r,p=spearmanr(x[g],y[g])
    return float(r),float(p),int(g.sum())

def percentile_text(a):
    q=np.nanpercentile(a,[16,50,84])
    return q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--min-count",type=int,default=1)
    ap.add_argument("--signal-frac",type=float,default=1e-3)
    ap.add_argument("--radial-bin-width",type=float,default=1.0)
    ap.add_argument("--prefix",default="rv_grain_correlations")
    a=ap.parse_args()

    z=np.load(a.npz)
    need=["rv","tau_v","dust_count","pixel_size_pkpc",
          "mean_grain_radius_massweighted_nm",
          "small_grain_mass_fraction",
          "carbon_mass_fraction_massweighted"]
    miss=[k for k in need if k not in z]
    if miss: raise KeyError(f"NPZ missing required fields: {miss}")

    rv=z["rv"].astype(float)
    tv=z["tau_v"].astype(float)
    nc=z["dust_count"].astype(int)
    dx=float(z["pixel_size_pkpc"])
    peak=np.nanmax(tv)
    threshold=a.signal_frac*peak
    base=np.isfinite(rv)&(nc>=a.min_count)&(tv>=threshold)

    props=[
      ("Small-grain mass fraction",z["small_grain_mass_fraction"].astype(float),
       r"$f_{\rm small}$","small_fraction"),
      ("Mass-weighted mean grain radius",z["mean_grain_radius_massweighted_nm"].astype(float),
       r"$\langle a\rangle_M$ [nm]","mean_radius"),
      ("Mass-weighted carbon fraction",z["carbon_mass_fraction_massweighted"].astype(float),
       r"$f_{\rm C}$","carbon_fraction")
    ]

    drv,rvtrend,rr=radial_residual(rv,base,dx,a.radial_bin_width)

    print("="*78)
    print("COSMICGRAIN R_V — GRAIN-PROPERTY CORRELATIONS")
    print("="*78)
    print(f"Input: {a.npz}")
    print(f"Pixel size: {dx:g} pkpc")
    print(f"Mask: N_dust >= {a.min_count}, tauV >= {a.signal_frac:g} * peak")
    print(f"Valid R_V pixels: {base.sum():,}/{base.size:,} ({100*base.mean():.2f}%)")
    print(f"Radial detrending: median in {a.radial_bin_width:g}-pkpc annuli\n")

    rows=[]
    residuals={}
    for name,p,label,key in props:
        m=base&np.isfinite(p)
        dp,ptrend,_=radial_residual(p,m,dx,a.radial_bin_width)
        raw_r,raw_p,nraw=rho(rv[m],p[m])
        both=m&np.isfinite(drv)&np.isfinite(dp)
        det_r,det_p,ndet=rho(drv[both],dp[both])
        q=percentile_text(p[m])
        rows.append((name,nraw,raw_r,raw_p,ndet,det_r,det_p,*q))
        residuals[key]=(p,dp,m,both,label)
        print(name+":")
        print(f"  property p16/median/p84 = {q[0]:.5g} / {q[1]:.5g} / {q[2]:.5g}")
        print(f"  raw:       Spearman rho = {raw_r:+.4f}  p={raw_p:.3e}  N={nraw:,}")
        print(f"  detrended: Spearman rho = {det_r:+.4f}  p={det_p:.3e}  N={ndet:,}\n")

    # Six-panel raw/residual scatter figure.
    fig,axs=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for j,(name,p,label,key) in enumerate(props):
        _,dp,m,both,_=residuals[key]
        raw_r=rows[j][2]; det_r=rows[j][5]
        axs[0,j].scatter(p[m],rv[m],s=6,alpha=.28,rasterized=True)
        axs[0,j].set_xlabel(label); axs[0,j].set_ylabel(r"$R_V$")
        axs[0,j].set_title(f"{name}\nraw Spearman $\\rho={raw_r:+.2f}$")
        axs[0,j].grid(alpha=.2)
        axs[1,j].scatter(dp[both],drv[both],s=6,alpha=.28,rasterized=True)
        axs[1,j].axhline(0,lw=.8,alpha=.5);axs[1,j].axvline(0,lw=.8,alpha=.5)
        if key=="mean_radius":
            xlab=r"$\delta\langle a\rangle_M$ [nm]"
        elif key=="small_fraction":
            xlab=r"$\delta f_{\rm small}$"
        else:
            xlab=r"$\delta f_{\rm C}$"
        axs[1,j].set_xlabel(xlab);axs[1,j].set_ylabel(r"$\delta R_V$")
        axs[1,j].set_title(f"radial residuals, $\\rho={det_r:+.2f}$")
        axs[1,j].grid(alpha=.2)
    fig.suptitle(r"$R_V$ versus local grain properties",fontsize=15)
    fig.savefig(a.prefix+"_scatter.png",dpi=220,bbox_inches="tight")
    fig.savefig(a.prefix+"_scatter.pdf",bbox_inches="tight")
    plt.close(fig)

    # Radial profile figure: medians and 16-84 percentile bands.
    rmax=np.nanmax(rr[base]); edges=np.arange(0,rmax+a.radial_bin_width,a.radial_bin_width)
    rc=.5*(edges[:-1]+edges[1:])
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    quantities=[("R_V",rv,r"$R_V$")]+[(n,p,l) for n,p,l,k in props]
    for ax,(name,q,label) in zip(axs.ravel(),quantities):
        med=[];lo=[];hi=[]
        for r0,r1 in zip(edges[:-1],edges[1:]):
            s=base&np.isfinite(q)&(rr>=r0)&(rr<r1)
            if s.any():
                zq=np.percentile(q[s],[16,50,84]);lo.append(zq[0]);med.append(zq[1]);hi.append(zq[2])
            else:lo.append(np.nan);med.append(np.nan);hi.append(np.nan)
        med=np.array(med);lo=np.array(lo);hi=np.array(hi)
        ax.plot(rc,med,marker="o",ms=3);ax.fill_between(rc,lo,hi,alpha=.18)
        ax.set_xlabel("Galactocentric radius [pkpc]");ax.set_ylabel(label);ax.set_title(name);ax.grid(alpha=.2)
    fig.savefig(a.prefix+"_radial_profiles.png",dpi=220,bbox_inches="tight")
    fig.savefig(a.prefix+"_radial_profiles.pdf",bbox_inches="tight")
    plt.close(fig)

    with open(a.prefix+"_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["property","N_raw","spearman_raw","p_raw","N_detrended",
                    "spearman_detrended","p_detrended","p16","median","p84"])
        w.writerows(rows)

    # Pixel residual table enables later cross-power/coherence work.
    X=(np.arange(rv.shape[1])-(rv.shape[1]-1)/2)*dx
    Y=(np.arange(rv.shape[0])-(rv.shape[0]-1)/2)*dx
    XX,YY=np.meshgrid(X,Y)
    with open(a.prefix+"_pixel_residuals.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["iy","ix","x_pkpc","y_pkpc","r_pkpc","Rv","delta_Rv",
                    "small_fraction","delta_small_fraction",
                    "mean_radius_nm","delta_mean_radius_nm",
                    "carbon_fraction","delta_carbon_fraction","dust_count","tauV"])
        ps=residuals["small_fraction"]; pr=residuals["mean_radius"]; pc=residuals["carbon_fraction"]
        for iy,ix in np.argwhere(base):
            w.writerow([iy,ix,XX[iy,ix],YY[iy,ix],rr[iy,ix],rv[iy,ix],drv[iy,ix],
                        ps[0][iy,ix],ps[1][iy,ix],pr[0][iy,ix],pr[1][iy,ix],
                        pc[0][iy,ix],pc[1][iy,ix],nc[iy,ix],tv[iy,ix]])

    print("Saved:")
    print(" ",a.prefix+"_scatter.png/pdf")
    print(" ",a.prefix+"_radial_profiles.png/pdf")
    print(" ",a.prefix+"_summary.csv")
    print(" ",a.prefix+"_pixel_residuals.csv")

if __name__=="__main__":
    main()
