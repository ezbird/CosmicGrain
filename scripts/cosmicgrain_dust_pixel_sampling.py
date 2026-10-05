#!/usr/bin/env python3
"""
cosmicgrain_dust_pixel_sampling.py

Measure how well dust particles sample face-on map pixels at several physical
pixel sizes, using the dust_count map saved in a finest-resolution CosmicGrain
master NPZ.

Reports, for each pixel size:
  - all-map and dust-occupied pixel counts
  - fractions with N_dust = 0, 1, 2, >=5, >=10, >=20
  - p16 / median / p84 dust count among occupied pixels
  - same statistics inside the R_V signal footprint (tau_V threshold)

Also produces:
  1) a simple sampling-fraction plot
  2) an occupied-pixel dust-count percentile plot
  3) a CSV summary
"""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f:
        raise ValueError(f"{a.shape} not divisible by rebin factor {f}")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def label(dx):
    return f"{dx*1000:.0f} pc" if dx < 1 else f"{dx:g} kpc"

def stats(n, footprint=None):
    if footprint is None:
        footprint=np.ones(n.shape,dtype=bool)
    vals=n[footprint]
    occ=vals>0
    pos=vals[occ]
    total=vals.size
    out={
        "total":total,
        "occupied":int(occ.sum()),
        "f0":float(np.mean(vals==0)),
        "f1":float(np.mean(vals==1)),
        "f2":float(np.mean(vals==2)),
        "fge5":float(np.mean(vals>=5)),
        "fge10":float(np.mean(vals>=10)),
        "fge20":float(np.mean(vals>=20)),
    }
    if pos.size:
        q=np.percentile(pos,[16,50,84])
        out.update(p16=float(q[0]),median=float(q[1]),p84=float(q[2]),
                   mean=float(np.mean(pos)),max=int(np.max(pos)))
    else:
        out.update(p16=np.nan,median=np.nan,p84=np.nan,mean=np.nan,max=0)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--pixel-sizes",nargs="+",type=float,
                    default=[0.125,0.25,0.5,1.0,2.0])
    ap.add_argument("--signal-frac",type=float,default=1e-3)
    ap.add_argument("--prefix",default="dust_pixel_sampling")
    a=ap.parse_args()

    z=np.load(a.npz)
    nc0=z["dust_count"].astype(int)
    tv0=z["tau_v"].astype(float)
    dx0=float(z["pixel_size_pkpc"])

    rows=[]
    print("="*82)
    print("COSMICGRAIN DUST-PIXEL SAMPLING DIAGNOSTIC")
    print("="*82)
    print(f"Master grid: {nc0.shape[0]}x{nc0.shape[1]}, dx={dx0:g} pkpc")
    print(f"Signal footprint at each scale: tau_V >= {a.signal_frac:g} * peak(tau_V)\n")

    for dx in a.pixel_sizes:
        q=dx/dx0; f=int(round(q))
        if f<1 or not np.isclose(q,f):
            raise ValueError(f"{dx} pkpc is not an integer multiple of master dx={dx0}")
        nc=block_sum(nc0,f).astype(int)
        tv=block_sum(tv0,f)
        sig=np.isfinite(tv)&(tv>=a.signal_frac*np.nanmax(tv))
        allst=stats(nc)
        sigst=stats(nc,sig)

        print(f"{label(dx)}: grid={nc.shape[0]}x{nc.shape[1]}")
        print(f"  ALL pixels:")
        print(f"    occupied={allst['occupied']:,}/{allst['total']:,} ({100*(1-allst['f0']):.2f}%)")
        print(f"    N=0: {100*allst['f0']:.2f}%   N=1: {100*allst['f1']:.2f}%   N=2: {100*allst['f2']:.2f}%")
        print(f"    N>=5: {100*allst['fge5']:.2f}%   N>=10: {100*allst['fge10']:.2f}%   N>=20: {100*allst['fge20']:.2f}%")
        print(f"    occupied N p16/median/p84 = {allst['p16']:.1f} / {allst['median']:.1f} / {allst['p84']:.1f}")
        print(f"  R_V SIGNAL footprint ({sig.sum():,} pixels):")
        print(f"    N>=5: {100*sigst['fge5']:.2f}%   N>=10: {100*sigst['fge10']:.2f}%   N>=20: {100*sigst['fge20']:.2f}%")
        print(f"    occupied N p16/median/p84 = {sigst['p16']:.1f} / {sigst['median']:.1f} / {sigst['p84']:.1f}\n")

        rows.append((dx,allst,sigst,int(sig.sum())))

    # Plot fractions for thresholds among all map pixels.
    fig,ax=plt.subplots(figsize=(8,5.5),constrained_layout=True)
    xs=np.array([r[0]*1000 for r in rows])
    for key,text in [("f0","N = 0"),("fge5","N ≥ 5"),("fge10","N ≥ 10"),("fge20","N ≥ 20")]:
        ys=np.array([r[1][key] for r in rows])*100
        ax.plot(xs,ys,marker="o",label=text)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("Map pixel size [pc]")
    ax.set_ylabel("Fraction of all map pixels [%]")
    ax.set_title("Dust-particle sampling versus map pixel size")
    ax.grid(True,which="both",alpha=.22)
    ax.legend()
    fig.savefig(a.prefix+"_fractions.png",dpi=240,bbox_inches="tight")
    fig.savefig(a.prefix+"_fractions.pdf",bbox_inches="tight")
    plt.close(fig)

    # Occupied-pixel count percentiles.
    fig,ax=plt.subplots(figsize=(8,5.5),constrained_layout=True)
    med=np.array([r[1]["median"] for r in rows])
    lo=np.array([r[1]["p16"] for r in rows])
    hi=np.array([r[1]["p84"] for r in rows])
    ax.plot(xs,med,marker="o",label="median")
    ax.fill_between(xs,lo,hi,alpha=.18,label="16–84%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("Map pixel size [pc]")
    ax.set_ylabel(r"$N_{dust}$ per occupied pixel")
    ax.set_title("Dust particles per occupied map pixel")
    ax.grid(True,which="both",alpha=.22)
    ax.legend()
    fig.savefig(a.prefix+"_counts.png",dpi=240,bbox_inches="tight")
    fig.savefig(a.prefix+"_counts.pdf",bbox_inches="tight")
    plt.close(fig)

    with open(a.prefix+"_summary.csv","w",newline="") as fh:
        w=csv.writer(fh)
        w.writerow(["pixel_size_pkpc","grid_pixels","occupied","occupied_fraction",
                    "f_N0","f_N1","f_N2","f_Nge5","f_Nge10","f_Nge20",
                    "occupied_p16","occupied_median","occupied_p84","occupied_mean","max",
                    "signal_pixels","signal_f_Nge5","signal_f_Nge10","signal_f_Nge20",
                    "signal_occupied_p16","signal_occupied_median","signal_occupied_p84"])
        for dx,s,t,nsig in rows:
            w.writerow([dx,s["total"],s["occupied"],1-s["f0"],s["f0"],s["f1"],s["f2"],
                        s["fge5"],s["fge10"],s["fge20"],s["p16"],s["median"],s["p84"],
                        s["mean"],s["max"],nsig,t["fge5"],t["fge10"],t["fge20"],
                        t["p16"],t["median"],t["p84"]])
    print("Saved:")
    print(" ",a.prefix+"_fractions.png/pdf")
    print(" ",a.prefix+"_counts.png/pdf")
    print(" ",a.prefix+"_summary.csv")

if __name__=="__main__":
    main()
