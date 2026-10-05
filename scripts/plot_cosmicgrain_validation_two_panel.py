#!/usr/bin/env python3
"""
Two-panel z=0 CosmicGrain galaxy-validation figure.

(a) Stellar mass--halo mass:
    CosmicGrain vs Moster et al. (2013) z=0 abundance-matching relation.
    M200c values come from the same CosmicGrain CSV used by the MZR panel.

(b) Mass--metallicity:
    SAGAbg direct-Te + Andrews & Martini (2013) + PHANGS direct regions
    + CosmicGrain SFR-weighted O/H.
    CG galaxies with no current SF are shown open at their all-gas
    mass-weighted O/H (currently CG06).

Uses scripts/cosmicgrain.mplstyle by default.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PHANGS_M = {
    "IC5332":9.67,"NGC0628":10.34,"NGC1087":9.93,"NGC1300":10.62,
    "NGC1365":10.99,"NGC1385":9.98,"NGC1433":10.87,"NGC1512":10.71,
    "NGC1566":10.78,"NGC1672":10.73,"NGC2835":10.00,"NGC3351":10.36,
    "NGC3627":10.83,"NGC4254":10.42,"NGC4303":10.52,"NGC4321":10.75,
    "NGC4535":10.53,"NGC5068":9.40,"NGC7496":10.00,
}

def fl(x):
    try: return float(str(x).strip())
    except (ValueError,TypeError): return np.nan

def read_saga(path):
    a=[]
    for l in open(path,encoding="utf-8"):
        if len(l)>=435 and l[:19].strip().isdigit():
            r=(fl(l[40:46]),fl(l[47:51]),fl(l[421:425]),fl(l[426:430]),fl(l[431:435]))
            if np.isfinite(r[1]) and np.isfinite(r[3]): a.append(r)
    return a

def norm(s):
    s=s.strip().upper().replace(" ","")
    if s.startswith("NGC") and s[3:].isdigit(): return "NGC"+s[3:].zfill(4)
    if s.startswith("IC") and s[2:].isdigit(): return "IC"+s[2:].zfill(4)
    return s

def read_phangs(path):
    g=defaultdict(list)
    for l in open(path,encoding="utf-8"):
        if len(l)<455: continue
        n=norm(l[:7]); ha=fl(l[77:86]); oh=fl(l[438:455])
        if n in PHANGS_M and np.isfinite(ha) and ha>0 and np.isfinite(oh):
            g[n].append((ha,oh))
    out=[]
    for n,v in sorted(g.items()):
        w=np.array([x[0] for x in v]); y=np.array([x[1] for x in v])
        lin=10**(y-12)
        mean=12+np.log10(np.sum(w*lin)/np.sum(w))
        wmean=np.sum(w*y)/np.sum(w)
        spread=np.sqrt(np.sum(w*(y-wmean)**2)/np.sum(w))
        out.append((n,PHANGS_M[n],mean,spread,len(v)))
    return out

def read_cg(path):
    a=[]
    for r in csv.DictReader(open(path,newline="",encoding="utf-8")):
        # tolerate likely column-name variants for M200
        m200=np.nan
        for k in ("M200c_Msun","M200_Msun","M200c","M200"):
            if k in r and np.isfinite(fl(r[k])):
                m200=fl(r[k]); break
        ns=0
        try: ns=int(float(r.get("N_SFgas_30pkpc",0) or 0))
        except ValueError: pass
        a.append(dict(
            name=r["CG"].strip(), logmstar=fl(r.get("log10_Mstar_Msun")),
            m200=m200, oh_sfr=fl(r.get("OH12_SFRweighted")),
            oh_all=fl(r.get("OH12_allgas_massweighted")),
            nsf=ns, sfr=fl(r.get("SFR_Msun_yr"))
        ))
    return a

def am13(x):
    return 8.798-np.log10(1+10**(0.640*(8.901-x)))

def moster13(mh):
    # z=0 parameters: Moster, Naab & White (2013)
    logM1=11.590; N=0.0351; beta=1.376; gamma=0.608
    M1=10**logM1
    ratio=2*N/((mh/M1)**(-beta)+(mh/M1)**gamma)
    return mh*ratio


def annotate_cg_consistent(ax, x, y, label, fontsize=9, fontweight="bold"):
    """Place every CosmicGrain label at the same offset, above all data."""
    return ax.annotate(
        label, (x, y),
        xytext=(7, 6), textcoords="offset points",
        fontsize=fontsize, fontweight=fontweight,
        ha="left", va="bottom",
        zorder=100,
        annotation_clip=False,
    )

def main():
    q=argparse.ArgumentParser()
    q.add_argument("--style",default="scripts/cosmicgrain.mplstyle")
    q.add_argument("--sagabg",default="scripts/apjad3042t1_mrt.txt")
    q.add_argument("--phangs",default="scripts/catalog.dat")
    q.add_argument("--cosmicgrain",default="cosmicgrain_sagabg_z0.csv")
    q.add_argument("--prefix",default="cosmicgrain_z0_validation")
    q.add_argument("--label-phangs",action="store_true")
    A=q.parse_args()

    plt.style.use(A.style)

    # Figure-1 typography: deliberately larger than the base CosmicGrain style.
    # CG point labels remain at fontsize=9 in annotate_cg_consistent().
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13.5,
    })
    S=read_saga(A.sagabg); P=read_phangs(A.phangs); C=read_cg(A.cosmicgrain)

    if len(S)!=120: print("WARNING: parsed",len(S),"SAGAbg galaxies, expected 120.")
    print("SAGAbg:",len(S))
    print("PHANGS:",sum(x[4] for x in P),"regions in",len(P),"galaxies")

    BLUE="#0072B2"
    PINK="#CC79A7"

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.8,10.0),sharex=True,
                              gridspec_kw={"height_ratios":[0.62,1.0],"hspace":0.06})

    # ---------- (a) Mstar-M200 ----------
    mh=np.logspace(10.3,12.7,500)
    ax1.plot(np.log10(moster13(mh)),np.log10(mh),ls="--",color="0.25",
             label="Moster et al. 2013")

    have_halo=[r for r in C if np.isfinite(r["m200"]) and r["m200"]>0 and np.isfinite(r["logmstar"])]
    if not have_halo:
        print("\nWARNING: no M200 column found in the CSV.")
        print("Panel (a) needs one of: M200c_Msun, M200_Msun, M200c, M200.")
        print("Re-run the CG extraction with M200c written to the CSV, or add that column.")
    else:
        x=np.array([r["logmstar"] for r in have_halo]); y=np.log10([r["m200"] for r in have_halo])
        ax1.scatter(x,y,s=105,color=BLUE,edgecolor="black",lw=.75,label="CosmicGrain",zorder=5)
        for r,xx,yy in zip(have_halo,x,y):
            annotate_cg_consistent(ax1, xx, yy, r["name"])

    ax1.set_xlim(7.0,12.0)
    ax1.set_ylabel(r"$\log_{10}(M_{200c}/M_\odot)$")
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc="upper left")
    ax1.text(.97,.04,"(a)",transform=ax1.transAxes,ha="right",va="bottom",
             fontweight="bold",color="0.20",fontsize=16,zorder=20)

    # ---------- (b) MZR ----------
    z,sm,p16,p50,p84=map(np.array,zip(*S))
    lo=p50-p16; hi=p84-p50
    ok=np.isfinite(lo)&np.isfinite(hi)&(lo>=0)&(hi>=0)
    ax2.errorbar(sm[ok],p50[ok],yerr=np.vstack((lo[ok],hi[ok])),fmt="none",
                 ecolor="0.62",elinewidth=.55,alpha=.38,capsize=0,zorder=1)
    ax2.scatter(sm,p50,s=23,color="0.48",alpha=.68,lw=0,
                label=rf"SAGAbg ({len(S)})",zorder=2)

    xx=np.linspace(7.4,10.5,350)
    ax2.plot(xx,am13(xx),ls="--",color="0.20",label="Andrews & Martini 2013",zorder=3)

    if P:
        px=np.array([r[1] for r in P]); py=np.array([r[2] for r in P]); ps=np.array([r[3] for r in P])
        ax2.errorbar(px,py,yerr=ps,fmt="none",color=PINK,elinewidth=1,alpha=.7,capsize=2.5,zorder=4)
        ax2.scatter(px,py,s=65,marker="D",color=PINK,edgecolor="black",lw=.6,
                    label=r"PHANGS-MUSE (H$\alpha$-weighted)",zorder=5)
        if A.label_phangs:
            for n,m,o,s,k in P:
                ax2.annotate(n,(m,o),xytext=(4,4),textcoords="offset points",fontsize=7)

    sf=[r for r in C if np.isfinite(r["logmstar"]) and np.isfinite(r["oh_sfr"])]
    nosf=[r for r in C if np.isfinite(r["logmstar"]) and not np.isfinite(r["oh_sfr"]) and np.isfinite(r["oh_all"])]

    if sf:
        ax2.scatter([r["logmstar"] for r in sf],[r["oh_sfr"] for r in sf],
                    s=110,color=BLUE,edgecolor="black",lw=.75,
                    label="CosmicGrain (SFR-weighted)",zorder=7)
        for r in sf:
            annotate_cg_consistent(ax2, r["logmstar"], r["oh_sfr"], r["name"])

    if nosf:
        ax2.scatter([r["logmstar"] for r in nosf],[r["oh_all"] for r in nosf],
                    s=110,facecolor="white",edgecolor=BLUE,lw=2,
                    label="CosmicGrain (no SF; all-gas)",zorder=7)
        for r in nosf:
            annotate_cg_consistent(ax2, r["logmstar"], r["oh_all"], r["name"])
            print(f"{r['name']}: open symbol at all-gas OH12={r['oh_all']:.3f}; "
                  f"SFR={r['sfr']:.3e}, N_SF={r['nsf']}")

    ax2.set_xlim(7.0,12.0); ax2.set_ylim(6.95,9.18)
    ax2.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax2.set_ylabel(r"$12+\log_{10}(\mathrm{O/H})$")
    ax2.legend(loc="upper left",fontsize=8.5)
    ax2.text(.97,.04,"(b)",transform=ax2.transAxes,ha="right",va="bottom",
             fontweight="bold",color="0.20",fontsize=16,zorder=20)

    fig.subplots_adjust(left=0.12,right=0.98,bottom=0.09,top=0.98,hspace=0.06)
    fig.savefig(A.prefix+".png")
    fig.savefig(A.prefix+".pdf")
    plt.close(fig)
    print("\nWrote",A.prefix+".png")
    print("Wrote",A.prefix+".pdf")

if __name__=="__main__":
    main()
