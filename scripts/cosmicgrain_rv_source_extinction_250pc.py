#!/usr/bin/env python3
import argparse, glob, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cosmicgrain_rv_maps as rvcore

RHO_C, RHO_SIL = 2.24, 3.50
SOURCE_NAMES = {0:"SNII",1:"AGB",2:"LRN"}

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)",s)]

def periodic_delta(coords, center, box):
    d=coords-center[None,:]
    d-=box*np.rint(d/box)
    return d

def block_sum(a,f):
    ny,nx=a.shape
    if ny%f or nx%f: raise ValueError("Map dimensions not divisible by block factor")
    return a.reshape(ny//f,f,nx//f,f).sum((1,3))

def rv_from_tau(tb,tv):
    d=tb-tv
    out=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    out[q]=tv[q]/d[q]
    return out

def safe_fraction(num,den):
    out=np.full_like(den,np.nan,dtype=float)
    q=np.isfinite(num)&np.isfinite(den)&(den>0)
    out[q]=num[q]/den[q]
    return out

def load_dust(files):
    C=[]; M=[]; R=[]; F=[]; S=[]
    a=box=h=None
    for fn in files:
        with h5py.File(fn,"r") as f:
            if a is None:
                a=float(f["Header"].attrs["Time"])
                box=float(f["Header"].attrs["BoxSize"])
                h=float(f["Parameters"].attrs["HubbleParam"])
            if "PartType6" not in f: continue
            g=f["PartType6"]
            C.append(np.asarray(g["Coordinates"][:],float))
            M.append(np.asarray(g["Masses"][:],float))
            R.append(np.asarray(g["GrainRadius"][:],float))
            F.append(np.asarray(g["CarbonMassFraction"][:],float))
            S.append(np.asarray(g["DustSource"][:],np.int64))
    if not C: raise RuntimeError("No PartType6 particles found")
    return tuple(map(np.concatenate,(C,M,R,F,S)))+(a,box,h)

def report(label,mask,rv,nd,tvsrc,esrc):
    print("\n"+"="*76+f"\n{label}\n"+"="*76)
    print(f"N pixels          = {mask.sum()}")
    if not mask.any(): return
    print(f"median R(V)       = {np.median(rv[mask]):.5f}")
    print(f"median N_dust     = {np.median(nd[mask]):.3f}")
    tvtot=sum(np.sum(tvsrc[s][mask]) for s in (0,1,2))
    etot=sum(np.sum(esrc[s][mask]) for s in (0,1,2))
    print("\nIntegrated source contribution:\n")
    print(f"{'Source':<8}{'tauV frac':>14}{'E(B-V) frac':>16}{'effective RV':>16}")
    for s in (0,1,2):
        tv=np.sum(tvsrc[s][mask]); e=np.sum(esrc[s][mask])
        print(f"{SOURCE_NAMES[s]:<8}{tv/tvtot if tvtot>0 else np.nan:14.5f}"
              f"{e/etot if etot>0 else np.nan:16.5f}{tv/e if e>0 else np.nan:16.5f}")
    for title,src in [("tau_V",tvsrc),("E(B-V)",esrc)]:
        total=sum(src.values())
        print(f"\nMedian per-pixel {title} fractions:")
        for s in (0,1,2):
            fr=safe_fraction(src[s],total)
            q=np.nanpercentile(fr[mask],[16,50,84])
            print(f"  {SOURCE_NAMES[s]:5s}: {q[1]:.5f} [{q[0]:.5f}, {q[2]:.5f}]")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("snapshot"); ap.add_argument("master_npz")
    ap.add_argument("rv_250pc"); ap.add_argument("count_250pc")
    ap.add_argument("--carbon-table",required=True); ap.add_argument("--silicate-table",required=True)
    ap.add_argument("--square-size",type=float,default=20.0)
    ap.add_argument("--pixel",type=float,default=.25); ap.add_argument("--radius",type=float,default=2.5)
    ap.add_argument("--prefix",default="halo441_current_rv_250pc_source_extinction")
    A=ap.parse_args()
    files=sorted(glob.glob(A.snapshot),key=natural_key)
    if not files: raise FileNotFoundError(A.snapshot)
    m=np.load(A.master_npz)
    center=np.asarray(m["center"],float); rot=np.asarray(m["rotation"],float)
    native=float(np.asarray(m["pixel_size_pkpc"]).squeeze())
    mr=float(np.asarray(m["plot_radius_pkpc"]).squeeze())
    zmax=float(np.asarray(m["zmax_pkpc"]).squeeze())
    print("\nMASTER GEOMETRY\n"+"="*76)
    print("center            =",center); print("rotation =\n",rot)
    print(f"native pixel      = {native:.6f} pkpc\nmaster radius     = {mr:.6f} pkpc\nzmax              = {zmax:.6f} pkpc")
    coords,mass,rnm,carbon,source,a,box,h=load_dust(files)
    print("\nSNAPSHOT\n"+"="*76)
    print(f"N PartType6       = {len(mass):,}\na                 = {a:.8f}\nh                 = {h:.8f}\nBoxSize           = {box:.8f}")
    xyz=periodic_delta(coords,center,box)*a/h
    xyz=xyz@rot.T
    x,y,z=xyz.T
    sel=(np.abs(x)<mr)&(np.abs(y)<mr)&(np.abs(z)<zmax)
    x,y,mass,rnm,carbon,source=x[sel],y[sel],mass[sel],rnm[sel],np.clip(carbon[sel],0,1),source[sel]
    print(f"Dust in face-on volume = {len(x):,}")

    # EXACT production optics
    rc,wc,qc=rvcore.read_qext_table(A.carbon_table)
    rs,ws,qs=rvcore.read_qext_table(A.silicate_table)
    qcbt=rvcore.q_at_lambda(rc,wc,qc,.44); qcvt=rvcore.q_at_lambda(rc,wc,qc,.55)
    qsbt=rvcore.q_at_lambda(rs,ws,qs,.44); qsvt=rvcore.q_at_lambda(rs,ws,qs,.55)
    rum=rnm*1e-3
    qcb=rvcore.interp_q_radius(rum,rc,qcbt); qcv=rvcore.interp_q_radius(rum,rc,qcvt)
    qsb=rvcore.interp_q_radius(rum,rs,qsbt); qsv=rvcore.interp_q_radius(rum,rs,qsvt)
    good=np.isfinite(qcb)&np.isfinite(qcv)&np.isfinite(qsb)&np.isfinite(qsv)&np.isfinite(mass)&(mass>0)&np.isfinite(rum)&(rum>0)
    x,y,mass,rum,carbon,source=x[good],y[good],mass[good],rum[good],carbon[good],source[good]
    qcb,qcv,qsb,qsv=qcb[good],qcv[good],qsb[good],qsv[good]
    wb=(mass/rum)*(carbon*qcb/RHO_C+(1-carbon)*qsb/RHO_SIL)
    wv=(mass/rum)*(carbon*qcv/RHO_C+(1-carbon)*qsv/RHO_SIL)

    nb=int(round(2*mr/native)); edges=np.linspace(-mr,mr,nb+1)
    cnt0=np.histogram2d(y,x,bins=(edges,edges))[0]
    tb0={}; tv0={}
    for s in (0,1,2):
        q=source==s
        tb0[s]=np.histogram2d(y[q],x[q],bins=(edges,edges),weights=wb[q])[0]
        tv0[s]=np.histogram2d(y[q],x[q],bins=(edges,edges),weights=wv[q])[0]
    nc=int(round(A.square_size/native)); st=(nb-nc)//2; en=st+nc
    f=int(round(A.pixel/native))
    if abs(f*native-A.pixel)>1e-10: raise ValueError("pixel must be integer multiple of native pixel")
    cnt=block_sum(cnt0[st:en,st:en],f)
    tb={s:block_sum(tb0[s][st:en,st:en],f) for s in (0,1,2)}
    tv={s:block_sum(tv0[s][st:en,st:en],f) for s in (0,1,2)}
    TB=sum(tb.values()); TV=sum(tv.values()); rvrec=rv_from_tau(TB,TV)
    rvref=np.loadtxt(A.rv_250pc); cref=np.loadtxt(A.count_250pc)

    print("\nEXACT VALIDATION\n"+"="*76)
    cd=cnt-cref
    print(f"Maximum absolute count difference = {np.max(np.abs(cd)):.0f}")
    print(f"Mismatched count pixels           = {np.count_nonzero(cd)} / {cnt.size}")
    if np.max(np.abs(cd))!=0 or np.count_nonzero(cd)!=0: raise RuntimeError("Count validation failed; refusing interpretation")
    print("COUNT VALIDATION PASSED EXACTLY.")
    frec=np.isfinite(rvrec); fref=np.isfinite(rvref)
    mm=np.count_nonzero(frec!=fref)
    print(f"Finite-mask mismatched pixels      = {mm} / {rvref.size}")
    both=frec&fref
    if not both.any(): raise RuntimeError("No common finite R(V) pixels")
    d=np.abs(rvrec[both]-rvref[both])
    print(f"Maximum |R(V) reconstruction-ref| = {d.max():.8e}")
    print(f"Median  |R(V) reconstruction-ref| = {np.median(d):.8e}")
    if mm or d.max()>1e-6: raise RuntimeError("R(V) optical reconstruction failed; refusing source-extinction interpretation")
    print("OPTICAL R(V) VALIDATION PASSED.")

    rv=rvref; nd=cref; ny,nx=rv.shape
    xx=(np.arange(nx)+.5-nx/2)*A.pixel; yy=(np.arange(ny)+.5-ny/2)*A.pixel
    X,Y=np.meshgrid(xx,yy); aperture=np.hypot(X,Y)<=A.radius
    valid=aperture&np.isfinite(rv)&(nd>=1)
    es={s:tb[s]-tv[s] for s in (0,1,2)}
    print("\nANALYSIS FOOTPRINT\n"+"="*76)
    print(f"footprint pixels = {aperture.sum()}\nvalid pixels     = {valid.sum()}\nmean R(V)        = {np.mean(rv[valid]):.6f}\nmedian R(V)      = {np.median(rv[valid]):.6f}")
    for label,mask in [("R(V)<=6",valid&(rv<=6)),("R(V)>6",valid&(rv>6)),("R(V)>10",valid&(rv>10)),("R(V)>15",valid&(rv>15))]:
        report(label,mask,rv,nd,tv,es)

    TV=sum(tv.values()); E=sum(es.values())
    rows=[]
    for iy,ix in zip(*np.where(valid)):
        row=[X[iy,ix],Y[iy,ix],rv[iy,ix],nd[iy,ix],TV[iy,ix],E[iy,ix]]
        for s in (0,1,2):
            row += [tv[s][iy,ix],es[s][iy,ix],
                    tv[s][iy,ix]/TV[iy,ix] if TV[iy,ix]>0 else np.nan,
                    es[s][iy,ix]/E[iy,ix] if E[iy,ix]>0 else np.nan]
        rows.append(row)
    header="x_pkpc y_pkpc RV Ndust tauV_total EBV_total tauV_SNII EBV_SNII tauVfrac_SNII EBVfrac_SNII tauV_AGB EBV_AGB tauVfrac_AGB EBVfrac_AGB tauV_LRN EBV_LRN tauVfrac_LRN EBVfrac_LRN"
    np.savetxt(A.prefix+"_pixels.txt",np.asarray(rows),header=header)

    agbt=safe_fraction(tv[1],TV); agbe=safe_fraction(es[1],E)
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    sc=ax.scatter(agbt[valid],rv[valid],c=np.log10(nd[valid]),s=45,alpha=.8)
    ax.axhline(6,ls="--",lw=1); ax.axhline(10,ls=":",lw=1)
    ax.set_xlabel(r"AGB fraction of $\tau_V$"); ax.set_ylabel(r"$R(V)$")
    ax.set_title("CG10: AGB contribution to optical extinction")
    fig.colorbar(sc,ax=ax).set_label(r"$\log_{10} N_{\rm dust}$")
    fig.tight_layout(); fig.savefig(A.prefix+"_rv_vs_agb_tauV.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,5.5))
    sc=ax.scatter(agbe[valid],agbt[valid],c=rv[valid],s=45,alpha=.8)
    ax.plot([0,1],[0,1],ls="--",lw=1)
    ax.set_xlabel(r"AGB fraction of $E(B-V)$"); ax.set_ylabel(r"AGB fraction of $\tau_V$")
    ax.set_title("CG10: source contribution to extinction")
    fig.colorbar(sc,ax=ax).set_label(r"$R(V)$")
    fig.tight_layout(); fig.savefig(A.prefix+"_agb_tauV_vs_ebv.png",dpi=220); plt.close(fig)

    print("\nOUTPUTS\n"+"="*76)
    print(A.prefix+"_pixels.txt")
    print(A.prefix+"_rv_vs_agb_tauV.png")
    print(A.prefix+"_agb_tauV_vs_ebv.png")

if __name__=="__main__":
    main()
