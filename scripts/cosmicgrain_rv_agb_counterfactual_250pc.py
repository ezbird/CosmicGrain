#!/usr/bin/env python3
"""
cosmicgrain_rv_agb_counterfactual_250pc.py

Counterfactual optical experiment for the CURRENT CG10 R(V) master map.

Question:
  Is the extreme R(V) of AGB-origin dust driven mainly by its large grain
  radius, its carbon-rich composition, or the combination?

The script uses the exact production optics functions from cosmicgrain_rv_maps.py
and the exact geometry stored in the master NPZ. It first reconstructs the
reference 250-pc R(V) map and refuses to continue unless the match is <1e-6.

Counterfactuals alter ONLY DustSource==1 (AGB) particles:
  baseline          : actual radius, actual carbon fraction
  AGB radius=10 nm  : actual carbon fraction, radius forced to 10 nm
  AGB carbon=0.10   : actual radius, carbon fraction forced to 0.10
  AGB both SNII-like: radius=10 nm and carbon fraction=0.10

The same particles, masses, positions, SNII/LRN dust, grid, aperture, and
pixel selection are retained. Results are reported both for:
  (1) the fixed pixels that are R(V)>10 in the REAL baseline map, and
  (2) the full valid aperture under each counterfactual.

This is an optical sensitivity experiment, not a self-consistent rerun of the
dust evolution.
"""

import argparse, glob, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cosmicgrain_rv_maps as rvcore

RHO_C = 2.24
RHO_SIL = 3.50
SRC_AGB = 1

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def periodic_delta(coords, center, box):
    d = coords - center[None, :]
    d -= box * np.rint(d / box)
    return d

def block_sum(a, f):
    ny, nx = a.shape
    if ny % f or nx % f:
        raise ValueError("Map dimensions not divisible by block factor")
    return a.reshape(ny//f, f, nx//f, f).sum((1,3))

def rv_from_tau(tb, tv):
    d = tb - tv
    out = np.full_like(tv, np.nan, dtype=float)
    q = np.isfinite(tb) & np.isfinite(tv) & (tv > 0) & (d > 0)
    out[q] = tv[q] / d[q]
    return out

def load_dust(files):
    C=[]; M=[]; R=[]; F=[]; S=[]
    a=box=h=None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if a is None:
                a = float(f["Header"].attrs["Time"])
                box = float(f["Header"].attrs["BoxSize"])
                h = float(f["Parameters"].attrs["HubbleParam"])
            if "PartType6" not in f:
                continue
            g=f["PartType6"]
            C.append(np.asarray(g["Coordinates"][:], float))
            M.append(np.asarray(g["Masses"][:], float))
            R.append(np.asarray(g["GrainRadius"][:], float))
            F.append(np.asarray(g["CarbonMassFraction"][:], float))
            S.append(np.asarray(g["DustSource"][:], np.int64))
    if not C:
        raise RuntimeError("No PartType6 particles found")
    return (np.concatenate(C), np.concatenate(M), np.concatenate(R),
            np.concatenate(F), np.concatenate(S), a, box, h)

def optics_tables(carbon_table, silicate_table):
    rc,wc,qc = rvcore.read_qext_table(carbon_table)
    rs,ws,qs = rvcore.read_qext_table(silicate_table)
    return {
        "rc":rc, "rs":rs,
        "qcb":rvcore.q_at_lambda(rc,wc,qc,0.44),
        "qcv":rvcore.q_at_lambda(rc,wc,qc,0.55),
        "qsb":rvcore.q_at_lambda(rs,ws,qs,0.44),
        "qsv":rvcore.q_at_lambda(rs,ws,qs,0.55),
    }

def weights(radius_nm, carbon, mass, tab):
    rum = radius_nm * 1e-3
    qcb = rvcore.interp_q_radius(rum, tab["rc"], tab["qcb"])
    qcv = rvcore.interp_q_radius(rum, tab["rc"], tab["qcv"])
    qsb = rvcore.interp_q_radius(rum, tab["rs"], tab["qsb"])
    qsv = rvcore.interp_q_radius(rum, tab["rs"], tab["qsv"])
    good = (np.isfinite(qcb)&np.isfinite(qcv)&np.isfinite(qsb)&np.isfinite(qsv)
            &np.isfinite(mass)&(mass>0)&np.isfinite(rum)&(rum>0))
    wb=np.full(len(mass),np.nan); wv=np.full(len(mass),np.nan)
    wb[good]=(mass[good]/rum[good])*(carbon[good]*qcb[good]/RHO_C
             +(1-carbon[good])*qsb[good]/RHO_SIL)
    wv[good]=(mass[good]/rum[good])*(carbon[good]*qcv[good]/RHO_C
             +(1-carbon[good])*qsv[good]/RHO_SIL)
    return wb,wv,good

def make_coarse_map(x,y,wb,wv,good,master_radius,native_pixel,
                    square_size,pixel):
    nb=int(round(2*master_radius/native_pixel))
    edges=np.linspace(-master_radius,master_radius,nb+1)
    tb0=np.histogram2d(y[good],x[good],bins=(edges,edges),weights=wb[good])[0]
    tv0=np.histogram2d(y[good],x[good],bins=(edges,edges),weights=wv[good])[0]
    c0=np.histogram2d(y[good],x[good],bins=(edges,edges))[0]
    nc=int(round(square_size/native_pixel))
    st=(nb-nc)//2; en=st+nc
    fac=int(round(pixel/native_pixel))
    tb=block_sum(tb0[st:en,st:en],fac)
    tv=block_sum(tv0[st:en,st:en],fac)
    cnt=block_sum(c0[st:en,st:en],fac)
    return rv_from_tau(tb,tv),tb,tv,cnt

def pct(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    if not len(a): return (np.nan,)*5
    q=np.percentile(a,[16,50,84])
    return len(a),np.mean(a),q[0],q[1],q[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("master_npz")
    ap.add_argument("rv_250pc")
    ap.add_argument("count_250pc")
    ap.add_argument("--carbon-table",required=True)
    ap.add_argument("--silicate-table",required=True)
    ap.add_argument("--square-size",type=float,default=20.0)
    ap.add_argument("--pixel",type=float,default=0.25)
    ap.add_argument("--radius",type=float,default=2.5)
    ap.add_argument("--agb-test-radius-nm",type=float,default=10.0)
    ap.add_argument("--agb-test-carbon",type=float,default=0.10)
    ap.add_argument("--high-rv",type=float,default=10.0)
    ap.add_argument("--prefix",default="halo441_current_rv_250pc_agb_counterfactual")
    A=ap.parse_args()

    files=sorted(glob.glob(A.snapshot),key=natural_key)
    if not files: raise FileNotFoundError(A.snapshot)
    master=np.load(A.master_npz)
    center=np.asarray(master["center"],float)
    rot=np.asarray(master["rotation"],float)
    native=float(np.asarray(master["pixel_size_pkpc"]).squeeze())
    mr=float(np.asarray(master["plot_radius_pkpc"]).squeeze())
    zmax=float(np.asarray(master["zmax_pkpc"]).squeeze())

    coords,mass,rnm,carbon,source,a,box,h=load_dust(files)
    xyz=periodic_delta(coords,center,box)*a/h
    xyz=xyz@rot.T
    x,y,z=xyz.T
    sel=(np.abs(x)<mr)&(np.abs(y)<mr)&(np.abs(z)<zmax)
    x,y=x[sel],y[sel]
    mass,rnm,carbon,source=mass[sel],rnm[sel],np.clip(carbon[sel],0,1),source[sel]

    print("INPUT")
    print("="*76)
    print(f"Dust in face-on volume = {len(x):,}")
    print(f"AGB particles          = {(source==SRC_AGB).sum():,}")
    print(f"test AGB radius        = {A.agb_test_radius_nm:g} nm")
    print(f"test AGB carbon frac   = {A.agb_test_carbon:g}")

    tab=optics_tables(A.carbon_table,A.silicate_table)

    scenarios=[
        ("baseline","Baseline",False,False),
        ("radius10","AGB radius -> 10 nm",True,False),
        ("carbon010","AGB carbon -> 0.10",False,True),
        ("both","AGB radius -> 10 nm; carbon -> 0.10",True,True),
    ]
    maps={}

    for key,label,change_r,change_c in scenarios:
        rr=rnm.copy(); cc=carbon.copy()
        agb=source==SRC_AGB
        if change_r: rr[agb]=A.agb_test_radius_nm
        if change_c: cc[agb]=A.agb_test_carbon
        wb,wv,good=weights(rr,cc,mass,tab)
        maps[key]=(label,)+make_coarse_map(
            x,y,wb,wv,good,mr,native,A.square_size,A.pixel
        )

    rvref=np.loadtxt(A.rv_250pc)
    cref=np.loadtxt(A.count_250pc)
    rv0,tb0,tv0,c0=maps["baseline"][1:]

    print("\nEXACT BASELINE VALIDATION")
    print("="*76)
    cd=c0-cref
    finite=np.isfinite(rv0)&np.isfinite(rvref)
    mm=np.count_nonzero(np.isfinite(rv0)!=np.isfinite(rvref))
    md=np.max(np.abs(rv0[finite]-rvref[finite]))
    print(f"max count difference    = {np.max(np.abs(cd)):.0f}")
    print(f"count mismatched pixels = {np.count_nonzero(cd)} / {cd.size}")
    print(f"finite-mask mismatches  = {mm} / {rvref.size}")
    print(f"max |RV baseline-ref|   = {md:.8e}")
    if np.max(np.abs(cd))!=0 or np.count_nonzero(cd)!=0 or mm or md>1e-6:
        raise RuntimeError("Baseline validation failed; refusing counterfactual interpretation")
    print("BASELINE VALIDATION PASSED.")

    ny,nx=rvref.shape
    xx=(np.arange(nx)+.5-nx/2)*A.pixel
    yy=(np.arange(ny)+.5-ny/2)*A.pixel
    X,Y=np.meshgrid(xx,yy)
    aperture=np.hypot(X,Y)<=A.radius
    base_valid=aperture&np.isfinite(rvref)&(cref>=1)
    high_base=base_valid&(rvref>A.high_rv)

    print("\nFIXED BASELINE HIGH-R(V) PIXELS")
    print("="*76)
    print(f"threshold = R(V) > {A.high_rv:g}")
    print(f"N pixels  = {high_base.sum()}")
    print("\nScenario                                  mean RV   median RV   p16      p84")
    summary=[]
    for key,label,_,_ in scenarios:
        rv=maps[key][1]
        vals=rv[high_base]
        n,mean,p16,med,p84=pct(vals)
        print(f"{label:<40s} {mean:8.3f} {med:10.3f} {p16:8.3f} {p84:8.3f}")
        summary.append((key,label,n,mean,p16,med,p84))

    print("\nFULL APERTURE UNDER EACH COUNTERFACTUAL")
    print("="*76)
    print("Scenario                                  median RV   frac>6   frac>10  frac>15")
    fullrows=[]
    for key,label,_,_ in scenarios:
        rv=maps[key][1]
        q=aperture&np.isfinite(rv)&(maps[key][4]>=1)
        vals=rv[q]
        f6=np.mean(vals>6); f10=np.mean(vals>10); f15=np.mean(vals>15)
        print(f"{label:<40s} {np.median(vals):10.3f} {f6:9.4f} {f10:9.4f} {f15:9.4f}")
        fullrows.append((key,label,len(vals),np.mean(vals),np.median(vals),np.std(vals),f6,f10,f15))

    # Save summary text.
    with open(A.prefix+"_summary.txt","w") as f:
        f.write("# Fixed baseline high-RV pixels\n")
        f.write("# key N mean p16 median p84\n")
        for key,label,n,mean,p16,med,p84 in summary:
            f.write(f"{key} {n} {mean:.8g} {p16:.8g} {med:.8g} {p84:.8g}\n")
        f.write("\n# Full aperture\n# key N mean median std frac_gt6 frac_gt10 frac_gt15\n")
        for row in fullrows:
            f.write(f"{row[0]} {row[2]} "+ " ".join(f"{v:.8g}" for v in row[3:])+"\n")

    # Matched-pixel scatter: baseline high-RV pixels.
    fig,ax=plt.subplots(figsize=(7.5,6))
    b=rvref[high_base]
    for key,label,_,_ in scenarios[1:]:
        ax.scatter(b,maps[key][1][high_base],s=48,alpha=.8,label=label)
    lo=min(np.nanmin(b),1); hi=max(np.nanmax(b),18)
    ax.plot([lo,hi],[lo,hi],ls="--",lw=1)
    ax.axhline(6,ls=":",lw=1); ax.axhline(10,ls=":",lw=1)
    ax.set_xlabel("Baseline R(V)")
    ax.set_ylabel("Counterfactual R(V)")
    ax.set_title(f"CG10: counterfactuals for baseline R(V) > {A.high_rv:g} pixels")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(A.prefix+"_matched_highrv.png",dpi=220)
    plt.close(fig)

    # Distribution on same fixed high-RV pixels.
    fig,ax=plt.subplots(figsize=(8,5.5))
    data=[maps[k][1][high_base] for k,_,_,_ in scenarios]
    labels=["Baseline","AGB a=10 nm","AGB fC=0.10","Both"]
    ax.boxplot(data,labels=labels,showfliers=True)
    ax.axhline(6,ls="--",lw=1); ax.axhline(10,ls=":",lw=1)
    ax.set_ylabel("R(V)")
    ax.set_title(f"Same {high_base.sum()} baseline high-R(V) pixels")
    fig.tight_layout()
    fig.savefig(A.prefix+"_fixed_pixels_boxplot.png",dpi=220)
    plt.close(fig)

    # Save counterfactual maps for later inspection.
    np.savez_compressed(
        A.prefix+"_maps.npz",
        rv_baseline=maps["baseline"][1],
        rv_agb_radius10=maps["radius10"][1],
        rv_agb_carbon010=maps["carbon010"][1],
        rv_agb_both=maps["both"][1],
        baseline_highrv_mask=high_base,
        aperture_mask=aperture,
        pixel_size_pkpc=A.pixel,
        aperture_radius_pkpc=A.radius,
        agb_test_radius_nm=A.agb_test_radius_nm,
        agb_test_carbon=A.agb_test_carbon,
    )

    print("\nOUTPUTS")
    print("="*76)
    print(A.prefix+"_summary.txt")
    print(A.prefix+"_matched_highrv.png")
    print(A.prefix+"_fixed_pixels_boxplot.png")
    print(A.prefix+"_maps.npz")

if __name__=="__main__":
    main()
