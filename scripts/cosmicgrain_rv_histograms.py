#!/usr/bin/env python3
import argparse,csv,numpy as np,matplotlib.pyplot as plt
def bs(x,f):
 n,m=x.shape; return x.reshape(n//f,f,m//f,f).sum((1,3))
def rv(tb,tv):
 d=tb-tv;r=np.full_like(tv,np.nan,float);q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0);r[q]=tv[q]/d[q];return r
def lab(x):return f"{x*1000:.0f} pc" if x<1 else f"{x:g} kpc"
p=argparse.ArgumentParser();p.add_argument("npz");p.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1,2]);p.add_argument("--min-count",type=int,default=1);p.add_argument("--signal-frac",type=float,default=1e-3);p.add_argument("--bins",type=int,default=70);p.add_argument("--xmin",type=float);p.add_argument("--xmax",type=float);p.add_argument("--prefix",default="halo441_rv_histograms");a=p.parse_args()
z=np.load(a.npz);tb0=z["tau_b"].astype(float);tv0=z["tau_v"].astype(float);nc0=z["dust_count"].astype(int);dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])
D=[];rows=[]
print("="*84);print("R_V DISTRIBUTIONS BY MAP PIXEL SIZE");print("="*84)
for dx in a.pixel_sizes:
 f=int(round(dx/dx0))
 if not np.isclose(dx/dx0,f):raise ValueError("pixel size not integer multiple of native")
 tb,tv,nc=bs(tb0,f),bs(tv0,f),bs(nc0,f).astype(int);r=rv(tb,tv);th=a.signal_frac*np.nanmax(tv);m=np.isfinite(r)&(nc>=a.min_count)&(tv>=th);v=r[m]
 q=np.percentile(v,[1,16,50,84,99]);row=[dx,len(v),m.mean(),v.mean(),v.std(),q[2],q[0],q[1],q[3],q[4],v.min(),v.max(),th];rows.append(row);D.append((dx,v))
 print(f"{lab(dx):>6}: N={len(v):6d} ({100*m.mean():6.2f}%) mean={v.mean():7.3f} std={v.std():7.3f} p16/med/p84={q[1]:6.3f}/{q[2]:6.3f}/{q[3]:6.3f} p01/p99={q[0]:6.3f}/{q[4]:6.3f}")
allv=np.concatenate([v for _,v in D]);xmin=a.xmin if a.xmin is not None else max(0,np.percentile(allv,.5));xmax=a.xmax if a.xmax is not None else np.percentile(allv,99.5);edges=np.linspace(xmin,xmax,a.bins+1)
fig,ax=plt.subplots(figsize=(8.4,5.5))
for dx,v in D: ax.hist(v,bins=edges,density=True,histtype="step",lw=1.8,label=lab(dx))
ax.set_xlabel(r"$R_V$");ax.set_ylabel("Probability density");ax.set_title("halo441 $R_V$ distribution");ax.grid(True,alpha=.18);ax.tick_params(direction="in",top=True,right=True);ax.legend(frameon=False,title="Pixel size");fig.tight_layout();fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight");fig.savefig(a.prefix+".pdf",bbox_inches="tight");plt.close(fig)
hdr=["pixel_kpc","N_valid","valid_fraction","mean","std","median","p01","p16","p84","p99","minimum","maximum","tauV_threshold"]
with open(a.prefix+"_stats.csv","w",newline="") as f:w=csv.writer(f);w.writerow(hdr);w.writerows(rows)
print(f"\nDisplay range R_V={xmin:.3f}..{xmax:.3f}");print("Saved PNG, PDF, and stats CSV with prefix:",a.prefix)
