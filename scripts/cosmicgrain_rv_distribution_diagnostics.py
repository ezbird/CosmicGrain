#!/usr/bin/env python3
import argparse,csv,numpy as np,matplotlib.pyplot as plt
def bs(a,f):
 n,m=a.shape
 return a.reshape(n//f,f,m//f,f).sum((1,3))
def mkRv(b,v):
 d=b-v;r=np.full_like(v,np.nan,float);q=np.isfinite(b)&np.isfinite(v)&(v>0)&(d>0);r[q]=v[q]/d[q];return r
def lab(x): return f"{x*1000:.0f} pc" if x<1 else f"{x:g} kpc"
def get(z,*ks):
 for k in ks:
  if k in z:return np.asarray(z[k],float),k
 return None,None
def hexp(x,y,xl,title,out,logx=False):
 q=np.isfinite(x)&np.isfinite(y)
 if logx:q&=x>0
 fig,ax=plt.subplots(figsize=(7.2,5.3))
 h=ax.hexbin(x[q],y[q],gridsize=55,bins="log",mincnt=1,xscale="log" if logx else "linear")
 fig.colorbar(h,ax=ax,label="log10 pixel count");ax.set_xlabel(xl);ax.set_ylabel(r"$R_V$");ax.set_title(title)
 ax.grid(True,alpha=.15);ax.tick_params(direction="in",top=True,right=True);fig.tight_layout()
 fig.savefig(out+".png",dpi=300,bbox_inches="tight");fig.savefig(out+".pdf",bbox_inches="tight");plt.close(fig)
p=argparse.ArgumentParser();p.add_argument("npz");p.add_argument("--count-cuts",nargs="+",type=int,default=[1,2,5,10]);p.add_argument("--signal-frac",type=float,default=1e-3);p.add_argument("--coarse-pixel-size",type=float,default=1.0);p.add_argument("--prefix",default="halo441_rv_diagnostics");a=p.parse_args()
z=np.load(a.npz);tb=z["tau_b"].astype(float);tv=z["tau_v"].astype(float);dm=z["dust_mass_code"].astype(float);nc=z["dust_count"].astype(int);dx=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0]);rv=mkRv(tb,tv)
base=np.isfinite(rv)&(tv>=a.signal_frac*np.nanmax(tv));rad,rk=get(z,"mean_grain_radius_massweighted_nm","mean_grain_radius_nm");small,sk=get(z,"small_grain_mass_fraction","small_grain_fraction");carbon,ck=get(z,"carbon_mass_fraction_massweighted","carbon_fraction_massweighted")
print("="*80);print("R_V DISTRIBUTION DIAGNOSTICS");print("="*80);print("native dx",dx,"grain fields:",rk,sk,ck)
# count-cut histograms
v0=rv[base&(nc>=min(a.count_cuts))];edges=np.linspace(max(0,np.percentile(v0,.5)),np.percentile(v0,99.5),75);rows=[]
fig,ax=plt.subplots(figsize=(8.3,5.4))
for c in a.count_cuts:
 m=base&(nc>=c);v=rv[m]
 if not len(v):continue
 q=np.percentile(v,[1,16,50,84,99]);rows.append([c,len(v),m.mean(),v.mean(),v.std(),*q])
 ax.hist(v,bins=edges,density=True,histtype="step",lw=1.8,label=fr"$N_{{dust}}\geq{c}$")
 print(f"N>={c}: N={len(v)}, mean={v.mean():.3f}, std={v.std():.3f}, p16/50/84={q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}, p99={q[4]:.3f}")
ax.set(xlabel=r"$R_V$",ylabel="Probability density",title=f"halo441 {dx*1000:.0f} pc $R_V$: particle-count sensitivity");ax.legend(frameon=False);ax.grid(True,alpha=.18);fig.tight_layout();fig.savefig(a.prefix+"_countcuts.png",dpi=300,bbox_inches="tight");fig.savefig(a.prefix+"_countcuts.pdf",bbox_inches="tight");plt.close(fig)
with open(a.prefix+"_countcuts.csv","w",newline="") as f:
 w=csv.writer(f);w.writerow(["Ncut","N","fraction","mean","std","p01","p16","median","p84","p99"]);w.writerows(rows)
m=base&(nc>=1);sig=dm/dx**2
hexp(tv[m],rv[m],r"$\tau_V$",f"halo441 {dx*1000:.0f} pc: $R_V$ vs extinction strength",a.prefix+"_rv_vs_tauv",True)
hexp(sig[m],rv[m],r"$\Sigma_{dust}$ (code mass kpc$^{-2}$)",f"halo441 {dx*1000:.0f} pc: $R_V$ vs dust surface density",a.prefix+"_rv_vs_dustdensity",True)
hexp(nc[m].astype(float),rv[m],r"$N_{dust}$",f"halo441 {dx*1000:.0f} pc: $R_V$ vs dust-particle count",a.prefix+"_rv_vs_count",True)
if rad is not None:
 mm=m&np.isfinite(rad);hexp(rad[mm],rv[mm],"Mass-weighted mean grain radius (nm)",f"halo441 {dx*1000:.0f} pc: $R_V$ vs mean grain radius",a.prefix+"_rv_vs_radius",True)
if small is not None:
 mm=m&np.isfinite(small);hexp(small[mm],rv[mm],"Small-grain mass fraction",f"halo441 {dx*1000:.0f} pc: $R_V$ vs small-grain fraction",a.prefix+"_rv_vs_smallgrain")
if carbon is not None:
 mm=m&np.isfinite(carbon);hexp(carbon[mm],rv[mm],"Mass-weighted carbon fraction",f"halo441 {dx*1000:.0f} pc: $R_V$ vs carbon fraction",a.prefix+"_rv_vs_carbon")
# coarse mixing
f=int(round(a.coarse_pixel_size/dx))
if not np.isclose(a.coarse_pixel_size/dx,f):raise ValueError("coarse size must be integer multiple of native")
ny,nx=rv.shape;tbc,tvc,ncc=bs(tb,f),bs(tv,f),bs(nc,f);rvc=mkRv(tbc,tvc);cm=np.isfinite(rvc)&(ncc>=1)&(tvc>=a.signal_frac*np.nanmax(tvc))
r4=rv.reshape(ny//f,f,nx//f,f);m4=m.reshape(ny//f,f,nx//f,f);mean=np.full(rvc.shape,np.nan);med=mean.copy();mn=mean.copy();mx=mean.copy();nv=np.zeros(rvc.shape,int)
for iy in range(rvc.shape[0]):
 for ix in range(rvc.shape[1]):
  vv=r4[iy,:,ix,:][m4[iy,:,ix,:]];nv[iy,ix]=len(vv)
  if len(vv):mean[iy,ix]=vv.mean();med[iy,ix]=np.median(vv);mn[iy,ix]=vv.min();mx[iy,ix]=vv.max()
q=cm&(nv>0)&np.isfinite(med);fig,axs=plt.subplots(1,2,figsize=(11,4.7))
for ax,x,xl in [(axs[0],mean[q],"Mean native $R_V$"),(axs[1],med[q],"Median native $R_V$")]:
 h=ax.hexbin(x,rvc[q],gridsize=45,bins="log",mincnt=1);lo=min(np.percentile(x,1),np.percentile(rvc[q],1));hi=max(np.percentile(x,99),np.percentile(rvc[q],99));ax.plot([lo,hi],[lo,hi],"k--",lw=1);ax.set_xlabel(xl);ax.set_ylabel(f"Recomputed {a.coarse_pixel_size:g} kpc $R_V$");ax.grid(True,alpha=.15)
fig.colorbar(h,ax=axs.ravel().tolist(),label="log10 coarse-pixel count");fig.suptitle(f"halo441: {dx*1000:.0f} pc R_V mixed into {a.coarse_pixel_size:g} kpc pixels");fig.subplots_adjust(left=.08,right=.9,bottom=.13,top=.86,wspace=.28);fig.savefig(a.prefix+"_coarse_mixing.png",dpi=300,bbox_inches="tight");fig.savefig(a.prefix+"_coarse_mixing.pdf",bbox_inches="tight");plt.close(fig)
np.savetxt(a.prefix+"_coarse_mixing.csv",np.column_stack([rvc[q],mean[q],med[q],mn[q],mx[q],nv[q]]),delimiter=",",header="Rv_coarse,Rv_native_mean,Rv_native_median,Rv_native_min,Rv_native_max,N_native_valid",comments="")
print(f"coarse mixing: {q.sum()} cells; median coarse={np.median(rvc[q]):.3f}, native mean={np.median(mean[q]):.3f}, native median={np.median(med[q]):.3f}, Nvalid={np.median(nv[q]):.1f}/{f*f}")
print("Saved all diagnostics with prefix",a.prefix)
