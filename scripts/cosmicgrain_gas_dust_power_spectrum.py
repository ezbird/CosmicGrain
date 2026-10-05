#!/usr/bin/env python3
import argparse,glob,numpy as np,matplotlib.pyplot as plt,h5py
def bs(a,f):
 n,m=a.shape;return a.reshape(n//f,f,m//f,f).sum((1,3))
def rot(j):
 j=np.array(j,float);j/=np.linalg.norm(j);z=np.array([0.,0.,1.]);x=np.cross(z,j)
 x=np.array([1.,0.,0.]) if np.linalg.norm(x)<1e-10 else x/np.linalg.norm(x);y=np.cross(j,x);return np.vstack([x,y,j])
def ps(a,dx):
 d=a-a.mean();F=dx**2*np.fft.fftshift(np.fft.fft2(d));P=np.abs(F)**2/(a.size*dx**2);ny,nx=a.shape
 kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx));ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dx));X,Y=np.meshgrid(kx,ky);K=np.hypot(X,Y);kf=2*np.pi/min(nx*dx,ny*dx);kn=np.pi/dx;ed=np.arange(.5*kf,kn+1.5*kf,kf);k=[];p=[];n=[]
 for lo,hi in zip(ed[:-1],ed[1:]):
  q=(K>=lo)&(K<hi)
  if q.any():k.append(K[q].mean());p.append(P[q].mean());n.append(q.sum())
 return np.array(k),np.array(p),np.array(n),kn
def lab(x):return f"{x*1000:.0f} pc" if x<1 else f"{x:g} kpc"
ap=argparse.ArgumentParser();ap.add_argument("snapshot");ap.add_argument("master_npz");ap.add_argument("--center",nargs=3,type=float,required=True);ap.add_argument("--jhat",nargs=3,type=float,required=True);ap.add_argument("--plot-radius",type=float,default=25);ap.add_argument("--zmax",type=float,default=20);ap.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5,1,2]);ap.add_argument("--min-modes",type=int,default=8);ap.add_argument("--prefix",default="halo441_gas_dust_power_spectrum");a=ap.parse_args()
files=sorted(glob.glob(a.snapshot));z=np.load(a.master_npz);dust0=z["dust_mass_code"].astype(float);dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0]);ny,nx=dust0.shape
if not files:raise FileNotFoundError(a.snapshot)
with h5py.File(files[0],"r") as f:
 h=f["Header"].attrs.get("HubbleParam",f.get("Parameters",{}).attrs.get("HubbleParam",None) if "Parameters" in f else None);sc=float(f["Header"].attrs.get("Time",1));box=float(f["Header"].attrs["BoxSize"])
if h is None:raise RuntimeError("HubbleParam not found")
R=rot(a.jhat);c=np.array(a.center);xs=[];ys=[];ms=[];ng=0
for fn in files:
 with h5py.File(fn,"r") as f:
  if "PartType0" not in f:continue
  g=f["PartType0"];p=np.asarray(g["Coordinates"],float);m=np.asarray(g["Masses"],float) if "Masses" in g else np.full(len(p),f["Header"].attrs["MassTable"][0])
 d=p-c;d-=box*np.rint(d/box);d*=sc/float(h);u=d@R.T;q=np.abs(u[:,2])<a.zmax;ng+=q.sum();xs.append(u[q,0]);ys.append(u[q,1]);ms.append(m[q])
x=np.concatenate(xs);y=np.concatenate(ys);m=np.concatenate(ms);edges=np.linspace(-a.plot_radius,a.plot_radius,nx+1);gas0=np.histogram2d(y,x,bins=[edges,edges],weights=m)[0]
print("="*80);print("GAS + DUST FRACTIONAL SURFACE-DENSITY POWER SPECTRA");print("="*80);print(f"a={sc:g} h={float(h):g} gas particles in slab={ng:,}");print("rotation matrix:\\n",R)
S={"gas":[],"dust":[]}
for dx in a.pixel_sizes:
 f=int(round(dx/dx0))
 if not np.isclose(dx/dx0,f):raise ValueError("pixel size must be integer multiple of native")
 gm,dm=bs(gas0,f),bs(dust0,f)
 for name,mass in [("gas",gm),("dust",dm)]:
  sig=mass/dx**2;delta=(sig-sig.mean())/sig.mean();k,P,n,kn=ps(delta,dx);S[name].append((dx,k,k*k*P,n,kn))
 print(f"{lab(dx):>6}: dMgas/M={gm.sum()/gas0.sum()-1:+.2e}, dMdust/M={dm.sum()/dust0.sum()-1:+.2e}, gas occupied={100*np.mean(gm>0):.1f}%, dust occupied={100*np.mean(dm>0):.1f}%")
print("\\nMedian P/P_native below 0.5 coarse kNyq:")
for name in S:
 print(name);_,kr,yr,_,_=S[name][0]
 for dx,k,yv,n,kn in S[name][1:]:
  q=(k<=.5*kn)&(n>=a.min_modes)&(yv>0);ref=np.interp(k[q],kr,yr,left=np.nan,right=np.nan);rr=yv[q]/ref;rr=rr[np.isfinite(rr)&(rr>0)]
  if len(rr):
   v=np.percentile(rr,[16,50,84]);print(f" {lab(dx):>6}: {v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}")
fig,axs=plt.subplots(1,2,figsize=(11.4,4.7))
for ax,name,title in zip(axs,["gas","dust"],["Gas surface density","Dust surface density"]):
 for dx,k,yv,n,kn in S[name]:
  q=(n>=a.min_modes)&(k>0)&(yv>0);ax.loglog(k[q],yv[q],lw=1.7,label=lab(dx))
 ax.set_title(title);ax.set_xlabel(r"$k$ (kpc$^{-1}$)");ax.set_ylabel(r"$k^2P_{\delta}(k)$");ax.grid(True,which="both",alpha=.18);ax.tick_params(direction="in",which="both",top=True,right=True)
fig.suptitle("halo441 power spectrum",fontsize=15);H,L=axs[0].get_legend_handles_labels();fig.legend(H,L,loc="lower center",ncol=len(L),frameon=False);fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(a.prefix+".png",dpi=300,bbox_inches="tight");fig.savefig(a.prefix+".pdf",bbox_inches="tight");plt.close(fig)
print("\\nSaved",a.prefix+".png and .pdf")
