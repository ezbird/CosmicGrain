#!/usr/bin/env python3
"""
cosmicgrain_stellar_morphology_quant.py

Quantify the z=0 stellar morphology of a GADGET-4 galaxy using the same
center/orientation convention as the morphology triptych.

Outputs:
  * circularity distribution using epsilon = j_z / (R * V_circ(R))
  * face-on stellar surface-density profile
  * vertical stellar density profile
  * simple exponential fits for disk scale length R_d and scale height h_z
  * summary text + NPZ

IMPORTANT: epsilon here uses j_circ(R) = R V_circ(R), where
V_circ(R)=sqrt(G M(<R)/R) from all particle types. This is a practical
circularity proxy, not the more expensive energy-matched j_circ(E).
"""
import argparse, glob, h5py, numpy as np, matplotlib.pyplot as plt

G = 4.30091e-6  # kpc (km/s)^2 / Msun

def scalar(x): return float(np.asarray(x).reshape(-1)[0])

def info(fn):
    with h5py.File(fn,"r") as f:
        H=f["Header"].attrs
        a=scalar(H["Time"]); box=scalar(H["BoxSize"])
        if "Parameters" in f and "HubbleParam" in f["Parameters"].attrs:
            h=scalar(f["Parameters"].attrs["HubbleParam"])
        elif "HubbleParam" in H: h=scalar(H["HubbleParam"])
        else: raise RuntimeError("HubbleParam not found; refusing h=1 fallback")
    return a,1/a-1,box,h

def load(files,t,need_vel=False):
    C=[]; M=[]; V=[]
    for fn in files:
        with h5py.File(fn,"r") as f:
            name=f"PartType{t}"
            if name not in f: continue
            g=f[name]; n=len(g["Coordinates"]); C.append(g["Coordinates"][:])
            if "Masses" in g: M.append(g["Masses"][:])
            else:
                m=f["Header"].attrs["MassTable"][t]
                if m<=0: raise RuntimeError(f"No masses for type {t}")
                M.append(np.full(n,m))
            if need_vel: V.append(g["Velocities"][:])
    if not C: return None
    out={"Coordinates":np.concatenate(C),"Masses":np.concatenate(M)}
    if need_vel: out["Velocities"]=np.concatenate(V)
    return out

def dperiodic(x,c,L):
    d=x.astype(float)-c; d-=L*np.rint(d/L); return d

def rotation(j):
    z=j/np.linalg.norm(j); ref=np.array([0.,0.,1.])
    if abs(ref@z)>.9: ref=np.array([0.,1.,0.])
    x=np.cross(ref,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.vstack((x,y,z))

def binned_profile(x,w,edges,volume=None):
    mass=np.histogram(x,bins=edges,weights=w)[0]
    if volume is not None: mass=mass/volume
    ctr=.5*(edges[:-1]+edges[1:])
    return ctr,mass

def expfit(x,y,xmin,xmax):
    m=(x>=xmin)&(x<=xmax)&np.isfinite(y)&(y>0)
    if m.sum()<3: return np.nan,np.nan,m
    slope,intercept=np.polyfit(x[m],np.log(y[m]),1)
    scale=-1/slope if slope<0 else np.nan
    return scale,intercept,m

p=argparse.ArgumentParser()
p.add_argument("snapshot")
p.add_argument("--center",nargs=3,type=float,required=True)
p.add_argument("--orientation-radius",type=float,default=20.)
p.add_argument("--analysis-radius",type=float,default=30.)
p.add_argument("--disk-zmax",type=float,default=5.)
p.add_argument("--radial-bin",type=float,default=.5)
p.add_argument("--vertical-bin",type=float,default=.25)
p.add_argument("--rd-fit",nargs=2,type=float,default=[3.,12.],
               metavar=("RMIN","RMAX"))
p.add_argument("--hz-fit",nargs=2,type=float,default=[.5,3.],
               metavar=("ZMIN","ZMAX"))
p.add_argument("--prefix",default="stellar_morphology_quant")
a=p.parse_args()

files=sorted(glob.glob(a.snapshot))
if not files: raise FileNotFoundError(a.snapshot)
aa,z,box,h=info(files[0]); c=np.array(a.center,float)

# Stars plus all mass-bearing particle types for M(<r).
s=load(files,4,True)
parts=[]
for t in range(7):
    q=load(files,t,False)
    if q is not None: parts.append((t,q))

rs=dperiodic(s["Coordinates"],c,box)*aa/h
ms=s["Masses"].astype(float)*1e10/h
vs=s["Velocities"].astype(float)

r3=np.linalg.norm(rs,axis=1)
ori=r3<a.orientation_radius
vb=np.average(vs[ori],axis=0,weights=ms[ori])
J=np.sum(np.cross(rs[ori],vs[ori]-vb)*ms[ori,None],axis=0)
jh=J/np.linalg.norm(J); Rot=rotation(jh)
r=rs@Rot.T; v=(vs-vb)@Rot.T
R=np.hypot(r[:,0],r[:,1])
jz=r[:,0]*v[:,1]-r[:,1]*v[:,0]

# Enclosed total mass curve from all particle types.
rall=[]; mall=[]
for t,q in parts:
    rp=np.linalg.norm(dperiodic(q["Coordinates"],c,box)*aa/h,axis=1)
    mp=q["Masses"].astype(float)*1e10/h
    keep=rp <= max(a.analysis_radius*1.5,50.)
    rall.append(rp[keep]); mall.append(mp[keep])
rall=np.concatenate(rall); mall=np.concatenate(mall)
order=np.argsort(rall); rall=rall[order]; cum=np.cumsum(mall[order])
Menc=np.interp(np.maximum(R,1e-6),rall,cum,left=cum[0],right=cum[-1])
vc=np.sqrt(G*Menc/np.maximum(R,1e-6))
jc=R*vc
eps=np.full_like(jz,np.nan,dtype=float)
ok=(jc>0)&np.isfinite(jc); eps[ok]=jz[ok]/jc[ok]

ana=r3<a.analysis_radius
epsa=eps[ana]; msa=ms[ana]
finite=np.isfinite(epsa)
disk70=np.sum(msa[finite & (epsa>0.7)])/np.sum(msa[finite])
disk50=np.sum(msa[finite & (epsa>0.5)])/np.sum(msa[finite])
counter=np.sum(msa[finite & (epsa<0)])/np.sum(msa[finite])

# Radial stellar surface density in |z| < disk_zmax.
redges=np.arange(0,a.analysis_radius+a.radial_bin,a.radial_bin)
rsel=ana&(np.abs(r[:,2])<a.disk_zmax)
areas=np.pi*(redges[1:]**2-redges[:-1]**2)
rc,Sigma=binned_profile(R[rsel],ms[rsel],redges,areas)
Rd,rdint,rdfit=expfit(rc,Sigma,*a.rd_fit)

# Vertical density profile within a radial disk annulus chosen to reduce bulge.
# Default annulus = same range as radial exponential fit.
absz=np.abs(r[:,2])
zedges=np.arange(0,10+a.vertical_bin,a.vertical_bin)
zsel=ana&(R>=a.rd_fit[0])&(R<=a.rd_fit[1])
ann_area=np.pi*(a.rd_fit[1]**2-a.rd_fit[0]**2)
# bins in |z| combine +z/-z; physical volume = 2*area*dz
vol=np.full(len(zedges)-1,2*ann_area*a.vertical_bin)
zc,rho=binned_profile(absz[zsel],ms[zsel],zedges,vol)
hz,hzint,hzfit=expfit(zc,rho,*a.hz_fit)

print("="*78)
print("STELLAR MORPHOLOGY QUANTIFICATION")
print("="*78)
print(f"a={aa:.8f} z={z:.8f} h={h:.8f}")
print(f"Nstar(<{a.analysis_radius:g} pkpc)={ana.sum():,}")
print("j_hat =",jh)
print(f"circularity proxy: epsilon = j_z/[R V_circ(R)]")
print(f"mass fraction epsilon > 0.7 = {disk70:.4f}")
print(f"mass fraction epsilon > 0.5 = {disk50:.4f}")
print(f"mass fraction epsilon < 0   = {counter:.4f}")
print(f"radial exponential fit: {a.rd_fit[0]:g}-{a.rd_fit[1]:g} kpc")
print(f"R_d = {Rd:.3f} kpc")
print(f"vertical fit annulus: R={a.rd_fit[0]:g}-{a.rd_fit[1]:g} kpc")
print(f"vertical exponential fit: |z|={a.hz_fit[0]:g}-{a.hz_fit[1]:g} kpc")
print(f"h_z = {hz:.3f} kpc")

fig,ax=plt.subplots(figsize=(6.5,4.8))
eb=np.linspace(-1.5,1.5,61)
hist,_=np.histogram(epsa[finite],bins=eb,weights=msa[finite],density=False)
hist=hist/hist.sum()
ec=.5*(eb[:-1]+eb[1:])
ax.step(ec,hist,where="mid")
ax.axvline(.7,ls="--",lw=1)
ax.set_xlabel(r"Circularity proxy $\epsilon=j_z/[R\,V_{\rm circ}(R)]$")
ax.set_ylabel("Stellar mass fraction per bin")
ax.set_xlim(-1.5,1.5)
fig.tight_layout()
fig.savefig(a.prefix+"_circularity.png",dpi=240,bbox_inches="tight")
fig.savefig(a.prefix+"_circularity.pdf",bbox_inches="tight")
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,4.8))
m=Sigma>0
ax.semilogy(rc[m],Sigma[m],"o-",ms=3)
if np.isfinite(Rd):
    xx=np.linspace(a.rd_fit[0],a.rd_fit[1],200)
    ax.plot(xx,np.exp(rdint-xx/Rd),ls="--",
            label=rf"$R_d={Rd:.2f}$ kpc")
    ax.legend()
ax.set_xlabel("Cylindrical radius (kpc)")
ax.set_ylabel(r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)")
fig.tight_layout()
fig.savefig(a.prefix+"_radial_profile.png",dpi=240,bbox_inches="tight")
fig.savefig(a.prefix+"_radial_profile.pdf",bbox_inches="tight")
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.5,4.8))
m=rho>0
ax.semilogy(zc[m],rho[m],"o-",ms=3)
if np.isfinite(hz):
    xx=np.linspace(a.hz_fit[0],a.hz_fit[1],200)
    ax.plot(xx,np.exp(hzint-xx/hz),ls="--",
            label=rf"$h_z={hz:.2f}$ kpc")
    ax.legend()
ax.set_xlabel(r"$|z'|$ (kpc)")
ax.set_ylabel(r"$\rho_\star$ ($M_\odot$ kpc$^{-3}$)")
fig.tight_layout()
fig.savefig(a.prefix+"_vertical_profile.png",dpi=240,bbox_inches="tight")
fig.savefig(a.prefix+"_vertical_profile.pdf",bbox_inches="tight")
plt.close(fig)

np.savez_compressed(a.prefix+"_data.npz",epsilon=epsa[finite],
                    epsilon_mass=msa[finite],R_profile=rc,Sigma_star=Sigma,
                    z_profile=zc,rho_star=rho,Rd_kpc=Rd,hz_kpc=hz,
                    disk_fraction_eps07=disk70,disk_fraction_eps05=disk50,
                    counterrotating_fraction=counter,j_hat=jh,rotation_matrix=Rot)

with open(a.prefix+"_summary.txt","w") as f:
    f.write(f"R_d_kpc {Rd}\n")
    f.write(f"h_z_kpc {hz}\n")
    f.write(f"disk_fraction_eps_gt_0.7 {disk70}\n")
    f.write(f"disk_fraction_eps_gt_0.5 {disk50}\n")
    f.write(f"counterrotating_fraction_eps_lt_0 {counter}\n")
    f.write("circularity_definition j_z/[R V_circ(R)]\n")

print("\nSaved circularity, radial-profile, vertical-profile PNG/PDF, data NPZ, and summary TXT.")
