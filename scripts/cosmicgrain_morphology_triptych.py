#!/usr/bin/env python3
import argparse, glob, h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load(files,ptype,fields):
    d={k:[] for k in fields}
    for fn in files:
        with h5py.File(fn,"r") as f:
            g=f.get(f"PartType{ptype}")
            if g is None: continue
            n=len(g["Coordinates"])
            for k in fields:
                if k in g: d[k].append(g[k][:])
                elif k=="Masses":
                    m=f["Header"].attrs["MassTable"][ptype]
                    if m<=0: raise KeyError(f"Masses missing for PartType{ptype}")
                    d[k].append(np.full(n,m))
                else: raise KeyError(f"{k} missing")
    return {k:np.concatenate(v) for k,v in d.items()}

def delta(x,c,L):
    q=x-c; q-=L*np.rint(q/L); return q

def rot(j):
    z=j/np.linalg.norm(j); ref=np.array([0.,0.,1.])
    if abs(ref@z)>.9: ref=np.array([0.,1.,0.])
    x=np.cross(ref,z); x/=np.linalg.norm(x)
    y=np.cross(z,x); return np.vstack((x,y,z))

def smap(x,y,m,xlim,ylim,pix):
    nx=int(round((xlim[1]-xlim[0])/pix)); ny=int(round((ylim[1]-ylim[0])/pix))
    H,ye,xe=np.histogram2d(y,x,bins=(ny,nx),range=(ylim,xlim),weights=m)
    area=(xlim[1]-xlim[0])*(ylim[1]-ylim[0])/(nx*ny)
    return H/area

def lim(a):
    q=a[a>0]
    lo,hi=np.percentile(q,[2,99.7])
    return max(lo,q.min()), max(hi,lo*1.01)

p=argparse.ArgumentParser(description="Stars face-on | gas face-on | stars edge-on morphology")
p.add_argument("snapshot")
p.add_argument("--center",nargs=3,type=float,required=True)
p.add_argument("--radius",type=float,default=25.)
p.add_argument("--edge-z",type=float,default=10.)
p.add_argument("--orientation-radius",type=float,default=20.)
p.add_argument("--pixel-size",type=float,default=.125)
p.add_argument("--prefix",default="morphology")
p.add_argument("--dpi",type=int,default=250)
a=p.parse_args()

files=sorted(glob.glob(a.snapshot))
if not files: raise FileNotFoundError(a.snapshot)
with h5py.File(files[0],"r") as f:
    h0=f["Header"].attrs; aa=float(h0["Time"]); box=float(h0["BoxSize"]); h=float(h0.get("HubbleParam",1))
z=1/aa-1; c=np.array(a.center)
s=load(files,4,["Coordinates","Velocities","Masses"])
g=load(files,0,["Coordinates","Masses"])
rs=delta(s["Coordinates"].astype(float),c,box)*aa/h
rg=delta(g["Coordinates"].astype(float),c,box)*aa/h
ms=s["Masses"].astype(float)*1e10/h; mg=g["Masses"].astype(float)*1e10/h

rr=np.linalg.norm(rs,axis=1); sel=rr<a.orientation_radius
vb=np.average(s["Velocities"][sel],axis=0,weights=ms[sel])
J=np.sum(np.cross(rs[sel],s["Velocities"][sel]-vb)*ms[sel,None],axis=0)
jh=J/np.linalg.norm(J); R=rot(jh)
rs=rs@R.T; rg=rg@R.T

rad=a.radius
sf=(abs(rs[:,0])<rad)&(abs(rs[:,1])<rad)&(abs(rs[:,2])<rad)
gf=(abs(rg[:,0])<rad)&(abs(rg[:,1])<rad)&(abs(rg[:,2])<rad)
se=(abs(rs[:,0])<rad)&(abs(rs[:,1])<rad)&(abs(rs[:,2])<a.edge_z)

S=smap(rs[sf,0],rs[sf,1],ms[sf],(-rad,rad),(-rad,rad),a.pixel_size)
G=smap(rg[gf,0],rg[gf,1],mg[gf],(-rad,rad),(-rad,rad),a.pixel_size)
E=smap(rs[se,0],rs[se,2],ms[se],(-rad,rad),(-a.edge_z,a.edge_z),a.pixel_size)

print("="*72)
print("COSMICGRAIN MORPHOLOGY TRIPTYCH")
print("="*72)
print(f"a={aa:.8f} z={z:.8f} h={h:g}")
print(f"orientation stars (<{a.orientation_radius:g} kpc): {sel.sum():,}")
print("stellar bulk velocity:",vb)
print("j_hat:",jh)
print("rotation matrix:\n",R)
print(f"stars face-on: N={sf.sum():,}, M={ms[sf].sum():.6e} Msun")
print(f"gas face-on:   N={gf.sum():,}, M={mg[gf].sum():.6e} Msun")

fig,axs=plt.subplots(1,3,figsize=(17,6))
items=[
 (S,(-rad,rad),(-rad,rad),"magma","Stars — Face-on",r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)"),
 (G,(-rad,rad),(-rad,rad),"viridis","Gas — Face-on",r"$\Sigma_{\rm gas}$ ($M_\odot$ kpc$^{-2}$)"),
 (E,(-rad,rad),(-a.edge_z,a.edge_z),"magma","Stars — Edge-on",r"$\Sigma_\star$ ($M_\odot$ kpc$^{-2}$)")
]
for i,(M,xl,yl,cm,title,clab) in enumerate(items):
    v0,v1=lim(M); mm=np.ma.masked_less_equal(M,0)
    im=axs[i].imshow(mm,origin="lower",extent=(*xl,*yl),cmap=cm,
                     norm=LogNorm(vmin=v0,vmax=v1),interpolation="nearest",aspect="equal")
    axs[i].set_title(title)
    axs[i].set_xlabel(r"$x'$ (kpc)")
    axs[i].set_ylabel(r"$y'$ (kpc)" if i<2 else r"$z'$ (kpc)")
    cb=fig.colorbar(im,ax=axs[i],orientation="horizontal",pad=.13,fraction=.055)
    cb.set_label(clab)

fig.tight_layout()
fig.savefig(a.prefix+"_triptych.png",dpi=a.dpi,bbox_inches="tight")
fig.savefig(a.prefix+"_triptych.pdf",bbox_inches="tight")
plt.close(fig)
np.savez_compressed(a.prefix+"_maps.npz",stellar_faceon=S,gas_faceon=G,stellar_edgeon=E,
                    center_ckpch=c,j_hat=jh,rotation_matrix=R,stellar_bulk_velocity=vb,
                    a=aa,z=z,h=h,pixel_size_kpc=a.pixel_size,
                    face_radius_kpc=rad,edge_z_kpc=a.edge_z)
print("Saved",a.prefix+"_triptych.png/pdf and "+a.prefix+"_maps.npz")
