#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def bsum(a,f):
    n,m=a.shape
    return a.reshape(n//f,f,m//f,f).sum((1,3))

def rvmap(tb,tv):
    d=tb-tv
    r=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    r[q]=tv[q]/d[q]
    return r

def save(path,a,q,dx,L,note):
    h=f"""CosmicGrain halo441 snapshot 029 face-on map
quantity: {q}
pixel_size_kpc: {dx}
region_side_kpc: {L}
grid_shape: {a.shape[0]} {a.shape[1]}
matrix convention: rows=y, columns=x
{note}"""
    np.savetxt(path,a,fmt="%.10e",header=h,comments="# ")

ap=argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--square-size",type=float,default=20.)
ap.add_argument("--pixel-sizes",nargs="+",type=float,default=[.125,.25,.5])
ap.add_argument("--outdir",default="halo441_maps_for_FT")
a=ap.parse_args()

z=np.load(a.npz)
tb0=np.asarray(z["tau_b"],float); tv0=np.asarray(z["tau_v"],float)
nc0=np.asarray(z["dust_count"],float)
dm0=np.asarray(z["dust_mass_code"],float) if "dust_mass_code" in z.files else None
dx0=float(np.asarray(z["pixel_size_pkpc"]).ravel()[0])

N=tb0.shape[0]; nn=int(round(a.square_size/dx0))
if not np.isclose(nn*dx0,a.square_size) or nn>N: raise ValueError("bad square size")
lo=(N-nn)//2; hi=lo+nn
tb0=tb0[lo:hi,lo:hi]; tv0=tv0[lo:hi,lo:hi]; nc0=nc0[lo:hi,lo:hi]
if dm0 is not None: dm0=dm0[lo:hi,lo:hi]

out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
summary=[]
for dx in a.pixel_sizes:
    f=int(round(dx/dx0))
    if not np.isclose(f*dx0,dx) or nn%f: raise ValueError(f"incompatible dx {dx}")
    tb=bsum(tb0,f); tv=bsum(tv0,f); nc=bsum(nc0,f); rv=rvmap(tb,tv)
    tag=f"{int(round(dx*1000))}pc" if dx<1 else f"{dx:g}kpc"
    save(out/f"{tag}_tauB.txt",tb,"tau_B",dx,a.square_size,"dimensionless; no science mask")
    save(out/f"{tag}_tauV.txt",tv,"tau_V",dx,a.square_size,"dimensionless; no science mask")
    save(out/f"{tag}_RV_unmasked.txt",rv,"R(V)",dx,a.square_size,
         "R(V)=tau_V/(tau_B-tau_V); NaN only where ratio is invalid")
    save(out/f"{tag}_dust_count.txt",nc.astype(int),"dust particle count",dx,a.square_size,
         "projected PartType6 superparticle count")
    if dm0 is not None:
        sig=bsum(dm0,f)/(dx*dx)
        save(out/f"{tag}_dust_surface_density_code_per_kpc2.txt",sig,
             "dust surface density",dx,a.square_size,"dust-mass code units per kpc^2")
    q=np.isfinite(rv)
    summary.append((tag,dx,rv.shape[0],q.sum(),q.mean(),np.nanmedian(rv),np.nanstd(rv)))

(out/"README.txt").write_text(f"""COSMICGRAIN HALO441 SNAPSHOT 029 — ASCII MAPS FOR INDEPENDENT FT
================================================================

Purpose
-------
Portable 2-D matrices for an independent Fourier/power-spectrum calculation.
No Python-specific format or CosmicGrain analysis code is required.

Geometry
--------
Galaxy: halo441 / CG10
Snapshot: 029 (z=0)
Map: face-on
Region: central {a.square_size:g} x {a.square_size:g} kpc
Matrix convention: rows=y, columns=x; array center is galaxy center.

Adopted simulation center (ckpc/h):
  [29459.01585206, 23170.70462420, 27860.34910496]

Face-on angular-momentum direction:
  j_hat = [0.57366824, -0.31402785, 0.75649935]

Files
-----
*_tauB.txt       B-band optical depth
*_tauV.txt       V-band optical depth
*_RV_unmasked.txt
                 R(V) = tau_V / (tau_B - tau_V)
*_dust_count.txt projected PartType6 dust-superparticle count
*_dust_surface_density_code_per_kpc2.txt
                 dust surface density, when available in source NPZ

Masking
-------
RV_unmasked has NO minimum-particle cut and NO relative tau/signal cut.
NaN is used only where the R(V) ratio is mathematically invalid.

Coarse graining
---------------
R(V) is never averaged directly. For every coarse pixel, tau_B and tau_V are
summed first and R(V) is recomputed:
  R(V)_coarse = sum(tau_V) / [sum(tau_B) - sum(tau_V)].

ASCII format
------------
Whitespace-separated matrices. Header lines begin '#'. NaN is written 'nan'.
Pixel size is recorded in every file header.

FT note
-------
For angular spatial wavenumber k, use physical pixel size to express k in
kpc^-1 (including the usual 2*pi relative to cycles/kpc). Missing-pixel/window
treatment is intentionally left to the recipient because it is part of the
independent masking test.
""")

with open(out/"MAP_SUMMARY.txt","w") as f:
    f.write("# tag pixel_kpc Nside finite_RV finite_fraction median_RV std_RV\n")
    for r in summary:
        f.write(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]:.8f} {r[5]:.8f} {r[6]:.8f}\n")

print("Wrote package to",out)
for r in summary:
    print(f"{r[0]}: {r[2]}x{r[2]}")
