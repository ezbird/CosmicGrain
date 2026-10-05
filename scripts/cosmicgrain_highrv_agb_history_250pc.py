#!/usr/bin/env python3
"""
cosmicgrain_highrv_agb_history_250pc.py

Trace the exact AGB-origin PartType6 superparticles that land in the CURRENT
CG10 baseline R(V)>10 pixels.

Outputs present-day grain properties, dust formation scale factor and age,
temperature, current face-on position, birth position (where usable), optical
weights, and per-pixel source purity.

The script reconstructs the production 250-pc R(V) map with the exact optics
functions imported from cosmicgrain_rv_maps.py and refuses interpretation
unless the map and particle counts reproduce the exported references.

DustFormationTime is treated as formation scale factor a_form. Dust ages are
computed from the flat LCDM cosmology supplied on the command line.
"""

import argparse, glob, re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cosmicgrain_rv_maps as rvcore

RHO_C = 2.24
RHO_SIL = 3.50
SRC_AGB = 1
MPC_KM = 3.0856775814913673e19
SEC_PER_GYR = 365.25 * 86400.0 * 1.0e9

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]

def periodic_delta(coords, center, box):
    d = coords - center[None, :]
    d -= box * np.rint(d / box)
    return d

def cosmic_age_gyr(a, h, om, ol):
    """Flat matter+Lambda analytic age; radiation neglected."""
    a = np.asarray(a, dtype=float)
    H0 = 100.0 * h / MPC_KM  # s^-1
    pref = 2.0 / (3.0 * H0 * np.sqrt(ol))
    val = pref * np.arcsinh(np.sqrt(ol / om) * np.power(a, 1.5))
    return val / SEC_PER_GYR

def load_dust(files):
    fields = {
        "coords": [], "mass": [], "radius": [], "carbon": [], "source": [],
        "form": [], "temp": [], "birth": [], "pid": []
    }
    a = box = h = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if a is None:
                a = float(f["Header"].attrs["Time"])
                box = float(f["Header"].attrs["BoxSize"])
                h = float(f["Parameters"].attrs["HubbleParam"])
            if "PartType6" not in f:
                continue
            g = f["PartType6"]
            fields["coords"].append(np.asarray(g["Coordinates"][:], float))
            fields["mass"].append(np.asarray(g["Masses"][:], float))
            fields["radius"].append(np.asarray(g["GrainRadius"][:], float))
            fields["carbon"].append(np.asarray(g["CarbonMassFraction"][:], float))
            fields["source"].append(np.asarray(g["DustSource"][:], np.int64))
            fields["form"].append(np.asarray(g["DustFormationTime"][:], float))
            fields["temp"].append(np.asarray(g["DustTemperature"][:], float))
            fields["birth"].append(np.asarray(g["BirthPos"][:], float))
            fields["pid"].append(np.asarray(g["ParticleIDs"][:], np.uint64))
    if not fields["coords"]:
        raise RuntimeError("No PartType6 particles found.")
    for k in fields:
        fields[k] = np.concatenate(fields[k])
    return fields, a, box, h

def optics(radius_nm, carbon, mass, carbon_table, silicate_table):
    rc,wc,qc = rvcore.read_qext_table(carbon_table)
    rs,ws,qs = rvcore.read_qext_table(silicate_table)
    qcb_t = rvcore.q_at_lambda(rc,wc,qc,0.44)
    qcv_t = rvcore.q_at_lambda(rc,wc,qc,0.55)
    qsb_t = rvcore.q_at_lambda(rs,ws,qs,0.44)
    qsv_t = rvcore.q_at_lambda(rs,ws,qs,0.55)
    rum = radius_nm * 1e-3
    qcb = rvcore.interp_q_radius(rum,rc,qcb_t)
    qcv = rvcore.interp_q_radius(rum,rc,qcv_t)
    qsb = rvcore.interp_q_radius(rum,rs,qsb_t)
    qsv = rvcore.interp_q_radius(rum,rs,qsv_t)
    good = (np.isfinite(qcb)&np.isfinite(qcv)&np.isfinite(qsb)&np.isfinite(qsv)
            &np.isfinite(mass)&(mass>0)&np.isfinite(rum)&(rum>0))
    wb = np.full(len(mass), np.nan)
    wv = np.full(len(mass), np.nan)
    wb[good] = (mass[good]/rum[good]) * (
        carbon[good]*qcb[good]/RHO_C +
        (1-carbon[good])*qsb[good]/RHO_SIL)
    wv[good] = (mass[good]/rum[good]) * (
        carbon[good]*qcv[good]/RHO_C +
        (1-carbon[good])*qsv[good]/RHO_SIL)
    return wb,wv,good

def rv_from_tau(tb,tv):
    d=tb-tv
    out=np.full_like(tv,np.nan,dtype=float)
    q=np.isfinite(tb)&np.isfinite(tv)&(tv>0)&(d>0)
    out[q]=tv[q]/d[q]
    return out

def summarize(name, q, age, radius, carbon, temp, mass):
    print("\n"+name)
    print("-"*76)
    print(f"N particles        = {q.sum()}")
    if not q.any(): return
    for label,arr,unit in [
        ("dust age",age,"Gyr"),("grain radius",radius,"nm"),
        ("carbon fraction",carbon,""),("dust temperature",temp,"K")]:
        v=arr[q]
        p=np.nanpercentile(v,[16,50,84])
        print(f"{label:<18s}= {p[1]:.5g} [{p[0]:.5g}, {p[2]:.5g}] {unit}")
    print(f"total dust mass    = {np.nansum(mass[q]):.8e} code mass")

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
    ap.add_argument("--high-rv",type=float,default=10.0)
    ap.add_argument("--omega-m",type=float,default=0.3158)
    ap.add_argument("--omega-l",type=float,default=0.6842)
    ap.add_argument("--prefix",default="halo441_current_highrv_agb_history_250pc")
    A=ap.parse_args()

    files=sorted(glob.glob(A.snapshot),key=natural_key)
    if not files: raise FileNotFoundError(A.snapshot)
    m=np.load(A.master_npz)
    center=np.asarray(m["center"],float)
    rot=np.asarray(m["rotation"],float)
    native=float(np.asarray(m["pixel_size_pkpc"]).squeeze())
    mr=float(np.asarray(m["plot_radius_pkpc"]).squeeze())
    zmax=float(np.asarray(m["zmax_pkpc"]).squeeze())

    D,a,box,h=load_dust(files)
    xyz=periodic_delta(D["coords"],center,box)*a/h
    xyz=xyz@rot.T
    x,y,z=xyz.T

    # BirthPos is assumed to use the same coordinate convention as Coordinates.
    # We report the transformed birth location and displacement, but do not use
    # it for any selection or validation.
    birth_xyz=periodic_delta(D["birth"],center,box)*a/h
    birth_xyz=birth_xyz@rot.T

    wb,wv,good=optics(D["radius"],np.clip(D["carbon"],0,1),D["mass"],
                      A.carbon_table,A.silicate_table)

    # Exact master face-on volume + optical-valid selection.
    sel=(np.abs(x)<mr)&(np.abs(y)<mr)&(np.abs(z)<zmax)&good
    idx=np.where(sel)[0]
    xs,ys=x[idx],y[idx]

    # Native pixel indices exactly corresponding to histogram2d(y,x).
    nb=int(round(2*mr/native))
    ix=np.floor((xs+mr)/native).astype(int)
    iy=np.floor((ys+mr)/native).astype(int)
    inside=(ix>=0)&(ix<nb)&(iy>=0)&(iy<nb)
    idx=idx[inside]; ix=ix[inside]; iy=iy[inside]

    # Central 20-kpc crop and native->250pc block mapping.
    nc=int(round(A.square_size/native))
    st=(nb-nc)//2; en=st+nc
    fac=int(round(A.pixel/native))
    if abs(fac*native-A.pixel)>1e-10:
        raise ValueError("Requested pixel must be an integer native multiple.")

    incrop=(ix>=st)&(ix<en)&(iy>=st)&(iy<en)
    idxc=idx[incrop]
    ixc=(ix[incrop]-st)//fac
    iyc=(iy[incrop]-st)//fac
    ncoarse=nc//fac

    # Reconstruct count/tau maps directly from particles.
    count=np.zeros((ncoarse,ncoarse),float)
    tb=np.zeros_like(count); tv=np.zeros_like(count)
    np.add.at(count,(iyc,ixc),1)
    np.add.at(tb,(iyc,ixc),wb[idxc])
    np.add.at(tv,(iyc,ixc),wv[idxc])
    rvrec=rv_from_tau(tb,tv)

    rvref=np.loadtxt(A.rv_250pc)
    cref=np.loadtxt(A.count_250pc)

    print("EXACT VALIDATION")
    print("="*76)
    cd=count-cref
    finite=np.isfinite(rvrec)&np.isfinite(rvref)
    fm=np.count_nonzero(np.isfinite(rvrec)!=np.isfinite(rvref))
    md=np.max(np.abs(rvrec[finite]-rvref[finite]))
    print(f"max count difference    = {np.max(np.abs(cd)):.0f}")
    print(f"count mismatched pixels = {np.count_nonzero(cd)} / {count.size}")
    print(f"finite-mask mismatches  = {fm} / {rvref.size}")
    print(f"max |RV reconstruction| = {md:.8e}")
    if np.max(np.abs(cd))!=0 or np.count_nonzero(cd)!=0 or fm or md>1e-6:
        raise RuntimeError("Validation failed; refusing particle-history interpretation.")
    print("VALIDATION PASSED.")

    # Aperture and baseline high-RV pixel mask.
    xx=(np.arange(ncoarse)+.5-ncoarse/2)*A.pixel
    yy=(np.arange(ncoarse)+.5-ncoarse/2)*A.pixel
    X,Y=np.meshgrid(xx,yy)
    aperture=np.hypot(X,Y)<=A.radius
    valid=aperture&np.isfinite(rvref)&(cref>=1)
    highpix=valid&(rvref>A.high_rv)

    # Particle membership in those exact pixels.
    particle_high=np.zeros(len(D["mass"]),dtype=bool)
    qpix=highpix[iyc,ixc]
    particle_high[idxc[qpix]]=True
    agb=D["source"]==SRC_AGB
    target=particle_high&agb

    # Also all AGB particles in valid aperture pixels for a fair comparison.
    valid_particle=np.zeros(len(D["mass"]),dtype=bool)
    qvalid=valid[iyc,ixc]
    valid_particle[idxc[qvalid]]=True
    agb_valid=valid_particle&agb
    agb_control=agb_valid&(~particle_high)

    # Ages from formation scale factor.
    tnow=float(cosmic_age_gyr(a,h,A.omega_m,A.omega_l))
    form=np.asarray(D["form"],float)
    valid_form=np.isfinite(form)&(form>0)&(form<=a)
    age=np.full(len(form),np.nan)
    age[valid_form]=tnow-cosmic_age_gyr(form[valid_form],h,A.omega_m,A.omega_l)

    print("\nSELECTION")
    print("="*76)
    print(f"cosmic age z=0             = {tnow:.5f} Gyr")
    print(f"valid aperture pixels      = {valid.sum()}")
    print(f"baseline R(V)>{A.high_rv:g} pixels     = {highpix.sum()}")
    print(f"all particles in those     = {particle_high.sum()}")
    print(f"AGB particles in those     = {target.sum()}")
    print(f"AGB particles other valid  = {agb_control.sum()}")

    summarize(f"AGB particles in R(V)>{A.high_rv:g} pixels",
              target,age,D["radius"],D["carbon"],D["temp"],D["mass"])
    summarize("AGB particles in other valid aperture pixels",
              agb_control,age,D["radius"],D["carbon"],D["temp"],D["mass"])

    # Age bins directly relevant to the user's recent dust-age work.
    print("\nAGE-BIN FRACTIONS")
    print("="*76)
    print("Population                         <3 Gyr     3-7 Gyr      >7 Gyr")
    for name,q in [(f"AGB in RV>{A.high_rv:g}",target),
                   ("other valid AGB",agb_control)]:
        aa=age[q]; aa=aa[np.isfinite(aa)]
        if len(aa):
            vals=[np.mean(aa<3),np.mean((aa>=3)&(aa<=7)),np.mean(aa>7)]
            print(f"{name:<32s} {vals[0]:9.4f} {vals[1]:11.4f} {vals[2]:11.4f}")

    # Per-target particle output.
    q=np.where(target)[0]
    # coarse pixel and baseline RV for each target
    # Build lookup from global particle index -> coarse indices
    pixx=np.full(len(D["mass"]),-1,int); pixy=np.full(len(D["mass"]),-1,int)
    pixx[idxc]=ixc; pixy[idxc]=iyc
    prv=np.array([rvref[pixy[j],pixx[j]] for j in q])
    pcount=np.array([cref[pixy[j],pixx[j]] for j in q])

    # Current galactocentric cylindrical radius and |z|.
    Rgal=np.hypot(x[q],y[q])
    # Birth displacement is a diagnostic only; BirthPos convention is not used
    # in any scientific selection.
    disp=np.linalg.norm(xyz[q]-birth_xyz[q],axis=1)

    arr=np.column_stack([
        D["pid"][q].astype(float), prv, pcount,
        D["form"][q], age[q], D["mass"][q], D["radius"][q],
        D["carbon"][q], D["temp"][q],
        x[q],y[q],z[q],Rgal,np.abs(z[q]),
        birth_xyz[q,0],birth_xyz[q,1],birth_xyz[q,2],disp,
        wb[q],wv[q],wb[q]-wv[q]
    ])
    header=("ParticleID RV_pixel Ndust_pixel formation_scale_factor age_Gyr "
            "mass_code grain_radius_nm carbon_mass_fraction dust_temperature_K "
            "x_pkpc y_pkpc z_pkpc Rgal_pkpc absz_pkpc "
            "birth_x_pkpc birth_y_pkpc birth_z_pkpc current_minus_birth_pkpc "
            "weight_B weight_V weight_EBV")
    np.savetxt(A.prefix+"_particles.txt",arr,header=header)

    # Compact summary CSV-like text for age/radius/carbon relationship.
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    sc=ax.scatter(age[target],D["radius"][target],
                  c=D["carbon"][target],s=60,alpha=.85)
    ax.set_xlabel("Dust age (Gyr)")
    ax.set_ylabel("Current grain radius (nm)")
    ax.set_title(f"AGB-origin dust in baseline R(V) > {A.high_rv:g} pixels")
    cb=fig.colorbar(sc,ax=ax); cb.set_label("Current carbon mass fraction")
    fig.tight_layout()
    fig.savefig(A.prefix+"_age_radius_carbon.png",dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,5.5))
    bins=np.linspace(0,tnow,15)
    if target.any():
        ax.hist(age[target][np.isfinite(age[target])],bins=bins,histtype="step",
                density=True,linewidth=2,label=f"AGB in R(V)>{A.high_rv:g}")
    if agb_control.any():
        ax.hist(age[agb_control][np.isfinite(age[agb_control])],bins=bins,
                histtype="step",density=True,linewidth=2,label="Other valid AGB")
    ax.set_xlabel("Dust age (Gyr)")
    ax.set_ylabel("Normalized density")
    ax.set_title("CG10 AGB dust-age comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(A.prefix+"_age_comparison.png",dpi=220)
    plt.close(fig)

    print("\nOUTPUTS")
    print("="*76)
    print(A.prefix+"_particles.txt")
    print(A.prefix+"_age_radius_carbon.png")
    print(A.prefix+"_age_comparison.png")

if __name__=="__main__":
    main()
