#!/usr/bin/env python3
'''
python zoom_halo_track.py output_zoom/snapshot_*.hdf5 output_zoom/fof_subhalo_tab_*.hdf5 \
  --ref-snap 061 --ref-halo 305 --csv track_halo305.csv --quick-plots
'''
import argparse, glob, os, re, sys, math, csv
import numpy as np, h5py

MSUN_IN_G=1.98847e33; CM_PER_MPC=3.085678e24; MPC_IN_M=3.085677581e22
G_SI=6.67430e-11; MSUN_IN_KG=1.98847e30; X_H=0.76; MP_G=1.6726219e-24

def rho_crit_Msun_per_Mpc3(z,H0,Om,Ol,Or=0.0):
    Ok=1-(Om+Ol+Or); Hz=H0*math.sqrt(Om*(1+z)**3+Or*(1+z)**4+Ok*(1+z)**2+Ol)
    Hs=Hz*1000.0/MPC_IN_M; return (3*Hs*Hs/(8*math.pi*G_SI))*(MPC_IN_M**3)/MSUN_IN_KG

def load_units(fn):
    with h5py.File(fn,'r') as f:
        H=f['Header'].attrs
        uL=float(H.get('UnitLength_in_cm',CM_PER_MPC))
        uM=float(H.get('UnitMass_in_g',1.989e43))
        z=float(H.get('Redshift',0.0)); L= float(H['BoxSize'])*(uL/CM_PER_MPC)
        return dict(uL=uL,uM=uM,z=z,box=L)

def fof_props(fof_fn):
    with h5py.File(fof_fn,'r') as f:
        H=f['Header'].attrs
        Om=float(H.get('Omega0',0.315)); Ol=float(H.get('OmegaLambda',0.685))
        H0=float(H.get('Hubble',100.0)); hub=float(H.get('HubbleParam',0.7))
        uL=float(H.get('UnitLength_in_cm',CM_PER_MPC)); uM=float(H.get('UnitMass_in_g',1.989e43))
        pos=np.array(f['Group/GroupPos'])*(uL/CM_PER_MPC)
        M=np.array(f['Group/GroupMass']); M_Msun= M*1e10 if (M.size and np.nanmax(M)<1e8) else M*(uM/MSUN_IN_G)
        R=None; M200=None
        if 'Group_R_Crit200' in f['Group']: R=np.array(f['Group/Group_R_Crit200'])/hub
        if 'Group_M_Crit200' in f['Group']:
            Mc=np.array(f['Group/Group_M_Crit200']); M200= Mc*1e10 if (Mc.size and np.nanmax(Mc)<1e8) else Mc*(uM/MSUN_IN_G)
        return dict(Om=Om,Ol=Ol,H0=H0,hub=hub,uL=uL,uM=uM,pos=pos,M=M_Msun,R200_kpc=R,M200=M200)

def most_bound_dm_particle(snap_fn, center_Mpc, R_Mpc):
    # Return ID of DM particle with minimum potential if available, otherwise min |v| or nearest to center
    with h5py.File(snap_fn,'r') as f:
        H=f['Header'].attrs; uL=float(H.get('UnitLength_in_cm',CM_PER_MPC)); to_Mpc=uL/CM_PER_MPC
        if 'PartType1' not in f: return None
        P=f['PartType1/Coordinates'][:]*to_Mpc; ids=f['PartType1/ParticleIDs'][:]
        d=P-center_Mpc[None,:]; L=float(H['BoxSize'])*to_Mpc; d-=np.round(d/L)*L; r=np.sqrt(np.einsum('ij,ij->i',d,d))
        m=(r<=R_Mpc)
        if not np.any(m): return None
        if 'PartType1/Potential' in f:
            pot=f['PartType1/Potential'][:][m]; ii=np.argmin(pot); return int(ids[m][ii])
        elif 'PartType1/Velocities' in f:
            v=f['PartType1/Velocities'][:][m]; s2=np.sum(v*v,axis=1); ii=np.argmin(s2); return int(ids[m][ii])
        else:
            ii=np.argmin(r[m]); return int(ids[m][ii])

def halo_index_containing_particle(fof_fn, pid):
    with h5py.File(fof_fn,'r') as f:
        if 'Group/GroupFirstSub' in f and 'IDs/ID' in f:
            # Not all catalogs store membership; fallback below if absent
            pass
    # Fallback: pick nearest FoF to particle position (reads snapshot instead)
    return None

def nearest_index(points, center, box):
    d=points-center[None,:]; d-=np.round(d/box)*box
    return int(np.argmin(np.einsum('ij,ij->i', d, d)))

def analyze_one(snap_fn, fof_fn, center_guess=None, idx_guess=None, r_mult=1.0):
    u=load_units(snap_fn); F=fof_props(fof_fn)
    # choose halo
    if idx_guess is not None:
        i=idx_guess; center=F['pos'][i]
    else:
        center=np.array(center_guess); box=u['box']; i=nearest_index(F['pos'], center, box)
        center=F['pos'][i]
    # R200/M200
    if F['R200_kpc'] is not None:
        R200=F['R200_kpc'][i]/1000.0; M200=float(F['M200'][i]) if F['M200'] is not None else float(F['M'][i])
    else:
        rho_c=rho_crit_Msun_per_Mpc3(u['z'],F['H0'],F['Om'],F['Ol'])
        M200=float(F['M'][i]); R200=float((3*M200/(4*np.pi*200*rho_c))**(1/3))
    # counts
    Rsel=r_mult*R200
    with h5py.File(snap_fn,'r') as f:
        to_Mpc=float(f['Header'].attrs.get('UnitLength_in_cm',CM_PER_MPC))/CM_PER_MPC
        L=float(f['Header'].attrs['BoxSize'])*to_Mpc
        def count(pt,R):
            if pt not in f: return 0
            P=f[f'{pt}/Coordinates'][:]*to_Mpc
            d=P-center[None,:]; d-=np.round(d/L)*L
            r=np.sqrt(np.einsum('ij,ij->i',d,d))
            return int(np.count_nonzero(r<=R))
        N1=count('PartType1',Rsel); N2=count('PartType2',Rsel)
        N1_2R=count('PartType1',2*R200); N2_2R=count('PartType2',2*R200)
    return dict(z=u['z'],R200=R200,M200=M200,halo=i,N1=N1,N2=N2,N1_2R=N1_2R,N2_2R=N2_2R)

def main():
    ap=argparse.ArgumentParser(description="Track one zoom halo across snapshots")
    ap.add_argument("snapshots_glob"); ap.add_argument("fof_glob")
    ap.add_argument("--ref-snap", type=int, required=True, help="Reference snapshot number (e.g., 061)")
    ap.add_argument("--ref-halo", type=int, help="FoF index at reference snapshot")
    ap.add_argument("--ref-center", type=float, nargs=3, help="If no ref-halo, pick nearest to this (Mpc)")
    ap.add_argument("--r-mult", type=float, default=1.0)
    ap.add_argument("--csv", default="halo_track.csv"); ap.add_argument("--quick-plots", action="store_true")
    args=ap.parse_args()

    snaps=sorted(glob.glob(args.snapshots_glob))
    fofs =sorted(glob.glob(args.fof_glob))
    # pair by snapshot number
    def num(s): m=re.search(r'(\d+)', os.path.basename(s)); return int(m.group(1)) if m else -1
    pairs=sorted([(s,f) for s in snaps for f in fofs if num(s)==num(f)], key=lambda x:num(x[0]))
    if not pairs: print("No matched snapshot/fof pairs."); sys.exit(1)

    # find ref pair
    ref_idx=[i for i,(s,f) in enumerate(pairs) if num(s)==args.ref_snap]
    if not ref_idx: print("Ref snapshot not found."); sys.exit(1)
    s_ref,f_ref=pairs[ref_idx[0]]

    # determine reference halo
    F=fof_props(f_ref)
    if args.ref_halo is not None:
        i_ref=args.ref_halo; center_ref=F['pos'][i_ref]
    else:
        if args.ref_center is None: print("Need --ref-halo or --ref-center."); sys.exit(1)
        center_ref=np.array(args.ref_center); i_ref=nearest_index(F['pos'], center_ref, load_units(s_ref)['box'])
        center_ref=F['pos'][i_ref]

    # (Optional) choose anchor particle (MBP) inside R200 at ref snapshot
    # For simplicity we just track by nearest in each snapshot to center_ref.
    rows=[]
    for s,f in pairs:
        R=analyze_one(s,f, center_guess=center_ref, idx_guess=None, r_mult=args.r_mult)
        rows.append(dict(snap=os.path.basename(s), z=R['z'], halo=R['halo'],
                         M200=R['M200'], R200=R['R200'],
                         N1=R['N1'], N2=R['N2'], contam=R['N2']/max(R['N1']+R['N2'],1),
                         N1_2R=R['N1_2R'], N2_2R=R['N2_2R'],
                         contam2R=R['N2_2R']/max(R['N1_2R']+R['N2_2R'],1)))
    # write CSV
    with open(args.csv,'w',newline='') as fp:
        w=csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[saved] {args.csv}")

    if args.quick_plots:
        try:
            import matplotlib.pyplot as plt
            zz=[r['z'] for r in rows]; M=[r['M200'] for r in rows]; R=[r['R200'] for r in rows]
            C=[r['contam'] for r in rows]
            plt.figure(); plt.plot(zz,M); plt.gca().invert_xaxis(); plt.xlabel('z'); plt.ylabel('M200 [Msun]')
            plt.tight_layout(); plt.savefig('track_M200.png',dpi=160); print("[plot] track_M200.png"); plt.close()
            plt.figure(); plt.plot(zz,R); plt.gca().invert_xaxis(); plt.xlabel('z'); plt.ylabel('R200 [Mpc]')
            plt.tight_layout(); plt.savefig('track_R200.png',dpi=160); print("[plot] track_R200.png"); plt.close()
            plt.figure(); plt.plot(zz,C); plt.gca().invert_xaxis(); plt.xlabel('z'); plt.ylabel('contam(<R200)')
            plt.tight_layout(); plt.savefig('track_contam.png',dpi=160); print("[plot] track_contam.png"); plt.close()
        except Exception as e:
            print(f"Plot error (skipping): {e}")
if __name__=="__main__":
    main()
