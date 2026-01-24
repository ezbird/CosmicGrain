#!/usr/bin/env python3
"""
zoom_halo_report.py
Post-process a finished Gadget-4 zoom snapshot: resolution, LR contamination,
isolation, spin/shape, gas readiness; robust to missing/zero R200/M200 in FoF.

- Finds halo by FoF index or nearest to (x,y,z).
- Converts units robustly (kpc vs kpc/h detection).
- If FoF R200/M200 invalid, estimates R200/M200 from snapshot particles via
  spherical-overdensity (200 × rho_crit at snapshot redshift).
- Optional plots: DM XY surface density and gas phase diagram.

Requirements: numpy, h5py, matplotlib (only if --make-plots)
"""

import argparse, json, os, sys, math
import numpy as np, h5py
from matplotlib.colors import LogNorm

# --- constants ---
MSUN_IN_G   = 1.98847e33
CM_PER_MPC  = 3.085678e24
CM_PER_KPC  = 3.085678e21
MPC_IN_M    = 3.085677581e22
G_SI        = 6.67430e-11
MSUN_IN_KG  = 1.98847e30
MP_G        = 1.6726219e-24
X_H         = 0.76
G_MPC_KMS2_MSUN = 4.30091e-9  # G in (Mpc * (km/s)^2 / Msun)

# --- helpers ---
def rho_crit_Msun_per_Mpc3(z, H0_km_s_Mpc, Om, Ol, Or=0.0):
    Ok = 1.0 - (Om + Ol + Or)
    Hz = H0_km_s_Mpc * math.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + Ol)
    H_s = Hz * 1000.0 / MPC_IN_M  # s^-1
    rho_c_SI = 3.0 * H_s*H_s / (8.0 * math.pi * G_SI)
    return rho_c_SI * (MPC_IN_M**3) / MSUN_IN_KG  # Msun / Mpc^3

def min_image_delta(a, b, box):
    d = a - b
    d -= np.round(d / box) * box
    return d

def nearest_index_periodic(points, center, box):
    d = min_image_delta(points, center[None,:], box)
    return int(np.argmin(np.einsum('ij,ij->i', d, d)))

def load_fof(fof_fn):
    with h5py.File(fof_fn,'r') as f:
        H = f['Header'].attrs
        Om  = float(H.get('Omega0',0.315))
        Ol  = float(H.get('OmegaLambda',0.685))
        H0  = float(H.get('Hubble',100.0))
        hub = float(H.get('HubbleParam',0.7))
        z   = float(H.get('Redshift',0.0))
        Lc  = float(H['BoxSize'])
        uL  = float(H.get('UnitLength_in_cm', CM_PER_MPC))
        uM  = float(H.get('UnitMass_in_g', 1.989e43))
        box = Lc * (uL/CM_PER_MPC)

        Gp = f.get('Group')
        if Gp is None: raise RuntimeError("FoF missing /Group")

        pos = np.array(Gp['GroupPos']) * (uL/CM_PER_MPC)
        M   = np.array(Gp['GroupMass'])
        M_Msun = M*1e10 if (M.size and np.nanmax(M)<1e8) else M*(uM/MSUN_IN_G)

        R200_kpc = None
        M200 = None
        Rcand = np.array(Gp['Group_R_Crit200']) if 'Group_R_Crit200' in Gp else None
        if 'Group_M_Crit200' in Gp:
            Mc = np.array(Gp['Group_M_Crit200'])
            M200 = Mc*1e10 if (Mc.size and np.nanmax(Mc)<1e8) else Mc*(uM/MSUN_IN_G)

        # detect kpc vs kpc/h with cross-check
        if Rcand is not None:
            if M200 is not None and np.isfinite(M200).any():
                rho_c = rho_crit_Msun_per_Mpc3(z,H0,Om,Ol)
                R_fromM_kpc = ((3*np.maximum(M200,1e-40))/(4*np.pi*200*rho_c))**(1/3)*1000.0
                err_kpc   = np.nanmedian(np.abs(Rcand - R_fromM_kpc))
                err_kpc_h = np.nanmedian(np.abs(Rcand/hub - R_fromM_kpc))
                R200_kpc = (Rcand/hub) if (err_kpc_h<err_kpc) else Rcand
            else:
                R200_kpc = Rcand / hub  # common default

        return dict(pos=pos, Mtot=M_Msun, R200_kpc=R200_kpc, M200=M200,
                    box=box, z=z, H0=H0, Om=Om, Ol=Ol, hub=hub)

def load_snapshot_units(snap):
    with h5py.File(snap,'r') as f:
        H = f['Header'].attrs
        uL = float(H.get('UnitLength_in_cm', CM_PER_MPC))
        uM = float(H.get('UnitMass_in_g', 1.989e43))
        uV = float(H.get('UnitVelocity_in_cm_per_s', 1e5))
        box = float(H['BoxSize'])*(uL/CM_PER_MPC)
        z   = float(H.get('Redshift',0.0))
        return dict(uL=uL,uM=uM,uV=uV,box=box,z=z,to_Mpc=uL/CM_PER_MPC,to_kms=uV/1e5)

def read_particles_in_sphere(snap, center, R, types=('PartType1','PartType2','PartType0','PartType4')):
    out={}
    with h5py.File(snap,'r') as f:
        H = f['Header'].attrs; uL=float(H.get('UnitLength_in_cm',CM_PER_MPC)); to_Mpc=uL/CM_PER_MPC
        L = float(H['BoxSize'])*to_Mpc
        for pt in types:
            if pt not in f: out[pt]=None; continue
            P=f[f'{pt}/Coordinates'][:]*to_Mpc
            d=min_image_delta(P, center[None,:], L)
            r=np.sqrt(np.einsum('ij,ij->i', d, d))
            msk=r<=R
            rec=dict(pos=P[msk], r=r[msk])
            if f'{pt}/Velocities' in f: rec['vel']=f[f'{pt}/Velocities'][:][msk]*(float(H.get('UnitVelocity_in_cm_per_s',1e5))/1e5)
            if f'{pt}/ParticleIDs' in f: rec['ids']=f[f'{pt}/ParticleIDs'][:][msk]
            if f'{pt}/Masses' in f: rec['mass']=f[f'{pt}/Masses'][:][msk]
            else:
                MT = H.get('MassTable')
                if MT is not None:
                    idx={'PartType0':0,'PartType1':1,'PartType2':2,'PartType3':3,'PartType4':4,'PartType5':5}[pt]
                    rec['mass']=np.full(msk.sum(), MT[idx], dtype=np.float64)
            if pt=='PartType0':
                for key in ('Temperature','Density','InternalEnergy'):
                    if f'{pt}/{key}' in f: rec[key]=f[f'{pt}/{key}'][:][msk]
            out[pt]=rec
    return out

def inertia_axis_ratios(positions, masses=None):
    if positions is None or positions.size==0: return (np.nan,np.nan)
    w = masses if masses is not None else np.ones(positions.shape[0])
    cen = np.average(positions, axis=0, weights=w)
    X = positions - cen
    I = np.einsum('ni,nj,n->ij', X, X, w)
    evals,_ = np.linalg.eigh(I); evals=np.sort(evals)
    a,b,c = math.sqrt(evals[2]+1e-30), math.sqrt(evals[1]+1e-30), math.sqrt(evals[0]+1e-30)
    return (b/a, c/a)

def spin_bullock(positions, velocities, masses, center, vbulk, M_Msun, R_Mpc):
    if positions is None or positions.size==0 or velocities is None: return np.nan
    w = masses if masses is not None else np.ones(positions.shape[0])
    x = positions - center[None,:]
    v = velocities - vbulk[None,:]
    J = np.sum(np.cross(x,v)*w[:,None], axis=0)
    Vc = math.sqrt(G_MPC_KMS2_MSUN * M_Msun / R_Mpc)
    return np.linalg.norm(J)/(math.sqrt(2)*M_Msun*Vc*R_Mpc)

# --- NEW: robust bulk velocity ---
def Msun_from_code(mass_code, unitmass_g):
    return mass_code*(unitmass_g/MSUN_IN_G) if mass_code is not None else None

def v_bulk(dmrecs, unitmass_g):
    """Mass-weighted bulk velocity; robust to missing/zero masses."""
    V_list, W_list = [], []
    for rec in dmrecs:
        if not rec:
            continue
        v = rec.get("vel")
        if v is None or v.size == 0:
            continue
        V_list.append(v)

        w = None
        m = rec.get("mass")
        if m is not None:
            w = Msun_from_code(m, unitmass_g)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w[w < 0] = 0.0
        if w is None or w.size != v.shape[0] or np.sum(w) <= 0.0:
            w = np.ones(v.shape[0], dtype=float)
        W_list.append(w)

    if not V_list:
        return np.zeros(3)
    V = np.vstack(V_list)
    W = np.hstack(W_list)
    wsum = float(np.sum(W))
    if not np.isfinite(wsum) or wsum <= 0.0:
        W = np.ones(V.shape[0], dtype=float); wsum = float(W.size)
    return np.sum(V * W[:, None], axis=0) / wsum

# --- NEW: SO(200c) fallback directly from particles ---
def estimate_R200_from_particles(snap_fn, center_Mpc, z, H0_km_s_Mpc, Om, Ol,
                                 r_max_Mpc=2.0, n_bins=240,
                                 include_gas=True, include_stars=True):
    """Return (R200_Mpc, M200_Msun) estimated from snapshot particles. np.nan if not found."""
    rho_c = rho_crit_Msun_per_Mpc3(z, H0_km_s_Mpc, Om, Ol)
    with h5py.File(snap_fn, "r") as f:
        H = f["Header"].attrs
        uL = float(H.get("UnitLength_in_cm", CM_PER_MPC))
        uM = float(H.get("UnitMass_in_g", 1.989e43))
        to_Mpc = uL / CM_PER_MPC
        box = float(H["BoxSize"]) * to_Mpc

        def get_massive(pt):
            if pt not in f: return None, None
            pos = f[f"{pt}/Coordinates"][:] * to_Mpc
            if f"{pt}/Masses" in f:
                m_code = f[f"{pt}/Masses"][:]
            else:
                MT = H.get("MassTable")
                if MT is None: return pos, None
                idx = {'PartType0':0,'PartType1':1,'PartType2':2,'PartType3':3,'PartType4':4,'PartType5':5}[pt]
                m_code = np.full(pos.shape[0], MT[idx], dtype=np.float64)
            m_sun = m_code * (uM / MSUN_IN_G)
            return pos, m_sun

        parts = []
        for pt in ("PartType1", "PartType2"):
            p, m = get_massive(pt)
            if p is not None and m is not None and p.size>0: parts.append((p, m))
        if include_gas:
            p, m = get_massive("PartType0")
            if p is not None and m is not None and p.size>0: parts.append((p, m))
        if include_stars:
            p, m = get_massive("PartType4")
            if p is not None and m is not None and p.size>0: parts.append((p, m))
        if not parts: return (np.nan, np.nan)

        c = np.asarray(center_Mpc)
        r_all, m_all = [], []
        for p, m in parts:
            d = p - c[None, :]
            d -= np.round(d / box) * box
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            sel = r <= r_max_Mpc
            if np.any(sel):
                r_all.append(r[sel]); m_all.append(m[sel])
        if not r_all: return (np.nan, np.nan)

        r = np.hstack(r_all); m = np.hstack(m_all)
        idx = np.argsort(r); r_sorted = r[idx]; m_sorted = m[idx]
        M_cum = np.cumsum(m_sorted)

        r_grid = np.linspace(max(1e-4, r_sorted.min()), min(r_max_Mpc, max(r_sorted.max(), 1e-3)), n_bins)
        j = np.searchsorted(r_sorted, r_grid, side="right") - 1
        j = np.clip(j, 0, len(M_cum)-1)
        M_of_r = M_cum[j]
        mean_rho = M_of_r / ((4.0/3.0) * math.pi * r_grid**3)

        target = 200.0 * rho_c
        above = mean_rho >= target
        if not np.any(above): return (np.nan, np.nan)
        k = np.where(above)[0][-1]
        if k == len(r_grid) - 1: return (np.nan, np.nan)
        r1, r2 = r_grid[k], r_grid[k+1]; y1, y2 = mean_rho[k], mean_rho[k+1]
        R200 = r2 if y1==y2 else r1 + (target - y1) * (r2 - r1) / (y2 - y1)
        M200 = float(np.interp(R200, r_grid, M_of_r))
        return (float(R200), M200)

# --- main ---
def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Zoom halo post-processing: resolution, contamination, gas readiness, spin/shape, isolation.")
    ap.add_argument("snapshot"); ap.add_argument("fof")
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--halo-index", type=int)
    g.add_argument("--center", type=float, nargs=3, metavar=("X","Y","Z"))
    ap.add_argument("--r-mult", type=float, default=1.0, help="Selection radius multiple of R200")
    ap.add_argument("--eps-dm-kpc", type=float, default=None, help="HR DM epsilon (physical kpc) for eps/R200")
    ap.add_argument("--make-plots", action="store_true"); ap.add_argument("--plot-dir", default="halo_plots")
    ap.add_argument("-o","--output", default=None); ap.add_argument("--json", default=None)
    args=ap.parse_args()

    fof=load_fof(args.fof); u=load_snapshot_units(args.snapshot)

    # choose halo
    if args.halo_index is not None:
        idx=int(args.halo_index); center=fof['pos'][idx]
    else:
        center_guess=np.array(args.center, dtype=float)
        idx=nearest_index_periodic(fof['pos'], center_guess, fof['box'])
        center=fof['pos'][idx]

    # --- robust R200/M200 with fallbacks ---
    R200_Mpc = np.nan; M200_Msun = np.nan; Rsrc = "unknown"
    if fof["R200_kpc"] is not None:
        R200_Mpc = float(fof["R200_kpc"][idx]) / 1000.0
        if fof["M200"] is not None and np.isfinite(fof["M200"][idx]) and fof["M200"][idx] > 0:
            M200_Msun = float(fof["M200"][idx])
        elif np.isfinite(fof["Mtot"][idx]) and fof["Mtot"][idx] > 0:
            M200_Msun = float(fof["Mtot"][idx])
        Rsrc = "Group_R_Crit200"

    if (not np.isfinite(R200_Mpc)) or R200_Mpc <= 0:
        if fof["M200"] is not None and np.isfinite(fof["M200"][idx]) and fof["M200"][idx] > 0:
            M200_Msun = float(fof["M200"][idx])
            rho_c = rho_crit_Msun_per_Mpc3(fof["z"], fof["H0"], fof["Om"], fof["Ol"])
            R200_Mpc = float((3.0 * M200_Msun / (4.0*np.pi*200.0*rho_c)) ** (1.0/3.0))
            Rsrc = "computed from Group_M_Crit200"

    if (not np.isfinite(R200_Mpc)) or R200_Mpc <= 0 or (not np.isfinite(M200_Msun)) or M200_Msun <= 0:
        R_est, M_est = estimate_R200_from_particles(
            args.snapshot, center,
            fof["z"], fof["H0"], fof["Om"], fof["Ol"],
            r_max_Mpc=2.0, n_bins=240, include_gas=True, include_stars=True
        )
        if np.isfinite(R_est) and R_est > 0:
            R200_Mpc = R_est; M200_Msun = M_est; Rsrc = "estimated from particles (SO 200c)"
        else:
            raise RuntimeError("Failed to determine R200 for this halo (catalog fields zero and SO estimate failed).")

    # Isolation (nearest heavier)
    d=min_image_delta(fof['pos'], center[None,:], fof['box'])
    rr=np.sqrt(np.einsum('ij,ij->i',d,d))
    heavier=fof['Mtot']>=M200_Msun
    heavier[idx]=False
    iso=float(np.min(rr[heavier])) if np.any(heavier) else np.inf

    # Select particles within radii
    Rsel = args.r_mult * R200_Mpc
    parts_R   = read_particles_in_sphere(args.snapshot, center, Rsel)
    parts_2R  = read_particles_in_sphere(args.snapshot, center, 2*R200_Mpc)

    # counts / contamination
    def count(pt, d): return 0 if d.get(pt) is None else d[pt]['pos'].shape[0]
    N_hr  = count('PartType1', parts_R)
    N_lr  = count('PartType2', parts_R)
    N_hr2 = count('PartType1', parts_2R)
    N_lr2 = count('PartType2', parts_2R)
    contam   = (N_lr / max(N_hr+N_lr,1))
    contam2R = (N_lr2 / max(N_hr2+N_lr2,1))

    # bulk velocity (robust)
    vbulk = v_bulk([parts_R.get('PartType1'), parts_R.get('PartType2')], u['uM'])

    # spin/shape (DM HR within R200)
    spin=np.nan; ba=ca=np.nan
    DM=parts_R.get('PartType1')
    if DM is not None and DM['pos'].size>0:
        m_dm = Msun_from_code(DM.get('mass'), u['uM']) if DM.get('mass') is not None else None
        spin=spin_bullock(DM['pos'], DM.get('vel', None), m_dm if m_dm is not None else np.ones(DM['pos'].shape[0]),
                          center, vbulk, M200_Msun, R200_Mpc)
        ba, ca = inertia_axis_ratios(DM['pos'], m_dm)

    # gas readiness
    gas_stats=None
    G=parts_R.get('PartType0')
    if G is not None and G['pos'].size>0:
        mg = Msun_from_code(G.get('mass'), u['uM']) if G.get('mass') is not None else None
        M_gas = float(np.sum(mg)) if mg is not None else 0.0
        nH=None
        if 'Density' in G:
            rho_code=G['Density']; rho_cgs = rho_code*(u['uM']/(u['uL']**3))
            nH = (X_H * rho_cgs)/MP_G
        T = G.get('Temperature', None)
        cold_frac = float(np.sum((T<300.0)*mg)/max(np.sum(mg),1.0)) if (T is not None and mg is not None) else np.nan
        dense_frac = float(np.sum((nH>10.0)*mg)/max(np.sum(mg),1.0)) if (nH is not None and mg is not None) else np.nan
        med_nH = float(np.median(nH)) if nH is not None else np.nan
        med_T  = float(np.median(T))  if T is not None  else np.nan
        gas_stats=dict(M_gas_Msun=M_gas,cold_frac=cold_frac,dense_frac=dense_frac,med_nH=med_nH,med_T=med_T)

    eps_over = (args.eps_dm_kpc/1000.0)/R200_Mpc if args.eps_dm_kpc is not None else None

    # report
    lines=[]
    lines += [f"=== Zoom Halo Report ===",
              f"Snapshot: {os.path.basename(args.snapshot)} | FoF: {os.path.basename(args.fof)}",
              f"z={u['z']:.3f} | Box={u['box']:.3f} Mpc | Halo index={idx}",
              f"Center (Mpc): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
              f"M200 = {M200_Msun:.3e} Msun | R200 = {R200_Mpc*1000:.1f} kpc  [{Rsrc}]",
              f"N_DM(<R200): HR={N_hr}  LR={N_lr}  | contam={contam*100:.2f}%",
              f"N_DM(<2R200): HR={N_hr2} LR={N_lr2} | contam(2R)={contam2R*100:.2f}%",
              f"Isolation (nearest heavier) = {iso:.3f} Mpc",
              f"DM spin λ' = {spin:.3f} | shape b/a={ba:.2f}, c/a={ca:.2f}",
              f"eps_DM/R200 = {eps_over:.3f}" if eps_over is not None else "eps_DM/R200 = n/a"]
    if gas_stats:
        lines += [f"Gas (<{args.r_mult:.2f} R200): M_gas={gas_stats['M_gas_Msun']:.3e} Msun | cold<300K={gas_stats['cold_frac']*100:.1f}% | dense nH>10={gas_stats['dense_frac']*100:.1f}%",
                  f"    med(nH)={gas_stats['med_nH']:.3g} cm^-3 | med(T)={gas_stats['med_T']:.3g} K"]
    report="\n".join(lines); print(report)
    if args.output:
        with open(args.output,'w') as fp: fp.write(report+"\n"); print(f"[saved] {args.output}")
    if args.json:
        js=dict(z=u['z'],box=u['box'],halo_index=idx,center=center.tolist(),M200=M200_Msun,R200=R200_Mpc,
                contam=contam,contam2R=contam2R,iso=iso,spin=spin,ba=ba,ca=ca,gas=gas_stats)
        with open(args.json,'w') as fp: json.dump(js,fp,indent=2); print(f"[saved] {args.json}")

    # plots
    if args.make_plots:
        import matplotlib.pyplot as plt
        os.makedirs(args.plot_dir, exist_ok=True)
        # DM HR XY within 2R200
        Rplot = 2*R200_Mpc
        DM2 = read_particles_in_sphere(args.snapshot, center, Rplot)['PartType1']
        if DM2 is not None and DM2['pos'].size>0:
            x,y = DM2['pos'][:,0], DM2['pos'][:,1]
            m   = Msun_from_code(DM2.get('mass'), u['uM']) if DM2.get('mass') is not None else None
            res=1024; rng=[[center[0]-Rplot, center[0]+Rplot],[center[1]-Rplot, center[1]+Rplot]]
            H, xe, ye = np.histogram2d(x, y, bins=res, range=rng, weights=m)
            px=((rng[0][1]-rng[0][0])/res)**2; S=(H.T)/px; S[S<=0]=np.nanmin(S[S>0])/100
            plt.figure(figsize=(6,5))
            plt.imshow(S, extent=[*rng[0],*rng[1]], origin='lower', cmap='magma_r', norm=LogNorm())
            plt.colorbar(label=r'$\Sigma\ [M_\odot\,\mathrm{Mpc}^{-2}]$')
            plt.xlabel('x [Mpc]'); plt.ylabel('y [Mpc]')
            plt.title(f'DM-HR XY, <2R200 (halo {idx})')
            outf=os.path.join(args.plot_dir,f"dm_xy_2R_halo{idx}.png")
            plt.tight_layout(); plt.savefig(outf,dpi=220); plt.close(); print(f"[plot] {outf}")

        # Gas phase diagram
        G2 = read_particles_in_sphere(args.snapshot, center, R200_Mpc).get('PartType0')
        if G2 is not None and G2.get('Density', None) is not None:
            rho_code=G2['Density']; rho_cgs=rho_code*(u['uM']/(u['uL']**3))
            nH=(X_H*rho_cgs)/MP_G; T=G2.get('Temperature', None)
            if T is not None:
                plt.figure(figsize=(6,5))
                plt.hist2d(np.log10(nH+1e-40), np.log10(T), bins=200, cmap="viridis")
                plt.xlabel(r'log $n_{\rm H}$ [cm$^{-3}$]'); plt.ylabel(r'log $T$ [K]')
                plt.title(f'Gas phase (R200, halo {idx})')
                outf=os.path.join(args.plot_dir,f"gas_phase_R_halo{idx}.png")
                plt.tight_layout(); plt.savefig(outf,dpi=220); plt.close(); print(f"[plot] {outf}")

if __name__=="__main__":
    main()
