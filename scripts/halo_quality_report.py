#!/usr/bin/env python3
"""
halo_quality_report.py
Inspect a chosen halo in a Gadget-4 zoom snapshot and print a "dust-readiness" scorecard.

Features
--------
- Works with DM-only or hydro snapshots (gas/stars optional).
- Finds halo by halo-index or nearest to (x,y,z) using FoF catalog.
- Robust unit handling; detects kpc vs kpc/h for Group_R_Crit200 by cross-checking Group_M_Crit200.
- Computes:
  * M200, R200 (Mpc)
  * Counts & masses inside R200 (DM by type, gas, stars)
  * LR contamination fraction (e.g., PartType2 inside R200)
  * Isolation: distance to nearest heavier neighbor
  * DM structure: spin parameter (Bullock λ'), shape (axis ratios via inertia tensor)
  * Simple resolution checks (N200, eps/R200 if provided)
- If gas present:
  * Cold & dense mass fractions (T<300 K; n_H bins)
  * Basic n_H and T histograms (optional plots)

Outputs
-------
- Text report to stdout (and optional -o file)
- Optional JSON (--json)
- Optional plots (--make-plots) saved into --plot-dir (default: ./halo_plots)

Requirements: numpy, h5py, matplotlib (only for --make-plots)
"""

import argparse, json, os, sys, math
import numpy as np
import h5py

MSUN_IN_G   = 1.98847e33
CM_PER_MPC  = 3.085678e24
CM_PER_KPC  = 3.085678e21
MPC_IN_M    = 3.085677581e22   # meters
G_SI        = 6.67430e-11      # m^3 kg^-1 s^-2
MSUN_IN_KG  = 1.98847e30
MP_G        = 1.6726219e-24    # g
X_H         = 0.76             # hydrogen mass fraction (approx)
G_MPC_KMS2_MSUN = 4.30091e-9   # G in (Mpc * (km/s)^2 / Msun)

def rho_crit_Msun_per_Mpc3(z, H0_km_s_Mpc, Om, Ol, Or=0.0):
    Ok = 1.0 - (Om + Ol + Or)
    Hz = H0_km_s_Mpc * math.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + Ol)
    H_s = Hz * 1000.0 / MPC_IN_M
    rho_c_SI = 3.0 * H_s*H_s / (8.0 * math.pi * G_SI)
    return rho_c_SI * (MPC_IN_M**3) / MSUN_IN_KG

def min_image_delta(a, b, box):
    d = a - b
    d -= np.round(d / box) * box
    return d

def nearest_index_periodic(points, center, box):
    d = min_image_delta(points, center[None,:], box)
    return int(np.argmin(np.einsum('ij,ij->i', d, d)))

def _unit_summary(ul_cm):
    if abs(ul_cm - CM_PER_MPC) / CM_PER_MPC < 1e-3: return "1 Mpc"
    if abs(ul_cm - CM_PER_KPC) / CM_PER_KPC < 1e-3: return "1 kpc (often kpc/h)"
    return f"{ul_cm:.3e} cm"

def load_fof(fof_fn):
    with h5py.File(fof_fn, "r") as f:
        H = f["Header"].attrs
        Om   = float(H.get("Omega0", 0.315))
        Ol   = float(H.get("OmegaLambda", 0.685))
        H0   = float(H.get("Hubble", 100.0))     # km/s/Mpc
        hub  = float(H.get("HubbleParam", 0.7))
        z    = float(H.get("Redshift", 0.0))
        Lcode= float(H["BoxSize"])
        uL   = float(H.get("UnitLength_in_cm", CM_PER_MPC))
        uM   = float(H.get("UnitMass_in_g", 1.989e43))
        to_Mpc = uL / CM_PER_MPC
        box_Mpc = Lcode * to_Mpc

        Gp = f.get("Group")
        if Gp is None:
            raise RuntimeError("FoF file missing /Group")

        pos = np.array(Gp["GroupPos"]) * to_Mpc                   # Mpc
        M   = np.array(Gp["GroupMass"])
        # convert GroupMass to Msun
        M_Msun = M * 1e10 if (M.size and np.nanmax(M) < 1e8) else M * (uM / MSUN_IN_G)

        R200_kpc = None
        if "Group_R_Crit200" in Gp:
            Rcand = np.array(Gp["Group_R_Crit200"])
        else:
            Rcand = None

        M200 = None
        if "Group_M_Crit200" in Gp:
            Mcand = np.array(Gp["Group_M_Crit200"])
            M200  = Mcand * 1e10 if (Mcand.size and np.nanmax(Mcand) < 1e8) else Mcand * (uM / MSUN_IN_G)

        # Decide R200 units by cross-check with M200 if possible
        R200_kpc_final = None
        if Rcand is not None:
            if M200 is not None and np.isfinite(M200).any():
                rho_c = rho_crit_Msun_per_Mpc3(z, H0, Om, Ol)
                R_from_M_kpc = ( (3.0 * np.maximum(M200,1e-40)) / (4.0*np.pi*200.0*rho_c) ) ** (1.0/3.0) * 1000.0
                # test Rcand as kpc (no /h) vs kpc/h
                err_kpc   = np.nanmedian(np.abs(Rcand        - R_from_M_kpc))
                err_kpc_h = np.nanmedian(np.abs(Rcand/hub    - R_from_M_kpc))
                R200_kpc_final = (Rcand/hub) if (err_kpc_h < err_kpc) else Rcand
            else:
                # Fallback: most SUBFIND catalogs are kpc/h
                R200_kpc_final = Rcand / hub

        return dict(pos=pos, Mtot=M_Msun, R200_kpc=R200_kpc_final, M200=M200,
                    box=box_Mpc, z=z, H0=H0, Om=Om, Ol=Ol, hub=hub,
                    unitlength_cm=uL)

def load_snapshot_minimal(snap_fn):
    with h5py.File(snap_fn, "r") as f:
        H = f["Header"].attrs
        uL = float(H.get("UnitLength_in_cm", CM_PER_MPC))
        uM = float(H.get("UnitMass_in_g", 1.989e43))
        uV = float(H.get("UnitVelocity_in_cm_per_s", 1e5))
        box_code = float(H["BoxSize"])
        to_Mpc   = uL / CM_PER_MPC
        to_kms   = uV / 1e5
        box_Mpc  = box_code * to_Mpc
        z        = float(H.get("Redshift", 0.0))
        hub      = float(H.get("HubbleParam", 0.7))

        def get_type(name):
            return name in f

        present = {pt: get_type(pt) for pt in ["PartType0","PartType1","PartType2","PartType4"]}
        return dict(unitlength_cm=uL, unitmass_g=uM, unitvel_cms=uV, box=box_Mpc,
                    to_Mpc=to_Mpc, to_kms=to_kms, z=z, hub=hub, present=present)

def read_particles_in_sphere(snap_fn, center_Mpc, R_Mpc, to_Mpc, to_kms, want_gas=True, want_stars=True):
    """Return dict with DM_HR (PT1), DM_LR (PT2 if present), gas (PT0), stars (PT4) within radius."""
    out = {}
    with h5py.File(snap_fn, "r") as f:
        box_code = float(f["Header"].attrs["BoxSize"])
        uL = float(f["Header"].attrs.get("UnitLength_in_cm", CM_PER_MPC))
        box = box_code * (uL/CM_PER_MPC)
        c = np.asarray(center_Mpc)

        def select(pt):
            if pt not in f: return None
            pos = f[f"{pt}/Coordinates"][:] * (uL/CM_PER_MPC)
            d   = min_image_delta(pos, c[None,:], box)
            r   = np.sqrt(np.einsum('ij,ij->i', d, d))
            msk = r <= R_Mpc
            out = dict(pos=pos[msk], r=r[msk], vel=None, mass=None, ids=None, extra={})
            if f"{pt}/Velocities" in f:
                out["vel"] = f[f"{pt}/Velocities"][:][msk] * to_kms   # km/s
            if f"{pt}/ParticleIDs" in f:
                out["ids"] = f[f"{pt}/ParticleIDs"][:][msk]
            # per-particle masses or MassTable
            if f"{pt}/Masses" in f:
                out["mass"] = f[f"{pt}/Masses"][:][msk]
            else:
                mt = f["Header"].attrs.get("MassTable")
                if mt is not None:
                    # MassTable is in code mass units; convert later
                    type_index = {'PartType0':0,'PartType1':1,'PartType2':2,'PartType3':3,'PartType4':4,'PartType5':5}[pt]
                    out["mass"] = np.full(msk.sum(), mt[type_index], dtype=np.float64)
            # gas extras
            if pt=="PartType0":
                for key in ["InternalEnergy","Temperature","Density"]:
                    if f"{pt}/{key}" in f:
                        out["extra"][key] = f[f"{pt}/{key}"][:][msk]
            return out

        out["pt1"] = select("PartType1")   # HR DM by convention
        out["pt2"] = select("PartType2")   # LR DM if present
        out["pt0"] = select("PartType0") if want_gas else None
        out["pt4"] = select("PartType4") if want_stars else None

    return out

def total_mass_Msun(m_code, unitmass_g):
    if m_code is None: return 0.0
    return float(np.sum(m_code) * (unitmass_g / MSUN_IN_G))

def inertia_tensor(positions, masses):
    # unweighted if masses None
    if positions.size == 0: return np.eye(3)
    w = masses if masses is not None else np.ones(positions.shape[0])
    x = positions - np.average(positions, axis=0, weights=w)
    I = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            I[i,j] = np.sum(w * x[:,i] * x[:,j])
    return I

def spin_parameter_bullock(positions, velocities, masses, center, vbulk, M_Msun, R_Mpc):
    """λ' = J / (sqrt(2) M Vc R) with Vc = sqrt(G M / R); units: Mpc, km/s, Msun."""
    if positions is None or positions.size==0: return np.nan
    w = masses if masses is not None else np.ones(positions.shape[0])
    x = positions - center[None,:]
    v = velocities - vbulk[None,:]
    J = np.sum(np.cross(x, v) * w[:,None], axis=0)  # Msun * Mpc * km/s (up to code mass conv)
    Vc = math.sqrt(G_MPC_KMS2_MSUN * M_Msun / R_Mpc)
    J_mag = np.linalg.norm(J) * (MSUN_IN_G / MSUN_IN_G)  # masses should already be in Msun-units scaling when Vc uses Msun
    return np.linalg.norm(J) / (math.sqrt(2.0) * M_Msun * Vc * R_Mpc)

def print_report(rep, out_txt=None):
    lines = []
    lines.append(f"=== Halo Quality Report ===")
    lines.append(f"Snapshot: {rep['snapshot']}")
    lines.append(f"FoF:      {rep['fof']}")
    lines.append(f"z = {rep['z']:.3f} | Box = {rep['box']:.3f} Mpc")
    lines.append(f"Target halo index: {rep['halo_index']}")
    lines.append(f"Center (Mpc): ({rep['center'][0]:.3f}, {rep['center'][1]:.3f}, {rep['center'][2]:.3f})")
    lines.append(f"M200 = {rep['M200_Msun']:.3e} Msun | R200 = {rep['R200_Mpc']*1000:.1f} kpc")
    lines.append(f"N_DM(<R200): HR={rep['N_dm_hr']}  LR={rep['N_dm_lr']}  (contam={rep['contam_frac']*100:.2f}%)")
    if rep.get("N_gas", None) is not None:
        lines.append(f"N_gas(<R200)={rep['N_gas']}  N_star(<R200)={rep.get('N_star',0)}")
    lines.append(f"Isolation: nearest heavier neighbor at {rep['isolation_Mpc']:.3f} Mpc")
    lines.append(f"Spin λ' (DM) = {rep['spin_lambda']:.3f} | Shape (b/a, c/a) = ({rep['shape_ba']:.2f}, {rep['shape_ca']:.2f})")
    lines.append(f"Resolution checks: N200 >= 1000? {'YES' if rep['N_dm_hr']>=1000 else 'NO'} | eps/R200 ~ {rep.get('eps_over_R200','n/a')}")
    if rep.get("gas_stats"):
        gs = rep["gas_stats"]
        lines.append(f"Gas (within R200): M_gas={gs['M_gas_Msun']:.3e} Msun | cold(<300K) frac={gs['cold_frac']*100:.1f}%")
        if 'dense_frac' in gs:
            lines.append(f"  Dense gas fraction (n_H>10 cm^-3): {gs['dense_frac']*100:.1f}%")
    text = "\n".join(lines)
    print(text)
    if out_txt:
        with open(out_txt, "w") as fp: fp.write(text + "\n")
        print(f"[saved] {out_txt}")

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Assess a chosen halo's suitability for dust physics in a Gadget-4 zoom snapshot.")
    ap.add_argument("snapshot")
    ap.add_argument("fof")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--halo-index", type=int, help="FoF halo index to analyze")
    sel.add_argument("--center", type=float, nargs=3, metavar=("X","Y","Z"), help="Pick halo nearest to (X,Y,Z) in Mpc (display units detected)")
    ap.add_argument("--r-mult", type=float, default=1.0, help="Multiple of R200 for selection radius")
    ap.add_argument("--eps-dm-kpc", type=float, default=None, help="(Optional) DM HR softening (physical kpc) to report eps/R200")
    ap.add_argument("--make-plots", action="store_true", help="Generate simple plots (requires matplotlib)")
    ap.add_argument("--plot-dir", default="halo_plots", help="Directory for plots if --make-plots")
    ap.add_argument("-o","--output", default=None, help="Save the text report to file")
    ap.add_argument("--json", default=None, help="Save a JSON copy of the report")
    args = ap.parse_args()

    fof = load_fof(args.fof)
    snap_min = load_snapshot_minimal(args.snapshot)

    # Pick halo
    if args.halo_index is not None:
        i = int(args.halo_index)
    else:
        ctr = np.array(args.center, dtype=float)
        i = nearest_index_periodic(fof["pos"], ctr, fof["box"])
    center = fof["pos"][i]

    # Decide R200
    if fof["R200_kpc"] is not None:
        R200_Mpc = float(fof["R200_kpc"][i]) / 1000.0
        M200_Msun = float(fof["M200"][i]) if fof["M200"] is not None else float(fof["Mtot"][i])
        Rsrc = "Group_R_Crit200"
    else:
        if fof["M200"] is None:
            raise RuntimeError("FoF lacks both Group_R_Crit200 and Group_M_Crit200.")
        M200_Msun = float(fof["M200"][i])
        rho_c = rho_crit_Msun_per_Mpc3(fof["z"], fof["H0"], fof["Om"], fof["Ol"])
        R200_Mpc = float((3.0 * M200_Msun / (4.0*np.pi*200.0*rho_c)) ** (1.0/3.0))
        Rsrc = "computed from Group_M_Crit200"

    # Isolation: nearest heavier neighbor
    dvec = min_image_delta(fof["pos"], center[None,:], fof["box"])
    rr   = np.sqrt(np.einsum('ij,ij->i', dvec, dvec))
    heavier = fof["Mtot"] >= M200_Msun * 1.0
    heavier[i] = False
    iso = float(np.min(rr[heavier])) if np.any(heavier) else np.inf

    # Read particles within radius
    Rsel = args.r_mult * R200_Mpc
    parts = read_particles_in_sphere(args.snapshot, center, Rsel,
                                     snap_min["to_Mpc"], snap_min["to_kms"],
                                     want_gas=True, want_stars=True)

    uM = snap_min["unitmass_g"]
    # Convert per-particle masses from code mass units to Msun if present
    def mass_to_Msun(marr):
        return marr * (uM / MSUN_IN_G) if marr is not None else None

    for pt in ["pt0","pt1","pt2","pt4"]:
        if parts[pt] is not None and parts[pt]["mass"] is not None:
            parts[pt]["mass_Msun"] = mass_to_Msun(parts[pt]["mass"])
        else:
            parts[pt] = parts[pt]  # unchanged

    # Counts & contamination
    N_hr = parts["pt1"]["pos"].shape[0] if parts["pt1"] is not None else 0
    N_lr = parts["pt2"]["pos"].shape[0] if parts["pt2"] is not None else 0
    contam = (N_lr / max(N_hr+N_lr,1)) if (N_hr+N_lr)>0 else 0.0

    # Bulk velocity (DM HR if present, else all DM)
    def stack(arrs):
        a = [x for x in arrs if x is not None and x["pos"].size>0]
        return a

    dm_for_bulk = stack([parts["pt1"], parts["pt2"]])
    if len(dm_for_bulk)==0: vbulk = np.zeros(3)
    else:
        vv = np.vstack([p["vel"] for p in dm_for_bulk if p["vel"] is not None])
        ww = np.hstack([p["mass_Msun"] if p.get("mass_Msun") is not None else np.ones(p["pos"].shape[0]) for p in dm_for_bulk])
        if vv.size==0:
            vbulk = np.zeros(3)
        else:
            vbulk = np.average(vv, axis=0, weights=ww)

    # Spin (use DM HR if present)
    spin = np.nan
    if parts["pt1"] is not None and parts["pt1"]["pos"].size>0 and parts["pt1"]["vel"] is not None:
        m_dm = parts["pt1"]["mass_Msun"] if parts["pt1"].get("mass_Msun") is not None else np.ones(parts["pt1"]["pos"].shape[0])
        spin = spin_parameter_bullock(parts["pt1"]["pos"], parts["pt1"]["vel"], m_dm, center, vbulk, M200_Msun, R200_Mpc)

    # Shape (DM HR positions)
    ba = ca = np.nan
    if parts["pt1"] is not None and parts["pt1"]["pos"].size>0:
        I = inertia_tensor(parts["pt1"]["pos"], parts["pt1"].get("mass_Msun"))
        w, _ = np.linalg.eigh(I)
        w_sorted = np.sort(w)  # ascending
        # semi-axes ∝ sqrt(eigenvalues)
        a = math.sqrt(max(w_sorted[2],1e-30))
        b = math.sqrt(max(w_sorted[1],1e-30))
        c = math.sqrt(max(w_sorted[0],1e-30))
        ba = b/a; ca = c/a

    # Gas diagnostics (if present)
    gas_stats = None
    if parts["pt0"] is not None and parts["pt0"]["pos"].size>0:
        mg = parts["pt0"]["mass_Msun"] if parts["pt0"].get("mass_Msun") is not None else None
        M_gas = float(np.sum(mg)) if mg is not None else 0.0

        # Try temperature; if not present, estimate from InternalEnergy (assumes monoatomic ideal gas, mu~0.6)
        T = None
        if "Temperature" in parts["pt0"]["extra"]:
            T = parts["pt0"]["extra"]["Temperature"]
        elif "InternalEnergy" in parts["pt0"]["extra"]:
            # very rough: u in code → (needs mean molecular weight, gamma, etc). Skip if unknown.
            pass

        cold_frac = np.nan
        if T is not None:
            cold_frac = float(np.sum((T<300.0) * (mg if mg is not None else 1.0)) / max(np.sum(mg) if mg is not None else len(T), 1.0))

        # Density → n_H
        dense_frac = np.nan
        if "Density" in parts["pt0"]["extra"]:
            # Gadget stores density in code mass units / (code length)^3.
            # Convert: rho_phys = rho_code * (unitmass_g / unitlength_cm^3)
            rho_code = parts["pt0"]["extra"]["Density"]
            rho_cgs  = rho_code * (snap_min["unitmass_g"] / (snap_min["unitlength_cm"]**3))  # g/cm^3
            n_H = (X_H * rho_cgs) / MP_G
            if mg is not None:
                dense_frac = float(np.sum((n_H>10.0) * mg) / max(np.sum(mg),1.0))
            else:
                dense_frac = float(np.mean(n_H>10.0))

        gas_stats = dict(M_gas_Msun=M_gas, cold_frac=cold_frac, dense_frac=dense_frac)

    # Simple resolution proxy: eps/R200 if user provides DM epsilon
    eps_over_R200 = None
    if args.eps_dm_kpc is not None:
        eps_over_R200 = (args.eps_dm_kpc/1000.0) / R200_Mpc

    # Build report dict
    rep = dict(
        snapshot=args.snapshot, fof=args.fof, z=fof["z"], box=fof["box"],
        halo_index=i, center=center.tolist(),
        M200_Msun=M200_Msun, R200_Mpc=R200_Mpc, R_source=Rsrc,
        N_dm_hr=N_hr, N_dm_lr=N_lr, contam_frac=contam,
        isolation_Mpc=iso, spin_lambda=float(spin), shape_ba=float(ba), shape_ca=float(ca),
        eps_over_R200=(f"{eps_over_R200:.3f}" if eps_over_R200 is not None else "n/a"),
        N_gas=(parts["pt0"]["pos"].shape[0] if parts["pt0"] is not None else None),
        N_star=(parts["pt4"]["pos"].shape[0] if parts["pt4"] is not None else None),
        gas_stats=gas_stats
    )

    # Print & save
    print_report(rep, args.output)
    if args.json:
        with open(args.json, "w") as fp: json.dump(rep, fp, indent=2)
        print(f"[saved] {args.json}")

    # Optional plots
    if args.make_plots:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        os.makedirs(args.plot_dir, exist_ok=True)

        # DM surface-density slice (full projection) centered on halo
        def sd_map(pt, title, fname):
            if pt is None or pt["pos"].size==0: return
            # shift to center with PBC and project
            # For quick look, use positions already absolute; just project a square of side 2*Rsel around center
            pos = pt["pos"]
            # recentre with min-image wrt halo center
            # (we already selected by radius; plotting all inside sphere)
            x = pos[:,0]; y = pos[:,1]; z = pos[:,2]
            # choose a plane (XY)
            res = 512
            box = fof["box"]
            # Put a fixed window of 2*Rsel around center
            xmin,xmax = center[0]-Rsel, center[0]+Rsel
            ymin,ymax = center[1]-Rsel, center[1]+Rsel
            m = (x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax)
            if not np.any(m): return
            H, xedges, yedges = np.histogram2d(x[m], y[m], bins=res, range=[[xmin,xmax],[ymin,ymax]],
                                               weights=(pt.get("mass_Msun", None)[m] if pt.get("mass_Msun", None) is not None else None))
            px = ((xmax-xmin)/res)**2
            S = (H.T) / px
            S[S<=0] = np.nanmin(S[S>0]) / 100.0
            plt.figure(figsize=(6,5))
            plt.imshow(S, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap="magma_r", norm=LogNorm())
            plt.colorbar(label=r'$\Sigma\ [M_\odot\, \mathrm{Mpc}^{-2}]$')
            plt.xlabel("x [Mpc]"); plt.ylabel("y [Mpc]")
            plt.title(title)
            outf = os.path.join(args.plot_dir, fname)
            plt.tight_layout(); plt.savefig(outf, dpi=200); plt.close()
            print(f"[plot] {outf}")

        sd_map(parts["pt1"], f"DM-HR around halo #{i} (XY, R={Rsel:.3f} Mpc)", f"dm_hr_xy_halo{i}.png")

        # Gas phase diagram if gas present
        if parts["pt0"] is not None and parts["pt0"]["pos"].size>0 and "Density" in parts["pt0"]["extra"]:
            rho_code = parts["pt0"]["extra"]["Density"]
            rho_cgs  = rho_code * (snap_min["unitmass_g"] / (snap_min["unitlength_cm"]**3))
            n_H = (X_H * rho_cgs) / MP_G
            T = parts["pt0"]["extra"].get("Temperature", None)
            if T is not None:
                plt.figure(figsize=(6,5))
                plt.hist2d(np.log10(n_H+1e-40), np.log10(T), bins=200, cmap="viridis")
                plt.xlabel(r"log $n_{\rm H}$ [cm$^{-3}$]"); plt.ylabel(r"log $T$ [K]")
                plt.title(f"Gas phase diagram (<{args.r_mult} R200)")
                outf = os.path.join(args.plot_dir, f"gas_phase_halo{i}.png")
                plt.tight_layout(); plt.savefig(outf, dpi=200); plt.close()
                print(f"[plot] {outf}")

if __name__ == "__main__":
    main()
