#!/usr/bin/env python3
"""
Extract ParticleIDs within N * R_200c of a target halo near a given (x,y,z),
and (optionally) print / dump the corresponding particle coordinates.

- Positions handled in Mpc (comoving), masses in Msun.
- Finds the FoF halo whose GroupPos is nearest to the provided (x,y,z) using periodic BCs.
- Uses Group_R_Crit200 if present; otherwise computes it from Group_M_Crit200 and rho_crit(z).
- Writes DM ParticleIDs (PartType1/ParticleIDs) within factor*R_200c to a text file.

Extras:
- --print-xyz N         print N sample coordinates to stdout
- --relative            print wrapped offsets from halo center instead of absolute coords
- --dump-xyz FILE       write all selected coords to FILE (optionally with IDs via --include-ids)

Usage:
python save_particleids_for_desired_halo.py ../output_genetic_128_parent/snapshot_061.hdf5 ../output_genetic_128_parent/fof_subhalo_tab_061.hdf5 \
      --center 5.861 46.592 37.217 --factor 2.0 \
      --print-xyz 20 \
      -o ids_2r200.txt

"""

import argparse
import h5py
import numpy as np
import sys
import math

MSUN_IN_G  = 1.98847e33
CM_PER_MPC = 3.085678e24
CM_PER_KPC = 3.085678e21
G_SI       = 6.67430e-11        # m^3 kg^-1 s^-2
MPC_IN_M   = 3.085677581e22     # meters
MSUN_IN_KG = 1.98847e30

def ez(z, Om, Ol, Or=0.0, Ok=None):
    if Ok is None:
        Ok = 1.0 - (Om + Ol + Or)
    return math.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + Ol)

def rho_crit_Msun_per_Mpc3(z, H0_km_s_Mpc, Om, Ol):
    Hz = H0_km_s_Mpc * ez(z, Om, Ol)            # km/s/Mpc
    H_s = Hz * 1000.0 / MPC_IN_M                # s^-1
    rho_c_SI = 3.0 * H_s*H_s / (8.0 * math.pi * G_SI)  # kg/m^3
    return rho_c_SI * (MPC_IN_M**3) / MSUN_IN_KG       # Msun/Mpc^3

def nearest_halo_index(pos_all, center, box):
    """Index of halo nearest to center (periodic minimum image)."""
    d = pos_all - center[None,:]
    d -= np.round(d / box) * box
    return int(np.argmin(np.sum(d*d, axis=1)))

def load_snapshot_dm(filename):
    with h5py.File(filename, "r") as f:
        hdr = f["Header"].attrs
        unit_len_cm = float(hdr.get("UnitLength_in_cm", CM_PER_MPC))
        box_code    = float(hdr.get("BoxSize"))
        to_Mpc      = unit_len_cm / CM_PER_MPC
        box_Mpc     = box_code * to_Mpc
        coords      = f["PartType1/Coordinates"][:] * to_Mpc   # (N,3) in Mpc
        pids        = f["PartType1/ParticleIDs"][:]            # (N,)
        z           = float(hdr.get("Redshift", 0.0))
    return coords, pids, box_Mpc, z, hdr

def load_fof(filename):
    with h5py.File(filename, "r") as f:
        hdr = f["Header"].attrs
        unit_len_cm = float(hdr.get("UnitLength_in_cm", CM_PER_MPC))
        unit_mass_g = float(hdr.get("UnitMass_in_g", 1.989e43))
        H0          = float(hdr.get("Hubble", 100.0))          # km/s/Mpc
        Om          = float(hdr.get("Omega0", 0.315))
        Ol          = float(hdr.get("OmegaLambda", 0.685))
        z           = float(hdr.get("Redshift", 0.0))
        box_code    = float(hdr.get("BoxSize"))
        to_Mpc      = unit_len_cm / CM_PER_MPC

        grp = f.get("Group")
        if grp is None:
            raise RuntimeError("FoF file missing /Group")

        pos = np.array(grp.get("GroupPos")) * to_Mpc           # Mpc
        M   = np.array(grp.get("GroupMass"))
        # Many FoF outputs use units of 1e10 Msun; detect and convert
        mass_Msun = M * 1.0e10 if (M.size and np.nanmax(M) < 1e8) else M * (unit_mass_g / MSUN_IN_G)

        R200_kpc = None
        if "Group_R_Crit200" in grp:
            Rcand = np.array(grp["Group_R_Crit200"])
            # Heuristic: <5 means likely in Mpc
            R200_kpc = Rcand * 1000.0 if np.nanmedian(Rcand) < 5.0 else Rcand

        M200 = None
        if "Group_M_Crit200" in grp:
            Mcand = np.array(grp["Group_M_Crit200"])
            M200 = Mcand * 1.0e10 if (Mcand.size and np.nanmax(Mcand) < 1e8) else Mcand * (unit_mass_g / MSUN_IN_G)

        box_Mpc = box_code * to_Mpc

    return dict(pos=pos, mass=mass_Msun, R200_kpc=R200_kpc, M200=M200,
                box_Mpc=box_Mpc, z=z, H0=H0, Om=Om, Ol=Ol)

def main():
    ap = argparse.ArgumentParser(description="Write ParticleIDs within N*R200c and (optionally) print/dump coordinates.")
    ap.add_argument("snapshot", help="Gadget HDF5 snapshot")
    ap.add_argument("fof", help="FoF HDF5 catalog (same snapshot)")
    ap.add_argument("--center", type=float, nargs=3, required=True, metavar=("X","Y","Z"),
                    help="Target center in Mpc (comoving)")
    ap.add_argument("--factor", type=float, default=2.0, help="Multiple of R200c to select (default 2.0)")
    ap.add_argument("--safety", type=float, default=1.0, help="Extra inflation factor on radius")
    ap.add_argument("-o","--output", default="particle_ids_within_radius.txt", help="Output text filename")
    # NEW:
    ap.add_argument("--print-xyz", type=int, default=0, help="Print N sample coordinates to stdout (0=off)")
    ap.add_argument("--relative", action="store_true", help="Print wrapped offsets from halo center instead of absolute coords")
    ap.add_argument("--dump-xyz", default=None, help="Optional file to dump ALL selected coordinates")
    ap.add_argument("--include_ids", action="store_true", help="Include ParticleIDs in printed/dumped coordinates")
    args = ap.parse_args()

    center = np.array(args.center, dtype=float)

    # Load data
    coords, pids, box, z_snap, _ = load_snapshot_dm(args.snapshot)
    fof = load_fof(args.fof)
    if abs(fof["box_Mpc"] - box) > 1e-5:
        print(f"WARNING: snapshot box={box:.6g} Mpc, FoF box={fof['box_Mpc']:.6g} Mpc")

    # Nearest halo (periodic)
    i = nearest_halo_index(fof["pos"], center, box)
    halo_center = fof["pos"][i]
    M_halo = fof["mass"][i]

    # R200c [Mpc]
    if fof["R200_kpc"] is not None:
        R200_Mpc = float(fof["R200_kpc"][i]) / 1000.0
        src = "Group_R_Crit200"
    elif fof["M200"] is not None:
        rho_c = rho_crit_Msun_per_Mpc3(fof["z"], fof["H0"], fof["Om"], fof["Ol"])
        R200_Mpc = (3.0 * fof["M200"][i] / (4.0 * math.pi * 200.0 * rho_c)) ** (1.0/3.0)
        src = "computed from Group_M_Crit200"
    else:
        print("ERROR: Neither Group_R_Crit200 nor Group_M_Crit200 present in FoF file.", file=sys.stderr)
        sys.exit(2)

    R_select = args.factor * args.safety * R200_Mpc

    # Select DM particles within radius (periodic minimum image)
    d = coords - halo_center[None,:]
    d -= np.round(d / box) * box                  # wrapped offsets from halo center
    r = np.sqrt(np.sum(d*d, axis=1))
    m = (r <= R_select)

    ids_in = pids[m]
    sel_coords_abs = coords[m]                    # absolute coords in box [0,Box)
    sel_offsets = d[m]                            # wrapped Δ to halo center
    count = ids_in.size

    # Diagnostics
    shift = (halo_center - center) - np.round((halo_center - center)/box)*box
    print(f"Nearest halo index: {i}")
    print(f"  Halo center (Mpc): ({halo_center[0]:.3f}, {halo_center[1]:.3f}, {halo_center[2]:.3f})")
    print(f"  Provided center:   ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  [Δ = {np.linalg.norm(shift):.4f} Mpc]")
    print(f"  M_halo:            {M_halo:.3e} Msun")
    print(f"  R200c source:      {src}")
    print(f"  R200c:             {R200_Mpc*1000:.1f} kpc  ({R200_Mpc:.3f} Mpc)")
    print(f"  Select radius:     factor={args.factor} * safety={args.safety} → {R_select:.3f} Mpc")
    print(f"  Selected DM count: {count} / {len(coords)}")

    # Spread in 3D (using wrapped offsets)
    if count > 0:
        dx, dy, dz = sel_offsets[:,0], sel_offsets[:,1], sel_offsets[:,2]
        print(f"Δx range [{dx.min():.4f}, {dx.max():.4f}] Mpc; std={dx.std():.4f}")
        print(f"Δy range [{dy.min():.4f}, {dy.max():.4f}] Mpc; std={dy.std():.4f}")
        print(f"Δz range [{dz.min():.4f}, {dz.max():.4f}] Mpc; std={dz.std():.4f}")

    # Print N sample coordinates to stdout
    if args.print_xyz and count > 0:
        n = min(args.print_xyz, count)
        # deterministic but shuffled sample
        rng = np.random.default_rng(1234)
        idx = rng.choice(count, size=n, replace=False)
        if args.relative:
            xyz = sel_offsets[idx]
            label = "dX dY dZ (Mpc, wrapped to halo)"
        else:
            xyz = sel_coords_abs[idx]
            label = "X Y Z (Mpc, absolute)"
        print(f"\nSample {n} coordinates [{label}]:")
        if args.include_ids:
            for pid, row in zip(ids_in[idx], xyz):
                print(f"{int(pid)}  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
        else:
            for row in xyz:
                print(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")

    # Optional dump of ALL selected coordinates
    if args.dump_xyz and count > 0:
        with open(args.dump_xyz, "w") as fp:
            if args.include_ids:
                for pid, row in zip(ids_in, sel_coords_abs if not args.relative else sel_offsets):
                    fp.write(f"{int(pid)} {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
            else:
                for row in (sel_coords_abs if not args.relative else sel_offsets):
                    fp.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        print(f"Wrote coordinates to: {args.dump_xyz}  ({'relative' if args.relative else 'absolute'})")

    # Always write IDs (original behavior)
    with open(args.output, "w") as fp:
        for pid in ids_in:
            fp.write(f"{int(pid)}\n")
    print(f"Wrote ParticleIDs to: {args.output}")

if __name__ == "__main__":
    main()
