#!/usr/bin/env python3
"""
cosmicgrain_zoom_postrun_audit.py

Post-run audit for the CosmicGrain zoom suite.

Uses zoom_halo_utils.py for a generic, particle-based halo definition and
prints catalog and particle-SO M200c/R200c side-by-side.

Apertures are specified in physical kpc and converted correctly to ckpc/h.
Masses are reported in physical Msun.
"""

import argparse
import h5py
import numpy as np

from zoom_halo_utils import (
    find_snapshot_and_group_files,
    get_zoom_halo,
    periodic_delta,
)


def read_type(snapshot_files, pt):
    coords, masses, fields = [], [], set()
    mt = None
    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            if mt is None:
                mt = np.asarray(f["Header"].attrs["MassTable"], dtype=float)
            name = f"PartType{pt}"
            if name not in f:
                continue
            g = f[name]
            fields.update(g.keys())
            c = np.asarray(g["Coordinates"], dtype=float)
            coords.append(c)
            if "Masses" in g:
                masses.append(np.asarray(g["Masses"], dtype=float))
            else:
                masses.append(np.full(len(c), mt[pt], dtype=float))
    if not coords:
        return None, None, fields
    return np.concatenate(coords), np.concatenate(masses), fields


def mass_inside(coords, masses, center, box, radius):
    d = periodic_delta(coords, center, box)
    r = np.linalg.norm(d, axis=1)
    q = r < radius
    return float(np.sum(masses[q])), int(np.count_nonzero(q)), r, q


def com_offset(coords, masses, center, box, radius):
    d = periodic_delta(coords, center, box)
    r = np.linalg.norm(d, axis=1)
    q = r < radius
    if not np.any(q) or np.sum(masses[q]) <= 0:
        return np.nan
    shift = np.average(d[q], axis=0, weights=masses[q])
    return float(np.linalg.norm(shift))


def read_scalar_field(snapshot_files, pt, field):
    out = []
    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            name = f"PartType{pt}"
            if name in f and field in f[name]:
                out.append(np.asarray(f[name][field], dtype=float))
    return np.concatenate(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--group-index", type=int, default=None)
    ap.add_argument("--galaxy-aperture-pkpc", type=float, default=30.0)
    ap.add_argument("--no-refine-center", action="store_true")
    args = ap.parse_args()

    snap_files, _ = find_snapshot_and_group_files(args.output_dir, args.snap)
    halo = get_zoom_halo(
        args.output_dir, args.snap,
        group_index=args.group_index,
        refine_center=not args.no_refine_center,
        verbose=False,
    )

    center = halo.chosen_center_ckpch
    r200 = halo.so_r200_ckpch
    box = halo.boxsize_ckpch
    a, h = halo.a, halo.h
    rap = args.galaxy_aperture_pkpc * h / a

    print("="*80)
    print("COSMICGRAIN ZOOM POST-RUN AUDIT")
    print("="*80)
    print(f"snap={args.snap:03d} z={halo.z:.6f} group={halo.group_index}")
    print()
    print("--- HALO DEFINITION ---")
    print(f"Catalog GroupPos       = {halo.catalog_center_ckpch}")
    print(f"Refined center         = {halo.refined_center_ckpch}")
    print(f"Refinement shift       = {halo.refinement_shift_ckpch:.3f} ckpc/h "
          f"= {halo.refinement_shift_ckpch*a/h:.3f} pkpc "
          f"= {halo.refinement_shift_ckpch/halo.catalog_r200_ckpch:.4f} catalog R200")
    print(f"Refinement accepted    = {halo.refinement_accepted}")
    print(f"Chosen center          = {halo.chosen_center_ckpch}")
    print()
    print(f"Catalog M200c          = {halo.catalog_m200_code*1e10/h:.6e} Msun")
    print(f"Particle-SO M200c      = {halo.so_m200_code*1e10/h:.6e} Msun")
    print(f"Catalog R200c          = {halo.catalog_r200_ckpch*a/h:.3f} pkpc")
    print(f"Particle-SO R200c      = {halo.so_r200_ckpch*a/h:.3f} pkpc")
    print(f"ΔM200 / catalog        = {(halo.so_m200_code/halo.catalog_m200_code-1)*100:+.2f}%")
    print(f"ΔR200 / catalog        = {(halo.so_r200_ckpch/halo.catalog_r200_ckpch-1)*100:+.2f}%")
    print(f"Galaxy aperture        = {args.galaxy_aperture_pkpc:.1f} pkpc "
          f"= {rap:.3f} ckpc/h")

    data = {}
    labels = {0:"Gas",1:"HR DM",2:"LR DM",4:"Stars",6:"Dust"}
    for pt in (0,1,2,4,6):
        data[pt] = read_type(snap_files, pt)

    print("\n--- HALO / GALAXY MASSES ---")
    aperture_values = {}
    for pt in (0,1,2,4,6):
        c,m,fields = data[pt]
        if c is None:
            print(f"{labels[pt]:8s}: absent")
            continue
        mh,nh,_,_ = mass_inside(c,m,center,box,r200)
        mg,ng,_,_ = mass_inside(c,m,center,box,rap)
        aperture_values[pt] = (mg,ng)
        print(
            f"{labels[pt]:8s}: <R200 N={nh:7d}, M={mh*1e10/h:12.5e} Msun"
            f" | <{args.galaxy_aperture_pkpc:g} pkpc N={ng:7d}, "
            f"M={mg*1e10/h:12.5e} Msun"
        )

    if 0 in aperture_values and 6 in aperture_values:
        gmass = aperture_values[0][0]
        dmass = aperture_values[6][0]
        if gmass > 0:
            print(f"D/G within {args.galaxy_aperture_pkpc:g} pkpc = {dmass/gmass:.6e}")

    print("\n--- CONTAMINATION ---")
    c2,m2,_ = data[2]
    if c2 is None:
        print("No PartType2 particles.")
    else:
        r2 = np.linalg.norm(periodic_delta(c2,center,box), axis=1)
        for level in np.unique(m2):
            q = np.isclose(m2,level,rtol=1e-7,atol=0)
            rq = r2[q]
            print(
                f"Type2 {level*1e10/h:.3e} Msun: "
                f"<R200={np.count_nonzero(rq<r200)}, "
                f"<2R200={np.count_nonzero(rq<2*r200)}, "
                f"<3R200={np.count_nonzero(rq<3*r200)}, "
                f"nearest={rq.min()/r200:.3f} R200"
            )

    print("\n--- CENTERING ---")
    for pt,label in ((4,"Stars"),(0,"Gas"),(6,"Dust")):
        c,m,_ = data[pt]
        if c is None:
            continue
        off = com_offset(c,m,center,box,rap)
        print(
            f"{label:8s} mass-weighted COM offset in aperture = "
            f"{off*a/h:.3f} pkpc"
        )

    print("\n--- NUMERICAL INTEGRITY ---")
    bad = False
    for pt in (0,1,2,4,6):
        c,m,fields = data[pt]
        if c is None:
            continue
        nf_c = int(np.count_nonzero(~np.isfinite(c)))
        nf_m = int(np.count_nonzero(~np.isfinite(m)))
        neg_m = int(np.count_nonzero(m < 0))
        print(
            f"PartType{pt}: nonfinite coords={nf_c}, "
            f"nonfinite masses={nf_m}, negative masses={neg_m}"
        )
        bad |= bool(nf_c or nf_m or neg_m)

    print("\n--- AVAILABLE SCIENCE FIELDS ---")
    for pt in (0,4,6):
        _,_,fields = data[pt]
        print(f"PartType{pt}: {', '.join(sorted(fields)) if fields else '(absent)'}")

    gas_c,gas_m,gas_fields = data[0]
    dust_c,dust_m,dust_fields = data[6]
    if gas_c is not None and dust_c is not None:
        zfield = next((x for x in ("Metallicity","GFM_Metallicity","Z") if x in gas_fields), None)
        if zfield:
            Z = read_scalar_field(snap_files,0,zfield)
            _,_,_,qg = mass_inside(gas_c,gas_m,center,box,rap)
            metal_mass = float(np.sum(gas_m[qg] * Z[qg]))
            dmass = aperture_values.get(6,(0.0,0))[0]
            if metal_mass > 0:
                print(
                    f"\nD/Z within {args.galaxy_aperture_pkpc:g} pkpc "
                    f"(using gas {zfield}) = {dmass/metal_mass:.6e}"
                )

    print("\nRESULT")
    print("PASS basic finite/mass integrity" if not bad
          else "WARNING: numerical integrity issues found")
    print("="*80)


if __name__ == "__main__":
    main()
