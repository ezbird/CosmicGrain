#!/usr/bin/env python3
"""
find_zoom_halos.py
------------------
Find all well-resolved, uncontaminated halos within the high-resolution zoom
region at z=0, and compute M_star and M_dust within R200 for each.

Results are written to a .npz file suitable for overlaying as individual
points on the M_dust vs M_star plot.

Usage:
    python find_zoom_halos.py /path/to/S10_output_1024/ [options]

    python find_zoom_halos.py ../S10_output_1024/ \\
        --snap 49 \\
        --output zoom_halos_z0.npz

    # Then overlay on the main plot:
    python plot_mdust_mstar.py ../S10_output_1024/ \\
        --simba-catalogs simba/snap_m50n512_151.hdf5 \\
        --zoom-halos zoom_halos_z0.npz \\
        --output mdust_mstar_final.png

Strategy
--------
1. Read the z=0 subfind FOF catalog to get GroupPos, R200, M200 for all groups.
2. For each group above a minimum mass threshold, load all particles within
   R200 from the snapshot.
3. Contamination check: if any particle has mass > hi_res_dm_mass * CONTAM_FACTOR,
   it is a low-resolution background DM particle (PartType2) — discard that halo.
4. Sum M_star (PartType4) and M_dust (PartType6) for clean halos.
5. Tag Halo 569 (the primary target) by matching its position to GroupPos.
"""

import argparse
import re
import sys
import numpy as np
import h5py
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_MASS = 1.989e33   # g

# Contamination threshold: a particle is "background" (low-res) if its mass
# exceeds the high-res DM particle mass by more than this factor.
# Low-res particles are typically 8x (level-1) or 64x (level-2) heavier.
CONTAM_FACTOR = 3.0   # conservative — flags anything > 3x the HR DM mass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_z0_snapshot(output_dir, snap_num=None):
    """
    Return (snap_file_0, catalog_file_0) for the requested snapshot number,
    or the highest-numbered snapshot if snap_num is None.
    """
    output_dir = Path(output_dir)

    if snap_num is not None:
        snapdir   = output_dir / f"snapdir_{snap_num:03d}"
        groups_dir = output_dir / f"groups_{snap_num:03d}"
    else:
        snapdirs = sorted(output_dir.glob("snapdir_*"))
        if not snapdirs:
            raise RuntimeError(f"No snapdir_* found in {output_dir}")
        snapdir = snapdirs[-1]
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        snap_num = int(m.group(1))
        groups_dir = output_dir / f"groups_{snap_num:03d}"

    snap_files = sorted(snapdir.glob("snap_*.hdf5")) + \
                 sorted(snapdir.glob("snapshot_*.hdf5"))
    cat_files  = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) \
                 if groups_dir.exists() else []

    if not snap_files:
        raise RuntimeError(f"No snapshot HDF5 files in {snapdir}")
    if not cat_files:
        raise RuntimeError(f"No subfind catalog in {groups_dir}")

    return snap_num, str(snap_files[0]), str(cat_files[0])


def get_snapshot_info(snap_file):
    """Return (redshift, h, box_kph, unit_mass_g, hi_res_dm_mass_code)."""
    with h5py.File(snap_file, "r") as f:
        hdr    = f["Header"].attrs
        params = f["Parameters"].attrs
        z      = float(hdr["Redshift"])
        h      = float(params["HubbleParam"])
        box    = float(hdr["BoxSize"])          # comoving kpc/h
        um     = float(params.get("UnitMass_in_g", 1.989e43))

        # High-res DM particle mass: read from MassTable[1] or from the
        # actual PartType1 Masses array (some runs store it there).
        mass_table = hdr.get("MassTable", None)
        hr_dm_mass = None
        if mass_table is not None and mass_table[1] > 0:
            hr_dm_mass = float(mass_table[1])
        elif "PartType1" in f and "Masses" in f["PartType1"]:
            hr_dm_mass = float(f["PartType1"]["Masses"][0])

    return z, h, box, um, hr_dm_mass


def load_all_group_info(catalog_file):
    """
    Load GroupPos, Group_R_Crit200, Group_M_Crit200 from all catalog chunks.
    Returns arrays sorted by descending M200 (as Gadget stores them).
    """
    p = Path(catalog_file)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    pos_list, r200_list, m200_list = [], [], []
    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_list.append(grp["GroupPos"][:])
            r200_list.append(grp["Group_R_Crit200"][:])
            m200_list.append(grp["Group_M_Crit200"][:])

    if not pos_list:
        return None, None, None

    return (np.concatenate(pos_list,  axis=0),
            np.concatenate(r200_list, axis=0),
            np.concatenate(m200_list, axis=0))


def load_particles_within_r200(snap_file_first, halo_center, r200,
                                box, part_types=(1, 4, 6)):
    """
    Load particles of requested types within r200 (comoving kpc/h).
    Returns dict {ptype: {'mass': arr, 'coords': arr}}.
    Handles multi-chunk snapshots.
    """
    p = Path(snap_file_first)
    stem = re.sub(r"\d+\.hdf5$", "", p.name)
    chunks = sorted(p.parent.glob(f"{stem}*.hdf5"))
    if not chunks:
        chunks = [p]

    result = {pt: {"mass": [], "coords": []} for pt in part_types}

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            mass_table = f["Header"].attrs.get("MassTable", None)

            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                coords = f[key]["Coordinates"][:]

                if "Masses" in f[key]:
                    masses = f[key]["Masses"][:]
                elif mass_table is not None and mass_table[pt] > 0:
                    masses = np.full(len(coords), float(mass_table[pt]))
                else:
                    continue

                # Periodic distance
                dx = coords - halo_center
                dx -= box * np.round(dx / box)
                r  = np.sqrt((dx**2).sum(axis=1))
                mask = r <= r200

                result[pt]["mass"].append(masses[mask])
                result[pt]["coords"].append(coords[mask])

    return {pt: {
                "mass":   np.concatenate(v["mass"])   if v["mass"]   else np.array([]),
                "coords": np.concatenate(v["coords"]) if v["coords"] else np.zeros((0, 3)),
            }
            for pt, v in result.items()}


def is_contaminated(dm_masses, hr_dm_mass, contam_factor=CONTAM_FACTOR):
    """
    Return True if any DM particle within R200 is a low-res background particle.
    """
    if len(dm_masses) == 0:
        return False
    return bool(np.any(dm_masses > hr_dm_mass * contam_factor))


def zoom_center_and_radius(all_group_pos, primary_idx=0,
                           zoom_radius_mpc_h=3500.0):
    """
    Estimate the zoom region center (from the primary halo) and a bounding
    radius in comoving kpc/h.  Default 3500 kpc/h ~ 5 Mpc/h is conservative
    for a typical 7 Mpc zoom patch.
    """
    center = all_group_pos[primary_idx]
    return center, zoom_radius_mpc_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find uncontaminated zoom halos at z=0 and measure M_star, M_dust")
    parser.add_argument("output_dir",
                        help="Gadget-4 output directory (e.g. ../S10_output_1024/)")
    parser.add_argument("--snap", type=int, default=None,
                        help="Snapshot number (default: last snapshot)")
    parser.add_argument("--output", default="zoom_halos_z0.npz",
                        help="Output .npz filename (default: zoom_halos_z0.npz)")
    parser.add_argument("--min-m200", type=float, default=1e10,
                        help="Minimum M200 in M_sun to consider (default: 1e10)")
    parser.add_argument("--zoom-radius", type=float, default=3500.0,
                        help="Zoom region radius in comoving kpc/h (default: 3500 "
                             "~ 5 Mpc/h, conservative for a 7 Mpc patch)")
    parser.add_argument("--primary-halo-index", type=int, default=0,
                        help="FOF index of the primary target halo (default: 0, "
                             "most massive = Halo 569)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Locate z=0 snapshot and catalog
    # ------------------------------------------------------------------
    snap_num, snap_file, catalog_file = find_z0_snapshot(
        args.output_dir, args.snap)
    z, h, box, unit_mass_g, hr_dm_mass = get_snapshot_info(snap_file)
    code_to_msun = unit_mass_g / SOLAR_MASS

    print(f"Snapshot {snap_num:03d}  z={z:.4f}")
    print(f"Box size:       {box:.1f} comoving kpc/h")
    print(f"HR DM mass:     {hr_dm_mass:.4e} code units "
          f"= {hr_dm_mass * code_to_msun:.3e} M_sun")
    print(f"Contam. floor:  {hr_dm_mass * CONTAM_FACTOR:.4e} code units\n")

    if hr_dm_mass is None:
        print("ERROR: could not determine high-res DM particle mass. "
              "Check MassTable in snapshot header.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load all group positions / R200 / M200
    # ------------------------------------------------------------------
    all_pos, all_r200, all_m200 = load_all_group_info(catalog_file)
    if all_pos is None:
        print("ERROR: no groups found in catalog.")
        sys.exit(1)

    n_total = len(all_pos)
    print(f"Total FOF groups in catalog: {n_total}")

    # Primary halo center defines the zoom region
    zoom_center, zoom_radius = zoom_center_and_radius(
        all_pos,
        primary_idx=args.primary_halo_index,
        zoom_radius_mpc_h=args.zoom_radius,
    )
    print(f"Zoom center (primary halo {args.primary_halo_index}): "
          f"{zoom_center} comoving kpc/h")
    print(f"Zoom search radius: {zoom_radius:.0f} comoving kpc/h "
          f"({zoom_radius / 1000 * h:.2f} Mpc)\n")

    # ------------------------------------------------------------------
    # 3. Pre-filter by mass and proximity to zoom center
    # ------------------------------------------------------------------
    min_m200_code = args.min_m200 / code_to_msun

    dx_to_center = all_pos - zoom_center
    dx_to_center -= box * np.round(dx_to_center / box)
    dist_to_center = np.sqrt((dx_to_center**2).sum(axis=1))

    candidate_mask = (all_m200 >= min_m200_code) & \
                     (dist_to_center <= zoom_radius) & \
                     (all_r200 > 0)
    candidate_idx = np.where(candidate_mask)[0]

    print(f"Candidate halos (M200 >= {args.min_m200:.0e} M_sun, "
          f"within {zoom_radius:.0f} kpc/h): {len(candidate_idx)}")

    # ------------------------------------------------------------------
    # 4. For each candidate: load particles, check contamination,
    #    sum M_star and M_dust
    # ------------------------------------------------------------------
    records = []

    for rank, gidx in enumerate(candidate_idx):
        pos  = all_pos[gidx]
        r200 = all_r200[gidx]
        m200 = all_m200[gidx] * code_to_msun
        is_primary = (gidx == args.primary_halo_index)

        parts = load_particles_within_r200(
            snap_file, pos, r200, box, part_types=(1, 4, 6))

        # Contamination check using DM (PartType1) within R200
        dm_masses = parts[1]["mass"]
        if len(dm_masses) == 0:
            print(f"  [{rank+1:3d}] group {gidx:4d}  SKIP (no DM particles found)")
            continue

        if is_contaminated(dm_masses, hr_dm_mass):
            lo_res_frac = (dm_masses > hr_dm_mass * CONTAM_FACTOR).mean()
            print(f"  [{rank+1:3d}] group {gidx:4d}  CONTAMINATED "
                  f"(low-res frac={lo_res_frac:.2f})  M200={m200:.2e} M_sun  SKIP")
            continue

        m_star = parts[4]["mass"].sum() * code_to_msun
        m_dust = parts[6]["mass"].sum() * code_to_msun

        if m_star <= 0:
            print(f"  [{rank+1:3d}] group {gidx:4d}  no stars  SKIP")
            continue

        log_ms = np.log10(m_star)
        log_md = np.log10(m_dust) if m_dust > 0 else np.nan

        tag = " ← PRIMARY (Halo 569)" if is_primary else ""
        print(f"  [{rank+1:3d}] group {gidx:4d}  "
              f"M200={m200:.2e}  R200={r200:.1f} kpc/h  "
              f"log(M*)={log_ms:.2f}  log(Md)={log_md:.2f}{tag}")

        records.append({
            "group_idx":  gidx,
            "is_primary": is_primary,
            "m200_msun":  m200,
            "r200_kph":   r200,
            "log_mstar":  log_ms,
            "log_mdust":  log_md,
        })

    print(f"\n{len(records)} clean halos found "
          f"({sum(r['is_primary'] for r in records)} primary)")

    if not records:
        print("No clean halos found — check --zoom-radius and --min-m200.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    out = Path(args.output)
    np.savez(
        out,
        group_idx  = np.array([r["group_idx"]  for r in records], dtype=int),
        is_primary = np.array([r["is_primary"]  for r in records], dtype=bool),
        m200_msun  = np.array([r["m200_msun"]   for r in records]),
        r200_kph   = np.array([r["r200_kph"]    for r in records]),
        log_mstar  = np.array([r["log_mstar"]   for r in records]),
        log_mdust  = np.array([r["log_mdust"]   for r in records]),
        redshift   = np.array([z]),
        snap_num   = np.array([snap_num], dtype=int),
    )
    print(f"\nSaved: {out}")
    print(f"\nTo overlay on the main plot, add:")
    print(f"  --zoom-halos {out}")


if __name__ == "__main__":
    main()
