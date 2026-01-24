#!/usr/bin/env python3
import os, glob, h5py, numpy as np
import argparse, math, sys

MSUN_IN_G  = 1.98847e33
CM_PER_MPC = 3.085678e24
CM_PER_KPC = 3.085678e21

# ----------------------------- utils ----------------------------- #
def _get_scale_length(header, target="mpc"):
    """Return factor so x[target] = x_code * scale."""
    ul = float(header.get("UnitLength_in_cm", CM_PER_MPC))
    if target == "mpc":  return ul / CM_PER_MPC
    if target == "kpc":  return ul / CM_PER_KPC
    return 1.0

def _mass_unit_to_Msun(header):
    """Return factor so m[Msun] = m_code * factor."""
    um = float(header.get("UnitMass_in_g", 1.989e43))  # 1e10 Msun in grams is common
    return um / MSUN_IN_G

def _mass_array(f, pgroup, mass_to_Msun):
    """Return masses in Msun for a PartType group, handling Masses vs MassTable."""
    if f"{pgroup}/Coordinates" not in f:
        return None
    N = f[f"{pgroup}/Coordinates"].shape[0]
    if f"{pgroup}/Masses" in f:
        return np.array(f[f"{pgroup}/Masses"]) * mass_to_Msun
    # constant particle mass from header MassTable?
    mt = f["Header"].attrs.get("MassTable", None)
    if mt is not None and len(mt) >= (int(pgroup[-1])+1) and mt[int(pgroup[-1])] > 0:
        return np.full(N, float(mt[int(pgroup[-1])]) * mass_to_Msun, dtype=np.float64)
    return None

def _coords_array(f, pgroup, len_scale):
    if f"{pgroup}/Coordinates" not in f:
        return None
    return np.array(f[f"{pgroup}/Coordinates"]) * len_scale

def _dataset_or_none(f, path):
    return np.array(f[path]) if path in f else None

def _gas_metallicity_array(f, pgroup="PartType0"):
    """Return scalar metallicity Z for gas if available."""
    # Common names
    for name in (f"{pgroup}/Metallicity", f"{pgroup}/GFM_Metallicity"):
        if name in f:
            return np.array(f[name])
    # Vector of species (e.g., GFM_Metals) → sum heavy elements
    if f"{pgroup}/GFM_Metals" in f:
        arr = np.array(f[f"{pgroup}/GFM_Metals"])
        if arr.ndim == 2:
            return np.sum(arr[:, 1:], axis=1)  # skip H
    return None

def _gas_sfr_array(f, pgroup="PartType0"):
    for name in (f"{pgroup}/Sfr", f"{pgroup}/SFR", f"{pgroup}/StarFormationRate"):
        if name in f:
            return np.array(f[name])
    return None

def _dust_mass_from_ptype(f, mass_to_Msun):
    if "PartType6" in f:
        m = _mass_array(f, "PartType6", mass_to_Msun)
        return np.sum(m) if m is not None else 0.0
    # fallback: gas-carried dust fraction, if present
    for cand in ("PartType0/DustFraction", "PartType0/Dust_MassFraction", "PartType0/DustFrac"):
        if cand in f:
            gas_m = _mass_array(f, "PartType0", mass_to_Msun)
            frac  = np.array(f[cand])
            if gas_m is not None and frac is not None:
                return float(np.sum(gas_m * frac))
    return 0.0

def _roi_mask(coords, center, radius, box):
    """Periodic spherical ROI mask."""
    if coords is None:
        return None
    d = coords - center[None,:]
    d -= np.round(d/box) * box
    r = np.sqrt((d*d).sum(axis=1))
    return r <= radius

# ------------------------- FoF helpers (optional) ---------------------- #
def _read_fof(fof_path):
    with h5py.File(fof_path, "r") as f:
        hdr = f["Header"].attrs
        len_scale = _get_scale_length(hdr, "mpc")
        mass_scale = _mass_unit_to_Msun(hdr)
        grp = f.get("Group")
        if grp is None:
            raise RuntimeError("FoF file missing /Group group")
        pos = np.array(grp["GroupPos"]) * len_scale
        # Prefer R200c if present
        R200_kpc = None
        if "Group_R_Crit200" in grp:
            R = np.array(grp["Group_R_Crit200"])
            R200_kpc = R * (1000.0 if np.nanmedian(R) < 5.0 else 1.0)  # Mpc→kpc if needed
        M200 = None
        if "Group_M_Crit200" in grp:
            M = np.array(grp["Group_M_Crit200"])
            M200 = M * (1e10 if np.nanmax(M) < 1e8 else mass_scale)
        # Generic FoF mass if no M200
        Mfof = np.array(grp["GroupMass"]) if "GroupMass" in grp else None
        if Mfof is not None:
            Mfof = Mfof * (1e10 if np.nanmax(Mfof) < 1e8 else mass_scale)
        return dict(pos=pos, R200_kpc=R200_kpc, M200=M200, Mfof=Mfof, header=hdr)

# -------------------------- core computation --------------------------- #
def compute_ratios(snapshot_path, units="mpc", center=None, radius=None, fof=None, group_index=None):
    # open first file (or loop all files in snapdir)
    is_dir = os.path.isdir(snapshot_path)
    files = sorted(glob.glob(os.path.join(snapshot_path, "*.hdf5"))) if is_dir else [snapshot_path]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found for {snapshot_path}")

    # probe header from first file
    with h5py.File(files[0], "r") as f0:
        hdr = f0["Header"].attrs
        len_scale = _get_scale_length(hdr, units)
        mass_scale = _mass_unit_to_Msun(hdr)
        box = float(hdr["BoxSize"]) * len_scale
        time = float(hdr.get("Time", 1.0))
        try: redshift = 1.0/time - 1.0
        except Exception: redshift = float(hdr.get("Redshift", 0.0))

    # If FoF is given and we need a center/radius, resolve it
    if fof is not None:
        cat = _read_fof(fof)
        pos = cat["pos"]
        if group_index is None and center is not None:
            d = pos - center
            d -= np.round(d/box)*box
            gi = int(np.argmin(np.sum(d*d, axis=1)))
        elif group_index is None:
            gi = 0
        else:
            gi = int(group_index)
        halo_center = pos[gi]
        if cat["R200_kpc"] is not None:
            R_Mpc = float(cat["R200_kpc"][gi])/1000.0
            halo_mass = float(cat["M200"][gi]) if cat["M200"] is not None else None
            source = "R200c (FoF)"
        else:
            # fallback: rough 0.2 * mean interparticle spacing as radius if nothing present (not ideal)
            R_Mpc = None
            halo_mass = cat["M200"][gi] if cat["M200"] is not None else (cat["Mfof"][gi] if cat["Mfof"] is not None else None)
            source = "FoF mass only"
        # if user didn't specify center/radius, use FoF halo
        if center is None: center = halo_center
        if radius is None and R_Mpc is not None: radius = R_Mpc
    # If user provided ROI w/out FoF, just use it
    if center is not None and radius is None:
        raise ValueError("ROI center provided but not radius. Pass --radius as well, or provide a FoF file.")

    # accumulators
    Mstar = 0.0
    Mgas  = 0.0
    Mdust = 0.0
    Mdm   = 0.0
    SFR_tot = 0.0
    metal_mass_gas = 0.0

    for fp in files:
        with h5py.File(fp, "r") as f:
            # masses & coords per type (only if present)
            def sum_type(pt, count_dm_as_halo=False):
                nonlocal Mdm, Mgas, Mstar, Mdust, SFR_tot, metal_mass_gas
                coords = _coords_array(f, pt, len_scale)
                masses = _mass_array(f, pt, mass_scale)
                if coords is None or masses is None:
                    return
                msel = np.ones(masses.shape[0], dtype=bool)
                if center is not None:
                    msel = _roi_mask(coords, np.asarray(center), radius, box)
                if not np.any(msel):
                    return
                msum = float(masses[msel].sum())

                if pt == "PartType0":
                    Mgas += msum
                    # SFR
                    sfr = _gas_sfr_array(f, pt)
                    if sfr is not None:
                        SFR_tot += float(np.sum(sfr[msel]))
                    # metallicity → metal mass in gas
                    Z = _gas_metallicity_array(f, pt)
                    if Z is not None:
                        Zsel = Z[msel]
                        # If Z looks like >2 on average, it may be in solar units; we keep it as-is but note later.
                        metal_mass_gas += float(np.sum((masses[msel]) * Zsel))
                elif pt in ("PartType1","PartType2","PartType3","PartType5"):
                    Mdm += msum if count_dm_as_halo else 0.0
                elif pt == "PartType4":
                    Mstar += msum
                elif pt == "PartType6":
                    Mdust += msum

            # gas, stars, dust
            sum_type("PartType0")
            sum_type("PartType4")
            sum_type("PartType6")
            # dark matter (all flavors) → halo mass proxy if global/ROI
            sum_type("PartType1", count_dm_as_halo=True)
            sum_type("PartType2", count_dm_as_halo=True)
            sum_type("PartType3", count_dm_as_halo=True)
            sum_type("PartType5", count_dm_as_halo=True)

    # If FoF gave an authoritative halo mass, prefer it
    Mhalo = Mdm
    if fof is not None:
        if 'halo_mass' in locals() and halo_mass is not None:
            Mhalo = halo_mass

    # Ratios
    ratios = {}
    ratios["Mstar_Mhalo"] = (Mstar / Mhalo) if (Mhalo > 0) else np.nan
    ratios["Mstar_over_SFR_yr"] = (Mstar / SFR_tot) if (SFR_tot > 0) else np.nan
    ratios["Mstar_over_SFR_Gyr"] = ratios["Mstar_over_SFR_yr"] / 1e9 if np.isfinite(ratios["Mstar_over_SFR_yr"]) else np.nan

    # Gas metallicity and M*/<Z_gas>
    Z_gas_mass_weighted = (metal_mass_gas / Mgas) if (Mgas > 0 and metal_mass_gas > 0) else np.nan
    ratios["Z_gas_mass_weighted"] = Z_gas_mass_weighted
    ratios["Mstar_over_Zgas"] = (Mstar / Z_gas_mass_weighted) if (Z_gas_mass_weighted > 0) else np.nan

    # Dust ratios
    ratios["DTM"] = (Mdust / metal_mass_gas) if (metal_mass_gas > 0) else np.nan
    ratios["DGR"] = (Mdust / Mgas) if (Mgas > 0) else np.nan

    summary = dict(
        redshift=redshift,
        box=box,
        roi_center=center.tolist() if isinstance(center, np.ndarray) else (center if center is None else list(center)),
        roi_radius=radius,
        masses=dict(Mstar=Mstar, Mgas=Mgas, Mdust=Mdust, Mdm=Mdm, Mhalo_used=Mhalo),
        SFR=SFR_tot,
        ratios=ratios,
        units=dict(length=units, mass="Msun", sfr="(snapshot units; often Msun/yr)")
    )
    if fof is not None:
        summary["halo_source"] = locals().get("source", "FoF")
    return summary

# ------------------------------- CLI ----------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compute key mass/metal/dust ratios from a Gadget-4 snapshot.")
    ap.add_argument("snapshot", help="Path to snapshot_###.hdf5 file OR snapdir_### directory")
    ap.add_argument("--units", choices=["mpc","kpc","code"], default="mpc",
                    help="Units for ROI center/radius and box reporting (default: mpc)")
    ap.add_argument("--center", type=float, nargs=3, metavar=("X","Y","Z"),
                    help="ROI sphere center in requested units (default: whole box)")
    ap.add_argument("--radius", type=float, help="ROI sphere radius in requested units")
    ap.add_argument("--fof", help="Optional FoF catalog (same snapshot) to use R200c/M200c")
    ap.add_argument("--group-index", type=int, help="FoF group index to use (overrides nearest-to-center)")
    ap.add_argument("-o","--output", default=None, help="Write a JSON summary to this path")
    args = ap.parse_args()

    center = np.array(args.center, float) if args.center is not None else None
    try:
        out = compute_ratios(args.snapshot, units=args.units, center=center,
                             radius=args.radius, fof=args.fof, group_index=args.group_index)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # pretty print
    r = out["ratios"]; m = out["masses"]
    print(f"# Snapshot summary  z={out['redshift']:.3f}  box={out['box']:.3f} {args.units}")
    if out["roi_center"] is not None:
        print(f"  ROI center={out['roi_center']}  radius={out['roi_radius']:.4g} {args.units}")
    if "halo_source" in out:
        print(f"  Halo mass source: {out['halo_source']}")
    print("\nMasses [Msun]:")
    print(f"  Mstar = {m['Mstar']:.3e}   Mgas = {m['Mgas']:.3e}   Mdust = {m['Mdust']:.3e}   Mdm = {m['Mdm']:.3e}")
    print(f"  Mhalo_used = {m['Mhalo_used']:.3e}")
    print(f"SFR (sum) = {out['SFR']:.3e}  (units depend on your SF model; often Msun/yr)\n")

    print("Ratios:")
    print(f"  Mstar / Mhalo           = {r['Mstar_Mhalo']:.4e}")
    print(f"  Mstar / SFR (yr)        = {r['Mstar_over_SFR_yr']:.4e}   (~ {r['Mstar_over_SFR_Gyr']:.3f} Gyr)")
    print(f"  <Z_gas> (mass-weighted) = {r['Z_gas_mass_weighted']:.4e} (dimensionless mass fraction)")
    print(f"  Mstar / <Z_gas>         = {r['Mstar_over_Zgas']:.4e} [Msun]")
    print(f"  Dust-to-metals (DTM)    = {r['DTM']:.4e}")
    print(f"  Dust-to-gas (DGR)       = {r['DGR']:.4e}")

    if args.output:
        import json, os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"\nSaved JSON: {args.output}")

if __name__ == "__main__":
    main()
