#!/usr/bin/env python3
"""
cosmicgrain_global_conservation_audit.py

Whole-box bookkeeping audit for a CosmicGrain snapshot or a sequence of
snapshots. Designed as a pre-commit validation check.

Reports:
  * particle counts and total masses by PartType
  * total baryonic mass = gas + stars + dust
  * gas metal mass from PartType0/Metallicity
  * stellar metal mass from PartType4/Metallicity
  * dust mass as a separate condensed-metal reservoir
  * gas C/N/O/Ne/Mg/Si/Fe masses from element mass-fraction fields
  * dust carbon/silicate mass inferred from CarbonMassFraction
  * changes relative to the first requested snapshot

Important:
  "gas metals + stellar metals + dust" is a bookkeeping diagnostic, not
  automatically a strictly conserved quantity unless the code's Metallicity
  convention excludes dust consistently and stellar yields/remnants are
  accounted for accordingly. Baryonic mass, however, should be an especially
  strong global conservation check in a closed periodic box.
"""

import argparse
import glob
import os
import re
import h5py
import numpy as np

PTYPES = {
    0: "Gas",
    1: "HRDM",
    2: "LRDM",
    3: "Type3",
    4: "Stars",
    5: "BH",
    6: "Dust",
}

ELEMENT_FIELDS = {
    "C":  "GasCarbonMassFraction",
    "N":  "GasNitrogenMassFraction",
    "O":  "GasOxygenMassFraction",
    "Ne": "GasNeonMassFraction",
    "Mg": "GasMagnesiumMassFraction",
    "Si": "GasSiliconMassFraction",
    "Fe": "GasIronMassFraction",
}


def snapshot_files(output_dir, snap):
    pats = [
        os.path.join(output_dir, f"snapdir_{snap:03d}", f"snapshot_{snap:03d}.*.hdf5"),
        os.path.join(output_dir, f"snapdir_{snap:03d}", f"snapshot_{snap:03d}.hdf5"),
        os.path.join(output_dir, f"snapshot_{snap:03d}.*.hdf5"),
        os.path.join(output_dir, f"snapshot_{snap:03d}.hdf5"),
    ]
    for pat in pats:
        fs = sorted(glob.glob(pat))
        if fs:
            multi = [x for x in fs if re.search(r"\.\d+\.hdf5$", x)]
            return multi if multi else fs
    raise FileNotFoundError(f"Could not find snapshot {snap:03d} under {output_dir}")


def available_snapshots(output_dir):
    found = set()
    for pat in [
        os.path.join(output_dir, "snapdir_*", "snapshot_*.hdf5"),
        os.path.join(output_dir, "snapshot_*.hdf5"),
    ]:
        for fn in glob.glob(pat):
            m = re.search(r"snapshot_(\d+)", os.path.basename(fn))
            if m:
                found.add(int(m.group(1)))
    return sorted(found)


def scalar_metallicity(z):
    """Return one metallicity value per particle."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        return z
    # Some builds store multiple components. Do not silently guess.
    if z.ndim == 2 and z.shape[1] == 1:
        return z[:, 0]
    raise RuntimeError(
        f"Metallicity has shape {z.shape}; expected scalar per particle. "
        "Inspect this field before interpreting total metal mass."
    )


def analyze_snapshot(output_dir, snap):
    files = snapshot_files(output_dir, snap)

    with h5py.File(files[0], "r") as f:
        hdr = f["Header"].attrs
        a = float(hdr["Time"])
        z = float(hdr.get("Redshift", 1.0/a - 1.0))
        mt = np.asarray(hdr["MassTable"], dtype=np.float64)
        if "Parameters" in f and "HubbleParam" in f["Parameters"].attrs:
            h = float(f["Parameters"].attrs["HubbleParam"])
        else:
            h = float(hdr["HubbleParam"])

    result = {
        "snap": snap, "a": a, "z": z, "h": h,
        "count": {p: 0 for p in PTYPES},
        "mass": {p: 0.0 for p in PTYPES},
        "gas_metal": 0.0,
        "star_metal": 0.0,
        "elements": {e: 0.0 for e in ELEMENT_FIELDS},
        "dust_carbon": 0.0,
        "dust_silicate": 0.0,
        "bad_cf": 0,
        "nonfinite_mass": 0,
        "negative_mass": 0,
    }

    for fn in files:
        with h5py.File(fn, "r") as f:
            for p in PTYPES:
                key = f"PartType{p}"
                if key not in f:
                    continue
                g = f[key]
                n = len(g["Coordinates"]) if "Coordinates" in g else 0
                result["count"][p] += n

                if "Masses" in g:
                    mcode = np.asarray(g["Masses"][()], dtype=np.float64)
                else:
                    mcode = np.full(n, mt[p], dtype=np.float64)

                result["nonfinite_mass"] += int(np.count_nonzero(~np.isfinite(mcode)))
                result["negative_mass"] += int(np.count_nonzero(mcode < 0))
                m = mcode * 1e10 / h
                result["mass"][p] += float(np.nansum(m))

                if p == 0:
                    if "Metallicity" in g:
                        zz = scalar_metallicity(g["Metallicity"][()])
                        result["gas_metal"] += float(np.nansum(m * zz))
                    for elem, field in ELEMENT_FIELDS.items():
                        if field in g:
                            frac = np.asarray(g[field][()], dtype=np.float64)
                            if frac.ndim > 1:
                                frac = np.squeeze(frac)
                            result["elements"][elem] += float(np.nansum(m * frac))

                elif p == 4:
                    if "Metallicity" in g:
                        zz = scalar_metallicity(g["Metallicity"][()])
                        result["star_metal"] += float(np.nansum(m * zz))

                elif p == 6:
                    if "CarbonMassFraction" in g:
                        cf = np.asarray(g["CarbonMassFraction"][()], dtype=np.float64)
                        cf = np.squeeze(cf)
                        good = np.isfinite(cf) & (cf >= 0.0) & (cf <= 1.0)
                        result["bad_cf"] += int(np.count_nonzero(~good))
                        result["dust_carbon"] += float(np.nansum(m[good] * cf[good]))
                        result["dust_silicate"] += float(np.nansum(m[good] * (1.0-cf[good])))

    result["baryon"] = result["mass"][0] + result["mass"][4] + result["mass"][5] + result["mass"][6]
    result["dm"] = result["mass"][1] + result["mass"][2] + result["mass"][3]
    result["dust"] = result["mass"][6]
    result["metal_plus_dust"] = result["gas_metal"] + result["star_metal"] + result["dust"]
    result["tracked_gas_elements"] = sum(result["elements"].values())
    return result


def frac(now, ref):
    return (now-ref)/ref if ref != 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--snaps", nargs="*", type=int,
                    help="Snapshots to compare; default = all available snapshots")
    args = ap.parse_args()

    snaps = args.snaps if args.snaps else available_snapshots(args.output_dir)
    if not snaps:
        raise RuntimeError("No snapshots found")

    rows = [analyze_snapshot(args.output_dir, s) for s in snaps]
    ref = rows[0]

    print("="*104)
    print("COSMICGRAIN GLOBAL CONSERVATION / BOOKKEEPING AUDIT")
    print("="*104)
    print(f"Reference snapshot: {ref['snap']:03d}  z={ref['z']:.6f}")
    print()

    for r in rows:
        print("-"*104)
        print(f"SNAP {r['snap']:03d}   a={r['a']:.8f}   z={r['z']:.6f}")
        print("Particle counts:")
        print("  " + "  ".join(f"{PTYPES[p]}={r['count'][p]:,}" for p in PTYPES if r['count'][p]))
        print("Masses [Msun]:")
        for p in PTYPES:
            if r["count"][p]:
                print(f"  {PTYPES[p]:8s} {r['mass'][p]:.9e}")
        print(f"  {'BARYON':8s} {r['baryon']:.9e}   Δ/ref={frac(r['baryon'],ref['baryon']):+.6e}")
        print(f"  {'DM':8s} {r['dm']:.9e}   Δ/ref={frac(r['dm'],ref['dm']):+.6e}")

        print("Metal / dust bookkeeping [Msun]:")
        print(f"  Gas metallicity reservoir  {r['gas_metal']:.9e}")
        print(f"  Stellar metal reservoir    {r['star_metal']:.9e}")
        print(f"  Dust reservoir             {r['dust']:.9e}")
        print(f"  GasZ + StarZ + Dust        {r['metal_plus_dust']:.9e}")
        print(f"  Tracked gas elements sum   {r['tracked_gas_elements']:.9e}")

        present = [(e, m) for e, m in r["elements"].items() if m != 0]
        if present:
            print("  Gas elements:")
            print("    " + "  ".join(f"{e}={m:.5e}" for e, m in present))

        if r["count"][6]:
            print(f"  Dust carbon mass           {r['dust_carbon']:.9e}")
            print(f"  Dust silicate mass         {r['dust_silicate']:.9e}")
            if r["dust"] > 0:
                closure = (r["dust_carbon"] + r["dust_silicate"]) / r["dust"]
                print(f"  C+silicate / dust          {closure:.12f}")

        print(f"Integrity: nonfinite masses={r['nonfinite_mass']}, "
              f"negative masses={r['negative_mass']}, bad dust CF={r['bad_cf']}")

    print("="*104)
    last = rows[-1]
    baryon_err = abs(frac(last["baryon"], ref["baryon"]))
    dm_err = abs(frac(last["dm"], ref["dm"]))
    print("FINAL SUMMARY")
    print(f"Baryonic mass fractional change : {baryon_err:.6e}")
    print(f"DM mass fractional change       : {dm_err:.6e}")
    print("Interpret GasZ+StarZ+Dust as a bookkeeping trend unless your exact")
    print("Metallicity/yield convention makes it a formally conserved reservoir.")
    if last["nonfinite_mass"] == 0 and last["negative_mass"] == 0 and last["bad_cf"] == 0:
        print("PASS finite/mass/composition-integrity checks")
    else:
        print("WARNING integrity failures detected")
    print("="*104)


if __name__ == "__main__":
    main()
