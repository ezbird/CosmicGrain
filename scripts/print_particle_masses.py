#!/usr/bin/env python3
import argparse
import glob
import h5py
import numpy as np
import os
import sys
from collections import Counter

MSUN_IN_G = 1.989e33

def fmt_mass(x, unit_label):
    if x == 0 or not np.isfinite(x):
        return f"{x} {unit_label}"
    expo = int(np.floor(np.log10(abs(x))))
    if expo >= 6 or expo <= -3:
        return f"{x:.3e} {unit_label}"
    return f"{x:,.3f} {unit_label}"

def read_header(files):
    with h5py.File(files[0], 'r') as f:
        hd = f['Header'].attrs
        params = f['Parameters'].attrs
        header = {
            'MassTable': np.array(hd.get('MassTable', np.zeros(6))),
            'NumPart_Total': np.array(hd.get('NumPart_Total', np.zeros(6, dtype=np.uint64))),
            'NumPart_Total_HighWord': np.array(hd.get('NumPart_Total_HighWord', np.zeros(6, dtype=np.uint32))),
            'Redshift': float(hd.get('Redshift', np.nan)),
            'Time': float(hd.get('Time', np.nan)),
            'BoxSize': float(hd.get('BoxSize', np.nan)),
            'HubbleParam': float(params.get('HubbleParam', np.nan)),
            'UnitMass_in_g': float(params.get('UnitMass_in_g', np.nan)),
        }
    return header

def detect_present_types(files, header):
    present = set()
    n_hilo = header.get('NumPart_Total', np.zeros(6)) + (header.get('NumPart_Total_HighWord', np.zeros(6)) << 32)
    for ptype in range(7):
        if ptype < len(n_hilo) and int(n_hilo[ptype]) > 0:
            present.add(ptype)
    for fn in files:
        with h5py.File(fn, 'r') as f:
            for ptype in range(7):
                if f.get(f'PartType{ptype}') is not None:
                    present.add(ptype)
    return sorted(list(present))

def load_per_particle_masses(files, ptype, max_read=None):
    masses = []
    total_len = 0
    for fn in files:
        with h5py.File(fn, 'r') as f:
            grp = f.get(f'PartType{ptype}')
            if grp is None:
                continue
            if 'Masses' in grp:
                arr = np.asarray(grp['Masses'])
                total_len += arr.size
                if arr.size == 0:
                    continue
                if max_read is not None and arr.size > max_read:
                    rng = np.random.RandomState(123)
                    idx = rng.choice(arr.size, size=max_read, replace=False)
                    arr = arr[idx]
                masses.append(arr)
    masses = np.concatenate(masses) if masses else np.array([], dtype=np.float64)
    return masses, total_len

def summarize_collisionless_buckets(mvals, unit_label):
    rounded = np.round(mvals, 6)
    from collections import Counter
    counts = Counter(rounded)
    tot = sum(counts.values())
    items = sorted(counts.items(), key=lambda kv: kv[0])
    lines = []
    for val, c in items:
        frac = 100.0 * c / tot if tot > 0 else 0.0
        lines.append(f"  - {fmt_mass(val, unit_label)}  (count={c}, {frac:.2f}%)")
    return "\n".join(lines) if lines else "  (no buckets)"

def report_parttype(files, ptype, header, to_msun, max_read=None):
    names = {0:'Gas',1:'DM',2:'DM2',3:'Tracers',4:'Stars',5:'Bndry',6:'Dust'}
    unit_label = "M_solar"
    masses, total_len_seen = load_per_particle_masses(files, ptype, max_read=max_read)
    mt = header['MassTable'][ptype] if 'MassTable' in header and len(header['MassTable'])>ptype else 0.0
    if ptype < 6:
        n_total = int(header.get('NumPart_Total', np.zeros(6))[ptype]) + (int(header.get('NumPart_Total_HighWord', np.zeros(6, dtype=np.uint32))[ptype]) << 32)
    else:
        n_total = total_len_seen

    if masses.size == 0:
        if mt > 0:
            m = mt * to_msun
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N={n_total}\n  uniform mass = {fmt_mass(m, unit_label)}\n")
        else:
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N={n_total}\n  masses not stored per-particle and MassTable=0\n")
        return

    mvals = masses * to_msun

    if ptype in (1,2,3,5,6):
        if max_read and total_len_seen>max_read:
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N≈{total_len_seen} (sampled)")
        else:
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N={total_len_seen}")
        print("  mass buckets (ALL present):")
        print(summarize_collisionless_buckets(mvals, unit_label))
        print("")
    else:
        p = np.percentile(mvals, [0,5,50,95,100])
        if max_read and total_len_seen>max_read:
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N≈{total_len_seen} (sampled)")
        else:
            print(f"PartType{ptype} ({names.get(ptype,'?')}): N={total_len_seen}")
        print(f"  mass distribution ({unit_label}):")
        print("  - min / 5% / median / 95% / max = " + " / ".join(fmt_mass(x, unit_label) for x in p))
        print("")

def main():
    ap = argparse.ArgumentParser(description="Print mass resolution per particle type from a Gadget-4 HDF5 snapshot (supports multi-part).")
    ap.add_argument("snap", help="Path to snapshot base or HDF5 file. Examples: snapshot_050.hdf5  OR  snapshot_050  OR  ./output/snapdir_050/snapshot_050")
    ap.add_argument("--max-read", type=int, default=400000, help="Max per-type particle masses to read (subsample if larger). Default: 400k")
    args = ap.parse_args()

    base = args.snap
    files = []
    if os.path.isfile(base) and base.endswith(".hdf5"):
        files = [base]
    else:
        candidates = []
        if os.path.isdir(base):
            candidates += sorted(glob.glob(os.path.join(base, "*.hdf5")))
        candidates += sorted(glob.glob(base + ".*.hdf5"))
        candidates += sorted(glob.glob(base + "_*.hdf5"))
        candidates += sorted(glob.glob(base + ".hdf5"))
        files = [fn for fn in candidates if os.path.isfile(fn)]
        files = sorted(list(dict.fromkeys(files)))
    if not files:
        print("No HDF5 snapshot files found for:", base, file=sys.stderr)
        sys.exit(2)

    header = read_header(files)
    a = header.get("Time", np.nan)
    z = (1.0/a - 1.0) if a and np.isfinite(a) and a>0 else np.nan

    # Unit conversion factor: code mass -> Msun
    if not np.isfinite(header.get("UnitMass_in_g", np.nan)) or header["UnitMass_in_g"] <= 0:
        print("# ERROR: UnitMass_in_g missing in header; cannot convert to solar masses.", file=sys.stderr)
        sys.exit(3)
    to_msun = header["UnitMass_in_g"] / MSUN_IN_G

    print(f"# Files: {len(files)}")
    try:
        print(f"# a = {a:.8g}   z = {z:.4f}   BoxSize = {header.get('BoxSize', np.nan)}")
    except Exception:
        print(f"# a = {a}   z = {z}   BoxSize = {header.get('BoxSize', np.nan)}")
    print(f"# H0/h = {header.get('HubbleParam', np.nan)}")
    print("# NOTE: All masses below are converted to solar masses (M_solar).")
    print("")

    present = detect_present_types(files, header)
    if not present:
        print("No PartType groups detected.", file=sys.stderr)
        sys.exit(4)

    for ptype in present:
        report_parttype(files, ptype, header, to_msun, max_read=args.max_read)

if __name__ == "__main__":
    main()
