#!/usr/bin/env python3
import os
import re
import h5py
import argparse
from collections import defaultdict, namedtuple
from datetime import datetime, timezone

SINGLE_RE = re.compile(r"(.*?/)?(snapshot_(\d{3}))\.hdf5$")
MULTI_RE  = re.compile(r"(.*?/)?(snapdir_(\d{3}))/snapshot_\3\.\d+\.hdf5$")

Series = namedtuple("Series", ["key", "files"])

def is_backup_or_temp(fn: str) -> bool:
    name = os.path.basename(fn).lower()
    if name.startswith("bak-"): return True
    if name.endswith(".bak.hdf5"): return True
    if "bak_snapshot" in name: return True
    if name.startswith("tmp-") or name.endswith(".tmp.hdf5"): return True
    if ".partial." in name or name.endswith(".part") or ".old" in name: return True
    if "backup" in name and not name.startswith("snapshot_"): return True
    return False

def discover_groups(root):
    """Return mapping: snap_index -> [Series(...)]"""
    by_key = defaultdict(list)
    idx_of_key = {}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".hdf5"):
                continue
            if is_backup_or_temp(fn):
                continue

            full = os.path.join(dirpath, fn)
            m_multi = MULTI_RE.search(full)
            m_single = SINGLE_RE.search(full)

            if m_multi:
                base = os.path.join(m_multi.group(1) or "", m_multi.group(2))
                idx = int(m_multi.group(3))
            elif m_single:
                base = os.path.join(m_single.group(1) or "", m_single.group(2))
                idx = int(m_single.group(3))
            else:
                continue

            by_key[base].append(full)
            idx_of_key[base] = idx

    by_index = defaultdict(list)
    for key, files in by_key.items():
        by_index[idx_of_key[key]].append(Series(key=key, files=sorted(files)))

    return by_index

def pick_newest_series(by_index):
    """Pick the newest mtime series for each snapshot index."""
    chosen = {}
    for idx, series_list in by_index.items():
        best = max(series_list, key=lambda s: max(os.path.getmtime(f) for f in s.files))
        chosen[idx] = best
    return chosen  # dict idx -> Series

def age_of_universe_gyr(z, Om, OL, h):
    import math
    H0 = (100.0 * h) / 3.085678e19
    a_now = 1.0 / (1.0 + max(z, 0.0))
    N = 2000
    la0, la1 = math.log(1e-8), math.log(a_now)
    acc = 0.0
    for i in range(N):
        a = math.exp(la0 + (i + 0.5) * (la1 - la0) / N)
        acc += 1.0 / math.sqrt(Om / a**3 + OL)
    t = acc * (la1 - la0) / N / H0
    return t / (3600 * 24 * 365.25 * 1e9)

def read_header_counts_cosmo(files):
    """
    Return (z, age_gyr, counts[0..6], newest_mtime, (Om,OL,h), unit_mass_in_g)
    counts are total NumPart_ThisFile summed across pieces.
    """
    counts = [0]*7
    z = None
    cosmo = None
    unit_mass_in_g = None
    newest_mtime = max(os.path.getmtime(f) for f in files)

    for f in files:
        with h5py.File(f, "r") as h:
            hdr = h["Header"].attrs

            if z is None:
                z = float(hdr.get("Redshift", 0.0))

            if cosmo is None:
                cosmo = (
                    float(hdr.get("Omega0", 0.3)),
                    float(hdr.get("OmegaLambda", 0.7)),
                    float(hdr.get("HubbleParam", 0.7)),
                )

            if unit_mass_in_g is None:
                # Gadget headers often store this as Header attrs; if missing, user knows it externally
                unit_mass_in_g = hdr.get("UnitMass_in_g", None)

            np_this = hdr.get("NumPart_ThisFile", None)
            if np_this is not None:
                for i in range(min(len(np_this), 7)):
                    counts[i] += int(np_this[i])
            else:
                for i in range(7):
                    g = f"PartType{i}"
                    if g in h and "Coordinates" in h[g]:
                        counts[i] += h[g]["Coordinates"].shape[0]

    if z is None:
        z = 0.0
    if cosmo is None:
        cosmo = (0.3, 0.7, 0.7)

    age = age_of_universe_gyr(z, *cosmo)
    return z, age, counts, newest_mtime, cosmo, unit_mass_in_g

def sum_ptype_mass(files, ptype):
    """
    Sum mass in code units for PartType{ptype}.
    Priority:
      - sum PartTypeX/Masses if present
      - else MassTable[ptype] * N if MassTable nonzero
      - else 0
    """
    import numpy as np
    total = 0.0
    for f in files:
        with h5py.File(f, "r") as h:
            gname = f"PartType{ptype}"
            if gname not in h:
                continue
            g = h[gname]

            if "Masses" in g:
                total += float(np.sum(g["Masses"][:]))
                continue

            hdr = h["Header"].attrs
            mt = hdr.get("MassTable", None)
            if mt is None or ptype >= len(mt):
                continue
            mconst = float(mt[ptype])
            if mconst <= 0:
                continue

            if "Coordinates" in g:
                n = g["Coordinates"].shape[0]
            else:
                ds = next(iter(g.values()))
                n = ds.shape[0]
            total += n * mconst
    return total

def fmt_int(n): return f"{n:,}"
def fmt_sci(x): return f"{x:.3e}"

def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(r):
        out = []
        for i, (c, w) in enumerate(zip(r, widths)):
            # left align first col, right align the rest
            out.append(c.ljust(w) if i == 0 else c.rjust(w))
        return "  ".join(out)

    line = "-" * (sum(widths) + 2*(len(widths)-1))
    return "\n".join([fmt(headers), line] + [fmt(r) for r in rows])

def main():
    ap = argparse.ArgumentParser(description="Compare dust vs no-dust Gadget snapshots by index.")
    ap.add_argument("dust_dir", help="Output dir for dust run")
    ap.add_argument("nodust_dir", help="Output dir for no-dust run")
    ap.add_argument("--tz", choices=["local", "utc"], default="local")
    ap.add_argument("--units", choices=["code", "Msun"], default="Msun",
                    help="Display masses in code units or Msun (default: Msun).")
    ap.add_argument("--max", type=int, default=None, help="Max rows to print (from earliest).")
    args = ap.parse_args()

    dust = pick_newest_series(discover_groups(args.dust_dir))
    nod  = pick_newest_series(discover_groups(args.nodust_dir))

    common = sorted(set(dust.keys()) & set(nod.keys()))
    if not common:
        print("No overlapping snapshot indices found between the two directories.")
        return

    headers = [
        "snap",
        "z_dust", "z_nodust",
        "Age_d(Gyr)", "Age_n(Gyr)",
        "Nstar_d", "Nstar_n",
        "M*_dust", "M*_nodust",
        "ratio", "dM*",
        "Ngas_d", "Ngas_n",
        "Mdust",  # will be 0 for nodust
        "LastMod_d", "LastMod_n"
    ]

    rows = []
    for idx in common[:args.max] if args.max else common:
        sd = dust[idx]
        sn = nod[idx]

        zd, aged, cd, mtd, cosd, umd = read_header_counts_cosmo(sd.files)
        zn, agen, cn, mtn, cosn, umn = read_header_counts_cosmo(sn.files)

        # masses in code units
        mstar_d_code = sum_ptype_mass(sd.files, 4)
        mstar_n_code = sum_ptype_mass(sn.files, 4)

        mdust_code   = sum_ptype_mass(sd.files, 6)  # will be 0 if absent

        # convert to Msun if requested
        # If UnitMass_in_g missing from file attrs, fall back to your known 1e10 Msun per code mass.
        CODE_TO_MSUN = 1.0e10

        if args.units == "Msun":
            mstar_d = mstar_d_code * CODE_TO_MSUN
            mstar_n = mstar_n_code * CODE_TO_MSUN
            mdust   = mdust_code   * CODE_TO_MSUN
        else:
            mstar_d = mstar_d_code
            mstar_n = mstar_n_code
            mdust   = mdust_code

        ratio = (mstar_d / mstar_n) if mstar_n > 0 else float("nan")
        dm    = (mstar_d - mstar_n)

        # timestamps
        dtd = datetime.fromtimestamp(mtd, tz=timezone.utc)
        dtn = datetime.fromtimestamp(mtn, tz=timezone.utc)
        if args.tz == "local":
            dtd = dtd.astimezone()
            dtn = dtn.astimezone()

        rows.append([
            f"{idx:03d}",
            f"{zd:.3f}", f"{zn:.3f}",
            f"{aged:.3f}", f"{agen:.3f}",
            fmt_int(cd[4]), fmt_int(cn[4]),
            fmt_sci(mstar_d), fmt_sci(mstar_n),
            f"{ratio:.3f}" if ratio == ratio else "nan",
            fmt_sci(dm),
            fmt_int(cd[0]), fmt_int(cn[0]),
            fmt_sci(mdust),
            dtd.strftime("%Y-%m-%d %H:%M:%S"),
            dtn.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    print(format_table(headers, rows))
    print()
    print("Notes:")
    print(f"  Mass units shown: {args.units} (conversion assumes 1 code mass = 1e10 Msun).")
    print("  Mdust is summed from PartType6 if present in the dust run.")
    print("  ratio = M*_dust / M*_nodust ; dM* = M*_dust - M*_nodust")

if __name__ == "__main__":
    main()

