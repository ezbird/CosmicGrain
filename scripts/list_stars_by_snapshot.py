#!/usr/bin/env python3
import os
import re
import h5py
import argparse
from collections import defaultdict, namedtuple
from datetime import datetime, timezone

SINGLE_RE = re.compile(r"(.*?/)?(snapshot_(\d{3}))\.hdf5$")
MULTI_RE  = re.compile(r"(.*?/)?(snapdir_(\d{3}))/snapshot_\3\.\d+\.hdf5$")

PartTypeLabels = {
    0: "P0(gas)",
    1: "P1(dm)",
    2: "P2",
    3: "P3",
    4: "P4(stars)",
    5: "P5",
    6: "P6(dust)",
}

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
    Series = namedtuple("Series", ["key", "files"])
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
    chosen = []
    for idx, series_list in by_index.items():
        best = max(series_list, key=lambda s: max(os.path.getmtime(f) for f in s.files))
        chosen.append((idx, best))
    return sorted(chosen, key=lambda x: x[0])

def read_header_and_counts(files):
    counts = {i: 0 for i in range(7)}
    redshift = None
    cosmo = None
    newest_mtime = max(os.path.getmtime(f) for f in files)

    for f in files:
        with h5py.File(f, "r") as h:
            hdr = h["Header"].attrs
            if redshift is None:
                redshift = float(hdr.get("Redshift", 0.0))

            if cosmo is None:
                cosmo = (
                    float(hdr.get("Omega0", 0.3)),
                    float(hdr.get("OmegaLambda", 0.7)),
                    float(hdr.get("HubbleParam", 0.7)),
                )

            np_this = hdr.get("NumPart_ThisFile")
            if np_this is not None:
                for i in range(min(len(np_this), 7)):
                    counts[i] += int(np_this[i])
            else:
                for i in range(7):
                    g = f"PartType{i}"
                    if g in h and "Coordinates" in h[g]:
                        counts[i] += h[g]["Coordinates"].shape[0]

    return (redshift or 0.0), counts, (cosmo or (0.3, 0.7, 0.7)), newest_mtime

def compute_stellar_mass(files):
    """
    Compute total stellar mass in code units.

    Priority:
      1) Sum PartType4/Masses if present
      2) Else use Header/MassTable[4] * Nstars
      3) Else 0
    """
    import numpy as np

    total = 0.0

    for f in files:
        with h5py.File(f, "r") as h:
            if "PartType4" not in h:
                continue
            g = h["PartType4"]

            # Best case: per-particle masses exist
            if "Masses" in g:
                m = g["Masses"][:]
                total += float(np.sum(m))
                continue

            # Fallback: constant mass from MassTable
            hdr = h["Header"].attrs
            mass_table = hdr.get("MassTable", None)
            if mass_table is None or len(mass_table) <= 4:
                continue

            mstar = float(mass_table[4])
            if mstar <= 0:
                continue

            if "Coordinates" in g:
                n = g["Coordinates"].shape[0]
            else:
                # last resort: count any dataset length
                ds = next(iter(g.values()))
                n = ds.shape[0]

            total += n * mstar

    return total

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

def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(r):
        # left-align SnapshotBase, right-align everything else
        out = []
        for i, (c, w) in enumerate(zip(r, widths)):
            out.append(c.ljust(w) if i == 0 else c.rjust(w))
        return "  ".join(out)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    out = [fmt(headers), line]
    out += [fmt(r) for r in rows]
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Snapshot overview table")
    ap.add_argument("output_dir", help="Path to output directory containing snapshots/snapdirs")
    ap.add_argument("--tz", choices=["local", "utc"], default="local",
                    help="Timestamp timezone (default: local)")
    ap.add_argument("--mass-unit", choices=["code", "1e10"], default="1e10",
                    help="Display M* in code units or divided by 1e10 (default: 1e10)")
    args = ap.parse_args()

    by_index = discover_groups(args.output_dir)
    if not by_index:
        print("No snapshots found.")
        return

    headers = ["SnapshotBase", "z", "Age(Gyr)"] + \
              [PartTypeLabels[i] for i in range(7)] + \
              ["M*", "LastModified"]

    rows = []

    for idx, series in pick_newest_series(by_index):
        z, counts, (Om, OL, hh), mtime = read_header_and_counts(series.files)
        age = age_of_universe_gyr(z, Om, OL, hh)

        mstar_code = compute_stellar_mass(series.files)

        # Convert code units -> Msun
        mstar_msun = mstar_code * 1.0e10
        mstar_str = f"{mstar_msun:.3e}"

        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        if args.tz == "local":
            dt = dt.astimezone()
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")

        label = os.path.basename(series.key).replace("snapdir_", "snapshot_")

        row = [label, f"{z:.3f}", f"{age:.3f}"]
        for i in range(7):
            row.append(f"{counts[i]:,}")
        row.append(mstar_str)
        row.append(ts)

        rows.append(row)

    print(format_table(headers, rows))

if __name__ == "__main__":
    main()

