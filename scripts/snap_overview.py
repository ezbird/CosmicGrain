#!/usr/bin/env python3
"""
snap_overview.py
----------------
Snapshot overview table for Gadget-4 zoom simulations.

Prints per-snapshot particle counts, stellar mass, dust mass, and halo
properties — all correctly converted from Gadget code units to physical
quantities.

Unit conventions (Gadget-4 defaults):
  - Positions:  comoving kpc/h  →  physical kpc  via  x_phys = x_code * a / h
  - Masses:     1e10 M_sun/h    →  M_sun          via  m_phys = m_code * 1e10 / h
  - R_200:      comoving kpc/h  →  physical kpc  via  r_phys = r_code * a / h
  - HubbleParam (h) is read from f["Parameters"].attrs, NOT f["Header"].attrs

Primary halo identification (zoom simulation safe):
  Reads ALL catalog chunks and selects the FOF group with the highest
  STELLAR mass (argmax GroupMassType[:,4]). This correctly identifies
  the zoom target galaxy rather than the most massive dark matter halo,
  which can be a star-poor group-scale object.
  Falls back to argmax(M200) at early epochs before stars form.

Sanity checks:
  R200_JUMP : >30% consecutive change, suppressed below --r200-min pkpc
  REGRESS   : R200 < --regress-frac of historical maximum

Usage:
    python snap_overview.py /path/to/output/ [options]
"""

import os
import re
import sys
import math
import argparse
import numpy as np
import h5py
from collections import defaultdict, namedtuple
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MSUN_PER_CODE  = 1e10
R200_JUMP_WARN = 0.30
R200_MIN_WARN  = 50.0
REGRESS_FRAC   = 0.50

PartTypeLabels = {
    0: "P0(gas)", 1: "P1(dm)", 2: "P2", 3: "P3",
    4: "P4(stars)", 5: "P5(bh)", 6: "P6(dust)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_RE = re.compile(r"(.*?/)?(snapshot_(\d{3}))\.hdf5$")
MULTI_RE  = re.compile(r"(.*?/)?(snapdir_(\d{3}))/snapshot_\3\.\d+\.hdf5$")

def is_backup_or_temp(fn):
    name = os.path.basename(fn).lower()
    if name.startswith(("bak-", "tmp-")):
        return True
    if name.endswith((".bak.hdf5", ".tmp.hdf5")):
        return True
    if any(x in name for x in ("bak_snapshot", ".partial.", ".old", "backup")):
        if not name.startswith("snapshot_"):
            return True
    return False

Series = namedtuple("Series", ["key", "files"])

def discover_groups(root):
    by_key = defaultdict(list)
    idx_of = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".hdf5") or is_backup_or_temp(fn):
                continue
            full = os.path.join(dirpath, fn)
            m = MULTI_RE.search(full) or SINGLE_RE.search(full)
            if not m:
                continue
            base = os.path.join(m.group(1) or "", m.group(2))
            idx  = int(m.group(3))
            by_key[base].append(full)
            idx_of[base] = idx

    by_index = defaultdict(list)
    for key, files in by_key.items():
        by_index[idx_of[key]].append(Series(key=key, files=sorted(files)))
    return by_index

def pick_newest_series(by_index):
    chosen = []
    for idx, slist in by_index.items():
        best = max(slist, key=lambda s: max(os.path.getmtime(f) for f in s.files))
        chosen.append((idx, best))
    return sorted(chosen, key=lambda x: x[0])

# ─────────────────────────────────────────────────────────────────────────────
# Header reading
# ─────────────────────────────────────────────────────────────────────────────

def read_header(files):
    counts   = {i: 0 for i in range(7)}
    redshift = a = h = Om = OL = box_code = None
    mtime    = max(os.path.getmtime(f) for f in files)

    for fname in files:
        with h5py.File(fname, "r") as f:
            hdr = f["Header"].attrs
            if redshift is None:
                redshift = float(hdr["Redshift"])
                a        = float(hdr["Time"])
                Om       = float(hdr.get("Omega0",      0.3))
                OL       = float(hdr.get("OmegaLambda", 0.7))
                box_code = float(hdr.get("BoxSize",      0.0))
            if h is None and "Parameters" in f:
                h = float(f["Parameters"].attrs["HubbleParam"])

            npt = hdr.get("NumPart_ThisFile")
            if npt is not None:
                for i in range(min(len(npt), 7)):
                    counts[i] += int(npt[i])
            else:
                for i in range(7):
                    g = f"PartType{i}"
                    if g in f and "Coordinates" in f[g]:
                        counts[i] += f[g]["Coordinates"].shape[0]

    if h is None:
        h = 0.6774
        print("WARNING: HubbleParam not found in Parameters — using 0.6774")

    box_phys_kpc = box_code * a / h if box_code else 0.0
    return redshift, a, h, Om, OL, box_phys_kpc, counts, mtime

# ─────────────────────────────────────────────────────────────────────────────
# Primary halo identification — stellar mass argmax across all chunks
# ─────────────────────────────────────────────────────────────────────────────

def find_catalog(snap_file, output_dir):
    m = re.search(r"snapshot[_-](\d{3})", snap_file)
    if not m:
        return None
    num        = m.group(1)
    groups_dir = Path(output_dir) / f"groups_{num}"
    if groups_dir.exists():
        cats = sorted(groups_dir.glob(f"fof_subhalo_tab_{num}*.hdf5"))
        if cats:
            return str(cats[0])
    return None


def get_halo_info(catalog_file):
    """
    Identify the primary halo as the group with the highest STELLAR mass
    (argmax GroupMassType[:,4]) across ALL catalog chunks.

    Using stellar mass rather than M200 correctly targets the zoom galaxy
    rather than potentially more massive but star-poor neighbouring halos.

    Falls back to argmax(Group_M_Crit200) at early epochs when no stars
    have formed yet (all stellar masses zero).

    Gadget-4 catalog units:
      GroupPos        : comoving kpc/h  →  physical kpc  (* a/h)
      Group_R_Crit200 : comoving kpc/h  →  physical kpc  (* a/h)
      Group_M_Crit200 : 1e10 M_sun/h   →  M_sun          (* 1e10/h)
      GroupMassType   : 1e10 M_sun/h   →  M_sun          (* 1e10/h)

    Returns dict: center_phys, r200_phys, m200_msun, mstar_msun,
                  a, h, group_idx, selection_by
    or None if catalog has no usable groups.
    """
    p         = Path(catalog_file)
    stem_base = re.sub(r"\.\d+$", "", p.stem)
    chunks    = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    all_pos   = []
    all_r200  = []
    all_m200  = []
    all_mstar = []
    a = h = None

    for chunk in chunks:
        with h5py.File(str(chunk), "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or "Group_M_Crit200" not in grp:
                continue
            if len(grp["GroupPos"]) == 0:
                continue
            if a is None:
                a = float(f["Header"].attrs["Time"])
                h = float(f["Parameters"].attrs["HubbleParam"])
            all_pos.append(grp["GroupPos"][:])
            all_r200.append(grp["Group_R_Crit200"][:])
            all_m200.append(grp["Group_M_Crit200"][:])
            if "GroupMassType" in grp:
                all_mstar.append(grp["GroupMassType"][:, 4])
            else:
                all_mstar.append(np.zeros(len(grp["GroupPos"])))

    if not all_m200 or a is None:
        return None

    pos_all   = np.concatenate(all_pos,   axis=0)
    r200_all  = np.concatenate(all_r200,  axis=0)
    m200_all  = np.concatenate(all_m200,  axis=0)
    mstar_all = np.concatenate(all_mstar, axis=0)

    # Select by stellar mass; fall back to M200 if no stars yet
    if mstar_all.max() > 0:
        idx          = int(np.argmax(mstar_all))
        selection_by = "M*"
    else:
        idx          = int(np.argmax(m200_all))
        selection_by = "M200"

    center_phys = pos_all[idx]         * a / h
    r200_phys   = float(r200_all[idx]) * a / h
    m200_msun   = float(m200_all[idx]) * MSUN_PER_CODE / h
    mstar_msun  = float(mstar_all[idx])* MSUN_PER_CODE / h

    return dict(
        center_phys  = center_phys,
        r200_phys    = r200_phys,
        m200_msun    = m200_msun,
        mstar_msun   = mstar_msun,
        a            = a,
        h            = h,
        group_idx    = idx,
        selection_by = selection_by,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Mass computation within R_200
# ─────────────────────────────────────────────────────────────────────────────

def masses_within_r200(snap_files, center_phys, r200_phys, a, h, box_phys_kpc):
    m_star = 0.0
    m_dust = 0.0
    for fname in snap_files:
        with h5py.File(fname, "r") as f:
            for pt, label in [(4, "star"), (6, "dust")]:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                g = f[key]
                if "Coordinates" not in g or "Masses" not in g:
                    continue
                pos  = g["Coordinates"][:] * a / h
                mass = g["Masses"][:]
                dx   = pos - center_phys
                if box_phys_kpc > 0:
                    dx -= box_phys_kpc * np.round(dx / box_phys_kpc)
                r    = np.sqrt((dx**2).sum(axis=1))
                m_msun = float(mass[r <= r200_phys].sum()) * MSUN_PER_CODE / h
                if label == "star":
                    m_star += m_msun
                else:
                    m_dust += m_msun
    return m_star, m_dust


def masses_global(snap_files, h):
    m_star = 0.0
    m_dust = 0.0
    for fname in snap_files:
        with h5py.File(fname, "r") as f:
            for pt, label in [(4, "star"), (6, "dust")]:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                g = f[key]
                if "Masses" not in g:
                    mt = f["Header"].attrs.get("MassTable")
                    if mt is not None and len(mt) > pt and mt[pt] > 0 \
                            and "Coordinates" in g:
                        m_code = float(mt[pt]) * g["Coordinates"].shape[0]
                    else:
                        continue
                else:
                    m_code = float(g["Masses"][:].sum())
                m_msun = m_code * MSUN_PER_CODE / h
                if label == "star":
                    m_star += m_msun
                else:
                    m_dust += m_msun
    return m_star, m_dust

# ─────────────────────────────────────────────────────────────────────────────
# Age of universe
# ─────────────────────────────────────────────────────────────────────────────

def age_of_universe_gyr(z, Om, OL, h):
    H0    = (100.0 * h) / 3.085678e19
    a_now = 1.0 / (1.0 + max(z, 0.0))
    N     = 2000
    la0, la1 = math.log(1e-8), math.log(a_now)
    acc = sum(
        1.0 / math.sqrt(Om / math.exp(la0 + (i+0.5)*(la1-la0)/N)**3 + OL)
        for i in range(N)
    )
    return acc * (la1 - la0) / N / H0 / (3600 * 24 * 365.25 * 1e9)

# ─────────────────────────────────────────────────────────────────────────────
# Table formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def fmt(r):
        out = []
        for i, (c, w) in enumerate(zip(r, widths)):
            out.append(str(c).ljust(w) if i == 0 else str(c).rjust(w))
        return "  ".join(out)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([fmt(headers), line] + [fmt(r) for r in rows])

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gadget-4 snapshot overview table")
    ap.add_argument("output_dir")
    ap.add_argument("--tz",           choices=["local", "utc"], default="local")
    ap.add_argument("--no-r200",      action="store_true",
                    help="Skip R200 aperture; sum all particles")
    ap.add_argument("--r200-warn",    type=float, default=R200_JUMP_WARN,
                    help=f"Fractional R200 change for R200_JUMP (default {R200_JUMP_WARN})")
    ap.add_argument("--r200-min",     type=float, default=R200_MIN_WARN,
                    help=f"Min R200 pkpc below which R200_JUMP suppressed (default {R200_MIN_WARN})")
    ap.add_argument("--regress-frac", type=float, default=REGRESS_FRAC,
                    help=f"Fraction of max R200 below which REGRESS fires (default {REGRESS_FRAC})")
    args = ap.parse_args()

    by_index = discover_groups(args.output_dir)
    if not by_index:
        print("No snapshots found.")
        sys.exit(1)

    headers = (["Snapshot", "z", "Age(Gyr)"]
               + [PartTypeLabels[i] for i in range(7)]
               + ["M*(Msun)[log]", "Mdust(Msun)[log]",
                  "R200(pkpc)", "M200(Msun)", "GrpIdx", "SelBy",
                  "Flags", "LastModified"])

    rows      = []
    warnings  = []
    prev_r200 = None
    max_r200  = 0.0

    for idx, series in pick_newest_series(by_index):
        z, a, h, Om, OL, box_phys, counts, mtime = read_header(series.files)
        age   = age_of_universe_gyr(z, Om, OL, h)
        snap0 = series.files[0]
        label = os.path.basename(series.key).replace("snapdir_", "snap_")

        cat_file = find_catalog(snap0, args.output_dir)
        halo     = get_halo_info(cat_file) if cat_file else None

        r200_str = "N/A"
        m200_str = "N/A"
        grp_str  = "N/A"
        sel_str  = "N/A"
        flags    = []

        if args.no_r200 or halo is None:
            m_star, m_dust = masses_global(series.files, h)
            flags.append("global" if args.no_r200 else "no-cat")
        else:
            r200_phys = halo["r200_phys"]
            m_star, m_dust = masses_within_r200(
                series.files, halo["center_phys"], r200_phys, a, h, box_phys)

            r200_str = f"{r200_phys:.1f}"
            m200_str = f"{halo['m200_msun']:.3e}"
            grp_str  = str(halo["group_idx"])
            sel_str  = halo["selection_by"]

            # R200_JUMP check
            if prev_r200 is not None and prev_r200 >= args.r200_min:
                frac = abs(r200_phys - prev_r200) / prev_r200
                if frac > args.r200_warn:
                    flags.append(f"R200_JUMP({frac*100:.0f}%)")
                    warnings.append(
                        f"  {label}: R200 {prev_r200:.1f}→{r200_phys:.1f} pkpc "
                        f"({frac*100:.0f}%), GrpIdx={halo['group_idx']} "
                        f"[sel by {halo['selection_by']}]"
                    )

            # REGRESS check
            if max_r200 >= args.r200_min and r200_phys < args.regress_frac * max_r200:
                flags.append(f"REGRESS(<{args.regress_frac*100:.0f}%max)")
                warnings.append(
                    f"  {label}: R200 {r200_phys:.1f} pkpc < "
                    f"{args.regress_frac*100:.0f}% of max {max_r200:.1f} pkpc "
                    f"(GrpIdx={halo['group_idx']}, sel by {halo['selection_by']})"
                )

            prev_r200 = r200_phys
            max_r200  = max(max_r200, r200_phys)

        def fmt_mass(m):
            if m <= 0:
                return "—"
            return f"{m:.3e} [{math.log10(m):.2f}]"

        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        if args.tz == "local":
            dt = dt.astimezone()
        ts = dt.strftime("%Y-%m-%d %H:%M")

        row = ([label, f"{z:.3f}", f"{age:.3f}"]
               + [f"{counts[i]:,}" for i in range(7)]
               + [fmt_mass(m_star), fmt_mass(m_dust),
                  r200_str, m200_str, grp_str, sel_str,
                  ",".join(flags) if flags else "ok",
                  ts])
        rows.append(row)

    print(format_table(headers, rows))

    if warnings:
        print()
        print("═" * 70)
        print(f"  {len(warnings)} WARNING(s):")
        print("═" * 70)
        for w in warnings:
            print(w)

    print()
    print("Unit notes:")
    print("  M*, Mdust : M_sun  (code units * 1e10 / h), log10 in brackets")
    print("  R200      : physical kpc  (comoving kpc/h * a / h)")
    print("  M200      : M_sun within R_200,crit of identified primary halo")
    print("  GrpIdx    : FOF group index selected as primary halo")
    print("  SelBy     : M* = selected by stellar mass (normal)")
    print("              M200 = fallback, no stars yet (early snapshots)")
    print("  h         : from f['Parameters'].attrs['HubbleParam']")
    print(f"  R200_JUMP : >={args.r200_warn*100:.0f}% change, "
          f"suppressed below {args.r200_min:.0f} pkpc")
    print(f"  REGRESS   : R200 < {args.regress_frac*100:.0f}% of historical max")

if __name__ == "__main__":
    main()
