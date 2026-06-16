#!/usr/bin/env python3
"""
parse_obs_data.py
-----------------
Parses downloaded VizieR/DustPedia files into clean numpy arrays and saves
obs_data/obs_dustmass.npz for use in plot_mdust_mstar.py.

Usage:
    python parse_obs_data.py obs_data/

Datasets:
  Galliano+2021   J/A+A/649/A18   tableh1.dat   798 galaxies
  Remy-Ruyer+2015 J/A+A/582/A121  table4.dat (Mstar) + table9.dat (Mdust)
  DustPedia CIGALE dustpedia_cigale_results.csv
"""

import sys, os
import numpy as np


def _read_col(path, b0, b1):
    """Read one fixed-width column (0-indexed b0:b1), return float array."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            raw = line[b0:b1].strip() if b1 <= len(line) else ""
            if not raw or raw.lstrip("<≥~").strip() == "-" or raw == "---":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(raw.lstrip("<≥~")))
                except ValueError:
                    vals.append(np.nan)
    return np.array(vals)


def _read_str_col(path, b0, b1):
    names = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            names.append(line[b0:b1].strip() if b1 <= len(line) else "")
    return names


def _parse_readme_table(readme_path, table_filename):
    """
    Extract {label: (b0, b1)} for a named table from a VizieR ReadMe.
    b0/b1 are 0-indexed Python slice endpoints.
    """
    cols = {}
    in_section = False
    with open(readme_path) as f:
        for line in f:
            if "Byte-by-byte" in line and table_filename in line:
                in_section = True
                cols = {}
                continue
            if not in_section:
                continue
            stripped = line.strip()
            if stripped.startswith("---") or stripped.startswith("==="):
                continue
            if stripped == "" and cols:
                break
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    bytespec = parts[0]
                    if "-" in bytespec:
                        b0s, b1s = bytespec.split("-")
                        b0, b1 = int(b0s) - 1, int(b1s)
                    else:
                        b0 = int(bytespec) - 1
                        b1 = b0 + 1
                    label = parts[3]
                    cols[label] = (b0, b1)
                except (ValueError, IndexError):
                    pass
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Galliano+2021
# ─────────────────────────────────────────────────────────────────────────────

def load_galliano2021(data_dir):
    dat    = os.path.join(data_dir, "galliano2021_tableh1.dat")
    readme = os.path.join(data_dir, "galliano2021_ReadMe")
    if not os.path.exists(dat):
        print(f"  [Galliano+2021] {dat} not found"); return None, None

    # Try ReadMe first
    mstar_bc = mdust_bc = None
    if os.path.exists(readme):
        cols = _parse_readme_table(readme, "tableh1")
        print(f"  [Galliano+2021] ReadMe columns: {list(cols.keys())}")
        mstar_bc = next((v for k, v in cols.items()
                         if "star" in k.lower() or k.lower() in ("logmstar","logm*")), None)
        mdust_bc = next((v for k, v in cols.items()
                         if "dust" in k.lower() or k.lower() in ("logmdust","logmd")), None)

    # Galliano+2021 tableh1.dat is whitespace-delimited (scientific notation).
    # Known column order from the paper (Table H.1):
    #   0=Name, 1=logMdust, 2=e_logMdust, 3=logU, 4=e_logU,
    #   5=logqAF, 6=e_logqAF, 7=logMgas, 8=e_logMgas,
    #   9=logfmol, 10=e_logfmol, 11=logMstar, 12=e_logMstar, ...
    # If the ReadMe parse succeeded, use those; otherwise fall back to indices.
    MSTAR_IDX = 11   # 0-indexed after splitting (position after Name)
    MDUST_IDX = 1

    # Try to refine indices from the ReadMe column order if available
    if mstar_bc or mdust_bc:
        pass   # ReadMe parse partial — use hardcoded indices below

    logMs_list, logMd_list = [], []
    with open(dat) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                ms = float(parts[MSTAR_IDX])
                md = float(parts[MDUST_IDX])
                logMs_list.append(ms)
                logMd_list.append(md)
            except (ValueError, IndexError):
                pass
    logMs = np.array(logMs_list)
    logMd = np.array(logMd_list)
    mask = np.isfinite(logMs) & np.isfinite(logMd) & (logMs > 4) & (logMd > 1)
    logMs, logMd = logMs[mask], logMd[mask]
    print(f"  [Galliano+2021] {len(logMs)} galaxies  "
          f"logMs {logMs.min():.1f}–{logMs.max():.1f}  "
          f"logMd {logMd.min():.1f}–{logMd.max():.1f}")
    return logMs, logMd


# ─────────────────────────────────────────────────────────────────────────────
# Remy-Ruyer+2015
# Confirmed from VizieR ReadMe (J/A+A/582/A121):
#   table4.dat: Name bytes 10-20 (0-idx 9:20), logM* bytes 25-29 (0-idx 24:29)
#   table9.dat: columns parsed from ReadMe
# ─────────────────────────────────────────────────────────────────────────────

def load_remyruyer2015(data_dir):
    t4     = os.path.join(data_dir, "remyruyer2015_table4.dat")
    t9     = os.path.join(data_dir, "remyruyer2015_table9.dat")
    readme = os.path.join(data_dir, "remyruyer2015_ReadMe")

    if not os.path.exists(t4) or not os.path.exists(t9):
        print("  [Remy-Ruyer+2015] table4.dat or table9.dat missing"); return None, None

    # table4: confirmed byte positions
    names4 = _read_str_col(t4, 9, 20)
    logMs4 = _read_col(t4, 24, 29)          # logM*  bytes 25-29
    print(f"  [Remy-Ruyer+2015] table4: {len(names4)} rows  "
          f"logMs {np.nanmin(logMs4):.1f}–{np.nanmax(logMs4):.1f}")

    # table9: parse from ReadMe
    t9_cols = {}
    if os.path.exists(readme):
        t9_cols = _parse_readme_table(readme, "table9")
        print(f"  [Remy-Ruyer+2015] table9 columns: {list(t9_cols.keys())}")

    name9_bc = t9_cols.get("Name", None)
    # Look for the main dust mass column — skip note (n_) and error (e_) columns
    # and flag columns that are single-character or start with l_/n_/e_
    mdust9_bc = next((v for k, v in t9_cols.items()
                      if ("dust" in k.lower() or k.lower().startswith("logmd"))
                      and "gas" not in k.lower()
                      and "pah" not in k.lower()
                      and not k.startswith(("n_", "e_", "l_", "u_", "f_"))
                      and (v[1] - v[0]) >= 4), None)   # must be ≥4 bytes wide

    if not mdust9_bc:
        # Hardcode from direct inspection of the data file.
        # table9.dat layout (confirmed from printed lines):
        #   Sample  bytes  1-8  (0-idx  0:8)
        #   Name    bytes 10-20 (0-idx  9:20)
        #   logMdustGr bytes 23-27 (0-idx 22:27)  — graphite model (standard)
        #   logMdustAC bytes 29-33 (0-idx 28:33)  — amorphous carbon model
        # We use logMdustGr (graphite) as the primary dust mass,
        # consistent with Remy-Ruyer+2015 Table 9 column 3.
        print("  [Remy-Ruyer+2015] Using hardcoded byte positions for logMdustGr (22:27)")
        mdust9_bc = (22, 28)
        name9_bc  = (9, 20)    # same as table4

    print(f"  [Remy-Ruyer+2015] logMdust at bytes {mdust9_bc}")
    logMd9 = _read_col(t9, *mdust9_bc)

    # Match table4 and table9 by galaxy name
    if name9_bc:
        names9 = _read_str_col(t9, *name9_bc)
        name_to_ms = {n: ms for n, ms in zip(names4, logMs4)}
        logMs_out, logMd_out = [], []
        for name, md in zip(names9, logMd9):
            logMs_out.append(name_to_ms.get(name, np.nan))
            logMd_out.append(md)
        logMs = np.array(logMs_out)
        logMd = np.array(logMd_out)
    else:
        logMs = logMs4   # assume same row order
        logMd = logMd9

    mask = np.isfinite(logMs) & np.isfinite(logMd) & (logMs > 4) & (logMd > 1)
    logMs, logMd = logMs[mask], logMd[mask]
    if len(logMs) == 0:
        print("  [Remy-Ruyer+2015] No valid rows after matching — check name alignment")
        # Diagnostic: print first few names from each table
        print(f"    table4 names[:5]: {names4[:5]}")
        if name9_bc:
            names9_peek = _read_str_col(t9, *name9_bc)
            print(f"    table9 names[:5]: {names9_peek[:5]}")
        return None, None
    print(f"  [Remy-Ruyer+2015] {len(logMs)} galaxies after match  "
          f"logMs {logMs.min():.1f}–{logMs.max():.1f}  "
          f"logMd {logMd.min():.1f}–{logMd.max():.1f}")
    return logMs, logMd


# ─────────────────────────────────────────────────────────────────────────────
# DustPedia CIGALE CSV (Nersesian+2019)
# ─────────────────────────────────────────────────────────────────────────────

def load_dustpedia_cigale(data_dir):
    csv_path = os.path.join(data_dir, "dustpedia_cigale_results.csv")
    if not os.path.exists(csv_path):
        print("  [DustPedia CIGALE] not found — skipping"); return None, None

    import csv
    mstar_vals, mdust_vals = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # Actual column names in this CSV are 'Mstar__Msol' and 'Mdust__Msol'
        mstar_col = next((h for h in header
                          if h.lower().startswith("mstar")
                          or ("stellar" in h.lower() and "m_star" in h.lower())), None)
        mdust_col = next((h for h in header
                          if h.lower().startswith("mdust")
                          or ("dust" in h.lower() and "mass" in h.lower()
                              and "err" not in h.lower())), None)
        if not mstar_col or not mdust_col:
            print(f"  [DustPedia CIGALE] columns not found. Header: {header}"); return None, None
        print(f"  [DustPedia CIGALE] using '{mstar_col}' and '{mdust_col}'")
        for row in reader:
            try:
                ms = float(row[mstar_col]); md = float(row[mdust_col])
                if ms > 0 and md > 0:
                    mstar_vals.append(np.log10(ms)); mdust_vals.append(np.log10(md))
            except (ValueError, KeyError):
                pass

    logMs, logMd = np.array(mstar_vals), np.array(mdust_vals)
    mask = np.isfinite(logMs) & np.isfinite(logMd) & (logMs > 4) & (logMd > 1)
    logMs, logMd = logMs[mask], logMd[mask]
    print(f"  [DustPedia CIGALE] {len(logMs)} galaxies  "
          f"logMs {logMs.min():.1f}–{logMs.max():.1f}  "
          f"logMd {logMd.min():.1f}–{logMd.max():.1f}")
    return logMs, logMd


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "obs_data"
    print("=" * 60)
    print(f"Loading datasets from: {data_dir}")
    print("=" * 60)

    datasets = {}
    for name, fn in [("galliano2021", load_galliano2021),
                     ("remyruyer2015", load_remyruyer2015),
                     ("dustpedia_cigale", load_dustpedia_cigale)]:
        print(f"\n--- {name} ---")
        r = fn(data_dir)
        if r[0] is not None:
            datasets[name] = r

    if not datasets:
        print("\nNo datasets loaded. Run:  bash download_obs_data.sh"); return

    out_npz = os.path.join(data_dir, "obs_dustmass.npz")
    save_dict = {}
    for key, (logMs, logMd) in datasets.items():
        save_dict[f"{key}_mstar"] = logMs
        save_dict[f"{key}_mdust"] = logMd
    np.savez(out_npz, **save_dict)

    print(f"\nSaved: {out_npz}")
    print("Keys:", list(save_dict.keys()))
    print("\nLoad in plot_mdust_mstar.py:")
    print("  obs = np.load('obs_data/obs_dustmass.npz')")
    for key in datasets:
        print(f"  {key}_mstar = obs['{key}_mstar']")
        print(f"  {key}_mdust = obs['{key}_mdust']")


if __name__ == "__main__":
    main()
