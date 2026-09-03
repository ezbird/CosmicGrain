#!/usr/bin/env python3
"""
cosmicgrain_dust_radial_diagnostic.py

Radial dust diagnostic for the CosmicGrain zoom suite.

Uses halo_utils.py for:
- generic target-halo identification
- conservative center refinement
- particle-based SO M200c/R200c

Outputs:
1. Cumulative dust particle count and dust mass at fixed physical apertures
   and fractions of R200.
2. Radial shell dust mass/count profile.
3. Dust-source breakdown versus radius.
4. Median grain radius, carbon fraction, dust temperature, and formation time
   in each radial shell when those fields are available.
5. A compact CSV table for later comparison across the 12-halo suite.
6. An interactive Plotly HTML with cumulative dust mass/count and source
   breakdown.

Example
-------
python3 cosmicgrain_dust_radial_diagnostic.py \
    ../halo295_output_512 \
    --snap 27 \
    --out-prefix halo295_512_z0_dust_radial
"""

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from halo_utils import (
    find_snapshot_and_group_files,
    get_zoom_halo,
    periodic_delta,
)


def read_dust(snapshot_files):
    fields = {}
    chunks = {}

    wanted = [
        "Coordinates",
        "Masses",
        "ParticleIDs",
        "DustSource",
        "GrainRadius",
        "CarbonMassFraction",
        "DustFormationTime",
        "DustTemperature",
        "BirthPos",
    ]

    mass_table = None

    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            if mass_table is None:
                mass_table = np.asarray(f["Header"].attrs["MassTable"], dtype=float)

            if "PartType6" not in f:
                continue

            g = f["PartType6"]
            for field in wanted:
                if field not in g:
                    continue
                chunks.setdefault(field, []).append(g[field][()])

            # Ensure Masses exists even if stored in MassTable.
            if "Masses" not in g:
                n = len(g["Coordinates"])
                chunks.setdefault("Masses", []).append(
                    np.full(n, mass_table[6], dtype=np.float64)
                )

    for field, vals in chunks.items():
        try:
            fields[field] = np.concatenate(vals, axis=0)
        except ValueError:
            fields[field] = np.concatenate([np.atleast_1d(v) for v in vals], axis=0)

    if "Coordinates" not in fields:
        raise RuntimeError("No PartType6 dust particles found.")

    return fields


def source_labels(values):
    """
    Infer source labels conservatively.

    Common CosmicGrain convention is expected to be:
      0 = SNII
      1 = AGB
      2 = LRN

    Any unknown values are reported literally rather than silently remapped.
    """
    unique = np.unique(values)
    mapping = {0: "SNII", 1: "AGB", 2: "LRN"}
    return {v: mapping.get(int(v), f"Source{int(v)}") for v in unique}


def safe_median(x):
    x = np.asarray(x)
    q = np.isfinite(x)
    if not np.any(q):
        return np.nan
    return float(np.median(x[q]))


def formation_time_to_redshift(aform):
    aform = np.asarray(aform, dtype=float)
    out = np.full_like(aform, np.nan, dtype=float)
    q = np.isfinite(aform) & (aform > 0)
    out[q] = 1.0 / aform[q] - 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--group-index", type=int, default=None)
    ap.add_argument("--no-refine-center", action="store_true")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--rmax-r200", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="dust_radial")
    args = ap.parse_args()

    snap_files, _ = find_snapshot_and_group_files(args.output_dir, args.snap)

    halo = get_zoom_halo(
        args.output_dir,
        args.snap,
        group_index=args.group_index,
        refine_center=not args.no_refine_center,
        verbose=False,
    )

    dust = read_dust(snap_files)

    center = halo.chosen_center_ckpch
    box = halo.boxsize_ckpch
    a = halo.a
    h = halo.h
    r200 = halo.so_r200_ckpch
    r200_pkpc = r200 * a / h

    coords = np.asarray(dust["Coordinates"], dtype=float)
    masses_code = np.asarray(dust["Masses"], dtype=float)
    masses_msun = masses_code * 1e10 / h

    d = periodic_delta(coords, center, box)
    r_ckpch = np.linalg.norm(d, axis=1)
    r_pkpc = r_ckpch * a / h
    r_r200 = r_ckpch / r200

    print("=" * 88)
    print("COSMICGRAIN DUST RADIAL DIAGNOSTIC")
    print("=" * 88)
    print(f"Run                  : {args.output_dir}")
    print(f"Snapshot             : {args.snap:03d}")
    print(f"z                    : {halo.z:.6f}")
    print(f"M200c                : {halo.so_m200_code*1e10/h:.6e} Msun")
    print(f"R200c                : {r200_pkpc:.3f} pkpc")
    print(f"Center               : {center}")
    print(f"Total dust particles : {len(coords):,}")
    print(f"Total dust mass      : {masses_msun.sum():.6e} Msun")
    print()

    # ------------------------------------------------------------
    # Cumulative apertures
    # ------------------------------------------------------------
    aperture_specs = [
        ("10 pkpc", 10.0, "pkpc"),
        ("20 pkpc", 20.0, "pkpc"),
        ("30 pkpc", 30.0, "pkpc"),
        ("50 pkpc", 50.0, "pkpc"),
        ("0.25 R200", 0.25, "r200"),
        ("0.50 R200", 0.50, "r200"),
        ("1.00 R200", 1.00, "r200"),
    ]

    print("--- CUMULATIVE DUST ---")
    print(f"{'Aperture':>12s} {'N_dust':>10s} {'M_dust [Msun]':>18s} {'Mass frac':>12s}")
    cumulative_rows = []

    total_within_r200 = masses_msun[r_r200 <= 1.0].sum()

    for label, val, kind in aperture_specs:
        if kind == "pkpc":
            q = r_pkpc <= val
        else:
            q = r_r200 <= val

        n = int(np.count_nonzero(q))
        md = float(np.sum(masses_msun[q]))
        frac = md / total_within_r200 if total_within_r200 > 0 else np.nan
        cumulative_rows.append((label, n, md, frac))
        print(f"{label:>12s} {n:10d} {md:18.6e} {frac:12.6f}")

    # ------------------------------------------------------------
    # Dust source breakdown
    # ------------------------------------------------------------
    src = dust.get("DustSource")
    src_map = None
    if src is not None:
        src = np.asarray(src)
        if src.ndim > 1:
            src = np.ravel(src)
        src_map = source_labels(src)

        print("\n--- DUST SOURCE BREAKDOWN ---")
        print("Assumed label convention: 0=SNII, 1=AGB, 2=LRN; unknown values are literal.")
        for sval, slabel in src_map.items():
            qsrc = (src == sval)
            n_all = int(np.count_nonzero(qsrc))
            m_all = float(np.sum(masses_msun[qsrc]))

            qhalo = qsrc & (r_r200 <= 1.0)
            n_halo = int(np.count_nonzero(qhalo))
            m_halo = float(np.sum(masses_msun[qhalo]))

            q30 = qsrc & (r_pkpc <= 30.0)
            n30 = int(np.count_nonzero(q30))
            m30 = float(np.sum(masses_msun[q30]))

            print(
                f"{slabel:>8s}: all N={n_all:6d} M={m_all:12.5e} | "
                f"<R200 N={n_halo:6d} M={m_halo:12.5e} | "
                f"<30pkpc N={n30:5d} M={m30:12.5e}"
            )

    # ------------------------------------------------------------
    # Radial shell profile
    # ------------------------------------------------------------
    rmax_pkpc = args.rmax_r200 * r200_pkpc
    edges = np.linspace(0.0, rmax_pkpc, args.nbins + 1)

    rows = []
    print("\n--- RADIAL SHELL PROFILE ---")
    print(
        f"{'r_lo':>9s} {'r_hi':>9s} {'N':>7s} {'Mdust':>12s} "
        f"{'a_med':>12s} {'CF_med':>10s} {'Td_med':>10s} {'zform_med':>11s}"
    )

    grain = dust.get("GrainRadius")
    cf = dust.get("CarbonMassFraction")
    temp = dust.get("DustTemperature")
    tform = dust.get("DustFormationTime")
    zform = formation_time_to_redshift(tform) if tform is not None else None

    for i in range(args.nbins):
        lo, hi = edges[i], edges[i + 1]
        q = (r_pkpc >= lo) & (r_pkpc < hi)

        row = {
            "r_lo_pkpc": lo,
            "r_hi_pkpc": hi,
            "r_mid_pkpc": 0.5 * (lo + hi),
            "r_mid_r200": 0.5 * (lo + hi) / r200_pkpc,
            "N_dust": int(np.count_nonzero(q)),
            "M_dust_Msun": float(np.sum(masses_msun[q])),
            "GrainRadius_median": safe_median(grain[q]) if grain is not None else np.nan,
            "CarbonMassFraction_median": safe_median(cf[q]) if cf is not None else np.nan,
            "DustTemperature_median": safe_median(temp[q]) if temp is not None else np.nan,
            "FormationRedshift_median": safe_median(zform[q]) if zform is not None else np.nan,
        }

        if src_map is not None:
            for sval, slabel in src_map.items():
                qs = q & (src == sval)
                row[f"N_{slabel}"] = int(np.count_nonzero(qs))
                row[f"M_{slabel}_Msun"] = float(np.sum(masses_msun[qs]))

        rows.append(row)

        print(
            f"{lo:9.2f} {hi:9.2f} {row['N_dust']:7d} "
            f"{row['M_dust_Msun']:12.4e} "
            f"{row['GrainRadius_median']:12.4e} "
            f"{row['CarbonMassFraction_median']:10.4f} "
            f"{row['DustTemperature_median']:10.3f} "
            f"{row['FormationRedshift_median']:11.3f}"
        )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    csv_path = f"{args.out_prefix}_profile.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    cum_csv = f"{args.out_prefix}_cumulative.csv"
    with open(cum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aperture", "N_dust", "M_dust_Msun", "fraction_of_dust_within_R200"])
        for row in cumulative_rows:
            w.writerow(row)

    print(f"\nSaved radial profile: {csv_path}")
    print(f"Saved cumulative table: {cum_csv}")

    # ------------------------------------------------------------
    # Interactive Plotly
    # ------------------------------------------------------------
    if go is not None:
        # Sort particles once for cumulative curves.
        order = np.argsort(r_pkpc)
        rs = r_pkpc[order]
        ms = masses_msun[order]
        cum_mass = np.cumsum(ms)
        cum_count = np.arange(1, len(rs) + 1)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rs,
            y=cum_mass,
            mode="lines",
            name="Cumulative dust mass",
            yaxis="y1",
        ))

        fig.add_trace(go.Scatter(
            x=rs,
            y=cum_count,
            mode="lines",
            name="Cumulative dust count",
            yaxis="y2",
        ))

        if src_map is not None:
            for sval, slabel in src_map.items():
                q = src == sval
                rr = r_pkpc[q]
                mm = masses_msun[q]
                if len(rr) == 0:
                    continue
                oo = np.argsort(rr)
                fig.add_trace(go.Scatter(
                    x=rr[oo],
                    y=np.cumsum(mm[oo]),
                    mode="lines",
                    name=f"{slabel} cumulative mass",
                    yaxis="y1",
                ))

        for x, text in [
            (10, "10 pkpc"),
            (20, "20 pkpc"),
            (30, "30 pkpc"),
            (50, "50 pkpc"),
            (0.25*r200_pkpc, "0.25 R200"),
            (0.5*r200_pkpc, "0.5 R200"),
            (r200_pkpc, "R200"),
        ]:
            fig.add_vline(x=x, line_width=1, opacity=0.35)

        fig.update_layout(
            title=(
                f"Dust radial distribution — {Path(args.output_dir).name}"
                f"<br><sup>snap={args.snap:03d}, z={halo.z:.4f}, "
                f"R200={r200_pkpc:.1f} pkpc</sup>"
            ),
            xaxis=dict(title="Radius [pkpc]", range=[0, args.rmax_r200*r200_pkpc]),
            yaxis=dict(title="Cumulative dust mass [Msun]"),
            yaxis2=dict(
                title="Cumulative dust particle count",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5),
        )

        html_path = f"{args.out_prefix}.html"
        fig.write_html(
            html_path,
            include_plotlyjs=True,
            full_html=True,
            config=dict(displayModeBar=True, displaylogo=False,
                        scrollZoom=True, responsive=True),
        )
        print(f"Saved interactive Plotly diagnostic: {html_path}")
    else:
        print("Plotly not installed; skipped HTML diagnostic.")

    print("=" * 88)


if __name__ == "__main__":
    main()
