#!/usr/bin/env python3
"""
make_3d_snapshot.py  —  Interactive 3-D Halo 569 snapshot viewer
=================================================================
Generates a self-contained HTML file (plotly) showing DM / gas / dust
as a 3-D scatter.  Open in any browser, rotate/zoom freely.

Usage
-----
  python make_3d_snapshot.py \\
      --snapdir /scratch/cygnus/CosmicGrain/output_s5_1024 \\
      --snap    80 \\
      --center  x y z   (physical kpc, optional) \\
      --size_pkpc 300 \\
      --out     halo569_z0.html

  # Loop over a few key epochs (ICs, z=5, z=2, z=1, z=0):
  for SNAP in 0 20 50 70 80; do
      python make_3d_snapshot.py --snapdir ... --snap $SNAP --out halo569_s${SNAP}.html
  done

Requirements: h5py, numpy, plotly (pip install plotly)
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import h5py
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    sys.exit("Install plotly first:  pip install plotly")


# ── re-use helpers from make_zoom_movie (or inline them here) ────────────────

def find_snap(snapdir, snap_num, snapbase="snap"):
    """Locate single or multi-file snapshot by number."""
    for pat in [
        os.path.join(snapdir, f"{snapbase}_{snap_num:03d}.hdf5"),
        os.path.join(snapdir, f"{snapbase}_{snap_num:04d}.hdf5"),
    ]:
        if os.path.exists(pat):
            return ("single", pat)
    for dname in [
        os.path.join(snapdir, f"{snapbase}_{snap_num:04d}"),
        os.path.join(snapdir, f"{snapbase}_{snap_num:03d}"),
    ]:
        if os.path.isdir(dname):
            pieces = sorted(glob.glob(os.path.join(dname, "*.hdf5")))
            if pieces:
                return ("multi", pieces)
    return None


def read_positions(snap_entry, ptype):
    kind, path = snap_entry
    files = path if kind == "multi" else [path]
    chunks = []
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                key = f"PartType{ptype}"
                if key in hf:
                    chunks.append(hf[key]["Coordinates"][:].astype(np.float32))
        except Exception:
            pass
    return np.concatenate(chunks) if chunks else None


def snap_header(snap_entry):
    kind, path = snap_entry
    fname = path[0] if kind == "multi" else path
    with h5py.File(fname, "r") as f:
        hdr = dict(f["Header"].attrs)
        params = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    h = float(params.get("HubbleParam", hdr.get("HubbleParam", 0.6774)))
    return hdr, h


def subsample(pos, n_max):
    if pos is None or len(pos) == 0:
        return None
    if len(pos) > n_max:
        idx = np.random.default_rng(42).choice(len(pos), n_max, replace=False)
        return pos[idx]
    return pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapdir",   required=True)
    p.add_argument("--snapbase",  default="snap")
    p.add_argument("--snap",      type=int, required=True, help="Snapshot number")
    p.add_argument("--out",       default="halo569_3d.html")
    p.add_argument("--center",    nargs=3, type=float, default=None,
                   metavar=("X","Y","Z"), help="Center in physical kpc")
    p.add_argument("--size_pkpc", type=float, default=300.0,
                   help="Half-box side to show in pkpc")
    p.add_argument("--n_dm",   type=int, default=80000)
    p.add_argument("--n_gas",  type=int, default=30000)
    p.add_argument("--n_dust", type=int, default=10000)
    p.add_argument("--n_star", type=int, default=10000)
    args = p.parse_args()

    snap_entry = find_snap(args.snapdir, args.snap, args.snapbase)
    if snap_entry is None:
        sys.exit(f"Snapshot {args.snap} not found in {args.snapdir}")

    hdr, h = snap_header(snap_entry)
    a  = float(hdr["Time"])
    z  = 1.0 / a - 1.0
    boxsize = float(hdr["BoxSize"]) * a / h  # physical kpc

    # Center
    if args.center:
        cen = np.array(args.center, dtype=float)
    else:
        # Crude: box center (replace with GroupPos lookup if catalogs present)
        cen = np.array([boxsize/2, boxsize/2, boxsize/2])
        print(f"No center specified — using box center {cen}.  "
              f"Pass --center X Y Z for Halo 569.")

    S = args.size_pkpc
    print(f"Snap {args.snap:04d}  z={z:.3f}  a={a:.4f}  center={cen}")

    def load(ptype, nmax, name):
        raw = read_positions(snap_entry, ptype)
        if raw is None:
            print(f"  {name}: not found")
            return None
        pos = raw.astype(np.float64) * a / h
        dx  = pos - cen
        mask= np.all(np.abs(dx) < S, axis=1)
        pos = dx[mask]
        n_in = len(pos)
        pos = subsample(pos, nmax)
        print(f"  {name}: {n_in:,} in window → plotting {len(pos):,}")
        return pos

    parts = {
        "Dark Matter": (load(1, args.n_dm,   "DM"),
                        "rgba(100,190,255,0.12)", 1.0),
        "Gas":         (load(0, args.n_gas,  "Gas"),
                        "rgba(255,120,40,0.25)",  1.5),
        "Dust":        (load(6, args.n_dust, "Dust"),
                        "rgba(255,60,220,0.60)",  2.0),
        "Stars":       (load(4, args.n_star, "Stars"),
                        "rgba(255,255,120,0.80)", 2.0),
    }

    traces = []
    for name, (pos, color, sz) in parts.items():
        if pos is None or len(pos) == 0:
            continue
        traces.append(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode="markers",
            marker=dict(size=sz, color=color, opacity=1.0),
            name=name,
        ))

    axis_style = dict(
        backgroundcolor="black",
        gridcolor="#2a2a2a",
        zerolinecolor="#444",
        color="white",
        showspikes=False,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>CosmicGrain — Halo 569</b><br>"
                 f"z = {z:.3f}  |  ±{S:.0f} pkpc window",
            font=dict(color="white", size=16),
            x=0.5,
        ),
        paper_bgcolor="#0a0a0a",
        scene=dict(
            xaxis=dict(title="Δx [pkpc]", **axis_style),
            yaxis=dict(title="Δy [pkpc]", **axis_style),
            zaxis=dict(title="Δz [pkpc]", **axis_style),
            bgcolor="black",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="white", size=12),
            bordercolor="#555",
            borderwidth=1,
        ),
        font=dict(color="white"),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # Add a thin bounding box wireframe
    corners_x = [-S, S, S,-S,-S,  S, S,-S,-S,  S,  S,-S]
    corners_y = [-S,-S, S, S,-S, -S, S, S,-S, -S,  S, S]
    corners_z = [-S,-S,-S,-S, S,  S, S, S,-S, -S, -S,-S]
    fig.add_trace(go.Scatter3d(
        x=corners_x, y=corners_y, z=corners_z,
        mode="lines",
        line=dict(color="#333", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"\n✓  3-D viewer written to: {args.out}")
    print(f"   Open in a browser and rotate freely!")


if __name__ == "__main__":
    main()
