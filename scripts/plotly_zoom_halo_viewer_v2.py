#!/usr/bin/env python3
"""
plotly_zoom_halo_viewer.py

Interactive 3-D Plotly viewer for the CosmicGrain zoom suite.

Uses zoom_halo_utils.py to:
- identify the target FOF halo,
- optionally refine the center conservatively,
- recompute M200c and R200c from the particles,
- compare those values to the catalog values.

The R200 display uses THREE great circles per radius shell (xy, xz, yz).
This avoids the visually confusing dense wireframe used by the earlier
version, where each single sphere produced many latitude/longitude circles.
"""

import argparse
import glob
import os
from typing import List

import h5py
import numpy as np
import plotly.graph_objects as go

from halo_utils import (
    find_snapshot_and_group_files,
    get_zoom_halo,
    periodic_delta,
)

def _piece_key(fn):
    try:
        return int(os.path.basename(fn).split(".")[-2])
    except Exception:
        return fn


def read_particle_type(snapshot_files: List[str], ptype: int):
    coords, masses = [], []
    mt = None
    for fn in snapshot_files:
        with h5py.File(fn, "r") as f:
            if mt is None:
                mt = np.asarray(f["Header"].attrs["MassTable"], dtype=float)
            gname = f"PartType{ptype}"
            if gname not in f:
                continue
            g = f[gname]
            c = np.asarray(g["Coordinates"], dtype=float)
            coords.append(c)
            if "Masses" in g:
                masses.append(np.asarray(g["Masses"], dtype=float))
            else:
                masses.append(np.full(len(c), mt[ptype], dtype=float))
    if not coords:
        return None, None
    return np.concatenate(coords), np.concatenate(masses)


def ptype_label(pt):
    return {0:"Gas", 1:"HR DM", 2:"LR DM", 3:"PartType3",
            4:"Stars", 5:"BH", 6:"Dust"}.get(pt, f"PartType{pt}")


def default_size(pt):
    return {0:1.2, 1:1.0, 2:2.0, 4:3.0, 6:2.6}.get(pt, 1.5)


def default_budget(pt):
    return {0:120000, 1:80000, 2:100000, 4:120000, 6:120000}.get(pt, 80000)


def subsample(idx, budget, rng):
    if budget <= 0 or len(idx) <= budget:
        return idx
    return rng.choice(idx, size=budget, replace=False)


def great_circle_traces(radius, label):
    t = np.linspace(0, 2*np.pi, 160)
    zero = np.zeros_like(t)
    circles = [
        (radius*np.cos(t), radius*np.sin(t), zero),  # xy
        (radius*np.cos(t), zero, radius*np.sin(t)),  # xz
        (zero, radius*np.cos(t), radius*np.sin(t)),  # yz
    ]
    out = []
    for i,(x,y,z) in enumerate(circles):
        out.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=2),
            name=label,
            legendgroup=label,
            showlegend=(i == 0),
            hoverinfo="skip",
            opacity=0.35,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--group-index", type=int, default=None)
    ap.add_argument("--types", type=int, nargs="+", default=[0,1,2,4,6])
    ap.add_argument("--rmax-r200", type=float, default=5.0)
    ap.add_argument("--max-points", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-refine-center", action="store_true")
    ap.add_argument("--no-spheres", action="store_true")
    ap.add_argument("--sphere-levels", type=float, nargs="+", default=[1,2,3])
    ap.add_argument("--out", default="zoom_halo_3d.html")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    snap_files, _ = find_snapshot_and_group_files(args.output_dir, args.snap)

    halo = get_zoom_halo(
        args.output_dir, args.snap,
        group_index=args.group_index,
        refine_center=not args.no_refine_center,
        verbose=True,
    )

    center = halo.chosen_center_ckpch
    r200 = halo.so_r200_ckpch
    m200 = halo.so_m200_code
    rmax = args.rmax_r200 * r200
    box = halo.boxsize_ckpch

    if args.max_points is None:
        budgets = {pt: default_budget(pt) for pt in args.types}
    elif len(args.max_points) == 1:
        budgets = {pt:int(args.max_points[0]) for pt in args.types}
    elif len(args.max_points) == len(args.types):
        budgets = {pt:int(v) for pt,v in zip(args.types,args.max_points)}
    else:
        raise ValueError("--max-points must have one value or match --types")

    traces = []

    for pt in args.types:
        coords, masses = read_particle_type(snap_files, pt)
        if coords is None:
            print(f"PartType{pt}: absent")
            continue

        d = periodic_delta(coords, center, box)
        rr = np.linalg.norm(d, axis=1)
        inside = np.where(rr <= rmax)[0]

        if pt == 2:
            for level in np.unique(masses):
                qlevel = np.isclose(masses, level, rtol=1e-7, atol=0)
                idx = inside[qlevel[inside]]
                chosen = subsample(idx, budgets[pt], rng)
                if not len(chosen):
                    continue
                label = f"LR DM ({level*1e10/halo.h:.2e} Msun)"
                traces.append(go.Scatter3d(
                    x=d[chosen,0], y=d[chosen,1], z=d[chosen,2],
                    mode="markers", name=label,
                    marker=dict(size=default_size(pt), opacity=0.70),
                    hovertemplate="Δx=%{x:.1f}<br>Δy=%{y:.1f}<br>Δz=%{z:.1f} ckpc/h<extra></extra>",
                ))
                print(f"{label}: within={len(idx):,}, plotted={len(chosen):,}")
        else:
            chosen = subsample(inside, budgets[pt], rng)
            if not len(chosen):
                print(f"{ptype_label(pt)}: 0 within plotting radius")
                continue
            traces.append(go.Scatter3d(
                x=d[chosen,0], y=d[chosen,1], z=d[chosen,2],
                mode="markers", name=f"{ptype_label(pt)} (PartType{pt})",
                marker=dict(
                    size=default_size(pt),
                    opacity=0.82 if pt in (4,6) else 0.55,
                ),
                hovertemplate="Δx=%{x:.1f}<br>Δy=%{y:.1f}<br>Δz=%{z:.1f} ckpc/h<extra></extra>",
            ))
            print(f"{ptype_label(pt)}: within={len(inside):,}, plotted={len(chosen):,}")

    if not args.no_spheres:
        for fac in args.sphere_levels:
            traces.extend(great_circle_traces(fac*r200, f"{fac:g} R200"))

    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        name="Chosen halo center",
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="Chosen halo center<extra></extra>",
    ))

    # Also show the catalog center if the refined center was accepted.
    if halo.refinement_accepted:
        dc = periodic_delta(halo.catalog_center_ckpch[None,:], center, box)[0]
        traces.append(go.Scatter3d(
            x=[dc[0]], y=[dc[1]], z=[dc[2]],
            mode="markers",
            name="Catalog GroupPos",
            marker=dict(size=5, symbol="x"),
            hovertemplate="Catalog GroupPos<extra></extra>",
        ))

    lim = rmax
    fig = go.Figure(traces)
    fig.update_layout(
        title=(
            f"CosmicGrain zoom halo — group {halo.group_index}"
            f"<br><sup>snap={args.snap:03d}, z={halo.z:.4f}, "
            f"particle-SO M200c={m200*1e10/halo.h:.3e} Msun, "
            f"R200c={r200*halo.a/halo.h:.1f} pkpc, "
            f"view={args.rmax_r200:g} R200</sup>"
        ),
        uirevision="lock",
        scene=dict(
            xaxis=dict(title="Δx [ckpc/h]", range=[-lim,lim], autorange=False),
            yaxis=dict(title="Δy [ckpc/h]", range=[-lim,lim], autorange=False),
            zaxis=dict(title="Δz [ckpc/h]", range=[-lim,lim], autorange=False),
            aspectmode="cube",
            dragmode="orbit",
        ),
        scene_camera=dict(eye=dict(x=1.45,y=1.45,z=1.15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="center", x=0.5),
        margin=dict(l=0,r=0,b=0,t=100),
    )

    fig.write_html(
        args.out,
        include_plotlyjs=True,
        full_html=True,
        config=dict(displayModeBar=True, displaylogo=False,
                    scrollZoom=True, responsive=True),
    )
    print(f"\nSaved: {args.out}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
