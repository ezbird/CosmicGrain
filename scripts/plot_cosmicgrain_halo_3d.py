#!/usr/bin/env python3
"""Interactive Plotly 3-D viewer for a CosmicGrain zoom halo."""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
import plotly.graph_objects as go

from halo_utils import get_zoom_halo, periodic_delta


CF_FIELDS = ("CarbonMassFraction", "CarbonFraction", "DustCarbonFraction", "CFRC")
RADIUS_FIELDS = ("GrainRadius", "DustGrainRadius", "GRAD")
SOURCE_FIELDS = ("DustSource", "Source", "DustSourceType", "DSRC")
SOURCE_NAMES = {0: "SNII", 1: "AGB", 2: "LRN"}


def snapshot_files(output_dir, snap):
    s = f"{snap:03d}"
    patterns = (
        os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.*.hdf5"),
        os.path.join(output_dir, f"snapdir_{s}", f"snapshot_{s}.hdf5"),
        os.path.join(output_dir, f"snapshot_{s}.*.hdf5"),
        os.path.join(output_dir, f"snapshot_{s}.hdf5"),
    )
    for pattern in patterns:
        found = sorted(set(glob.glob(pattern)))
        if found:
            return found
    raise FileNotFoundError(f"No snapshot {s} beneath {output_dir}")


def discover_last_snapshot(output_dir):
    values = set()
    for path in glob.glob(os.path.join(output_dir, "snapdir_*")):
        try:
            values.add(int(os.path.basename(path).split("_")[-1]))
        except ValueError:
            pass
    for path in glob.glob(os.path.join(output_dir, "snapshot_*.hdf5")):
        name = os.path.basename(path)
        try:
            values.add(int(name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass
    if not values:
        raise FileNotFoundError(f"No snapshots beneath {output_dir}")
    return max(values)


def first_field(names, candidates):
    names = set(names)
    return next((field for field in candidates if field in names), None)


def load_particles(files, ptype, center, box, radius_ckpch, wanted=()):
    positions = []
    fields = {name: [] for name in wanted}
    available = set()
    for filename in files:
        with h5py.File(filename, "r") as f:
            group_name = f"PartType{ptype}"
            if group_name not in f:
                continue
            group = f[group_name]
            available.update(group.keys())
            coords = np.asarray(group["Coordinates"], dtype=np.float64)
            delta = periodic_delta(coords, center, box)
            keep = np.einsum("ij,ij->i", delta, delta) < radius_ckpch**2
            if not np.any(keep):
                continue
            positions.append(delta[keep])
            for name in wanted:
                if name in group:
                    fields[name].append(np.asarray(group[name])[keep])
    if not positions:
        return None
    return {
        "position": np.concatenate(positions),
        "fields": {name: np.concatenate(parts) for name, parts in fields.items() if parts},
        "available": sorted(available),
    }


def sample_indices(n, maximum, seed):
    if maximum <= 0 or n <= maximum:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


def xyz(data, indices, length_to_pkpc):
    p = data["position"][indices] * length_to_pkpc
    return p[:, 0], p[:, 1], p[:, 2]


def main():
    ap = argparse.ArgumentParser(description="Create an orbitable Plotly 3-D CosmicGrain halo")
    ap.add_argument("output_dir")
    ap.add_argument("--group-index", type=int, required=True)
    ap.add_argument("--snap", type=int, default=None,
                    help="Snapshot number; default is the highest available")
    ap.add_argument("--radius-pkpc", type=float, default=30.0)
    ap.add_argument("--max-gas", type=int, default=50000)
    ap.add_argument("--max-stars", type=int, default=50000)
    ap.add_argument("--max-dust", type=int, default=50000)
    ap.add_argument("--max-dm", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=441)
    ap.add_argument("--show-dm", action="store_true",
                    help="Show HR dark matter initially (it is otherwise legend-only)")
    ap.add_argument("--no-refine-center", action="store_true")
    ap.add_argument("--output", default=None,
                    help="Output HTML path; default is inside output_dir")
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    snap = args.snap if args.snap is not None else discover_last_snapshot(output_dir)
    files = snapshot_files(output_dir, snap)
    halo = get_zoom_halo(output_dir, snap, group_index=args.group_index,
                         refine_center=not args.no_refine_center, verbose=False)
    center = halo.chosen_center_ckpch
    length_to_pkpc = halo.a / halo.h
    radius_ckpch = args.radius_pkpc / length_to_pkpc

    gas = load_particles(files, 0, center, halo.boxsize_ckpch, radius_ckpch)
    dm = load_particles(files, 1, center, halo.boxsize_ckpch, radius_ckpch)
    stars = load_particles(files, 4, center, halo.boxsize_ckpch, radius_ckpch)
    dust = load_particles(files, 6, center, halo.boxsize_ckpch, radius_ckpch,
                          CF_FIELDS + RADIUS_FIELDS + SOURCE_FIELDS)

    fig = go.Figure()
    counts = {}

    if dm is not None:
        ind = sample_indices(len(dm["position"]), args.max_dm, args.seed + 1)
        x, y, z = xyz(dm, ind, length_to_pkpc)
        counts["HR dark matter"] = (len(dm["position"]), len(ind))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="HR dark matter",
            visible=True if args.show_dm else "legendonly",
            marker=dict(size=1.2, color="#8b93a6", opacity=0.12),
            hovertemplate="HR DM<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f} pkpc<extra></extra>"))

    if gas is not None:
        ind = sample_indices(len(gas["position"]), args.max_gas, args.seed + 2)
        x, y, z = xyz(gas, ind, length_to_pkpc)
        counts["Gas"] = (len(gas["position"]), len(ind))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="Gas",
            marker=dict(size=2.0, color="#38bdf8", opacity=0.22),
            hovertemplate="Gas<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f} pkpc<extra></extra>"))

    if stars is not None:
        ind = sample_indices(len(stars["position"]), args.max_stars, args.seed + 3)
        x, y, z = xyz(stars, ind, length_to_pkpc)
        counts["Stars"] = (len(stars["position"]), len(ind))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="Stars",
            marker=dict(size=3.0, color="#ffd166", opacity=0.80),
            hovertemplate="Star<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f} pkpc<extra></extra>"))

    if dust is not None:
        ind = sample_indices(len(dust["position"]), args.max_dust, args.seed + 4)
        x, y, z = xyz(dust, ind, length_to_pkpc)
        counts["Dust"] = (len(dust["position"]), len(ind))
        cf_field = first_field(dust["fields"], CF_FIELDS)
        radius_field = first_field(dust["fields"], RADIUS_FIELDS)
        source_field = first_field(dust["fields"], SOURCE_FIELDS)
        cf = np.asarray(dust["fields"][cf_field][ind], dtype=float).reshape(-1) if cf_field else np.full(len(ind), 0.5)
        if radius_field:
            radius = np.asarray(dust["fields"][radius_field][ind], dtype=float).reshape(-1)
            safe_radius = np.clip(radius, 1.0e-30, None)
            lo, hi = np.nanpercentile(np.log10(safe_radius), [5, 95])
            marker_size = 2.2 + 3.8 * np.clip((np.log10(safe_radius)-lo)/max(hi-lo, 1e-12), 0, 1)
        else:
            radius = np.full(len(ind), np.nan)
            marker_size = 3.0
        if source_field:
            source = np.asarray(dust["fields"][source_field][ind]).reshape(-1)
            source_text = np.array([SOURCE_NAMES.get(int(v), f"source {v}") for v in source])
        else:
            source_text = np.full(len(ind), "unknown")
        custom = np.column_stack((radius, cf, source_text))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="Dust",
            customdata=custom,
            marker=dict(size=marker_size, color=cf, cmin=0, cmax=1,
                        colorscale=[[0, "#9c6644"], [0.5, "#c9ada7"], [1, "#202020"]],
                        opacity=0.72,
                        colorbar=dict(title="Dust carbon<br>mass fraction", thickness=16)),
            hovertemplate=("Dust (%{customdata[2]})<br>x=%{x:.2f}<br>y=%{y:.2f}<br>"
                           "z=%{z:.2f} pkpc<br>radius=%{customdata[0]:.3g} nm<br>"
                           "carbon fraction=%{customdata[1]:.3f}<extra></extra>")))

    limit = args.radius_pkpc
    fig.update_layout(
        title=(f"{os.path.basename(output_dir)} — snapshot {snap:03d}, z={halo.z:.3f}, "
               f"group {halo.group_index}, r < {limit:g} pkpc"),
        dragmode="orbit",
        template="plotly_dark", width=1100, height=850,
        margin=dict(l=0, r=0, b=0, t=55),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.35)"),
        scene=dict(
            dragmode="orbit", aspectmode="cube",
            xaxis=dict(title="x − center [pkpc]", range=[-limit, limit]),
            yaxis=dict(title="y − center [pkpc]", range=[-limit, limit]),
            zaxis=dict(title="z − center [pkpc]", range=[-limit, limit]),
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15), up=dict(x=0, y=0, z=1))),
        annotations=[dict(
            text="Left-drag: orbit · right-drag: pan · wheel: zoom · double-click: reset",
            x=0.5, y=0.01, xref="paper", yref="paper", showarrow=False,
            bgcolor="rgba(0,0,0,0.45)")])

    output = args.output or os.path.join(
        output_dir, f"{os.path.basename(output_dir)}_snap{snap:03d}_3d_{limit:g}pkpc.html")
    fig.write_html(output, include_plotlyjs=True, full_html=True,
                   config={"scrollZoom": True, "displaylogo": False,
                           "responsive": True,
                           "modeBarButtonsToRemove": ["zoom3d"],
                           "modeBarButtonsToAdd": ["resetCameraLastSave3d"]},
                   post_script="""
const haloPlot = document.getElementById('{plot_id}');
Plotly.relayout(haloPlot, {'dragmode': 'orbit', 'scene.dragmode': 'orbit'});
haloPlot.addEventListener('contextmenu', function(event) {
    event.preventDefault();
}, true);
""")
    print(f"Halo center: {center} ckpc/h")
    print(f"Particle-SO R200c: {halo.so_r200_ckpch*length_to_pkpc:.3f} pkpc")
    for name, (total, shown) in counts.items():
        print(f"{name}: {total:,} inside radius; plotted {shown:,}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
