#!/usr/bin/env python3
"""Interactive 3-D Plotly viewer for the CosmicGrain zoom suite.

The default invocation is intentionally simple::

    python3 scripts/plotly_zoom_halo_viewer_v3.py halo9235_output_1024

For standard ``halo<ID>_output_<resolution>`` suite directories, the latest
completed catalog and target group are inferred through ``halo_utils``.  An
explicit snapshot/group remains available for unusual or diagnostic cases.

Defaults emphasize visible baryonic structure: gas, stars, and dust inside
R200c, with larger markers for rare stellar and dust particles.  Buttons in
the HTML switch between full-halo, inner-halo, and galaxy axis ranges without
discarding the particles outside the current view.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import plotly.graph_objects as go

from halo_utils import (
    find_last_snap_num,
    find_snapshot_and_group_files,
    get_zoom_halo,
    get_zoom_halo_series,
    infer_zoom_target_group,
    periodic_delta,
)


TYPE_LABELS = {
    0: "Gas",
    1: "HR DM",
    2: "LR DM",
    3: "PartType3",
    4: "Stars",
    5: "BH",
    6: "Dust",
}

TYPE_COLORS = {
    0: "#32B8FF",
    1: "#9AA0A6",
    2: "#FF2D95",
    3: "#A78BFA",
    4: "#FFE066",
    5: "#FFFFFF",
    6: "#FF6B35",
}

TYPE_SIZES = {
    0: 3.0,
    1: 1.2,
    2: 5.0,
    3: 2.0,
    4: 8.0,
    5: 9.0,
    6: 6.0,
}

TYPE_OPACITIES = {
    0: 0.48,
    1: 0.18,
    2: 0.90,
    3: 0.45,
    4: 0.95,
    5: 1.00,
    6: 0.88,
}

TYPE_BUDGETS = {
    0: 120_000,
    1: 80_000,
    2: 100_000,
    4: 120_000,
    6: 120_000,
}


def _piece_key(filename: str):
    try:
        return int(os.path.basename(filename).split(".")[-2])
    except Exception:
        return filename


def ptype_label(ptype: int) -> str:
    return TYPE_LABELS.get(ptype, f"PartType{ptype}")


def default_budget(ptype: int) -> int:
    return TYPE_BUDGETS.get(ptype, 80_000)


def read_particle_type(snapshot_files: Sequence[str], ptype: int):
    """Read coordinates and masses for one particle type from all chunks."""
    coords, masses = [], []
    mass_table = None
    for filename in sorted(snapshot_files, key=_piece_key):
        with h5py.File(filename, "r") as handle:
            if mass_table is None:
                mass_table = np.asarray(
                    handle["Header"].attrs["MassTable"], dtype=float
                )
            name = f"PartType{ptype}"
            if name not in handle:
                continue
            group = handle[name]
            chunk_coords = np.asarray(group["Coordinates"], dtype=float)
            coords.append(chunk_coords)
            if "Masses" in group:
                masses.append(np.asarray(group["Masses"], dtype=float))
            else:
                masses.append(
                    np.full(len(chunk_coords), mass_table[ptype], dtype=float)
                )
    if not coords:
        return None, None
    return np.concatenate(coords), np.concatenate(masses)


def subsample(indices: np.ndarray, budget: int, rng) -> np.ndarray:
    if budget <= 0 or len(indices) <= budget:
        return indices
    return np.sort(rng.choice(indices, size=budget, replace=False))


def great_circle_traces(radius_pkpc: float, label: str, color: str):
    """Three uncluttered great circles representing one radius shell."""
    angle = np.linspace(0.0, 2.0 * np.pi, 180)
    zero = np.zeros_like(angle)
    circles = (
        (radius_pkpc*np.cos(angle), radius_pkpc*np.sin(angle), zero),
        (radius_pkpc*np.cos(angle), zero, radius_pkpc*np.sin(angle)),
        (zero, radius_pkpc*np.cos(angle), radius_pkpc*np.sin(angle)),
    )
    traces = []
    for number, (x, y, z) in enumerate(circles):
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(width=3, color=color),
            name=label,
            legendgroup=label,
            showlegend=(number == 0),
            hoverinfo="skip",
            opacity=0.38,
        ))
    return traces


def parse_budgets(
    particle_types: Sequence[int], values: Optional[Sequence[int]]
) -> Dict[int, int]:
    if values is None:
        return {ptype: default_budget(ptype) for ptype in particle_types}
    if len(values) == 1:
        return {ptype: int(values[0]) for ptype in particle_types}
    if len(values) == len(particle_types):
        return {
            ptype: int(value)
            for ptype, value in zip(particle_types, values)
        }
    raise ValueError("--max-points must have one value or match --types")


def resolve_target(args, parser):
    """Return ``(snapshot, group_index)`` with safe automatic inference."""
    if args.group_index is not None:
        snap = args.snap
        if snap is None:
            snap = find_last_snap_num(args.output_dir)
        if snap is None:
            parser.error("no completed group catalog was found")
        return int(snap), int(args.group_index)

    try:
        final_group, reference_snap, _ = infer_zoom_target_group(
            args.output_dir,
            reference_snap=None,
            verbose=True,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(f"automatic target selection failed: {exc}")

    if args.snap is None or int(args.snap) == reference_snap:
        return reference_snap, final_group

    requested_snap = int(args.snap)
    try:
        history = get_zoom_halo_series(
            args.output_dir,
            [requested_snap, reference_snap],
            group_index=final_group,
            reference_snap=reference_snap,
            verbose=False,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(
            f"could not track the automatic target to snapshot "
            f"{requested_snap:03d}: {exc}"
        )
    target = history.get(requested_snap)
    if target is None:
        parser.error(
            f"the target has no reliable match at snapshot {requested_snap:03d}"
        )
    return requested_snap, int(target["group_idx"])


def axis_range_update(limit: float) -> dict:
    return {
        "scene.xaxis.range": [-limit, limit],
        "scene.yaxis.range": [-limit, limit],
        "scene.zaxis.range": [-limit, limit],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Interactive target-centered CosmicGrain halo viewer"
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--snap", type=int,
        help="Snapshot to display (default: latest completed catalog)",
    )
    parser.add_argument(
        "--group-index", type=int,
        help="Explicit group override (normally inferred automatically)",
    )
    parser.add_argument(
        "--types", type=int, nargs="+", default=[0, 4, 6],
        help="Particle types to load (default: gas stars dust)",
    )
    parser.add_argument(
        "--rmax-r200", type=float, default=1.0,
        help="Particles loaded within this multiple of R200c (default: 1)",
    )
    parser.add_argument(
        "--max-points", type=int, nargs="*", default=None,
        help="One plotting budget or one value per requested type",
    )
    parser.add_argument(
        "--marker-scale", type=float, default=1.0,
        help="Multiply every marker size by this value (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-refine-center", action="store_true")
    parser.add_argument(
        "--center", type=float, nargs=3,
        metavar=("X", "Y", "Z"),
        help="Manual halo center [ckpc/h]; bypasses get_zoom_halo",
    )
    parser.add_argument(
        "--r200-pkpc", type=float,
        help="Manual R200c [physical kpc]; required with --center",
    )
    parser.add_argument(
        "--m200-msun", type=float,
        help="Manual M200c [Msun], used for the display title",
    )
    parser.add_argument("--no-spheres", action="store_true")
    parser.add_argument(
        "--sphere-levels", type=float, nargs="+", default=[0.1, 1.0],
        help="Radius circles in units of R200c (default: 0.1 1)",
    )
    parser.add_argument(
        "--out",
        help="Output HTML (default: inside output_dir with halo/snapshot name)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.rmax_r200 <= 0:
        parser.error("--rmax-r200 must be positive")
    if args.marker_scale <= 0:
        parser.error("--marker-scale must be positive")
    if not args.types or any(ptype < 0 or ptype > 6 for ptype in args.types):
        parser.error("--types must contain particle types from 0 through 6")

    snap, group_index = resolve_target(args, parser)
    rng = np.random.default_rng(args.seed)
    snap_files, _ = find_snapshot_and_group_files(args.output_dir, snap)

    if args.center is not None:
        if args.r200_pkpc is None:
            parser.error("--r200-pkpc is required when --center is supplied")

        # Backward-compatible/manual mode for older CosmicGrain runs whose
        # FOF catalogs do not contain Group_M_Crit200 / Group_R_Crit200.
        with h5py.File(snap_files[0], "r") as handle:
            header = handle["Header"].attrs
            halo_a = float(header["Time"])
            halo_h = float(handle["Parameters"].attrs["HubbleParam"])
            box = float(header["BoxSize"])

        center = np.asarray(args.center, dtype=float)
        r200_pkpc = float(args.r200_pkpc)
        r200_ckpch = r200_pkpc * halo_h / halo_a
        m200_msun = (
            float(args.m200_msun)
            if args.m200_msun is not None
            else float("nan")
        )
        halo_z = 1.0 / halo_a - 1.0
        halo_group_index = group_index
        refinement_accepted = False

        print("\nMANUAL HALO DEFINITION")
        print(f"  group index : {group_index}")
        print(f"  center      : {center} ckpc/h")
        print(f"  R200c       : {r200_pkpc:.3f} pkpc")
        print(f"  M200c       : {m200_msun:.3e} Msun")
        print(
            f"  a / z / h   : {halo_a:.8f} / "
            f"{halo_z:.8f} / {halo_h:.6f}"
        )
    else:
        halo = get_zoom_halo(
            args.output_dir,
            snap,
            group_index=group_index,
            refine_center=not args.no_refine_center,
            verbose=True,
        )

        center = halo.chosen_center_ckpch
        r200_ckpch = halo.so_r200_ckpch
        r200_pkpc = r200_ckpch * halo_a / halo_h
        m200_msun = halo.so_m200_code * 1.0e10 / halo_h
        box = halo.boxsize_ckpch

        halo_a = halo.a
        halo_h = halo.h
        halo_z = halo.z
        halo_group_index = halo.group_index
        refinement_accepted = halo.refinement_accepted

    rmax_ckpch = args.rmax_r200 * r200_ckpch
    rmax_pkpc = args.rmax_r200 * r200_pkpc
    budgets = parse_budgets(args.types, args.max_points)
    traces = []

    print("\nPARTICLES IN VIEW")
    for ptype in args.types:
        coords, masses = read_particle_type(snap_files, ptype)
        if coords is None:
            print(f"PartType{ptype}: absent")
            continue

        delta_ckpch = periodic_delta(coords, center, box)
        radius_ckpch = np.linalg.norm(delta_ckpch, axis=1)
        inside = np.flatnonzero(radius_ckpch <= rmax_ckpch)

        if ptype == 2:
            for level in np.unique(masses):
                at_level = np.isclose(masses, level, rtol=1.0e-7, atol=0.0)
                indices = inside[at_level[inside]]
                chosen = subsample(indices, budgets[ptype], rng)
                if not len(chosen):
                    continue
                delta_pkpc = delta_ckpch[chosen] * halo_a / halo_h
                radius_pkpc = radius_ckpch[chosen] * halo_a / halo_h
                mass_msun = masses[chosen] * 1.0e10 / halo_h
                custom = np.column_stack((radius_pkpc, mass_msun))
                label = f"LR DM {level*1e10/halo.h:.2e} Msun"
                traces.append(go.Scatter3d(
                    x=delta_pkpc[:, 0],
                    y=delta_pkpc[:, 1],
                    z=delta_pkpc[:, 2],
                    mode="markers",
                    name=f"{label} — {len(indices):,}",
                    marker=dict(
                        size=TYPE_SIZES[2] * args.marker_scale,
                        color=TYPE_COLORS[2],
                        opacity=TYPE_OPACITIES[2],
                    ),
                    customdata=custom,
                    hovertemplate=(
                        "LR DM<br>r=%{customdata[0]:.2f} pkpc"
                        "<br>mass=%{customdata[1]:.3e} Msun"
                        "<extra></extra>"
                    ),
                ))
                print(
                    f"{label}: within={len(indices):,}, "
                    f"plotted={len(chosen):,}"
                )
            continue

        chosen = subsample(inside, budgets[ptype], rng)
        if not len(chosen):
            print(f"{ptype_label(ptype)}: 0 within plotting radius")
            continue
        delta_pkpc = delta_ckpch[chosen] * halo_a / halo_h
        radius_pkpc = radius_ckpch[chosen] * halo_a / halo_h
        mass_msun = masses[chosen] * 1.0e10 / halo_h
        custom = np.column_stack((radius_pkpc, mass_msun))
        label = ptype_label(ptype)
        traces.append(go.Scatter3d(
            x=delta_pkpc[:, 0],
            y=delta_pkpc[:, 1],
            z=delta_pkpc[:, 2],
            mode="markers",
            name=f"{label} — {len(inside):,}",
            marker=dict(
                size=TYPE_SIZES.get(ptype, 2.0) * args.marker_scale,
                color=TYPE_COLORS.get(ptype, "#FFFFFF"),
                opacity=TYPE_OPACITIES.get(ptype, 0.6),
                line=(
                    dict(width=0.5, color="#FFF4D6")
                    if ptype in (4, 6) else dict(width=0)
                ),
            ),
            customdata=custom,
            hovertemplate=(
                f"{label}<br>r=%{{customdata[0]:.2f}} pkpc"
                "<br>mass=%{customdata[1]:.3e} Msun"
                "<extra></extra>"
            ),
        ))
        print(
            f"{label}: within={len(inside):,}, plotted={len(chosen):,}"
        )

    if not args.no_spheres:
        palette = ("#F4F4F5", "#80CBC4", "#B39DDB", "#EF9A9A")
        for number, factor in enumerate(args.sphere_levels):
            traces.extend(great_circle_traces(
                factor * r200_pkpc,
                f"{factor:g} R200",
                palette[number % len(palette)],
            ))

    traces.append(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode="markers",
        name="Halo center",
        marker=dict(size=8*args.marker_scale, symbol="diamond", color="white"),
        hovertemplate="Chosen halo center<extra></extra>",
    ))

    if args.center is None and refinement_accepted:
        catalog_offset = periodic_delta(
            halo.catalog_center_ckpch[None, :], center, box
        )[0] * halo_a / halo_h
        traces.append(go.Scatter3d(
            x=[catalog_offset[0]],
            y=[catalog_offset[1]],
            z=[catalog_offset[2]],
            mode="markers",
            name="Catalog GroupPos",
            marker=dict(size=7*args.marker_scale, symbol="x", color="#FF4D4D"),
            hovertemplate="Catalog GroupPos<extra></extra>",
        ))

    full_limit = rmax_pkpc
    inner_limit = min(0.3 * r200_pkpc, full_limit)
    galaxy_limit = min(0.1 * r200_pkpc, full_limit)
    buttons = [
        dict(
            label=f"Full ({args.rmax_r200:g} R200)",
            method="relayout",
            args=[axis_range_update(full_limit)],
        ),
        dict(
            label="Inner (0.3 R200)",
            method="relayout",
            args=[axis_range_update(inner_limit)],
        ),
        dict(
            label="Galaxy (0.1 R200)",
            method="relayout",
            args=[axis_range_update(galaxy_limit)],
        ),
    ]

    fig = go.Figure(traces)
    fig.update_layout(
        template="plotly_dark",
        title=(
            f"CosmicGrain target halo — group {halo_group_index}"
            f"<br><sup>snap={snap:03d}, z={halo_z:.4f}, "
            f"M200c={m200_msun:.3e} Msun, R200c={r200_pkpc:.1f} pkpc, "
            f"loaded to {args.rmax_r200:g} R200</sup>"
        ),
        uirevision="cosmicgrain-halo-view",
        scene=dict(
            xaxis=dict(
                title="Δx [pkpc]", range=[-full_limit, full_limit],
                autorange=False, backgroundcolor="#080B12",
            ),
            yaxis=dict(
                title="Δy [pkpc]", range=[-full_limit, full_limit],
                autorange=False, backgroundcolor="#080B12",
            ),
            zaxis=dict(
                title="Δz [pkpc]", range=[-full_limit, full_limit],
                autorange=False, backgroundcolor="#080B12",
            ),
            aspectmode="cube",
            dragmode="orbit",
        ),
        scene_camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            buttons=buttons,
            x=0.5,
            xanchor="center",
            y=0.995,
            yanchor="top",
            pad=dict(t=4, r=4),
        )],
        margin=dict(l=0, r=0, b=0, t=130),
    )

    if args.out:
        output = Path(args.out)
    else:
        output = Path(args.output_dir) / f"target_snap{snap:03d}_3d.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config=dict(
            displayModeBar=True,
            displaylogo=False,
            scrollZoom=True,
            responsive=True,
        ),
    )
    print(f"\nSaved: {output}")
    print("Mouse: left-drag orbit, wheel zoom; legend: click hide/show, double-click isolate.")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
