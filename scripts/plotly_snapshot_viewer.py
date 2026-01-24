#!/usr/bin/env python3
"""
plotly_snapshot_viewer.py

Interactive 3D Plotly viewer for Gadget-4 HDF5 snapshots.

Features:
- Loads one snapshot from a snapdir_XXX directory (multi-file snapshots supported)
- Plots selected particle types as separate Scatter3d traces
- Default log10 color scaling
- Interactive dropdown to switch color-mapped field between:
    Density, Metallicity, DustTemperature, DustMass
- Legend pinned top-left; colorbar pushed right
- Title includes output folder name
- Marker shapes: Gas=circle, Stars=star, Dust=square (approx cube)

Example:
  python plotly_snapshot_viewer.py ../run/snapdir_049 --snap 49 --types 0 4 6 \
    --color-by DustTemperature --out snap049_dusttemp.html
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import plotly.graph_objects as go


# ----------------------------
# Helpers
# ----------------------------

def _die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _log10_safe(arr: np.ndarray) -> np.ndarray:
    """log10 with non-positive -> NaN."""
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    m = arr > 0
    out[m] = np.log10(arr[m])
    return out


def _snap_str(snap: int) -> str:
    """Return 3-digit snapshot string without octal gotchas."""
    return f"{int(snap):03d}"


def _find_snapshot_files(snapdir: str, snap: int) -> List[str]:
    s = _snap_str(snap)
    patt = os.path.join(snapdir, f"snapshot_{s}.*.hdf5")
    files = sorted(glob.glob(patt))
    if not files:
        # Sometimes people have snapshot_###.hdf5 (single-file)
        patt2 = os.path.join(snapdir, f"snapshot_{s}.hdf5")
        files = sorted(glob.glob(patt2))
    return files


def _read_attr_group(g: h5py.Group) -> Dict[str, object]:
    out = {}
    for k in g.attrs.keys():
        out[k] = g.attrs[k]
    return out


def _as_float(x) -> Optional[float]:
    try:
        return float(np.array(x))
    except Exception:
        return None


# ----------------------------
# Data container
# ----------------------------

@dataclass
class PartData:
    ptype: int
    label: str
    pos: np.ndarray                 # (N,3)
    vel: Optional[np.ndarray]        # (N,3) or None
    mass: np.ndarray                # (N,)
    fields: Dict[str, np.ndarray]    # ProperCase keys matching HDF5 datasets (e.g., "Density")


# ----------------------------
# Snapshot reading
# ----------------------------

WANTED_FIELDS = ["Density", "Metallicity", "DustTemperature"]  # add more if you like


def _read_part_from_file(
    f: h5py.File,
    ptype: int,
    header: Dict[str, object],
    wanted_fields: List[str],
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, np.ndarray]]]:
    gname = f"PartType{ptype}"
    if gname not in f:
        return None

    g = f[gname]

    if "Coordinates" not in g:
        return None

    pos = g["Coordinates"][:].astype(np.float32, copy=False)

    vel = g["Velocities"][:].astype(np.float32, copy=False) if "Velocities" in g else None

    # Masses may be absent for constant-mass species; use Header MassTable in that case.
    if "Masses" in g:
        mass = g["Masses"][:].astype(np.float32, copy=False)
    else:
        mt = header.get("MassTable", None)
        if mt is None:
            _die("No Masses dataset and Header/MassTable missing.")
        mt = np.array(mt, dtype=float)
        if ptype >= len(mt):
            _die(f"MassTable does not contain entry for PartType{ptype}.")
        mconst = float(mt[ptype])
        if mconst <= 0:
            # Some types (e.g., stars) might have variable masses but still missing dataset in some configs.
            # If that's your case, you *must* output Masses.
            _die(f"PartType{ptype} has no Masses dataset and MassTable[{ptype}]={mconst} (<=0).")
        mass = np.full(pos.shape[0], mconst, dtype=np.float32)

    fields: Dict[str, np.ndarray] = {}
    for key in wanted_fields:
        if key in g:
            fields[key] = g[key][:]

    return pos, vel, mass, fields


def load_snapshot(snapdir: str, snap: int, types: List[int]) -> Tuple[Dict[int, PartData], Dict[str, object]]:
    files = _find_snapshot_files(snapdir, snap)
    if not files:
        _die(f"No snapshot files found in {snapdir} for snap={snap}.")

    # Read header from first file
    with h5py.File(files[0], "r") as f0:
        if "Header" not in f0:
            _die("HDF5 file missing /Header group.")
        header = _read_attr_group(f0["Header"])

    parts: Dict[int, PartData] = {}

    for ptype in types:
        all_pos = []
        all_vel = []
        all_mass = []
        all_fields: Dict[str, List[np.ndarray]] = {k: [] for k in WANTED_FIELDS}

        for fn in files:
            with h5py.File(fn, "r") as f:
                got = _read_part_from_file(f, ptype, header, WANTED_FIELDS)
                if got is None:
                    continue
                pos, vel, mass, fields = got
                all_pos.append(pos)
                all_mass.append(mass)
                if vel is not None:
                    all_vel.append(vel)

                for k in WANTED_FIELDS:
                    if k in fields:
                        all_fields[k].append(fields[k])

        if not all_pos:
            continue

        pos = np.concatenate(all_pos, axis=0)
        mass = np.concatenate(all_mass, axis=0)
        vel = np.concatenate(all_vel, axis=0) if all_vel else None

        fields_out: Dict[str, np.ndarray] = {}
        for k, chunks in all_fields.items():
            if chunks:
                fields_out[k] = np.concatenate(chunks, axis=0)

        label = {
            0: "Gas",
            1: "DM",
            2: "PartType2",
            3: "PartType3",
            4: "Stars",
            5: "BH",
            6: "Dust",
        }.get(ptype, f"PartType{ptype}")

        parts[ptype] = PartData(ptype=ptype, label=label, pos=pos, vel=vel, mass=mass, fields=fields_out)

    return parts, header


# ----------------------------
# Plotting
# ----------------------------

COLOR_CHOICES = ["Density", "Metallicity", "DustTemperature", "DustMass"]


def _get_color_array(pd: PartData, field: str) -> Optional[np.ndarray]:
    """
    Return the raw (linear) array for a requested field, or None if not available.
    Field names are ProperCase, matching HDF5 dataset naming (except DustMass, which uses Masses).
    """
    if field == "DustMass":
        # Use per-particle mass (works for any type, but most meaningful for dust)
        return pd.mass.astype(float, copy=False)

    return pd.fields.get(field, None)


def _choose_default_field(parts: Dict[int, PartData]) -> str:
    # Prefer DustTemperature if dust exists, else Density if gas exists, else Metallicity.
    if 6 in parts and _get_color_array(parts[6], "DustTemperature") is not None:
        return "DustTemperature"
    if 0 in parts and _get_color_array(parts[0], "Density") is not None:
        return "Density"
    if any(_get_color_array(p, "Metallicity") is not None for p in parts.values()):
        return "Metallicity"
    return "DustMass"


def make_plot(parts: Dict[int, PartData], header: Dict[str, object], snapdir: str,
              title: str, output_html: Optional[str], initial_color_by: Optional[str]) -> None:

    fig = go.Figure()

    outfolder = os.path.basename(os.path.abspath(os.path.join(snapdir, "..")))

    # Marker symbols: Scatter3d supports a subset; "star" works in many Plotly builds.
    symbol_map = {0: "circle-open", 1: "diamond-open", 4: "circle", 6: "square"}
    size_map   = {0: 2,0: 2,        4: 4,      6: 3}

    # Header subtitle bits
    subtitle = []
    a = _as_float(header.get("Time", None))
    z = _as_float(header.get("Redshift", None))
    box = _as_float(header.get("BoxSize", None))
    if a is not None:
        subtitle.append(f"a={a:.6g}")
    if z is not None:
        subtitle.append(f"z={z:.6g}")
    if box is not None:
        subtitle.append(f"Box={box:.0f}")

    # Determine initial field
    default_field = _choose_default_field(parts)
    color_by = initial_color_by if initial_color_by else default_field
    if color_by not in COLOR_CHOICES:
        _die(f"--color-by must be one of {COLOR_CHOICES} (got {color_by})")

    # Precompute log10 color arrays for each trace for each field
    trace_ptypes = list(parts.keys())  # keep insertion order from loading
    colors_by_field: Dict[str, List[Optional[np.ndarray]]] = {f: [] for f in COLOR_CHOICES}

    for ptype in trace_ptypes:
        pd = parts[ptype]
        for f in COLOR_CHOICES:
            arr = _get_color_array(pd, f)
            if arr is None:
                colors_by_field[f].append(None)
            else:
                colors_by_field[f].append(_log10_safe(arr))

    labels_by_field = {
        "Density":        "log₁₀(Density)",
        "Metallicity":    "log₁₀(Metallicity)",
        "DustTemperature":"log₁₀(DustTemperature)",
        "DustMass":       "log₁₀(DustMass)",
    }

    # Build traces with initial field
    show_scale_for_first_colored = True
    for i, ptype in enumerate(trace_ptypes):
        pd = parts[ptype]
        x, y, zc = pd.pos[:, 0], pd.pos[:, 1], pd.pos[:, 2]

        marker = dict(
            size=size_map.get(ptype, 2),
            symbol=symbol_map.get(ptype, "circle"),
        )

        init_c = colors_by_field[color_by][i]
        if init_c is not None:
            marker.update(dict(
                color=init_c,
                colorscale="Viridis",
                showscale=show_scale_for_first_colored,
                colorbar=dict(
                    title=dict(text=labels_by_field[color_by]),
                    x=1.02,
                ),
            ))
            show_scale_for_first_colored = False

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=zc,
            mode="markers",
            name=pd.label,
            marker=marker,
        ))

    # Dropdown to switch fields
    buttons = []
    ntr = len(fig.data)

    for field in COLOR_CHOICES:
        new_colors = [colors_by_field[field][i] for i in range(ntr)]

        # Show a SINGLE colorbar: first trace that has data for this field
        new_showscales = [False] * ntr
        scale_trace = None
        for i in range(ntr):
            if new_colors[i] is not None:
                new_showscales[i] = True
                scale_trace = i
                break

        # Update colorbar title text only on the trace showing the scale
        new_titles = [None] * ntr
        if scale_trace is not None:
            new_titles[scale_trace] = labels_by_field[field]

        buttons.append(dict(
            label=field,
            method="restyle",
            args=[{
                "marker.color": new_colors,
                "marker.showscale": new_showscales,
                "marker.colorbar.title.text": new_titles,
            }],
        ))

    fig.update_layout(
        title=f"{title} — {outfolder}" + ((" — " + ", ".join(subtitle)) if subtitle else ""),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend=dict(
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.86, y=0.95,          # ← moved near colorbar
            xanchor="left",
            yanchor="top",
            showactive=True,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        )],
        margin=dict(l=0, r=140, t=70, b=0),
    )

    # Helpful label for the dropdown
    fig.add_annotation(
        text="Color by (log₁₀):",
        x=0.86, y=0.985,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
    )

    if output_html:
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"Results saved in: {output_html}")
    else:
        fig.show()


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapdir", help="Path to snapdir_XXX directory")
    ap.add_argument("--snap", type=int, required=True, help="Snapshot number (e.g. 49)")
    ap.add_argument("--types", nargs="+", type=int, default=[0, 4, 6],
                    help="Particle types to plot (e.g. 0 4 6)")
    ap.add_argument("--color-by", default=None,
                    help="Initial color field (ProperCase): Density, Metallicity, DustTemperature, DustMass")
    ap.add_argument("--title", default="Gadget-4 Snapshot Viewer", help="Base title for plot")
    ap.add_argument("--out", default=None, help="Output HTML file (if omitted, opens interactive window)")

    args = ap.parse_args()

    snapdir = os.path.abspath(args.snapdir)
    if not os.path.isdir(snapdir):
        _die(f"Not a directory: {snapdir}")

    parts, header = load_snapshot(snapdir, args.snap, args.types)
    if not parts:
        _die("No particle data loaded. Check --types and that PartType groups exist in the snapshot.")

    make_plot(
        parts=parts,
        header=header,
        snapdir=snapdir,
        title=args.title,
        output_html=args.out,
        initial_color_by=args.color_by,
    )


if __name__ == "__main__":
    main()

