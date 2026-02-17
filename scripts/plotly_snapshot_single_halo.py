#!/usr/bin/env python3
"""
plotly_snapshot_viewer.py

Interactive 3D Plotly viewer for Gadget-4 HDF5 snapshots with:
- Optional halo extraction mode using halo_utils
- radius-limited plotting
- per-type point budgets
- subsampling that can be applied ONLY to selected types (default: gas only)
- robust color-by handling (log scaling, percentile clipping)
- fixed axis ranges AND fixed aspect ratio so toggling traces doesn't rescale the scene

Examples:
  # Normal mode (full snapshot):
  python plotly_snapshot_viewer.py ../snapdir_015 --snap 15 \
      --types 0 4 6 --color-by DustTemperature \
      --center 25000 25000 25000 --rmax 3000

  # HALO MODE (extract target halo automatically):
  python plotly_snapshot_single_halo.py ../7_output_zoom_2048_halo569_50Mpc_dust/snapdir_009 --snap 9 --catalog ../7_output_zoom_2048_halo569_50Mpc_dust/groups_009/fof_subhalo_tab_009.0.hdf5 --types 0 1 4 6 --rmax 200 --out my_halo.html
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: This script requires h5py. Try: pip install h5py", file=sys.stderr)
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: This script requires plotly. Try: pip install plotly", file=sys.stderr)
    sys.exit(1)

# Try to import halo_utils
HALO_UTILS_AVAILABLE = False
try:
    from halo_utils import load_target_halo, extract_dust_spatially, convert_to_physical_units
    HALO_UTILS_AVAILABLE = True
except ImportError:
    pass


# -------------------------
# Helpers: reading snapshots
# -------------------------

def _find_snapshot_files(path: str, snap: Optional[int]) -> List[str]:
    """Accept either:
      - a directory containing snapshot pieces (snapdir_XXX)
      - a single HDF5 snapshot file
    """
    if os.path.isfile(path) and path.endswith((".hdf5", ".h5")):
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    patterns = []
    if snap is not None:
        s = f"{snap:03d}"
        patterns += [
            os.path.join(path, f"snapshot_{s}.*.hdf5"),
            os.path.join(path, f"snapshot_{s}.*.h5"),
            os.path.join(path, f"snap_{s}.*.hdf5"),
            os.path.join(path, f"snap_{s}.*.h5"),
            os.path.join(path, f"snapshot_{s}.hdf5"),
            os.path.join(path, f"snapshot_{s}.h5"),
        ]
    else:
        patterns += [
            os.path.join(path, "snapshot_*.hdf5"),
            os.path.join(path, "snapshot_*.h5"),
            os.path.join(path, "snap_*.hdf5"),
            os.path.join(path, "snap_*.h5"),
        ]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No snapshot HDF5 files found in {path}. "
            f"Try providing a direct .hdf5 file or use --snap with a snapdir."
        )
    return files


def _get_snapshot_base(path: str, snap: Optional[int]) -> str:
    """Get the snapshot base path for halo_utils (e.g., 'snapdir_049/snapshot_049')."""
    if os.path.isfile(path):
        # Remove .X.hdf5 suffix if present
        base = path
        if '.hdf5' in base:
            base = base[:base.rfind('.hdf5')]
            # If it ends with .0, .1, etc., remove that too
            if base[-2:].startswith('.') and base[-1].isdigit():
                base = base[:-2]
        return base
    
    # Directory path
    if snap is None:
        raise ValueError("Must provide --snap when using directory path with --catalog")
    
    s = f"{snap:03d}"
    # Try different naming conventions
    for prefix in ['snapshot', 'snap']:
        candidate = os.path.join(path, f"{prefix}_{s}")
        test_files = glob.glob(f"{candidate}.*.hdf5") or glob.glob(f"{candidate}.hdf5")
        if test_files:
            return candidate
    
    raise FileNotFoundError(f"Cannot determine snapshot base in {path} for snap {snap}")


def _read_block(files: List[str], ptype: int, block: str) -> Optional[np.ndarray]:
    """Read a dataset for a given PartType across snapshot pieces and concatenate.
    Returns None if the dataset does not exist.
    """
    parts = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            gname = f"PartType{ptype}"
            if gname not in f:
                continue
            grp = f[gname]
            if block not in grp:
                continue
            parts.append(grp[block][()])

    if not parts:
        return None

    try:
        return np.concatenate(parts, axis=0)
    except ValueError:
        return np.concatenate([np.atleast_1d(x) for x in parts], axis=0)


def _read_header_attr(files: List[str], key: str) -> Optional[np.ndarray]:
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "Header" in f and key in f["Header"].attrs:
                return f["Header"].attrs[key]
    return None


def _get_redshift(files: List[str]) -> Optional[float]:
    z = _read_header_attr(files, "Redshift")
    if z is not None:
        try:
            return float(np.array(z).squeeze())
        except Exception:
            pass

    a = _read_header_attr(files, "Time")
    if a is not None:
        try:
            a = float(np.array(a).squeeze())
            if a > 0:
                return (1.0 / a) - 1.0
        except Exception:
            pass

    return None


# -------------------------
# Subsampling
# -------------------------

def _apply_radius_cut(coords: np.ndarray, center: np.ndarray, rmax: Optional[float]) -> np.ndarray:
    if rmax is None:
        return np.ones(coords.shape[0], dtype=bool)
    d = coords - center[None, :]
    r = np.sqrt(np.sum(d * d, axis=1))
    return r <= rmax


def _subsample_indices_stride(n: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    return np.arange(0, n, stride, dtype=np.int64)


def _subsample_indices_fraction(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    frac = float(frac)
    if frac >= 1.0:
        return np.arange(n, dtype=np.int64)
    if frac <= 0.0:
        return np.array([], dtype=np.int64)
    k = max(1, int(round(frac * n)))
    return rng.choice(n, size=k, replace=False)


def _subsample_indices_weighted(weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement using numpy.choice."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, weights.size)

    w = np.array(weights, dtype=np.float64)
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        return rng.choice(weights.size, size=k, replace=False)
    w /= s
    return rng.choice(weights.size, size=k, replace=False, p=w)


def _compute_density_weights(density: np.ndarray, power: float = 1.0, eps: float = 1e-30) -> np.ndarray:
    d = np.array(density, dtype=np.float64)
    d[~np.isfinite(d)] = 0.0
    d = np.maximum(d, eps)
    w = d ** float(power)
    w[~np.isfinite(w)] = 0.0
    return w


# -------------------------
# Color handling
# -------------------------

def _prep_color_array(color: np.ndarray,
                      log_color: bool,
                      clip_percentiles: Tuple[float, float]) -> Tuple[np.ndarray, float, float]:
    c = np.array(color, dtype=np.float64)
    c[~np.isfinite(c)] = np.nan

    if log_color:
        c = np.where(c > 0, np.log10(c), np.nan)

    lo_p, hi_p = clip_percentiles
    finite = np.isfinite(c)
    if np.count_nonzero(finite) == 0:
        c = np.zeros_like(c)
        return c, 0.0, 1.0

    lo = np.nanpercentile(c, lo_p)
    hi = np.nanpercentile(c, hi_p)
    if not np.isfinite(lo):
        lo = np.nanmin(c)
    if not np.isfinite(hi):
        hi = np.nanmax(c)
    if hi <= lo:
        hi = lo + 1.0

    c = np.clip(c, lo, hi)
    return c, float(lo), float(hi)


# -------------------------
# Plotting helpers
# -------------------------

def _ptype_label(pt: int) -> str:
    return {
        0: "Gas",
        1: "DM",
        2: "Disk",
        3: "Bulge",
        4: "Stars",
        5: "BH",
        6: "Dust",
    }.get(pt, f"PartType{pt}")


def _default_marker_size(pt: int) -> float:
    return {
        0: 1.0,   # gas
        1: 0.9,   # DM
        4: 3.0,   # stars
        6: 2.0,   # dust
    }.get(pt, 1.0)


def _lock_scene_ranges_from_data(per_trace_xyz: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 pad_frac: float = 0.02) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    xs = [t[0] for t in per_trace_xyz]
    ys = [t[1] for t in per_trace_xyz]
    zs = [t[2] for t in per_trace_xyz]

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))

    def pad(lo: float, hi: float) -> Tuple[float, float]:
        span = hi - lo
        if span <= 0:
            span = 1.0
        p = pad_frac * span
        return lo - p, hi + p

    return pad(xmin, xmax), pad(ymin, ymax), pad(zmin, zmax)


def _aspectratio_from_ranges(xr, yr, zr) -> Dict[str, float]:
    sx = max(1e-12, float(xr[1] - xr[0]))
    sy = max(1e-12, float(yr[1] - yr[0]))
    sz = max(1e-12, float(zr[1] - zr[0]))
    m = max(sx, sy, sz)
    return dict(x=sx / m, y=sy / m, z=sz / m)


def build_figure(traces: List[go.Scatter3d],
                 title_html: str,
                 xr: Tuple[float, float],
                 yr: Tuple[float, float],
                 zr: Tuple[float, float]) -> go.Figure:
    fig = go.Figure(data=traces)

    aspectratio = _aspectratio_from_ranges(xr, yr, zr)

    fig.update_layout(
        title=title_html,
        uirevision="lock",
        scene=dict(
            xaxis=dict(title="x [kpc]", range=[xr[0], xr[1]], autorange=False),
            yaxis=dict(title="y [kpc]", range=[yr[0], yr[1]], autorange=False),
            zaxis=dict(title="z [kpc]", range=[zr[0], zr[1]], autorange=False),
            aspectmode="manual",
            aspectratio=aspectratio,
            dragmode="orbit",
        ),
        scene_camera=dict(eye=dict(x=1.4, y=1.4, z=1.15)),
        margin=dict(l=0, r=0, t=120, b=0),  # Increased top margin to make room
        legend=dict(
            orientation="h",      # Horizontal layout
            yanchor="bottom",
            y=1.02,               # Just above the plot area
            xanchor="center",
            x=0.5,                # Centered
            itemsizing="constant"
        ),
    )
    return fig


# -------------------------
# CLI
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Plotly 3D viewer for Gadget-4 HDF5 snapshots.")

    ap.add_argument("path", help="Snapshot .hdf5 file OR snapdir directory.")
    ap.add_argument("--snap", type=int, default=None, help="Snapshot number (used when path is a snapdir).")

    # HALO MODE
    ap.add_argument("--catalog", default=None, 
                    help="Subfind catalog (e.g., fof_subhalo_tab_049.0.hdf5) to extract target halo. "
                         "Enables halo extraction mode.")

    ap.add_argument("--types", type=int, nargs="+", default=[0, 4, 6],
                    help="PartTypes to plot (e.g. 0 4 6). Default: 0 4 6")

    ap.add_argument("--out", default="snapshot_plot.html", help="Output HTML filename.")
    ap.add_argument("--plotlyjs", choices=["embed", "cdn"], default="embed",
                    help="embed=offline self-contained, cdn=smaller HTML. Default: embed")

    ap.add_argument("--center", type=float, nargs=3, default=None,
                    help="Center for radius cut. In halo mode, defaults to halo center. Otherwise: 0 0 0")
    ap.add_argument("--rmax", type=float, default=None,
                    help="Max radius to include. In halo mode, defaults to 2*halfmass_rad.")

    ap.add_argument("--color-by", default=None,
                    help="Dataset name inside each PartType group to color points by.")
    ap.add_argument("--log-color", action="store_true",
                    help="Apply log10 to color-by values (positive-only).")

    ap.add_argument("--clip", type=float, nargs=2, default=[1.0, 99.0],
                    help="Percentile clip for color range. Default: 1 99")

    ap.add_argument("--subsample", choices=["none", "stride", "fraction", "density"], default="density",
                    help="Subsampling mode. Default: density")
    ap.add_argument("--subsample-types", type=int, nargs="+", default=[0],
                    help="Which PartTypes the subsampling applies to. Default: 0 (gas only).")

    ap.add_argument("--stride", type=int, default=5,
                    help="Stride for --subsample stride. Default: 5")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="Fraction for --subsample fraction. Default: 0.05")

    ap.add_argument("--max-points", type=int, nargs="*", default=None,
                    help="Max points per type, same length as --types.")

    ap.add_argument("--density-field", default="Density",
                    help="Field to use for density-weighted subsampling. Default: Density")
    ap.add_argument("--density-power", type=float, default=1.0,
                    help="Weights = Density^power for density subsampling. Default: 1.0")

    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    ap.add_argument("--show", action="store_true", help="Open in browser.")
    
    ap.add_argument("--convert-mass", action="store_true",
                    help="Convert masses from code units (1e10 Msun) to Msun. Only in halo mode.")

    return ap.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Check if halo mode is requested
    halo_mode = args.catalog is not None
    
    if halo_mode and not HALO_UTILS_AVAILABLE:
        print("ERROR: --catalog requires halo_utils.py. Make sure it's in the same directory.", file=sys.stderr)
        sys.exit(1)

    files = _find_snapshot_files(args.path, args.snap)
    redshift = _get_redshift(files)

    # Default budgets
    default_budget = {
        0: 200_000,   # gas
        1: 120_000,   # DM
        4: 120_000,   # stars
        6: 120_000,   # dust
    }

    if args.max_points is None:
        max_points_map = {pt: default_budget.get(pt, 80_000) for pt in args.types}
    else:
        if len(args.max_points) == 1 and len(args.types) > 1:
            max_points_map = {pt: int(args.max_points[0]) for pt in args.types}
        elif len(args.max_points) != len(args.types):
            print("ERROR: --max-points must have length 1 or match --types length.", file=sys.stderr)
            sys.exit(2)
        else:
            max_points_map = {pt: int(m) for pt, m in zip(args.types, args.max_points)}

    subsample_types = set(int(x) for x in args.subsample_types)

    # ======================
    # HALO EXTRACTION MODE
    # ======================
    if halo_mode:
        print("=" * 60)
        print("HALO EXTRACTION MODE")
        print("=" * 60)
        
        snapshot_base = _get_snapshot_base(args.path, args.snap)
        print(f"Snapshot base: {snapshot_base}")
        print(f"Catalog: {args.catalog}")
        print()
        
        # Extract halo
        halo = load_target_halo(
            args.catalog,
            snapshot_base,
            particle_types=args.types,
            verbose=True
        )
        
        # Get halo properties
        halo_info = halo['halo_info']
        halo_pos = halo_info['position']
        halo_mass = halo_info['mass']
        halo_halfmass = halo_info['halfmass_rad']
        
        # Set defaults for center and rmax
        center = np.array(args.center) if args.center is not None else halo_pos
        rmax = args.rmax if args.rmax is not None else (halo_halfmass * 2.0)
        
        print(f"\nUsing center: {center}")
        print(f"Using rmax: {rmax:.2f} kpc")
        
        # Extract dust spatially if requested
        if 6 in args.types:
            print("\nExtracting dust spatially (Subfind doesn't track PartType6)...")
            dust_data = extract_dust_spatially(snapshot_base, halo_pos, radius_kpc=rmax, verbose=True)
            if dust_data is not None:
                halo['dust'] = dust_data
        
        # Convert units if requested
        if args.convert_mass:
            print("\nConverting masses to M_sun...")
            ptype_names = {0: 'gas', 1: 'dm', 2: 'dm2', 4: 'stars', 5: 'bh', 6: 'dust'}
            for pt in args.types:
                pname = ptype_names.get(pt)
                if pname and pname in halo:
                    convert_to_physical_units(halo[pname], mass_in_msun=True)
        
        # Prepare data dict for plotting
        # Map particle type numbers to names
        ptype_names = {0: 'gas', 1: 'dm', 2: 'dm2', 4: 'stars', 5: 'bh', 6: 'dust'}
        halo_data = {}
        for pt in args.types:
            pname = ptype_names.get(pt)
            if pname and pname in halo:
                halo_data[pt] = halo[pname]
        
        print("\n" + "=" * 60)
    else:
        # Normal mode
        center = np.array(args.center) if args.center is not None else np.array([0.0, 0.0, 0.0])
        rmax = args.rmax
        halo_data = None
        halo_info = None

    # ======================
    # PLOTTING
    # ======================
    traces: List[go.Scatter3d] = []
    per_trace_xyz: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    available_counts: Dict[int, int] = {}
    plotted_counts: Dict[int, int] = {}
    total_plotted = 0

    for pt in args.types:
        # Get coordinates
        if halo_mode and halo_data and pt in halo_data:
            coords = halo_data[pt]['Coordinates']
        else:
            coords = _read_block(files, pt, "Coordinates")
        
        if coords is None:
            print(f"Warning: PartType{pt} has no Coordinates. Skipping.")
            continue

        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            print(f"Warning: PartType{pt} Coordinates shape {coords.shape} unexpected. Skipping.")
            continue

        # Radius cut (in halo mode, this is already applied, but user might specify additional rmax)
        rad_mask = _apply_radius_cut(coords, center, rmax)
        idx0 = np.nonzero(rad_mask)[0]
        n_in = int(idx0.size)
        available_counts[pt] = n_in

        if n_in == 0:
            print(f"Info: PartType{pt} has 0 particles within rmax. Skipping.")
            continue

        budget = int(max_points_map.get(pt, 80_000))
        budget = max(0, budget)

        # Decide whether subsampling applies
        do_subsample = (args.subsample != "none") and (pt in subsample_types)

        rel = np.arange(n_in, dtype=np.int64)

        if do_subsample:
            density = None
            if args.subsample == "density":
                if halo_mode and halo_data and pt in halo_data:
                    density = halo_data[pt].get(args.density_field)
                else:
                    density = _read_block(files, pt, args.density_field)
                
                if density is None:
                    print(f"Warning: PartType{pt} missing '{args.density_field}' for density subsample.")
                    density = None

            if args.subsample == "stride":
                rel = _subsample_indices_stride(n_in, args.stride)

            elif args.subsample == "fraction":
                rel = _subsample_indices_fraction(n_in, args.fraction, rng)

            elif args.subsample == "density":
                k = min(budget, n_in) if budget > 0 else n_in
                if density is None:
                    rel = rng.choice(n_in, size=min(k, n_in), replace=False)
                else:
                    d_in = np.asarray(density, dtype=np.float64)[idx0]
                    w = _compute_density_weights(d_in, power=args.density_power)
                    rel = _subsample_indices_weighted(w, k=min(k, n_in), rng=rng)

        # Enforce budget
        if budget > 0 and rel.size > budget:
            rel = rng.choice(rel, size=budget, replace=False)

        idx = idx0[rel]
        coords_sel = coords[idx]

        # Color array
        showscale = False
        ctitle = None
        if args.color_by is not None:
            if halo_mode and halo_data and pt in halo_data:
                c_raw = halo_data[pt].get(args.color_by)
            else:
                c_raw = _read_block(files, pt, args.color_by)
            
            if c_raw is None:
                print(f"Warning: PartType{pt} missing '{args.color_by}'. Using uniform color.")
                c = np.zeros(coords_sel.shape[0], dtype=np.float64)
                cmin, cmax = 0.0, 1.0
                showscale = False
            else:
                c_raw = np.asarray(c_raw)
                if c_raw.ndim > 1:
                    c_raw = np.linalg.norm(c_raw, axis=-1)
                c_sel = np.asarray(c_raw, dtype=np.float64)[idx]
                c, cmin, cmax = _prep_color_array(
                    c_sel, log_color=args.log_color,
                    clip_percentiles=(args.clip[0], args.clip[1]),
                )
                showscale = True
                ctitle = args.color_by + (" (log10)" if args.log_color else "")
        else:
            c = np.zeros(coords_sel.shape[0], dtype=np.float64)
            cmin, cmax = 0.0, 1.0
            showscale = False

        label = f"{_ptype_label(pt)} (PartType{pt})"
        plotted_counts[pt] = int(coords_sel.shape[0])
        total_plotted += coords_sel.shape[0]

        mode_note = args.subsample if do_subsample else "none"
        print(f"{label}: in r-cut = {n_in:,}, plotted = {coords_sel.shape[0]:,} "
              f"(budget={budget:,}, subsample={mode_note})")

        marker = dict(
            size=_default_marker_size(pt),
            opacity=1.0,
        )

        # Shared colorbar on first trace
        if args.color_by is not None and showscale:
            first_pt_with_scale = None
            for tpt in args.types:
                if plotted_counts.get(tpt, 0) > 0:
                    first_pt_with_scale = tpt
                    break

            marker.update(dict(
                color=c,
                cmin=cmin,
                cmax=cmax,
                colorscale="Viridis",
                colorbar=dict(title=ctitle) if pt == first_pt_with_scale else None,
                showscale=(pt == first_pt_with_scale),
            ))

        trace = go.Scatter3d(
            x=coords_sel[:, 0],
            y=coords_sel[:, 1],
            z=coords_sel[:, 2],
            mode="markers",
            name=label,
            marker=marker,
            hoverinfo="skip",
            hovertemplate=None,
        )

        traces.append(trace)
        per_trace_xyz.append((coords_sel[:, 0], coords_sel[:, 1], coords_sel[:, 2]))

    if not traces:
        print("ERROR: No traces to plot.", file=sys.stderr)
        sys.exit(3)

    # Lock axis ranges
    xr, yr, zr = _lock_scene_ranges_from_data(per_trace_xyz, pad_frac=0.02)

    # Title
    snap_str = f"{args.snap:03d}" if args.snap is not None else "N/A"
    z_str = f"{redshift:.6g}" if redshift is not None else "N/A"

    parts_lines = []
    for pt in args.types:
        a = available_counts.get(pt, 0)
        p = plotted_counts.get(pt, 0)
        parts_lines.append(f"{_ptype_label(pt)}: {a:,} avail / {p:,} plotted")
    counts_str = " | ".join(parts_lines)

    if halo_mode and halo_info:
        title_html = (
            f"Target Halo (ID={halo_info['id']}, M={halo_info['mass']:.2e})"
            f"<br><sup>"
            f"snap={snap_str} | z={z_str}"
            f"<br>{counts_str}"
            f"<br>center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) | rmax={rmax:.1f} kpc"
            f"</sup>"
        )
    else:
        title_html = (
            "Gadget-4 Snapshot Viewer"
            f"<br><sup>"
            f"snap={snap_str} | z={z_str}"
            f"<br>{counts_str}"
            f"<br>center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
            + (f" | rmax={rmax:.1f}" if rmax is not None else "")
            + f"</sup>"
        )

    fig = build_figure(traces, title_html, xr, yr, zr)

    config = dict(
        displayModeBar=True,
        displaylogo=False,
        scrollZoom=True,
        responsive=True,
    )

    include_plotlyjs = "cdn" if args.plotlyjs == "embed" else "True"
    fig.write_html(args.out, include_plotlyjs=include_plotlyjs, full_html=True, config=config)

    print("\nCounts summary:")
    for pt in args.types:
        a = available_counts.get(pt, 0)
        p = plotted_counts.get(pt, 0)
        print(f"  {_ptype_label(pt)} (PartType{pt}): {a:,} avail / {p:,} plotted")

    print(f"\nSaved: {args.out}")
    if halo_mode and halo_info:
        print(f"Halo mass: {halo_info['mass']:.2e} (code units)")
        print(f"Halfmass radius: {halo_info['halfmass_rad']:.2f} kpc")
    print(f"Total plotted points: {total_plotted:,}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
