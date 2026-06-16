#!/usr/bin/env python3
"""
make_zoom_movie.py  —  CosmicGrain / Gadget-4 zoom evolution visualizer
========================================================================
Renders a multi-panel movie of Halo 569 from z~99 → z=0.

Panels:
  Left  : dark matter (PartType1) — projected density, viridis
  Center: gas (PartType0) — projected density, inferno
  Right : dust (PartType6) — projected density, plasma  [skipped pre-dust snaps]

Optionally overlays a 4th panel: stellar mass (PartType4) once stars appear.

Usage
-----
  python make_zoom_movie.py \\
      --snapdir  /scratch/cygnus/CosmicGrain/output_s5_1024 \\
      --snapbase snap \\
      --outdir   ./frames \\
      --movie    zoom_evolution.mp4 \\
      --res 512 \\
      --size_pkpc 600 \\
      --fps 12 \\
      --nproc 8

For a quick test on a handful of snaps:
  python make_zoom_movie.py ... --snap_range 0 10

Requirements:  numpy, h5py, matplotlib, scipy, tqdm, ffmpeg (system)
Optional:      multiprocessing (parallel frame rendering)
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import multiprocessing as mp

# ──────────────────────────────────────────────────────────────────────────────
# Halo 569 verified comoving centers  (ckpc/h)
#
# These are density-weighted shrinking-sphere centers derived from gas
# particles at z=0 using a tight initial radius (150 ckpc/h) to avoid
# FOF superstructure contamination.  GroupPos[0] from the SubFind catalogs
# is NOT used here — it is unreliable at all resolutions because the FOF
# linker bridges Halo 569 to surrounding structure, offsetting the centroid
# by up to ~460 pkpc (verified and documented in the analysis pipeline).
#
# These coordinates are comoving and stable: to get the physical center at
# any snapshot, compute  center_phys = CENTER_CKPC_H * a / h  (physical kpc).
# Using a fixed comoving anchor is correct for a movie: the main progenitor
# drifts <2 Mpc over the Hubble time, well within any reasonable window.
#
# R200 is fixed at 85.95 ckpc/h (127.7 pkpc) from the 1024³ Subfind z=0
# catalog and applied at all resolutions.
# ──────────────────────────────────────────────────────────────────────────────
HALO569_CENTERS_CKPC_H = {
    512:  np.array([23083.102, 23519.314, 23665.764]),
    1024: np.array([23060.507, 23523.082, 23657.845]),  # DM potential minimum, snap 047
    2048: np.array([23084.035, 23511.898, 23649.725]),
}
HALO569_R200_CKPC_H = 85.95   # ckpc/h  →  127.7 pkpc at z=0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_snapshots(snapdir, snapbase="snap"):
    """
    Return sorted list of (snap_number, snap_entry) for all HDF5 snapshots.

    Handles three layouts:
      1. snapdir/snap_NNN.hdf5                     (single-file, custom base)
      2. snapdir/snap_NNN/snap_NNN.K.hdf5          (multi-file, custom base)
      3. snapdir/snapdir_NNN/snapshot_NNN.K.hdf5   (Gadget-4 default layout)
         also catches the single-file variant: snapdir_NNN/snapshot_NNN.hdf5
    """
    snaps = {}

    # ── Layout 1 & 2: user-supplied snapbase ──────────────────────────────
    for pat in [f"{snapbase}_???.hdf5", f"{snapbase}_????.hdf5"]:
        for p in sorted(glob.glob(os.path.join(snapdir, pat))):
            num = int(Path(p).stem.split("_")[-1])
            snaps[num] = ("single", p)

    for pat in [f"{snapbase}_???", f"{snapbase}_????"]:
        for d in sorted(glob.glob(os.path.join(snapdir, pat))):
            if os.path.isdir(d):
                pieces = sorted(glob.glob(os.path.join(d, "*.hdf5")))
                if pieces:
                    num = int(Path(d).name.split("_")[-1])
                    snaps[num] = ("multi", pieces)

    # ── Layout 3: Gadget-4 default  snapdir_NNN/snapshot_NNN[.K].hdf5 ────
    if not snaps:
        for d in sorted(glob.glob(os.path.join(snapdir, "snapdir_???"))):
            if not os.path.isdir(d):
                continue
            num = int(Path(d).name.split("_")[-1])
            pieces = sorted(glob.glob(os.path.join(d, "snapshot_???.*.hdf5")))
            if not pieces:
                # single-file variant
                pieces = sorted(glob.glob(os.path.join(d, "snapshot_???.hdf5")))
            if pieces:
                if len(pieces) == 1:
                    snaps[num] = ("single", pieces[0])
                else:
                    snaps[num] = ("multi", pieces)

    return [(k, v) for k, v in sorted(snaps.items())]


def open_snap_header(snap_entry):
    """Return (h5file_handle, header_dict) for the first chunk."""
    kind, path = snap_entry
    fname = path[0] if kind == "multi" else path
    f = h5py.File(fname, "r")
    hdr = dict(f["Header"].attrs)
    params = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    h = float(params.get("HubbleParam", hdr.get("HubbleParam", 0.6774)))
    return f, hdr, h


def read_part_positions(snap_entry, part_type):
    """
    Read all particle positions for a given PartType across all chunks.
    Returns (N,3) float32 array in code units (comoving kpc/h).
    Returns None if PartType not present.
    """
    kind, path = snap_entry
    files = path if kind == "multi" else [path]
    chunks = []
    for fname in files:
        try:
            with h5py.File(fname, "r") as f:
                key = f"PartType{part_type}"
                if key not in f:
                    continue
                pos = f[key]["Coordinates"][:]
                chunks.append(pos.astype(np.float32))
        except Exception:
            continue
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def read_part_masses(snap_entry, part_type, fallback_mass=None):
    """
    Read particle masses.  For DM (type 1) Gadget stores a single MassTable entry.
    Returns 1-D float32 array, or None.
    """
    kind, path = snap_entry
    files = path if kind == "multi" else [path]
    chunks = []
    mass_table_val = None
    npart_total = 0

    for fname in files:
        try:
            with h5py.File(fname, "r") as f:
                hdr = dict(f["Header"].attrs)
                if mass_table_val is None:
                    mt = hdr.get("MassTable", [0]*6)
                    if part_type < len(mt):
                        mass_table_val = float(mt[part_type])

                key = f"PartType{part_type}"
                if key not in f:
                    continue
                npart_total += f[key]["Coordinates"].shape[0]
                if "Masses" in f[key]:
                    chunks.append(f[key]["Masses"][:].astype(np.float32))
        except Exception:
            continue

    if chunks:
        return np.concatenate(chunks, axis=0)
    if mass_table_val and mass_table_val > 0 and npart_total > 0:
        return np.full(npart_total, mass_table_val, dtype=np.float32)
    if fallback_mass and npart_total > 0:
        return np.full(npart_total, fallback_mass, dtype=np.float32)
    return None


def get_halo_center(snap_entry, hdr, h, user_center=None, resolution=1024):
    """
    Returns the center of Halo 569 in PHYSICAL kpc at the snapshot's epoch.

    Priority
    --------
    1. --center X Y Z CLI override  (physical kpc, applied as-is every frame)
    2. HALO569_CENTERS_CKPC_H[resolution]  — verified shrinking-sphere comoving
       anchor, converted to physical via  pos_phys = pos_ckpc_h * a / h
    3. Box center  (last-resort fallback; prints a warning)

    NOTE: FOF GroupPos is intentionally NOT used.  It is unreliable for Halo 569
    at all resolutions because the FOF linker bridges to surrounding structure,
    producing centroid offsets of 340–460 pkpc relative to the true nucleus.
    """
    a = float(hdr["Time"])  # scale factor

    # 1. CLI override (physical kpc, constant across all frames)
    if user_center is not None:
        return np.array(user_center, dtype=float)

    # 2. Verified comoving anchor → physical kpc
    center_ckpc_h = HALO569_CENTERS_CKPC_H.get(resolution)
    if center_ckpc_h is not None:
        return center_ckpc_h * a / h  # physical kpc

    # 3. Box center fallback (only reached if resolution not in table)
    boxsize = float(hdr["BoxSize"])  # ckpc/h
    print(f"  WARNING: resolution {resolution} not in HALO569_CENTERS_CKPC_H — "
          f"falling back to box center.  Add the shrinking-sphere center to the table.")
    return np.array([boxsize / 2, boxsize / 2, boxsize / 2]) * a / h


def project_density(pos, weights, center_phys, size_pkpc, npix, smooth_kpc=None):
    """
    2D projected density map.
    pos       : (N,3) physical kpc
    weights   : (N,) masses or None (equal weight)
    center    : (3,) physical kpc
    size_pkpc : half-size of the box in pkpc
    npix      : output grid size
    Returns   : (npix, npix) float32 map (log10 of sum)
    """
    dx = pos[:, 0] - center_phys[0]
    dy = pos[:, 1] - center_phys[1]
    dz = pos[:, 2] - center_phys[2]

    mask = (
        (np.abs(dx) < size_pkpc) &
        (np.abs(dy) < size_pkpc) &
        (np.abs(dz) < size_pkpc)
    )
    dx, dy = dx[mask], dy[mask]
    if weights is not None:
        w = weights[mask]
    else:
        w = np.ones(mask.sum(), dtype=np.float32)

    # Bin
    bins = np.linspace(-size_pkpc, size_pkpc, npix + 1)
    H, _, _ = np.histogram2d(dx, dy, bins=[bins, bins], weights=w)
    H = H.astype(np.float32)

    if smooth_kpc is not None:
        pix_kpc = 2 * size_pkpc / npix
        sigma = smooth_kpc / pix_kpc
        H = gaussian_filter(H, sigma=max(sigma, 0.5))

    # log scale with floor
    H = np.log10(H + 1e-10)
    return H


def make_colormap_with_alpha(base_cmap_name, alpha_power=1.0):
    """Returns a colormap where alpha fades to 0 at low values."""
    base = plt.cm.get_cmap(base_cmap_name)
    colors = base(np.linspace(0, 1, 256))
    colors[:, 3] = np.linspace(0, 1, 256) ** alpha_power
    return mcolors.ListedColormap(colors)


# ──────────────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(args_tuple):
    (snap_num, snap_entry, outdir, cfg) = args_tuple

    npix      = cfg["res"]
    size_pkpc = cfg["size_pkpc"]
    smooth    = cfg["smooth_kpc"]
    resolution= cfg["resolution"]
    user_cen  = cfg["center"]
    dpi       = cfg["dpi"]
    with_dust = cfg["with_dust"]
    with_stars= cfg["with_stars"]

    outpath = os.path.join(outdir, f"frame_{snap_num:05d}.png")
    if os.path.exists(outpath) and not cfg.get("overwrite", False):
        return outpath

    try:
        f, hdr, h = open_snap_header(snap_entry)
        f.close()
    except Exception as e:
        print(f"  [skip {snap_num}] header error: {e}")
        return None

    a     = float(hdr["Time"])
    z     = 1.0 / a - 1.0
    t_Gyr = _a_to_Gyr(a)

    center = get_halo_center(snap_entry, hdr, h, user_cen, resolution)

    # ── Read particles ──────────────────────────────────────────────────────
    dm_pos  = read_part_positions(snap_entry, 1)
    gas_pos = read_part_positions(snap_entry, 0)
    dm_mass = read_part_masses(snap_entry, 1)
    gas_mass= read_part_masses(snap_entry, 0)

    # Physical coordinates
    def to_phys(pos_code):
        if pos_code is None: return None
        return pos_code.astype(np.float64) * a / h  # physical kpc

    dm_pos_p  = to_phys(dm_pos)
    gas_pos_p = to_phys(gas_pos)

    dust_pos_p = star_pos_p = None
    dust_mass = star_mass = None

    if with_dust:
        dust_pos  = read_part_positions(snap_entry, 6)
        dust_mass = read_part_masses(snap_entry, 6)
        dust_pos_p = to_phys(dust_pos)

    if with_stars:
        star_pos  = read_part_positions(snap_entry, 4)
        star_mass = read_part_masses(snap_entry, 4)
        star_pos_p = to_phys(star_pos)

    # ── Build maps ─────────────────────────────────────────────────────────
    dm_map  = project_density(dm_pos_p,  dm_mass,  center, size_pkpc, npix, smooth) if dm_pos_p  is not None else None
    gas_map = project_density(gas_pos_p, gas_mass, center, size_pkpc, npix, smooth) if gas_pos_p is not None else None
    dust_map= project_density(dust_pos_p,dust_mass,center, size_pkpc, npix, smooth) if dust_pos_p is not None else None
    star_map= project_density(star_pos_p,star_mass,center, size_pkpc, npix, smooth) if star_pos_p is not None else None

    # ── Layout ─────────────────────────────────────────────────────────────
    panels = [
        (dm_map,   "viridis", "Dark Matter"),
        (gas_map,  "inferno", "Gas"),
    ]
    if with_dust:
        panels.append((dust_map, "plasma",  "Dust (CosmicGrain)"))
    if with_stars:
        panels.append((star_map, "YlOrBr",  "Stars"))
    n_panels = len(panels)

    # Build a figure where every panel is exactly square.
    # Strategy: use pixel-exact sizing.  Each panel maps npix×npix data
    # pixels onto PANEL_PX screen pixels, padded by fixed margins.
    PANEL_PX   = npix          # one screen pixel per data pixel
    MARGIN_L   = 52            # pixels: left margin (y-axis label + ticks)
    MARGIN_R   = 8             # pixels: right margin
    MARGIN_BOT = 44            # pixels: bottom (x-label + ticks)
    MARGIN_TOP = 56            # pixels: top (suptitle, 2 lines)
    GAP        = 6             # pixels: gap between panels

    total_w = MARGIN_L + n_panels * PANEL_PX + (n_panels - 1) * GAP + MARGIN_R
    total_h = MARGIN_BOT + PANEL_PX + MARGIN_TOP

    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi),
                     dpi=dpi, facecolor="black")

    # Convert pixel margins to figure fractions
    left   = MARGIN_L / total_w
    right  = 1.0 - MARGIN_R / total_w
    bottom = MARGIN_BOT / total_h
    top    = 1.0 - MARGIN_TOP / total_h
    wspace_frac = GAP / PANEL_PX   # GridSpec wspace is relative to panel width

    gs = GridSpec(1, n_panels, figure=fig,
                  left=left, right=right,
                  bottom=bottom, top=top,
                  wspace=wspace_frac)

    extent = [-size_pkpc, size_pkpc, -size_pkpc, size_pkpc]

    def add_panel(ax, idx, dmap, cmap_name, title, vmin_pct=2, vmax_pct=99.5):
        ax.set_facecolor("black")
        ax.set_xlim(-size_pkpc, size_pkpc)
        ax.set_ylim(-size_pkpc, size_pkpc)
        # Do NOT set aspect here — figure geometry already guarantees square axes
        if dmap is not None:
            finite = dmap[np.isfinite(dmap)]
            populated = finite[finite > np.log10(1e-9)]
            if populated.size > 10:
                vmin = np.percentile(populated, vmin_pct)
                vmax = np.percentile(finite, vmax_pct)
            else:
                vmin, vmax = -5.0, 0.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            ax.imshow(dmap.T, origin="lower", extent=extent,
                      cmap=cmap_name, vmin=vmin, vmax=vmax,
                      interpolation="bilinear",
                      aspect="auto")   # axes shape is already square; auto fills it
        else:
            ax.text(0.5, 0.5, "no data", color="#555", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11)

        ax.set_title(title, color="white", fontsize=11, pad=4)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        if idx == 0:
            ax.set_ylabel("y  [pkpc]", color="white", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("x  [pkpc]", color="white", fontsize=9)

        # Scale bar (bottom-left corner)
        raw = size_pkpc / 4
        mag = 10 ** np.floor(np.log10(max(raw, 1e-6)))
        bar_len = round(raw / mag) * mag
        x0 = -size_pkpc * 0.85
        y0 = -size_pkpc * 0.88
        ax.plot([x0, x0 + bar_len], [y0, y0], color="white", lw=2,
                solid_capstyle="butt")
        ax.text(x0 + bar_len / 2, y0 + size_pkpc * 0.04,
                f"{int(bar_len)} pkpc", color="white", ha="center",
                va="bottom", fontsize=7)

    for idx, (dmap, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        add_panel(ax, idx, dmap, cmap, title)

    # ── Title / info bar ───────────────────────────────────────────────────
    n_dm  = len(dm_pos_p)   if dm_pos_p   is not None else 0
    n_gas = len(gas_pos_p)  if gas_pos_p  is not None else 0
    n_dust= len(dust_pos_p) if dust_pos_p is not None else 0

    fig.suptitle(
        f"CosmicGrain  •  Halo 569  •  z = {z:.2f}   (t = {t_Gyr:.2f} Gyr)\n"
        f"DM: {n_dm:,}   Gas: {n_gas:,}   Dust: {n_dust:,}   "
        f"Box shown: {2*size_pkpc:.0f} pkpc",
        color="white", fontsize=10,
        y=1.0 - 4 / total_h,
        va="top",
    )

    fig.savefig(outpath, dpi=dpi, facecolor="black",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return outpath


def _a_to_Gyr(a, H0=67.74, Om=0.3089, Ol=0.6911):
    """Crude numerical integration for age of universe."""
    from scipy.integrate import quad
    def integrand(z):
        zp1 = 1 + z
        E = np.sqrt(Om * zp1**3 + Ol)
        return 1.0 / (zp1 * E)
    z = 1.0 / a - 1.0
    t, _ = quad(integrand, z, np.inf)
    return t * 977.8 / H0  # Gyr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render a CosmicGrain zoom evolution movie.")
    parser.add_argument("--snapdir",   required=True, help="Directory with snapshots")
    parser.add_argument("--snapbase",  default="snap", help="Snapshot filename prefix")
    parser.add_argument("--outdir",    default="./frames", help="Frame output directory")
    parser.add_argument("--movie",     default="zoom_evolution.mp4", help="Output movie filename")
    parser.add_argument("--res",       type=int, default=512, help="Projection grid pixels (per side)")
    parser.add_argument("--size_pkpc", type=float, default=600.0,
                        help="Half-size of the projection window in pkpc")
    parser.add_argument("--smooth_kpc",type=float, default=2.0,
                        help="Gaussian smoothing kernel in pkpc (0=off)")
    parser.add_argument("--fps",       type=int, default=12, help="Movie frames per second")
    parser.add_argument("--dpi",       type=int, default=150, help="Frame DPI")
    parser.add_argument("--nproc",     type=int, default=4, help="Parallel processes")
    parser.add_argument("--snap_range",nargs=2, type=int, default=None,
                        metavar=("FIRST","LAST"), help="Restrict to snap numbers FIRST..LAST")
    parser.add_argument("--center",    nargs=3, type=float, default=None,
                        metavar=("X","Y","Z"), help="Halo center override in physical kpc")
    parser.add_argument("--resolution",type=int, default=1024,
                        help="Simulation resolution tier (512/1024/2048) for center lookup")
    parser.add_argument("--no_dust",   action="store_true", help="Skip PartType6")
    parser.add_argument("--no_stars",  action="store_true", help="Skip PartType4")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing frames")
    parser.add_argument("--frames_only",action="store_true", help="Render frames but skip ffmpeg")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    snaps = find_snapshots(args.snapdir, args.snapbase)
    if not snaps:
        sys.exit(f"No snapshots found in {args.snapdir} with base '{args.snapbase}'")

    if args.snap_range:
        lo, hi = args.snap_range
        snaps = [(n, e) for n, e in snaps if lo <= n <= hi]

    print(f"Found {len(snaps)} snapshots.  Rendering to {args.outdir}/")

    cfg = dict(
        res       = args.res,
        size_pkpc = args.size_pkpc,
        smooth_kpc= args.smooth_kpc if args.smooth_kpc > 0 else None,
        resolution= args.resolution,
        center    = args.center,
        dpi       = args.dpi,
        with_dust = not args.no_dust,
        with_stars= not args.no_stars,
        overwrite = args.overwrite,
    )

    work = [(snap_num, snap_entry, args.outdir, cfg)
            for snap_num, snap_entry in snaps]

    if args.nproc > 1:
        with mp.Pool(args.nproc) as pool:
            frames = list(tqdm(pool.imap(render_frame, work), total=len(work),
                               desc="Rendering frames"))
    else:
        frames = [render_frame(w) for w in tqdm(work, desc="Rendering frames")]

    frames = [f for f in frames if f is not None]
    print(f"Rendered {len(frames)} frames.")

    if args.frames_only or not frames:
        return

    # ── Assemble movie with ffmpeg ─────────────────────────────────────────
    # Create a sorted file list
    listfile = os.path.join(args.outdir, "frame_list.txt")
    frames_sorted = sorted(frames)
    with open(listfile, "w") as fh:
        for p in frames_sorted:
            fh.write(f"file '{os.path.abspath(p)}'\n")
            fh.write(f"duration {1.0/args.fps:.4f}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", listfile,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # ensure even dims
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        args.movie
    ]
    print("Running ffmpeg …")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:\n", result.stderr)
        # Try fallback with glob pattern instead of concat list
        pattern = os.path.join(args.outdir, "frame_%05d.png")
        cmd2 = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-pattern_type", "glob",
            "-i", os.path.join(args.outdir, "frame_?????.png"),
            "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p",
            args.movie
        ]
        print("Trying fallback ffmpeg command…")
        subprocess.run(cmd2)
    else:
        print(f"\n✓ Movie written to: {args.movie}")


# ──────────────────────────────────────────────────────────────────────────────
# 3-D interactive version (open in browser)
# ──────────────────────────────────────────────────────────────────────────────

def make_3d_snapshot(snap_entry, snap_num, center_pkpc, size_pkpc,
                     h, outpath="halo569_3d.html",
                     n_dm=50000, n_gas=20000, n_dust=5000):
    """
    Render an interactive 3-D scatter plot of one snapshot using plotly.
    Subsamples particles so the HTML stays manageable.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping 3-D render.")
        return

    _, hdr, _ = open_snap_header(snap_entry)
    a = float(hdr["Time"])
    z = 1.0 / a - 1.0

    def load_sub(ptype, nmax):
        pos = read_part_positions(snap_entry, ptype)
        if pos is None: return None
        pos = pos.astype(np.float64) * a / h
        dx = pos - center_pkcp
        mask = np.all(np.abs(dx) < size_pkpc, axis=1)
        pos = pos[mask]
        if len(pos) > nmax:
            idx = np.random.choice(len(pos), nmax, replace=False)
            pos = pos[idx]
        return pos

    center_pkcp = np.array(center_pkpc)

    traces = []
    dm  = load_sub(1, n_dm)
    gas = load_sub(0, n_gas)
    dst = load_sub(6, n_dust)

    for pts, name, color in [
        (dm,  "Dark Matter", "rgba(100,180,255,0.15)"),
        (gas, "Gas",         "rgba(255,120, 60,0.25)"),
        (dst, "Dust",        "rgba(255, 60,200,0.50)"),
    ]:
        if pts is None: continue
        traces.append(go.Scatter3d(
            x=pts[:,0]-center_pkcp[0],
            y=pts[:,1]-center_pkcp[1],
            z=pts[:,2]-center_pkcp[2],
            mode="markers",
            marker=dict(size=1.2, color=color, opacity=0.5),
            name=name,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"CosmicGrain — Halo 569  |  z = {z:.2f}",
        paper_bgcolor="black", plot_bgcolor="black",
        scene=dict(
            xaxis=dict(title="x [pkpc]", backgroundcolor="black",
                       gridcolor="#333", color="white"),
            yaxis=dict(title="y [pkpc]", backgroundcolor="black",
                       gridcolor="#333", color="white"),
            zaxis=dict(title="z [pkpc]", backgroundcolor="black",
                       gridcolor="#333", color="white"),
        ),
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    )
    fig.write_html(outpath)
    print(f"3-D interactive plot written to {outpath}")


if __name__ == "__main__":
    main()
