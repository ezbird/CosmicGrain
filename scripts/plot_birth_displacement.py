#!/usr/bin/env python3
"""
plot_birth_displacement.py

Measures how far dust particles travel from their birth position
before being destroyed, accreted, or surviving to the observed epoch.

Three figures:
  1. CDF of displacement at z=0  (ISM / CGM / all)
  2. Median displacement ± 1σ vs redshift  (ISM / CGM)
  3. 2D hexbin: displacement vs galactocentric radius at z=0

Optional 4th output: event-log statistics for destroyed/accreted particles
printed to stdout (no position column required — only if logs have PosX..Z).

Usage:
  python plot_birth_displacement.py
  python plot_birth_displacement.py --snap-dir /path/to/output --snaps 30 40 50
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration (edit or override via CLI) ───────────────────────────────────

SNAP_DIR        = Path("/scratch/user/CosmicGrain/output")
GROUP_DIR       = Path("/scratch/user/CosmicGrain/output")
LOG_DIR         = None   # set via --log-dir or defaults to snap-dir
OUT_DIR         = Path(".")

# Snapshot numbers to analyse — last entry treated as z=0
SNAP_NUMBERS    = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# ISM/CGM boundary (physical kpc) — consistent with stellar_mass_profile.py
ISM_RADIUS_PKPC = 20.0

# ── Plotting style (project conventions) ──────────────────────────────────────

C_ISM  = "#2a9d8f"   # teal  — primary simulation colour
C_CGM  = "#e76f51"   # orange
C_ALL  = "#264653"   # dark slate

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.facecolor"   : "white",
    "figure.facecolor" : "white",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── Path helpers ───────────────────────────────────────────────────────────────

def snap_path(n):
    # Multi-file layout: snapdir_NNN/snapshot_NNN.*.hdf5
    snap_dir = SNAP_DIR / f"snapdir_{n:03d}"
    chunks   = sorted(snap_dir.glob(f"snapshot_{n:03d}.*.hdf5"))
    return chunks   # returns a list; empty list = not found

def group_path(n):
    """
    Try several Gadget-4 group-catalog layouts, including multi-file.
    Returns path to the first file found (chunk 0 for multi-file), or None.
    """
    candidates = [
        GROUP_DIR / f"fof_subhalo_tab_{n:03d}.hdf5",
        GROUP_DIR / f"groups_{n:03d}" / f"fof_subhalo_tab_{n:03d}.hdf5",
        GROUP_DIR / f"groups_{n:03d}" / f"fof_subhalo_tab_{n:03d}.0.hdf5",
        GROUP_DIR / f"fof_subhalo_tab_{n:03d}.0.hdf5",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ── Data loading ───────────────────────────────────────────────────────────────

def load_halo_center_com(n):
    """
    Return halo-0 center in comoving kpc/h.
    Prefers SubhaloPos[0] (most-bound particle); falls back to GroupPos[0].
    Returns None if catalog not available.
    """
    gp = group_path(n)
    if gp is None:
        return None
    with h5py.File(gp, "r") as f:
        if "Subhalo/SubhaloPos" in f and len(f["Subhalo/SubhaloPos"]) > 0:
            return f["Subhalo/SubhaloPos"][0].copy()
        if "Group/GroupPos" in f and len(f["Group/GroupPos"]) > 0:
            return f["Group/GroupPos"][0].copy()
    return None


def load_dust(n):
    """
    Load PartType6 Coordinates and BPOS from a multi-file snapshot.
    Concatenates across all chunks (snapshot_NNN.0.hdf5, .1, .2, ...).
    Returns dict or None if no dust present anywhere.
    """
    chunks = snap_path(n)   # list of Path objects
    if not chunks:
        print(f"  [skip] snapdir_{n:03d} not found or empty")
        return None

    # Header scalars from chunk 0
    with h5py.File(chunks[0], "r") as f:
        h       = float(f["Parameters"].attrs["HubbleParam"])
        a       = float(f["Header"].attrs["Time"])
        z       = float(f["Header"].attrs["Redshift"])
        boxsize = float(f["Header"].attrs["BoxSize"])   # comoving kpc/h

    # Auto-detect birth-position field name
    bpos_key = None
    with h5py.File(chunks[0], "r") as _f:
        pt6 = _f.get("PartType6", {})
        for candidate in ["BPOS", "BirthPos", "BirthPosition", "InitialCoordinates"]:
            if candidate in pt6:
                bpos_key = candidate
                break
        if bpos_key is None:
            available = list(pt6.keys()) if pt6 else []
            print(f"\n  ERROR: birth-position field not found in PartType6.")
            print(f"  Available PartType6 fields in chunk 0: {available}")
            print(f"  Add the correct key to the candidate list above.")
            return None

    # Accumulate PartType6 arrays across all chunks
    pos_list  = []
    bpos_list = []

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            npart6 = int(f["Header"].attrs["NumPart_ThisFile"][6])
            if npart6 == 0 or "PartType6" not in f:
                continue
            pos_list.append(f["PartType6/Coordinates"][:])
            bpos_list.append(f["PartType6"][bpos_key][:])

    if not pos_list:
        print(f"  [skip] snap {n:03d} (z={z:.2f}): no PartType6 in any chunk")
        return None

    pos  = np.concatenate(pos_list,  axis=0)
    bpos = np.concatenate(bpos_list, axis=0)

    return dict(pos=pos, bpos=bpos, a=a, z=z, h=h, boxsize=boxsize, n=n)

# ── Core computation ───────────────────────────────────────────────────────────

def compute_displacement(data, center_com=None):
    """
    Physical displacement from birth position (pkpc) for each particle.

    dr_phys = (pos_com - bpos_com) * a / h

    Periodic wrapping is applied before conversion. This gives the
    displacement at the *current epoch*, not the integrated path length —
    appropriate for asking "how far from its birthplace is this grain now?"

    Parameters
    ----------
    data : dict from load_dust()
    center_com : array (3,) halo center in comoving kpc/h, or None

    Returns
    -------
    displacement : (N,) physical kpc
    r_gal        : (N,) physical kpc galactocentric radius, or None
    """
    pos  = data["pos"]
    bpos = data["bpos"]
    a    = data["a"]
    h    = data["h"]
    box  = data["boxsize"]

    # Birth-displacement vector, periodic wrap
    dr = pos - bpos
    dr -= box * np.round(dr / box)
    dr_phys      = dr * a / h                          # physical kpc
    displacement = np.linalg.norm(dr_phys, axis=1)

    r_gal = None
    if center_com is not None:
        dp = pos - center_com
        dp -= box * np.round(dp / box)
        r_gal = np.linalg.norm(dp * a / h, axis=1)    # physical kpc

    return displacement, r_gal


def _stats(arr):
    """Median and 16th/84th percentile; NaN-safe."""
    if len(arr) == 0:
        return dict(median=np.nan, p16=np.nan, p84=np.nan, n=0)
    return dict(
        median = float(np.median(arr)),
        p16    = float(np.percentile(arr, 16)),
        p84    = float(np.percentile(arr, 84)),
        n      = len(arr),
    )

# ── Snapshot sweep ─────────────────────────────────────────────────────────────

def run_snapshot_analysis(snap_numbers):
    """
    Sweep over snapshots. Returns list of per-snapshot result dicts.
    The final snapshot's raw displacement arrays are retained for z=0 plots.
    """
    results = []
    last_loaded = None   # will track the last snap that actually loaded

    for n in snap_numbers:
        print(f"  snap {n:03d} ...", end=" ", flush=True)
        data = load_dust(n)
        if data is None:
            continue

        center = load_halo_center_com(n)
        if center is None:
            print(f"(no group catalog — galactocentric radii unavailable)", end=" ")

        disp, r_gal = compute_displacement(data, center_com=center)

        if r_gal is not None:
            ism_mask = r_gal < ISM_RADIUS_PKPC
        else:
            # Can't split without halo center; treat everything as ISM
            ism_mask = np.ones(len(disp), dtype=bool)

        cgm_mask = ~ism_mask

        entry = dict(
            n        = n,
            z        = data["z"],
            a        = data["a"],
            all      = _stats(disp),
            ism      = _stats(disp[ism_mask]),
            cgm      = _stats(disp[cgm_mask]),
            # Raw arrays kept — we overwrite each time so the last loaded snap wins
            disp_z0     = disp,
            ism_mask_z0 = ism_mask,
            r_gal_z0    = r_gal,
        )
        results.append(entry)

        print(f"z={data['z']:.2f}  N={len(disp):,}  "
              f"ISM median={entry['ism']['median']:.1f} pkpc  "
              f"CGM median={entry['cgm']['median']:.1f} pkpc")

    return results

# ── Figures ────────────────────────────────────────────────────────────────────

def fig_cdf_z0(results, out_dir):
    """Figure 1: CDF of birth displacement at z≈0, split ISM / CGM / all."""
    z0 = next((r for r in reversed(results) if r["disp_z0"] is not None), None)
    if z0 is None:
        print("  [skip] no z=0 displacement data for CDF")
        return

    disp     = z0["disp_z0"]
    ism_mask = z0["ism_mask_z0"]
    cgm_mask = ~ism_mask

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    populations = [
        (disp,           "All",  C_ALL, "-",  1.8),
        (disp[ism_mask], "ISM",  C_ISM, "--", 2.0),
        (disp[cgm_mask], "CGM",  C_CGM, ":",  2.0),
    ]

    for arr, label, color, ls, lw in populations:
        if len(arr) == 0:
            continue
        x = np.sort(arr)
        y = np.linspace(0, 1, len(x))
        ax.plot(x, y, color=color, ls=ls, lw=lw,
                label=rf"{label}  ($N={len(arr):,}$)")

    # Annotate medians with vertical lines + text
    text_y = {"ISM": 0.52, "CGM": 0.38}
    for arr, label, color in [(disp[ism_mask], "ISM", C_ISM),
                               (disp[cgm_mask], "CGM", C_CGM)]:
        if len(arr) == 0:
            continue
        med = np.median(arr)
        ax.axvline(med, color=color, lw=1.0, alpha=0.55, ls="--")
        ax.text(med * 1.15, text_y[label], f"{med:.0f} pkpc",
                color=color, fontsize=9, va="center")

    ax.set_xscale("log")
    ax.set_xlim(left=0.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Displacement from birth position (pkpc)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(r"Birth displacement, $z \approx 0$")
    ax.legend(framealpha=0.9, fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))

    fig.tight_layout()
    out = out_dir / "birth_displacement_cdf_z0.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


def fig_median_vs_z(results, out_dir):
    """Figure 2: Median ± 1σ birth displacement vs redshift."""
    if not results:
        return

    zs      = np.array([r["z"]              for r in results])
    ism_med = np.array([r["ism"]["median"]  for r in results])
    ism_lo  = np.array([r["ism"]["p16"]     for r in results])
    ism_hi  = np.array([r["ism"]["p84"]     for r in results])
    cgm_med = np.array([r["cgm"]["median"]  for r in results])
    cgm_lo  = np.array([r["cgm"]["p16"]     for r in results])
    cgm_hi  = np.array([r["cgm"]["p84"]     for r in results])

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    for med, lo, hi, label, color in [
        (ism_med, ism_lo, ism_hi, "ISM", C_ISM),
        (cgm_med, cgm_lo, cgm_hi, "CGM", C_CGM),
    ]:
        mask = np.isfinite(med)
        if mask.sum() == 0:
            continue
        ax.fill_between(zs[mask], lo[mask], hi[mask],
                        alpha=0.15, color=color)
        ax.plot(zs[mask], med[mask], color=color, lw=2, label=label)

    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Median birth displacement (pkpc)")
    ax.set_title("Birth displacement vs redshift")
    ax.legend(framealpha=0.9, fontsize=10)

    # Right-axis: label ISM/CGM directly at z=0
    for med, color, label in [(ism_med, C_ISM, "ISM"),
                               (cgm_med, C_CGM, "CGM")]:
        if np.isfinite(med[-1]):
            ax.text(zs[-1] - 0.05, med[-1], label,
                    color=color, fontsize=10, va="center", ha="right")

    fig.tight_layout()
    out = out_dir / "birth_displacement_vs_z.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


def fig_displacement_vs_rgal(results, out_dir):
    """Figure 3: 2D hexbin of displacement vs galactocentric radius at z≈0."""
    z0 = next((r for r in reversed(results) if r["disp_z0"] is not None), None)
    if z0 is None or z0["r_gal_z0"] is None:
        print("  [skip] no galactocentric radius data for hexbin")
        return

    disp  = z0["disp_z0"]
    r_gal = z0["r_gal_z0"]

    # Clip zeros before log-scale hexbin
    r_gal = np.clip(r_gal, 0.01, None)
    disp  = np.clip(disp,  0.01, None)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    hb = ax.hexbin(r_gal, disp, gridsize=60,
                   cmap="YlOrBr", xscale="log", yscale="log",
                   mincnt=1, linewidths=0.2)

    # ISM/CGM boundary
    ax.axvline(ISM_RADIUS_PKPC, color="gray", lw=1.2, ls="--", alpha=0.7)
    ylims = ax.get_ylim()
    ax.text(ISM_RADIUS_PKPC * 0.65, ylims[0] * 1.8, "ISM",
            color="gray", fontsize=9, ha="right")
    ax.text(ISM_RADIUS_PKPC * 1.35, ylims[0] * 1.8, "CGM",
            color="gray", fontsize=9, ha="left")

    # 1:1 reference line (displacement = r_gal; born at halo centre)
    lim = np.array([max(r_gal.min(), disp.min()),
                    min(r_gal.max(), disp.max())])
    ax.plot(lim, lim, color="white", lw=1.0, ls=":", alpha=0.6,
            label="disp = $r_{\\rm gal}$")

    ax.set_xlabel(r"Galactocentric radius, $r$ (pkpc)")
    ax.set_ylabel("Displacement from birth position (pkpc)")
    ax.set_title(r"Birth displacement vs location, $z \approx 0$")

    # Inline colorbar label
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Particle count", fontsize=9)

    fig.tight_layout()
    out = out_dir / "birth_displacement_vs_rgal.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)

# ── Event-log analysis ─────────────────────────────────────────────────────────

def analyse_event_logs(log_files, h_cosmo):
    """
    Parse dust_log_task*.txt files.

    Confirmed column layout (16 cols):
      0   ParticleID
      1   a_birth        (scale factor at spawn)
      2   a_event        (scale factor at this event)
      3-5 PosX/Y/Z       (comoving kpc/h at event)
      6-8 BirthPosX/Y/Z  (comoving kpc/h)
      9   |displacement|  (comoving kpc/h, pre-computed)
      10  grain radius
      11  carbon fraction
      12  temperature/density
      13  rate/mass
      14  event_type flag
      15  event_type flag

    Physical displacement = col9 * a_event / h
    """
    print(f"  Parsing {len(log_files)} log files ...")

    # Accumulate across all task files
    disp_com_all = []
    a_event_all  = []
    etype14_all  = []
    etype15_all  = []

    for lf in log_files:
        try:
            raw = np.loadtxt(lf, comments="#")
        except Exception as e:
            print(f"    Warning: {lf.name}: {e}")
            continue
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        if len(raw) == 0:
            continue
        disp_com_all.append(raw[:, 9])
        a_event_all.append(raw[:, 2])
        etype14_all.append(raw[:, 14].astype(int))
        etype15_all.append(raw[:, 15].astype(int))

    if not disp_com_all:
        print("  No data parsed from log files.")
        return

    disp_com = np.concatenate(disp_com_all)
    a_event  = np.concatenate(a_event_all)
    etype14  = np.concatenate(etype14_all)
    etype15  = np.concatenate(etype15_all)

    # Physical displacement at event epoch
    disp_phys = disp_com * a_event / h_cosmo

    print(f"  Total events: {len(disp_phys):,}")
    print(f"  Unique col-14 values: {np.unique(etype14)}")
    print(f"  Unique col-15 values: {np.unique(etype15)}")
    print()

    # Overall distribution (all events, regardless of type)
    print(f"  {'Population':<20} {'N':>9}  {'p16':>8}  {'median':>8}  {'p84':>8}  (pkpc)")
    print(f"  {'-'*62}")
    print(f"  {'All events':<20} {len(disp_phys):>9,}  "
          f"{np.percentile(disp_phys,16):>8.1f}  "
          f"{np.median(disp_phys):>8.1f}  "
          f"{np.percentile(disp_phys,84):>8.1f}")

    # Break down by col-14 event type
    for val in np.unique(etype14):
        mask = etype14 == val
        d = disp_phys[mask]
        print(f"  col14={val:<15} {mask.sum():>9,}  "
              f"{np.percentile(d,16):>8.1f}  "
              f"{np.median(d):>8.1f}  "
              f"{np.percentile(d,84):>8.1f}")

    # Redshift bins: high-z (a < 0.33, z>2), mid (0.33-0.67), low (a>0.67)
    print()
    print(f"  {'Epoch':<20} {'N':>9}  {'median disp':>12}  (pkpc physical)")
    print(f"  {'-'*48}")
    for label, lo, hi in [("z > 2  (a<0.33)",  0.0,  0.333),
                           ("1<z<2  (0.33-0.5)", 0.333, 0.500),
                           ("z < 1  (a>0.5)",   0.500, 1.001)]:
        mask = (a_event >= lo) & (a_event < hi)
        if mask.sum() == 0:
            continue
        d = disp_phys[mask]
        print(f"  {label:<20} {mask.sum():>9,}  {np.median(d):>12.1f}")

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap-dir",  default=str(SNAP_DIR))
    p.add_argument("--group-dir", default=None,
                   help="Group catalog root (default: same as --snap-dir)")
    p.add_argument("--log-dir",   default=None,
                   help="Directory containing dust_log_task*.txt (default: <snap-dir>/dust_logs)")
    p.add_argument("--out-dir",   default=str(OUT_DIR))
    p.add_argument("--snaps", nargs="+", type=int, default=SNAP_NUMBERS,
                   metavar="N",
                   help="Snapshot numbers to process (last = z≈0)")
    p.add_argument("--ism-radius", type=float, default=ISM_RADIUS_PKPC,
                   metavar="PKPC",
                   help="ISM/CGM boundary in physical kpc (default 20)")
    return p.parse_args()


def main():
    args = parse_args()

    global SNAP_DIR, GROUP_DIR, LOG_DIR, OUT_DIR, ISM_RADIUS_PKPC
    SNAP_DIR        = Path(args.snap_dir)
    GROUP_DIR       = Path(args.group_dir) if args.group_dir else Path(args.snap_dir)
    LOG_DIR         = Path(args.log_dir) if args.log_dir else Path(args.snap_dir) / "dust_logs"
    OUT_DIR         = Path(args.out_dir)
    ISM_RADIUS_PKPC = args.ism_radius

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap_numbers = sorted(args.snaps)

    print("=" * 60)
    print("Birth displacement analysis — CosmicGrain")
    print(f"  Snapshots : {snap_numbers}")
    print(f"  ISM/CGM boundary : {ISM_RADIUS_PKPC} pkpc")
    print("=" * 60)

    print("\n── Snapshot sweep ──────────────────────────────────────────")
    results = run_snapshot_analysis(snap_numbers)

    if not results:
        print("No valid snapshots found. Check SNAP_DIR and snap numbers.")
        sys.exit(1)

    print("\n── Figures ─────────────────────────────────────────────────")
    fig_cdf_z0(results, OUT_DIR)
    fig_median_vs_z(results, OUT_DIR)
    fig_displacement_vs_rgal(results, OUT_DIR)

    # Event-log analysis (optional; requires position columns in log)
    if LOG_DIR.exists():
        log_files = sorted(LOG_DIR.glob("dust_log_task*.txt"))
        if log_files:
            print("\n── Event log analysis (destroyed/accreted particles) ───────")
            h_cosmo = None
            for n in reversed(snap_numbers):
                chunks = snap_path(n)
                if chunks:
                    with h5py.File(chunks[0], "r") as f:
                        h_cosmo = float(f["Parameters"].attrs["HubbleParam"])
                    break
            if h_cosmo is not None:
                analyse_event_logs(log_files, h_cosmo)
        else:
            print(f"\n  No dust_log_task*.txt files found in {LOG_DIR}")
    else:
        print(f"\n  Log dir {LOG_DIR} not found; skipping event log analysis.")

    print("\nDone.")


if __name__ == "__main__":
    main()
