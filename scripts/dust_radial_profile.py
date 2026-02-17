#!/usr/bin/env python3
"""
dust_radial_profile.py

Compute radial distribution of dust (PartType6) in Gadget-4 HDF5 snapshots.
Handles split snapshots (snapshot_XXX.*.hdf5) inside snapdir_XXX folders.

Outputs:
- radial bin centers [kpc]
- dust count per bin
- dust mass per bin
- cumulative mass profile

Centering options:
- box: box center
- com: center of mass of chosen particle type (default PartType1 DM)
- shrinksphere: iterative shrinking sphere on chosen type (default PartType1 DM)
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py


# -----------------------------
# Utilities
# -----------------------------
def shell_stats(r: np.ndarray,
                mass: Optional[np.ndarray],
                edges: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute per-shell counts and (optionally) mass, for custom bin edges.
    edges: array of bin edges [kpc], length N+1.
    """
    counts, _ = np.histogram(r, bins=edges)
    out = {"counts": counts}

    if mass is not None:
        msum, _ = np.histogram(r, bins=edges, weights=mass)
        out["mass"] = msum
        out["mass_cum"] = np.cumsum(msum)
    return out


def print_shell_table(edges: np.ndarray,
                      stats: Dict[str, np.ndarray],
                      label: str = "dust") -> None:
    """
    Pretty terminal table.
    """
    counts = stats["counts"]
    has_mass = "mass" in stats
    mass = stats.get("mass", None)

    print("\n" + "="*72)
    print(f"[shells] {label} distribution")
    if has_mass:
        print(f"{'r_in-r_out [kpc]':>18}  {'N':>10}  {'Mass':>14}")
    else:
        print(f"{'r_in-r_out [kpc]':>18}  {'N':>10}")
    print("-"*72)

    for i in range(len(edges) - 1):
        rin = edges[i]
        rout = edges[i+1]
        if has_mass:
            print(f"{rin:7.1f}-{rout:7.1f}      {counts[i]:10d}  {mass[i]:14.6e}")
        else:
            print(f"{rin:7.1f}-{rout:7.1f}      {counts[i]:10d}")

    print("-"*72)
    print(f"{'TOTAL':>18}  {counts.sum():10d}")
    print("="*72 + "\n")

def make_png_plot(out_png: str,
                  r_centers: np.ndarray,
                  counts: np.ndarray,
                  mass_in_bin: Optional[np.ndarray],
                  mass_cum: Optional[np.ndarray],
                  rmax: float,
                  snapnum: int,
                  center_info: str) -> None:
    import matplotlib.pyplot as plt

    # Figure 1: histogram-like line for counts
    plt.figure(figsize=(8, 5))
    plt.plot(r_centers, counts, drawstyle="steps-mid")
    plt.xlabel("r [kpc]")
    plt.ylabel("Dust count per bin")
    plt.title(f"Dust radial distribution (PartType6) | snap {snapnum:03d}")
    plt.xlim(0, rmax)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_counts.png"), dpi=200)
    plt.close()

    # Figure 2: mass per bin + cumulative mass (if available)
    if mass_in_bin is not None and mass_cum is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(r_centers, mass_in_bin, drawstyle="steps-mid", label="mass per bin")
        plt.plot(r_centers, mass_cum, label="cumulative mass")
        plt.xlabel("r [kpc]")
        plt.ylabel("Dust mass")
        plt.title(f"Dust mass profile (PartType6) | snap {snapnum:03d}")
        plt.xlim(0, rmax)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", "_mass.png"), dpi=200)
        plt.close()

    # (Optional) a tiny text dump of center method in a separate .txt for bookkeeping
    with open(out_png.replace(".png", "_meta.txt"), "w") as f:
        f.write(f"snapnum: {snapnum}\n")
        f.write(f"rmax: {rmax}\n")
        f.write(f"center_info: {center_info}\n")


def list_snapshot_files(path: str, snapnum: int) -> List[str]:
    """
    Accepts either:
      - a snapdir_XXX directory
      - a path to a single snapshot file
      - a directory containing snapshots

    Returns a sorted list of HDF5 files belonging to snapshot snapnum.
    """
    path = os.path.abspath(path)

    # If they pass a file directly
    if os.path.isfile(path) and path.endswith(".hdf5"):
        return [path]

    # If they pass snapdir_XXX
    if os.path.isdir(path) and os.path.basename(path).startswith("snapdir_"):
        pat = os.path.join(path, f"snapshot_{snapnum:03d}.*.hdf5")
        files = sorted(glob.glob(pat))
        if not files:
            # Sometimes naming differs (rare)
            pat2 = os.path.join(path, f"snapshot_{snapnum:03d}.hdf5")
            files = sorted(glob.glob(pat2))
        return files

    # Otherwise: treat as a directory that contains snapdir_XXX
    if os.path.isdir(path):
        snapdir = os.path.join(path, f"snapdir_{snapnum:03d}")
        if os.path.isdir(snapdir):
            return list_snapshot_files(snapdir, snapnum)

        # Or flat files in this directory
        pat = os.path.join(path, f"snapshot_{snapnum:03d}.*.hdf5")
        files = sorted(glob.glob(pat))
        if not files:
            pat2 = os.path.join(path, f"snapshot_{snapnum:03d}.hdf5")
            files = sorted(glob.glob(pat2))
        return files

    raise FileNotFoundError(f"Could not interpret path: {path}")


def periodic_displacement(dx: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Minimal image convention for displacement dx in a periodic box [0, boxsize).
    """
    return dx - boxsize * np.round(dx / boxsize)


def read_header_units(first_file: str) -> Tuple[float, float]:
    """
    Return (BoxSize, HubbleParam) from Header of first file.
    """
    with h5py.File(first_file, "r") as f:
        hdr = f["Header"].attrs
        boxsize = float(hdr["BoxSize"])
        hubble = float(hdr.get("HubbleParam", 1.0))
    return boxsize, hubble


def concat_parttype_dataset(files: List[str], ptype: int, dset: str) -> Optional[np.ndarray]:
    """
    Concatenate a dataset across split snapshot files for a given PartType.
    Returns None if that PartType or dataset does not exist in any file.
    """
    key = f"PartType{ptype}/{dset}"
    chunks = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            if f"PartType{ptype}" not in f:
                continue
            grp = f[f"PartType{ptype}"]
            if dset not in grp:
                continue
            arr = grp[dset][...]
            if arr.size == 0:
                continue
            chunks.append(arr)

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


# -----------------------------
# Center finding
# -----------------------------

@dataclass
class CenterResult:
    center: np.ndarray  # shape (3,)
    info: str


def center_box(boxsize: float) -> CenterResult:
    c = np.array([0.5 * boxsize, 0.5 * boxsize, 0.5 * boxsize], dtype=np.float64)
    return CenterResult(c, "center=box (BoxSize/2)")


def center_com(pos: np.ndarray, mass: Optional[np.ndarray], boxsize: float,
               inner_fraction: float = 1.0) -> CenterResult:
    """
    Center-of-mass in a periodic box.
    Simple COM is tricky in periodic boundaries; we do a robust approach:
      - pick a reference particle
      - unwrap all positions relative to it using minimal image
      - compute COM in unwrapped space
      - wrap back into box

    inner_fraction: optionally restrict to the particles within the inner_fraction
    of the radius distribution around a rough center (box center). This can help
    ignore far-out junk.
    """
    if pos is None or len(pos) == 0:
        raise ValueError("No particles for COM center.")

    # Optional restriction based on rough radii from box center
    if inner_fraction < 1.0:
        rough_c = np.array([0.5 * boxsize]*3, dtype=np.float64)
        disp = periodic_displacement(pos - rough_c[None, :], boxsize)
        r = np.sqrt((disp*disp).sum(axis=1))
        cut = np.quantile(r, inner_fraction)
        sel = r <= cut
        pos = pos[sel]
        if mass is not None:
            mass = mass[sel]

    ref = pos[0].astype(np.float64)
    disp = periodic_displacement(pos - ref[None, :], boxsize)
    unwrapped = ref[None, :] + disp  # around ref, continuous

    if mass is None:
        c_unwrap = unwrapped.mean(axis=0)
    else:
        w = mass.astype(np.float64)
        c_unwrap = (unwrapped * w[:, None]).sum(axis=0) / w.sum()

    c = np.mod(c_unwrap, boxsize)
    return CenterResult(c, f"center=com (inner_fraction={inner_fraction})")


def center_shrinking_sphere(pos: np.ndarray,
                           mass: Optional[np.ndarray],
                           boxsize: float,
                           r0: Optional[float] = None,
                           shrink: float = 0.7,
                           nmin: int = 5000,
                           max_iter: int = 50) -> CenterResult:
    """
    Classic shrinking-sphere center finder:
      Start with an initial center (box center).
      Compute COM of particles within radius R.
      Shrink R by factor 'shrink' each iteration until particle count < nmin.

    Uses periodic wrapping properly by unwrapping around current center.
    """
    if pos is None or len(pos) == 0:
        raise ValueError("No particles for shrinking-sphere center.")

    c = np.array([0.5 * boxsize]*3, dtype=np.float64)

    # Choose initial radius: default to half the box diagonal-ish, or user-specified
    if r0 is None:
        r0 = 0.5 * boxsize
    R = float(r0)

    info_bits = [f"shrinksphere(r0={R:.3g}, shrink={shrink}, nmin={nmin}, max_iter={max_iter})"]

    for it in range(max_iter):
        disp = periodic_displacement(pos - c[None, :], boxsize)
        r = np.sqrt((disp*disp).sum(axis=1))
        sel = r <= R
        nsel = int(sel.sum())
        if nsel == 0:
            info_bits.append(f"iter={it}: nsel=0 (stopped)")
            break

        psel = pos[sel]
        if mass is not None:
            msel = mass[sel]
        else:
            msel = None

        # unwrap around current center for COM
        disp_sel = periodic_displacement(psel - c[None, :], boxsize)
        unwrapped = c[None, :] + disp_sel

        if msel is None:
            c_new = unwrapped.mean(axis=0)
        else:
            w = msel.astype(np.float64)
            c_new = (unwrapped * w[:, None]).sum(axis=0) / w.sum()

        c = np.mod(c_new, boxsize)

        info_bits.append(f"iter={it}: R={R:.3g} nsel={nsel}")

        if nsel < nmin:
            break

        R *= shrink

    return CenterResult(c, "center=" + " | ".join(info_bits))


# -----------------------------
# Main profile computation
# -----------------------------

def radial_profile(pos: np.ndarray,
                   mass: Optional[np.ndarray],
                   center: np.ndarray,
                   boxsize: float,
                   rmax: float,
                   nbins: int) -> Dict[str, np.ndarray]:
    disp = periodic_displacement(pos - center[None, :], boxsize)
    r = np.sqrt((disp*disp).sum(axis=1))

    sel = r <= rmax
    r = r[sel]
    if mass is not None:
        mass = mass[sel]

    bins = np.linspace(0.0, rmax, nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])

    counts, _ = np.histogram(r, bins=bins)

    if mass is None:
        mprof = np.zeros_like(rc)
        mcum = np.zeros_like(rc)
    else:
        mprof, _ = np.histogram(r, bins=bins, weights=mass)
        mcum = np.cumsum(mprof)

    return {
        "r_bin_edges": bins,
        "r_centers": rc,
        "counts": counts,
        "mass_in_bin": mprof,
        "mass_cum": mcum,
        "n_selected": np.array([len(r)]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="snapdir_XXX, a directory containing it, or a single snapshot file")
    ap.add_argument("snapnum", type=int, help="snapshot number, e.g. 17")
    ap.add_argument("--rmax", type=float, default=300.0, help="max radius in kpc (code units), default 300")
    ap.add_argument("--nbins", type=int, default=60, help="number of radial bins")
    ap.add_argument("--center", choices=["box", "com", "shrinksphere"], default="shrinksphere",
                    help="how to choose the center")
    ap.add_argument("--center-type", type=int, default=1,
                    help="particle type used for center finding (default 1=DM). Other: 0=gas, 4=stars")
    ap.add_argument("--inner-fraction", type=float, default=0.2,
                    help="for COM center: restrict to inner fraction of particles around box center (default 0.2). "
                         "Ignored for other methods.")
    ap.add_argument("--nmin", type=int, default=5000, help="shrinksphere: minimum particles to stop")
    ap.add_argument("--shrink", type=float, default=0.7, help="shrinksphere: shrink factor per iter")
    ap.add_argument("--out", default=None, help="output .npz filename (default: dust_profile_snapXXX.npz)")
    ap.add_argument("--png", default=None,
                    help="Base output png name (default: dust_profile_snapXXX.png). "
                         "Will actually write *_counts.png and *_mass.png")
    ap.add_argument("--shell-edges", default="0,5,10,20,50,100,200,300",
                    help="Comma-separated radial shell edges in kpc for terminal counts (default: 0,5,10,20,50,100,200,300)")

    args = ap.parse_args()

    files = list_snapshot_files(args.path, args.snapnum)
    if not files:
        raise FileNotFoundError("No snapshot files found.")
    print(f"[info] Found {len(files)} file(s) for snapshot {args.snapnum:03d}")
    for fn in files[:3]:
        print(f"  {fn}")
    if len(files) > 3:
        print("  ...")

    boxsize, hubble = read_header_units(files[0])
    print(f"[info] BoxSize={boxsize:.6g} (code length units, typically kpc) | HubbleParam={hubble}")

    # Read dust
    dust_pos = concat_parttype_dataset(files, 6, "Coordinates")
    if dust_pos is None:
        raise RuntimeError("No PartType6/Coordinates found (no dust in this snapshot?).")
    dust_mass = concat_parttype_dataset(files, 6, "Masses")  # might be None if constant mass in header

    print(f"[info] Dust particles: {len(dust_pos)}")
    if dust_mass is None:
        print("[info] Dust masses not found as dataset; profile will be in counts only.")

    # Read center-finding particles
    c_pos = concat_parttype_dataset(files, args.center_type, "Coordinates")
    c_mass = concat_parttype_dataset(files, args.center_type, "Masses")

    if args.center == "box":
        c_res = center_box(boxsize)
    elif args.center == "com":
        if c_pos is None:
            raise RuntimeError(f"No PartType{args.center_type}/Coordinates for COM centering.")
        c_res = center_com(c_pos, c_mass, boxsize, inner_fraction=args.inner_fraction)
    else:
        if c_pos is None:
            raise RuntimeError(f"No PartType{args.center_type}/Coordinates for shrinking-sphere centering.")
        c_res = center_shrinking_sphere(
            c_pos, c_mass, boxsize,
            r0=0.5 * boxsize,
            shrink=args.shrink,
            nmin=args.nmin,
            max_iter=60
        )

    print(f"[info] {c_res.info}")
    print(f"[info] Center = ({c_res.center[0]:.6g}, {c_res.center[1]:.6g}, {c_res.center[2]:.6g})")

    prof = radial_profile(dust_pos, dust_mass, c_res.center, boxsize, args.rmax, args.nbins)

    # Compute radii for *all* dust (within rmax) so we can shell-print
    disp = periodic_displacement(dust_pos - c_res.center[None, :], boxsize)
    r_all = np.sqrt((disp*disp).sum(axis=1))

    # shell edges for printout
    shell_edges = np.array([float(x) for x in args.shell_edges.split(",")], dtype=np.float64)
    shell_edges = np.unique(shell_edges)
    shell_edges = shell_edges[shell_edges >= 0.0]
    if shell_edges[0] != 0.0:
        shell_edges = np.insert(shell_edges, 0, 0.0)

    # Restrict shell stats to within max(shell_edges) to match the table
    r_shell_max = shell_edges[-1]
    sel_shell = r_all <= r_shell_max
    r_shell = r_all[sel_shell]
    m_shell = dust_mass[sel_shell] if dust_mass is not None else None

    stats = shell_stats(r_shell, m_shell, shell_edges)
    print_shell_table(shell_edges, stats, label="dust (PartType6)")

    # Make PNG plots
    png_base = args.png or f"dust_profile_snap{args.snapnum:03d}.png"
    make_png_plot(
        png_base,
        prof["r_centers"],
        prof["counts"],
        None if dust_mass is None else prof["mass_in_bin"],
        None if dust_mass is None else prof["mass_cum"],
        args.rmax,
        args.snapnum,
        c_res.info
    )
    print(f"[done] Wrote {png_base.replace('.png','_counts.png')}")
    if dust_mass is not None:
        print(f"[done] Wrote {png_base.replace('.png','_mass.png')}")

    out = args.out or f"dust_profile_snap{args.snapnum:03d}.npz"
    np.savez(
        out,
        snapnum=args.snapnum,
        boxsize=boxsize,
        center=c_res.center,
        center_info=c_res.info,
        rmax=args.rmax,
        nbins=args.nbins,
        **prof
    )
    print(f"[done] Wrote {out}")
    print(f"[done] Selected dust within r<= {args.rmax} kpc: {int(prof['n_selected'][0])}")


if __name__ == "__main__":
    main()
