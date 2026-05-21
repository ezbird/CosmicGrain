#!/usr/bin/env python3
"""
plot_mdust_mstar.py
-------------------
Plot the total dust mass vs. stellar mass evolutionary track for Halo 569,
overlaid on observational data compiled in Osman et al. 2025 (DUSTY-GAEA,
arXiv:2512.15902) Figure 2.

Both M_dust and M_star are measured within R_200 at each snapshot.

Usage:
    python plot_mdust_mstar.py /path/to/output/ [options]

    python plot_mdust_mstar.py ../5_output_zoom_1024_halo569_50Mpc_dust/ \\
        --output mdust_mstar_1024.png

    # Multiple resolutions on one plot:
    python plot_mdust_mstar.py \\
        ../5_output_zoom_512_halo569_50Mpc_dust/ \\
        ../5_output_zoom_1024_halo569_50Mpc_dust/ \\
        --labels "512^3" "1024^3" \\
        --output mdust_mstar_convergence.png
"""

import sys
import os
import glob
import re
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

try:
    from halo_utils import load_target_halo, extract_dust_spatially
    HALO_UTILS_AVAILABLE = True
except ImportError:
    HALO_UTILS_AVAILABLE = False
    print("Warning: halo_utils not found — will fall back to density centroid.")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_MASS = 1.989e33          # g
PC_IN_CM   = 3.0857e18         # cm per parsec

# ---------------------------------------------------------------------------
# Observational data loader
# ---------------------------------------------------------------------------

def load_obs_data(npz_path):
    """
    Load real observational data from the .npz produced by parse_obs_data.py.
    Falls back to empty arrays with a warning if the file is missing.

    Expected keys in the .npz:
        galliano2021_mstar / galliano2021_mdust   — 784 DustPedia galaxies, z~0
        remyruyer2015_mstar / remyruyer2015_mdust — 109 DGS+KINGFISH galaxies, z~0
        dustpedia_cigale_mstar / dustpedia_cigale_mdust — 815 galaxies, z~0
    """
    if not os.path.exists(npz_path):
        print(f"WARNING: obs data not found at {npz_path}")
        print("         Run:  python parse_obs_data.py obs_data/")
        print("         Using empty arrays — observational points will be absent.")
        empty = np.array([])
        return {k: empty for k in (
            "galliano2021_mstar", "galliano2021_mdust",
            "remyruyer2015_mstar", "remyruyer2015_mdust",
            "dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
        )}
    obs = np.load(npz_path)
    print(f"Loaded obs data from {npz_path}  keys: {list(obs.keys())}")
    return dict(obs)

# ---------------------------------------------------------------------------

def find_snapshots(output_dir):
    """
    Return sorted list of (snap_num, snapdir_path, groups_path) tuples.
    Skips entries where no groups catalog exists.
    """
    output_dir = Path(output_dir)
    entries = []

    snapdirs = sorted(output_dir.glob("snapdir_*"))
    for snapdir in snapdirs:
        m = re.search(r"snapdir_(\d+)", snapdir.name)
        if not m:
            continue
        snap_num = int(m.group(1))

        # Locate a groups catalog
        groups_dir = output_dir / f"groups_{snap_num:03d}"
        catalog_files = sorted(groups_dir.glob("fof_subhalo_tab_*.hdf5")) if groups_dir.exists() else []
        if not catalog_files:
            continue   # no subfind catalog → skip

        # Locate first HDF5 chunk of the snapshot
        snap_files = sorted(snapdir.glob("snap_*.hdf5")) + sorted(snapdir.glob("snapshot_*.hdf5"))
        if not snap_files:
            continue

        entries.append((snap_num, snapdir, str(snap_files[0]), str(catalog_files[0])))

    return entries


def get_header(snap_file):
    """Return (redshift, h, BoxSize_kpc) from an HDF5 snapshot header."""
    with h5py.File(snap_file, "r") as f:
        hdr = f["Header"].attrs
        params = f["Parameters"].attrs
        z    = float(hdr["Redshift"])
        h    = float(params["HubbleParam"])
        box  = float(hdr["BoxSize"])   # comoving kpc/h
    return z, h, box


def get_unit_mass(snap_file):
    """Return UnitMass_in_g from snapshot (fall back to Gadget default)."""
    with h5py.File(snap_file, "r") as f:
        params = f.get("Parameters") or f.get("Config") or {}
        um = None
        if params:
            um = params.attrs.get("UnitMass_in_g", None)
        if um is None:
            um = 1.989e43   # 10^10 M_sun in grams (Gadget default)
    return float(um)


def load_particles_within_r200(snap_file_first, halo_center_kph, r200_kph,
                                part_types=(4, 6)):
    """
    Load all particles of requested PartTypes within r200 (comoving kpc/h).
    Returns dict {ptype: {'mass': array_in_code_units}}.

    Handles multi-chunk snapshots by globbing sibling files.
    """
    # Collect all chunks
    p = Path(snap_file_first)
    chunks = sorted(p.parent.glob(p.name.split(".")[0].rstrip("0123456789") + "*.hdf5"))
    if not chunks:
        chunks = [p]

    result = {pt: [] for pt in part_types}

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            for pt in part_types:
                key = f"PartType{pt}"
                if key not in f:
                    continue
                coords = f[key]["Coordinates"][:]   # comoving kpc/h
                mass_key = "Masses" if "Masses" in f[key] else None

                # Dust particles always have explicit Masses
                # Stars may use a mass table (check header)
                if mass_key is None:
                    hdr = f["Header"].attrs
                    mass_table = hdr.get("MassTable", None)
                    if mass_table is not None and mass_table[pt] > 0:
                        n = len(coords)
                        masses = np.full(n, mass_table[pt])
                    else:
                        continue
                else:
                    masses = f[key][mass_key][:]

                # Periodic distance check
                box = float(f["Header"].attrs["BoxSize"])
                dx = coords - halo_center_kph
                dx = dx - box * np.round(dx / box)
                r  = np.sqrt(np.sum(dx**2, axis=1))

                mask = r <= r200_kph
                result[pt].append(masses[mask])

    return {pt: np.concatenate(result[pt]) if result[pt] else np.array([])
            for pt in part_types}


def extract_r200_from_catalog(catalog_file):
    """
    Return (halo_center_kph, r200_kph, m200_code) from the primary SubFind group.

    Handles multi-chunk catalogs (fof_subhalo_tab_NNN.0.hdf5, .1.hdf5, …) by
    globbing siblings.  Returns None if no groups exist yet (early snapshots).
    """
    p = Path(catalog_file)
    # Glob all chunks: strip trailing digits from stem to get base name
    stem_base = re.sub(r"\.\d+$", "", p.stem)   # e.g. fof_subhalo_tab_000
    chunks = sorted(p.parent.glob(f"{stem_base}*.hdf5"))
    if not chunks:
        chunks = [p]

    pos_list  = []
    r200_list = []
    m200_list = []

    for chunk in chunks:
        with h5py.File(chunk, "r") as f:
            if "Group" not in f:
                continue
            grp = f["Group"]
            if "GroupPos" not in grp or len(grp["GroupPos"]) == 0:
                continue
            pos_list.append(grp["GroupPos"][:])
            r200_list.append(grp["Group_R_Crit200"][:])
            m200_list.append(grp["Group_M_Crit200"][:])

    if not pos_list:
        return None   # no groups yet (high-z snapshot)

    all_pos  = np.concatenate(pos_list,  axis=0)
    all_r200 = np.concatenate(r200_list, axis=0)
    all_m200 = np.concatenate(m200_list, axis=0)

    # Primary group = most massive (index 0 after Gadget sorts by mass desc)
    return all_pos[0], float(all_r200[0]), float(all_m200[0])


def process_snapshot(snap_num, snapdir, snap_file, catalog_file, unit_mass_g=None):
    """
    For one snapshot return (z, log_mstar_msun, log_mdust_msun) or None.
    """
    z, h, box = get_header(snap_file)
    if unit_mass_g is None:
        unit_mass_g = get_unit_mass(snap_file)

    # Code unit → solar masses conversion
    code_to_msun = unit_mass_g / SOLAR_MASS   # typically 1e10 / h

    # Halo center and R200 — read directly from subfind Group catalog.
    # halo_utils requires Subhalo which is absent at high-z or in FOF-only
    # catalogs, so we bypass it here and use the Group table directly.
    result = extract_r200_from_catalog(catalog_file)
    if result is None:
        return None   # no groups yet at this redshift — skip silently
    halo_center, r200, _ = result
    if r200 <= 0:
        return None

    # Load masses within R200 for stars (pt=4) and dust (pt=6)
    particles = load_particles_within_r200(snap_file, halo_center, r200,
                                           part_types=(4, 6))

    m_star_code = particles[4].sum() if len(particles[4]) else 0.0
    m_dust_code = particles[6].sum() if len(particles[6]) else 0.0

    m_star_msun = m_star_code * code_to_msun
    m_dust_msun = m_dust_code * code_to_msun

    if m_star_msun <= 0 or m_dust_msun <= 0:
        return None

    print(f"  snap {snap_num:03d}  z={z:.3f}  "
          f"log(M*/Msun)={np.log10(m_star_msun):.2f}  "
          f"log(Md/Msun)={np.log10(m_dust_msun):.2f}  "
          f"R200={r200:.1f} kpc/h")

    return z, np.log10(m_star_msun), np.log10(m_dust_msun)


def run_simulation(output_dir, label, color, skip_every=1):
    """
    Iterate over all snapshots with subfind catalogs and return arrays
    (z, log_mstar, log_mdust), sorted by descending z.
    """
    snapshots = find_snapshots(output_dir)
    if not snapshots:
        raise RuntimeError(f"No snapshots with subfind catalogs found in {output_dir}")

    print(f"\n[{label}] Found {len(snapshots)} snapshots with catalogs")

    # Read unit_mass once from the first available snapshot
    unit_mass_g = get_unit_mass(snapshots[0][2])

    results = []
    for i, (snap_num, snapdir, snap_file, catalog_file) in enumerate(snapshots):
        if i % skip_every != 0:
            continue
        r = process_snapshot(snap_num, snapdir, snap_file, catalog_file, unit_mass_g)
        if r is not None:
            results.append(r)

    if not results:
        raise RuntimeError(f"No valid snapshots processed for {label}")

    results = sorted(results, key=lambda x: -x[0])   # high-z first
    z_arr      = np.array([r[0] for r in results])
    mstar_arr  = np.array([r[1] for r in results])
    mdust_arr  = np.array([r[2] for r in results])
    return z_arr, mstar_arr, mdust_arr


def make_plot(sim_tracks, output_path, obs):
    """
    sim_tracks: list of (label, color, z_arr, mstar_arr, mdust_arr)
    obs:        dict from load_obs_data()
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # ------------------------------------------------------------------
    # Observational data  (real data from parse_obs_data.py)
    # ------------------------------------------------------------------
    def _scatter(key_ms, key_md, marker, color, label, size=12, alpha=0.6):
        ms = obs.get(key_ms, np.array([]))
        md = obs.get(key_md, np.array([]))
        if len(ms) > 0:
            ax.scatter(ms, md, s=size, marker=marker, color=color,
                       alpha=alpha, zorder=1, label=label, linewidths=0)

    # Galliano+2021: best single z~0 dataset (784 DustPedia galaxies,
    # hierarchical Bayesian SED, J/A+A/649/A18)
    _scatter("galliano2021_mstar", "galliano2021_mdust",
             "o", "0.55", "Galliano et al 2021 (DustPedia; z~0)", size=10, alpha=0.5)

    # Remy-Ruyer+2015: DGS+KINGFISH (109 galaxies, covers dwarf end,
    # J/A+A/582/A121)
    _scatter("remyruyer2015_mstar", "remyruyer2015_mdust",
             "s", "0.30", "Rémy-Ruyer et al 2015 (DGS+KINGFISH; z~0)", size=14, alpha=0.7)

    # DustPedia CIGALE (Nersesian+2019, 815 galaxies)
    # Only shown if Galliano+21 not available (same sample, older dust masses)
    if len(obs.get("galliano2021_mstar", [])) == 0:
        _scatter("dustpedia_cigale_mstar", "dustpedia_cigale_mdust",
                 "^", "0.55", "DustPedia CIGALE (z~0)", size=10, alpha=0.5)

    # ------------------------------------------------------------------
    # Simulation tracks
    # ------------------------------------------------------------------
    # Redshift label positions: annotate a few z milestones along each track
    Z_LABELS = [6.0, 4.5, 3.5, 2.5, 1.5, 0.5, 0.0]

    for (label, color, z_arr, mstar_arr, mdust_arr) in sim_tracks:
        # Color the track by redshift
        from matplotlib.collections import LineCollection
        points  = np.array([mstar_arr, mdust_arr]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        norm    = plt.Normalize(vmin=0, vmax=6)
        lc      = LineCollection(segs, cmap="plasma_r", norm=norm,
                                 linewidth=2.0, zorder=5, label=label)
        lc.set_array(0.5 * (z_arr[:-1] + z_arr[1:]))
        ax.add_collection(lc)

        # Mark z milestones — labels left of dot on rising section (z>2), right otherwise
        for z_target in Z_LABELS:
            idx = np.argmin(np.abs(z_arr - z_target))
            if np.abs(z_arr[idx] - z_target) < 0.3:
                ax.scatter(mstar_arr[idx], mdust_arr[idx],
                           s=35, color=cm.plasma_r(norm(z_arr[idx])),
                           zorder=6, edgecolors="k", linewidths=0.4)
                # Rising section (z >= 2): label to the left; late-time: to the right
                if z_arr[idx] >= 2.0:
                    ax.annotate(f"z={z_arr[idx]:.1f}",
                                (mstar_arr[idx], mdust_arr[idx]),
                                textcoords="offset points", xytext=(-38, 3),
                                fontsize=6.5, color="0.2", zorder=7,
                                ha="left")
                else:
                    ax.annotate(f"z={z_arr[idx]:.1f}",
                                (mstar_arr[idx], mdust_arr[idx]),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=6.5, color="0.2", zorder=7,
                                ha="left")

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax.set_xlabel(r"$\log\,(M_\star\,/\,\mathrm{M}_\odot)$", fontsize=11)
    ax.set_ylabel(r"$\log\,(M_\mathrm{dust}\,/\,\mathrm{M}_\odot)$", fontsize=11)
    ax.set_xlim(7.0, 12.0)
    ax.set_ylim(3.5, 9.5)
    ax.tick_params(labelsize=9)

    ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
              handlelength=1.5, labelspacing=0.3, borderpad=0.6)

    ax.set_title("Evolution of Halo 569", fontsize=10)

    # Grid behind everything — set_axisbelow pushes grid under all artists
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="major", color="0.88", linewidth=0.5)
    ax.grid(True, which="minor", color="0.93", linewidth=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot M_dust vs M_star track for CosmicGrain runs")
    parser.add_argument("output_dirs", nargs="+",
                        help="One or more Gadget-4 output directories")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels for each run (default: directory names)")
    parser.add_argument("--skip-every", type=int, default=1,
                        help="Process every N-th snapshot (e.g. 2 = every other)")
    parser.add_argument("--output", default="mdust_mstar.png",
                        help="Output figure filename")
    parser.add_argument("--obs-data", default="obs_data/obs_dustmass.npz",
                        help="Path to obs_dustmass.npz from parse_obs_data.py "
                             "(default: obs_data/obs_dustmass.npz)")
    args = parser.parse_args()

    n = len(args.output_dirs)
    labels = args.labels if args.labels else [Path(d).name for d in args.output_dirs]
    if len(labels) != n:
        parser.error("--labels must match the number of output_dirs")

    # Load observational data
    obs = load_obs_data(args.obs_data)

    # Cycle through a few distinct colors (used for legend label; track is colored by z)
    track_colors = ["#e07b39", "#4682b4", "#55a868", "#c44e52", "#8172b2"]

    sim_tracks = []
    for i, (d, lbl) in enumerate(zip(args.output_dirs, labels)):
        color = track_colors[i % len(track_colors)]
        print(f"\nProcessing: {lbl}")
        z_arr, mstar_arr, mdust_arr = run_simulation(d, lbl, color,
                                                      skip_every=args.skip_every)
        sim_tracks.append((lbl, color, z_arr, mstar_arr, mdust_arr))

    make_plot(sim_tracks, args.output, obs)


if __name__ == "__main__":
    main()
