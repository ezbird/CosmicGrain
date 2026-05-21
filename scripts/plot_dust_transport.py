#!/usr/bin/env python3
"""
plot_dust_transport.py
----------------------
Four-panel dust transport figure for the CosmicGrain methods paper.

Panel 1 — Displacement PDF by grain size
Panel 2 — Net radial migration vs birth radius (2D histogram)
Panel 3 — Displacement vs particle age, coloured by GrainType
Panel 4 — Dust-gas velocity offset Δv vs gas density n_H  [NEW]
           Tests whether Epstein drag is producing physically correct
           coupling. Overplots analytical t_stop contours from
           McKinnon+2018 for a=100 nm silicate grains.

Usage
-----
    python plot_dust_transport.py \\
        --snap-dir   ../S10_output_1024 \\
        --log-dir    ../S10_output_1024/dust_logs \\
        --snap-num   49 \\
        --r-max-pkpc 350 \\
        --output     dust_transport_1024.png

    # With explicit coupling snapshot (defaults to --snap-num)
    python plot_dust_transport.py \\
        --snap-dir   ../S10_output_2048 \\
        --log-dir    ../S10_output_2048/dust_logs \\
        --snap-num   49 \\
        --coupling-snap-num 30 \\
        --coupling-nsample  50000 \\
        --output     dust_transport_2048.png

Notes
-----
  Panel 4 reads PartType0 (gas) and PartType6 (dust) from a snapshot,
  subsamples --coupling-nsample dust particles for tractability at 2048^3,
  matches each to its nearest gas cell via scipy KDTree, computes
  |v_dust - v_gas| and n_H, and plots a 2D log-log histogram.

  Analytical t_stop contours assume:
    a = 100 nm, rho_grain = 2.4 g/cm^3, mu = 0.6, gamma = 5/3, T = 1e4 K
  t_stop ∝ a * rho_grain / (rho_gas * c_s)
  Terminal drift: v_drift = F * t_stop / m_grain  (not computed; contours
  mark where t_stop equals 0.001, 0.1, 10 Myr as reference timescales).
"""

import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# ── Physical constants (CGS) ──────────────────────────────────────────────────
BOLTZMANN   = 1.38064852e-16  # erg/K
PROTONMASS  = 1.6726219e-24   # g
SEC_PER_MYR = 3.15576e13      # s/Myr
PARSEC      = 3.085678e18     # cm

# ── Cosmology (matched to CosmicGrain runs) ───────────────────────────────────
OMEGA_M  = 0.3089
OMEGA_L  = 0.6911
OMEGA_R  = 0.0
H0_KM_S_MPC = 67.74  # km/s/Mpc


# ═══════════════════════════════════════════════════════════════════════════════
# Cosmological time integration
# ═══════════════════════════════════════════════════════════════════════════════

def a_to_gyr(a_arr):
    """Convert scale factor array to lookback-time array in Gyr (t=0 at Big Bang)."""
    from scipy.integrate import quad
    H0_s = H0_KM_S_MPC * 1e5 / (PARSEC * 1e3 * 1e3)  # 1/s

    def integrand(a):
        return 1.0 / (a * np.sqrt(OMEGA_M / a**3 + OMEGA_L + OMEGA_R / a**4))

    t_arr = []
    for a in np.atleast_1d(a_arr):
        t, _ = quad(integrand, 0.0, a, limit=200)
        t_arr.append(t / H0_s / (1e9 * 365.25 * 24 * 3600))
    return np.array(t_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# Snapshot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_snap_files(snap_dir, snap_num):
    """Return sorted list of HDF5 chunk files for a given snapshot number."""
    pattern_single = os.path.join(snap_dir,
                                  f"snapdir_{snap_num:03d}",
                                  f"snapshot_{snap_num:03d}.hdf5")
    pattern_multi  = os.path.join(snap_dir,
                                  f"snapdir_{snap_num:03d}",
                                  f"snapshot_{snap_num:03d}.*.hdf5")
    files = sorted(glob.glob(pattern_multi))
    if not files:
        if os.path.exists(pattern_single):
            files = [pattern_single]
    if not files:
        raise FileNotFoundError(
            f"No snapshot files found for snap {snap_num} in {snap_dir}")
    return files


def read_header(snap_file):
    """Read box size, scale factor, HubbleParam from a snapshot chunk."""
    with h5py.File(snap_file, "r") as f:
        box  = float(f["Header"].attrs["BoxSize"])    # comoving kpc/h
        a    = float(f["Header"].attrs["Time"])
        h    = float(f["Parameters"].attrs["HubbleParam"])
    return box, a, h


def load_dust_survivors(snap_dir, snap_num):
    """
    Load surviving PartType6 from all chunks of a snapshot.
    Returns dict with keys: pos, vel, mass, grain_radius, carbon_frac,
                             grain_type, birth_pos, birth_a
    Positions in comoving kpc/h; velocities in km/s (peculiar).
    """
    files = find_snap_files(snap_dir, snap_num)
    print(f"  Found {len(files)} snapshot chunk(s) ...")

    arrs = {k: [] for k in
            ["pos", "vel", "mass", "grain_radius", "carbon_frac",
             "grain_type", "birth_pos", "birth_a"]}

    box, a_snap, h = None, None, None

    for f_path in files:
        with h5py.File(f_path, "r") as f:
            if box is None:
                box  = float(f["Header"].attrs["BoxSize"])
                a_snap = float(f["Header"].attrs["Time"])
                h    = float(f["Parameters"].attrs["HubbleParam"])

            if "PartType6" not in f:
                continue
            pt6 = f["PartType6"]

            pos   = pt6["Coordinates"][:]        # ckpc/h
            vel   = pt6["Velocities"][:]          # km/s peculiar
            mass  = pt6["Masses"][:]
            rad   = pt6["GrainRadius"][:]         # nm
            cf    = pt6["CarbonFraction"][:]
            gt    = pt6["GrainType"][:]
            bpos  = pt6["BirthPos"][:]            # ckpc/h
            ba    = pt6["StellarAge"][:]           # birth scale factor

            arrs["pos"].append(pos)
            arrs["vel"].append(vel)
            arrs["mass"].append(mass)
            arrs["grain_radius"].append(rad)
            arrs["carbon_frac"].append(cf)
            arrs["grain_type"].append(gt)
            arrs["birth_pos"].append(bpos)
            arrs["birth_a"].append(ba)

    result = {k: np.concatenate(arrs[k]) for k in arrs}
    result["box"]    = box
    result["a_snap"] = a_snap
    result["h"]      = h

    n_total = len(result["pos"])
    missing = np.all(result["birth_pos"] == 0, axis=1)
    result["valid"]  = ~missing

    print(f"  Found {n_total:,} surviving dust particles")
    print(f"  Excluding {missing.sum():,} particles with missing BirthPos")
    return result


def load_gas_for_coupling(snap_dir, snap_num, nsample=None, rng=None):
    """
    Load PartType0 (gas) positions, velocities, density from a snapshot.
    Returns arrays: pos (ckpc/h), vel (km/s), density (code), n_H (cm^-3).
    If nsample is not None, returns a random subsample of gas particles.
    """
    files = find_snap_files(snap_dir, snap_num)
    pos_list, vel_list, rho_list = [], [], []

    unit_density = None

    for f_path in files:
        with h5py.File(f_path, "r") as f:
            if unit_density is None:
                # UnitDensity_in_cgs from Parameters
                ud = f["Parameters"].attrs.get("UnitDensity_in_cgs", None)
                if ud is None:
                    # Reconstruct from UnitLength and UnitMass
                    ul = float(f["Parameters"].attrs["UnitLength_in_cm"])
                    um = float(f["Parameters"].attrs["UnitMass_in_g"])
                    unit_density = um / ul**3
                else:
                    unit_density = float(ud)
                a_snap = float(f["Header"].attrs["Time"])
                h      = float(f["Parameters"].attrs["HubbleParam"])

            if "PartType0" not in f:
                continue
            pt0 = f["PartType0"]
            pos_list.append(pt0["Coordinates"][:])
            vel_list.append(pt0["Velocities"][:])
            rho_list.append(pt0["Density"][:])

    pos = np.concatenate(pos_list)
    vel = np.concatenate(vel_list)
    rho = np.concatenate(rho_list)   # code units

    # Physical density in CGS
    cf_a3inv = 1.0 / a_snap**3
    rho_cgs  = rho * cf_a3inv * unit_density * h * h
    HYDROGEN_MASSFRAC = 0.76
    n_H = rho_cgs * HYDROGEN_MASSFRAC / PROTONMASS

    if nsample is not None and nsample < len(pos):
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(pos), size=nsample, replace=False)
        pos, vel, n_H = pos[idx], vel[idx], n_H[idx]

    return pos, vel, n_H, a_snap, h


# ═══════════════════════════════════════════════════════════════════════════════
# Halo center tracking
# ═══════════════════════════════════════════════════════════════════════════════

def get_halo_center(snap_dir, snap_num):
    """Find halo center via halo_utils, with PartType6 median fallback."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import halo_utils
        files    = find_snap_files(snap_dir, snap_num)
        snap_base = files[0].replace(".0.hdf5", "").replace(".hdf5", "")
        cat_dir  = os.path.dirname(files[0]).replace(
                       f"snapdir_{snap_num:03d}",
                       f"groups_{snap_num:03d}")
        cat_path = os.path.join(cat_dir,
                                f"fof_subhalo_tab_{snap_num:03d}.hdf5")
        if not os.path.exists(cat_path):
            # Try without subdirectory
            cat_path = os.path.join(
                snap_dir, f"groups_{snap_num:03d}",
                f"fof_subhalo_tab_{snap_num:03d}.0.hdf5")
        halo = halo_utils.load_target_halo(cat_path, snap_base,
                                            particle_types=[0], verbose=False)
        center = np.array(halo["GroupPos"])
        print(f"  Halo center (halo_utils): {center} ckpc/h")
        return center
    except Exception as e:
        print(f"  halo_utils failed ({e}), using PartType6 median")
        files = find_snap_files(snap_dir, snap_num)
        coords = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                if "PartType6" in f:
                    coords.append(f["PartType6"]["Coordinates"][:])
        if not coords:
            raise RuntimeError("No PartType6 found for halo center fallback")
        c = np.median(np.concatenate(coords), axis=0)
        print(f"  Halo center (PartType6 median): {c} ckpc/h")
        return c


def build_halo_track(snap_dir, n_snaps=50):
    """
    Build halo center track from available group catalogs.
    Returns (track_a, track_center) sorted by scale factor.
    """
    track_a, track_c = [], []
    snap_dirs = sorted(glob.glob(os.path.join(snap_dir, "snapdir_*")))
    snap_nums = [int(os.path.basename(d).split("_")[-1]) for d in snap_dirs]

    # Thin to n_snaps evenly spaced
    if len(snap_nums) > n_snaps:
        idx = np.round(np.linspace(0, len(snap_nums)-1, n_snaps)).astype(int)
        snap_nums = [snap_nums[i] for i in idx]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import halo_utils
        use_halo_utils = True
    except ImportError:
        use_halo_utils = False

    for sn in snap_nums:
        files = find_snap_files(snap_dir, sn)
        if not files:
            continue
        try:
            _, a, _ = read_header(files[0])
            if use_halo_utils:
                snap_base = files[0].replace(".0.hdf5", "").replace(".hdf5","")
                cat_path  = files[0].replace(
                    f"snapdir_{sn:03d}", f"groups_{sn:03d}").replace(
                    f"snapshot_{sn:03d}.0.hdf5",
                    f"fof_subhalo_tab_{sn:03d}.0.hdf5").replace(
                    f"snapshot_{sn:03d}.hdf5",
                    f"fof_subhalo_tab_{sn:03d}.0.hdf5")
                if not os.path.exists(cat_path):
                    continue
                halo = halo_utils.load_target_halo(cat_path, snap_base,
                                                    particle_types=[0],
                                                    verbose=False)
                center = np.array(halo["GroupPos"])
            else:
                # Fallback: PartType0 CoM within central region
                coords = []
                for fp in files:
                    with h5py.File(fp, "r") as f:
                        if "PartType0" in f:
                            coords.append(f["PartType0"]["Coordinates"][:])
                if not coords:
                    continue
                center = np.median(np.concatenate(coords), axis=0)

            track_a.append(a)
            track_c.append(center)
        except Exception:
            continue

    if len(track_a) < 2:
        raise RuntimeError("Could not build halo track (< 2 catalog entries)")

    order = np.argsort(track_a)
    track_a = np.array(track_a)[order]
    track_c = np.array(track_c)[order]
    print(f"  Halo track: {len(track_a)} catalog entries  "
          f"a = [{track_a[0]:.3f} … {track_a[-1]:.3f}]")
    return track_a, track_c


def interpolate_halo_center(track_a, track_center, query_a):
    """Linearly interpolate halo center, clamping outside track range."""
    query_a = np.asarray(query_a)
    interp  = interp1d(track_a, track_center, axis=0,
                       kind="linear", bounds_error=False,
                       fill_value=(track_center[0], track_center[-1]))
    return interp(query_a)


# ═══════════════════════════════════════════════════════════════════════════════
# Log file loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dust_logs(log_dir):
    """
    Load all dust event log files from log_dir.
    Expected columns (space-separated):
      event_type  ID  birth_x  birth_y  birth_z  birth_a
      event_x  event_y  event_z  event_a
      mass  grain_radius  carbon_frac  grain_type  gas_idx  task
    Returns a DataFrame with these columns.
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "dust_log_*.txt")))
    if not log_files:
        raise FileNotFoundError(f"No dust log files found in {log_dir}")
    print(f"  Loading {len(log_files)} log files from {log_dir} ...")

    col_names = [
        "event_type",
        "id",
        "birth_x", "birth_y", "birth_z", "birth_a",
        "event_x", "event_y", "event_z", "event_a",
        "mass", "grain_radius", "carbon_frac", "grain_type",
        "gas_idx", "task"
    ]

    dfs = []
    for lf in log_files:
        try:
            df = pd.read_csv(lf, sep=r"\s+", comment="#",
                             names=col_names, on_bad_lines="skip")
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {lf}: {e}")

    if not dfs:
        raise RuntimeError("No log data could be loaded")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total logged events: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Periodic distance helper
# ═══════════════════════════════════════════════════════════════════════════════

def periodic_diff(a, b, box):
    """Minimum-image difference a - b in a periodic box of size box."""
    d = a - b
    d = d - box * np.round(d / box)
    return d


def periodic_dist(a, b, box):
    """Scalar periodic distance |a - b|."""
    return np.linalg.norm(periodic_diff(a, b, box), axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical Epstein t_stop contour
# ═══════════════════════════════════════════════════════════════════════════════

def t_stop_myr(n_H_cm3, a_nm=100.0, T_K=1e4,
               rho_grain_gcc=2.4, mu=0.6, gamma=5.0/3.0):
    """
    Epstein stopping time in Myr for subsonic grains.
    t_stop = sqrt(pi*gamma/8) * a*rho_grain / (rho_gas * c_s)
    rho_gas from n_H assuming hydrogen mass fraction 0.76.
    """
    rho_gas = n_H_cm3 * PROTONMASS / 0.76        # g/cm³
    c_s     = np.sqrt(gamma * BOLTZMANN * T_K / (mu * PROTONMASS))  # cm/s
    a_cm    = a_nm * 1e-7
    t_s     = (np.sqrt(np.pi * gamma / 8.0) * a_cm * rho_grain_gcc
               / (rho_gas * c_s))
    return t_s / SEC_PER_MYR


# ═══════════════════════════════════════════════════════════════════════════════
# Velocity coupling panel
# ═══════════════════════════════════════════════════════════════════════════════

def make_coupling_panel(ax, snap_dir, snap_num, h_cosmo,
                        nsample=50000, rng=None):
    """
    Panel 4: Δv vs n_H 2D log-log histogram with t_stop contours.

    Loads gas and dust from snap_num, matches each (subsampled) dust particle
    to its nearest gas neighbour via KDTree, computes |v_dust - v_gas| and
    n_H from gas density, then plots as a 2D histogram.

    Overplots t_stop contours at 0.001, 0.1, 10 Myr for a=100 nm silicate.
    """
    print(f"\n  [Coupling panel] Loading gas from snap {snap_num} ...")
    gas_pos, gas_vel, gas_nH, a_snap, _ = load_gas_for_coupling(
        snap_dir, snap_num, nsample=None, rng=rng)

    print(f"  [Coupling panel] Loading dust from snap {snap_num} ...")
    dust = load_dust_survivors(snap_dir, snap_num)
    mask = dust["valid"]
    d_pos = dust["pos"][mask]
    d_vel = dust["vel"][mask]

    # Subsample dust for tractability
    n_dust = len(d_pos)
    if n_dust > nsample:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(n_dust, size=nsample, replace=False)
        d_pos = d_pos[idx]
        d_vel = d_vel[idx]
    print(f"  [Coupling panel] Matching {len(d_pos):,} dust → gas via KDTree ...")

    # Build KDTree on gas positions — no periodic wrapping for now
    # (acceptable for a statistical diagnostic within the zoom region)
    tree     = cKDTree(gas_pos)
    _, gi    = tree.query(d_pos, k=1, workers=-1)

    v_diff   = d_vel - gas_vel[gi]                        # km/s
    dv_kms   = np.linalg.norm(v_diff, axis=1)            # km/s
    nH_match = gas_nH[gi]                                 # cm^-3

    # Filter unphysical zeros and negatives
    good = (dv_kms > 1e-6) & (nH_match > 1e-8)
    dv_kms   = dv_kms[good]
    nH_match = nH_match[good]
    print(f"  [Coupling panel] {good.sum():,} pairs after filtering")

    # 2D log histogram
    nH_bins  = np.logspace(-6, 4, 60)
    dv_bins  = np.logspace(-3, 4, 60)
    H, xedge, yedge = np.histogram2d(nH_match, dv_kms,
                                      bins=[nH_bins, dv_bins])
    # Mask empty cells
    H = np.ma.masked_where(H == 0, H)

    pcm = ax.pcolormesh(xedge, yedge, H.T,
                        norm=mcolors.LogNorm(vmin=1),
                        cmap="viridis", shading="auto")
    plt.colorbar(pcm, ax=ax, label="N particles")

    # ── Analytical t_stop contours ────────────────────────────────────────────
    nH_line = np.logspace(-6, 4, 300)
    colors_ts  = ["#ff7f0e", "#d62728", "#9467bd"]
    labels_ts  = ["$t_{\\rm stop}=0.001$ Myr", "$t_{\\rm stop}=0.1$ Myr",
                  "$t_{\\rm stop}=10$ Myr"]
    t_targets  = [0.001, 0.1, 10.0]

    for t_targ, col, lab in zip(t_targets, colors_ts, labels_ts):
        ts_arr = t_stop_myr(nH_line)
        # t_stop contour is vertical in (nH, Δv) space — it marks nH where
        # t_stop = t_targ. The terminal velocity v = F*t_stop/m is uncertain
        # without knowing F, so we instead draw vertical lines at the nH
        # where t_stop equals our target, across all Δv.
        nH_targ = np.interp(t_targ, ts_arr[::-1], nH_line[::-1],
                            left=np.nan, right=np.nan)
        if np.isfinite(nH_targ):
            ax.axvline(nH_targ, color=col, ls="--", lw=1.5, label=lab)

    # Literature reference: McKinnon+2018 found Δv < 1 km/s in ISM
    ax.axhline(1.0, color="white", ls=":", lw=1.2, alpha=0.8,
               label="$\\Delta v = 1$ km/s (McKinnon+2018 ISM)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\rm H}\ [\rm cm^{-3}]$",   fontsize=11)
    ax.set_ylabel(r"$|\Delta v|\ [\rm km\,s^{-1}]$", fontsize=11)
    ax.set_title(f"Dust–gas velocity coupling  (snap {snap_num}, z={1/a_snap-1:.2f})",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(1e-6, 1e4)
    ax.set_ylim(1e-3, 1e4)

    # Stats to terminal
    print(f"  [Coupling panel] Δv stats (km/s):")
    print(f"    median = {np.median(dv_kms):.2f}")
    print(f"    p90    = {np.percentile(dv_kms, 90):.2f}")
    print(f"    max    = {dv_kms.max():.2f}")
    print(f"    fraction Δv < 1 km/s (ISM-coupled): "
          f"{(dv_kms < 1.0).mean()*100:.1f}%")
    ism = nH_match > 0.1
    if ism.sum() > 10:
        print(f"    ISM (nH>0.1): median Δv = {np.median(dv_kms[ism]):.2f} km/s  "
              f"(N={ism.sum():,})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(df_log, survivors, track_a, track_center,
                r_max_pkpc, snap_dir, snap_num,
                coupling_snap_num, coupling_nsample,
                output_path):
    """Build the 4-panel figure."""

    box  = survivors["box"]
    a_z0 = survivors["a_snap"]
    h    = survivors["h"]

    # ── Combine log events + survivors into one DataFrame ─────────────────────
    # Destroyed particles come from the log
    df_dest = df_log.rename(columns={
        "birth_x": "birth_x", "birth_y": "birth_y", "birth_z": "birth_z",
        "event_x": "x",       "event_y": "y",       "event_z": "z",
    }).copy()
    df_dest["is_survivor"] = False

    # Survivors
    mask = survivors["valid"]
    df_surv = pd.DataFrame({
        "birth_x":      survivors["birth_pos"][mask, 0],
        "birth_y":      survivors["birth_pos"][mask, 1],
        "birth_z":      survivors["birth_pos"][mask, 2],
        "birth_a":      survivors["birth_a"][mask],
        "x":            survivors["pos"][mask, 0],
        "y":            survivors["pos"][mask, 1],
        "z":            survivors["pos"][mask, 2],
        "event_a":      np.full(mask.sum(), a_z0),
        "mass":         survivors["mass"][mask],
        "grain_radius": survivors["grain_radius"][mask],
        "carbon_frac":  survivors["carbon_frac"][mask],
        "grain_type":   survivors["grain_type"][mask],
        "event_type":   0,
        "is_survivor":  True,
    })
    df_surv["event_x"] = df_surv["x"]
    df_surv["event_y"] = df_surv["y"]
    df_surv["event_z"] = df_surv["z"]

    df = pd.concat([df_dest, df_surv], ignore_index=True, sort=False)
    df["birth_a"]  = pd.to_numeric(df["birth_a"],  errors="coerce")
    df["event_a"]  = pd.to_numeric(df["event_a"],  errors="coerce")
    df.dropna(subset=["birth_a", "event_a", "birth_x", "birth_y", "birth_z",
                       "x", "y", "z"], inplace=True)

    print(f"  Combined dataset: {len(df):,} particles "
          f"({df['is_survivor'].sum():,} survivors, "
          f"{(~df['is_survivor']).sum():,} destruction events)")

    # ── Halo-center-tracked radii ─────────────────────────────────────────────
    birth_xyz   = df[["birth_x", "birth_y", "birth_z"]].values
    event_xyz   = df[["x", "y", "z"]].values
    birth_a_arr = df["birth_a"].values
    event_a_arr = df["event_a"].values

    birth_centers = interpolate_halo_center(track_a, track_center, birth_a_arr)
    event_centers = interpolate_halo_center(track_a, track_center, event_a_arr)

    # Physical kpc
    birth_sep = np.linalg.norm(birth_xyz - birth_centers, axis=1)
    event_sep = np.linalg.norm(event_xyz - event_centers, axis=1)

    df["birth_radius"] = birth_sep * birth_a_arr / h
    df["event_radius"] = event_sep * event_a_arr / h
    df["dr"]           = df["event_radius"] - df["birth_radius"]

    # Physical displacement |event - birth| with periodic wrap
    diff = periodic_diff(event_xyz, birth_xyz, box)
    df["displacement"] = np.linalg.norm(diff, axis=1) * a_z0 / h

    # ── r_max filter ──────────────────────────────────────────────────────────
    n_before = len(df)
    if r_max_pkpc > 0:
        df = df[df["birth_radius"] <= r_max_pkpc].copy()
    print(f"  r_max filter ({r_max_pkpc} pkpc): {n_before:,} → {len(df):,} particles "
          f"({100*(n_before - len(df))/n_before:.1f}% removed)")

    if df.empty:
        print("ERROR: no particles remain after r_max filter.")
        return

    br = df["birth_radius"].values
    print(f"  Birth radius stats (physical kpc):")
    print(f"    min={br.min():.2f}  p10={np.percentile(br,10):.2f}  "
          f"median={np.median(br):.2f}  p90={np.percentile(br,90):.2f}  "
          f"max={br.max():.2f}")

    # ── Particle ages ─────────────────────────────────────────────────────────
    print("  Computing particle ages (Friedmann integration) ...")
    unique_a = np.unique(np.concatenate([birth_a_arr, event_a_arr]))
    unique_a = unique_a[(unique_a > 0) & (unique_a <= 1.0)]
    t_of_a   = dict(zip(unique_a, a_to_gyr(unique_a)))

    df["age_gyr"] = (df["event_a"].map(t_of_a) -
                     df["birth_a"].map(t_of_a)).clip(lower=0)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ─── Panel 1: Displacement PDF by grain size ──────────────────────────────
    size_bins  = [(0, 50, "Small (<50 nm)", "steelblue"),
                  (50, 150, "Medium (50–150 nm)", "darkorange"),
                  (150, 1e6, "Large (>150 nm)", "forestgreen")]
    disp       = df["displacement"].values
    grain_r    = df["grain_radius"].values

    bins_hist = np.logspace(-1, np.log10(max(disp.max(), 1.0)+0.1), 60)
    for lo, hi, label, col in size_bins:
        mask_s = (grain_r >= lo) & (grain_r < hi) & (disp > 0)
        if mask_s.sum() > 0:
            ax1.hist(disp[mask_s], bins=bins_hist, histtype="step",
                     density=True, lw=1.8, color=col, label=f"{label} (N={mask_s.sum():,})")

    ax1.set_xscale("log")
    ax1.set_xlabel("Displacement (pkpc)", fontsize=11)
    ax1.set_ylabel("Probability density", fontsize=11)
    ax1.set_title("Displacement PDF by grain size", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: Radial migration vs birth radius ───────────────────────────
    valid2 = np.isfinite(df["dr"]) & np.isfinite(df["birth_radius"])
    x2 = df["birth_radius"].values[valid2]
    y2 = df["dr"].values[valid2]

    rbins = np.linspace(0, r_max_pkcp := (r_max_pkpc if r_max_pkpc > 0 else x2.max()), 60)
    drbins = np.linspace(y2.min(), y2.max(), 60)
    H2, xe2, ye2 = np.histogram2d(x2, y2, bins=[rbins, drbins])
    H2 = np.ma.masked_where(H2 == 0, H2)
    ax2.pcolormesh(xe2, ye2, H2.T,
                   norm=mcolors.LogNorm(vmin=1), cmap="plasma", shading="auto")
    ax2.axhline(0, color="white", ls="--", lw=1.0, alpha=0.7)

    # Median dr per birth radius bin
    mid_r = 0.5 * (rbins[:-1] + rbins[1:])
    med_dr = np.array([np.median(y2[np.abs(x2 - r) < (rbins[1]-rbins[0])]) for r in mid_r])
    ax2.plot(mid_r, med_dr, color="white", lw=2.0, ls="-", label="Median $\\Delta r$")

    ax2.set_xlabel("Birth radius (pkpc)", fontsize=11)
    ax2.set_ylabel("Net radial migration $\\Delta r$ (pkpc)", fontsize=11)
    ax2.set_title("Radial migration vs birth radius", fontsize=11)
    ax2.legend(fontsize=9)

    # ─── Panel 3: Displacement vs age, coloured by grain type ────────────────
    gt_map = {0: ("SNII silicate", "steelblue"),
              1: ("AGB carbon",    "darkorange"),
              2: ("Mixed",         "grey")}
    valid3 = np.isfinite(df["age_gyr"]) & (disp > 0)

    for gt_val, (gt_label, gt_col) in gt_map.items():
        m = valid3 & (df["grain_type"].values == gt_val)
        if m.sum() < 2:
            continue
        # Subsample for scatter
        nsub = min(5000, m.sum())
        idx  = np.random.choice(np.where(m)[0], nsub, replace=False)
        ax3.scatter(df["age_gyr"].values[idx], disp[idx],
                    s=1, alpha=0.3, color=gt_col, label=f"{gt_label} (N={m.sum():,})")

        # Running median
        age_sorted = np.sort(df["age_gyr"].values[m])
        q_bins     = np.percentile(age_sorted, np.linspace(5, 95, 20))
        q_mid      = 0.5 * (q_bins[:-1] + q_bins[1:])
        q_med      = [np.median(disp[m & (df["age_gyr"].values >= q_bins[i]) &
                                     (df["age_gyr"].values < q_bins[i+1])])
                      for i in range(len(q_bins)-1)]
        ax3.plot(q_mid, q_med, color=gt_col, lw=2.0)

    ax3.set_yscale("log")
    ax3.set_xlabel("Particle age (Gyr)", fontsize=11)
    ax3.set_ylabel("Displacement (pkpc)", fontsize=11)
    ax3.set_title("Displacement vs age by grain type", fontsize=11)
    ax3.legend(fontsize=9, markerscale=5)
    ax3.grid(True, alpha=0.3)

    # ─── Panel 4: Velocity coupling ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    make_coupling_panel(ax4, snap_dir,
                        coupling_snap_num if coupling_snap_num is not None else snap_num,
                        h, nsample=coupling_nsample, rng=rng)

    fig.suptitle("CosmicGrain Dust Transport Diagnostics", fontsize=14,
                 fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CosmicGrain dust transport + velocity coupling figure")
    p.add_argument("--snap-dir",    required=True,
                   help="Directory containing snapdir_XXX/ subdirectories")
    p.add_argument("--log-dir",     required=True,
                   help="Directory containing dust_log_*.txt files")
    p.add_argument("--snap-num",    type=int, required=True,
                   help="Snapshot number for survivors (e.g. 49 for z=0)")
    p.add_argument("--r-max-pkpc",  type=float, default=350.0,
                   help="Filter: max birth radius in physical kpc (0 = disable)")
    p.add_argument("--output",      default="dust_transport.png",
                   help="Output figure path")
    p.add_argument("--halo-center", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Manual halo center in comoving kpc/h (overrides auto)")
    # Coupling panel options
    p.add_argument("--coupling-snap-num", type=int, default=None,
                   help="Snapshot for velocity coupling panel "
                        "(default: same as --snap-num)")
    p.add_argument("--coupling-nsample",  type=int, default=50000,
                   help="Number of dust particles to sample for coupling panel "
                        "(default: 50000; reduce if memory limited)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("CosmicGrain dust transport analysis")
    print("=" * 60)

    # ── 1. Load dust event logs ───────────────────────────────────────────────
    print("\n[1] Loading dust event log ...")
    df_log = load_dust_logs(args.log_dir)

    # ── 2. Load survivors ─────────────────────────────────────────────────────
    print("\n[2] Loading survivors ...")
    survivors = load_dust_survivors(args.snap_dir, args.snap_num)
    box  = survivors["box"]
    a_z0 = survivors["a_snap"]
    h    = survivors["h"]
    print(f"  Box size: {box:.1f} comoving kpc/h")
    print(f"  Scale factor: a = {a_z0:.4f}")
    print(f"  h = {h:.4f}")
    print(f"  Conversion: 1 ckpc/h = {a_z0/h:.4f} physical kpc")

    # ── 3. Halo center track ──────────────────────────────────────────────────
    print("\n[3] Building halo center track ...")
    if args.halo_center is not None:
        halo_center  = np.array(args.halo_center)
        print(f"  Halo center (manual): {halo_center} ckpc/h")
        track_a      = np.array([0.01, 1.0])
        track_center = np.array([halo_center, halo_center])
    else:
        halo_center          = get_halo_center(args.snap_dir, args.snap_num)
        track_a, track_center = build_halo_track(args.snap_dir, n_snaps=50)
        # Cross-check at z=0
        pt6_files = find_snap_files(args.snap_dir, args.snap_num)
        coords = []
        for fp in pt6_files:
            with h5py.File(fp, "r") as f:
                if "PartType6" in f:
                    coords.append(f["PartType6"]["Coordinates"][:])
        if coords:
            pt6_med = np.median(np.concatenate(coords), axis=0)
            off_ckpch = np.linalg.norm(halo_center - pt6_med)
            off_pkpc  = off_ckpch * a_z0 / h
            print(f"  Cross-check (z=0 informational):")
            print(f"    halo_utils z=0:      {halo_center} ckpc/h")
            print(f"    PartType6 z=0 median:{pt6_med} ckpc/h")
            print(f"    Offset: {off_ckpch:.1f} ckpc/h = {off_pkpc:.1f} pkpc")
            print(f"  INFO: offset expected at S10 — track interpolation handles this.")

    # ── 4. Build figure ───────────────────────────────────────────────────────
    print("\n[4] Building figure ...")
    make_figure(
        df_log, survivors, track_a, track_center,
        r_max_pkpc=args.r_max_pkpc,
        snap_dir=args.snap_dir,
        snap_num=args.snap_num,
        coupling_snap_num=args.coupling_snap_num,
        coupling_nsample=args.coupling_nsample,
        output_path=args.output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
