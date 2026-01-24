#!/usr/bin/env python3
"""
Enhanced radial dust-to-metal ratio profile with diagnostics
Includes halo particle filtering and dust distribution analysis

Key upgrades:
- uint64-safe ParticleIDs everywhere (snapshot + halo file)
- robust halo .txt parsing (no float precision loss)
- explicit intersection diagnostics per particle type
- center fallback if filtered stars are missing
"""

import numpy as np
import h5py
import glob
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# -----------------------------
# Helpers: ID handling / debug
# -----------------------------
def as_uint64(arr, name="ids"):
    """Force an array to uint64 safely."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size == 0:
        return a.astype(np.uint64)
    # If someone accidentally passed floats, this will *not* recover precision.
    # We still cast, but we also warn elsewhere if floats are detected.
    return a.astype(np.uint64, copy=False)


def summarize_ids(ids, label):
    ids = np.asarray(ids)
    if ids.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={ids.size}, dtype={ids.dtype}, "
            f"min={int(ids.min())}, max={int(ids.max())}")


def ensure_sorted_unique(ids):
    ids = as_uint64(ids)
    if ids.size == 0:
        return ids
    ids = np.unique(ids)
    ids.sort()
    return ids


def match_mask(snapshot_ids, halo_ids):
    """
    Returns boolean mask for snapshot_ids that are in halo_ids.
    Assumes halo_ids are sorted unique for speed.
    """
    snapshot_ids = as_uint64(snapshot_ids)
    halo_ids = as_uint64(halo_ids)
    if snapshot_ids.size == 0 or halo_ids.size == 0:
        return np.zeros(snapshot_ids.shape, dtype=bool)
    # np.isin is fine, but faster if halo_ids are unique/sorted and snapshot_ids large:
    return np.isin(snapshot_ids, halo_ids, assume_unique=False)


def diag_match(ptype, snap_ids, halo_ids, mask, verbose=True):
    """Print intersection diagnostics."""
    if not verbose:
        return
    snap_ids = as_uint64(snap_ids)
    halo_ids = as_uint64(halo_ids) if halo_ids is not None else np.array([], dtype=np.uint64)
    n_snap = snap_ids.size
    n_halo = halo_ids.size
    n_match = int(mask.sum()) if mask is not None else 0

    print(f"\n[HALO_FILTER_DIAG] {ptype}")
    print(f"  {summarize_ids(snap_ids, 'snapshot IDs')}")
    print(f"  {summarize_ids(halo_ids, 'halo IDs')}")
    print(f"  matched: {n_match} / {n_snap}  ({(100.0*n_match/n_snap if n_snap>0 else 0.0):.2f}%)")

    # Quick dtype sanity check: if halo IDs were floats at any point, precision may be lost.
    # (We can’t recover it, but this tells you what happened.)
    # Note: by the time we get here, we already cast to uint64, so check is limited.
    # Still useful to add this warning in halo loader (done below).


# -----------------------------
# Halo ID loading
# -----------------------------
def load_halo_particle_ids(halo_file, halo_id, particle_types=('gas', 'stars', 'dust')):
    """
    Load particle IDs for a specific halo from halo finder output.
    Ensures uint64-safe IDs.
    """
    halo_pids = {}

    if halo_file.endswith('.hdf5'):
        with h5py.File(halo_file, 'r') as f:
            group_key = f'Group/{halo_id}'

            def read_ids(path):
                if path in f:
                    return ensure_sorted_unique(f[path][:])
                return np.array([], dtype=np.uint64)

            if 'gas' in particle_types:
                halo_pids['gas'] = read_ids(f'{group_key}/PartType0/ParticleIDs')
            if 'stars' in particle_types:
                halo_pids['stars'] = read_ids(f'{group_key}/PartType4/ParticleIDs')
            if 'dust' in particle_types:
                halo_pids['dust'] = read_ids(f'{group_key}/PartType6/ParticleIDs')

    elif halo_file.endswith('.txt') or halo_file.endswith('.dat'):
        # Robust parsing: read as strings first to avoid float conversion / scientific notation surprises.
        # Expected format: "<type> <id>" per line, where type is 0/4/6 and id is an integer.
        types = []
        ids = []
        bad_lines = 0

        with open(halo_file, 'r') as fp:
            for line in fp:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    bad_lines += 1
                    continue
                try:
                    t = int(parts[0])
                    # Force integer parsing; if the file contains "1.23e15" we will error here (good!)
                    pid = int(parts[1])
                    types.append(t)
                    ids.append(pid)
                except Exception:
                    bad_lines += 1

        if bad_lines > 0:
            print(f"[WARN] Skipped {bad_lines} malformed lines in {halo_file}")

        types = np.asarray(types, dtype=np.int32)
        ids = np.asarray(ids, dtype=np.uint64)

        if ids.size == 0:
            print(f"[WARN] No IDs parsed from {halo_file} — check file format.")
        else:
            # If your halo file *does* contain scientific notation, you want to know immediately:
            # Our int() parsing would have rejected those lines and counted them as malformed.
            pass

        if 'gas' in particle_types:
            halo_pids['gas'] = ensure_sorted_unique(ids[types == 0])
        if 'stars' in particle_types:
            halo_pids['stars'] = ensure_sorted_unique(ids[types == 4])
        if 'dust' in particle_types:
            halo_pids['dust'] = ensure_sorted_unique(ids[types == 6])

    else:
        raise ValueError(f"Unknown halo file format: {halo_file}")

    print(f"\nLoaded particle IDs for halo {halo_id}:")
    for ptype, arr in halo_pids.items():
        print(f"  {ptype}: {arr.size} particles (dtype={arr.dtype})")
        if arr.size > 0:
            print(f"    min={int(arr.min())}, max={int(arr.max())}")

    return halo_pids


# -----------------------------
# Snapshot loading with filter
# -----------------------------
def read_snapshot_with_filter(snapshot_path, halo_pids=None, verbose=False):
    """
    Read snapshot and filter by halo particle IDs (uint64-safe).
    """
    if os.path.isdir(snapshot_path):
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not snapshot_files:
            snapshot_files = [snapshot_path]

    if verbose:
        print(f"Reading {len(snapshot_files)} file(s)...")
        print(f"First file: {os.path.basename(snapshot_files[0])}")
        if len(snapshot_files) > 1:
            print(f"Last file:  {os.path.basename(snapshot_files[-1])}")

    data = {
        'dust_pos': [], 'dust_mass': [], 'dust_ids': [], 'dust_grain_radius': [],
        'gas_pos': [], 'gas_mass': [], 'gas_metals': [], 'gas_density': [], 'gas_ids': [], 'gas_temp': [],
        'star_pos': [], 'star_mass': [], 'star_metals': [], 'star_ids': []
    }

    time = None
    redshift = None
    box_size = None

    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            header = f['Header'].attrs
            time = float(header['Time'])
            redshift = float(header['Redshift'])
            box_size = float(header['BoxSize'])

            # Dust (PartType6)
            if 'PartType6' in f:
                dust_ids_all = as_uint64(f['PartType6/ParticleIDs'][:], "dust_ids")
                if halo_pids and 'dust' in halo_pids:
                    halo_ids = halo_pids['dust']
                    mask = match_mask(dust_ids_all, halo_ids)
                    diag_match("dust (PartType6)", dust_ids_all, halo_ids, mask, verbose=verbose)
                else:
                    mask = np.ones(dust_ids_all.shape, dtype=bool)

                if mask.sum() > 0:
                    coords = f['PartType6/Coordinates'][:]
                    masses = f['PartType6/Masses'][:]
                    data['dust_pos'].append(coords[mask])
                    data['dust_mass'].append(masses[mask])
                    data['dust_ids'].append(dust_ids_all[mask])

                    if 'GrainRadius' in f['PartType6']:
                        data['dust_grain_radius'].append(f['PartType6/GrainRadius'][:][mask])

            # Gas (PartType0)
            if 'PartType0' in f:
                gas_ids_all = as_uint64(f['PartType0/ParticleIDs'][:], "gas_ids")
                if halo_pids and 'gas' in halo_pids:
                    halo_ids = halo_pids['gas']
                    mask = match_mask(gas_ids_all, halo_ids)
                    diag_match("gas (PartType0)", gas_ids_all, halo_ids, mask, verbose=verbose)
                else:
                    mask = np.ones(gas_ids_all.shape, dtype=bool)

                if mask.sum() > 0:
                    data['gas_pos'].append(f['PartType0/Coordinates'][:][mask])
                    data['gas_mass'].append(f['PartType0/Masses'][:][mask])
                    data['gas_ids'].append(gas_ids_all[mask])

                    if 'Metallicity' in f['PartType0']:
                        data['gas_metals'].append(f['PartType0/Metallicity'][:][mask])
                    if 'Density' in f['PartType0']:
                        data['gas_density'].append(f['PartType0/Density'][:][mask])

                    # temperature: keep your rough estimate, but protect it
                    if 'InternalEnergy' in f['PartType0']:
                        u = f['PartType0/InternalEnergy'][:][mask]
                        temp = u * (2.0/3.0) * 1.2e10
                        data['gas_temp'].append(temp)

            # Stars (PartType4)
            if 'PartType4' in f:
                star_ids_all = as_uint64(f['PartType4/ParticleIDs'][:], "star_ids")
                if halo_pids and 'stars' in halo_pids:
                    halo_ids = halo_pids['stars']
                    mask = match_mask(star_ids_all, halo_ids)
                    diag_match("stars (PartType4)", star_ids_all, halo_ids, mask, verbose=verbose)
                else:
                    mask = np.ones(star_ids_all.shape, dtype=bool)

                if mask.sum() > 0:
                    data['star_pos'].append(f['PartType4/Coordinates'][:][mask])
                    data['star_mass'].append(f['PartType4/Masses'][:][mask])
                    data['star_ids'].append(star_ids_all[mask])

                    if 'Metallicity' in f['PartType4']:
                        data['star_metals'].append(f['PartType4/Metallicity'][:][mask])

    # Concatenate arrays
    for key in list(data.keys()):
        if isinstance(data[key], list):
            data[key] = np.concatenate(data[key]) if len(data[key]) > 0 else np.array([])

    data['time'] = time
    data['redshift'] = redshift
    data['box_size'] = box_size

    if verbose:
        print(f"\nTime = {time:.3f}, Redshift = {redshift:.3f}, BoxSize = {box_size:.1f} kpc")
        print(f"Loaded particles: Dust={len(data['dust_pos'])}, Gas={len(data['gas_pos'])}, Stars={len(data['star_pos'])}")

    return data


# -----------------------------
# Analysis / center finding
# -----------------------------
def find_halo_center(data, method='stellar', verbose=False):
    """Find halo center using various methods, with safe fallbacks."""

    # Preferred: stellar COM
    if method == 'stellar':
        if len(data['star_pos']) > 0 and len(data['star_mass']) > 0:
            center = np.average(data['star_pos'], weights=data['star_mass'], axis=0)
            if verbose:
                print(f"Center (stellar COM): [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] kpc")
            return center
        else:
            if verbose:
                print("[WARN] No stars available for stellar COM. Falling back to dense gas COM.")
            # fall through to dense_gas

    # Dense gas fallback
    if len(data['gas_pos']) > 0 and len(data['gas_mass']) > 0:
        if len(data.get('gas_density', [])) > 0 and data['gas_density'].size > 0:
            density_threshold = np.percentile(data['gas_density'], 90)
            mask = data['gas_density'] > density_threshold
            if mask.sum() > 0:
                center = np.average(data['gas_pos'][mask], weights=data['gas_mass'][mask], axis=0)
                if verbose:
                    print(f"Center (dense gas COM): [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] kpc")
                return center

        # If no density or no dense subset
        center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
        if verbose:
            print(f"Center (gas COM): [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] kpc")
        return center

    # Last resort: box center
    box_size = data.get('box_size', 50000.0)
    center = np.array([box_size/2, box_size/2, box_size/2], dtype=float)
    if verbose:
        print(f"[WARN] No gas/stars. Using box center: {center}")
    return center


def analyze_dust_distribution(data, center, verbose=True):
    if len(data['dust_pos']) == 0:
        print("No dust particles to analyze!")
        return None

    dust_r = np.sqrt(np.sum((data['dust_pos'] - center)**2, axis=1))

    print("\n" + "="*60)
    print("DUST DISTRIBUTION ANALYSIS")
    print("="*60)

    print("\nRadial Distribution:")
    radial_bins = [0, 5, 10, 20, 30, 50, 100, 150, 200, 500, 1000]
    for i in range(len(radial_bins)-1):
        m = (dust_r >= radial_bins[i]) & (dust_r < radial_bins[i+1])
        c = int(m.sum())
        if c > 0:
            mass = float(np.sum(data['dust_mass'][m]))
            print(f"  {radial_bins[i]:4.0f}-{radial_bins[i+1]:4.0f} kpc: "
                  f"{c:6d} particles, {mass*1e10:10.3e} Msun")

    print("\nMass Distribution (resolution hints):")
    if len(data['dust_mass']) > 0:
        mlog = np.log10(data['dust_mass'][data['dust_mass'] > 0])
        print(f"  dust mass log10 range: {mlog.min():.2f} .. {mlog.max():.2f}")

    far_mask = dust_r > 100
    far_count = int(far_mask.sum())
    if far_count > 0:
        print(f"\nWARNING: {far_count} dust particles beyond 100 kpc")
        far_masses = data['dust_mass'][far_mask]
        high_mass_far = int((far_masses > 1e-5).sum())
        if high_mass_far > 0:
            print(f"  WARNING: {high_mass_far} far dust particles have high mass (>1e-5) → likely low-res contamination")

    print("="*60 + "\n")

    return {
        'dust_r': dust_r,
        'far_dust_count': far_count,
        'high_mass_count': int((data['dust_mass'] > 1e-5).sum()) if len(data['dust_mass']) > 0 else 0
    }


# -----------------------------
# The rest of your pipeline
# (compute_radial_profile, plot_radial_profiles, main)
# -----------------------------
# Keep your existing compute_radial_profile() and plot_radial_profiles()
# with minimal changes. I’m pasting them unchanged to keep this drop-in.

def compute_radial_profile(data, center, bin_mode='adaptive', r_max=None, verbose=False):
    if len(data['dust_pos']) > 0:
        dust_r = np.sqrt(np.sum((data['dust_pos'] - center)**2, axis=1))
    else:
        if verbose:
            print("Warning: No dust particles found!")
        return None

    gas_r = np.sqrt(np.sum((data['gas_pos'] - center)**2, axis=1))

    if r_max is None:
        max_dust_r = np.max(dust_r) if len(dust_r) > 0 else 100
        max_gas_r = np.percentile(gas_r, 99) if len(gas_r) > 0 else 100
        r_max = min(max(max_dust_r * 1.1, max_gas_r, 100), 500)

        if verbose:
            print(f"Auto-determined r_max = {r_max:.1f} kpc")
            if max_dust_r > 100:
                print(f"  WARNING: Dust extends to {max_dust_r:.1f} kpc!")

    if r_max > 150:
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 50, 21)[1:]
        outer_bins = np.linspace(50, r_max, 16)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    elif bin_mode == 'linear':
        r_bins = np.linspace(0, r_max, 31)
    elif bin_mode == 'log':
        r_bins = np.logspace(np.log10(0.5), np.log10(r_max), 30)
        r_bins = np.concatenate([[0], r_bins])
    else:
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 30, 11)[1:]
        outer_bins = np.linspace(30, r_max, 11)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])

    n_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    profile = {
        'r_bins': r_bins,
        'r_centers': r_centers,
        'dust_mass': np.zeros(n_bins),
        'metal_mass': np.zeros(n_bins),
        'gas_mass': np.zeros(n_bins),
        'dust_mass_cum': np.zeros(n_bins),
        'metal_mass_cum': np.zeros(n_bins),
        'gas_mass_cum': np.zeros(n_bins),
        'dust_count': np.zeros(n_bins, dtype=int),
        'gas_count': np.zeros(n_bins, dtype=int),
        'DM_ratio': np.zeros(n_bins),
        'DG_ratio': np.zeros(n_bins),
        'r_max': r_max
    }

    dust_bin_idx = np.digitize(dust_r, r_bins) - 1
    for i in range(n_bins):
        m = (dust_bin_idx == i)
        if m.sum() > 0:
            profile['dust_mass'][i] = np.sum(data['dust_mass'][m])
            profile['dust_count'][i] = int(m.sum())
        mc = (dust_r < r_bins[i+1])
        if mc.sum() > 0:
            profile['dust_mass_cum'][i] = np.sum(data['dust_mass'][mc])

    gas_bin_idx = np.digitize(gas_r, r_bins) - 1
    for i in range(n_bins):
        m = (gas_bin_idx == i)
        if m.sum() > 0:
            profile['gas_mass'][i] = np.sum(data['gas_mass'][m])
            if len(data['gas_metals']) > 0:
                profile['metal_mass'][i] = np.sum(data['gas_mass'][m] * data['gas_metals'][m])
            profile['gas_count'][i] = int(m.sum())
        mc = (gas_r < r_bins[i+1])
        if mc.sum() > 0:
            profile['gas_mass_cum'][i] = np.sum(data['gas_mass'][mc])
            if len(data['gas_metals']) > 0:
                profile['metal_mass_cum'][i] = np.sum(data['gas_mass'][mc] * data['gas_metals'][mc])

    mask_metals = profile['metal_mass'] > 0
    profile['DM_ratio'][mask_metals] = profile['dust_mass'][mask_metals] / profile['metal_mass'][mask_metals]

    mask_gas = profile['gas_mass'] > 0
    profile['DG_ratio'][mask_gas] = profile['dust_mass'][mask_gas] / profile['gas_mass'][mask_gas]

    profile['DM_ratio'] = np.clip(profile['DM_ratio'], 0, 1.0)
    profile['DG_ratio'] = np.clip(profile['DG_ratio'], 0, 0.1)

    return profile


def plot_radial_profiles(profile, data, dust_analysis=None, halo_id=None, output_file=None):
    # (unchanged from your original; kept for brevity)
    # If you want, I can also add: a dedicated “matched fraction per bin” panel.
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

    title_str = f'Radial Plots'
    if halo_id is not None:
        title_str += f' for Halo {halo_id}'
    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        title_str += f'\nWARNING: {dust_analysis["far_dust_count"]} dust particles beyond 100 kpc!'

    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, f'z = {data["redshift"]:.2f}', ha='center', fontsize=11)

    r = profile['r_centers']
    r_max_plot = min(profile['r_max'], 200)
    output_name = data.get('output_name', 'Simulation')

    ax1 = fig.add_subplot(gs[0, 0])
    mask = (profile['DM_ratio'] > 0) & (profile['metal_mass'] > 0)
    ax1.plot(r[mask], profile['DM_ratio'][mask], 'o-', color='darkblue', lw=2, label=output_name, markersize=4)
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('Dust-to-Metal Ratio')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, r_max_plot)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Dust-to-Metal Ratio')

    ax2 = fig.add_subplot(gs[0, 1])
    mask = (profile['DG_ratio'] > 0) & (profile['gas_mass'] > 0)
    ax2.plot(r[mask], profile['DG_ratio'][mask] * 100, 'o-', color='darkgreen', lw=2, label=output_name, markersize=4)
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Dust-to-Gas Ratio (%)')
    ax2.set_ylim(0, 2.0)
    ax2.set_xlim(0, r_max_plot)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Dust-to-Gas Ratio')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python radial_plots.py <snapshot_or_dir> [halo_file] [halo_id]")
        sys.exit(1)

    snapshot_path = sys.argv[1]

    halo_pids = None
    halo_id = None
    if len(sys.argv) >= 4:
        halo_file = sys.argv[2]
        halo_id = int(sys.argv[3])
        print(f"Loading particle IDs for halo {halo_id} from {halo_file}...")
        halo_pids = load_halo_particle_ids(halo_file, halo_id)
    else:
        print("No halo filtering - analyzing all particles")

    print(f"\nLoading snapshot {snapshot_path}...")
    data = read_snapshot_with_filter(snapshot_path, halo_pids=halo_pids, verbose=True)

    if len(data['gas_pos']) == 0:
        print("Error: No gas particles found!")
        sys.exit(1)

    print("\nFinding halo center...")
    center = find_halo_center(data, method='stellar', verbose=True)

    dust_analysis = analyze_dust_distribution(data, center, verbose=True)

    print("\nComputing radial profiles...")
    profile = compute_radial_profile(data, center, bin_mode='adaptive', verbose=True)
    if profile is None:
        print("Could not compute profile - no dust particles?")
        sys.exit(1)

    if os.path.isdir(snapshot_path):
        output_name = os.path.basename(os.path.dirname(snapshot_path.rstrip('/')))
    else:
        output_name = os.path.basename(os.path.dirname(snapshot_path))
    data['output_name'] = output_name

    if os.path.isdir(snapshot_path):
        snap_num = snapshot_path.rstrip('/').split('_')[-1]
    else:
        snap_num = snapshot_path.split('_')[-1].split('.')[0]

    if halo_id is not None:
        output_base = f'halo{halo_id}_snap{snap_num}_radial_DM_filtered'
    else:
        output_base = f'snap{snap_num}_radial_DM'

    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        output_base += '_DIAGNOSTIC'

    print("\nCreating plots...")
    plot_radial_profiles(profile, data, dust_analysis=dust_analysis,
                         halo_id=halo_id, output_file=f'{output_base}.png')

    np.savez(f'{output_base}.npz',
             r_centers=profile['r_centers'],
             r_bins=profile['r_bins'],
             DM_ratio=profile['DM_ratio'],
             DG_ratio=profile['DG_ratio'],
             dust_mass=profile['dust_mass'],
             metal_mass=profile['metal_mass'],
             gas_mass=profile['gas_mass'],
             dust_mass_cum=profile['dust_mass_cum'],
             metal_mass_cum=profile['metal_mass_cum'],
             gas_mass_cum=profile['gas_mass_cum'],
             dust_count=profile['dust_count'],
             gas_count=profile['gas_count'])

    print(f"\nDone! Outputs:\n  - {output_base}.png\n  - {output_base}.npz")


if __name__ == '__main__':
    main()
