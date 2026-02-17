#!/usr/bin/env python3
import h5py
import numpy as np
import glob

# Configuration
snapshot_base = '../7_output_zoom_1024_halo569_50Mpc_dust/snapdir_049/snapshot_049'
halo_center = np.array([23062.72, 23522.66, 23665.23])  # From your output

# Load all dust particles
print("Loading dust particles...")
files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))

all_coords = []
all_masses = []
all_vels = []

for file in files:
    with h5py.File(file, 'r') as f:
        if 'PartType6' not in f:
            continue
        coords = f['PartType6']['Coordinates'][:]
        masses = f['PartType6']['Masses'][:] * 1e10  # Convert to M☉
        vels = f['PartType6']['Velocities'][:]
        
        all_coords.append(coords)
        all_masses.append(masses)
        all_vels.append(vels)

coords = np.vstack(all_coords)
masses = np.concatenate(all_masses)
vels = np.vstack(all_vels)

print(f"Total dust particles: {len(masses):,}")
print(f"Total dust mass: {masses.sum():.2e} M☉")

# Compute distances from halo center
dx = coords - halo_center
r = np.sqrt((dx**2).sum(axis=1))

# Compute velocities
v = np.sqrt((vels**2).sum(axis=1))

print(f"\n{'='*70}")
print("RADIAL DISTRIBUTION")
print(f"{'='*70}")

bins = [0, 10, 30, 50, 100, 200, 500, 1000, 5000, 10000, 25000]
for i in range(len(bins)-1):
    mask = (r >= bins[i]) & (r < bins[i+1])
    count = mask.sum()
    if count > 0:
        mass = masses[mask].sum()
        frac = 100.0 * count / len(masses)
        mass_frac = 100.0 * mass / masses.sum()
        v_median = np.median(v[mask])
        print(f"  {bins[i]:5.0f}-{bins[i+1]:5.0f} kpc: {count:6d} particles ({frac:5.1f}%), "
              f"mass={mass:.2e} M☉ ({mass_frac:5.1f}%), v_med={v_median:.1f} km/s")

print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")
print(f"Median radius: {np.median(r):.1f} kpc")
print(f"Mean radius: {np.mean(r):.1f} kpc")
print(f"Max radius: {np.max(r):.1f} kpc")
print(f"\nMedian velocity: {np.median(v):.1f} km/s")
print(f"Mean velocity: {np.mean(v):.1f} km/s")
print(f"Max velocity: {np.max(v):.1f} km/s")

# Check for escapees
escaped = r > 1000
if escaped.sum() > 0:
    print(f"\n{'='*70}")
    print(f"WARNING: {escaped.sum():,} particles beyond 1 Mpc ({100.0*escaped.sum()/len(r):.1f}%)")
    print(f"         Mass: {masses[escaped].sum():.2e} M☉ ({100.0*masses[escaped].sum()/masses.sum():.1f}%)")
    print(f"         Median velocity: {np.median(v[escaped]):.1f} km/s")
    print(f"{'='*70}")

# Check velocities in different regions
print(f"\n{'='*70}")
print("VELOCITY BY REGION")
print(f"{'='*70}")
regions = [
    (0, 50, "Galaxy core"),
    (50, 200, "Halo"),
    (200, 1000, "CGM/Ejected"),
    (1000, 50000, "IGM/Escaped")
]

for r_min, r_max, label in regions:
    mask = (r >= r_min) & (r < r_max)
    if mask.sum() > 0:
        v_med = np.median(v[mask])
        v_mean = np.mean(v[mask])
        v_max = np.max(v[mask])
        count = mask.sum()
        print(f"  {label:20s}: {count:6d} particles, v = {v_med:6.1f} km/s (median), "
              f"{v_mean:6.1f} (mean), {v_max:6.1f} (max)")

# Add to find_escaped_dust.py
core_mask = r < 50
v_r = np.sum(dx[core_mask] * vels[core_mask], axis=1) / r[core_mask]  # Radial velocity

print(f"\nCore dust radial velocities:")
print(f"  Infall (v_r < -50 km/s): {(v_r < -50).sum()} particles")
print(f"  Outflow (v_r > +50 km/s): {(v_r > +50).sum()} particles")