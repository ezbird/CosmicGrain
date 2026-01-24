#!/usr/bin/env python3
"""Find center of mass of stars in high-resolution region"""

import numpy as np
import h5py
import glob
import sys
import os

def read_stars(snapshot_path):
    # Handle directory or file input
    if os.path.isdir(snapshot_path):
        # It's a directory - find snapshot files in it
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        # It's a file - use multi-file logic
        base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not snapshot_files:
            snapshot_files = [snapshot_path]
    
    print(f"Reading {len(snapshot_files)} file(s)...")
    if len(snapshot_files) > 1:
        print(f"First: {os.path.basename(snapshot_files[0])}")
        print(f"Last:  {os.path.basename(snapshot_files[-1])}")
    
    star_pos, star_mass = [], []
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            if 'PartType4' in f:
                star_pos.append(f['PartType4/Coordinates'][:])
                star_mass.append(f['PartType4/Masses'][:])
    
    if star_pos:
        star_pos = np.concatenate(star_pos)
        star_mass = np.concatenate(star_mass)
    else:
        star_pos = np.array([])
        star_mass = np.array([])
    
    return star_pos, star_mass

if len(sys.argv) < 2:
    print("Usage: python find_halo_center.py <snapshot_or_dir>")
    print("\nExamples:")
    print("  python find_halo_center.py snapshot_026.hdf5")
    print("  python find_halo_center.py snapshot_026.0.hdf5")
    print("  python find_halo_center.py snapdir_026/")
    sys.exit(1)

snapshot = sys.argv[1]
print(f"Reading {snapshot}...")

star_pos, star_mass = read_stars(snapshot)

if len(star_pos) == 0:
    print("No stars found!")
    sys.exit(1)

# Overall center of mass
com = np.average(star_pos, weights=star_mass, axis=0)
print(f"\nAll stars COM: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}] kpc")
print(f"Total stellar mass: {np.sum(star_mass):.2e}")
print(f"Number of stars: {len(star_pos)}")

# Find densest region (most massive halo) - only if enough stars
dense_com = com.copy()  # Initialize with overall COM
if len(star_pos) > 100:
    mass_threshold = np.percentile(star_mass, 90)
    dense_mask = star_mass > mass_threshold
    if np.sum(dense_mask) > 0:
        dense_com = np.average(star_pos[dense_mask], weights=star_mass[dense_mask], axis=0)
        print(f"\nDense region COM (top 10% by mass): [{dense_com[0]:.2f}, {dense_com[1]:.2f}, {dense_com[2]:.2f}] kpc")
else:
    print(f"\nToo few stars ({len(star_pos)}) to find dense region - using overall COM")

# Iterative refinement: find center of densest sphere
print(f"\nIterative refinement:")
center = com.copy()
radii = [100, 50, 30, 20, 10]  # Try different radii
for radius in radii:
    r = np.sqrt(np.sum((star_pos - center)**2, axis=1))
    mask = r < radius
    n_within = np.sum(mask)
    if n_within > 10:
        new_center = np.average(star_pos[mask], weights=star_mass[mask], axis=0)
        shift = np.sqrt(np.sum((new_center - center)**2))
        center = new_center
        total_mass = np.sum(star_mass[mask])
        print(f"  R={radius:3d} kpc: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] - {n_within:5d} stars, mass={total_mass:.2e}, shift={shift:.2f} kpc")
        if shift < 1.0:  # Converged
            break
    else:
        print(f"  R={radius:3d} kpc: Only {n_within} stars - skipping")

print(f"\n*** Use this center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] ***")
