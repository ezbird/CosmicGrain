#!/usr/bin/env python3
"""
Quick diagnostic: Where are the dust particles actually located?
"""

import numpy as np
import h5py
import glob
import sys

if len(sys.argv) < 2:
    print("Usage: python dust_location_check.py <snapshot_dir_or_file>")
    sys.exit(1)

snap_path = sys.argv[1]

# Find snapshot files
if snap_path.endswith('.hdf5'):
    base_path = snap_path.replace('.0.hdf5', '').replace('.hdf5', '')
else:
    # Directory - find latest snapshot
    import os
    snap_files = sorted(glob.glob(os.path.join(snap_path, 'snapshot_*.0.hdf5')))
    if snap_files:
        base_path = snap_files[-1].replace('.0.hdf5', '')
    else:
        snap_files = sorted(glob.glob(os.path.join(snap_path, 'snapshot_*.hdf5')))
        base_path = snap_files[-1].replace('.hdf5', '')

snap_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
if not snap_files:
    snap_files = [base_path + '.hdf5']

print(f"Reading {len(snap_files)} file(s)...")

# Read all particles
dust_pos = []
dust_mass = []
gas_pos = []
gas_mass = []
star_pos = []
star_mass = []

for snap_file in snap_files:
    print(f"  Reading {snap_file}...")
    with h5py.File(snap_file, 'r') as f:
        # Dust
        if 'PartType6' in f:
            dust_pos.append(f['PartType6/Coordinates'][:])
            dust_mass.append(f['PartType6/Masses'][:])
            print(f"    Dust: {len(f['PartType6/Masses'])} particles")
        
        # Gas
        if 'PartType0' in f:
            gas_pos.append(f['PartType0/Coordinates'][:])
            gas_mass.append(f['PartType0/Masses'][:])
            print(f"    Gas:  {len(f['PartType0/Masses'])} particles")
        
        # Stars
        if 'PartType4' in f:
            star_pos.append(f['PartType4/Coordinates'][:])
            star_mass.append(f['PartType4/Masses'][:])
            print(f"    Stars: {len(f['PartType4/Masses'])} particles")

# Concatenate
dust_pos = np.concatenate(dust_pos) if dust_pos else np.array([])
dust_mass = np.concatenate(dust_mass) if dust_mass else np.array([])
gas_pos = np.concatenate(gas_pos) if gas_pos else np.array([])
gas_mass = np.concatenate(gas_mass) if gas_mass else np.array([])
star_pos = np.concatenate(star_pos) if star_pos else np.array([])
star_mass = np.concatenate(star_mass) if star_mass else np.array([])

print("\n" + "="*70)
print("PARTICLE LOCATIONS")
print("="*70)

# Compute centers
if len(star_pos) > 0:
    star_center = np.average(star_pos, weights=star_mass, axis=0)
    print(f"\nStellar center: [{star_center[0]:.2f}, {star_center[1]:.2f}, {star_center[2]:.2f}]")
    print(f"Stellar extent:")
    print(f"  X: {star_pos[:, 0].min():.2f} to {star_pos[:, 0].max():.2f} (range: {star_pos[:, 0].max() - star_pos[:, 0].min():.2f})")
    print(f"  Y: {star_pos[:, 1].min():.2f} to {star_pos[:, 1].max():.2f} (range: {star_pos[:, 1].max() - star_pos[:, 1].min():.2f})")
    print(f"  Z: {star_pos[:, 2].min():.2f} to {star_pos[:, 2].max():.2f} (range: {star_pos[:, 2].max() - star_pos[:, 2].min():.2f})")
else:
    star_center = None
    print("\nNo stars found!")

if len(gas_pos) > 0:
    gas_center = np.average(gas_pos, weights=gas_mass, axis=0)
    print(f"\nGas center: [{gas_center[0]:.2f}, {gas_center[1]:.2f}, {gas_center[2]:.2f}]")
    print(f"Gas extent:")
    print(f"  X: {gas_pos[:, 0].min():.2f} to {gas_pos[:, 0].max():.2f} (range: {gas_pos[:, 0].max() - gas_pos[:, 0].min():.2f})")
    print(f"  Y: {gas_pos[:, 1].min():.2f} to {gas_pos[:, 1].max():.2f} (range: {gas_pos[:, 1].max() - gas_pos[:, 1].min():.2f})")
    print(f"  Z: {gas_pos[:, 2].min():.2f} to {gas_pos[:, 2].max():.2f} (range: {gas_pos[:, 2].max() - gas_pos[:, 2].min():.2f})")

if len(dust_pos) > 0:
    dust_center = np.average(dust_pos, weights=dust_mass, axis=0)
    print(f"\nDust center: [{dust_center[0]:.2f}, {dust_center[1]:.2f}, {dust_center[2]:.2f}]")
    print(f"Dust extent:")
    print(f"  X: {dust_pos[:, 0].min():.2f} to {dust_pos[:, 0].max():.2f} (range: {dust_pos[:, 0].max() - dust_pos[:, 0].min():.2f})")
    print(f"  Y: {dust_pos[:, 1].min():.2f} to {dust_pos[:, 1].max():.2f} (range: {dust_pos[:, 1].max() - dust_pos[:, 1].min():.2f})")
    print(f"  Z: {dust_pos[:, 2].min():.2f} to {dust_pos[:, 2].max():.2f} (range: {dust_pos[:, 2].max() - dust_pos[:, 2].min():.2f})")
    
    if star_center is not None:
        # Radial distribution from stellar center
        dust_r = np.linalg.norm(dust_pos - star_center, axis=1)
        print(f"\nDust radial distribution (from stellar center):")
        print(f"  Min radius: {dust_r.min():.2f} kpc")
        print(f"  Median radius: {np.median(dust_r):.2f} kpc")
        print(f"  90th percentile: {np.percentile(dust_r, 90):.2f} kpc")
        print(f"  Max radius: {dust_r.max():.2f} kpc")
        
        # How many dust particles in different regions?
        n_inner = np.sum(dust_r < 5)
        n_disk = np.sum((dust_r >= 5) & (dust_r < 15))
        n_outer = np.sum((dust_r >= 15) & (dust_r < 30))
        n_halo = np.sum(dust_r >= 30)
        
        print(f"\nDust particle distribution:")
        print(f"  r < 5 kpc (inner disk):  {n_inner:6,} ({100*n_inner/len(dust_r):5.1f}%)")
        print(f"  5-15 kpc (disk):         {n_disk:6,} ({100*n_disk/len(dust_r):5.1f}%)")
        print(f"  15-30 kpc (outer disk):  {n_outer:6,} ({100*n_outer/len(dust_r):5.1f}%)")
        print(f"  r > 30 kpc (halo):       {n_halo:6,} ({100*n_halo/len(dust_r):5.1f}%)")
        
        if n_halo > 0.5 * len(dust_r):
            print(f"\n⚠️  WARNING: >50% of dust is in the halo (r > 30 kpc)!")
            print(f"    This is unusual - dust should be concentrated in the disk.")
            print(f"    Possible issues:")
            print(f"      - Wrong stellar center calculation")
            print(f"      - Dust in strong outflows")
            print(f"      - Bug in dust particle spawning locations")

print("="*70)
