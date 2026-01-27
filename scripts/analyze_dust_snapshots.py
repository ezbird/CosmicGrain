#!/usr/bin/env python3
"""
Comprehensive dust analysis from GADGET snapshots
Reads all particles from all ranks for accurate statistics
"""

import numpy as np
import h5py
import glob
import sys
import os

def read_snapshot(snap_path):
    """
    Read all files for a snapshot
    Handles both:
      - snapshot_049/ (directory with snapshot_049.*.hdf5 inside)
      - snapshot_049 (base name, looks for snapshot_049.*.hdf5)
    """
    # Remove trailing slash
    snap_path = snap_path.rstrip('/')
    
    # Determine if input is a directory or base name
    if os.path.isdir(snap_path):
        # It's a directory like snapshot_049/
        snap_name = os.path.basename(snap_path)
        pattern = os.path.join(snap_path, f"{snap_name}.*.hdf5")
    else:
        # It's a base name like snapshot_049
        pattern = f"{snap_path}.*.hdf5"
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"ERROR: No files found matching: {pattern}")
        return None
    
    print(f"Found {len(files)} snapshot files")
    
    data = {
        'dust_pos': [],
        'dust_vel': [],
        'dust_mass': [],
        'dust_radius': [],
        'dust_carbon': [],
        'gas_pos': [],
        'gas_vel': [],
        'gas_density': [],
        'gas_temp': [],
    }
    
    for f in files:
        with h5py.File(f, 'r') as hf:
            # Read header (only once)
            if f == files[0]:
                header = dict(hf['Header'].attrs)
                data['redshift'] = header['Redshift']
                data['time'] = header['Time']
                data['boxsize'] = header['BoxSize']
                print(f"Redshift: {data['redshift']:.3f}")
            
            # Read dust (PartType6)
            if 'PartType6' in hf:
                pt6 = hf['PartType6']
                n_dust = len(pt6['Masses'])
                if n_dust > 0:
                    data['dust_pos'].append(pt6['Coordinates'][:])
                    data['dust_vel'].append(pt6['Velocities'][:])
                    data['dust_mass'].append(pt6['Masses'][:])
                    
                    # Dust-specific fields
                    if 'GrainRadius' in pt6:
                        data['dust_radius'].append(pt6['GrainRadius'][:])
                    if 'CarbonFraction' in pt6:
                        data['dust_carbon'].append(pt6['CarbonFraction'][:])
                    
                    print(f"  {os.path.basename(f)}: {n_dust:,} dust particles")
            
            # Read gas (PartType0) for nearest neighbor analysis
            if 'PartType0' in hf:
                pt0 = hf['PartType0']
                data['gas_pos'].append(pt0['Coordinates'][:])
                data['gas_vel'].append(pt0['Velocities'][:])
                data['gas_density'].append(pt0['Density'][:])
                # Temperature if available
                if 'Temperature' in pt0:
                    data['gas_temp'].append(pt0['Temperature'][:])
    
    # Concatenate arrays
    for key in data:
        if isinstance(data[key], list) and len(data[key]) > 0:
            data[key] = np.concatenate(data[key])
            if key == 'dust_pos':
                print(f"Total dust particles: {len(data[key]):,}")
            elif key == 'gas_pos':
                print(f"Total gas particles: {len(data[key]):,}")
    
    return data

def calc_grain_distribution(radii, masses):
    """Calculate grain size distribution"""
    bins = [0, 10, 50, 100, 150, 200, 500]
    counts = np.histogram(radii, bins=bins)[0]
    mass_bins = []
    
    for i in range(len(bins)-1):
        mask = (radii >= bins[i]) & (radii < bins[i+1])
        mass_bins.append(np.sum(masses[mask]))
    
    return bins, counts, mass_bins

def calc_radial_distribution(pos, center, masses):
    """Calculate radial distribution"""
    dx = pos - center
    r = np.sqrt(np.sum(dx**2, axis=1))
    
    bins = [0, 10, 50, 100, 250, 500, 1000, 2000, 5000]
    counts = np.histogram(r, bins=bins)[0]
    mass_bins = []
    
    for i in range(len(bins)-1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        mass_bins.append(np.sum(masses[mask]))
    
    return bins, counts, mass_bins, r

def find_nearest_gas(dust_pos, gas_pos, gas_vel, max_search=10.0):
    """Find nearest gas particle to each dust particle"""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("WARNING: scipy not available, skipping coupling analysis")
        return None, None, np.zeros(len(dust_pos), dtype=bool)
    
    print(f"  Building KD-tree for {len(gas_pos):,} gas particles...")
    tree = cKDTree(gas_pos)
    
    print(f"  Finding nearest neighbors...")
    distances, indices = tree.query(dust_pos, k=1, distance_upper_bound=max_search)
    
    # Get gas velocities for nearest neighbors
    valid = distances < max_search
    gas_vel_nearest = np.zeros_like(dust_pos)
    gas_vel_nearest[valid] = gas_vel[indices[valid]]
    
    return distances, gas_vel_nearest, valid

def analyze_snapshot(snap_path, halo_center=None):
    """Complete dust analysis for one snapshot"""
    print(f"\n{'='*70}")
    print(f"Reading: {snap_path}")
    print(f"{'='*70}")
    
    data = read_snapshot(snap_path)
    if data is None:
        return None
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS: z = {data['redshift']:.3f}")
    print(f"{'='*70}\n")
    
    if len(data['dust_mass']) == 0:
        print("No dust particles found!")
        return None
    
    # Basic stats
    print(f"1. BASIC STATISTICS")
    print(f"   Dust particles: {len(data['dust_mass']):,}")
    print(f"   Total dust mass: {np.sum(data['dust_mass']):.3e} Msun")
    print(f"   Mean grain radius: {np.mean(data['dust_radius']):.2f} nm")
    print(f"   Median grain radius: {np.median(data['dust_radius']):.2f} nm")
    print()
    
    # Grain size distribution
    print(f"2. GRAIN SIZE DISTRIBUTION")
    bins, counts, masses = calc_grain_distribution(data['dust_radius'], data['dust_mass'])
    total_count = np.sum(counts)
    total_mass = np.sum(masses)
    
    for i in range(len(bins)-1):
        frac_num = 100 * counts[i] / total_count if total_count > 0 else 0
        frac_mass = 100 * masses[i] / total_mass if total_mass > 0 else 0
        print(f"   [{bins[i]:3.0f}-{bins[i+1]:3.0f} nm]: {counts[i]:6d} ({frac_num:5.1f}%), {masses[i]:.2e} Msun ({frac_mass:5.1f}%)")
    print()
    
    # Radial distribution
    if halo_center is None:
        halo_center = np.array([data['boxsize']/2]*3)
        print(f"   (Using box center as halo center)")
    
    print(f"3. RADIAL DISTRIBUTION")
    print(f"   Center: ({halo_center[0]:.1f}, {halo_center[1]:.1f}, {halo_center[2]:.1f}) kpc")
    bins, counts, masses, radii = calc_radial_distribution(data['dust_pos'], halo_center, data['dust_mass'])
    
    for i in range(len(bins)-1):
        frac_num = 100 * counts[i] / total_count if total_count > 0 else 0
        frac_mass = 100 * masses[i] / total_mass if total_mass > 0 else 0
        print(f"   [{bins[i]:4.0f}-{bins[i+1]:4.0f} kpc]: {counts[i]:6d} ({frac_num:5.1f}%), {masses[i]:.2e} Msun ({frac_mass:5.1f}%)")
    
    print(f"\n   Median radius: {np.median(radii):.1f} kpc")
    print(f"   90th percentile: {np.percentile(radii, 90):.1f} kpc")
    print(f"   99th percentile: {np.percentile(radii, 99):.1f} kpc")
    print(f"   Max radius: {np.max(radii):.1f} kpc")
    
    n_escape = np.sum(radii > 1000)
    if n_escape > 0:
        print(f"   ⚠️  {n_escape:,} particles beyond 1000 kpc ({100*n_escape/len(radii):.1f}%)")
    print()
    
    # Dust-gas coupling
    print(f"4. DUST-GAS COUPLING")
    if len(data['gas_pos']) > 0:
        distances, gas_vel_nearest, valid = find_nearest_gas(
            data['dust_pos'], data['gas_pos'], data['gas_vel']
        )
        
        if gas_vel_nearest is not None:
            dv = data['dust_vel'] - gas_vel_nearest
            dv_mag = np.sqrt(np.sum(dv**2, axis=1))
            
            print(f"   Dust with nearby gas: {np.sum(valid):,} / {len(valid):,} ({100*np.sum(valid)/len(valid):.1f}%)")
            if np.sum(valid) > 0:
                print(f"   Mean Δv: {np.mean(dv_mag[valid]):.1f} km/s")
                print(f"   Median Δv: {np.median(dv_mag[valid]):.1f} km/s")
                print(f"   90th %ile Δv: {np.percentile(dv_mag[valid], 90):.1f} km/s")
                print(f"   Dust with Δv > 50 km/s: {np.sum(dv_mag[valid] > 50):,} ({100*np.sum(dv_mag[valid] > 50)/np.sum(valid):.1f}%)")
                print(f"   Dust with Δv < 10 km/s: {np.sum(dv_mag[valid] < 10):,} ({100*np.sum(dv_mag[valid] < 10)/np.sum(valid):.1f}%)")
            
            print(f"\n   Coupling by radius:")
            rad_bins = [0, 50, 100, 500, 1000, 2000]
            for i in range(len(rad_bins)-1):
                mask = valid & (radii >= rad_bins[i]) & (radii < rad_bins[i+1])
                if np.sum(mask) > 0:
                    print(f"     {rad_bins[i]:4d}-{rad_bins[i+1]:4d} kpc: Δv = {np.median(dv_mag[mask]):5.1f} km/s (n={np.sum(mask):,})")
    else:
        print("   No gas data")
    print()
    
    # Carbon fraction
    if len(data.get('dust_carbon', [])) > 0:
        print(f"5. COMPOSITION")
        carb = data['dust_carbon']
        sil_mask = carb < 0.5
        carb_mask = carb >= 0.5
        
        print(f"   Silicate (CF<0.5): {np.sum(sil_mask):,} ({100*np.sum(sil_mask)/len(carb):.1f}%)")
        print(f"   Carbon (CF≥0.5): {np.sum(carb_mask):,} ({100*np.sum(carb_mask)/len(carb):.1f}%)")
        print()
    
    return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze dust from GADGET snapshots')
    parser.add_argument('snapshots', nargs='+', help='Snapshot directories or base names')
    parser.add_argument('--center', nargs=3, type=float, help='Halo center (x y z) in kpc')
    args = parser.parse_args()
    
    halo_center = np.array(args.center) if args.center else None
    
    for snap in args.snapshots:
        analyze_snapshot(snap, halo_center)
