#!/usr/bin/env python3
"""
Find where dust particles are distributed relative to a halo center
Helps diagnose why dust particles aren't being captured in sphere extraction
"""

import numpy as np
import h5py
import glob
import sys
import matplotlib.pyplot as plt

def read_snapshot_positions(snapshot_path):
    """Read particle positions and IDs"""
    
    base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
    snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
    if not snapshot_files:
        snapshot_files = [snapshot_path]
    
    print(f"Reading {len(snapshot_files)} file(s)...")
    
    data = {
        'dust_pos': [], 'dust_ids': [],
        'gas_pos': [], 'gas_ids': [],
        'star_pos': [], 'star_ids': []
    }
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            # Dust (PartType6)
            if 'PartType6' in f:
                data['dust_pos'].append(f['PartType6/Coordinates'][:])
                data['dust_ids'].append(f['PartType6/ParticleIDs'][:])
            
            # Gas
            if 'PartType0' in f:
                data['gas_pos'].append(f['PartType0/Coordinates'][:])
                data['gas_ids'].append(f['PartType0/ParticleIDs'][:])
            
            # Stars
            if 'PartType4' in f:
                data['star_pos'].append(f['PartType4/Coordinates'][:])
                data['star_ids'].append(f['PartType4/ParticleIDs'][:])
    
    # Concatenate
    for key in data:
        if data[key]:
            data[key] = np.concatenate(data[key])
        else:
            data[key] = np.array([])
    
    return data


def analyze_dust_distribution(data, halo_center, halo_radius=200):
    """Analyze where dust particles are relative to halo"""
    
    print(f"\n{'='*60}")
    print(f"Halo center: [{halo_center[0]:.2f}, {halo_center[1]:.2f}, {halo_center[2]:.2f}] kpc")
    print(f"Search radius: {halo_radius} kpc")
    print(f"{'='*60}")
    
    # Overall particle counts
    print(f"\nTotal particles in snapshot:")
    print(f"  Gas:   {len(data['gas_pos']):,}")
    print(f"  Stars: {len(data['star_pos']):,}")
    print(f"  Dust:  {len(data['dust_pos']):,}")
    
    if len(data['dust_pos']) == 0:
        print("\nERROR: No dust particles in snapshot!")
        return
    
    # Compute distances from halo center
    dust_r = np.sqrt(np.sum((data['dust_pos'] - halo_center)**2, axis=1))
    gas_r = np.sqrt(np.sum((data['gas_pos'] - halo_center)**2, axis=1))
    star_r = np.sqrt(np.sum((data['star_pos'] - halo_center)**2, axis=1))
    
    # Particles within halo radius
    dust_in_halo = np.sum(dust_r < halo_radius)
    gas_in_halo = np.sum(gas_r < halo_radius)
    star_in_halo = np.sum(star_r < halo_radius)
    
    print(f"\nParticles within {halo_radius} kpc:")
    print(f"  Gas:   {gas_in_halo:,} ({100*gas_in_halo/len(gas_r):.1f}%)")
    print(f"  Stars: {star_in_halo:,} ({100*star_in_halo/len(star_r):.1f}%)")
    print(f"  Dust:  {dust_in_halo:,} ({100*dust_in_halo/len(dust_r):.1f}%)")
    
    # Dust distribution statistics
    print(f"\nDust radial distribution:")
    print(f"  Minimum:  {np.min(dust_r):.1f} kpc")
    print(f"  Median:   {np.median(dust_r):.1f} kpc")
    print(f"  Mean:     {np.mean(dust_r):.1f} kpc")
    print(f"  Maximum:  {np.max(dust_r):.1f} kpc")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        r_p = np.percentile(dust_r, p)
        print(f"    {p:2d}%: {r_p:7.1f} kpc")
    
    # Find recommended radius to capture most dust
    for target_fraction in [0.80, 0.90, 0.95, 0.99]:
        radius_needed = np.percentile(dust_r, target_fraction * 100)
        count = np.sum(dust_r < radius_needed)
        print(f"\n  To capture {target_fraction*100:.0f}% of dust ({count:,} particles):")
        print(f"    Need radius: {radius_needed:.1f} kpc")
    
    # Check spatial distribution
    print(f"\nDust spatial extent:")
    print(f"  X: [{np.min(data['dust_pos'][:,0]):.1f}, {np.max(data['dust_pos'][:,0]):.1f}] kpc")
    print(f"  Y: [{np.min(data['dust_pos'][:,1]):.1f}, {np.max(data['dust_pos'][:,1]):.1f}] kpc")
    print(f"  Z: [{np.min(data['dust_pos'][:,2]):.1f}, {np.max(data['dust_pos'][:,2]):.1f}] kpc")
    
    print(f"\nHalo center:")
    print(f"  X: {halo_center[0]:.1f} kpc")
    print(f"  Y: {halo_center[1]:.1f} kpc")
    print(f"  Z: {halo_center[2]:.1f} kpc")
    
    # Make diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Radial histogram
    ax = axes[0, 0]
    bins = np.linspace(0, np.min([np.max(dust_r), 1000]), 50)
    ax.hist(dust_r, bins=bins, alpha=0.7, label='Dust', color='brown')
    ax.hist(gas_r[::100], bins=bins, alpha=0.5, label='Gas (sampled)', color='blue')
    ax.axvline(halo_radius, color='red', ls='--', lw=2, label=f'Search radius ({halo_radius} kpc)')
    ax.set_xlabel('Distance from halo center [kpc]')
    ax.set_ylabel('Number of particles')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Radial Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_dust_r = np.sort(dust_r)
    cumulative = np.arange(len(sorted_dust_r)) / len(sorted_dust_r)
    ax.plot(sorted_dust_r, cumulative * 100, 'o-', color='brown', markersize=2)
    ax.axvline(halo_radius, color='red', ls='--', lw=2, label=f'Search radius')
    ax.axhline(90, color='gray', ls=':', alpha=0.5, label='90%')
    ax.axhline(95, color='gray', ls=':', alpha=0.5, label='95%')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Cumulative fraction [%]')
    ax.set_xlim(0, np.min([np.max(dust_r), 1000]))
    ax.legend()
    ax.set_title('Cumulative Dust Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. XY projection
    ax = axes[1, 0]
    # Plot subset for clarity
    sample_size = min(1000, len(data['dust_pos']))
    sample_idx = np.random.choice(len(data['dust_pos']), sample_size, replace=False)
    ax.scatter(data['dust_pos'][sample_idx, 0], data['dust_pos'][sample_idx, 1], 
               s=1, alpha=0.5, color='brown', label='Dust')
    
    # Halo center and radius
    circle = plt.Circle((halo_center[0], halo_center[1]), halo_radius, 
                       fill=False, color='red', ls='--', lw=2, label=f'{halo_radius} kpc sphere')
    ax.add_patch(circle)
    ax.plot(halo_center[0], halo_center[1], 'r*', markersize=15, label='Halo center')
    
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('XY Projection')
    
    # 4. XZ projection
    ax = axes[1, 1]
    ax.scatter(data['dust_pos'][sample_idx, 0], data['dust_pos'][sample_idx, 2], 
               s=1, alpha=0.5, color='brown', label='Dust')
    
    # Halo center and radius
    circle = plt.Circle((halo_center[0], halo_center[2]), halo_radius, 
                       fill=False, color='red', ls='--', lw=2, label=f'{halo_radius} kpc sphere')
    ax.add_patch(circle)
    ax.plot(halo_center[0], halo_center[2], 'r*', markersize=15, label='Halo center')
    
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Z [kpc]')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('XZ Projection')
    
    plt.tight_layout()
    output_file = 'dust_distribution_diagnostic.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 5:
        print("Diagnose dust particle distribution relative to halo")
        print("\nUsage:")
        print("  python diagnose_dust_location.py <snapshot> <halo_x> <halo_y> <halo_z> [search_radius]")
        print("\nExample:")
        print("  python diagnose_dust_location.py snapshot_049.0.hdf5 23090 23624 23693 200")
        sys.exit(1)
    
    snapshot_path = sys.argv[1]
    halo_center = np.array([float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
    halo_radius = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0
    
    print("Loading snapshot...")
    data = read_snapshot_positions(snapshot_path)
    
    print(f"\nAnalyzing dust distribution...")
    analyze_dust_distribution(data, halo_center, halo_radius)


if __name__ == '__main__':
    main()
