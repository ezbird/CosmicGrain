#!/usr/bin/env python3
"""
Analyze dust grain sizes from Gadget-4 snapshots
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob

def read_dust_data(snapshot_base):
    """Read all dust particle data from multi-file snapshot"""
    
    # Find all files for this snapshot
    files = sorted(glob.glob(f"{snapshot_base}.[0-9]*.hdf5"))
    
    if not files:
        files = [f"{snapshot_base}.hdf5"]
    
    grain_sizes = []
    dust_masses = []
    dust_temps = []
    
    for fname in files:
        with h5py.File(fname, 'r') as f:
            if 'PartType6' not in f:
                continue
            
            # Read dust properties
            if 'GrainRadius' in f['PartType6']:
                sizes = f['PartType6']['GrainRadius'][:]
                grain_sizes.extend(sizes)
            
            if 'Masses' in f['PartType6']:
                masses = f['PartType6']['Masses'][:]
                dust_masses.extend(masses)
            
            if 'DustTemperature' in f['PartType6']:
                temps = f['PartType6']['DustTemperature'][:]
                dust_temps.extend(temps)
    
    return {
        'grain_sizes': np.array(grain_sizes),
        'dust_masses': np.array(dust_masses),
        'dust_temps': np.array(dust_temps)
    }

def plot_grain_size_histogram(data, outfile='grain_size_hist.png'):
    """Plot histogram of grain sizes"""
    
    sizes_micron = data['grain_sizes'] * 1e4  # Convert cm to microns
    masses = data['dust_masses']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Number-weighted histogram
    ax1.hist(sizes_micron, bins=50, range=(0, 0.35), 
             alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Grain size [μm]', fontsize=12)
    ax1.set_ylabel('Number of grains', fontsize=12)
    ax1.set_title('Grain Size Distribution (Number)', fontsize=13, fontweight='bold')
    ax1.axvline(np.median(sizes_micron), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(sizes_micron):.3f} μm')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Mass-weighted histogram
    ax2.hist(sizes_micron, bins=50, range=(0, 0.35), 
             weights=masses, alpha=0.7, edgecolor='black', color='green')
    ax2.set_xlabel('Grain size [μm]', fontsize=12)
    ax2.set_ylabel('Dust mass [code units]', fontsize=12)
    ax2.set_title('Grain Size Distribution (Mass)', fontsize=13, fontweight='bold')
    
    # Calculate mass-weighted mean
    mass_weighted_mean = np.average(sizes_micron, weights=masses)
    ax2.axvline(mass_weighted_mean, color='red', linestyle='--', 
                linewidth=2, label=f'Mass-weighted: {mass_weighted_mean:.3f} μm')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.close()

def compare_snapshots(snap_list):
    """Compare grain sizes across multiple snapshots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, snap_base in enumerate(snap_list[:6]):
        data = read_dust_data(snap_base)
        sizes_micron = data['grain_sizes'] * 1e4
        
        if len(sizes_micron) == 0:
            continue
        
        ax = axes[idx]
        ax.hist(sizes_micron, bins=40, range=(0, 0.35), 
                alpha=0.7, edgecolor='black')
        ax.set_xlabel('Grain size [μm]', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f"Snapshot {snap_base.split('_')[-1]}", fontsize=11)
        ax.axvline(np.median(sizes_micron), color='red', linestyle='--', linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Grain Size Evolution Across Snapshots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('snapshot_comparison.png', dpi=150)
    print("Saved: snapshot_comparison.png")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_dust_snapshots.py <snapshot_base> [snapshot_base2 ...]")
        print("Example: python analyze_dust_snapshots.py snapdir_047/snapshot_047")
        sys.exit(1)
    
    snap_bases = sys.argv[1:]
    
    if len(snap_bases) == 1:
        # Single snapshot analysis
        print(f"Analyzing {snap_bases[0]}...")
        data = read_dust_data(snap_bases[0])
        
        if len(data['grain_sizes']) > 0:
            sizes_micron = data['grain_sizes'] * 1e4
            print(f"\nGrain size statistics:")
            print(f"  Count: {len(sizes_micron)}")
            print(f"  Mean: {np.mean(sizes_micron):.4f} μm")
            print(f"  Median: {np.median(sizes_micron):.4f} μm")
            print(f"  Std dev: {np.std(sizes_micron):.4f} μm")
            print(f"  Min: {np.min(sizes_micron):.4f} μm")
            print(f"  Max: {np.max(sizes_micron):.4f} μm")
            
            plot_grain_size_histogram(data)
        else:
            print("No dust particles found!")
    else:
        # Multi-snapshot comparison
        print(f"Comparing {len(snap_bases)} snapshots...")
        compare_snapshots(snap_bases)
