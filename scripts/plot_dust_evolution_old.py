#!/usr/bin/env python3
"""
Plot dust evolution directly from Gadget-4 snapshot files.
Only counts high-resolution gas particles in the zoom region.

Usage:
    python plot_dust_from_snapshots.py /path/to/output/ [--highres-threshold 1e-5]
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)


def read_snapshot_masses(snap_path, highres_method='auto', mass_threshold=None):
    """
    Read dust mass and high-res gas mass from a snapshot.
    Handles distributed snapshots (snapshot split across multiple files).
    
    Args:
        snap_path: path to snapshot file (or first file if distributed)
        highres_method: 'auto' or 'threshold'
        mass_threshold: mass threshold for high-res identification (if using threshold)
    
    Returns:
        dict with: redshift, dust_mass, gas_mass_highres, gas_mass_total, 
                   n_dust, n_gas_highres, n_gas_total
    """
    result = {
        'redshift': None,
        'time': None,
        'dust_mass': 0.0,
        'gas_mass_highres': 0.0,
        'gas_mass_total': 0.0,
        'n_dust': 0,
        'n_gas_highres': 0,
        'n_gas_total': 0,
    }
    
    # Check if this is a distributed snapshot (snapshot_XXX.0.hdf5)
    snap_files = [snap_path]
    if '.0.hdf5' in str(snap_path):
        # Find all files for this snapshot
        base = str(snap_path).replace('.0.hdf5', '')
        snap_dir = snap_path.parent
        snap_files = sorted(snap_dir.glob(f"{snap_path.stem.split('.')[0]}.*.hdf5"))
    
    try:
        # Read header from first file
        with h5py.File(snap_files[0], 'r') as f:
            result['redshift'] = f['Header'].attrs['Redshift']
            result['time'] = f['Header'].attrs['Time']
        
        # Collect all gas masses first (for high-res detection)
        all_gas_masses = []
        for snap_file in snap_files:
            with h5py.File(snap_file, 'r') as f:
                if 'PartType0' in f:
                    all_gas_masses.append(f['PartType0']['Masses'][:])
        
        # Combine all gas masses
        if all_gas_masses:
            all_gas_masses = np.concatenate(all_gas_masses)
            result['gas_mass_total'] = np.sum(all_gas_masses)
            result['n_gas_total'] = len(all_gas_masses)
            
            # Identify high-res gas using combined masses
            if highres_method == 'threshold' and mass_threshold is not None:
                highres_mask = all_gas_masses < mass_threshold
            else:
                # Automatic detection using histogram
                hist, bin_edges = np.histogram(np.log10(all_gas_masses), bins=50)
                peak_bin = np.argmax(hist)
                peak_mass = 10**bin_edges[peak_bin]
                highres_mask = (all_gas_masses > peak_mass * 0.5) & (all_gas_masses < peak_mass * 2.0)
                
                if np.sum(highres_mask) / len(all_gas_masses) > 0.8:
                    # Probably not a zoom sim, use all gas
                    highres_mask = np.ones_like(all_gas_masses, dtype=bool)
            
            result['gas_mass_highres'] = np.sum(all_gas_masses[highres_mask])
            result['n_gas_highres'] = np.sum(highres_mask)
        
        # Read dust from all files
        for snap_file in snap_files:
            with h5py.File(snap_file, 'r') as f:
                if 'PartType6' in f:
                    dust_masses = f['PartType6']['Masses'][:]
                    result['dust_mass'] += np.sum(dust_masses)
                    result['n_dust'] += len(dust_masses)
    
    except Exception as e:
        print(f"  Error reading {snap_path}: {e}")
        return None
    
    return result


def find_snapshots(snapdir):
    """
    Find all snapshot files in directory.
    Handles both:
    - Snapshots directly in directory: snapshot_000.hdf5
    - Snapshots in subdirectories: snapdir_001/snapshot_001.0.hdf5
    """
    snapdir = Path(snapdir)
    snapfiles = []
    
    # First, look for snapshots directly in the directory
    snapfiles = sorted(snapdir.glob("snapshot_*.hdf5"))
    if not snapfiles:
        snapfiles = sorted(snapdir.glob("snap_*.hdf5"))
    
    # If not found, look in snapdir_* subdirectories
    if not snapfiles:
        snapdirs = sorted(snapdir.glob("snapdir_*"))
        for sdir in snapdirs:
            # Look for snapshot files (might be distributed: snapshot_XXX.0.hdf5, etc.)
            snap_candidates = sorted(sdir.glob("snapshot_*.hdf5"))
            if not snap_candidates:
                snap_candidates = sorted(sdir.glob("snap_*.hdf5"))
            
            # If distributed format (snapshot_000.0.hdf5), take only .0 files
            if snap_candidates:
                # Check if distributed format
                if '.0.hdf5' in str(snap_candidates[0]):
                    # Only keep .0 files (first part of distributed snapshot)
                    snap_candidates = [s for s in snap_candidates if '.0.hdf5' in str(s)]
                
                snapfiles.extend(snap_candidates)
    
    return snapfiles


def read_all_snapshots(snapdir, highres_method='auto', mass_threshold=None):
    """
    Read dust and gas evolution from all snapshots.
    
    Returns:
        dict with arrays: redshift, dust_mass, gas_mass_highres, etc.
    """
    snapfiles = find_snapshots(snapdir)
    
    if not snapfiles:
        print(f"ERROR: No snapshot files found in {snapdir}")
        return None
    
    print(f"Found {len(snapfiles)} snapshots")
    
    data = {
        'redshift': [],
        'dust_mass': [],
        'gas_mass_highres': [],
        'gas_mass_total': [],
        'n_dust': [],
        'n_gas_highres': [],
        'n_gas_total': [],
    }
    
    for i, snapfile in enumerate(snapfiles):
        if i % 10 == 0:
            print(f"  Reading snapshot {i+1}/{len(snapfiles)}...", end='\r')
        
        snap_data = read_snapshot_masses(snapfile, highres_method, mass_threshold)
        
        if snap_data is not None:
            for key in data.keys():
                data[key].append(snap_data[key])
    
    print(f"  Reading snapshot {len(snapfiles)}/{len(snapfiles)}... Done!")
    
    # Convert to numpy arrays
    for key in data.keys():
        data[key] = np.array(data[key])
    
    # Calculate D/G ratio
    data['dust_to_gas'] = np.zeros_like(data['dust_mass'])
    nonzero = data['gas_mass_highres'] > 0
    data['dust_to_gas'][nonzero] = data['dust_mass'][nonzero] / data['gas_mass_highres'][nonzero]
    
    return data


def plot_dust_evolution(data, output_file='dust_evolution.png'):
    """
    Create two-panel plot of dust evolution.
    
    Top panel: Gas mass and dust mass vs redshift
    Bottom panel: Dust-to-gas ratio vs redshift
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    z = data['redshift']
    dust_mass = data['dust_mass']
    gas_mass = data['gas_mass_highres']
    dg_ratio = data['dust_to_gas']
    
    # Top panel: Masses
    ax1.semilogy(z, dust_mass * 1e10, 'b-', linewidth=2.5, label='Dust mass', 
                marker='o', markersize=4, markevery=max(1, len(z)//50))
    ax1.semilogy(z, gas_mass * 1e10, 'r-', linewidth=2.5, label='Gas mass (high-res)', 
                marker='s', markersize=4, markevery=max(1, len(z)//50))
    
    ax1.set_ylabel(r'Mass [M$_\odot$]', fontsize=13)
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    #ax1.set_title('Dust and Gas Evolution - Gadget-4 Zoom Simulation', fontsize=14, fontweight='bold')
    
    # Bottom panel: D/G ratio
    nonzero = dg_ratio > 0
    if np.any(nonzero):
        ax2.semilogy(z[nonzero], dg_ratio[nonzero], 'g-', linewidth=2.5, 
                    marker='o', markersize=4, markevery=max(1, np.sum(nonzero)//50))
        ax2.fill_between(z[nonzero], dg_ratio[nonzero], alpha=0.3, color='green')
    
    # Reference lines
    ax2.axhline(1e-7, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, 
               label=r'D/G = $10^{-7}$ (dwarf-ish)')
    ax2.axhline(1e-6, color='gray', linestyle=':', alpha=0.6, linewidth=1.5,
               label=r'D/G = $10^{-6}$ (MW-ish)')
    
    # Mark peak D/G if any
    if np.any(nonzero):
        peak_idx = np.argmax(dg_ratio)
        if dg_ratio[peak_idx] > 0:
            ax2.plot(z[peak_idx], dg_ratio[peak_idx], 'r*', markersize=15, 
                    label=f'Peak: D/G={dg_ratio[peak_idx]:.2e} at z={z[peak_idx]:.2f}')
    
    ax2.set_xlabel('Redshift', fontsize=13)
    ax2.set_ylabel('Dust-to-Gas Ratio', fontsize=13)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-9, 1e-5)
    
    # Set x-axis limits: z=20 to z=0
    ax2.set_xlim(15, -0.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_file}")
    
    return fig, (ax1, ax2)


def print_statistics(data):
    """Print summary statistics of dust evolution."""
    print("\n" + "="*70)
    print("DUST EVOLUTION STATISTICS")
    print("="*70)
    
    z = data['redshift']
    dust_mass = data['dust_mass']
    gas_mass = data['gas_mass_highres']
    dg_ratio = data['dust_to_gas']
    n_dust = data['n_dust']
    n_gas = data['n_gas_highres']
    
    print(f"\nSnapshot range:")
    print(f"  Redshift: z = {z.max():.2f} to {z.min():.3f}")
    print(f"  Number of snapshots: {len(z)}")
    
    print(f"\nGas properties:")
    print(f"  Initial gas mass (high-res): {gas_mass[0]*1e10:.2e} Msun ({n_gas[0]} particles)")
    print(f"  Final gas mass (high-res):   {gas_mass[-1]*1e10:.2e} Msun ({n_gas[-1]} particles)")
    print(f"  Peak gas mass: {gas_mass.max()*1e10:.2e} Msun")
    
    # Find peak dust mass
    if np.any(dust_mass > 0):
        peak_idx = np.argmax(dust_mass)
        print(f"\nPeak dust mass:")
        print(f"  Redshift: z = {z[peak_idx]:.3f}")
        print(f"  Mass: {dust_mass[peak_idx]*1e10:.2e} Msun")
        print(f"  Particles: {n_dust[peak_idx]}")
    
    # Find peak D/G ratio
    nonzero = dg_ratio > 0
    if np.any(nonzero):
        peak_dg_idx = np.argmax(dg_ratio[nonzero])
        z_nonzero = z[nonzero]
        dg_nonzero = dg_ratio[nonzero]
        print(f"\nPeak D/G ratio:")
        print(f"  Redshift: z = {z_nonzero[peak_dg_idx]:.3f}")
        print(f"  D/G = {dg_nonzero[peak_dg_idx]:.2e}")
        
        # Final state
        print(f"\nFinal state (z = {z[-1]:.3f}):")
        print(f"  Dust mass: {dust_mass[-1]*1e10:.2e} Msun")
        print(f"  Dust particles: {n_dust[-1]}")
        print(f"  D/G ratio: {dg_ratio[-1]:.2e}")
        
        # When did dust first appear?
        first_dust_idx = np.where(n_dust > 0)[0]
        if len(first_dust_idx) > 0:
            print(f"\nFirst dust created:")
            print(f"  Redshift: z = {z[first_dust_idx[0]]:.3f}")
            print(f"  Initial D/G: {dg_ratio[first_dust_idx[0]]:.2e}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot dust evolution from Gadget-4 snapshots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic high-res detection
  python plot_dust_from_snapshots.py /path/to/output/
  
  # Specify mass threshold for high-res gas (in 10^10 Msun)
  python plot_dust_from_snapshots.py /path/to/output/ --highres-threshold 1e-5
  
  # Custom output filename
  python plot_dust_from_snapshots.py /path/to/output/ --output my_plot.png
        """
    )
    parser.add_argument('snapdir', help='Directory containing snapshot files')
    parser.add_argument('--highres-threshold', type=float, default=None,
                       help='Mass threshold for high-res gas particles (in code units, 10^10 Msun)')
    parser.add_argument('--output', default='dust_evolution.png',
                       help='Output plot filename (default: dust_evolution.png)')
    parser.add_argument('--show-all-gas', action='store_true',
                       help='Plot total gas mass instead of high-res only')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not Path(args.snapdir).exists():
        print(f"ERROR: Directory {args.snapdir} does not exist")
        sys.exit(1)
    
    # Read all snapshots
    print(f"Reading snapshots from {args.snapdir}...")
    if args.highres_threshold is not None:
        print(f"Using mass threshold: {args.highres_threshold:.2e} (10^10 Msun)")
        data = read_all_snapshots(args.snapdir, 'threshold', args.highres_threshold)
    else:
        print("Using automatic high-res detection")
        data = read_all_snapshots(args.snapdir, 'auto', None)
    
    if data is None:
        sys.exit(1)
    
    # Use total gas if requested
    if args.show_all_gas:
        print("\nNote: Plotting TOTAL gas mass (not just high-res)")
        data['gas_mass_highres'] = data['gas_mass_total']
        data['n_gas_highres'] = data['n_gas_total']
    
    # Print statistics
    print_statistics(data)
    
    # Create plot
    print(f"Creating plot...")
    plot_dust_evolution(data, output_file=args.output)
    
    print(f"\n✓ Done! Check {args.output}")


if __name__ == '__main__':
    main()
