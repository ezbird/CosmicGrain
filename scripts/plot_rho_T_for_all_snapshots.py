#!/usr/bin/env python3
"""
Phase diagram highlighting particles at the HSML floor.
Now processes entire output directories and shows star counts.

OPTIMIZED VERSION:
- Auto-skips backup snapshots (bak-*)
- Simpler CLI: just pass directory path
- Much faster: caches high-res region, optional filtering

Usage:
    # Process entire output directory (default behavior)
    python plot_rho_T_for_all_snapshots.py ./output_zoom_halo3698_100Mpc_feedback
    
    # Single snapshot
    python plot_rho_T_for_all_snapshots.py --snapshot output/snapdir_099/
    
    # Skip high-res filtering (much faster)
    python plot_rho_T_for_all_snapshots.py ./output_zoom/ --no-filter
"""
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob

# Physical constants
BOLTZMANN = 1.38064852e-16  # erg/K
PROTONMASS = 1.6726219e-24  # g
GAMMA = 5.0/3.0

# ✅ Global cache for high-res region (compute once, use for all snapshots)
_HIGHRES_REGION_CACHE = None

def find_snapshot_files(snapshot_input):
    """Find all snapshot files."""
    if os.path.isdir(snapshot_input):
        pattern = os.path.join(snapshot_input, "*.hdf5")
        files = glob.glob(pattern)
        if not files:
            pattern = os.path.join(snapshot_input, "snapshot_*.*.hdf5")
            files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in directory: {snapshot_input}")
    elif os.path.isfile(snapshot_input):
        return [snapshot_input]
    else:
        pattern = f"{snapshot_input}.*.hdf5"
        files = glob.glob(pattern)
        if not files:
            single_file = f"{snapshot_input}.hdf5"
            if os.path.isfile(single_file):
                return [single_file]
            raise FileNotFoundError(f"No snapshot files found for pattern: {snapshot_input}")
    
    files = sorted(files)
    print(f"Found {len(files)} snapshot file(s)")
    return files

def find_all_snapdirs(output_dir):
    """Find all snapdir_XXX directories in output folder, excluding backups."""
    snapdirs = glob.glob(os.path.join(output_dir, "snapdir_*"))
    
    # ✅ Filter out backup directories (bak-*)
    snapdirs = [s for s in snapdirs if not os.path.basename(s).startswith("bak-")]
    
    # Also check for snapshot_XXX.hdf5 files in root
    snapshot_files = glob.glob(os.path.join(output_dir, "snapshot_*.hdf5"))
    
    # ✅ Filter out backup files (bak-*)
    snapshot_files = [s for s in snapshot_files if not os.path.basename(s).startswith("bak-")]
    
    if not snapdirs and not snapshot_files:
        raise FileNotFoundError(f"No snapdir_* directories or snapshot_*.hdf5 files found in {output_dir}")
    
    # Sort by snapshot number
    if snapdirs:
        snapdirs = sorted(snapdirs, key=lambda x: int(x.split('_')[-1]))
        print(f"Found {len(snapdirs)} snapshot directories (excluding backups)")
        return snapdirs
    else:
        print(f"Found {len(snapshot_files)} snapshot files (excluding backups)")
        return snapshot_files

def count_stars(snapshot_files):
    """Count total number of star particles."""
    if isinstance(snapshot_files, str):
        snapshot_files = find_snapshot_files(snapshot_files)
    
    total_stars = 0
    total_stellar_mass = 0.0
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            # Get units
            if 'Parameters' in f:
                UnitMass_in_g = f['Parameters'].attrs.get('UnitMass_in_g', 1.989e43)
            else:
                UnitMass_in_g = 1.989e43
            
            SOLAR_MASS = 1.989e33
            
            if 'PartType4' in f:
                n_stars = len(f['PartType4/Masses'])
                masses = np.array(f['PartType4/Masses'])
                
                # Convert to solar masses using actual units
                stellar_mass_msun = (masses * UnitMass_in_g / SOLAR_MASS).sum()
                
                total_stars += n_stars
                total_stellar_mass += stellar_mass_msun
    
    return total_stars, total_stellar_mass

def identify_highres_region_cached(snapshot_files):
    """
    ✅ OPTIMIZED: Identify high-res region ONCE and cache it.
    Only reads from FIRST snapshot file to define region.
    """
    global _HIGHRES_REGION_CACHE
    
    if _HIGHRES_REGION_CACHE is not None:
        return _HIGHRES_REGION_CACHE
    
    print("Identifying high-resolution region (one-time calculation)...")
    
    # Only read from first file
    first_file = snapshot_files[0] if isinstance(snapshot_files, list) else find_snapshot_files(snapshot_files)[0]
    
    with h5py.File(first_file, 'r') as f:
        if 'PartType1' not in f or 'Coordinates' not in f['PartType1']:
            print("  No high-res DM found, skipping region filtering")
            _HIGHRES_REGION_CACHE = None
            return None
        
        # Read high-res DM positions (only from first file - good enough!)
        highres_dm_pos = np.array(f['PartType1/Coordinates'])
        
        # Define high-res region as bounding box
        hr_min = highres_dm_pos.min(axis=0)
        hr_max = highres_dm_pos.max(axis=0)
        hr_center = (hr_min + hr_max) / 2
        hr_extent = hr_max - hr_min
        
        print(f"  High-res DM particles: {len(highres_dm_pos):,}")
        print(f"  High-res region center: [{hr_center[0]:.2f}, {hr_center[1]:.2f}, {hr_center[2]:.2f}]")
        print(f"  High-res region extent: [{hr_extent[0]:.2f}, {hr_extent[1]:.2f}, {hr_extent[2]:.2f}]")
        
        _HIGHRES_REGION_CACHE = {
            'min': hr_min,
            'max': hr_max,
            'center': hr_center,
            'extent': hr_extent
        }
    
    return _HIGHRES_REGION_CACHE

def read_gas_properties_zoom(snapshot_files, hsml_floor_tolerance=1.01, 
                              filter_highres=True, contamination_check=False):
    """
    ✅ OPTIMIZED: Read gas properties for zoom simulations.
    - Uses cached high-res region (computed once)
    - Disabled contamination check by default (slow and rarely needed)
    """
    if isinstance(snapshot_files, str):
        snapshot_files = find_snapshot_files(snapshot_files)
    
    # Read header and parameters
    with h5py.File(snapshot_files[0], 'r') as f:
        header = f['Header'].attrs
        time = header['Time']
        redshift = header['Redshift']
        box_size = header['BoxSize']
        
        if 'Parameters' in f:
            parameters = f['Parameters'].attrs
            hubble = parameters['HubbleParam']
            UnitLength_in_cm = parameters.get('UnitLength_in_cm', 3.085678e21)
            UnitMass_in_g = parameters.get('UnitMass_in_g', 1.989e43)
            UnitVelocity_in_cm_per_s = parameters.get('UnitVelocity_in_cm_per_s', 1e5)
        else:
            hubble = header.get('HubbleParam', 1.0)
            UnitLength_in_cm = 3.085678e21
            UnitMass_in_g = 1.989e43
            UnitVelocity_in_cm_per_s = 1e5
    
    print(f"\nSnapshot: z={redshift:.2f}, a={time:.6f}")
    
    # ✅ Get cached high-res region (or compute once)
    highres_region = None
    if filter_highres:
        highres_region = identify_highres_region_cached(snapshot_files)
    
    # Read gas particles
    density_list = []
    masses_list = []
    internal_energy_list = []
    ne_list = []
    hsml_list = []
    positions_list = []
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            if 'PartType0' not in f:
                continue
            
            gas = f['PartType0']
            density_list.append(np.array(gas['Density']))
            masses_list.append(np.array(gas['Masses']))
            internal_energy_list.append(np.array(gas['InternalEnergy']))
            positions_list.append(np.array(gas['Coordinates']))
            
            if 'SmoothingLength' in gas:
                hsml_list.append(np.array(gas['SmoothingLength']))
            else:
                hsml_list.append(np.zeros(len(masses_list[-1])))
            
            if 'ElectronAbundance' in gas:
                ne_list.append(np.array(gas['ElectronAbundance']))
            else:
                ne_list.append(np.ones(len(masses_list[-1])))
    
    if not density_list:
        raise ValueError("No gas particles found!")
    
    density = np.concatenate(density_list)
    masses = np.concatenate(masses_list)
    internal_energy = np.concatenate(internal_energy_list)
    ne = np.concatenate(ne_list)
    hsml = np.concatenate(hsml_list)
    positions = np.concatenate(positions_list)
    
    print(f"Total gas particles: {len(density):,}")
    
    # ✅ Apply high-res filter if requested (using cached region)
    if filter_highres and highres_region is not None:
        hr_min = highres_region['min']
        hr_max = highres_region['max']
        
        # Expand region slightly (10% buffer)
        buffer = 0.1
        extent = hr_max - hr_min
        hr_min_buffered = hr_min - buffer * extent
        hr_max_buffered = hr_max + buffer * extent
        
        # Filter gas particles in high-res region
        in_highres = ((positions[:, 0] >= hr_min_buffered[0]) & (positions[:, 0] <= hr_max_buffered[0]) &
                      (positions[:, 1] >= hr_min_buffered[1]) & (positions[:, 1] <= hr_max_buffered[1]) &
                      (positions[:, 2] >= hr_min_buffered[2]) & (positions[:, 2] <= hr_max_buffered[2]))
        
        print(f"  Filtering to high-res region: {in_highres.sum():,} / {len(density):,} ({100*in_highres.sum()/len(density):.1f}%)")
        
        density = density[in_highres]
        masses = masses[in_highres]
        internal_energy = internal_energy[in_highres]
        ne = ne[in_highres]
        hsml = hsml[in_highres]
        positions = positions[in_highres]
    
    # Convert to physical units
    density_cgs = density * UnitMass_in_g / (UnitLength_in_cm**3) / (time**3)
    u_cgs = internal_energy * UnitVelocity_in_cm_per_s**2
    
    # Temperature calculation
    mu = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * ne)
    temperature = (GAMMA - 1.0) * u_cgs * mu * PROTONMASS / BOLTZMANN
    
    # Mass in solar masses
    SOLAR_MASS = 1.989e33
    mass_msun = masses * UnitMass_in_g / SOLAR_MASS
    
    # HSML in Mpc/h comoving
    hsml_mpc_h = hsml
    
    # Count stars
    n_stars, stellar_mass = count_stars(snapshot_files)
    
    return {
        'density': density_cgs,
        'temperature': temperature,
        'mass': mass_msun,
        'hsml': hsml_mpc_h,
        'time': time,
        'redshift': redshift,
        'n_stars': n_stars,
        'stellar_mass': stellar_mass
    }

def create_phase_diagram_with_hsml(data, output_file, hsml_floor, hsml_tolerance=1.01,
                                   density_range=(1e-30, 1e-20),
                                   temp_range=(1e2, 1e8),
                                   nbins=200):
    """Create phase diagram with HSML floor particles highlighted."""
    
    density = data['density']
    temperature = data['temperature']
    mass = data['mass']
    hsml = data['hsml']
    time = data['time']
    redshift = data['redshift']
    n_stars = data['n_stars']
    stellar_mass = data['stellar_mass']
    
    # Identify particles at HSML floor
    at_floor = hsml <= (hsml_floor * hsml_tolerance)
    
    print(f"  Particles at floor: {at_floor.sum():,} / {len(hsml):,} ({100*at_floor.sum()/len(hsml):.2f}%)")
    
    if at_floor.sum() > 0:
        print(f"  Floor particles - T range: {temperature[at_floor].min():.2e} - {temperature[at_floor].max():.2e} K")
        print(f"  Floor particles - ρ range: {density[at_floor].min():.2e} - {density[at_floor].max():.2e} g/cm^3")
    
    # Create bins
    density_bins = np.logspace(np.log10(density_range[0]), np.log10(density_range[1]), nbins)
    temp_bins = np.logspace(np.log10(temp_range[0]), np.log10(temp_range[1]), nbins)
    
    # All particles histogram (mass-weighted)
    H_all, xedges, yedges = np.histogram2d(
        density, temperature,
        bins=[density_bins, temp_bins],
        weights=mass
    )
    H_all = H_all.T
    H_all = np.ma.masked_where(H_all <= 0, H_all)
    
    # HSML floor particles histogram (mass-weighted)
    if at_floor.sum() > 0:
        H_floor, _, _ = np.histogram2d(
            density[at_floor], temperature[at_floor],
            bins=[density_bins, temp_bins],
            weights=mass[at_floor]
        )
        H_floor = H_floor.T
        H_floor = np.ma.masked_where(H_floor <= 0, H_floor)
    else:
        H_floor = np.zeros_like(H_all)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all particles
    im = ax.pcolormesh(xedges, yedges, H_all,
                       cmap='viridis',
                       norm=LogNorm(vmin=H_all[H_all>0].min(), vmax=H_all.max()),
                       shading='auto', alpha=0.8)
    
    cbar = plt.colorbar(im, ax=ax, label='Total Gas Mass [M$_\\odot$]')
    
    # Overlay HSML floor particles as red contours
    if at_floor.sum() > 0 and np.any(H_floor > 0):
        # Create contour levels
        levels = np.logspace(np.log10(H_floor[H_floor>0].min()), 
                           np.log10(H_floor.max()), 5)
        
        # Plot contours
        X, Y = np.meshgrid(xedges, yedges)
        CS = ax.contour(X[:-1, :-1], Y[:-1, :-1], H_floor,
                       levels=levels, colors='red', linewidths=2, alpha=0.8)
        
        # Add label for HSML floor particles
        ax.plot([], [], 'r-', linewidth=2, label=f'Particles at HSML floor (n={at_floor.sum():,})')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mass Density [g cm$^{-3}$]', fontsize=14)
    ax.set_ylabel('Temperature [K]', fontsize=14)
    ax.set_title(f'Gas Phase Diagram, z = {redshift:.2f} (a = {time:.4f})', fontsize=16)
    
    # Reference lines
    ax.axvline(1e-24, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
              label='SF density threshold')
    ax.axhline(1e4, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='~10$^4$ K')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add info text with star count
    info_text = f'HSML floor: {hsml_floor:.4f} Mpc/h\n'
    info_text += f'At floor: {at_floor.sum():,} ({100*at_floor.sum()/len(hsml):.1f}%)\n'
    info_text += f'\n★ Stars: {n_stars:,}\n'
    info_text += f'M★: {stellar_mass:.2e} M☉'
    
    ax.text(0.97, 0.03, info_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10,
            family='monospace')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved phase diagram to {output_file}")
    plt.close()

def process_single_snapshot(snapshot_path, output_file, hsml_floor, hsml_tolerance,
                            density_range, temp_range, nbins, filter_highres):
    """Process a single snapshot."""
    print(f"\n{'='*60}")
    print(f"Processing: {snapshot_path}")
    print(f"{'='*60}")
    
    data = read_gas_properties_zoom(snapshot_path, hsml_tolerance, 
                                    filter_highres=filter_highres)

    create_phase_diagram_with_hsml(
        data, output_file, hsml_floor, hsml_tolerance,
        density_range=density_range,
        temp_range=temp_range,
        nbins=nbins
    )

def process_all_snapshots(output_dir, hsml_floor, hsml_tolerance,
                          density_range, temp_range, nbins, output_subdir, filter_highres):
    """Process all snapshots in output directory."""
    
    # Find all snapshot directories or files
    snapshots = find_all_snapdirs(output_dir)
    
    # Create output directory for plots
    plot_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nSaving plots to: {plot_dir}")
    
    print(f"\nProcessing {len(snapshots)} snapshots...")
    
    for snapshot_path in snapshots:
        # Extract snapshot number from path
        if 'snapdir_' in snapshot_path:
            snap_num = snapshot_path.split('snapdir_')[-1]
            output_file = os.path.join(plot_dir, f'phase_diagram_{snap_num}.png')
        else:
            # For snapshot_XXX.hdf5 files
            basename = os.path.basename(snapshot_path)
            snap_num = basename.replace('snapshot_', '').replace('.hdf5', '')
            output_file = os.path.join(plot_dir, f'phase_diagram_{snap_num}.png')
        
        try:
            process_single_snapshot(snapshot_path, output_file, hsml_floor, hsml_tolerance,
                                   density_range, temp_range, nbins, filter_highres)
        except Exception as e:
            print(f"ERROR processing {snapshot_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed! All plots saved to: {plot_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Phase diagram with HSML floor particles highlighted',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire output directory (simplest!)
  python plot_rho_T_for_all_snapshots.py ./output_zoom/
  
  # Process single snapshot
  python plot_rho_T_for_all_snapshots.py --snapshot snapdir_099/
  
  # Skip high-res filtering for speed
  python plot_rho_T_for_all_snapshots.py ./output_zoom/ --no-filter
        """
    )
    
    # ✅ Make output directory a positional argument (simpler!)
    parser.add_argument('output_dir', nargs='?',
                       help='Path to output directory containing all snapshots')
    
    # Input options
    parser.add_argument('--snapshot',
                       help='Path to single snapshot directory or file')
    
    # Output options
    parser.add_argument('--output', default='phase_diagram_hsml_floor.png',
                       help='Output image file (for single snapshot mode)')
    parser.add_argument('--output-subdir', default='phase_diagrams',
                       help='Subdirectory name for plots (for output-dir mode)')
    
    # Analysis parameters
    parser.add_argument('--hsml-floor', type=float, default=0.00025,
                       help='HSML floor value from MinGasHsml parameter (default: 0.00025)')
    parser.add_argument('--hsml-tolerance', type=float, default=1.01,
                       help='Tolerance factor for identifying floor particles (default: 1.01)')
    parser.add_argument('--nbins', type=int, default=200,
                       help='Number of bins (default: 200)')
    parser.add_argument('--density-range', nargs=2, type=float,
                       default=[1e-30, 1e-20],
                       help='Density range in g/cm^3')
    parser.add_argument('--temp-range', nargs=2, type=float,
                       default=[1e2, 1e8],
                       help='Temperature range in K')
    
    # ✅ Performance option
    parser.add_argument('--no-filter', action='store_true',
                       help='Skip high-res region filtering (much faster)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.snapshot and not args.output_dir:
        parser.error("Either output_dir or --snapshot must be provided")
    
    filter_highres = not args.no_filter
    
    if args.snapshot:
        # Single snapshot mode
        print(f"Reading snapshot: {args.snapshot}")
        process_single_snapshot(args.snapshot, args.output, args.hsml_floor, args.hsml_tolerance,
                               args.density_range, args.temp_range, args.nbins, filter_highres)
    else:
        # Multi-snapshot mode
        print(f"Processing all snapshots in: {args.output_dir}")
        process_all_snapshots(args.output_dir, args.hsml_floor, args.hsml_tolerance,
                             args.density_range, args.temp_range, args.nbins, 
                             args.output_subdir, filter_highres)
    
    print("\nDone!")

if __name__ == '__main__':
    main()