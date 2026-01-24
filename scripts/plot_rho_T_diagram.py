#!/usr/bin/env python3
"""
Create a density-temperature phase diagram from Gadget-4 zoom simulation,
showing only high-resolution gas particles with color representing total gas mass.

Handles both single-file and multi-file snapshots.

Usage:
    # Single file
    python plot_phase_diagram.py \
        --snapshot output/snapshot_099.hdf5 \
        --output phase_diagram_z0.png
    
    # Multi-file (directory or base pattern)
    python plot_phase_diagram.py \
        --snapshot output/snapdir_099/ \
        --output phase_diagram_z99.png
    
    # Or specify base name
    python plot_phase_diagram.py \
        --snapshot output/snapdir_099/snapshot_099 \
        --output phase_diagram_z99.png
"""
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob

plt.style.use('sleek.mplstyle')

# Physical constants
BOLTZMANN = 1.38064852e-16  # erg/K
PROTONMASS = 1.6726219e-24  # g
GAMMA = 5.0/3.0  # Adiabatic index for monoatomic gas

def find_snapshot_files(snapshot_input):
    """
    Find all snapshot files from various input formats:
    - Single file: snapshot_099.hdf5
    - Directory: snapdir_099/
    - Base pattern: snapdir_099/snapshot_099
    
    Returns sorted list of snapshot files.
    """
    # Check if it's a directory
    if os.path.isdir(snapshot_input):
        # Find all .hdf5 files in directory
        pattern = os.path.join(snapshot_input, "*.hdf5")
        files = glob.glob(pattern)
        if not files:
            # Try with numbered files
            pattern = os.path.join(snapshot_input, "snapshot_*.*.hdf5")
            files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in directory: {snapshot_input}")
    
    # Check if it's a single file
    elif os.path.isfile(snapshot_input):
        return [snapshot_input]
    
    # Otherwise, assume it's a base pattern (e.g., snapdir_099/snapshot_099)
    else:
        # Try numbered files: snapshot_099.0.hdf5, snapshot_099.1.hdf5, etc.
        pattern = f"{snapshot_input}.*.hdf5"
        files = glob.glob(pattern)
        
        if not files:
            # Try without extension: snapshot_099.0, snapshot_099.1, etc.
            pattern = f"{snapshot_input}.*"
            files = glob.glob(pattern)
            files = [f for f in files if not f.endswith('.hdf5')]
        
        if not files:
            # Try single file with .hdf5 extension
            single_file = f"{snapshot_input}.hdf5"
            if os.path.isfile(single_file):
                return [single_file]
            raise FileNotFoundError(f"No snapshot files found for pattern: {snapshot_input}")
    
    # Sort files to ensure correct order
    files = sorted(files)
    
    print(f"Found {len(files)} snapshot file(s):")
    for f in files[:5]:  # Show first 5
        print(f"  {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... and {len(files)-5} more")
    
    return files

def read_gas_properties(snapshot_files):
    """
    Read gas particle properties from one or more snapshot files.
    Returns dict with density, temperature, mass, and coordinates.
    """
    # If input is string, convert to list
    if isinstance(snapshot_files, str):
        snapshot_files = find_snapshot_files(snapshot_files)
    
    # Read header from first file
    with h5py.File(snapshot_files[0], 'r') as f:
        header = f['Header'].attrs
        parameters = f['Parameters'].attrs 
        time = header['Time']
        redshift = header['Redshift']
        hubble = parameters['HubbleParam']
        box_size = header['BoxSize']
        num_files = header.get('NumFilesPerSnapshot', 1)
    
    # Unit conversions (Gadget-4 uses Mpc, 10^10 Msun, km/s)
    UnitLength_in_cm = 3.085678e24  # Mpc
    UnitMass_in_g = 1.989e43        # 10^10 Msun
    UnitVelocity_in_cm_per_s = 1e5  # km/s
    
    print(f"\nSnapshot info:")
    print(f"  Time = {time:.6f} (a = {time:.6f})")
    print(f"  Redshift = {redshift:.4f}")
    print(f"  Box size = {box_size:.2f} Mpc/h")
    print(f"  Number of files: {num_files}")
    
    # Read from all files
    density_list = []
    masses_list = []
    internal_energy_list = []
    ne_list = []
    
    # Particle counts
    n_stars = 0
    n_dm = 0
    dm_masses_list = []
    
    print(f"\nReading gas particles from {len(snapshot_files)} file(s)...")
    
    for i, snap_file in enumerate(snapshot_files):
        with h5py.File(snap_file, 'r') as f:
            # Check if gas particles exist in this file
            if 'PartType0' in f:
                gas = f['PartType0']
                
                # Read particle data
                density_list.append(np.array(gas['Density']))
                masses_list.append(np.array(gas['Masses']))
                internal_energy_list.append(np.array(gas['InternalEnergy']))
                
                # Electron abundance (if available)
                if 'ElectronAbundance' in gas:
                    ne_list.append(np.array(gas['ElectronAbundance']))
                else:
                    ne_list.append(np.ones(len(masses_list[-1])))
            
            # Count star particles (PartType4)
            if 'PartType4' in f:
                n_stars += len(f['PartType4']['Masses'])
            
            # Count DM particles and get masses (PartType1)
            if 'PartType1' in f:
                dm = f['PartType1']
                n_dm += len(dm['Masses'])
                dm_masses_list.append(np.array(dm['Masses']))
        
        if (i + 1) % 5 == 0 or (i + 1) == len(snapshot_files):
            print(f"  Processed {i+1}/{len(snapshot_files)} files...")
    
    # Concatenate all arrays
    if not density_list:
        raise ValueError("No gas particles (PartType0) found in any snapshot files!")
    
    density = np.concatenate(density_list)
    masses = np.concatenate(masses_list)
    internal_energy = np.concatenate(internal_energy_list)
    ne = np.concatenate(ne_list)

    # Process DM masses if available
    if dm_masses_list:
        dm_masses = np.concatenate(dm_masses_list)
        dm_mass_msun = dm_masses * 1e10 / hubble  # Convert to Msun
    else:
        dm_mass_msun = np.array([])

    # Mean molecular weight
    # For neutral gas: mu ≈ 0.6 (mostly H)
    # For fully ionized: mu ≈ 0.59
    # Approximate: mu = 4/(1 + 3*X + 4*X*ne) where X = 0.76
    X_H = 0.76  # Hydrogen mass fraction
    mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*ne)
    
    # Convert internal energy to temperature
    # u = (k*T) / ((gamma-1) * mu * m_p)
    # T = u * (gamma-1) * mu * m_p / k
    u_cgs = internal_energy * UnitVelocity_in_cm_per_s**2  # erg/g
    temperature = u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN
    
    # Convert density to physical units (g/cm^3)
    # Gadget density is in comoving code units
    a = time  # scale factor
    rho_code = density  # 10^10 Msun/h / (Mpc/h)^3
    rho_cgs = rho_code * (UnitMass_in_g / UnitLength_in_cm**3)  # g/cm^3 comoving
    rho_physical = rho_cgs / a**3  # Convert to physical density
    
    # Convert to hydrogen number density (cm^-3)
    n_H = rho_physical * X_H / PROTONMASS
    
    # Convert masses to physical units (Msun)
    mass_msun = masses * 1e10 / hubble  # Msun
    
    print(f"\nTotal gas particle statistics:")
    print(f"  Total particles: {len(masses):,}")
    print(f"  Total gas mass: {mass_msun.sum():.2e} Msun")
    print(f"  Mass range: {mass_msun.min():.2e} - {mass_msun.max():.2e} Msun")
    print(f"  Temperature range: {temperature.min():.2e} - {temperature.max():.2e} K")
    print(f"  Mass density range: {rho_physical.min():.2e} - {rho_physical.max():.2e} g/cm^3")
    print(f"  Number density range: {n_H.min():.2e} - {n_H.max():.2e} cm^-3")
    print(f"\nOther particle types:")
    print(f"  Star particles (Type 4): {n_stars:,}")
    print(f"  DM particles (Type 1): {n_dm:,}")
    
    return {
        'density': rho_physical,  # g/cm^3 (MASS density, not number density)
        'temperature': temperature,  # K
        'mass': mass_msun,        # Msun
        'redshift': redshift,
        'time': time,
        'n_stars': n_stars,
        'n_dm': n_dm,
        'dm_masses': dm_mass_msun  # For filtering high-res DM
    }

def filter_highres_particles(data, mass_threshold_factor=10):
    """
    Filter for high-resolution particles based on mass.
    High-res particles have uniform low mass, low-res have higher mass.
    
    Returns mask for high-res gas particles and count of high-res DM particles.
    """
    masses = data['mass']
    
    # Find the mode (most common) mass - this should be high-res
    hist, edges = np.histogram(np.log10(masses), bins=100)
    mode_idx = np.argmax(hist)
    mode_mass = 10**((edges[mode_idx] + edges[mode_idx+1])/2)
    
    # High-res particles are those within factor of mass_threshold_factor of mode mass
    mask = masses < (mode_mass * mass_threshold_factor)
    
    # Filter DM particles similarly if available
    n_dm_highres = 0
    if len(data['dm_masses']) > 0:
        dm_masses = data['dm_masses']
        hist_dm, edges_dm = np.histogram(np.log10(dm_masses), bins=100)
        mode_idx_dm = np.argmax(hist_dm)
        mode_mass_dm = 10**((edges_dm[mode_idx_dm] + edges_dm[mode_idx_dm+1])/2)
        mask_dm = dm_masses < (mode_mass_dm * mass_threshold_factor)
        n_dm_highres = mask_dm.sum()
    
    print(f"\nHigh-res particle filtering:")
    print(f"  Gas mode mass: {mode_mass:.2e} Msun")
    print(f"  Threshold: {mode_mass * mass_threshold_factor:.2e} Msun")
    print(f"  High-res gas particles: {mask.sum():,} / {len(mask):,} ({100*mask.sum()/len(mask):.1f}%)")
    if n_dm_highres > 0:
        print(f"  High-res DM particles: {n_dm_highres:,} / {len(data['dm_masses']):,} ({100*n_dm_highres/len(data['dm_masses']):.1f}%)")
    
    return mask, n_dm_highres

def create_phase_diagram(data, highres_mask, n_dm_highres, output_file, 
                        density_range=(1e-30, 1e-20), temp_range=(1e2, 1e8),
                        nbins=200):
    """
    Create density-temperature phase diagram with mass-weighted bins.
    Includes inset showing particle counts.
    """
    # Filter for high-res particles
    density = data['density'][highres_mask]
    temperature = data['temperature'][highres_mask]
    mass = data['mass'][highres_mask]
    redshift = data['redshift']
    n_stars = data['n_stars']
    n_gas_highres = highres_mask.sum()
    
    # Create log-spaced bins
    density_bins = np.logspace(np.log10(density_range[0]), 
                               np.log10(density_range[1]), nbins)
    temp_bins = np.logspace(np.log10(temp_range[0]), 
                           np.log10(temp_range[1]), nbins)
    
    # Create 2D histogram weighted by mass
    H, xedges, yedges = np.histogram2d(
        density, temperature, 
        bins=[density_bins, temp_bins],
        weights=mass
    )
    
    # Transpose for correct orientation (density on x, temp on y)
    H = H.T
    
    # Mask zero bins
    H = np.ma.masked_where(H <= 0, H)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot 2D histogram
    im = ax.pcolormesh(xedges, yedges, H, 
                       cmap='viridis', 
                       norm=LogNorm(vmin=H[H>0].min(), vmax=H.max()),
                       shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Total Gas Mass (M$_\\odot$)')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mass Density (g cm$^{-3}$)', fontsize=14)
    ax.set_ylabel('Temperature (K)', fontsize=14)
    ax.set_title(f'z = {redshift:.2f}', fontsize=16)
    
    # Add reference lines
    # Star formation threshold (typical: rho ~ 10^-24 g/cm^3, T ~ 10^4 K)
    ax.axvline(1e-24, color='k', linestyle='--', alpha=0.5, linewidth=1.5, 
              label='SF density threshold')
    ax.axhline(1e4, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='~10$^4$ K')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add inset with particle counts
    # Add particle counts as text in top right
    # Format counts
    if n_gas_highres >= 1e6:
        gas_str = f'{n_gas_highres:.2e}'
    else:
        gas_str = f'{n_gas_highres:,}'
    
    if n_dm_highres >= 1e6:
        dm_str = f'{n_dm_highres:.2e}'
    else:
        dm_str = f'{n_dm_highres:,}'
    
    if n_stars >= 1e6:
        stars_str = f'{n_stars:.2e}'
    else:
        stars_str = f'{n_stars:,}'
    
    # Add text box in upper right
    text_str = f'High-Res Region:\n'
    text_str += f'Gas: {gas_str}\n'
    text_str += f'DM: {dm_str}\n'
    text_str += f'Stars: {stars_str}'
    
    ax.text(0.97, 0.97, text_str,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=11,
            family='monospace')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved phase diagram to {output_file}")
    
    # Print statistics
    total_mass = mass.sum()
    print(f"\nPhase diagram statistics:")
    print(f"  Total high-res gas mass: {total_mass:.2e} Msun")
    print(f"  Mass in bins: {H.sum():.2e} Msun")
    print(f"  Number of non-empty bins: {np.sum(H > 0)}")

def main():
    parser = argparse.ArgumentParser(
        description='Create phase diagram from Gadget-4 zoom simulation'
    )
    parser.add_argument('--snapshot', required=True,
                       help='Path to snapshot file (HDF5)')
    parser.add_argument('--output', default='phase_diagram.png',
                       help='Output image file')
    parser.add_argument('--nbins', type=int, default=200,
                       help='Number of bins in each dimension (default: 200)')
    parser.add_argument('--density-range', nargs=2, type=float,
                       default=[1e-30, 1e-20],
                       help='Mass density range in g/cm^3 (default: 1e-30 1e-20)')
    parser.add_argument('--temp-range', nargs=2, type=float,
                       default=[1e2, 1e8],
                       help='Temperature range in K (default: 1e2 1e8)')
    parser.add_argument('--mass-threshold', type=float, default=10,
                       help='Factor above mode mass for high-res cutoff (default: 10)')
    
    args = parser.parse_args()
    
    # Read snapshot
    print(f"Reading snapshot: {args.snapshot}")
    data = read_gas_properties(args.snapshot)
    
    # Filter for high-res particles
    highres_mask, n_dm_highres = filter_highres_particles(data, args.mass_threshold)
    
    # Create phase diagram
    create_phase_diagram(
        data, highres_mask, n_dm_highres, args.output,
        density_range=args.density_range,
        temp_range=args.temp_range,
        nbins=args.nbins
    )
    
    print("\nDone!")

if __name__ == '__main__':
    main()