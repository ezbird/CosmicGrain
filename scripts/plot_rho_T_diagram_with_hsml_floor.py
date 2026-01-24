#!/usr/bin/env python3
"""
Phase diagram highlighting particles at the HSML floor.

Usage:
    python plot_phase_diagram_with_hsml_floor.py \
        --snapshot output/snapdir_099/ \
        --output phase_diagram_hsml_floor.png \
        --hsml-floor 0.0005  # Your MinGasHsml value
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

def read_gas_properties_with_hsml(snapshot_files, hsml_floor_tolerance=1.01):
    """Read gas properties including Hsml smoothing lengths."""
    if isinstance(snapshot_files, str):
        snapshot_files = find_snapshot_files(snapshot_files)
    
    # Read header
    with h5py.File(snapshot_files[0], 'r') as f:
        header = f['Header'].attrs
        parameters = f['Parameters'].attrs
        time = header['Time']
        redshift = header['Redshift']
        hubble = parameters['HubbleParam']
        box_size = header['BoxSize']
    
    UnitLength_in_cm = 3.085678e24
    UnitMass_in_g = 1.989e43
    UnitVelocity_in_cm_per_s = 1e5
    
    print(f"\nSnapshot: z={redshift:.2f}")
    
    # Read all gas particles
    density_list = []
    masses_list = []
    internal_energy_list = []
    ne_list = []
    hsml_list = []  # Smoothing lengths
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            if 'PartType0' not in f:
                continue
            
            gas = f['PartType0']
            density_list.append(np.array(gas['Density']))
            masses_list.append(np.array(gas['Masses']))
            internal_energy_list.append(np.array(gas['InternalEnergy']))
            
            # Read smoothing lengths
            if 'SmoothingLength' in gas:
                hsml_list.append(np.array(gas['SmoothingLength']))
            else:
                # Fallback if not available
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
    
    # Calculate temperature and density
    X_H = 0.76
    mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*ne)
    
    u_cgs = internal_energy * UnitVelocity_in_cm_per_s**2
    temperature = u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN
    
    a = time
    rho_code = density
    rho_cgs = rho_code * (UnitMass_in_g / UnitLength_in_cm**3)
    rho_physical = rho_cgs / a**3
    
    mass_msun = masses * 1e10 / hubble
    
    print(f"Total gas particles: {len(masses):,}")
    print(f"Temperature range: {temperature.min():.2e} - {temperature.max():.2e} K")
    print(f"Density range: {rho_physical.min():.2e} - {rho_physical.max():.2e} g/cm^3")
    
    return {
        'density': rho_physical,
        'temperature': temperature,
        'mass': mass_msun,
        'hsml': hsml,  # Smoothing lengths
        'redshift': redshift
    }

def create_phase_diagram_with_hsml(data, output_file, hsml_floor, hsml_tolerance=1.01,
                                    density_range=(1e-30, 1e-20), temp_range=(1e2, 1e8),
                                    nbins=200):
    """Create phase diagram highlighting particles at HSML floor."""
    
    density = data['density']
    temperature = data['temperature']
    mass = data['mass']
    hsml = data['hsml']
    redshift = data['redshift']
    
    # Identify particles at or very close to HSML floor
    at_floor = hsml <= (hsml_floor * hsml_tolerance)
    
    print(f"\nHSML floor analysis:")
    print(f"  HSML floor: {hsml_floor:.4f} Mpc/h")
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
    ax.set_title(f'Gas Phase Diagram with HSML Floor Particles, z = {redshift:.2f}', fontsize=16)
    
    # Reference lines
    ax.axvline(1e-24, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
              label='SF density threshold')
    ax.axhline(1e4, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='~10$^4$ K')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add info text
    info_text = f'HSML floor: {hsml_floor:.4f} Mpc/h\n'
    info_text += f'At floor: {at_floor.sum():,} ({100*at_floor.sum()/len(hsml):.1f}%)'
    
    ax.text(0.97, 0.03, info_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10,
            family='monospace')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved phase diagram to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Phase diagram with HSML floor particles highlighted'
    )
    parser.add_argument('--snapshot', required=True,
                       help='Path to snapshot')
    parser.add_argument('--output', default='phase_diagram_hsml_floor.png',
                       help='Output image file')
    parser.add_argument('--hsml-floor', type=float, default=0.0005,
                       help='HSML floor value from MinGasHsml parameter (default: 0.0005)')
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
    
    args = parser.parse_args()
    
    print(f"Reading snapshot: {args.snapshot}")
    data = read_gas_properties_with_hsml(args.snapshot)
    
    create_phase_diagram_with_hsml(
        data, args.output, args.hsml_floor, args.hsml_tolerance,
        density_range=args.density_range,
        temp_range=args.temp_range,
        nbins=args.nbins
    )
    
    print("\nDone!")

if __name__ == '__main__':
    main()