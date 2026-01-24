#!/usr/bin/env python3
"""
Analyze dust-to-metal ratio evolution in Gadget-4 simulations
Compares results to MW, LMC, and high-z observations
"""

import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic_2d

# Observational benchmarks
OBS_DATA = {
    'MW': {'D/M': 0.45, 'D/M_err': 0.05, 'Z_gas': 8.69, 'label': 'Milky Way'},
    'LMC': {'D/M': 0.25, 'D/M_err': 0.05, 'Z_gas': 8.35, 'label': 'LMC'},
    'SMC': {'D/M': 0.15, 'D/M_err': 0.05, 'Z_gas': 8.0, 'label': 'SMC'},
    'high_z_low': {'D/M': 0.1, 'D/M_err': 0.05, 'Z_gas': 8.3, 'label': 'High-z (lower limit)'},
    'high_z_high': {'D/M': 0.3, 'D/M_err': 0.05, 'Z_gas': 8.5, 'label': 'High-z (upper limit)'},
}


def read_snapshot(snapshot_path, verbose=False):
    """
    Read a Gadget-4 snapshot and extract dust and metal information
    
    Parameters:
    -----------
    snapshot_path : str
        Path to snapshot file (can be multi-file, will handle .0, .1, etc)
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    dict with keys:
        - time : simulation time
        - redshift : redshift
        - dust_mass : array of dust particle masses
        - dust_pos : array of dust particle positions
        - gas_mass : array of gas particle masses
        - gas_metal : array of gas particle metal masses
        - gas_pos : array of gas particle positions
        - gas_metallicity : array of gas particle metallicities (mass fraction)
        - boxsize : simulation box size
    """
    
    # Handle multi-file snapshots
    base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
    snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
    if not snapshot_files:
        snapshot_files = [snapshot_path]
    
    if verbose:
        print(f"Reading {len(snapshot_files)} file(s): {base_path}")
    
    data = {
        'dust_mass': [],
        'dust_pos': [],
        'gas_mass': [],
        'gas_metal': [],
        'gas_pos': [],
        'gas_metallicity': [],
    }
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            # Read header (only once)
            if 'time' not in data:
                data['time'] = f['Header'].attrs['Time']
                data['redshift'] = f['Header'].attrs['Redshift']
                data['boxsize'] = f['Header'].attrs['BoxSize']
                data['hubble'] = f['Header'].attrs['HubbleParam']
            
            # Read dust particles (PartType6)
            if 'PartType6' in f:
                masses = f['PartType6/Masses'][:]
                coords = f['PartType6/Coordinates'][:]
                data['dust_mass'].append(masses)
                data['dust_pos'].append(coords)
            
            # Read gas particles (PartType0)
            if 'PartType0' in f:
                masses = f['PartType0/Masses'][:]
                coords = f['PartType0/Coordinates'][:]
                
                # Metallicity can be stored in different ways
                if 'GFM_Metallicity' in f['PartType0']:
                    metallicity = f['PartType0/GFM_Metallicity'][:]
                elif 'Metallicity' in f['PartType0']:
                    metallicity = f['PartType0/Metallicity'][:]
                else:
                    if verbose:
                        print("Warning: No metallicity field found, assuming Z=0")
                    metallicity = np.zeros_like(masses)
                
                metal_mass = masses * metallicity
                
                data['gas_mass'].append(masses)
                data['gas_metal'].append(metal_mass)
                data['gas_pos'].append(coords)
                data['gas_metallicity'].append(metallicity)
    
    # Concatenate arrays from all files
    for key in ['dust_mass', 'dust_pos', 'gas_mass', 'gas_metal', 'gas_pos', 'gas_metallicity']:
        if data[key]:
            data[key] = np.concatenate(data[key])
        else:
            data[key] = np.array([])
    
    if verbose:
        print(f"  Time = {data['time']:.3f}, Redshift = {data['redshift']:.3f}")
        print(f"  N_dust = {len(data['dust_mass'])}, N_gas = {len(data['gas_mass'])}")
        print(f"  Total dust mass = {np.sum(data['dust_mass']):.3e}")
        print(f"  Total metal mass = {np.sum(data['gas_metal']):.3e}")
    
    return data


def compute_global_DM(data):
    """
    Compute global dust-to-metal ratio
    
    Returns:
    --------
    D/M : float
        Global dust-to-metal ratio
    """
    total_dust = np.sum(data['dust_mass'])
    total_metals = np.sum(data['gas_metal'])
    
    if total_metals == 0:
        return 0.0
    
    return total_dust / total_metals


def compute_DM_vs_metallicity(data, Z_bins=np.logspace(-4, -1, 20)):
    """
    Compute D/M as a function of gas-phase metallicity
    
    Parameters:
    -----------
    data : dict
        Snapshot data
    Z_bins : array
        Metallicity bins (mass fraction)
        
    Returns:
    --------
    Z_centers : array
        Bin centers in metallicity
    DM_ratio : array
        D/M in each bin
    counts : array
        Number of particles per bin
    """
    
    if len(data['dust_mass']) == 0 or len(data['gas_metal']) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Create a spatial grid to match dust to gas
    # Use a fine grid - bins in 3D
    n_bins = 50
    
    # Get galaxy center (use gas center of mass)
    if len(data['gas_pos']) > 0:
        center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
    else:
        center = data['boxsize'] / 2
    
    # Define spatial extent (e.g., within 50 kpc of center)
    extent = 50.0  # kpc, adjust as needed
    
    # Bin dust mass in 3D grid
    dust_hist, edges = np.histogramdd(
        data['dust_pos'] - center,
        bins=n_bins,
        range=[(-extent, extent)] * 3,
        weights=data['dust_mass']
    )
    
    # Bin metal mass in same grid
    metal_hist, _ = np.histogramdd(
        data['gas_pos'] - center,
        bins=n_bins,
        range=[(-extent, extent)] * 3,
        weights=data['gas_metal']
    )
    
    # Bin gas mass to get average metallicity
    gas_hist, _ = np.histogramdd(
        data['gas_pos'] - center,
        bins=n_bins,
        range=[(-extent, extent)] * 3,
        weights=data['gas_mass']
    )
    
    # Compute metallicity in each cell
    Z_grid = np.zeros_like(metal_hist)
    mask = gas_hist > 0
    Z_grid[mask] = metal_hist[mask] / gas_hist[mask]
    
    # Flatten arrays
    Z_flat = Z_grid.flatten()
    dust_flat = dust_hist.flatten()
    metal_flat = metal_hist.flatten()
    
    # Only keep cells with both dust and metals
    valid = (metal_flat > 0) & (Z_flat > 0)
    Z_valid = Z_flat[valid]
    DM_valid = dust_flat[valid] / metal_flat[valid]
    
    # Bin by metallicity
    Z_centers = 0.5 * (Z_bins[1:] + Z_bins[:-1])
    DM_binned = []
    counts = []
    
    for i in range(len(Z_bins) - 1):
        in_bin = (Z_valid >= Z_bins[i]) & (Z_valid < Z_bins[i+1])
        if np.sum(in_bin) > 0:
            DM_binned.append(np.median(DM_valid[in_bin]))
            counts.append(np.sum(in_bin))
        else:
            DM_binned.append(np.nan)
            counts.append(0)
    
    return Z_centers, np.array(DM_binned), np.array(counts)


def compute_radial_profile(data, r_bins=np.linspace(0, 50, 25)):
    """
    Compute D/M as a function of radius from galaxy center
    
    Parameters:
    -----------
    data : dict
        Snapshot data
    r_bins : array
        Radial bins in kpc
        
    Returns:
    --------
    r_centers : array
        Bin centers in kpc
    DM_ratio : array
        D/M in each radial bin
    """
    
    if len(data['dust_mass']) == 0 or len(data['gas_metal']) == 0:
        return np.array([]), np.array([])
    
    # Get galaxy center
    center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
    
    # Compute radii
    dust_r = np.linalg.norm(data['dust_pos'] - center, axis=1)
    gas_r = np.linalg.norm(data['gas_pos'] - center, axis=1)
    
    # Bin by radius
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    DM_radial = []
    
    for i in range(len(r_bins) - 1):
        dust_in_bin = np.sum(data['dust_mass'][(dust_r >= r_bins[i]) & (dust_r < r_bins[i+1])])
        metal_in_bin = np.sum(data['gas_metal'][(gas_r >= r_bins[i]) & (gas_r < r_bins[i+1])])
        
        if metal_in_bin > 0:
            DM_radial.append(dust_in_bin / metal_in_bin)
        else:
            DM_radial.append(np.nan)
    
    return r_centers, np.array(DM_radial)


def analyze_snapshots(snapshot_dir, output_prefix='dust_analysis'):
    """
    Analyze all snapshots in a directory
    
    Parameters:
    -----------
    snapshot_dir : str
        Directory containing snapshot files
    output_prefix : str
        Prefix for output files
    """
    
    # Find all snapshots
    snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_*.0.hdf5')))
    if not snapshot_files:
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_*.hdf5')))
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return
    
    print(f"Found {len(snapshot_files)} snapshots")
    
    # Arrays to store evolution
    times = []
    redshifts = []
    DM_global = []
    total_dust = []
    total_metals = []
    
    # For the latest snapshot, do detailed analysis
    latest_data = None
    
    for i, snap_file in enumerate(snapshot_files):
        print(f"\nProcessing {os.path.basename(snap_file)} ({i+1}/{len(snapshot_files)})")
        data = read_snapshot(snap_file, verbose=True)
        
        times.append(data['time'])
        redshifts.append(data['redshift'])
        DM_global.append(compute_global_DM(data))
        total_dust.append(np.sum(data['dust_mass']))
        total_metals.append(np.sum(data['gas_metal']))
        
        # Save latest for detailed analysis
        if i == len(snapshot_files) - 1:
            latest_data = data
    
    times = np.array(times)
    redshifts = np.array(redshifts)
    DM_global = np.array(DM_global)
    total_dust = np.array(total_dust)
    total_metals = np.array(total_metals)
    
    # Create plots
    print("\nCreating plots...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: D/M evolution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, DM_global, 'o-', linewidth=2, markersize=6, color='darkblue')
    ax1.axhline(OBS_DATA['MW']['D/M'], color='green', linestyle='--', label='MW', linewidth=2)
    ax1.axhline(OBS_DATA['LMC']['D/M'], color='orange', linestyle='--', label='LMC', linewidth=2)
    ax1.set_xlabel('Time [code units]', fontsize=12)
    ax1.set_ylabel('Global D/M', fontsize=12)
    ax1.set_title('Dust-to-Metal Ratio Evolution', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, max(1.0, np.nanmax(DM_global) * 1.1))
    
    # Plot 2: Total masses
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogy(times, total_dust, 'o-', label='Dust', color='brown', markersize=4)
    ax2.semilogy(times, total_metals, 's-', label='Metals', color='gray', markersize=4)
    ax2.set_xlabel('Time [code units]', fontsize=12)
    ax2.set_ylabel('Total Mass [code units]', fontsize=12)
    ax2.set_title('Mass Evolution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: D/M vs metallicity (latest snapshot)
    if latest_data is not None:
        ax3 = fig.add_subplot(gs[1, :2])
        Z_centers, DM_Z, counts = compute_DM_vs_metallicity(latest_data)
        
        if len(Z_centers) > 0:
            # Convert to 12+log(O/H) assuming solar O/H = 8.69
            Z_solar = 0.0134  # solar metallicity
            OH_centers = 8.69 + np.log10(Z_centers / Z_solar)
            
            valid = ~np.isnan(DM_Z) & (counts > 5)
            ax3.plot(OH_centers[valid], DM_Z[valid], 'o-', linewidth=2, markersize=8, 
                    color='purple', label='Simulation')
            
            # Plot observations
            for key, obs in OBS_DATA.items():
                ax3.errorbar(obs['Z_gas'], obs['D/M'], yerr=obs['D/M_err'],
                           fmt='s', markersize=10, capsize=5, label=obs['label'], alpha=0.7)
            
            ax3.set_xlabel('12 + log(O/H)', fontsize=12)
            ax3.set_ylabel('D/M', fontsize=12)
            ax3.set_title('D/M vs Metallicity (Latest Snapshot)', fontsize=14, fontweight='bold')
            ax3.grid(alpha=0.3)
            ax3.legend(fontsize=9)
            ax3.set_ylim(0, 1.0)
    
    # Plot 4: Radial profile (latest snapshot)
    if latest_data is not None:
        ax4 = fig.add_subplot(gs[1, 2])
        r_centers, DM_r = compute_radial_profile(latest_data)
        
        if len(r_centers) > 0:
            valid = ~np.isnan(DM_r)
            ax4.plot(r_centers[valid], DM_r[valid], 'o-', linewidth=2, markersize=6, color='teal')
            ax4.axhline(OBS_DATA['MW']['D/M'], color='green', linestyle='--', label='MW', linewidth=2)
            ax4.set_xlabel('Radius [kpc]', fontsize=12)
            ax4.set_ylabel('D/M', fontsize=12)
            ax4.set_title('Radial D/M Profile', fontsize=14, fontweight='bold')
            ax4.grid(alpha=0.3)
            ax4.legend()
            ax4.set_ylim(0, max(1.0, np.nanmax(DM_r) * 1.1))
    
    # Plot 5: 2D dust map (latest snapshot)
    if latest_data is not None and len(latest_data['dust_pos']) > 0:
        ax5 = fig.add_subplot(gs[2, 0])
        center_dust = np.average(latest_data['gas_pos'], weights=latest_data['gas_mass'], axis=0)
        
        pos_centered_dust = latest_data['dust_pos'] - center_dust
        extent_dust = 30  # kpc
        
        H, xedges, yedges = np.histogram2d(
            pos_centered_dust[:, 0], pos_centered_dust[:, 1],
            bins=50, range=[(-extent_dust, extent_dust), (-extent_dust, extent_dust)],
            weights=latest_data['dust_mass']
        )
        
        im = ax5.imshow(H.T, origin='lower', extent=[-extent_dust, extent_dust, -extent_dust, extent_dust],
                       cmap='hot', aspect='auto', interpolation='nearest')
        ax5.set_xlabel('X [kpc]', fontsize=12)
        ax5.set_ylabel('Y [kpc]', fontsize=12)
        ax5.set_title('Dust Surface Density', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax5, label='Dust Mass [code units]')
    
    # Plot 6: 2D metal map (latest snapshot)
    if latest_data is not None and len(latest_data['gas_pos']) > 0:
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Recompute center for this plot
        center_metal = np.average(latest_data['gas_pos'], weights=latest_data['gas_mass'], axis=0)
        extent_metal = 30  # kpc
        
        pos_centered_metal = latest_data['gas_pos'] - center_metal
        
        H, xedges, yedges = np.histogram2d(
            pos_centered_metal[:, 0], pos_centered_metal[:, 1],
            bins=50, range=[(-extent_metal, extent_metal), (-extent_metal, extent_metal)],
            weights=latest_data['gas_metal']
        )
        
        im = ax6.imshow(H.T, origin='lower', extent=[-extent_metal, extent_metal, -extent_metal, extent_metal],
                       cmap='viridis', aspect='auto', interpolation='nearest')
        ax6.set_xlabel('X [kpc]', fontsize=12)
        ax6.set_ylabel('Y [kpc]', fontsize=12)
        ax6.set_title('Metal Surface Density', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax6, label='Metal Mass [code units]')
    
    # Plot 7: 2D D/M map (latest snapshot)
    if latest_data is not None and len(latest_data['dust_pos']) > 0:
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Recompute center and positions for this plot
        center_dm = np.average(latest_data['gas_pos'], weights=latest_data['gas_mass'], axis=0)
        extent_dm = 30  # kpc
        
        # Compute D/M in 2D bins
        pos_dust_centered = latest_data['dust_pos'] - center_dm
        dust_hist, xedges, yedges = np.histogram2d(
            pos_dust_centered[:, 0], pos_dust_centered[:, 1],
            bins=30, range=[(-extent_dm, extent_dm), (-extent_dm, extent_dm)],
            weights=latest_data['dust_mass']
        )
        
        pos_gas_centered = latest_data['gas_pos'] - center_dm
        metal_hist, _, _ = np.histogram2d(
            pos_gas_centered[:, 0], pos_gas_centered[:, 1],
            bins=30, range=[(-extent_dm, extent_dm), (-extent_dm, extent_dm)],
            weights=latest_data['gas_metal']
        )
        
        DM_map = np.zeros_like(dust_hist)
        mask = metal_hist > 0
        DM_map[mask] = dust_hist[mask] / metal_hist[mask]
        DM_map[~mask] = np.nan
        
        im = ax7.imshow(DM_map.T, origin='lower', extent=[-extent_dm, extent_dm, -extent_dm, extent_dm],
                       cmap='RdYlBu_r', aspect='auto', interpolation='nearest',
                       vmin=0, vmax=1.0)
        ax7.set_xlabel('X [kpc]', fontsize=12)
        ax7.set_ylabel('Y [kpc]', fontsize=12)
        ax7.set_title('D/M Ratio Map', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax7, label='D/M')
    
    plt.suptitle(f'Dust-to-Metal Ratio Analysis: {os.path.basename(snapshot_dir)}',
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_file = f'{output_prefix}_DM_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    
    # Save numerical data
    np.savez(f'{output_prefix}_DM_data.npz',
             times=times, redshifts=redshifts, DM_global=DM_global,
             total_dust=total_dust, total_metals=total_metals)
    print(f"Saved data to {output_prefix}_DM_data.npz")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final D/M ratio: {DM_global[-1]:.3f}")
    print(f"  MW observed:   {OBS_DATA['MW']['D/M']:.3f}")
    print(f"  LMC observed:  {OBS_DATA['LMC']['D/M']:.3f}")
    print(f"\nFinal dust mass:  {total_dust[-1]:.3e} code units")
    print(f"Final metal mass: {total_metals[-1]:.3e} code units")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_dust_to_metal.py <snapshot_directory> [output_prefix]")
        print("\nThis script analyzes ALL snapshots in the given directory.")
        print("\nExamples:")
        print("  python analyze_dust_to_metal.py ../5_output_zoom_512_halo569_50Mpc_dust")
        print("  python analyze_dust_to_metal.py ../snapdir_049 my_analysis")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'dust_analysis'
    
    # Check if input is actually a directory
    if not os.path.isdir(snapshot_dir):
        print(f"ERROR: {snapshot_dir} is not a directory")
        print("This script expects a directory containing snapshot files.")
        print("\nIf you want to analyze a single snapshot, use:")
        print("  python radial_DM_profile.py <snapshot_file>")
        sys.exit(1)
    
    analyze_snapshots(snapshot_dir, output_prefix)
