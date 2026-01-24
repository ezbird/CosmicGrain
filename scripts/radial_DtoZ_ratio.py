#!/usr/bin/env python3
"""
Radial dust-to-metal ratio profile for a specific halo
Focuses on detailed radial structure and statistics
"""

import numpy as np
import h5py
import glob
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def find_halo_center(data, method='gas_com', use_highres_only=True):
    """
    Find the center of the halo using different methods
    
    Parameters:
    -----------
    data : dict
        Snapshot data
    method : str
        'gas_com' - gas center of mass
        'dense_gas' - center of mass of densest 10% of gas
        'stellar' - stellar center of mass
        'potential' - minimum potential (if available)
    use_highres_only : bool
        For zoom simulations, filter out low-resolution boundary particles
        
    Returns:
    --------
    center : array
        3D position of halo center [x, y, z]
    """
    
    # For zoom simulations, identify high-res particles by their smaller mass
    # Low-res boundary particles are typically 8-1000x more massive
    if use_highres_only:
        if len(data['star_mass']) > 10:
            # Use stellar masses to identify resolution
            star_mass_median = np.median(data['star_mass'])
            star_highres_mask = data['star_mass'] < star_mass_median * 3  # Keep particles within 3x median
        else:
            star_highres_mask = np.ones(len(data['star_mass']), dtype=bool)
        
        if len(data['gas_mass']) > 10:
            gas_mass_median = np.median(data['gas_mass'])
            gas_highres_mask = data['gas_mass'] < gas_mass_median * 3
        else:
            gas_highres_mask = np.ones(len(data['gas_mass']), dtype=bool)
    else:
        star_highres_mask = np.ones(len(data['star_mass']), dtype=bool)
        gas_highres_mask = np.ones(len(data['gas_mass']), dtype=bool)
    
    if method == 'gas_com':
        center = np.average(data['gas_pos'][gas_highres_mask], 
                          weights=data['gas_mass'][gas_highres_mask], axis=0)
    
    elif method == 'dense_gas':
        # Use densest gas as proxy for galaxy center
        if 'gas_density' in data and len(data['gas_density']) > 0:
            gas_hr = data['gas_pos'][gas_highres_mask]
            gas_mass_hr = data['gas_mass'][gas_highres_mask]
            gas_dens_hr = data['gas_density'][gas_highres_mask]
            
            density_threshold = np.percentile(gas_dens_hr, 90)
            dense_mask = gas_dens_hr > density_threshold
            center = np.average(gas_hr[dense_mask], weights=gas_mass_hr[dense_mask], axis=0)
        else:
            # Fall back to regular COM
            center = np.average(data['gas_pos'][gas_highres_mask], 
                              weights=data['gas_mass'][gas_highres_mask], axis=0)
    
    elif method == 'stellar':
        if len(data['star_pos']) > 0:
            star_hr = data['star_pos'][star_highres_mask]
            star_mass_hr = data['star_mass'][star_highres_mask]
            center = np.average(star_hr, weights=star_mass_hr, axis=0)
        else:
            # Fall back to gas COM
            center = np.average(data['gas_pos'][gas_highres_mask], 
                              weights=data['gas_mass'][gas_highres_mask], axis=0)
    
    else:
        # Default to gas COM
        center = np.average(data['gas_pos'][gas_highres_mask], 
                          weights=data['gas_mass'][gas_highres_mask], axis=0)
    
    return center


def read_snapshot_full(snapshot_path, verbose=False):
    """
    Read snapshot with additional fields for radial analysis
    """
    
    # Handle multi-file snapshots
    base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
    snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
    if not snapshot_files:
        snapshot_files = [snapshot_path]
    
    if verbose:
        print(f"Reading {len(snapshot_files)} file(s)...")
    
    data = {
        'dust_mass': [],
        'dust_pos': [],
        'gas_mass': [],
        'gas_metal': [],
        'gas_pos': [],
        'gas_metallicity': [],
        'gas_density': [],
        'gas_temp': [],
        'star_mass': [],
        'star_pos': [],
    }
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            # Read header (only once)
            if 'time' not in data:
                data['time'] = f['Header'].attrs['Time']
                data['redshift'] = f['Header'].attrs['Redshift']
                data['boxsize'] = f['Header'].attrs['BoxSize']
            
            # Dust particles (PartType6)
            if 'PartType6' in f:
                data['dust_mass'].append(f['PartType6/Masses'][:])
                data['dust_pos'].append(f['PartType6/Coordinates'][:])
            
            # Gas particles (PartType0)
            if 'PartType0' in f:
                data['gas_mass'].append(f['PartType0/Masses'][:])
                data['gas_pos'].append(f['PartType0/Coordinates'][:])
                
                # Metallicity
                if 'GFM_Metallicity' in f['PartType0']:
                    metallicity = f['PartType0/GFM_Metallicity'][:]
                elif 'Metallicity' in f['PartType0']:
                    metallicity = f['PartType0/Metallicity'][:]
                else:
                    metallicity = np.zeros(len(f['PartType0/Masses']))
                
                data['gas_metallicity'].append(metallicity)
                data['gas_metal'].append(f['PartType0/Masses'][:] * metallicity)
                
                # Density
                if 'Density' in f['PartType0']:
                    data['gas_density'].append(f['PartType0/Density'][:])
                
                # Temperature
                if 'Temperature' in f['PartType0']:
                    data['gas_temp'].append(f['PartType0/Temperature'][:])
                elif 'InternalEnergy' in f['PartType0']:
                    # Approximate temperature from internal energy
                    u = f['PartType0/InternalEnergy'][:]
                    # Assuming mean molecular weight ~ 0.6 and gamma=5/3
                    T = (2.0/3.0) * 0.6 * 1.67e-24 * u * (3.086e21)**2 / 1.38e-16  # rough estimate
                    data['gas_temp'].append(T)
            
            # Star particles (PartType4)
            if 'PartType4' in f:
                data['star_mass'].append(f['PartType4/Masses'][:])
                data['star_pos'].append(f['PartType4/Coordinates'][:])
    
    # Concatenate arrays
    for key in data.keys():
        if isinstance(data[key], list) and data[key]:
            data[key] = np.concatenate(data[key])
        elif isinstance(data[key], list):
            data[key] = np.array([])
    
    if verbose:
        print(f"Time = {data['time']:.3f}, Redshift = {data['redshift']:.3f}")
        print(f"Particles: Dust={len(data['dust_mass'])}, Gas={len(data['gas_mass'])}, Stars={len(data['star_mass'])}")
    
    return data


def compute_radial_profile_detailed(data, center, r_bins='adaptive'):
    """
    Compute detailed radial profiles
    
    Parameters:
    -----------
    data : dict
        Snapshot data
    center : array
        Halo center coordinates
    r_bins : str or array
        'adaptive' - fine bins in center, coarse outside
        'logarithmic' - log-spaced bins
        array - custom bin edges in kpc
        
    Returns:
    --------
    dict with radial profiles
    """
    
    # Define radial bins
    if r_bins == 'adaptive':
        # Coarser resolution for better statistics with limited dust particles
        inner_bins = np.arange(0, 10, 2)    # 2 kpc bins to 10 kpc
        middle_bins = np.arange(10, 30, 5)  # 5 kpc bins to 30 kpc
        outer_bins = np.arange(30, 100, 10) # 10 kpc bins to 100 kpc
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    elif r_bins == 'coarse':
        # Very coarse bins for sparse dust
        r_bins = np.array([0, 5, 10, 15, 20, 30, 50, 100])
    elif r_bins == 'logarithmic':
        r_bins = np.logspace(-0.5, 2, 40)  # ~0.3 to 100 kpc
    
    # Compute radii
    dust_r = np.linalg.norm(data['dust_pos'] - center, axis=1)
    gas_r = np.linalg.norm(data['gas_pos'] - center, axis=1)
    
    # Initialize output
    n_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    profile = {
        'r_bins': r_bins,
        'r_centers': r_centers,
        'DM_ratio': np.zeros(n_bins),
        'dust_mass': np.zeros(n_bins),
        'metal_mass': np.zeros(n_bins),
        'gas_mass': np.zeros(n_bins),
        'n_dust': np.zeros(n_bins, dtype=int),
        'n_gas': np.zeros(n_bins, dtype=int),
        'avg_metallicity': np.zeros(n_bins),
        'avg_density': np.zeros(n_bins),
        'avg_temp': np.zeros(n_bins),
    }
    
    # Bin data
    for i in range(n_bins):
        # Dust in bin
        dust_mask = (dust_r >= r_bins[i]) & (dust_r < r_bins[i+1])
        profile['dust_mass'][i] = np.sum(data['dust_mass'][dust_mask])
        profile['n_dust'][i] = np.sum(dust_mask)
        
        # Gas/metals in bin
        gas_mask = (gas_r >= r_bins[i]) & (gas_r < r_bins[i+1])
        profile['metal_mass'][i] = np.sum(data['gas_metal'][gas_mask])
        profile['gas_mass'][i] = np.sum(data['gas_mass'][gas_mask])
        profile['n_gas'][i] = np.sum(gas_mask)
        
        if np.sum(gas_mask) > 0:
            profile['avg_metallicity'][i] = np.average(data['gas_metallicity'][gas_mask])
            if len(data['gas_density']) > 0:
                profile['avg_density'][i] = np.average(data['gas_density'][gas_mask])
            if len(data['gas_temp']) > 0:
                profile['avg_temp'][i] = np.average(data['gas_temp'][gas_mask])
        
        # D/M ratio
        if profile['metal_mass'][i] > 0:
            profile['DM_ratio'][i] = profile['dust_mass'][i] / profile['metal_mass'][i]
        else:
            profile['DM_ratio'][i] = np.nan
    
    return profile


def plot_radial_profiles(profile, data, bin_type='adaptive', output_file='radial_DM_profile.png'):
    """
    Create comprehensive radial profile plots
    
    Parameters:
    -----------
    profile : dict
        Radial profile data
    data : dict  
        Snapshot data
    bin_type : str
        Binning type used ('adaptive', 'coarse', 'logarithmic')
    output_file : str
        Output filename
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    r = profile['r_centers']
    
    # Adjust particle count requirement based on bin type
    # Coarse/logarithmic bins are wider, so we can require fewer particles
    if bin_type == 'coarse':
        min_dust = 1  # Very coarse bins should have enough particles
    elif bin_type == 'logarithmic':
        min_dust = 2  # Log bins vary in size
    else:
        min_dust = 3  # Adaptive bins
    
    # Plot 1: D/M ratio vs radius
    ax1 = fig.add_subplot(gs[0, :])
    # Require at least 20 gas particles AND min_dust dust particles for reliability
    valid = (~np.isnan(profile['DM_ratio']) & 
             (profile['n_gas'] > 20) & 
             (profile['n_dust'] >= min_dust) &
             (profile['DM_ratio'] <= 1.0))  # D/M cannot exceed 1.0
    
    if np.sum(valid) == 0:
        ax1.text(0.5, 0.5, 'Insufficient particles for reliable D/M profile', 
                ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.set_xlabel('Radius [kpc]', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Dust-to-Metal Ratio (D/M)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Radial D/M Profile - Halo #569 (z={data["redshift"]:.2f})', 
                      fontsize=16, fontweight='bold')
    else:
        ax1.plot(r[valid], profile['DM_ratio'][valid], 'o-', linewidth=2.5, 
                 markersize=8, color='darkblue', label='Simulation')
        
        # Observational benchmarks
        ax1.axhline(0.45, color='green', linestyle='--', linewidth=2, label='MW (D/M ~ 0.45)', alpha=0.7)
        ax1.axhline(0.25, color='orange', linestyle='--', linewidth=2, label='LMC (D/M ~ 0.25)', alpha=0.7)
        ax1.axhline(0.15, color='red', linestyle='--', linewidth=2, label='SMC (D/M ~ 0.15)', alpha=0.7)
        
        ax1.set_xlabel('Radius [kpc]', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Dust-to-Metal Ratio (D/M)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Radial D/M Profile - Halo #569 (z={data["redshift"]:.2f})', 
                      fontsize=16, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, min(1.0, np.nanmax(profile['DM_ratio'][valid]) * 1.1))
        ax1.set_xlim(0, np.max(r[valid]) * 1.05)
    
    # Plot 2: Dust and metal mass vs radius
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(r, profile['dust_mass'], 'o-', label='Dust', color='brown', linewidth=2, markersize=6)
    ax2.semilogy(r, profile['metal_mass'], 's-', label='Metals', color='gray', linewidth=2, markersize=6)
    ax2.set_xlabel('Radius [kpc]', fontsize=12)
    ax2.set_ylabel('Mass [code units]', fontsize=12)
    ax2.set_title('Radial Mass Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Particle counts
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(r, profile['n_dust'], 'o-', label='Dust particles', color='brown', linewidth=2, markersize=6)
    ax3.semilogy(r, profile['n_gas'], 's-', label='Gas particles', color='blue', linewidth=2, markersize=6)
    ax3.set_xlabel('Radius [kpc]', fontsize=12)
    ax3.set_ylabel('Number of Particles', fontsize=12)
    ax3.set_title('Particle Counts per Bin', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Average metallicity
    ax4 = fig.add_subplot(gs[1, 2])
    valid_Z = (profile['avg_metallicity'] > 0) & (profile['n_gas'] > 10)
    ax4.plot(r[valid_Z], profile['avg_metallicity'][valid_Z], 'o-', 
             color='purple', linewidth=2, markersize=6)
    ax4.set_xlabel('Radius [kpc]', fontsize=12)
    ax4.set_ylabel('Average Metallicity [Z]', fontsize=12)
    ax4.set_title('Metallicity Profile', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # Plot 5: Cumulative D/M
    ax5 = fig.add_subplot(gs[2, 0])
    cumulative_dust = np.cumsum(profile['dust_mass'])
    cumulative_metal = np.cumsum(profile['metal_mass'])
    cumulative_DM = np.zeros_like(cumulative_dust)
    mask_cumul = cumulative_metal > 0
    cumulative_DM[mask_cumul] = cumulative_dust[mask_cumul] / cumulative_metal[mask_cumul]
    
    ax5.plot(r, cumulative_DM, 'o-', color='teal', linewidth=2, markersize=6)
    ax5.axhline(0.45, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax5.axhline(0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax5.set_xlabel('Radius [kpc]', fontsize=12)
    ax5.set_ylabel('Cumulative D/M', fontsize=12)
    ax5.set_title('Cumulative D/M (enclosed)', fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1.0)
    
    # Plot 6: Surface density
    ax6 = fig.add_subplot(gs[2, 1])
    # Compute surface densities (mass / annulus area)
    r_bins = profile['r_bins']
    annulus_area = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    dust_surface_density = profile['dust_mass'] / annulus_area
    metal_surface_density = profile['metal_mass'] / annulus_area
    
    valid_surf = (dust_surface_density > 0) & (metal_surface_density > 0)
    ax6.loglog(r[valid_surf], dust_surface_density[valid_surf], 'o-', 
              label='Dust', color='brown', linewidth=2, markersize=6)
    ax6.loglog(r[valid_surf], metal_surface_density[valid_surf], 's-', 
              label='Metals', color='gray', linewidth=2, markersize=6)
    ax6.set_xlabel('Radius [kpc]', fontsize=12)
    ax6.set_ylabel('Surface Density [M$_\\odot$/kpc$^2$]', fontsize=12)
    ax6.set_title('Surface Density Profiles', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3, which='both')
    
    # Plot 7: D/M vs metallicity (colored by radius)
    ax7 = fig.add_subplot(gs[2, 2])
    valid_scatter = ~np.isnan(profile['DM_ratio']) & (profile['avg_metallicity'] > 0) & (profile['n_gas'] > 10)
    
    scatter = ax7.scatter(profile['avg_metallicity'][valid_scatter], 
                         profile['DM_ratio'][valid_scatter],
                         c=r[valid_scatter], s=100, cmap='viridis', 
                         edgecolors='black', linewidth=1)
    
    ax7.axhline(0.45, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax7.axhline(0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax7.set_xlabel('Metallicity [Z]', fontsize=12)
    ax7.set_ylabel('D/M', fontsize=12)
    ax7.set_title('D/M vs Metallicity', fontsize=13, fontweight='bold')
    ax7.grid(alpha=0.3)
    ax7.set_ylim(0, 1.0)
    
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Radius [kpc]', fontsize=11)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    
    return fig


def print_summary(profile, data, bin_type='adaptive'):
    """Print summary statistics"""
    
    # Adjust particle count requirement based on bin type
    if bin_type == 'coarse':
        min_dust = 1
    elif bin_type == 'logarithmic':
        min_dust = 2
    else:
        min_dust = 3
    
    print("\n" + "="*70)
    print(f"RADIAL D/M PROFILE SUMMARY - Halo #569")
    print("="*70)
    print(f"Snapshot time: {data['time']:.3f} code units")
    print(f"Redshift: {data['redshift']:.3f}")
    print(f"Binning: {bin_type}")
    print(f"\nTotal particles:")
    print(f"  Dust: {len(data['dust_mass']):,}")
    print(f"  Gas:  {len(data['gas_mass']):,}")
    print(f"  Stars: {len(data['star_mass']):,}")
    
    # Find radii with good statistics (updated criteria)
    valid = (~np.isnan(profile['DM_ratio']) & 
             (profile['n_gas'] > 20) & 
             (profile['n_dust'] >= min_dust) &
             (profile['DM_ratio'] <= 1.0))
    
    if np.sum(valid) > 0:
        print(f"\nRadial D/M statistics (bins with >20 gas AND >={min_dust} dust particles):")
        print(f"  Number of valid bins: {np.sum(valid)}")
        print(f"  Radial range: {profile['r_centers'][valid].min():.1f} - {profile['r_centers'][valid].max():.1f} kpc")
        
        print(f"\n  Inner region (r < 5 kpc):")
        inner_mask = valid & (profile['r_centers'] < 5)
        if np.sum(inner_mask) > 0:
            print(f"    D/M = {np.median(profile['DM_ratio'][inner_mask]):.3f} (median)")
            print(f"    D/M range = [{np.min(profile['DM_ratio'][inner_mask]):.3f}, {np.max(profile['DM_ratio'][inner_mask]):.3f}]")
        else:
            print(f"    No valid bins")
        
        print(f"  Middle region (5 < r < 15 kpc):")
        middle_mask = valid & (profile['r_centers'] >= 5) & (profile['r_centers'] < 15)
        if np.sum(middle_mask) > 0:
            print(f"    D/M = {np.median(profile['DM_ratio'][middle_mask]):.3f} (median)")
            print(f"    D/M range = [{np.min(profile['DM_ratio'][middle_mask]):.3f}, {np.max(profile['DM_ratio'][middle_mask]):.3f}]")
        else:
            print(f"    No valid bins")
        
        print(f"  Outer region (r > 15 kpc):")
        outer_mask = valid & (profile['r_centers'] >= 15)
        if np.sum(outer_mask) > 0:
            print(f"    D/M = {np.median(profile['DM_ratio'][outer_mask]):.3f} (median)")
            print(f"    D/M range = [{np.min(profile['DM_ratio'][outer_mask]):.3f}, {np.max(profile['DM_ratio'][outer_mask]):.3f}]")
        else:
            print(f"    No valid bins")
    else:
        print(f"\n⚠️  WARNING: No bins meet quality criteria (>20 gas, >={min_dust} dust particles)")
        print(f"    This usually means dust particles are too sparse for radial analysis.")
        print(f"    Try using coarser bins: python script.py <snapshot> coarse")
    # Global values
    total_dust = np.sum(profile['dust_mass'])
    total_metal = np.sum(profile['metal_mass'])
    global_DM = total_dust / total_metal if total_metal > 0 else 0
    
    print(f"\nGlobal values:")
    print(f"  Total dust:  {total_dust:.4e} code units")
    print(f"  Total metal: {total_metal:.4e} code units")
    print(f"  Global D/M:  {global_DM:.3f}")
    print(f"\nComparison to observations:")
    print(f"  MW  (D/M ~ 0.45): {global_DM/0.45:.2f}x")
    print(f"  LMC (D/M ~ 0.25): {global_DM/0.25:.2f}x")
    print(f"  SMC (D/M ~ 0.15): {global_DM/0.15:.2f}x")
    print("="*70)


if __name__ == '__main__':
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python radial_DM_profile.py <snapshot_directory_or_file> [bin_type]")
        print("\nBin types:")
        print("  adaptive    - 2-10 kpc bins (default) - good for ~10k+ dust particles")
        print("  coarse      - Very wide bins (5-20 kpc) - use for sparse dust (<20k particles)")
        print("  logarithmic - Log-spaced bins from 0.3 to 100 kpc")
        print("\nExamples:")
        print("  python radial_DM_profile.py ../snapdir_049/")
        print("  python radial_DM_profile.py ../snapdir_049/ coarse")
        print("  python radial_DM_profile.py ../snapdir_049/snapshot_049.0.hdf5")
        sys.exit(1)
    
    input_path = sys.argv[1]
    bin_type = sys.argv[2] if len(sys.argv) > 2 else 'adaptive'
    
    # Determine if input is a directory or file
    if os.path.isdir(input_path):
        # Find snapshot files in directory
        print(f"Searching for snapshots in {input_path}...")
        snapshot_files = sorted(glob.glob(os.path.join(input_path, 'snapshot_*.0.hdf5')))
        if not snapshot_files:
            snapshot_files = sorted(glob.glob(os.path.join(input_path, 'snapshot_*.hdf5')))
        
        if not snapshot_files:
            print(f"ERROR: No snapshot files found in {input_path}")
            print("Looking for files matching: snapshot_*.hdf5 or snapshot_*.0.hdf5")
            sys.exit(1)
        
        # Use the last (most recent) snapshot
        snapshot_file = snapshot_files[-1]
        print(f"Found {len(snapshot_files)} snapshot(s), using latest: {os.path.basename(snapshot_file)}")
        
        # Get snapshot number for output naming
        snap_num = os.path.basename(snapshot_file).split('_')[1].split('.')[0]
    else:
        # Input is a file
        snapshot_file = input_path
        snap_num = os.path.basename(snapshot_file).split('_')[1].split('.')[0]
    
    print(f"\nLoading snapshot...")
    data = read_snapshot_full(snapshot_file, verbose=True)
    
    print("\nFinding halo center...")
    
    # Show particle mass ranges to diagnose multi-resolution issues
    if len(data['star_mass']) > 0:
        print(f"Star particle masses: min={np.min(data['star_mass']):.3e}, "
              f"median={np.median(data['star_mass']):.3e}, max={np.max(data['star_mass']):.3e}")
        star_mass_ratio = np.max(data['star_mass']) / np.min(data['star_mass'])
        if star_mass_ratio > 10:
            print(f"  → Mass range spans {star_mass_ratio:.1f}x - multi-resolution zoom detected!")
            print(f"  → Filtering out low-res particles (keeping mass < {np.median(data['star_mass'])*3:.3e})")
    
    if len(data['gas_mass']) > 0:
        print(f"Gas particle masses:  min={np.min(data['gas_mass']):.3e}, "
              f"median={np.median(data['gas_mass']):.3e}, max={np.max(data['gas_mass']):.3e}")
        gas_mass_ratio = np.max(data['gas_mass']) / np.min(data['gas_mass'])
        if gas_mass_ratio > 10:
            print(f"  → Mass range spans {gas_mass_ratio:.1f}x - multi-resolution zoom detected!")
    
    # Compute both centers for diagnostics
    if len(data['star_pos']) > 0:
        stellar_center = find_halo_center(data, method='stellar', use_highres_only=True)
        print(f"\nStellar center (high-res only): [{stellar_center[0]:.2f}, {stellar_center[1]:.2f}, {stellar_center[2]:.2f}] kpc")
    else:
        stellar_center = None
        print("WARNING: No stellar particles found!")
    
    gas_center = find_halo_center(data, method='gas_com', use_highres_only=True)
    print(f"Gas center (high-res only):     [{gas_center[0]:.2f}, {gas_center[1]:.2f}, {gas_center[2]:.2f}] kpc")
    
    if stellar_center is not None:
        offset = np.linalg.norm(stellar_center - gas_center)
        print(f"Offset between centers: {offset:.2f} kpc")
        if offset > 20:
            print("  → Large offset detected! Using stellar center (more reliable for galaxies)")
    
    center = find_halo_center(data, method='stellar', use_highres_only=True)
    print(f"\nUsing center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] kpc")
    
    print(f"\nComputing radial profiles with '{bin_type}' binning...")
    profile = compute_radial_profile_detailed(data, center, r_bins=bin_type)
    
    print("\nCreating plots...")
    output_file = f'halo569_snap{snap_num}_radial_DM_{bin_type}.png'
    plot_radial_profiles(profile, data, bin_type=bin_type, output_file=output_file)
    
    print_summary(profile, data, bin_type=bin_type)
    
    # Save numerical data
    data_file = f'halo569_snap{snap_num}_radial_DM_{bin_type}.npz'
    np.savez(data_file, **profile, center=center)
    print(f"\nSaved data to {data_file}")
