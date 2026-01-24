#!/usr/bin/env python3
"""
Enhanced radial dust-to-metal ratio profile with diagnostics
Includes halo particle filtering and dust distribution analysis
"""

import numpy as np
import h5py
import glob
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_halo_particle_ids(halo_file, halo_id, particle_types=['gas', 'stars', 'dust']):
    """
    Load particle IDs for a specific halo from halo finder output
    """
    halo_pids = {}
    
    if halo_file.endswith('.hdf5'):
        with h5py.File(halo_file, 'r') as f:
            try:
                group_key = f'Group/{halo_id}'
                
                if 'gas' in particle_types:
                    if f'{group_key}/PartType0/ParticleIDs' in f:
                        halo_pids['gas'] = f[f'{group_key}/PartType0/ParticleIDs'][:]
                
                if 'stars' in particle_types:
                    if f'{group_key}/PartType4/ParticleIDs' in f:
                        halo_pids['stars'] = f[f'{group_key}/PartType4/ParticleIDs'][:]
                
                if 'dust' in particle_types:
                    if f'{group_key}/PartType6/ParticleIDs' in f:
                        halo_pids['dust'] = f[f'{group_key}/PartType6/ParticleIDs'][:]
                        
            except KeyError:
                print(f"Warning: Could not find halo {halo_id} in {halo_file}")
                print("Available keys:", list(f.keys()))
    
    elif halo_file.endswith('.txt') or halo_file.endswith('.dat'):
        data = np.loadtxt(halo_file, dtype={'names': ('type', 'id'), 
                                             'formats': ('i4', 'i8')})
        
        if 'gas' in particle_types:
            halo_pids['gas'] = np.unique(data['id'][data['type'] == 0])
        if 'stars' in particle_types:
            halo_pids['stars'] = np.unique(data['id'][data['type'] == 4])
        if 'dust' in particle_types:
            halo_pids['dust'] = np.unique(data['id'][data['type'] == 6])
    
    else:
        raise ValueError(f"Unknown halo file format: {halo_file}")
    
    print(f"\nLoaded particle IDs for halo {halo_id}:")
    for ptype, ids in halo_pids.items():
        print(f"  {ptype}: {len(ids)} particles")
    
    return halo_pids


def read_snapshot_with_filter(snapshot_path, halo_pids=None, verbose=False):
    """
    Read snapshot and filter by halo particle IDs
    """
    
    if os.path.isdir(snapshot_path):
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not snapshot_files:
            snapshot_files = [snapshot_path]
    
    if verbose:
        print(f"Reading {len(snapshot_files)} file(s)...")
        print(f"First file: {os.path.basename(snapshot_files[0])}")
        if len(snapshot_files) > 1:
            print(f"Last file:  {os.path.basename(snapshot_files[-1])}")
    
    data = {
        'dust_pos': [], 'dust_mass': [], 'dust_ids': [], 'dust_grain_radius': [],
        'gas_pos': [], 'gas_mass': [], 'gas_metals': [], 'gas_density': [], 'gas_ids': [], 'gas_temp': [],
        'star_pos': [], 'star_mass': [], 'star_metals': [], 'star_ids': []
    }
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            header = f['Header'].attrs
            time = header['Time']
            redshift = header['Redshift']
            box_size = header['BoxSize']
            
            # Dust (PartType6)
            if 'PartType6' in f:
                dust_ids = f['PartType6/ParticleIDs'][:]
                
                if halo_pids and 'dust' in halo_pids:
                    mask = np.isin(dust_ids, halo_pids['dust'])
                    dust_ids = dust_ids[mask]
                else:
                    mask = np.ones(len(dust_ids), dtype=bool)
                
                if np.sum(mask) > 0:
                    data['dust_pos'].append(f['PartType6/Coordinates'][:][mask])
                    data['dust_mass'].append(f['PartType6/Masses'][:][mask])
                    data['dust_ids'].append(dust_ids)
                    
                    # Try to get grain radius if available
                    if 'GrainRadius' in f['PartType6']:
                        data['dust_grain_radius'].append(f['PartType6/GrainRadius'][:][mask])
            
            # Gas (PartType0)
            if 'PartType0' in f:
                gas_ids = f['PartType0/ParticleIDs'][:]
                
                if halo_pids and 'gas' in halo_pids:
                    mask = np.isin(gas_ids, halo_pids['gas'])
                    gas_ids = gas_ids[mask]
                else:
                    mask = np.ones(len(gas_ids), dtype=bool)
                
                if np.sum(mask) > 0:
                    data['gas_pos'].append(f['PartType0/Coordinates'][:][mask])
                    data['gas_mass'].append(f['PartType0/Masses'][:][mask])
                    data['gas_ids'].append(gas_ids)
                    
                    if 'Metallicity' in f['PartType0']:
                        data['gas_metals'].append(f['PartType0/Metallicity'][:][mask])
                    
                    if 'Density' in f['PartType0']:
                        data['gas_density'].append(f['PartType0/Density'][:][mask])
                    
                    # Try to get temperature
                    if 'InternalEnergy' in f['PartType0']:
                        u = f['PartType0/InternalEnergy'][:][mask]
                        # Rough temperature estimate (adjust units as needed)
                        temp = u * (2.0/3.0) * 1.2e10  # Approximate conversion
                        data['gas_temp'].append(temp)
            
            # Stars (PartType4)
            if 'PartType4' in f:
                star_ids = f['PartType4/ParticleIDs'][:]
                
                if halo_pids and 'stars' in halo_pids:
                    mask = np.isin(star_ids, halo_pids['stars'])
                    star_ids = star_ids[mask]
                else:
                    mask = np.ones(len(star_ids), dtype=bool)
                
                if np.sum(mask) > 0:
                    data['star_pos'].append(f['PartType4/Coordinates'][:][mask])
                    data['star_mass'].append(f['PartType4/Masses'][:][mask])
                    data['star_ids'].append(star_ids)
                    
                    if 'Metallicity' in f['PartType4']:
                        data['star_metals'].append(f['PartType4/Metallicity'][:][mask])
    
    # Concatenate arrays
    for key in data:
        if data[key]:
            data[key] = np.concatenate(data[key])
        else:
            data[key] = np.array([])
    
    data['time'] = time
    data['redshift'] = redshift
    data['box_size'] = box_size
    
    if verbose:
        print(f"Time = {time:.3f}, Redshift = {redshift:.3f}, BoxSize = {box_size:.1f} kpc")
        print(f"Loaded particles: Dust={len(data['dust_pos'])}, "
              f"Gas={len(data['gas_pos'])}, Stars={len(data['star_pos'])}")
    
    return data


def analyze_dust_distribution(data, center, verbose=True):
    """
    Comprehensive dust distribution analysis
    """
    if len(data['dust_pos']) == 0:
        print("No dust particles to analyze!")
        return None
    
    # Calculate radii
    dust_r = np.sqrt(np.sum((data['dust_pos'] - center)**2, axis=1))
    
    print("\n" + "="*60)
    print("DUST DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Radial distribution
    print("\nRadial Distribution:")
    radial_bins = [0, 5, 10, 20, 30, 50, 100, 150, 200, 500, 1000]
    for i in range(len(radial_bins)-1):
        mask = (dust_r >= radial_bins[i]) & (dust_r < radial_bins[i+1])
        count = np.sum(mask)
        if count > 0:
            mass = np.sum(data['dust_mass'][mask])
            print(f"  {radial_bins[i]:4.0f}-{radial_bins[i+1]:4.0f} kpc: "
                  f"{count:6d} particles, {mass*1e10:10.3e} Msun")
    
    # Mass distribution analysis
    print("\nMass Distribution (identifying low-res vs high-res):")
    mass_ranges = [
        (1e-9, 1e-8, "Ultra-low (likely high-res)"),
        (1e-8, 1e-7, "Low (typical high-res)"),
        (1e-7, 1e-6, "Medium (boundary)"),
        (1e-6, 1e-5, "High (possibly low-res)"),
        (1e-5, 1e-4, "Very high (likely low-res)"),
        (1e-4, 1e0,  "Extreme (definitely low-res)")
    ]
    
    for min_m, max_m, label in mass_ranges:
        mask = (data['dust_mass'] >= min_m) & (data['dust_mass'] < max_m)
        count = np.sum(mask)
        if count > 0:
            radii = dust_r[mask]
            print(f"  {label}:")
            print(f"    Count: {count}, Avg radius: {np.mean(radii):.1f} kpc, "
                  f"Max radius: {np.max(radii):.1f} kpc")
    
    # Identify problematic particles
    print("\nProblematic Dust Particles:")
    far_mask = dust_r > 100  # Beyond 100 kpc
    if np.sum(far_mask) > 0:
        print(f"  Found {np.sum(far_mask)} dust particles beyond 100 kpc:")
        far_masses = data['dust_mass'][far_mask]
        far_radii = dust_r[far_mask]
        
        # Show first 10
        for i in range(min(10, len(far_masses))):
            print(f"    Particle {i+1}: r={far_radii[i]:.1f} kpc, "
                  f"mass={far_masses[i]:.3e} ({far_masses[i]*1e10:.3e} Msun)")
        
        # Check if these are low-res particles
        high_mass_far = np.sum(far_masses > 1e-5)
        if high_mass_far > 0:
            print(f"\n  WARNING: {high_mass_far} far dust particles have high mass (>1e-5)")
            print("  These are likely from LOW-RESOLUTION region!")
    
    # Grain size distribution (if available)
    if len(data['dust_grain_radius']) > 0:
        print("\nGrain Size Distribution:")
        grain_nm = data['dust_grain_radius'] * 1e7  # Convert to nm
        
        # Check for invalid grains
        invalid = (grain_nm <= 0) | ~np.isfinite(grain_nm)
        if np.sum(invalid) > 0:
            print(f"  WARNING: {np.sum(invalid)} particles have invalid grain size!")
        
        valid = ~invalid
        if np.sum(valid) > 0:
            print(f"  Min: {np.min(grain_nm[valid]):.2f} nm")
            print(f"  Max: {np.max(grain_nm[valid]):.2f} nm")
            print(f"  Mean: {np.mean(grain_nm[valid]):.2f} nm")
            print(f"  Median: {np.median(grain_nm[valid]):.2f} nm")
    
    # Temperature environment (if gas temp available)
    if len(data['gas_temp']) > 0:
        print("\nDust Environment Analysis:")
        # Find nearest gas for some dust particles
        sample_size = min(100, len(data['dust_pos']))
        sample_indices = np.random.choice(len(data['dust_pos']), sample_size, replace=False)
        
        hot_dust_count = 0
        for idx in sample_indices:
            dust_pos = data['dust_pos'][idx]
            # Simple nearest neighbor - just for diagnostic
            gas_dist = np.sqrt(np.sum((data['gas_pos'] - dust_pos)**2, axis=1))
            if len(gas_dist) > 0:
                nearest = np.argmin(gas_dist)
                if data['gas_temp'][nearest] > 1e6:
                    hot_dust_count += 1
        
        if hot_dust_count > 0:
            print(f"  WARNING: {hot_dust_count}/{sample_size} sampled dust particles "
                  f"near hot gas (T>1e6 K)")
            print("  These should be destroyed by thermal sputtering!")
    
    print("="*60 + "\n")
    
    return {
        'dust_r': dust_r,
        'far_dust_count': np.sum(far_mask) if 'far_mask' in locals() else 0,
        'high_mass_count': np.sum(data['dust_mass'] > 1e-5)
    }


def find_halo_center(data, method='stellar', verbose=False):
    """Find halo center using various methods"""
    
    if method == 'stellar' and len(data['star_pos']) > 0:
        center = np.average(data['star_pos'], weights=data['star_mass'], axis=0)
        if verbose:
            print(f"Center (stellar COM): [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] kpc")
    
    elif method == 'dense_gas' and len(data['gas_pos']) > 0:
        if len(data['gas_density']) > 0:
            density_threshold = np.percentile(data['gas_density'], 90)
            mask = data['gas_density'] > density_threshold
            if np.sum(mask) > 0:
                center = np.average(data['gas_pos'][mask], weights=data['gas_mass'][mask], axis=0)
            else:
                center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
        else:
            center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
    
    else:
        if len(data['gas_pos']) > 0:
            center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
        else:
            # Use box center as last resort
            box_size = data.get('box_size', 50000)
            center = np.array([box_size/2, box_size/2, box_size/2])
            if verbose:
                print(f"Warning: Using box center as halo center: {center}")
    
    return center


def compute_radial_profile(data, center, bin_mode='adaptive', r_max=None, verbose=False):
    """Compute radial D/M profile with extended range for diagnostics"""
    
    # Compute radii
    if len(data['dust_pos']) > 0:
        dust_r = np.sqrt(np.sum((data['dust_pos'] - center)**2, axis=1))
    else:
        dust_r = np.array([])
        if verbose:
            print("Warning: No dust particles found!")
        return None
    
    gas_r = np.sqrt(np.sum((data['gas_pos'] - center)**2, axis=1))
    
    # Determine r_max - extend if dust is found at large radii
    if r_max is None:
        max_dust_r = np.max(dust_r) if len(dust_r) > 0 else 100
        max_gas_r = np.percentile(gas_r, 99) if len(gas_r) > 0 else 100
        
        # Use larger of the two, but cap at reasonable value
        r_max = min(max(max_dust_r * 1.1, max_gas_r, 100), 500)
        
        if verbose:
            print(f"Auto-determined r_max = {r_max:.1f} kpc")
            if max_dust_r > 100:
                print(f"  WARNING: Dust extends to {max_dust_r:.1f} kpc!")
    
    # Create bins - use more bins if extending to large radii
    if r_max > 150:
        # Extended bins for diagnostic purposes
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 50, 21)[1:]
        outer_bins = np.linspace(50, r_max, 16)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    elif bin_mode == 'linear':
        r_bins = np.linspace(0, r_max, 31)
    elif bin_mode == 'log':
        r_bins = np.logspace(np.log10(0.5), np.log10(r_max), 30)
        r_bins = np.concatenate([[0], r_bins])
    else:  # adaptive
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 30, 11)[1:]
        outer_bins = np.linspace(30, r_max, 11)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    
    n_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # Initialize arrays
    profile = {
        'r_bins': r_bins,
        'r_centers': r_centers,
        'dust_mass': np.zeros(n_bins),
        'metal_mass': np.zeros(n_bins),
        'gas_mass': np.zeros(n_bins),
        'dust_mass_cum': np.zeros(n_bins),
        'metal_mass_cum': np.zeros(n_bins),
        'gas_mass_cum': np.zeros(n_bins),
        'dust_count': np.zeros(n_bins, dtype=int),
        'gas_count': np.zeros(n_bins, dtype=int),
        'DM_ratio': np.zeros(n_bins),
        'DG_ratio': np.zeros(n_bins),
        'r_max': r_max
    }
    
    # Bin dust particles
    dust_bin_idx = np.digitize(dust_r, r_bins) - 1
    for i in range(n_bins):
        mask = (dust_bin_idx == i)
        if np.sum(mask) > 0:
            profile['dust_mass'][i] = np.sum(data['dust_mass'][mask])
            profile['dust_count'][i] = np.sum(mask)
        
        mask_cum = (dust_r < r_bins[i+1])
        if np.sum(mask_cum) > 0:
            profile['dust_mass_cum'][i] = np.sum(data['dust_mass'][mask_cum])
    
    # Bin gas particles
    gas_bin_idx = np.digitize(gas_r, r_bins) - 1
    for i in range(n_bins):
        mask = (gas_bin_idx == i)
        if np.sum(mask) > 0:
            profile['gas_mass'][i] = np.sum(data['gas_mass'][mask])
            if len(data['gas_metals']) > 0:
                profile['metal_mass'][i] = np.sum(data['gas_mass'][mask] * data['gas_metals'][mask])
            profile['gas_count'][i] = np.sum(mask)
        
        mask_cum = (gas_r < r_bins[i+1])
        if np.sum(mask_cum) > 0:
            profile['gas_mass_cum'][i] = np.sum(data['gas_mass'][mask_cum])
            if len(data['gas_metals']) > 0:
                profile['metal_mass_cum'][i] = np.sum(data['gas_mass'][mask_cum] * 
                                                      data['gas_metals'][mask_cum])
    
    # Compute ratios
    mask_metals = profile['metal_mass'] > 0
    profile['DM_ratio'][mask_metals] = (profile['dust_mass'][mask_metals] / 
                                        profile['metal_mass'][mask_metals])
    
    mask_gas = profile['gas_mass'] > 0
    profile['DG_ratio'][mask_gas] = profile['dust_mass'][mask_gas] / profile['gas_mass'][mask_gas]
    
    # Clip unrealistic values
    profile['DM_ratio'] = np.clip(profile['DM_ratio'], 0, 1.0)
    profile['DG_ratio'] = np.clip(profile['DG_ratio'], 0, 0.1)
    
    return profile


def plot_radial_profiles(profile, data, dust_analysis=None, halo_id=None, output_file=None):
    """Enhanced plotting with diagnostic information"""
    
    if profile is None:
        print("No profile data to plot!")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Add title with diagnostic info
    title_str = f'Radial Plots'
    if halo_id is not None:
        title_str += f' for Halo {halo_id}'
    
    # Add warning if dust at large radii
    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        title_str += f'\nWARNING: {dust_analysis["far_dust_count"]} dust particles beyond 100 kpc!'
    
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, f'z = {data["redshift"]:.2f}', ha='center', fontsize=11)
    
    r = profile['r_centers']
    r_max_plot = min(profile['r_max'], 200)  # Limit plot range for clarity
    
    output_name = data.get('output_name', 'Simulation')
    
    # Plot 1: Dust-to-Metal ratio
    ax1 = fig.add_subplot(gs[0, 0])
    mask = (profile['DM_ratio'] > 0) & (profile['metal_mass'] > 0)
    ax1.plot(r[mask], profile['DM_ratio'][mask], 'o-', color='darkblue', lw=2, 
             label=output_name, markersize=4)
    
    # Observational references
    r_mw_disk = np.linspace(0, 8, 100)
    ax1.plot(r_mw_disk, np.full_like(r_mw_disk, 0.5), '--', 
             color='orange', alpha=0.7, lw=2, label='MW disk')
    
    r_mw_outer = np.linspace(8, 20, 100)
    dm_outer = 0.5 - 0.2 * (r_mw_outer - 8) / 12
    ax1.plot(r_mw_outer, dm_outer, '--', color='orange', alpha=0.7, lw=2)
    
    ax1.axhspan(0.1, 0.3, alpha=0.15, color='red', label='Dwarf galaxies')
    
    # Mark suspicious regions
    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        ax1.axvspan(100, r_max_plot, alpha=0.2, color='red', label='Anomalous region')
    
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('Dust-to-Metal Ratio')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, r_max_plot)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Dust-to-Metal Ratio')
    
    # Plot 2: Dust-to-Gas ratio
    ax2 = fig.add_subplot(gs[0, 1])
    mask = (profile['DG_ratio'] > 0) & (profile['gas_mass'] > 0)
    ax2.plot(r[mask], profile['DG_ratio'][mask] * 100, 'o-', 
             color='darkgreen', lw=2, label=output_name, markersize=4)
    
    # References
    r_mw_disk = np.linspace(0, 8, 100)
    ax2.plot(r_mw_disk, np.full_like(r_mw_disk, 0.7), '--', 
             color='orange', alpha=0.7, lw=2, label='MW disk')
    
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Dust-to-Gas Ratio (%)')
    ax2.set_ylim(0, 2.0)
    ax2.set_xlim(0, r_max_plot)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Dust-to-Gas Ratio')
    
    # Plot 3: Dust mass profile (cumulative)
    ax3 = fig.add_subplot(gs[1, 0])
    mask = profile['dust_mass_cum'] > 0
    if np.sum(mask) > 0:
        dust_mass_msun = profile['dust_mass_cum'][mask] * 1e10
        ax3.semilogy(r[mask], dust_mass_msun, 'o-', color='brown', lw=2, label=output_name)
        ax3.axhline(1e7, color='orange', ls='--', alpha=0.5, lw=1.5, label='MW total')
    ax3.set_xlabel('Radius (kpc)')
    ax3.set_ylabel(r'Dust Mass (M$_{\odot}$)')
    ax3.set_xlim(0, r_max_plot)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Cumulative Dust Mass')
    ax3.legend()
    
    # Plot 4: Metal mass profile (cumulative)
    ax4 = fig.add_subplot(gs[1, 1])
    mask = profile['metal_mass_cum'] > 0
    if np.sum(mask) > 0:
        metal_mass_msun = profile['metal_mass_cum'][mask] * 1e10
        ax4.semilogy(r[mask], metal_mass_msun, 'o-', color='purple', lw=2, label=output_name)
    ax4.set_xlabel('Radius (kpc)')
    ax4.set_ylabel(r'Metal Mass (M$_{\odot}$)')
    ax4.set_xlim(0, r_max_plot)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Cumulative Metal Mass')
    ax4.legend()
    
    # Plot 5: Particle counts (LOG SCALE)
    ax5 = fig.add_subplot(gs[2, 0])
    mask_dust = profile['dust_count'] > 0
    mask_gas = profile['gas_count'] > 0
    
    if np.sum(mask_dust) > 0:
        ax5.semilogy(r[mask_dust], profile['dust_count'][mask_dust], 
                     'o-', color='brown', label='Dust', lw=2)
    if np.sum(mask_gas) > 0:
        ax5.semilogy(r[mask_gas], profile['gas_count'][mask_gas], 
                     's-', color='blue', label='Gas', lw=1, alpha=0.5, markersize=3)
    
    ax5.set_xlabel('Radius (kpc)')
    ax5.set_ylabel('Particle Count (log scale)')
    ax5.set_xlim(0, r_max_plot)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title('Particle Counts')
    
    # Plot 6: Dust radial distribution (new diagnostic plot)
    ax6 = fig.add_subplot(gs[2, 1])
    if dust_analysis and 'dust_r' in dust_analysis:
        dust_r = dust_analysis['dust_r']
        
        # Histogram of dust locations
        hist_bins = np.linspace(0, min(np.max(dust_r), r_max_plot), 50)
        ax6.hist(dust_r[dust_r < r_max_plot], bins=hist_bins, 
                 color='brown', alpha=0.7, edgecolor='black')
        
        # Mark regions
        ax6.axvline(20, color='green', ls='--', alpha=0.5, label='Typical disk')
        ax6.axvline(50, color='orange', ls='--', alpha=0.5, label='Extended halo')
        ax6.axvline(100, color='red', ls='--', alpha=0.5, label='Anomalous')
        
        ax6.set_xlabel('Radius (kpc)')
        ax6.set_ylabel('Number of Dust Particles')
        ax6.set_xlim(0, r_max_plot)
        ax6.set_title('Dust Particle Distribution')
        ax6.legend(fontsize=8)
    
    # Plot 7: Surface densities
    ax7 = fig.add_subplot(gs[3, 0])
    r_bins = profile['r_bins']
    annulus_area = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    
    # Prevent division by zero for very small annuli
    annulus_area[annulus_area < 1e-10] = 1e-10
    
    dust_surf = profile['dust_mass'] / annulus_area * 1e10
    metal_surf = profile['metal_mass'] / annulus_area * 1e10
    
    mask_dust = (dust_surf > 0) & (r < r_max_plot)
    mask_metal = (metal_surf > 0) & (r < r_max_plot)
    
    if np.sum(mask_dust) > 0:
        ax7.semilogy(r[mask_dust], dust_surf[mask_dust], 'o-', 
                     color='brown', label='Dust', lw=2)
    if np.sum(mask_metal) > 0:
        ax7.semilogy(r[mask_metal], metal_surf[mask_metal], 's-', 
                     color='purple', label='Metals', lw=1, alpha=0.7, markersize=3)
    
    ax7.set_xlabel('Radius (kpc)')
    ax7.set_ylabel(r'Surface Density (M$_{\odot}$ kpc$^{-2}$)')
    ax7.set_xlim(0, r_max_plot)
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.set_title('Surface Densities')
    
    # Plot 8: Mass distribution diagnostic
    ax8 = fig.add_subplot(gs[3, 1])
    if len(data['dust_mass']) > 0:
        dust_masses_log = np.log10(data['dust_mass'][data['dust_mass'] > 0])
        ax8.hist(dust_masses_log, bins=30, color='brown', alpha=0.7, edgecolor='black')
        
        # Mark regions
        ax8.axvline(np.log10(1e-7), color='green', ls='--', alpha=0.5, label='Typical high-res')
        ax8.axvline(np.log10(1e-5), color='red', ls='--', alpha=0.5, label='Low-res boundary')
        
        ax8.set_xlabel('log10(Dust Mass) [code units]')
        ax8.set_ylabel('Count')
        ax8.set_title('Dust Mass Distribution')
        ax8.legend(fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python radial_plots.py <snapshot_or_dir> [halo_file] [halo_id]")
        print("\nExamples:")
        print("  # No filtering:")
        print("  python radial_plots.py snapshot_049.hdf5")
        print("  python radial_plots.py snapdir_049/")
        print("\n  # With halo filtering:")
        print("  python radial_plots.py snapshot_049.hdf5 halo569_particles.txt 569")
        print("  python radial_plots.py snapdir_049/ halo569_particles.txt 569")
        sys.exit(1)
    
    snapshot_path = sys.argv[1]
    
    # Check if halo filtering requested
    halo_pids = None
    halo_id = None
    if len(sys.argv) >= 4:
        halo_file = sys.argv[2]
        halo_id = int(sys.argv[3])
        print(f"Loading particle IDs for halo {halo_id} from {halo_file}...")
        halo_pids = load_halo_particle_ids(halo_file, halo_id)
    else:
        print("No halo filtering - analyzing all particles")
    
    # Load snapshot
    print(f"\nLoading snapshot {snapshot_path}...")
    data = read_snapshot_with_filter(snapshot_path, halo_pids=halo_pids, verbose=True)
    
    if len(data['gas_pos']) == 0:
        print("Error: No gas particles found!")
        sys.exit(1)
    
    # Find halo center
    print("\nFinding halo center...")
    center = find_halo_center(data, method='stellar', verbose=True)
    
    # Analyze dust distribution BEFORE computing profiles
    dust_analysis = analyze_dust_distribution(data, center, verbose=True)
    
    # Compute radial profile
    print("\nComputing radial profiles...")
    profile = compute_radial_profile(data, center, bin_mode='adaptive', verbose=True)
    
    if profile is None:
        print("Could not compute profile - no dust particles?")
        sys.exit(1)
    
    # Extract output folder name for legend
    if os.path.isdir(snapshot_path):
        output_name = os.path.basename(os.path.dirname(snapshot_path.rstrip('/')))
    else:
        output_name = os.path.basename(os.path.dirname(snapshot_path))
    
    data['output_name'] = output_name
    
    # Create output filename
    if os.path.isdir(snapshot_path):
        snap_num = snapshot_path.rstrip('/').split('_')[-1]
    else:
        snap_num = snapshot_path.split('_')[-1].split('.')[0]
    
    if halo_id is not None:
        output_base = f'halo{halo_id}_snap{snap_num}_radial_DM_filtered'
    else:
        output_base = f'snap{snap_num}_radial_DM'
    
    # Add diagnostic suffix if issues found
    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        output_base += '_DIAGNOSTIC'
    
    # Plot
    print("\nCreating plots...")
    plot_radial_profiles(profile, data, dust_analysis=dust_analysis,
                        halo_id=halo_id, output_file=f'{output_base}.png')
    
    # Save data
    np.savez(f'{output_base}.npz',
             r_centers=profile['r_centers'],
             r_bins=profile['r_bins'],
             DM_ratio=profile['DM_ratio'],
             DG_ratio=profile['DG_ratio'],
             dust_mass=profile['dust_mass'],
             metal_mass=profile['metal_mass'],
             gas_mass=profile['gas_mass'],
             dust_mass_cum=profile['dust_mass_cum'],
             metal_mass_cum=profile['metal_mass_cum'],
             gas_mass_cum=profile['gas_mass_cum'],
             dust_count=profile['dust_count'],
             gas_count=profile['gas_count'])
    
    print(f"\nDone! Outputs:")
    print(f"  - {output_base}.png")
    print(f"  - {output_base}.npz")
    
    # Print summary warnings
    if dust_analysis:
        if dust_analysis['far_dust_count'] > 0:
            print("\n" + "!"*60)
            print("WARNING: Dust particles found at anomalous distances!")
            print(f"  {dust_analysis['far_dust_count']} particles beyond 100 kpc")
            print(f"  {dust_analysis['high_mass_count']} high-mass (likely low-res) particles")
            print("  Check dust creation in low-resolution regions!")
            print("!"*60)


if __name__ == '__main__':
    main()
