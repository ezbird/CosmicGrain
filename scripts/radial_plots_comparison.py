#!/usr/bin/env python3
"""
Radial dust-to-metal ratio profile comparison script
Compare two simulations side-by-side with the same 6-panel layout
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
    
    # Check file format
    if halo_file.endswith('.hdf5'):
        # SUBFIND HDF5 format
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
    
    # Print summary
    print(f"\nLoaded particle IDs for halo {halo_id}:")
    for ptype, ids in halo_pids.items():
        print(f"  {ptype}: {len(ids)} particles")
    
    return halo_pids


def read_snapshot_with_filter(snapshot_path, halo_pids=None, verbose=False):
    """
    Read snapshot and filter by halo particle IDs
    """
    
    import os
    
    # Handle multi-file snapshots
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
        'dust_pos': [], 'dust_mass': [], 'dust_ids': [],
        'gas_pos': [], 'gas_mass': [], 'gas_metals': [], 'gas_density': [], 'gas_ids': [],
        'star_pos': [], 'star_mass': [], 'star_metals': [], 'star_ids': []
    }
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            # Read header
            header = f['Header'].attrs
            time = header['Time']
            redshift = header['Redshift']
            
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
    
    if verbose:
        print(f"Time = {time:.3f}, Redshift = {redshift:.3f}")
        print(f"Filtered particles: Dust={len(data['dust_pos'])}, "
              f"Gas={len(data['gas_pos'])}, Stars={len(data['star_pos'])}")
    
    return data


def find_halo_center(data, method='stellar', verbose=False):
    """Find the center of the halo"""
    
    if method == 'stellar' and len(data['star_pos']) > 0:
        center = np.average(data['star_pos'], axis=0, weights=data['star_mass'])
        if verbose:
            print(f"Stellar center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    elif len(data['gas_pos']) > 0:
        center = np.average(data['gas_pos'], axis=0, weights=data['gas_mass'])
        if verbose:
            print(f"Gas center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    else:
        center = np.array([0.0, 0.0, 0.0])
        if verbose:
            print("Warning: No particles found, using origin as center")
    
    return center


def compute_radial_profile(data, center, bin_mode='adaptive', n_bins=20, max_radius=None, verbose=False):
    """Compute radial profiles"""
    
    # Compute distances from center
    if len(data['dust_pos']) > 0:
        dust_r = np.sqrt(np.sum((data['dust_pos'] - center)**2, axis=1))
    else:
        dust_r = np.array([])
    
    if len(data['gas_pos']) > 0:
        gas_r = np.sqrt(np.sum((data['gas_pos'] - center)**2, axis=1))
    else:
        gas_r = np.array([])
    
    # Determine max radius
    if max_radius is None:
        if len(gas_r) > 0:
            max_radius = np.percentile(gas_r, 99)
        else:
            max_radius = 100.0
    
    # Create bins
    if bin_mode == 'adaptive':
        r_bins = np.logspace(np.log10(0.1), np.log10(max_radius), n_bins+1)
    else:
        r_bins = np.linspace(0, max_radius, n_bins+1)
    
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # Initialize profile arrays
    profile = {
        'r_centers': r_centers,
        'r_bins': r_bins,
        'dust_mass': np.zeros(n_bins),           # Mass per bin
        'metal_mass': np.zeros(n_bins),          # Mass per bin
        'gas_mass': np.zeros(n_bins),            # Mass per bin
        'dust_mass_cum': np.zeros(n_bins),       # Cumulative mass
        'metal_mass_cum': np.zeros(n_bins),      # Cumulative mass
        'gas_mass_cum': np.zeros(n_bins),        # Cumulative mass
        'dust_count': np.zeros(n_bins, dtype=int),
        'gas_count': np.zeros(n_bins, dtype=int),
        'DM_ratio': np.zeros(n_bins),
        'DG_ratio': np.zeros(n_bins)
    }
    
    # Bin particles
    for i in range(n_bins):
        r_min, r_max = r_bins[i], r_bins[i+1]
        
        # Dust (per bin)
        dust_mask = (dust_r >= r_min) & (dust_r < r_max)
        profile['dust_mass'][i] = np.sum(data['dust_mass'][dust_mask])
        profile['dust_count'][i] = np.sum(dust_mask)
        
        # Dust (cumulative - within this radius)
        dust_mask_cum = (dust_r < r_max)
        profile['dust_mass_cum'][i] = np.sum(data['dust_mass'][dust_mask_cum])
        
        # Gas (per bin)
        gas_mask = (gas_r >= r_min) & (gas_r < r_max)
        profile['gas_mass'][i] = np.sum(data['gas_mass'][gas_mask])
        profile['gas_count'][i] = np.sum(gas_mask)
        
        # Gas (cumulative)
        gas_mask_cum = (gas_r < r_max)
        profile['gas_mass_cum'][i] = np.sum(data['gas_mass'][gas_mask_cum])
        
        # Metals in gas (per bin)
        if len(data['gas_metals']) > 0:
            profile['metal_mass'][i] = np.sum(data['gas_mass'][gas_mask] * data['gas_metals'][gas_mask])
            # Cumulative
            profile['metal_mass_cum'][i] = np.sum(data['gas_mass'][gas_mask_cum] * data['gas_metals'][gas_mask_cum])
        
        # Total metals = gas metals + dust (dust is 100% metal)
        total_metals = profile['metal_mass'][i] + profile['dust_mass'][i]
        
        # D/Z ratio (use per-bin for annular D/Z)
        if total_metals > 0:
            profile['DM_ratio'][i] = profile['dust_mass'][i] / total_metals
        
        # D/G ratio (use per-bin)
        if profile['gas_mass'][i] > 0:
            profile['DG_ratio'][i] = profile['dust_mass'][i] / profile['gas_mass'][i]
    
    if verbose:
        print(f"Radial bins: {n_bins} from {r_bins[0]:.2f} to {r_bins[-1]:.2f} kpc")
        total_dust = np.sum(profile['dust_mass']) * 1e10
        total_metal = np.sum(profile['metal_mass'] + profile['dust_mass']) * 1e10
        global_DM = np.sum(profile['dust_mass']) / np.sum(profile['metal_mass'] + profile['dust_mass'])
        print(f"Total dust: {total_dust:.2e} Msun")
        print(f"Total metals: {total_metal:.2e} Msun")
        print(f"Global D/Z: {global_DM:.4f}")
    
    return profile


def plot_comparison(profile1, profile2, data1, data2, label1, label2, 
                   halo_id=None, output_file=None):
    """
    Plot comparison of two simulations
    
    Parameters:
    -----------
    profile1, profile2 : dict
        Radial profiles from both simulations
    data1, data2 : dict
        Full datasets from both simulations
    label1, label2 : str
        Labels for the two simulations (e.g., folder names)
    """
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Add title
    title = 'Radial Profile Comparison'
    if halo_id is not None:
        title += f' (Halo {halo_id})'
    title += f'\nz={data1["redshift"]:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Get radius arrays
    r1 = profile1['r_centers']
    r2 = profile2['r_centers']
    max_r = max(r1.max(), r2.max())
    
    # ========== Plot 1: Dust-to-Metal Ratio ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Simulation 1 (solid line)
    mask1 = (profile1['DM_ratio'] > 0) & (profile1['metal_mass'] > 0)
    ax1.plot(r1[mask1], profile1['DM_ratio'][mask1], 'o-', 
             color='darkblue', lw=2.5, markersize=6, label=label1, zorder=10)
    
    # Simulation 2 (dotted line)
    mask2 = (profile2['DM_ratio'] > 0) & (profile2['metal_mass'] > 0)
    ax1.plot(r2[mask2], profile2['DM_ratio'][mask2], 's:', 
             color='darkred', lw=2.5, markersize=6, label=label2, zorder=9)
    
    # Reference lines
    r_mw_disk = np.linspace(0, 8, 100)
    ax1.plot(r_mw_disk, np.full_like(r_mw_disk, 0.5), '--', 
             color='orange', alpha=0.7, lw=2,
             label='MW inner disk (Galliano+ 2021)')
    
    r_mw_outer = np.linspace(8, 20, 100)
    dm_outer = 0.5 - 0.2 * (r_mw_outer - 8) / 12
    ax1.plot(r_mw_outer, dm_outer, '--', color='orange', alpha=0.7, lw=2)
    
    ax1.axhspan(0.1, 0.3, alpha=0.15, color='red', 
                label='Dwarf galaxies (Rémy-Ruyer+ 2014)')
    
    r_cgm = np.linspace(20, 100, 100)
    ax1.plot(r_cgm, np.full_like(r_cgm, 0.05), '-.', 
             color='gray', alpha=0.5, lw=1.5,
             label='CGM upper limit (Ménard+ 2010)')
    
    ax1.set_xlabel('Radius (kpc)', fontsize=11)
    ax1.set_ylabel('Dust-to-Metal Ratio', fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, max_r)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Dust-to-Metal Ratio', fontweight='bold')
    
    # ========== Plot 2: Dust-to-Gas ratio ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Simulation 1
    mask1 = (profile1['DG_ratio'] > 0) & (profile1['gas_mass'] > 0)
    ax2.plot(r1[mask1], profile1['DG_ratio'][mask1] * 100, 'o-', 
             color='darkgreen', lw=2.5, markersize=6, label=label1, zorder=10)
    
    # Simulation 2
    mask2 = (profile2['DG_ratio'] > 0) & (profile2['gas_mass'] > 0)
    ax2.plot(r2[mask2], profile2['DG_ratio'][mask2] * 100, 's:', 
             color='darkmagenta', lw=2.5, markersize=6, label=label2, zorder=9)
    
    # References
    r_mw_disk = np.linspace(0, 8, 100)
    ax2.plot(r_mw_disk, np.full_like(r_mw_disk, 0.7), '--', 
             color='orange', alpha=0.7, lw=2,
             label='MW disk (Draine 2011)')
    
    r_mw_outer = np.linspace(8, 20, 100)
    dg_outer = 0.7 - 0.4 * (r_mw_outer - 8) / 12
    ax2.plot(r_mw_outer, dg_outer, '--', color='orange', alpha=0.7, lw=2)
    
    ax2.axhspan(0.1, 0.3, alpha=0.15, color='red',
                label='Low-Z dwarfs')
    
    r_cgm = np.linspace(20, 100, 100)
    ax2.plot(r_cgm, np.full_like(r_cgm, 0.01), '-.', 
             color='gray', alpha=0.5, lw=1.5,
             label='CGM (if any)')
    
    ax2.set_xlabel('Radius (kpc)', fontsize=11)
    ax2.set_ylabel('Dust-to-Gas Ratio (%)', fontsize=11)
    ax2.set_ylim(0, 2.0)
    ax2.set_xlim(0, max_r)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Dust-to-Gas Ratio', fontweight='bold')
    
    # ========== Plot 3: Dust mass profile (CUMULATIVE) ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Use cumulative mass for clearer comparison
    mask1 = profile1['dust_mass_cum'] > 0
    dust_mass1_msun = profile1['dust_mass_cum'][mask1] * 1e10
    ax3.semilogy(r1[mask1], dust_mass1_msun, 'o-', color='brown', 
                 lw=2.5, markersize=6, label=label1)
    
    mask2 = profile2['dust_mass_cum'] > 0
    dust_mass2_msun = profile2['dust_mass_cum'][mask2] * 1e10
    ax3.semilogy(r2[mask2], dust_mass2_msun, 's:', color='darkorange', 
                 lw=2.5, markersize=6, label=label2)
    
    ax3.axhline(1e7, color='orange', ls='--', alpha=0.5, lw=1.5,
                label='MW total dust')
    
    ax3.set_xlabel('Radius (kpc)', fontsize=11)
    ax3.set_ylabel(r'Dust Mass (M$_{\odot}$)', fontsize=11)
    ax3.set_xlim(0, max_r)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.set_title('Cumulative Dust Mass', fontweight='bold')
    
    # ========== Plot 4: Metal mass profile (CUMULATIVE) ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Use cumulative mass for clearer comparison
    mask1 = profile1['metal_mass_cum'] > 0
    metal_mass1_msun = profile1['metal_mass_cum'][mask1] * 1e10
    ax4.semilogy(r1[mask1], metal_mass1_msun, 'o-', color='purple', 
                 lw=2.5, markersize=6, label=label1)
    
    mask2 = profile2['metal_mass_cum'] > 0
    metal_mass2_msun = profile2['metal_mass_cum'][mask2] * 1e10
    ax4.semilogy(r2[mask2], metal_mass2_msun, 's:', color='indigo', 
                 lw=2.5, markersize=6, label=label2)
    
    ax4.set_xlabel('Radius (kpc)', fontsize=11)
    ax4.set_ylabel(r'Metal Mass (M$_{\odot}$)', fontsize=11)
    ax4.set_xlim(0, max_r)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_title('Cumulative Metal Mass', fontweight='bold')
    
    # ========== Plot 5: Particle counts ==========
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Sim 1
    ax5.semilogy(r1, profile1['dust_count'], 'o-', color='brown', 
                 label=f'{label1} Dust', lw=2.5, markersize=5)
    ax5.semilogy(r1, profile1['gas_count'], 's-', color='blue', 
                 label=f'{label1} Gas', lw=2.5, alpha=0.7, markersize=5)
    
    # Sim 2
    ax5.semilogy(r2, profile2['dust_count'], 'o:', color='darkorange', 
                 label=f'{label2} Dust', lw=2.5, markersize=5)
    ax5.semilogy(r2, profile2['gas_count'], 's:', color='darkblue', 
                 label=f'{label2} Gas', lw=2.5, alpha=0.7, markersize=5)
    
    ax5.set_xlabel('Radius (kpc)', fontsize=11)
    ax5.set_ylabel('Particle Count', fontsize=11)
    ax5.set_xlim(0, max_r)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    ax5.set_title('Particle Counts', fontweight='bold')
    
    # ========== Plot 6: Surface densities ==========
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Sim 1
    r_bins1 = profile1['r_bins']
    annulus_area1 = np.pi * (r_bins1[1:]**2 - r_bins1[:-1]**2)
    dust_surf1 = profile1['dust_mass'] / annulus_area1 * 1e10
    metal_surf1 = profile1['metal_mass'] / annulus_area1 * 1e10
    
    mask_dust1 = dust_surf1 > 0
    mask_metal1 = metal_surf1 > 0
    ax6.semilogy(r1[mask_dust1], dust_surf1[mask_dust1], 'o-', 
                 color='brown', label=f'{label1} Dust', lw=2.5, markersize=5)
    ax6.semilogy(r1[mask_metal1], metal_surf1[mask_metal1], 's-', 
                 color='purple', label=f'{label1} Metals', lw=2.5, alpha=0.7, markersize=5)
    
    # Sim 2
    r_bins2 = profile2['r_bins']
    annulus_area2 = np.pi * (r_bins2[1:]**2 - r_bins2[:-1]**2)
    dust_surf2 = profile2['dust_mass'] / annulus_area2 * 1e10
    metal_surf2 = profile2['metal_mass'] / annulus_area2 * 1e10
    
    mask_dust2 = dust_surf2 > 0
    mask_metal2 = metal_surf2 > 0
    ax6.semilogy(r2[mask_dust2], dust_surf2[mask_dust2], 'o:', 
                 color='darkorange', label=f'{label2} Dust', lw=2.5, markersize=5)
    ax6.semilogy(r2[mask_metal2], metal_surf2[mask_metal2], 's:', 
                 color='indigo', label=f'{label2} Metals', lw=2.5, alpha=0.7, markersize=5)
    
    # MW reference
    r_ref = np.linspace(0, 20, 100)
    sigma_dust_mw = 1e6 * np.exp(-r_ref / 5.0)
    ax6.plot(r_ref, sigma_dust_mw, '--', color='orange', alpha=0.5, lw=1.5,
             label='MW disk model')
    
    ax6.set_xlabel('Radius (kpc)', fontsize=11)
    ax6.set_ylabel(r'Surface Density (M$_{\odot}$ kpc$^{-2}$)', fontsize=11)
    ax6.set_xlim(0, max_r)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=7, ncol=2)
    ax6.set_title('Surface Densities', fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python radial_DtoZ_compare.py <snapshot1> <snapshot2> [halo_file] [halo_id]")
        print("\nExamples:")
        print("  # Compare two simulations:")
        print("  python radial_DtoZ_compare.py output1/snapdir_049/ output2/snapdir_049/")
        print("  python radial_DtoZ_compare.py output1/snapshot_049.hdf5 output2/snapshot_049.hdf5")
        print("\n  # With halo filtering:")
        print("  python radial_DtoZ_compare.py output1/snapdir_049/ output2/snapdir_049/ halo.txt 569")
        sys.exit(1)
    
    snapshot_path1 = sys.argv[1]
    snapshot_path2 = sys.argv[2]
    
    # Extract labels from folder names
    if os.path.isdir(snapshot_path1):
        label1 = os.path.basename(os.path.dirname(snapshot_path1.rstrip('/')))
    else:
        label1 = os.path.basename(os.path.dirname(snapshot_path1))
    
    if os.path.isdir(snapshot_path2):
        label2 = os.path.basename(os.path.dirname(snapshot_path2.rstrip('/')))
    else:
        label2 = os.path.basename(os.path.dirname(snapshot_path2))
    
    # Check if halo filtering requested
    halo_pids = None
    halo_id = None
    if len(sys.argv) >= 5:
        halo_file = sys.argv[3]
        halo_id = int(sys.argv[4])
        print(f"Loading particle IDs for halo {halo_id} from {halo_file}...")
        halo_pids = load_halo_particle_ids(halo_file, halo_id)
    else:
        print("No halo filtering - analyzing all particles")
    
    # Load first simulation
    print(f"\n{'='*70}")
    print(f"SIMULATION 1: {label1}")
    print(f"{'='*70}")
    print(f"Loading snapshot {snapshot_path1}...")
    data1 = read_snapshot_with_filter(snapshot_path1, halo_pids=halo_pids, verbose=True)
    
    if len(data1['gas_pos']) == 0:
        print("Error: No particles found in simulation 1!")
        sys.exit(1)
    
    print("\nFinding halo center...")
    center1 = find_halo_center(data1, method='stellar', verbose=True)
    
    print("\nComputing radial profiles...")
    profile1 = compute_radial_profile(data1, center1, bin_mode='adaptive', verbose=True)
    
    # Load second simulation
    print(f"\n{'='*70}")
    print(f"SIMULATION 2: {label2}")
    print(f"{'='*70}")
    print(f"Loading snapshot {snapshot_path2}...")
    data2 = read_snapshot_with_filter(snapshot_path2, halo_pids=halo_pids, verbose=True)
    
    if len(data2['gas_pos']) == 0:
        print("Error: No particles found in simulation 2!")
        sys.exit(1)
    
    print("\nFinding halo center...")
    center2 = find_halo_center(data2, method='stellar', verbose=True)
    
    print("\nComputing radial profiles...")
    profile2 = compute_radial_profile(data2, center2, bin_mode='adaptive', verbose=True)
    
    # Create output filename
    if os.path.isdir(snapshot_path1):
        snap_num = snapshot_path1.rstrip('/').split('_')[-1]
    else:
        snap_num = snapshot_path1.split('_')[-1].split('.')[0]
    
    if halo_id is not None:
        output_base = f'comparison_halo{halo_id}_snap{snap_num}'
    else:
        output_base = f'comparison_snap{snap_num}'
    
    # Plot comparison
    print(f"\n{'='*70}")
    print("Creating comparison plots...")
    print(f"{'='*70}")
    plot_comparison(profile1, profile2, data1, data2, label1, label2,
                   halo_id=halo_id, output_file=f'{output_base}.png')
    
    # Save data for both
    np.savez(f'{output_base}_data.npz',
             # Sim 1
             r_centers_1=profile1['r_centers'],
             DM_ratio_1=profile1['DM_ratio'],
             DG_ratio_1=profile1['DG_ratio'],
             dust_mass_1=profile1['dust_mass'],          # Per bin
             metal_mass_1=profile1['metal_mass'],        # Per bin
             gas_mass_1=profile1['gas_mass'],            # Per bin
             dust_mass_cum_1=profile1['dust_mass_cum'],  # Cumulative
             metal_mass_cum_1=profile1['metal_mass_cum'],# Cumulative
             gas_mass_cum_1=profile1['gas_mass_cum'],    # Cumulative
             # Sim 2
             r_centers_2=profile2['r_centers'],
             DM_ratio_2=profile2['DM_ratio'],
             DG_ratio_2=profile2['DG_ratio'],
             dust_mass_2=profile2['dust_mass'],          # Per bin
             metal_mass_2=profile2['metal_mass'],        # Per bin
             gas_mass_2=profile2['gas_mass'],            # Per bin
             dust_mass_cum_2=profile2['dust_mass_cum'],  # Cumulative
             metal_mass_cum_2=profile2['metal_mass_cum'],# Cumulative
             gas_mass_cum_2=profile2['gas_mass_cum'],    # Cumulative
             # Labels
             label_1=label1,
             label_2=label2)
    
    print(f"\nDone! Outputs:")
    print(f"  - {output_base}.png")
    print(f"  - {output_base}_data.npz")


if __name__ == '__main__':
    main()
