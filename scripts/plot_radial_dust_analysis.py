#!/usr/bin/env python3
"""
plot_radial_dust_analysis.py

Radial dust-to-metal and dust-to-gas ratio profiles with TEMPERATURE FILTERING.

CRITICAL: Measures D/M in WARM ISM (T<5e5 K) where dust survives thermal sputtering!

The threshold is set to DustThermalSputteringTemp (5e5 K), NOT molecular cloud 
temperature (1e4 K), because dust can survive in warm ionized gas up to ~5e5 K.

Usage:
    python plot_radial_dust_analysis.py \
        --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 \
        --snapshot ../snapdir_049/snapshot_049 \
        --out radial_dust_analysis.png \
        --rmax 200
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py

try:
    from halo_utils import compute_radial_distance
except ImportError:
    print("ERROR: This script requires halo_utils.py in the same directory")
    exit(1)


def get_target_halo_center(catalog_path):
    """Extract the position of the target halo (most massive subhalo)."""
    with h5py.File(catalog_path, 'r') as f:
        pos = f['Subhalo/SubhaloPos'][0]  # kpc
        mass = f['Subhalo/SubhaloMass'][0] * 1e10  # Convert to M☉
        
    print(f"Target halo:")
    print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] kpc")
    print(f"  Mass: {mass:.2e} M☉")
    
    return pos, mass


def get_gas_temperature(internal_energy, electron_abundance):
    """
    Calculate gas temperature from internal energy.
    
    Args:
        internal_energy: Specific internal energy in Gadget units (km/s)^2
        electron_abundance: Electron abundance (Ne)
    
    Returns:
        Temperature in Kelvin
    """
    # Physical constants
    GAMMA = 5.0/3.0
    BOLTZMANN = 1.38064852e-16  # erg/K
    PROTONMASS = 1.6726219e-24  # g
    HYDROGEN_MASSFRAC = 0.76
    
    # Mean molecular weight
    XH = HYDROGEN_MASSFRAC
    Y = (1.0 - XH) / (4.0 * XH)
    mu = (1.0 + 4.0 * Y) / (1.0 + Y + electron_abundance)
    
    # InternalEnergy in Gadget-4 is in (km/s)^2
    # Convert to (cm/s)^2 (which equals erg/g for specific energy)
    UnitVelocity_in_cm_per_s = 1e5  # 1 km/s = 1e5 cm/s
    u_cgs = internal_energy * UnitVelocity_in_cm_per_s**2
    
    # Temperature = (gamma-1) * u * m_p * mu / k_B
    temp = (GAMMA - 1.0) * u_cgs * PROTONMASS * mu / BOLTZMANN
    
    return temp


def extract_particles_in_sphere(snapshot_base, center_pos, radius_kpc, particle_type):
    """
    Extract ALL particles of given type within radius of center.
    
    For gas (type 0), also loads InternalEnergy and ElectronAbundance for temperature.
    """
    import glob
    
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        print(f"ERROR: No snapshot files found matching {snapshot_base}.*.hdf5")
        return None
    
    print(f"  Found {len(files)} snapshot files")
    
    ptype_names = {0: 'PartType0', 4: 'PartType4', 6: 'PartType6'}
    ptype_name = ptype_names.get(particle_type, f'PartType{particle_type}')
    
    all_coords = []
    all_masses = []
    all_metallicity = []
    all_internal_energy = []
    all_electron_abundance = []
    
    with h5py.File(files[0], 'r') as f:
        box_size = f['Header'].attrs['BoxSize']
    
    print(f"  Extracting {ptype_name} particles within {radius_kpc} kpc...")
    
    for file in files:
        with h5py.File(file, 'r') as f:
            if ptype_name not in f:
                continue
            
            coords = f[ptype_name]['Coordinates'][:]  # kpc
            masses = f[ptype_name]['Masses'][:]  # 1e10 M☉
            
            # Periodic distance
            dx = coords[:, 0] - center_pos[0]
            dy = coords[:, 1] - center_pos[1]
            dz = coords[:, 2] - center_pos[2]
            
            dx = np.where(dx > box_size/2, dx - box_size, dx)
            dx = np.where(dx < -box_size/2, dx + box_size, dx)
            dy = np.where(dy > box_size/2, dy - box_size, dy)
            dy = np.where(dy < -box_size/2, dy + box_size, dy)
            dz = np.where(dz > box_size/2, dz - box_size, dz)
            dz = np.where(dz < -box_size/2, dz + box_size, dz)
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            mask = r < radius_kpc
            n_selected = mask.sum()
            
            if n_selected > 0:
                all_coords.append(coords[mask])
                all_masses.append(masses[mask])
                
                # Get metallicity for gas
                if particle_type == 0:
                    if 'Metallicity' in f[ptype_name]:
                        all_metallicity.append(f[ptype_name]['Metallicity'][:][mask])
                    
                    # Get internal energy and electron abundance for temperature
                    if 'InternalEnergy' in f[ptype_name]:
                        all_internal_energy.append(f[ptype_name]['InternalEnergy'][:][mask])
                    if 'ElectronAbundance' in f[ptype_name]:
                        all_electron_abundance.append(f[ptype_name]['ElectronAbundance'][:][mask])
    
    if len(all_coords) == 0:
        print(f"  WARNING: No {ptype_name} particles found!")
        return None
    
    result = {
        'Coordinates': np.vstack(all_coords),
        'Masses': np.concatenate(all_masses) * 1e10  # Convert to M☉
    }
    
    if len(all_metallicity) > 0:
        result['Metallicity'] = np.concatenate(all_metallicity)
    
    if len(all_internal_energy) > 0:
        result['InternalEnergy'] = np.concatenate(all_internal_energy)
    
    if len(all_electron_abundance) > 0:
        result['ElectronAbundance'] = np.concatenate(all_electron_abundance)
    
    # Calculate temperature for gas
    if particle_type == 0 and 'InternalEnergy' in result:
        ne = result.get('ElectronAbundance', np.ones(len(result['Masses'])))
        result['Temperature'] = get_gas_temperature(result['InternalEnergy'], ne)
        
        T = result['Temperature']
        print(f"  Temperature range: {T.min():.1e} - {T.max():.1e} K")
        
        # Dust survival threshold (T < 5e5 K)
        warm_mask = T < 5e5
        hot_mask = T >= 5e5
        
        print(f"  Gas below sputtering (T<5e5 K): {warm_mask.sum():,} particles ({100*warm_mask.sum()/len(T):.1f}%)")
        print(f"  Gas above sputtering (T>5e5 K): {hot_mask.sum():,} particles ({100*hot_mask.sum()/len(T):.1f}%)")
    
    print(f"  Extracted {len(result['Masses']):,} particles, "
          f"total mass = {result['Masses'].sum():.2e} M☉")
    
    return result


def compute_radial_profile(halo, halo_pos, r_max=200):
    """
    Compute radial profiles with TEMPERATURE FILTERING.
    
    Returns profiles for:
    - All gas (traditional measurement)
    - Warm ISM only (T < 5e5 K, below thermal sputtering threshold)
    """
    
    # Set up radial bins
    if r_max > 150:
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 50, 21)[1:]
        outer_bins = np.linspace(50, r_max, 16)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    else:
        inner_bins = np.linspace(0, 10, 11)
        middle_bins = np.linspace(10, 30, 11)[1:]
        outer_bins = np.linspace(30, r_max, 11)[1:]
        r_bins = np.concatenate([inner_bins, middle_bins, outer_bins])
    
    n_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    
    # Initialize profile arrays
    profile = {
        'r_bins': r_bins,
        'r_centers': r_centers,
        'dust_mass': np.zeros(n_bins),
        'metal_mass_all': np.zeros(n_bins),   # All gas
        'metal_mass_warm': np.zeros(n_bins),  # Warm ISM (T<5e5 K)
        'gas_mass_all': np.zeros(n_bins),
        'gas_mass_warm': np.zeros(n_bins),
        'dust_count': np.zeros(n_bins, dtype=int),
        'gas_count_all': np.zeros(n_bins, dtype=int),
        'gas_count_warm': np.zeros(n_bins, dtype=int),
        'DM_ratio_all': np.zeros(n_bins),
        'DM_ratio_warm': np.zeros(n_bins),
        'DG_ratio_all': np.zeros(n_bins),
        'DG_ratio_warm': np.zeros(n_bins),
    }
    
    # Dust particles (unchanged)
    if 'dust' in halo:
        dust_r = compute_radial_distance(halo['dust']['Coordinates'], halo_pos)
        dust_bin_idx = np.digitize(dust_r, r_bins) - 1
        
        for i in range(n_bins):
            mask = (dust_bin_idx == i)
            if mask.sum() > 0:
                profile['dust_mass'][i] = np.sum(halo['dust']['Masses'][mask])
                profile['dust_count'][i] = int(mask.sum())
    
    # Gas particles with temperature filter
    if 'gas' in halo:
        gas_r = compute_radial_distance(halo['gas']['Coordinates'], halo_pos)
        gas_bin_idx = np.digitize(gas_r, r_bins) - 1
        
        # Temperature filter: warm ISM where dust survives (T < 5e5 K)
        T_gas = halo['gas'].get('Temperature', np.ones(len(halo['gas']['Masses'])) * 1e5)
        warm_mask = T_gas < 5e5  # K - below thermal sputtering threshold
        
        for i in range(n_bins):
            mask_all = (gas_bin_idx == i)
            mask_warm = mask_all & warm_mask
            
            # All gas statistics
            if mask_all.sum() > 0:
                profile['gas_mass_all'][i] = np.sum(halo['gas']['Masses'][mask_all])
                profile['gas_count_all'][i] = int(mask_all.sum())
                
                if 'Metallicity' in halo['gas']:
                    profile['metal_mass_all'][i] = np.sum(
                        halo['gas']['Masses'][mask_all] * halo['gas']['Metallicity'][mask_all]
                    )
            
            # Warm ISM statistics
            if mask_warm.sum() > 0:
                profile['gas_mass_warm'][i] = np.sum(halo['gas']['Masses'][mask_warm])
                profile['gas_count_warm'][i] = int(mask_warm.sum())
                
                if 'Metallicity' in halo['gas']:
                    profile['metal_mass_warm'][i] = np.sum(
                        halo['gas']['Masses'][mask_warm] * halo['gas']['Metallicity'][mask_warm]
                    )
    
    # Compute ratios for ALL gas
    mask_metals = profile['metal_mass_all'] > 0
    profile['DM_ratio_all'][mask_metals] = (
        profile['dust_mass'][mask_metals] / profile['metal_mass_all'][mask_metals]
    )
    
    mask_gas = profile['gas_mass_all'] > 0
    profile['DG_ratio_all'][mask_gas] = (
        profile['dust_mass'][mask_gas] / profile['gas_mass_all'][mask_gas]
    )
    
    # Compute ratios for WARM ISM only
    mask_metals_warm = profile['metal_mass_warm'] > 0
    profile['DM_ratio_warm'][mask_metals_warm] = (
        profile['dust_mass'][mask_metals_warm] / profile['metal_mass_warm'][mask_metals_warm]
    )
    
    mask_gas_warm = profile['gas_mass_warm'] > 0
    profile['DG_ratio_warm'][mask_gas_warm] = (
        profile['dust_mass'][mask_gas_warm] / profile['gas_mass_warm'][mask_gas_warm]
    )
    
    # Clip to reasonable ranges
    profile['DM_ratio_all'] = np.clip(profile['DM_ratio_all'], 0, 1.0)
    profile['DM_ratio_warm'] = np.clip(profile['DM_ratio_warm'], 0, 1.0)
    profile['DG_ratio_all'] = np.clip(profile['DG_ratio_all'], 0, 0.1)
    profile['DG_ratio_warm'] = np.clip(profile['DG_ratio_warm'], 0, 0.1)
    
    return profile


def analyze_dust_distribution(halo, halo_pos):
    """Print dust distribution diagnostics."""
    
    if 'dust' not in halo or len(halo['dust']['Coordinates']) == 0:
        print("No dust particles to analyze!")
        return None
    
    dust_r = compute_radial_distance(halo['dust']['Coordinates'], halo_pos)
    dust_masses = halo['dust']['Masses']
    
    print("\n" + "="*60)
    print("DUST DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal dust particles: {len(dust_r):,}")
    print(f"Total dust mass: {dust_masses.sum():.2e} M☉")
    
    print("\nRadial Distribution:")
    radial_bins = [0, 5, 10, 20, 30, 50, 100, 150, 200, 500]
    for i in range(len(radial_bins)-1):
        mask = (dust_r >= radial_bins[i]) & (dust_r < radial_bins[i+1])
        count = int(mask.sum())
        if count > 0:
            mass = float(np.sum(dust_masses[mask]))
            print(f"  {radial_bins[i]:4.0f}-{radial_bins[i+1]:4.0f} kpc: "
                  f"{count:6d} particles, {mass:10.3e} M☉")
    
    print("\nMass Distribution:")
    print(f"  Min mass: {dust_masses.min():.2e} M☉")
    print(f"  Max mass: {dust_masses.max():.2e} M☉")
    print(f"  Median mass: {np.median(dust_masses):.2e} M☉")
    
    # Check for far particles
    far_mask = dust_r > 100
    far_count = int(far_mask.sum())
    if far_count > 0:
        print(f"\nWARNING: {far_count} dust particles beyond 100 kpc")
        far_frac = 100.0 * far_count / len(dust_r)
        print(f"  ({far_frac:.1f}% of total)")
    
    print("="*60 + "\n")
    
    return {
        'dust_r': dust_r,
        'far_dust_count': far_count,
        'dust_masses': dust_masses
    }


def plot_radial_profiles(profile, halo_mass, redshift, dust_analysis=None, output_file=None):
    """Create comprehensive radial profile plots with temperature-filtered ratios."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, top=0.92)
    
    # Title
    title_str = f'Radial Dust Analysis (Thermal Sputtering Threshold) - Target Halo (M={halo_mass:.2e} M☉) - z = {redshift:.3f}'
  
    fig.suptitle(title_str, fontsize=14, fontweight='bold')
    
    r = profile['r_centers']
    r_max_plot = min(profile['r_bins'][-1], 200)
    
    # 1. Dust-to-Metal Ratio (BOTH all gas and warm ISM)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # All gas (gray, dashed)
    mask_all = (profile['DM_ratio_all'] > 0) & (profile['metal_mass_all'] > 0)
    ax1.plot(r[mask_all], profile['DM_ratio_all'][mask_all], 's--', 
             color='gray', lw=1.5, markersize=4, alpha=0.5, label='D/M (all gas)')
    
    # Warm ISM (blue, solid) - THIS IS THE PHYSICAL MEASUREMENT
    mask_warm = (profile['DM_ratio_warm'] > 0) & (profile['metal_mass_warm'] > 0)
    ax1.plot(r[mask_warm], profile['DM_ratio_warm'][mask_warm], 'o-', 
             color='darkblue', lw=2, markersize=5, label='D/M (warm ISM, T<5e5 K)')
    
    ax1.axhline(0.4, color='red', linestyle='--', alpha=0.7, lw=2, label='MW ISM (0.4)')
    ax1.set_xlabel('Radius (kpc)', fontsize=10)
    ax1.set_ylabel('Dust-to-Metal Ratio', fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, r_max_plot)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    ax1.set_title('Dust-to-Metal Ratio (Warm ISM vs All Gas)', fontweight='bold')
    
    # 2. Dust-to-Gas Ratio (BOTH)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # All gas (gray, dashed)
    mask_all = (profile['DG_ratio_all'] > 0) & (profile['gas_mass_all'] > 0)
    ax2.plot(r[mask_all], profile['DG_ratio_all'][mask_all] * 100, 's--', 
             color='gray', lw=1.5, markersize=4, alpha=0.5, label='D/G (all gas)')
    
    # Warm ISM (green, solid)
    mask_warm = (profile['DG_ratio_warm'] > 0) & (profile['gas_mass_warm'] > 0)
    ax2.plot(r[mask_warm], profile['DG_ratio_warm'][mask_warm] * 100, 'o-', 
             color='darkgreen', lw=2, markersize=5, label='D/G (warm ISM, T<5e5 K)')
    
    ax2.axhline(0.7, color='red', linestyle='--', alpha=0.7, lw=2, label='MW value (~0.7%)')
    ax2.set_xlabel('Radius (kpc)', fontsize=10)
    ax2.set_ylabel('Dust-to-Gas Ratio (%)', fontsize=10)
    ax2.set_ylim(0, 5.0)
    ax2.set_xlim(0, r_max_plot)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    ax2.set_title('Dust-to-Gas Ratio (Warm ISM vs All Gas)', fontweight='bold')
    
    # 3. Dust Mass Profile
    ax3 = fig.add_subplot(gs[0, 2])
    mask = profile['dust_mass'] > 0
    ax3.semilogy(r[mask], profile['dust_mass'][mask], 'o-', color='brown', 
                 lw=2, markersize=5)
    ax3.set_xlabel('Radius (kpc)', fontsize=10)
    ax3.set_ylabel('Dust Mass per bin (M$_\\odot$)', fontsize=10)
    ax3.set_xlim(0, r_max_plot)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Dust Mass Profile', fontweight='bold')
    
    # 4. Gas Mass Profile (All vs Warm)
    ax4 = fig.add_subplot(gs[1, 0])
    mask_all = profile['gas_mass_all'] > 0
    mask_warm = profile['gas_mass_warm'] > 0
    ax4.semilogy(r[mask_all], profile['gas_mass_all'][mask_all], 's--', 
                color='lightblue', lw=1.5, markersize=4, alpha=0.7, label='All gas')
    ax4.semilogy(r[mask_warm], profile['gas_mass_warm'][mask_warm], 'o-', 
                color='steelblue', lw=2, markersize=5, label='Warm ISM (T<5e5 K)')
    ax4.set_xlabel('Radius (kpc)', fontsize=10)
    ax4.set_ylabel('Gas Mass per bin (M$_\\odot$)', fontsize=10)
    ax4.set_xlim(0, r_max_plot)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=8)
    ax4.set_title('Gas Mass Profile (All vs Warm ISM)', fontweight='bold')
    
    # 5. Metal Mass Profile (All vs Warm)
    ax5 = fig.add_subplot(gs[1, 1])
    mask_all = profile['metal_mass_all'] > 0
    mask_warm = profile['metal_mass_warm'] > 0
    ax5.semilogy(r[mask_all], profile['metal_mass_all'][mask_all], 's--', 
                color='orange', lw=1.5, markersize=4, alpha=0.5, label='All gas')
    ax5.semilogy(r[mask_warm], profile['metal_mass_warm'][mask_warm], 'o-', 
                color='darkorange', lw=2, markersize=5, label='Warm ISM (T<5e5 K)')
    ax5.set_xlabel('Radius (kpc)', fontsize=10)
    ax5.set_ylabel('Metal Mass per bin (M$_\\odot$)', fontsize=10)
    ax5.set_xlim(0, r_max_plot)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=8)
    ax5.set_title('Metal Mass Profile (All vs Warm ISM)', fontweight='bold')
    
    # 6. Particle Counts
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(r, profile['dust_count'], 'o-', color='brown', 
                 lw=2, markersize=5, label='Dust')
    ax6.semilogy(r, profile['gas_count_all'], 's--', color='lightblue', 
                 lw=1.5, markersize=4, alpha=0.7, label='Gas (all)')
    ax6.semilogy(r, profile['gas_count_warm'], 'o-', color='steelblue', 
                 lw=2, markersize=5, label='Gas (warm ISM)')
    ax6.set_xlabel('Radius (kpc)', fontsize=10)
    ax6.set_ylabel('Particle Count per bin', fontsize=10)
    ax6.set_xlim(0, r_max_plot)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=8)
    ax6.set_title('Particle Counts', fontweight='bold')
    
    # 7. Dust Mass Distribution Histogram
    if dust_analysis:
        ax7 = fig.add_subplot(gs[2, 0])
        dust_masses = dust_analysis['dust_masses']
        ax7.hist(np.log10(dust_masses), bins=50, color='brown', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('log$_{10}$(Dust Mass / M$_\\odot$)', fontsize=10)
        ax7.set_ylabel('Count', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('Dust Mass Distribution', fontweight='bold')
    
    # 8. Radial Dust Distribution
    if dust_analysis:
        ax8 = fig.add_subplot(gs[2, 1])
        dust_r = dust_analysis['dust_r']
        ax8.hist(dust_r, bins=50, range=(0, r_max_plot), color='brown', 
                alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Radius (kpc)', fontsize=10)
        ax8.set_ylabel('Dust Particle Count', fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_title('Dust Radial Distribution', fontweight='bold')
    
    # 9. Summary Statistics (WARM ISM FOCUS)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate totals
    total_dust = profile['dust_mass'].sum()
    total_gas_all = profile['gas_mass_all'].sum()
    total_gas_warm = profile['gas_mass_warm'].sum()
    total_metals_all = profile['metal_mass_all'].sum()
    total_metals_warm = profile['metal_mass_warm'].sum()
    
    # Inner disk only (r < 30 kpc)
    mask_inner = profile['r_centers'] < 30
    dust_inner = profile['dust_mass'][mask_inner].sum()
    metals_warm_inner = profile['metal_mass_warm'][mask_inner].sum()
    
    avg_dm_all = total_dust / total_metals_all if total_metals_all > 0 else 0
    avg_dm_warm = total_dust / total_metals_warm if total_metals_warm > 0 else 0
    avg_dm_inner = dust_inner / metals_warm_inner if metals_warm_inner > 0 else 0
    
    avg_dg_all = total_dust / total_gas_all if total_gas_all > 0 else 0
    avg_dg_warm = total_dust / total_gas_warm if total_gas_warm > 0 else 0
    
    warm_frac = 100 * total_gas_warm / total_gas_all if total_gas_all > 0 else 0
    
    stats_text = f"""Summary Statistics:

Total Dust:         {total_dust:.2e} M☉
Total Gas (all):    {total_gas_all:.2e} M☉
Total Gas (warm):   {total_gas_warm:.2e} M☉ ({warm_frac:.1f}%)
Total Metals (all): {total_metals_all:.2e} M☉
Total Metals (warm):{total_metals_warm:.2e} M☉

D/M (all gas):      {avg_dm_all:.3f}
D/M (warm ISM):     {avg_dm_warm:.3f} ← PHYSICAL!
D/M (r<30 kpc):     {avg_dm_inner:.3f} ← DISK ONLY!

D/G (all gas):      {avg_dg_all*100:.3f}%
D/G (warm ISM):     {avg_dg_warm*100:.3f}%

Dust particles:     {profile['dust_count'].sum():,}
Gas (all):          {profile['gas_count_all'].sum():,}
Gas (warm ISM):     {profile['gas_count_warm'].sum():,}
"""
    
    if dust_analysis and dust_analysis['far_dust_count'] > 0:
        stats_text += f"\n⚠ {dust_analysis['far_dust_count']} dust\n  beyond 100 kpc"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Radial dust analysis with temperature filtering')
    parser.add_argument('--catalog', required=True, 
                       help='Path to Subfind catalog (fof_subhalo_tab_*.hdf5)')
    parser.add_argument('--snapshot', required=True, 
                       help='Base path to snapshot (e.g., snapdir_049/snapshot_049)')
    parser.add_argument('--out', default='radial_dust_analysis.png', 
                       help='Output filename')
    parser.add_argument('--rmax', type=float, default=200, 
                       help='Maximum radius for analysis (kpc). Default: 200')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RADIAL DUST ANALYSIS - THERMAL SPUTTERING THRESHOLD")
    print("="*60)
    
    # Get halo center from catalog
    print("\nReading target halo from catalog...")
    halo_pos, halo_mass = get_target_halo_center(args.catalog)
    
    # Get redshift
    with h5py.File(f'{args.snapshot}.0.hdf5', 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])
    print(f"Redshift: z = {redshift:.3f}")
    
    # Extract gas and dust using SAME spatial sphere
    print(f"\n{'='*60}")
    print(f"SPATIAL EXTRACTION (r < {args.rmax} kpc)")
    print(f"{'='*60}")
    
    halo = {}
    
    print("\n1. Extracting GAS particles (with temperature):")
    halo['gas'] = extract_particles_in_sphere(args.snapshot, halo_pos, args.rmax, particle_type=0)
    
    print("\n2. Extracting DUST particles:")
    halo['dust'] = extract_particles_in_sphere(args.snapshot, halo_pos, args.rmax, particle_type=6)
    
    if halo['gas'] is None:
        print("ERROR: No gas particles found!")
        return
    
    if halo['dust'] is None:
        print("WARNING: No dust particles found!")
    
    # Analyze dust distribution
    print("\n" + "="*60)
    dust_analysis = None
    if halo['dust'] is not None:
        dust_analysis = analyze_dust_distribution(halo, halo_pos)
    
    # Compute radial profiles (with temperature filtering)
    print(f"\nComputing radial profiles with temperature filtering...")
    profile = compute_radial_profile(halo, halo_pos, r_max=args.rmax)
    
    # Create plots
    print("\nCreating plots...")
    plot_radial_profiles(profile, halo_mass, redshift, 
                        dust_analysis=dust_analysis, 
                        output_file=args.out)
    
    # Save data
    npz_file = args.out.replace('.png', '.npz')
    np.savez(npz_file,
             r_centers=profile['r_centers'],
             r_bins=profile['r_bins'],
             DM_ratio_all=profile['DM_ratio_all'],
             DM_ratio_warm=profile['DM_ratio_warm'],
             DG_ratio_all=profile['DG_ratio_all'],
             DG_ratio_warm=profile['DG_ratio_warm'],
             dust_mass=profile['dust_mass'],
             metal_mass_all=profile['metal_mass_all'],
             metal_mass_warm=profile['metal_mass_warm'],
             gas_mass_all=profile['gas_mass_all'],
             gas_mass_warm=profile['gas_mass_warm'],
             dust_count=profile['dust_count'],
             gas_count_all=profile['gas_count_all'],
             gas_count_warm=profile['gas_count_warm'])
    
    print(f"Saved data: {npz_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()