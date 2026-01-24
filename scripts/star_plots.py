#!/usr/bin/env python3
"""
Three-panel plot: Stellar-to-Halo Mass Ratio, Lilly-Madau Diagram, and Absolute SFR
- Panel 1: M*/M_halo evolution with Behroozi+2013 comparison
- Panel 2: Cosmic SFR density (entire high-res region) vs Madau & Dickinson 2014
- Panel 3: Absolute SFR of main halo vs observations of MW-mass galaxies

Usage:
  python plot_halo_evolution.py <output_dir> <halo_particle_file> [options]
  
Example:
  python plot_halo_evolution.py ../5_output_zoom_512_halo569_50Mpc_dust/ halo569_2Rvir_particles.txt
"""

import numpy as np
import h5py
import glob
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

# Observational data from Madau & Dickinson 2014
OBS_Z = np.array([0.01, 0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.75, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5])
OBS_SFR = np.array([0.015, 0.017, 0.03, 0.055, 0.08, 0.10, 0.11, 0.12, 0.10, 0.09, 0.07, 0.045, 0.03, 0.01, 0.003, 0.001])
OBS_ERR = OBS_SFR * 0.2  # Assume 20% errors


def load_halo_particle_ids(halo_file):
    """
    Load gas and star particle IDs for a specific halo from text file.
    
    Format of text file:
    # Comment lines
    TYPE PARTICLE_ID
    TYPE PARTICLE_ID
    ...
    
    Where TYPE: 0=Gas, 4=Stars
    """
    print(f"Loading halo particle IDs from: {halo_file}")
    
    data = np.loadtxt(halo_file, dtype={'names': ('type', 'id'), 'formats': ('i4', 'i8')})
    
    gas_ids = data['id'][data['type'] == 0]
    star_ids = data['id'][data['type'] == 4]
    
    print(f"  Found {len(gas_ids)} gas particle IDs")
    print(f"  Found {len(star_ids)} star particle IDs")
    
    return {'gas': gas_ids, 'stars': star_ids}


def load_snapshot_header(snapshot_file):
    """Load header information from snapshot"""
    with h5py.File(snapshot_file, 'r') as f:
        header = f['Header'].attrs
        return {
            'Time': header['Time'],
            'Redshift': header['Redshift'],
            'BoxSize': header['BoxSize'],
            'HubbleParam': header.get('HubbleParam', 0.6774),
            'Omega0': header.get('Omega0', 0.3089),
            'OmegaLambda': header.get('OmegaLambda', 0.6911)
        }


def compute_halo_sfr(snapshot_files, halo_pids):
    """
    Compute total star formation rate for halo particles only.
    
    Returns:
        redshift, total_sfr (M_sun/yr)
    """
    # Read all files to get complete particle data
    all_ids = []
    all_sfr = []
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            if 'PartType0' in f:
                ids = f['PartType0']['ParticleIDs'][:]
                sfr = f['PartType0']['StarFormationRate'][:]
                
                all_ids.append(ids)
                all_sfr.append(sfr)
    
    # Concatenate all data
    all_ids = np.concatenate(all_ids)
    all_sfr = np.concatenate(all_sfr)
    
    # Filter to only halo particles
    halo_mask = np.isin(all_ids, halo_pids['gas'])
    halo_sfr = all_sfr[halo_mask]
    
    total_sfr = np.sum(halo_sfr[halo_sfr > 0])  # Only count actively star-forming gas
    
    # Get header info from first file
    header = load_snapshot_header(snapshot_files[0])
    
    return header['Redshift'], total_sfr


def compute_halo_sfr_density(snapshot_files, halo_pids):
    """
    Compute SFR density for the halo region.
    
    Note: Volume is approximate - uses R_vir estimate
    For more accurate results, you could compute volume from particle distribution
    """
    z, total_sfr = compute_halo_sfr(snapshot_files, halo_pids)
    
    # Estimate halo volume (very rough - assumes ~200 kpc radius for MW-mass halo)
    # Better approach: calculate from actual particle distribution
    R_vir_kpc = 200.0  # Approximate virial radius in kpc
    volume_mpc3 = (4.0/3.0) * np.pi * (R_vir_kpc / 1000.0)**3  # Mpc^3
    
    sfr_density = total_sfr / volume_mpc3 if volume_mpc3 > 0 else 0.0
    
    return z, sfr_density


def compute_cosmic_sfr_density(snapshot_files, zoom_radius_mpc=3.5):
    """
    Compute cosmic average SFR density in the entire high-res zoom region.
    Uses ALL high-res particles, not just one halo.
    
    Parameters:
    -----------
    snapshot_files : list
        List of snapshot files for this redshift
    zoom_radius_mpc : float
        Radius of high-res region in comoving Mpc (default: 3.5 for 7 Mpc diameter)
    
    Returns:
    --------
    redshift, sfr_density (M_sun/yr/Mpc^3)
    """
    # Get header info
    with h5py.File(snapshot_files[0], 'r') as f:
        header = f['Header'].attrs
        redshift = header['Redshift']
        h = header.get('HubbleParam', 0.6774)
    
    # Calculate high-res region volume
    volume_mpc3 = (4.0/3.0) * np.pi * zoom_radius_mpc**3
    
    # Sum SFR from ALL gas particles in the snapshot
    # In a zoom simulation, snapdir_* should primarily contain high-res particles
    total_sfr = 0.0
    total_gas_mass = 0.0
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            if 'PartType0' in f:
                masses = f['PartType0']['Masses'][:]
                sfr = f['PartType0']['StarFormationRate'][:]
                
                # Identify high-res by finding the mode (most common) mass
                # High-res particles should dominate and have uniform mass
                if len(masses) > 0:
                    # Find the most common mass (high-res particles)
                    # Use histogram to identify the peak
                    hist, bin_edges = np.histogram(np.log10(masses + 1e-20), bins=50)
                    peak_bin = np.argmax(hist)
                    peak_mass = 10.0**((bin_edges[peak_bin] + bin_edges[peak_bin+1]) / 2.0)
                    
                    # High-res particles are within factor of 2 of the peak mass
                    highres_mask = (masses > peak_mass * 0.5) & (masses < peak_mass * 2.0)
                    
                    # Sum SFR from high-res particles only
                    total_sfr += np.sum(sfr[highres_mask & (sfr > 0)])
                    total_gas_mass += np.sum(masses[highres_mask])
    
    sfr_density = total_sfr / volume_mpc3 if volume_mpc3 > 0 else 0.0
    
    return redshift, sfr_density


def behroozi_2013_shmr(M_halo, z):
    """
    Stellar-to-halo mass ratio from Behroozi et al. 2013
    
    Parameters:
    -----------
    M_halo : float or array
        Halo mass in M_sun
    z : float or array
        Redshift
        
    Returns:
    --------
    M_star : stellar mass in M_sun
    ratio : M_star / M_halo
    """
    # Convert to log10 mass
    log_M_halo = np.log10(M_halo)
    
    # Behroozi+2013 parametrization
    a = 1.0 / (1.0 + z)  # Scale factor
    
    # Characteristic mass M_1 (log10)
    log_M1 = 11.514 + (-1.793 * (a - 1.0)) + (-0.251 * z)
    
    # Normalization
    epsilon = -1.777 + (-0.006 * (a - 1.0)) + (-0.000 * z) + (-0.119 * (a - 1.0))
    
    # Low-mass slope
    alpha = -1.412 + (0.731 * (a - 1.0))
    
    # High-mass slope  
    delta = 3.508 + (2.608 * (a - 1.0)) + (-0.043 * z)
    
    # Transition width
    gamma = 0.316 + (1.319 * (a - 1.0)) + (0.279 * z)
    
    # Calculate f(x) where x = log_M_halo - log_M1
    x = log_M_halo - log_M1
    
    f_x = -np.log10(10.0**(alpha * x) + 1.0) + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10.0**(-x)))
    
    log_M_star = epsilon + log_M1 + f_x - np.log10(2.0)
    M_star = 10.0**log_M_star
    
    ratio = M_star / M_halo
    
    return M_star, ratio


def process_snapshots_for_sfr(snapshot_dir, halo_pids):
    """
    Process all snapshots in directory to compute both:
    1. Cosmic SFR density (entire high-res region)
    2. Halo absolute SFR (halo 569 only)
    
    Returns:
    --------
    redshifts, cosmic_sfr_densities, halo_absolute_sfr
    """
    # Find all snapshot directories
    snap_dirs = sorted(glob.glob(os.path.join(snapshot_dir, 'snapdir_*')))
    
    if not snap_dirs:
        print(f"No snapshot directories found matching: {snapshot_dir}/snapdir_*")
        return np.array([]), np.array([]), np.array([])
    
    print(f"\nProcessing {len(snap_dirs)} snapshots for SFR calculation...")
    
    redshifts = []
    cosmic_sfr_densities = []
    halo_absolute_sfrs = []
    
    for snap_dir in snap_dirs:
        # Get all files for this snapshot
        snap_files = sorted(glob.glob(os.path.join(snap_dir, 'snapshot_*.hdf5')))
        
        if not snap_files:
            continue
        
        # Cosmic SFR density (entire zoom region)
        z_cosmic, sfr_density_cosmic = compute_cosmic_sfr_density(snap_files)
        
        # Halo absolute SFR
        z_halo, sfr_halo = compute_halo_sfr(snap_files, halo_pids)
        
        redshifts.append(z_cosmic)
        cosmic_sfr_densities.append(sfr_density_cosmic)
        halo_absolute_sfrs.append(sfr_halo)
        
        print(f"  z={z_cosmic:.2f}: Cosmic SFR density = {sfr_density_cosmic:.3e} M_sun/yr/Mpc^3, "
              f"Halo SFR = {sfr_halo:.2f} M_sun/yr")
    
    return np.array(redshifts), np.array(cosmic_sfr_densities), np.array(halo_absolute_sfrs)


def load_stellar_evolution(output_dir):
    """
    Load stellar evolution data from stellar_evolution.txt
    
    Expected format:
    # scale_factor  stellar_mass  halo_mass  ratio
    """
    stellar_file = os.path.join(output_dir, 'stellar_evolution.txt')
    
    if not os.path.exists(stellar_file):
        print(f"ERROR: Could not find {stellar_file}")
        print("Make sure your simulation is tracking stellar evolution!")
        return None
    
    print(f"\nLoading stellar evolution from: {stellar_file}")
    
    data = np.loadtxt(stellar_file)
    
    # Convert scale factor to redshift
    scale_factor = data[:, 0]
    redshift = 1.0 / scale_factor - 1.0
    
    stellar_mass = data[:, 1]  # M_sun
    halo_mass = data[:, 2]     # M_sun
    ratio = data[:, 3]
    
    print(f"  Loaded {len(redshift)} snapshots")
    print(f"  Redshift range: {redshift.min():.2f} to {redshift.max():.2f}")
    print(f"  Final stellar mass: {stellar_mass[-1]:.3e} M_sun")
    print(f"  Final halo mass: {halo_mass[-1]:.3e} M_sun")
    print(f"  Final ratio: {ratio[-1]:.3e}")
    
    return {
        'redshift': redshift,
        'stellar_mass': stellar_mass,
        'halo_mass': halo_mass,
        'ratio': ratio
    }


def plot_combined(stellar_data, sfr_redshifts, cosmic_sfr_densities, halo_absolute_sfrs, 
                  output_file='halo_evolution.png', run_name=''):
    """
    Create three-panel plot:
    - Top: Stellar-to-halo mass ratio vs redshift (with age of universe on top axis)
    - Middle: Absolute SFR of main halo vs MW-mass galaxy observations
    - Bottom: Cosmic SFR density (Lilly-Madau diagram)
    """
    fig = plt.figure(figsize=(10, 14))
    gs = GridSpec(3, 1, hspace=0.25, top=0.92, figure=fig)  # Reduced hspace from 0.35 to 0.25
    
    # Add master title with run name (moved down slightly)
    if run_name:
        fig.suptitle(f'{run_name}', fontsize=14, fontweight='bold', color='green', y=0.985)
    
    # ===== TOP PANEL: Stellar-to-Halo Mass Ratio =====
    ax1 = fig.add_subplot(gs[0])
    
    # Plot simulation data
    ax1.semilogy(stellar_data['redshift'], stellar_data['ratio'], 
                 'b-', linewidth=2, label='Simulation (Halo 569)')
    
    # Plot Behroozi+2013 prediction using actual halo mass evolution
    # Use the halo masses from the simulation data
    _, behroozi_ratio = behroozi_2013_shmr(stellar_data['halo_mass'], stellar_data['redshift'])
    
    # Get final halo mass for legend
    final_halo_mass = stellar_data['halo_mass'][-1]
    ax1.semilogy(stellar_data['redshift'], behroozi_ratio, 'r--', linewidth=2, alpha=0.7,
                 label=r'Behroozi et al 2013 ($M_h = {:.1f} \times 10^{{12}}$ M$_\odot$)'.format(final_halo_mass/1e12))
    
    # Mark peak efficiency at ~2% (from Behroozi+2013, Moster+2013)
    ax1.axhline(0.02, color='gray', linestyle=':', alpha=0.5, 
                label='Peak efficiency (~2%)')
    
    ax1.set_ylabel(r'$M_\star / M_{\rm halo}$', fontsize=12)
    ax1.set_xlim(stellar_data['redshift'].max(), 0)
    ax1.set_ylim(1e-4, 0.1)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_title('Stellar-to-Halo Mass Ratio', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='x', labelbottom=False)
    
    # Add age of universe on top x-axis
    ax1_top = ax1.twiny()
    
    # Calculate age of universe at different redshifts
    # Using Planck 2018 cosmology: H0=67.4, Ωm=0.315, ΩΛ=0.685
    def age_at_redshift(z, H0=67.4, Om=0.315, OL=0.685):
        """Calculate age of universe at redshift z in Gyr"""
        def integrand(zp):
            return 1.0 / ((1.0 + zp) * np.sqrt(Om * (1.0 + zp)**3 + OL))
        age_Gyr = (1.0 / H0) * quad(integrand, z, np.inf)[0] * 977.8  # Convert to Gyr
        return age_Gyr
    
    # Set up age axis to match redshift axis
    z_ticks = ax1.get_xticks()
    z_ticks = z_ticks[(z_ticks >= 0) & (z_ticks <= stellar_data['redshift'].max())]
    age_ticks = [age_at_redshift(z) for z in z_ticks]
    
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(z_ticks)
    ax1_top.set_xticklabels([f'{age:.1f}' for age in age_ticks])
    ax1_top.set_xlabel('Age of Universe (Gyr)', fontsize=12)
    ax1_top.tick_params(axis='x', labelsize=10)
    
    # ===== MIDDLE PANEL: Absolute SFR of Main Halo =====
    ax2 = fig.add_subplot(gs[1])
    
    # Plot simulation halo SFR
    ax2.semilogy(sfr_redshifts, halo_absolute_sfrs, 
                 'b-o', linewidth=2, markersize=6, label='Simulation (Halo 569)')
    
    # Add observational ranges for MW-mass galaxies
    obs_z_points = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    obs_sfr_typical = np.array([1.5, 3.0, 7.0, 15.0, 25.0, 20.0, 15.0, 10.0, 8.0])  # M_sun/yr
    obs_sfr_lower = obs_sfr_typical * 0.5
    obs_sfr_upper = obs_sfr_typical * 2.0
    
    # Plot observational range as shaded region
    ax2.fill_between(obs_z_points, obs_sfr_lower, obs_sfr_upper, 
                     color='gray', alpha=0.3, label=r'MW-mass range')
    ax2.plot(obs_z_points, obs_sfr_typical, 'k--', linewidth=2, alpha=0.7,
             label=r'Typical MW-mass ($M_\star \sim 5 \times 10^{10}$ M$_\odot$)')
    
    # Add cosmic noon shading (don't add to legend)
    ax2.axvspan(1.5, 2.5, alpha=0.1, color='orange', zorder=0)
    
    ax2.set_ylabel(r'Star Formation Rate (M$_\odot$ yr$^{-1}$)', fontsize=12)
    ax2.set_xlim(stellar_data['redshift'].max(), 0)
    ax2.set_ylim(0.1, 100)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_title('Absolute Star Formation Rate', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='x', labelbottom=False)
    
    # ===== BOTTOM PANEL: Cosmic SFR Density (Lilly-Madau) =====
    ax3 = fig.add_subplot(gs[2])
    
    # Plot simulation cosmic SFR density (entire high-res region)
    ax3.semilogy(sfr_redshifts, cosmic_sfr_densities, 
                 'b-o', linewidth=2, markersize=6, label='Simulation (Entire high-res region)')
    
    # Plot observational data
    ax3.errorbar(OBS_Z, OBS_SFR, yerr=OBS_ERR, fmt='^', color='black',
                 ecolor='gray', elinewidth=2, capsize=4, markersize=7,
                 label='Madau & Dickinson 2014', zorder=5)
    
    # Add cosmic noon shading (don't add to legend)
    ax3.axvspan(1.5, 2.5, alpha=0.1, color='orange', zorder=0)
    
    ax3.set_xlabel('Redshift', fontsize=14)
    ax3.set_ylabel(r'SFR Density (M$_\odot$ yr$^{-1}$ Mpc$^{-3}$)', fontsize=12)
    ax3.set_xlim(stellar_data['redshift'].max(), 0)
    ax3.set_ylim(1e-4, 1e0)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_title('Cosmic Star Formation History', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=10)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    plt.show()


def print_usage():
    """Print usage information"""
    print("\n" + "="*70)
    print("🌌 Halo Evolution Plotter 🌌")
    print("="*70)
    print("\nThis script creates a three-panel plot showing:")
    print("  1. Stellar-to-halo mass ratio vs redshift")
    print("  2. Cosmic SFR density (Lilly-Madau, entire high-res region)")
    print("  3. Absolute SFR of main halo vs MW-mass galaxy observations")
    print("\n" + "="*70)
    print("\nUsage:")
    print("  python plot_halo_evolution.py <output_dir> <halo_particle_file>\n")
    print("Arguments:")
    print("  output_dir         : Directory containing simulation output")
    print("                       (should have snapdir_* subdirectories and stellar_evolution.txt)")
    print("  halo_particle_file : Text file with particle IDs for main halo")
    print("                       (format: TYPE PARTICLE_ID, e.g., halo569_2Rvir_particles.txt)")
    print("\nExample:")
    print("  python plot_halo_evolution.py \\")
    print("      ../5_output_zoom_512_halo569_50Mpc_dust/ \\")
    print("      halo569_2Rvir_particles.txt")
    print("\n" + "="*70 + "\n")


def main():
    """Main function"""
    
    # Check arguments
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    output_dir = sys.argv[1]
    halo_file = sys.argv[2]
    
    # Check if directories/files exist
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)
    
    if not os.path.exists(halo_file):
        print(f"ERROR: Halo particle file not found: {halo_file}")
        sys.exit(1)
    
    # Extract run name from output directory
    run_name = os.path.basename(os.path.normpath(output_dir))
    
    print("\n" + "="*70)
    print("🌌 Halo Evolution Analysis (3-Panel Edition) 🌌")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Run name: {run_name}")
    print(f"Halo particle file: {halo_file}")
    
    # Load halo particle IDs
    halo_pids = load_halo_particle_ids(halo_file)
    
    # Load stellar evolution data
    stellar_data = load_stellar_evolution(output_dir)
    if stellar_data is None:
        sys.exit(1)
    
    # Process snapshots for SFR (both cosmic and halo-specific)
    sfr_redshifts, cosmic_sfr_densities, halo_absolute_sfrs = process_snapshots_for_sfr(output_dir, halo_pids)
    
    if len(sfr_redshifts) == 0:
        print("ERROR: No snapshot data found!")
        sys.exit(1)
    
    # Create combined 3-panel plot with run name
    output_file = os.path.join(output_dir, 'halo_evolution.png')
    plot_combined(stellar_data, sfr_redshifts, cosmic_sfr_densities, halo_absolute_sfrs, 
                  output_file, run_name=run_name)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
