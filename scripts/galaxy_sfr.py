#!/usr/bin/env python3
"""
Analyze galaxy star formation rate from Gadget-4 zoom simulation.
For ZOOM simulations, calculates SFR of the main galaxy/halo.

Usage:
    python plot_galaxy_sfr.py /path/to/output
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob
import sys
import re
from pathlib import Path
from collections import defaultdict

# Constants
SOLAR_MASS = 1.989e33  # g
YR_IN_SEC = 3.156e7    # s

def redshift_to_age(z, h=0.7, Om=0.3):
    """Convert redshift to age of universe (Gyr)."""
    tH = 13.8  # Hubble time in Gyr
    a = 1 / (1 + z)
    lookback = tH * (1 - a)
    age = tH - lookback
    return age

def scale_factor_to_time(a, h=0.7, Om=0.3):
    """Convert scale factor to cosmic time (Gyr)."""
    if a <= 0:
        return 0.0
    z = (1.0 / a) - 1.0
    return redshift_to_age(z, h, Om)

def stellar_mass_to_halo_mass(mstar):
    """
    Approximate halo mass from stellar mass using abundance matching.
    Based on Behroozi et al. (2013) at z~0.
    Returns halo mass in M☉.
    """
    # Very rough approximation: M_halo ~ 50 * M_star for MW-mass galaxies
    return mstar * 50

def main_sequence_sfr(mstar, z):
    """
    Star-forming main sequence from Speagle et al. (2014).
    Returns expected SFR in M☉/yr for a star-forming galaxy.
    
    Parameters:
        mstar: stellar mass in M☉
        z: redshift
    """
    # Speagle et al. (2014) parameterization
    log_mstar = np.log10(mstar)
    t = redshift_to_age(z)  # Age in Gyr
    
    # Main sequence: log(SFR) = (0.84 - 0.026*t) * log(M*) - (6.51 - 0.11*t)
    log_sfr = (0.84 - 0.026 * t) * log_mstar - (6.51 - 0.11 * t)
    
    return 10**log_sfr

def ssfr_evolution(z):
    """
    Specific SFR evolution from Speagle et al. (2014).
    Returns sSFR in Gyr^-1.
    """
    t = redshift_to_age(z)
    # sSFR ∝ (1+z)^2.2 or equivalently sSFR ∝ t^-1.1
    # Normalized to ~2 Gyr^-1 at z=2
    ssfr_gyr = 2.0 * ((1 + z) / 3.0)**2.2
    return ssfr_gyr

def find_snapshot_files(base_path):
    """Find all snapshot files organized by snapshot number."""
    base_path = Path(base_path)
    snapshots = defaultdict(list)
    
    snapdirs = sorted(base_path.glob('snapdir_*'))
    
    if snapdirs:
        for snapdir in snapdirs:
            snap_files = list(snapdir.glob('snap_*.hdf5')) + list(snapdir.glob('snapshot_*.hdf5'))
            for snap_file in snap_files:
                match = re.search(r'snap(?:shot)?_(\d+)', snap_file.name)
                if match:
                    snap_num = int(match.group(1))
                    snapshots[snap_num].append(str(snap_file))
    else:
        snap_files = list(base_path.glob('snap_*.hdf5')) + list(base_path.glob('snapshot_*.hdf5'))
        for snap_file in snap_files:
            match = re.search(r'snap(?:shot)?_(\d+)', snap_file.name)
            if match:
                snap_num = int(match.group(1))
                snapshots[snap_num].append(str(snap_file))
    
    for snap_num in snapshots:
        snapshots[snap_num] = sorted(snapshots[snap_num])
    
    return dict(sorted(snapshots.items()))

def read_snapshot(filenames):
    """Read Gadget-4 HDF5 snapshot from one or more files."""
    if isinstance(filenames, str):
        filenames = [filenames]
    
    with h5py.File(filenames[0], 'r') as f:
        header = f['Header'].attrs
        parameters = f['Parameters'].attrs
        redshift = header['Redshift']
        time = header['Time']
        hubble_param = parameters['HubbleParam']
        unit_mass = parameters['UnitMass_in_g']
        unit_length = parameters['UnitLength_in_cm']
        
        if 'UnitTime_in_s' in parameters:
            unit_time = parameters['UnitTime_in_s']
        elif 'UnitLength_in_cm' in parameters and 'UnitVelocity_in_cm_per_s' in parameters:
            unit_time = parameters['UnitLength_in_cm'] / parameters['UnitVelocity_in_cm_per_s']
        else:
            unit_time = unit_length / 1e5
    
    all_masses = []
    all_formation_times = []
    
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            if 'PartType4' in f:
                stars = f['PartType4']
                masses = stars['Masses'][:]
                
                if 'StellarFormationTime' in stars:
                    formation_times = stars['StellarFormationTime'][:]
                elif 'GFM_StellarFormationTime' in stars:
                    formation_times = stars['GFM_StellarFormationTime'][:]
                else:
                    formation_times = np.zeros_like(masses)
                
                all_masses.append(masses)
                all_formation_times.append(formation_times)
    
    if all_masses:
        masses = np.concatenate(all_masses)
        formation_times = np.concatenate(all_formation_times)
    else:
        masses = np.array([])
        formation_times = np.array([])
    
    return {
        'masses': masses,
        'formation_times': formation_times,
        'redshift': redshift,
        'time': time,
        'unit_mass': unit_mass,
        'unit_length': unit_length,
        'unit_time': unit_time,
        'hubble_param': hubble_param
    }

def calculate_galaxy_sfr(snapshots, time_window=100e6):
    """
    Calculate galaxy SFR from snapshots.
    
    Parameters:
        snapshots: list of snapshot data dictionaries
        time_window: time window for SFR calculation in years
    
    Returns:
        redshifts, ages, sfrs, stellar_masses (all in proper units)
    """
    redshifts = []
    ages = []
    sfrs = []
    stellar_masses = []
    
    time_window_gyr = time_window / 1e9
    
    for snap in snapshots:
        if len(snap['masses']) == 0:
            continue
        
        z = snap['redshift']
        current_time = snap['time']
        h = snap['hubble_param']
        
        # Total stellar mass
        total_mass_code = np.sum(snap['masses'])
        total_mass_msun = total_mass_code * snap['unit_mass'] / SOLAR_MASS / h
        
        # Determine if formation times are scale factors
        if snap['formation_times'].max() < 10:  # Scale factor format
            current_time_gyr = scale_factor_to_time(current_time, h=h)
            formation_times_gyr = np.array([scale_factor_to_time(a, h=h) 
                                           for a in snap['formation_times']])
            
            age_when_formed = formation_times_gyr
            current_age = current_time_gyr
            recent_stars = (current_age - age_when_formed) < time_window_gyr
        else:
            time_window_code = time_window * YR_IN_SEC / snap['unit_time']
            recent_stars = (current_time - snap['formation_times']) < time_window_code
        
        if np.sum(recent_stars) > 0:
            mass_formed_code = np.sum(snap['masses'][recent_stars])
            mass_formed_msun = mass_formed_code * snap['unit_mass'] / SOLAR_MASS / h
            sfr = mass_formed_msun / time_window  # M☉/yr
        else:
            sfr = 0.0
        
        redshifts.append(z)
        ages.append(redshift_to_age(z, h=h))
        sfrs.append(sfr)
        stellar_masses.append(total_mass_msun)
    
    return np.array(redshifts), np.array(ages), np.array(sfrs), np.array(stellar_masses)

def plot_galaxy_sfr(sim_z, sim_age, sim_sfr, sim_mstar, output_file='galaxy_sfr.png'):
    """Create plot of galaxy SFR evolution with observational comparisons."""
    
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Panel 1: SFR vs stellar mass (Main Sequence)
    ax1 = plt.subplot(gs[0])
    
    # Plot observational main sequence at different redshifts with scatter
    mstar_range = np.logspace(8, 12, 100)
    colors_ms = plt.cm.viridis(np.linspace(0, 1, 5))
    z_values = [0, 1, 2, 3, 4]
    
    # Typical scatter around main sequence is ~0.3 dex
    ms_scatter = 0.3  # dex
    
    for i, z_ms in enumerate(z_values):
        sfr_ms = np.array([main_sequence_sfr(m, z_ms) for m in mstar_range])
        
        # Plot scatter band (±0.3 dex is typical for star-forming galaxies)
        ax1.fill_between(mstar_range, sfr_ms * 10**(-ms_scatter), sfr_ms * 10**(ms_scatter),
                        alpha=0.15, color=colors_ms[i], zorder=0)
        
        # Plot median line
        ax1.plot(mstar_range, sfr_ms, '--', color=colors_ms[i], alpha=0.7, 
                linewidth=2, label=f'MS z={z_ms}', zorder=1)
    
    # Color code simulation points by redshift
    scatter = ax1.scatter(sim_mstar, sim_sfr, c=sim_z, s=60, cmap='plasma', 
                         edgecolors='black', linewidth=0.5, zorder=5)
    
    # Connect points chronologically
    ax1.plot(sim_mstar, sim_sfr, '-', color='gray', alpha=0.3, linewidth=1, zorder=4)
    
    ax1.set_xlabel('Stellar Mass (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SFR (M$_\\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e8, max(sim_mstar.max() * 2, 1e11))
    ax1.set_ylim(1e-2, max(sim_sfr.max() * 2, 100))
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)
    ax1.set_title('Star-Forming Main Sequence (shaded = ±0.3 dex scatter)', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar for redshift
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Redshift', fontsize=10)
    
    # Panel 2: sSFR vs redshift
    ax2 = plt.subplot(gs[1])
    
    # Observational sSFR evolution with scatter
    z_obs = np.linspace(0, 6, 100)
    ssfr_obs = np.array([ssfr_evolution(z) for z in z_obs])
    age_obs = np.array([redshift_to_age(z) for z in z_obs])
    
    # Typical factor of 2-3 scatter in sSFR observations
    ax2.fill_between(age_obs, ssfr_obs * 0.3, ssfr_obs * 3.0, 
                     alpha=0.15, color='blue', label='Observed scatter (×0.3-3)', zorder=0)
    ax2.fill_between(age_obs, ssfr_obs * 0.5, ssfr_obs * 2.0, 
                     alpha=0.2, color='gray', label='Typical range (×0.5-2)', zorder=1)
    ax2.plot(age_obs, ssfr_obs, 'k-', linewidth=2.5, label='Median (Speagle+2014)', zorder=2)
    
    # Simulation sSFR
    ssfr_sim = sim_sfr / sim_mstar * 1e9  # Convert to Gyr^-1
    ax2.plot(sim_age, ssfr_sim, 'o-', linewidth=2, markersize=7, 
            color='#e74c3c', label='Simulation', zorder=2)
    
    ax2.set_ylabel('sSFR (Gyr$^{-1}$)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 14)
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add redshift axis on top
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    z_ticks = [10, 8, 6, 4, 2, 0]
    age_ticks = [redshift_to_age(z) for z in z_ticks]
    ax2_top.set_xticks(age_ticks)
    ax2_top.set_xticklabels([str(z) for z in z_ticks])
    ax2_top.set_xlabel('Redshift', fontsize=12)
    ax2.tick_params(labelbottom=False)
    
    # Panel 3: Stellar mass growth
    ax3 = plt.subplot(gs[2])
    ax3.plot(sim_age, sim_mstar, 'o-', linewidth=2.5, markersize=7, color='#3498db', 
            label='Simulation', zorder=3)
    ax3.set_xlabel('Age of Universe (Gyr)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Stellar Mass (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 14)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add shaded regions for different galaxy types
    ax3.axhspan(3e10, 8e10, alpha=0.1, color='green', label='MW-mass range', zorder=0)
    ax3.axhspan(8e10, 5e11, alpha=0.1, color='orange', label='Massive galaxies', zorder=0)
    ax3.axhspan(1e9, 3e10, alpha=0.1, color='purple', label='Small galaxies', zorder=0)
    
    # Add reference lines
    ax3.axhline(5e10, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.text(0.5, 5e10 * 1.15, 'Milky Way', fontsize=9, color='darkgreen', 
            verticalalignment='bottom')
    
    ax3.legend(loc='lower right', fontsize=9)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Save data
    data_file = output_file.replace('.png', '.txt')
    with open(data_file, 'w') as f:
        f.write("# Galaxy Star Formation History\n")
        f.write("# Columns: Redshift, Age(Gyr), SFR(Msun/yr), Mstar(Msun), sSFR(Gyr^-1), MS_SFR(Msun/yr), sSFR_obs(Gyr^-1)\n")
        for i in range(len(sim_z)):
            ssfr_val = sim_sfr[i] / sim_mstar[i] * 1e9 if sim_mstar[i] > 0 else 0
            ms_sfr = main_sequence_sfr(sim_mstar[i], sim_z[i])
            ssfr_obs_val = ssfr_evolution(sim_z[i])
            f.write(f"{sim_z[i]:.4f}  {sim_age[i]:.4f}  {sim_sfr[i]:.6e}  "
                   f"{sim_mstar[i]:.6e}  {ssfr_val:.6e}  {ms_sfr:.6e}  {ssfr_obs_val:.6e}\n")
    print(f"Data saved to {data_file}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("GALAXY PROPERTIES AT z=0 (final snapshot):")
    print(f"{'='*70}")
    final_idx = np.argmin(np.abs(sim_z))
    final_mstar = sim_mstar[final_idx]
    final_sfr = sim_sfr[final_idx]
    final_ssfr = final_sfr / final_mstar * 1e9
    
    print(f"Stellar mass: {final_mstar:.2e} M☉")
    print(f"SFR: {final_sfr:.2f} M☉/yr")
    print(f"sSFR: {final_ssfr:.2f} Gyr⁻¹")
    
    # Compare to main sequence
    ms_sfr_final = main_sequence_sfr(final_mstar, sim_z[final_idx])
    offset = np.log10(final_sfr / ms_sfr_final)
    print(f"\nMain Sequence SFR (expected): {ms_sfr_final:.2f} M☉/yr")
    print(f"Offset from MS: {offset:.2f} dex")
    
    if offset > 0.3:
        print("  → Galaxy is a STARBURST (above main sequence)")
    elif offset < -0.3:
        print("  → Galaxy is QUENCHED (below main sequence)")
    else:
        print("  → Galaxy is ON the main sequence ✓")
    
    # Estimate halo mass
    mhalo = stellar_mass_to_halo_mass(final_mstar)
    print(f"\nEstimated halo mass: {mhalo:.2e} M☉")
    
    if final_mstar > 1e11:
        print(f"This is a MASSIVE galaxy (M* > 10^11 M☉)")
    elif final_mstar > 3e10:
        print(f"This is a MILKY WAY-MASS galaxy")
    elif final_mstar > 1e9:
        print(f"This is a small galaxy / LMC-mass")
    else:
        print(f"This is a DWARF galaxy")
    
    print(f"{'='*70}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_galaxy_sfr.py <output_directory>")
        print("Example: python plot_galaxy_sfr.py /path/to/zoom/simulation/output")
        sys.exit(1)
    
    base_path = sys.argv[1]
    
    print(f"Scanning for snapshots in: {base_path}")
    snapshot_dict = find_snapshot_files(base_path)
    
    if len(snapshot_dict) == 0:
        print(f"No snapshots found in {base_path}")
        sys.exit(1)
    
    print(f"\nFound {len(snapshot_dict)} snapshots")
    
    print("\nReading snapshots...")
    snapshots = []
    for snap_num in sorted(snapshot_dict.keys()):
        files = snapshot_dict[snap_num]
        try:
            snap = read_snapshot(files)
            if len(snap['masses']) > 0:  # Only include snapshots with stars
                snapshots.append(snap)
                if snap_num % 10 == 0:  # Print every 10th snapshot
                    print(f"  Snapshot {snap_num:03d}: z={snap['redshift']:.3f}, "
                          f"{len(snap['masses'])} stars")
        except Exception as e:
            print(f"  Warning: Could not read snapshot {snap_num}: {e}")
    
    if len(snapshots) == 0:
        print("\nNo snapshots with star particles found!")
        sys.exit(1)
    
    print(f"\nCalculating galaxy SFR...")
    z, age, sfr, mstar = calculate_galaxy_sfr(snapshots, time_window=100e6)
    
    print(f"\nProcessed {len(z)} snapshots")
    print(f"Redshift range: z = {z.min():.2f} to {z.max():.2f}")
    print(f"SFR range: {sfr.min():.2e} to {sfr.max():.2e} M☉/yr")
    print(f"Stellar mass range: {mstar.min():.2e} to {mstar.max():.2e} M☉")
    
    print("\nCreating plots...")
    plot_galaxy_sfr(z, age, sfr, mstar)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
