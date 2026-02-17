#!/usr/bin/env python3
"""
plot_temp_sfr_analysis.py

Temperature and star formation rate profiles to diagnose dust physics.

Usage:
    python plot_temp_sfr_analysis.py \
        --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 \
        --snapshot ../snapdir_049/snapshot_049 \
        --out temp_sfr_analysis.png \
        --rmax 200
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from halo_utils import load_target_halo, compute_radial_distance
except ImportError:
    print("ERROR: This script requires halo_utils.py in the same directory")
    exit(1)


# Physical constants
HYDROGEN_MASSFRAC = 0.76
PROTONMASS = 1.67262178e-24  # g
BOLTZMANN = 1.38065e-16  # erg/K
GAMMA = 5.0/3.0  # Adiabatic index
SOLAR_MASS = 1.989e33  # g


def compute_gas_temperature(internal_energy, electron_abundance=None):
    """
    Convert internal energy to temperature.
    
    T = (gamma - 1) * u * mu * m_p / k_B
    
    where mu is the mean molecular weight.
    
    Parameters
    ----------
    internal_energy : array
        Internal energy per unit mass in (km/s)^2
    electron_abundance : array, optional
        Electron abundance for ionization correction
    
    Returns
    -------
    temperature : array
        Temperature in Kelvin
    """
    
    # Mean molecular weight
    if electron_abundance is not None:
        # Full ionization treatment
        # mu = 4 / (1 + 3*X + 4*X*ne) where X is hydrogen fraction
        X = HYDROGEN_MASSFRAC
        mu = 4.0 / (1.0 + 3.0*X + 4.0*X*electron_abundance)
    else:
        # Neutral gas approximation
        mu = 0.59  # Roughly solar composition
    
    # Convert internal energy from (km/s)^2 to cgs (cm/s)^2
    u_cgs = internal_energy * 1e10
    
    # Temperature in Kelvin
    temperature = (GAMMA - 1.0) * u_cgs * mu * PROTONMASS / BOLTZMANN
    
    return temperature


def compute_thermal_profiles(halo, halo_pos, r_max=200, sfr_time_window=0.1):
    """
    Compute temperature and SFR radial profiles.
    
    Parameters
    ----------
    halo : dict
        Halo data from load_target_halo
    halo_pos : array
        Halo center position
    r_max : float
        Maximum radius for profiles (kpc)
    sfr_time_window : float
        Time window for SFR calculation in scale factor units (default: 0.1)
    
    Returns dict with thermal and star formation profiles.
    """
    
    # Set up radial bins (same as dust analysis)
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
        'temperature': np.zeros(n_bins),
        'temp_std': np.zeros(n_bins),
        'temp_median': np.zeros(n_bins),
        'sfr': np.zeros(n_bins),
        'sfr_surface_density': np.zeros(n_bins),
        'gas_mass': np.zeros(n_bins),
        'stellar_mass_young': np.zeros(n_bins),
        'gas_count': np.zeros(n_bins, dtype=int),
        'star_count': np.zeros(n_bins, dtype=int),
    }
    
    # Process gas particles
    if 'gas' in halo and 'InternalEnergy' in halo['gas']:
        print("  Computing gas temperatures...")
        
        # Calculate temperatures
        electron_abundance = halo['gas'].get('ElectronAbundance', None)
        temperatures = compute_gas_temperature(
            halo['gas']['InternalEnergy'],
            electron_abundance
        )
        
        # Get radial distances
        gas_r = compute_radial_distance(halo['gas']['Coordinates'], halo_pos)
        gas_masses = halo['gas']['Masses']
        
        # Bin temperatures (mass-weighted)
        gas_bin_idx = np.digitize(gas_r, r_bins) - 1
        
        for i in range(n_bins):
            mask = (gas_bin_idx == i)
            if mask.sum() > 0:
                shell_temp = temperatures[mask]
                shell_mass = gas_masses[mask]
                
                # Mass-weighted mean temperature
                profile['temperature'][i] = np.average(shell_temp, weights=shell_mass)
                
                # Standard deviation
                profile['temp_std'][i] = np.sqrt(
                    np.average((shell_temp - profile['temperature'][i])**2, weights=shell_mass)
                )
                
                # Median temperature
                profile['temp_median'][i] = np.median(shell_temp)
                
                # Total gas mass in shell
                profile['gas_mass'][i] = np.sum(shell_mass)
                profile['gas_count'][i] = int(mask.sum())
    
    # Process gas particles for SFR (use StarFormationRate field if available)
    if 'gas' in halo and 'StarFormationRate' in halo['gas']:
        print("  Computing star formation rates from gas SFR field...")
        
        # Get gas SFR data (this is instantaneous SFR in code units)
        gas_sfr = halo['gas']['StarFormationRate']
        gas_pos = halo['gas']['Coordinates']
        
        # Get radial distances
        gas_r = compute_radial_distance(gas_pos, halo_pos)
        
        # Bin SFR
        gas_bin_idx = np.digitize(gas_r, r_bins) - 1
        
        # Count star-forming gas particles
        sf_mask = gas_sfr > 0
        n_sf_gas = sf_mask.sum()
        print(f"    Found {n_sf_gas} star-forming gas particles")
        
        if n_sf_gas == 0:
            print(f"    WARNING: No star-forming gas detected!")
            print(f"    This could mean:")
            print(f"      - Star formation is suppressed by feedback")
            print(f"      - Gas is too hot to form stars")
            print(f"      - Halo has quenched")
        
        for i in range(n_bins):
            mask = (gas_bin_idx == i) & sf_mask
            if mask.sum() > 0:
                # Sum SFR in this shell
                # SFR is in code units (1e10 Msun / 0.978 Gyr for Gadget default units)
                # Convert to M_sun/yr
                sfr_code = np.sum(gas_sfr[mask])
                
                # Gadget SFR units: 1e10 Msun / (0.978 Gyr) = 1e10 Msun / (9.78e8 yr)
                # So multiply by 1e10 / 9.78e8 to get Msun/yr
                profile['sfr'][i] = sfr_code * (1e10 / 9.78e8)
                
                # Track number of SF gas particles
                profile['star_count'][i] = int(mask.sum())
        
        # Compute SFR surface density (M_sun/yr/kpc^2)
        for i in range(n_bins):
            if profile['sfr'][i] > 0:
                # Area of shell
                area = np.pi * (r_bins[i+1]**2 - r_bins[i]**2)
                profile['sfr_surface_density'][i] = profile['sfr'][i] / area
    
    # Fallback: try to compute from stellar particles if gas SFR not available
    elif 'stars' in halo and 'GFM_StellarFormationTime' in halo['stars']:
        print("  Computing star formation rates from stellar formation times...")
        print("    (Note: gas StarFormationRate field not found, using stellar particles)")
        
        # Get stellar data
        star_pos = halo['stars']['Coordinates']
        star_mass = halo['stars']['Masses']
        star_formation_time = halo['stars']['GFM_StellarFormationTime']
        
        # Get current scale factor from header
        current_time = halo.get('Time', 1.0)  # Scale factor
        
        # Time window for SFR
        time_window = sfr_time_window
        
        # Get radial distances
        star_r = compute_radial_distance(star_pos, halo_pos)
        
        # Select recently formed stars
        age_mask = (star_formation_time > 0) & (star_formation_time > current_time - time_window)
        
        n_young_stars = age_mask.sum()
        print(f"    Found {n_young_stars} stars formed in last da={time_window:.3f}")
        
        if n_young_stars == 0:
            print(f"    WARNING: No recent star formation detected!")
            print(f"    Consider increasing --sfr-time-window (current: {time_window:.3f})")
        
        # Bin stellar mass
        star_bin_idx = np.digitize(star_r, r_bins) - 1
        
        for i in range(n_bins):
            mask = (star_bin_idx == i) & age_mask
            if mask.sum() > 0:
                # Total stellar mass formed recently in this shell
                mass_formed = np.sum(star_mass[mask])  # Already in code units
                
                # Convert to M_sun (if in 1e10 M_sun units)
                mass_formed_msun = mass_formed * 1e10
                
                # Convert time window to years
                time_yr = 1e8  # 100 Myr in years
                
                # SFR in M_sun/yr
                profile['sfr'][i] = mass_formed_msun / time_yr
                
                # Also track young stellar mass
                profile['stellar_mass_young'][i] = mass_formed_msun
                profile['star_count'][i] = int(mask.sum())
        
        # Compute SFR surface density (M_sun/yr/kpc^2)
        for i in range(n_bins):
            if profile['sfr'][i] > 0:
                # Area of shell
                area = np.pi * (r_bins[i+1]**2 - r_bins[i]**2)
                profile['sfr_surface_density'][i] = profile['sfr'][i] / area
    
    return profile


def analyze_temperature_distribution(halo, halo_pos):
    """Print temperature distribution diagnostics."""
    
    if 'gas' not in halo or 'InternalEnergy' not in halo['gas']:
        print("No gas internal energy data to analyze!")
        return None
    
    # Calculate temperatures
    electron_abundance = halo['gas'].get('ElectronAbundance', None)
    temperatures = compute_gas_temperature(
        halo['gas']['InternalEnergy'],
        electron_abundance
    )
    
    gas_r = compute_radial_distance(halo['gas']['Coordinates'], halo_pos)
    gas_masses = halo['gas']['Masses']
    
    print("\n" + "="*60)
    print("TEMPERATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal gas particles: {len(temperatures):,}")
    
    print("\nTemperature Statistics:")
    print(f"  Min:    {temperatures.min():.2e} K")
    print(f"  Max:    {temperatures.max():.2e} K")
    print(f"  Median: {np.median(temperatures):.2e} K")
    print(f"  Mean:   {np.mean(temperatures):.2e} K")
    print(f"  Mass-weighted mean: {np.average(temperatures, weights=gas_masses):.2e} K")
    
    print("\nTemperature Ranges:")
    temp_bins = [(0, 1e4, "Cold"), (1e4, 1e5, "Warm"), (1e5, 1e6, "Hot"), (1e6, 1e8, "Very Hot")]
    for t_min, t_max, label in temp_bins:
        mask = (temperatures >= t_min) & (temperatures < t_max)
        count = int(mask.sum())
        if count > 0:
            frac = 100.0 * count / len(temperatures)
            mass_frac = 100.0 * np.sum(gas_masses[mask]) / np.sum(gas_masses)
            print(f"  {label:10s} ({t_min:.0e}-{t_max:.0e} K): {frac:5.1f}% particles, {mass_frac:5.1f}% mass")
    
    print("\nRadial Temperature Distribution:")
    radial_bins = [0, 10, 30, 50, 100, 150, 200]
    for i in range(len(radial_bins)-1):
        mask = (gas_r >= radial_bins[i]) & (gas_r < radial_bins[i+1])
        if mask.sum() > 0:
            temp_mean = np.average(temperatures[mask], weights=gas_masses[mask])
            temp_median = np.median(temperatures[mask])
            print(f"  {radial_bins[i]:4.0f}-{radial_bins[i+1]:4.0f} kpc: "
                  f"Mean = {temp_mean:.2e} K, Median = {temp_median:.2e} K")
    
    print("="*60 + "\n")
    
    return {
        'temperatures': temperatures,
        'gas_r': gas_r,
        'gas_masses': gas_masses
    }


def plot_thermal_profiles(profile, halo_info, redshift, temp_analysis=None, output_file=None):
    """Create comprehensive temperature and SFR plots."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, top=0.92)
    
    # Title
    title_str = f'Temperature & SFR Analysis - Target Halo (M={halo_info["mass"]:.2e}) - z = {redshift:.3f}'
    fig.suptitle(title_str, fontsize=14, fontweight='bold')
    
    r = profile['r_centers']
    r_max_plot = min(profile['r_bins'][-1], 200)
    
    # 1. Temperature Profile (mass-weighted)
    ax1 = fig.add_subplot(gs[0, 0])
    mask = profile['temperature'] > 0
    ax1.semilogy(r[mask], profile['temperature'][mask], 'o-', color='red', 
                 lw=2, markersize=6, label='Mass-weighted mean')
    ax1.semilogy(r[mask], profile['temp_median'][mask], 's--', color='orange', 
                 lw=1.5, markersize=4, alpha=0.7, label='Median')
    ax1.axhline(1e4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(1e5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(1e6, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(r_max_plot*0.8, 1e4, '10⁴ K', fontsize=8, alpha=0.6)
    ax1.text(r_max_plot*0.8, 1e5, '10⁵ K', fontsize=8, alpha=0.6)
    ax1.text(r_max_plot*0.8, 1e6, '10⁶ K', fontsize=8, alpha=0.6)
    ax1.set_xlabel('Radius (kpc)', fontsize=10)
    ax1.set_ylabel('Temperature (K)', fontsize=10)
    ax1.set_xlim(0, r_max_plot)
    ax1.set_ylim(1e3, 1e7)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    ax1.set_title('Gas Temperature Profile', fontweight='bold')
    
    # 2. Star Formation Rate Profile
    ax2 = fig.add_subplot(gs[0, 1])
    mask = profile['sfr'] > 0
    if mask.sum() > 0:
        ax2.semilogy(r[mask], profile['sfr'][mask], 'o-', color='blue', 
                     lw=2, markersize=6)
        ax2.set_ylim(bottom=1e-4)
    else:
        # No SFR detected - add informative text
        ax2.text(0.5, 0.5, 'No recent star formation\ndetected in time window', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
    ax2.set_xlabel('Radius (kpc)', fontsize=10)
    ax2.set_ylabel('SFR (M$_\\odot$/yr)', fontsize=10)
    ax2.set_xlim(0, r_max_plot)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Star Formation Rate Profile', fontweight='bold')
    
    # 3. SFR Surface Density
    ax3 = fig.add_subplot(gs[0, 2])
    mask = profile['sfr_surface_density'] > 0
    if mask.sum() > 0:
        ax3.semilogy(r[mask], profile['sfr_surface_density'][mask], 'o-', 
                     color='green', lw=2, markersize=6)
        ax3.set_ylim(bottom=1e-6)
    else:
        # No SFR detected
        ax3.text(0.5, 0.5, 'No recent star formation\ndetected in time window', 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
    ax3.set_xlabel('Radius (kpc)', fontsize=10)
    ax3.set_ylabel('$\\Sigma_{\\rm SFR}$ (M$_\\odot$/yr/kpc²)', fontsize=10)
    ax3.set_xlim(0, r_max_plot)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('SFR Surface Density', fontweight='bold')
    
    # 4. Temperature vs Gas Mass
    ax4 = fig.add_subplot(gs[1, 0])
    mask = (profile['gas_mass'] > 0) & (profile['temperature'] > 0)
    if mask.sum() > 0:
        ax4.loglog(profile['gas_mass'][mask], profile['temperature'][mask], 
                   'o', color='purple', markersize=8, alpha=0.6)
    ax4.set_xlabel('Gas Mass in Shell (M$_\\odot$)', fontsize=10)
    ax4.set_ylabel('Temperature (K)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Temperature vs Gas Mass', fontweight='bold')
    
    # 5. SFR vs Temperature
    ax5 = fig.add_subplot(gs[1, 1])
    mask = (profile['sfr'] > 0) & (profile['temperature'] > 0)
    if mask.sum() > 0:
        scatter = ax5.scatter(profile['temperature'][mask], profile['sfr'][mask],
                            c=r[mask], s=60, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax5, label='Radius (kpc)')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
    else:
        # No SFR detected
        ax5.text(0.5, 0.5, 'No recent star formation\ndetected in time window', 
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
    ax5.set_xlabel('Temperature (K)', fontsize=10)
    ax5.set_ylabel('SFR (M$_\\odot$/yr)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('SFR vs Temperature', fontweight='bold')
    
    # 6. Gas Mass Profile
    ax6 = fig.add_subplot(gs[1, 2])
    mask = profile['gas_mass'] > 0
    ax6.semilogy(r[mask], profile['gas_mass'][mask], 'o-', color='steelblue', 
                 lw=2, markersize=6)
    ax6.set_xlabel('Radius (kpc)', fontsize=10)
    ax6.set_ylabel('Gas Mass per bin (M$_\\odot$)', fontsize=10)
    ax6.set_xlim(0, r_max_plot)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Gas Mass Profile', fontweight='bold')
    
    # 7. Temperature Distribution Histogram
    if temp_analysis:
        ax7 = fig.add_subplot(gs[2, 0])
        temps = temp_analysis['temperatures']
        ax7.hist(np.log10(temps), bins=50, color='red', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('log$_{10}$(Temperature / K)', fontsize=10)
        ax7.set_ylabel('Count', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('Temperature Distribution', fontweight='bold')
    
    # 8. Radial Temperature Distribution (2D histogram)
    if temp_analysis:
        ax8 = fig.add_subplot(gs[2, 1])
        temps = temp_analysis['temperatures']
        gas_r = temp_analysis['gas_r']
        
        # Create 2D histogram
        r_edges = np.linspace(0, r_max_plot, 50)
        t_edges = np.logspace(3, 7, 50)
        
        H, xedges, yedges = np.histogram2d(gas_r, temps, bins=[r_edges, t_edges])
        
        # Plot with blue-to-red colormap
        im = ax8.pcolormesh(xedges, yedges, H.T, cmap='RdYlBu_r', shading='auto', 
                           norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=H.max()))
        plt.colorbar(im, ax=ax8, label='Particle count')
        ax8.set_xlabel('Radius (kpc)', fontsize=10)
        ax8.set_ylabel('Temperature (K)', fontsize=10)
        ax8.set_yscale('log')
        ax8.set_xlim(0, r_max_plot)
        ax8.set_title('Temperature-Radius Distribution', fontweight='bold')
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    total_gas = profile['gas_mass'].sum()
    total_sfr = profile['sfr'].sum()
    
    max_temp = profile['temperature'].max() if profile['temperature'].max() > 0 else 0
    max_sfr = profile['sfr'].max() if profile['sfr'].max() > 0 else 0
    
    # Find radius of peak SFR
    if max_sfr > 0:
        peak_sfr_idx = np.argmax(profile['sfr'])
        peak_sfr_radius = r[peak_sfr_idx]
    else:
        peak_sfr_radius = 0
    
    # Find radius of peak temperature
    if max_temp > 0:
        peak_temp_idx = np.argmax(profile['temperature'])
        peak_temp_radius = r[peak_temp_idx]
    else:
        peak_temp_radius = 0
    
    stats_text = f"""Summary Statistics:

Total Gas:     {total_gas:.2e} M☉
Total SFR:     {total_sfr:.3f} M☉/yr

Max Temperature: {max_temp:.2e} K
  at radius:     {peak_temp_radius:.1f} kpc

Peak SFR:      {max_sfr:.3f} M☉/yr
  at radius:   {peak_sfr_radius:.1f} kpc

Gas particles: {profile['gas_count'].sum():,}
SF gas cells:  {profile['star_count'].sum():,}
"""
    
    # Add warning if hot gas near metal peak
    if temp_analysis:
        # Check 40-60 kpc region
        mask_4060 = (temp_analysis['gas_r'] >= 40) & (temp_analysis['gas_r'] <= 60)
        if mask_4060.sum() > 0:
            temp_4060 = np.average(temp_analysis['temperatures'][mask_4060], 
                                  weights=temp_analysis['gas_masses'][mask_4060])
            if temp_4060 > 5e5:
                stats_text += f"\n⚠ Hot gas (T={temp_4060:.1e} K)\n  at 40-60 kpc"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_file}")
    
    plt.show()


def print_focus_region_analysis(profile, r_min=40, r_max=60):
    """Print detailed analysis of the metal peak region."""
    
    print("\n" + "="*60)
    print(f"FOCUS ON {r_min}-{r_max} kpc REGION (where metal peak occurs)")
    print("="*60)
    
    r = profile['r_centers']
    mask = (r >= r_min) & (r <= r_max)
    
    if not mask.any():
        print("No data in this region!")
        return
    
    print(f"\n{'Radius':>8s} {'Temp(K)':>12s} {'SFR':>12s} {'Gas Mass':>12s}")
    print(f"{'(kpc)':>8s} {'':>12s} {'(Msun/yr)':>12s} {'(Msun)':>12s}")
    print("-"*60)
    
    for i in np.where(mask)[0]:
        print(f"{r[i]:8.1f} {profile['temperature'][i]:12.2e} "
              f"{profile['sfr'][i]:12.3e} {profile['gas_mass'][i]:12.2e}")
    
    # Summary for this region
    if mask.sum() > 0:
        avg_temp = np.average(profile['temperature'][mask], 
                            weights=profile['gas_mass'][mask])
        total_sfr = profile['sfr'][mask].sum()
        total_gas = profile['gas_mass'][mask].sum()
        n_sf_cells = profile['star_count'][mask].sum()
        
        print("-"*60)
        print(f"\nRegion Summary ({r_min}-{r_max} kpc):")
        print(f"  Avg temperature: {avg_temp:.2e} K")
        print(f"  Total SFR:       {total_sfr:.3e} M☉/yr")
        print(f"  SF gas cells:    {n_sf_cells}")
        print(f"  Total gas mass:  {total_gas:.2e} M☉")
        
        # Diagnosis
        print(f"\nDiagnosis:")
        if avg_temp > 5e5:
            print("  ✓ High temperature regime (T > 5×10⁵ K)")
            print("    → Thermal sputtering likely destroying dust")
            print("    → Metals present but can't condense to dust")
        elif avg_temp > 1e5:
            print("  ✓ Warm/hot gas (T > 10⁵ K)")
            print("    → Reduced dust condensation efficiency")
        else:
            print("  ✓ Cool gas (T < 10⁵ K)")
            print("    → Dust condensation should be efficient")
        
        if total_sfr > 0.01:
            print(f"  ✓ Active star formation ({total_sfr:.3f} M☉/yr)")
            print("    → Recent stellar feedback heating gas")
            print("    → Fresh metal enrichment from SNe")
        else:
            print(f"  ✗ No/minimal star formation ({total_sfr:.3e} M☉/yr)")
            print("    → Feedback may have heated gas above SF threshold")
            print("    → Or gas already fully converted to stars")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Temperature and SFR analysis for target halo')
    parser.add_argument('--catalog', required=True, 
                       help='Path to Subfind catalog (fof_subhalo_tab_*.hdf5)')
    parser.add_argument('--snapshot', required=True, 
                       help='Base path to snapshot (e.g., snapshot_049)')
    parser.add_argument('--out', default='temp_sfr_analysis.png', 
                       help='Output filename')
    parser.add_argument('--rmax', type=float, default=200, 
                       help='Maximum radius for analysis (kpc). Default: 200')
    parser.add_argument('--focus-rmin', type=float, default=40,
                       help='Focus region minimum radius (kpc). Default: 40')
    parser.add_argument('--focus-rmax', type=float, default=60,
                       help='Focus region maximum radius (kpc). Default: 60')
    parser.add_argument('--sfr-time-window', type=float, default=0.1,
                       help='Time window for SFR calculation (scale factor units). Default: 0.1. '
                            'Use larger values (e.g., 0.3) at low redshift.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TEMPERATURE & SFR ANALYSIS")
    print("="*60)
    
    # Extract target halo (need gas and stars)
    print("\nExtracting target halo...")
    halo = load_target_halo(
        args.catalog,
        args.snapshot,
        particle_types=[0, 4],  # Gas and stars
        verbose=True
    )
    
    halo_info = halo['halo_info']
    halo_pos = halo_info['position']
    
    # Get redshift and time
    import h5py
    with h5py.File(f'{args.snapshot}.0.hdf5', 'r') as f:
        redshift = float(f['Header'].attrs['Redshift'])
        time = float(f['Header'].attrs['Time'])
    
    halo['Time'] = time  # Store for SFR calculation
    
    # Analyze temperature distribution
    temp_analysis = analyze_temperature_distribution(halo, halo_pos)
    
    # Compute thermal profiles
    print(f"\nComputing temperature and SFR profiles (rmax={args.rmax} kpc)...")
    print(f"  SFR time window: da = {args.sfr_time_window:.3f}")
    profile = compute_thermal_profiles(halo, halo_pos, 
                                      r_max=args.rmax,
                                      sfr_time_window=args.sfr_time_window)
    
    # Print focus region analysis
    print_focus_region_analysis(profile, args.focus_rmin, args.focus_rmax)
    
    # Create plots
    print("\nCreating plots...")
    plot_thermal_profiles(profile, halo_info, redshift, 
                         temp_analysis=temp_analysis,
                         output_file=args.out)
    
    # Save data
    npz_file = args.out.replace('.png', '.npz')
    np.savez(npz_file,
             r_centers=profile['r_centers'],
             r_bins=profile['r_bins'],
             temperature=profile['temperature'],
             temp_std=profile['temp_std'],
             temp_median=profile['temp_median'],
             sfr=profile['sfr'],
             sfr_surface_density=profile['sfr_surface_density'],
             gas_mass=profile['gas_mass'],
             stellar_mass_young=profile['stellar_mass_young'],
             gas_count=profile['gas_count'],
             star_count=profile['star_count'])
    
    print(f"Saved data: {npz_file}")
    print("\nDone!")


if __name__ == '__main__':
    main()
