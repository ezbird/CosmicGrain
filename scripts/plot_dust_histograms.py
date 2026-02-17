#!/usr/bin/env python3
"""
plot_dust_histograms.py

Create a grid of histograms showing dust particle properties from a Gadget-4 simulation
using halo_utils to extract the target halo.

Usage:
    python plot_dust_histograms.py \
        --catalog ../groups_049/fof_subhalo_tab_049.0.hdf5 \
        --snapshot ../snapdir_049/snapshot_049 \
        --out dust_histograms.png \
        --rmax 300
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import glob

try:
    from halo_utils import load_target_halo, extract_dust_spatially
except ImportError:
    print("ERROR: This script requires halo_utils.py in the same directory")
    exit(1)


def get_snapshot_info(snapshot_base):
    """Get current time/scale factor and cosmology from snapshot header."""
    files = sorted(glob.glob(f'{snapshot_base}.*.hdf5'))
    if not files:
        files = [f'{snapshot_base}.hdf5']
    
    with h5py.File(files[0], 'r') as f:
        header = f['Header'].attrs
        
        info = {}
        if 'Time' in header:
            info['Time'] = float(header['Time'])
        if 'Redshift' in header:
            info['Redshift'] = float(header['Redshift'])
        if 'HubbleParam' in header:
            info['HubbleParam'] = float(header['HubbleParam'])
        if 'Omega0' in header:
            info['Omega0'] = float(header['Omega0'])
        if 'OmegaLambda' in header:
            info['OmegaLambda'] = float(header['OmegaLambda'])
        
        # Check what fields exist for PartType6
        if 'PartType6' in f:
            info['dust_fields'] = list(f['PartType6'].keys())
        
    return info


def scale_factor_to_age(a, h=0.7, Om=0.3, OL=0.7):
    """
    Convert scale factor to age of universe in Gyr for flat ΛCDM.
    
    Parameters:
    -----------
    a : float or array
        Scale factor
    h : float
        Hubble parameter (H0 = h * 100 km/s/Mpc)
    Om : float
        Matter density parameter
    OL : float
        Dark energy density parameter
    
    Returns:
    --------
    age : float or array
        Age in Gyr
    """
    from scipy.integrate import quad
    
    def integrand(a_prime):
        # dt/da = 1/(a * H(a)) where H(a) = H0 * sqrt(Om/a^3 + OL)
        return 1.0 / (a_prime * np.sqrt(Om / a_prime**3 + OL))
    
    # Hubble constant in proper units
    # H0 = h * 100 km/s/Mpc
    # Need to convert to Gyr^-1
    # 1 km = 1e3 m
    # 1 Mpc = 3.0857e22 m  
    # 1 s = 1 s
    # 1 Gyr = 3.1536e16 s
    # H0 [s^-1] = (h * 100 km/s/Mpc) * (1e3 m/km) / (3.0857e22 m/Mpc)
    #           = h * 100 * 1e3 / 3.0857e22 s^-1
    #           = h * 3.241e-18 s^-1
    # H0 [Gyr^-1] = h * 3.241e-18 * 3.1536e16 Gyr^-1
    #             = h * 1.0226e-1 Gyr^-1
    # Hubble time = 1/H0 = 9.778 / h Gyr
    
    t_H = 9.778 / h  # Hubble time in Gyr
    
    # Integrate from 0 to a
    age, _ = quad(integrand, 0, a)
    return age * t_H


def compute_velocity_magnitude(velocities):
    """Compute velocity magnitude from velocity vectors."""
    return np.sqrt(np.sum(velocities**2, axis=1))


def make_histogram(ax, data, xlabel, title, bins=50, log_x=False, log_y=False, color='steelblue'):
    """Create a histogram with nice formatting."""
    
    # Remove non-finite values
    data = data[np.isfinite(data)]
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Create histogram
    if log_x:
        # Use log-spaced bins
        data_pos = data[data > 0]
        if len(data_pos) == 0:
            ax.text(0.5, 0.5, 'No positive data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        bins = np.logspace(np.log10(data_pos.min()), np.log10(data_pos.max()), bins)
        data = data_pos
    
    counts, bin_edges, patches = ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Formatting
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text
    stats_text = f'N = {len(data):,}\nMedian = {np.median(data):.2e}\nMean = {np.mean(data):.2e}'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description='Plot dust particle histograms for target halo')
    parser.add_argument('--catalog', required=True, help='Path to Subfind catalog (fof_subhalo_tab_*.hdf5)')
    parser.add_argument('--snapshot', required=True, help='Base path to snapshot (e.g., snapshot_049)')
    parser.add_argument('--out', default='dust_histograms.png', help='Output filename')
    parser.add_argument('--rmax', type=float, default=None, 
                        help='Max radius for dust extraction (kpc). Default: 2*halfmass_rad')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins per histogram')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 9], 
                        help='Figure size in inches (width height)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DUST HISTOGRAM PLOTTER")
    print("="*60)
    
    # Get snapshot info
    snap_info = get_snapshot_info(args.snapshot)
    current_time = snap_info.get('Time')
    redshift = snap_info.get('Redshift')
    
    print(f"\nSnapshot information:")
    print(f"  Time (scale factor): {current_time:.6f}" if current_time else "  Time: Not found")
    print(f"  Redshift: {redshift:.6f}" if redshift else "  Redshift: Not found")
    if 'dust_fields' in snap_info:
        print(f"  PartType6 fields available: {snap_info['dust_fields']}")
    
    # Load target halo
    print("\nLoading halo info...")
    halo = load_target_halo(
        args.catalog,
        args.snapshot,
        particle_types=[],
        verbose=True
    )
    
    halo_info = halo['halo_info']
    halo_pos = halo_info['position']
    halo_mass = halo_info['mass']
    halo_halfmass = halo_info['halfmass_rad']
    
    # Set extraction radius
    rmax = args.rmax if args.rmax is not None else (halo_halfmass * 2.0)
    print(f"\nExtracting dust within {rmax:.2f} kpc of halo center...")
    
    # Extract dust particles spatially
    dust_data = extract_dust_spatially(
        args.snapshot,
        halo_pos,
        radius_kpc=rmax,
        verbose=True
    )
    
    if dust_data is None or len(dust_data['Coordinates']) == 0:
        print("ERROR: No dust particles found!")
        return
    
    print(f"\nPreparing histograms...")
    
    # Extract dust properties
    grain_radius = dust_data['GrainRadius']
    carbon_frac = dust_data['CarbonFraction']
    masses = dust_data['Masses'] * 1e10
    velocities = dust_data['Velocities']
    dust_temp = dust_data['DustTemperature']
    
    # Diagnose DustFormationTime
    print("\n" + "="*60)
    print("DIAGNOSING DUST FORMATION TIME")
    print("="*60)
    
    has_formation_time = False
    if 'DustFormationTime' in dust_data:
        dust_formation = dust_data['DustFormationTime']
        has_formation_time = True
        print("✓ DustFormationTime field found")
    elif 'StellarFormationTime' in dust_data:
        dust_formation = dust_data['StellarFormationTime']
        has_formation_time = True
        print("✓ Using StellarFormationTime (DustFormationTime not found)")
    else:
        dust_formation = np.zeros(len(grain_radius))
        print("✗ No formation time field found - using zeros")
    
    if has_formation_time:
        print(f"  Min value: {dust_formation.min():.6f}")
        print(f"  Max value: {dust_formation.max():.6f}")
        print(f"  Median: {np.median(dust_formation):.6f}")
        print(f"  Number of zeros: {np.sum(dust_formation == 0):,} / {len(dust_formation):,}")
        print(f"  Number of non-zeros: {np.sum(dust_formation > 0):,}")
        
        # Check if values look like scale factors (0 < a < 1)
        nonzero = dust_formation[dust_formation > 0]
        if len(nonzero) > 0:
            if np.all((nonzero > 0) & (nonzero <= 1)):
                print("  → Values appear to be scale factors (0 < a ≤ 1)")
            else:
                print("  → Values do NOT look like scale factors")
    
    # Calculate age
    if has_formation_time and current_time is not None:
        # If formation times are scale factors and current_time is scale factor
        if np.any(dust_formation > 0):
            try:
                from scipy.integrate import quad
                
                # Get cosmology
                h = snap_info.get('HubbleParam', 0.7)
                Om = snap_info.get('Omega0', 0.3)
                OL = snap_info.get('OmegaLambda', 0.7)
                
                print(f"\nCalculating ages using cosmology:")
                print(f"  h = {h}, Ωm = {Om}, ΩΛ = {OL}")
                
                # Calculate current age
                current_age = scale_factor_to_age(current_time, h, Om, OL)
                print(f"  Current age of universe: {current_age:.3f} Gyr")
                
                # Calculate ages for each particle
                dust_age_gyr = np.zeros(len(dust_formation))
                for i, a_form in enumerate(dust_formation):
                    if a_form > 0 and a_form <= current_time:
                        age_at_form = scale_factor_to_age(a_form, h, Om, OL)
                        dust_age_gyr[i] = current_age - age_at_form
                    elif a_form == 0:
                        # Assume formed at z=0 (current time) if zero
                        dust_age_gyr[i] = 0.0
                
                age_label = 'Age (Gyr)'
                age_title = 'Age'
                print(f"  Age range: {dust_age_gyr.min():.3f} to {dust_age_gyr.max():.3f} Gyr")
                
            except ImportError:
                print("  Warning: scipy not available, using simple approximation")
                dust_age = current_time - dust_formation
                dust_age_gyr = dust_age * 13.8
                age_label = 'Age (approx, Gyr)'
                age_title = 'Age (approx)'
        else:
            print("  All formation times are zero - showing as zero age")
            dust_age_gyr = np.zeros(len(dust_formation))
            age_label = 'Age (Gyr)'
            age_title = 'Age'
    else:
        print("\nCannot calculate age - using formation time directly")
        dust_age_gyr = dust_formation
        age_label = 'Formation Time (scale factor)'
        age_title = 'Formation Scale Factor'
    
    print("="*60 + "\n")
    
    # Compute velocity magnitude
    vel_mag = compute_velocity_magnitude(velocities)
    
    # Create figure with 2x3 grid - more space at top
    fig = plt.figure(figsize=args.figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, top=0.89)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Plot histograms
    make_histogram(ax1, grain_radius, 'Grain Radius (nm)', 'Grain Radius', 
                   bins=args.bins, log_x=False, log_y=False, color='steelblue')
    
    make_histogram(ax2, carbon_frac, 'Carbon Fraction', 'Carbon Fraction', 
                   bins=args.bins, log_x=False, log_y=False, color='forestgreen')
    
    make_histogram(ax3, masses, 'Mass (M$_\\odot$)', 'Masses', 
                   bins=args.bins, log_x=True, log_y=False, color='coral')
    
    make_histogram(ax4, vel_mag, 'Velocity (km/s)', 'Velocity Magnitude', 
                   bins=args.bins, log_x=False, log_y=False, color='purple')
    
    make_histogram(ax5, dust_temp, 'Temperature (K)', 'Temperature', 
                   bins=args.bins, log_x=False, log_y=False, color='crimson')
    
    make_histogram(ax6, dust_age_gyr, age_label, age_title, 
                   bins=args.bins, log_x=False, log_y=False, color='darkorange')
    
    halo_mass*=1e10
    # Add overall title with two lines
    fig.text(0.5, 0.975, 'Dust Properties', 
             fontsize=18, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.94, f'Target Halo (M={halo_mass:.2e}, R<{rmax:.1f} kpc)', 
             fontsize=11, ha='center', va='top')
    
    # Save
    plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {args.out}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total dust particles: {len(grain_radius):,}")
    print(f"\nGrain Radius (nm):    min={grain_radius.min():.2e}, max={grain_radius.max():.2e}, median={np.median(grain_radius):.2e}")
    print(f"Carbon Fraction:      min={carbon_frac.min():.3f}, max={carbon_frac.max():.3f}, median={np.median(carbon_frac):.3f}")
    print(f"Masses (Msun):        min={masses.min():.2e}, max={masses.max():.2e}, median={np.median(masses):.2e}")
    print(f"Velocity (km/s):      min={vel_mag.min():.1f}, max={vel_mag.max():.1f}, median={np.median(vel_mag):.1f}")
    print(f"Temperature (K):      min={dust_temp.min():.1f}, max={dust_temp.max():.1f}, median={np.median(dust_temp):.1f}")
    print(f"Age (Gyr):            min={dust_age_gyr.min():.3f}, max={dust_age_gyr.max():.3f}, median={np.median(dust_age_gyr):.3f}")
    
    plt.show()


if __name__ == "__main__":
    main()
