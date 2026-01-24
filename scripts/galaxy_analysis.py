#!/usr/bin/env python3
"""
Galaxy Formation Diagnostics for Gadget-4 Simulations
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_galaxy_evolution():
    """Plot galaxy formation history from log file"""
    
    # Read the galaxy evolution log
    try:
        data = np.loadtxt('galaxy_evolution.txt')
        time = data[:, 0]
        redshift = data[:, 1] 
        n_stars = data[:, 2]
        stellar_mass = data[:, 3]
    except:
        print("No galaxy_evolution.txt found. Run simulation with diagnostics first.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Star formation history
    ax1.plot(redshift, n_stars, 'b-o', markersize=4)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('Number of Star Particles')
    ax1.set_title('Star Formation History')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher z on left
    
    # Plot 2: Stellar mass evolution
    ax2.semilogy(redshift, stellar_mass, 'r-s', markersize=4)
    ax2.set_xlabel('Redshift')  
    ax2.set_ylabel('Total Stellar Mass [M☉]')
    ax2.set_title('Stellar Mass Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Star formation rate
    if len(time) > 1:
        dt = np.diff(time)
        dm = np.diff(stellar_mass)
        sfr = dm / dt / 1e6  # Msun/yr (approximate conversion)
        t_mid = 0.5 * (time[1:] + time[:-1])
        z_mid = 1.0/t_mid - 1.0
        
        ax3.semilogy(z_mid, np.abs(sfr), 'g-^', markersize=4)
        ax3.set_xlabel('Redshift')
        ax3.set_ylabel('Star Formation Rate [M☉/yr]')
        ax3.set_title('Star Formation Rate')
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
    
    # Plot 4: Galaxy count estimate
    # Rough estimate: 1 galaxy per ~100-1000 star particles
    galaxy_estimate = n_stars / 500  # Adjust this factor
    ax4.plot(redshift, galaxy_estimate, 'm-d', markersize=4)
    ax4.set_xlabel('Redshift')
    ax4.set_ylabel('Estimated Galaxy Count')
    ax4.set_title('Galaxy Number Estimate')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('galaxy_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Current status:")
    print(f"  Redshift: {redshift[-1]:.2f}")
    print(f"  Star particles: {n_stars[-1]:d}")
    print(f"  Stellar mass: {stellar_mass[-1]:.2e} Msun")
    print(f"  Estimated galaxies: {galaxy_estimate[-1]:.1f}")

def analyze_galaxy_catalogs():
    """Analyze detailed galaxy catalog files"""
    
    catalog_files = sorted(glob.glob('galaxies_*.txt'))
    if not catalog_files:
        print("No galaxy catalog files found.")
        return
    
    print(f"Found {len(catalog_files)} galaxy catalogs")
    
    # Analyze most recent catalog
    latest_file = catalog_files[-1]
    print(f"Analyzing: {latest_file}")
    
    try:
        # Skip comment lines
        data = np.loadtxt(latest_file)
        
        if len(data) == 0:
            print("No galaxies found in catalog")
            return
            
        galaxy_id = data[:, 0]
        x_pos = data[:, 1]
        y_pos = data[:, 2] 
        z_pos = data[:, 3]
        stellar_mass = data[:, 4]
        age = data[:, 5]
        metallicity = data[:, 6]
        
        # Create analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Spatial distribution
        scatter = ax1.scatter(x_pos, y_pos, c=stellar_mass, s=20, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('X [kpc]')
        ax1.set_ylabel('Y [kpc]')
        ax1.set_title('Galaxy Spatial Distribution')
        plt.colorbar(scatter, ax=ax1, label='Stellar Mass [M☉]')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mass function
        log_mass = np.log10(stellar_mass)
        ax2.hist(log_mass, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('log₁₀(Stellar Mass [M☉])')
        ax2.set_ylabel('Number of Galaxies')
        ax2.set_title('Galaxy Stellar Mass Function')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Age distribution
        ax3.hist(age, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_xlabel('Age [Gyr]')
        ax3.set_ylabel('Number of Galaxies')
        ax3.set_title('Galaxy Age Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mass vs Metallicity
        ax4.scatter(log_mass, metallicity, alpha=0.7, s=20)
        ax4.set_xlabel('log₁₀(Stellar Mass [M☉])')
        ax4.set_ylabel('Metallicity')
        ax4.set_title('Mass-Metallicity Relation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('galaxy_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\nGalaxy Statistics:")
        print(f"  Total galaxies: {len(stellar_mass)}")
        print(f"  Mass range: {stellar_mass.min():.2e} - {stellar_mass.max():.2e} Msun")
        print(f"  Median mass: {np.median(stellar_mass):.2e} Msun")
        print(f"  Age range: {age.min():.2f} - {age.max():.2f} Gyr") 
        print(f"  Metallicity range: {metallicity.min():.4f} - {metallicity.max():.4f}")
        
    except Exception as e:
        print(f"Error reading catalog: {e}")

def quick_snapshot_analysis(snapshot_file):
    """Quick analysis of Gadget snapshot file"""
    
    try:
        # This would need to be adapted for your specific snapshot format
        # Using basic numpy for demonstration
        
        print(f"Analyzing snapshot: {snapshot_file}")
        print("Note: This requires proper snapshot reading code")
        
        # Placeholder for actual snapshot analysis
        # You'd typically use pygadgetreader, h5py, or similar
        
    except Exception as e:
        print(f"Error reading snapshot: {e}")

def main():
    """Main analysis function"""
    
    print("Galaxy Formation Diagnostics")
    print("=" * 40)
    
    # Check what data is available
    has_evolution_log = os.path.exists('galaxy_evolution.txt')
    has_catalogs = len(glob.glob('galaxies_*.txt')) > 0
    
    if has_evolution_log:
        print("✓ Found galaxy evolution log")
        plot_galaxy_evolution()
    else:
        print("✗ No galaxy evolution log found")
        print("  Add count_galaxies_simple() calls to your simulation")
    
    if has_catalogs:
        print("✓ Found galaxy catalogs") 
        analyze_galaxy_catalogs()
    else:
        print("✗ No galaxy catalogs found")
        print("  Add output_galaxy_catalog() calls to your simulation")
    
    if not has_evolution_log and not has_catalogs:
        print("\nTo get galaxy diagnostics:")
        print("1. Add the diagnostic functions to your simulation code")
        print("2. Call run_galaxy_diagnostics(Sp) in your main loop")
        print("3. Re-run this analysis script")

if __name__ == "__main__":
    main()