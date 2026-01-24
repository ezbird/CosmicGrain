#!/usr/bin/env python3
"""
Analyze metallicity as a function of gas density to check if metals
are concentrated in high-density regions (galaxies) vs low-density IGM.

For uniform box simulations, this reveals if enrichment is happening locally
in galaxies even if box-averaged metallicity is low.

Usage: 
    python metallicity_vs_density.py snapshot_040.hdf5
    python metallicity_vs_density.py snapdir_040/
"""

import sys
import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Observational expectations (from literature)
OBS_DATA = {
    'igm_z': {
        'value': 0.001,  # ~10^-3 Z☉ (Peeples+2014, Tumlinson+2011)
        'range': (0.0001, 0.01),
        'description': 'IGM/warm-hot gas (T~10^5-10^6 K)',
        'refs': 'Peeples+14, Tumlinson+11'
    },
    'cgm_z': {
        'value': 0.01,  # ~10^-2 Z☉ (Werk+2014)
        'range': (0.001, 0.1),
        'description': 'CGM (circumgalactic medium)',
        'refs': 'Werk+14, Prochaska+17'
    },
    'ism_dwarf_z': {
        'value': 0.2,  # 0.2 Z☉ for 10^9 M☉ galaxies
        'range': (0.1, 0.4),
        'description': 'Dwarf galaxy ISM (10^8-10^9 M☉)',
        'refs': 'Zahid+14, Sanders+21'
    },
    'ism_mw_z': {
        'value': 0.5,  # 0.5 Z☉ for 10^10 M☉ galaxies
        'range': (0.3, 0.7),
        'description': 'MW-mass galaxy ISM (10^10 M☉)',
        'refs': 'Zahid+14, Tremonti+04'
    },
    'ism_massive_z': {
        'value': 0.8,  # 0.8 Z☉ for 10^11 M☉ galaxies
        'range': (0.6, 1.0),
        'description': 'Massive galaxy ISM (10^11 M☉)',
        'refs': 'Zahid+14, Gallazzi+05'
    },
    'enrichment_ratio': {
        'value': 100,
        'range': (50, 1000),
        'description': 'Galaxy/IGM metallicity ratio',
        'refs': 'Peeples+14'
    }
}

def find_snapshot_files(path):
    """Find snapshot file(s) - handles both single files and snapdir folders."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # It's a snapdir - find all HDF5 files
        files = sorted(glob.glob(os.path.join(path, '*.hdf5')) + 
                      glob.glob(os.path.join(path, '*.h5')))
        if not files:
            print(f"ERROR: No HDF5 files found in {path}")
            sys.exit(1)
        return files
    else:
        print(f"ERROR: {path} is neither a file nor a directory")
        sys.exit(1)

def load_gas_data(snapshot_files):
    """Load and combine gas data from multiple snapshot files."""
    print(f"Reading {len(snapshot_files)} file(s)...")
    
    all_density = []
    all_Z = []
    all_masses = []
    redshift = None
    time = None
    
    for i, snap_file in enumerate(snapshot_files):
        try:
            with h5py.File(snap_file, 'r') as f:
                if 'PartType0' not in f:
                    continue
                
                gas = f['PartType0']
                
                # Read data from this file
                density = np.array(gas['Density'])
                masses = np.array(gas['Masses'])
                
                if 'Metallicity' not in gas:
                    print(f"WARNING: No metallicity in {os.path.basename(snap_file)}")
                    continue
                
                Z = np.array(gas['Metallicity'])
                if len(Z.shape) > 1:
                    Z = Z.sum(axis=1)
                
                all_density.append(density)
                all_Z.append(Z)
                all_masses.append(masses)
                
                # Get time info from first file
                if redshift is None:
                    header = f['Header'].attrs
                    redshift = header['Redshift']
                    time = header['Time']
                
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(snapshot_files)} files...")
                    
        except Exception as e:
            print(f"WARNING: Error reading {snap_file}: {e}")
            continue
    
    if not all_density:
        print("ERROR: No gas data loaded")
        return None
    
    # Combine arrays
    density = np.concatenate(all_density)
    Z = np.concatenate(all_Z)
    masses = np.concatenate(all_masses)
    
    print(f"✓ Loaded {len(density):,} gas particles")
    
    return {
        'density': density,
        'Z': Z,
        'masses': masses,
        'redshift': redshift,
        'time': time
    }

def analyze_Z_vs_density(snapshot_path):
def analyze_Z_vs_density(snapshot_path):
    """Analyze metallicity stratification by density."""
    
    print("="*70)
    print("METALLICITY vs DENSITY ANALYSIS")
    print("="*70)
    print()
    
    # Find and load snapshot file(s)
    snapshot_files = find_snapshot_files(snapshot_path)
    data = load_gas_data(snapshot_files)
    
    if data is None:
        return
    
    density = data['density']
    Z = data['Z']
    masses = data['masses']
    redshift = data['redshift']
    time = data['time']
    
    # Get physical density
    UnitMass = 1.989e43  # 10^10 Msun in g
    UnitLength = 3.085678e24  # Mpc in cm
    rho_cgs = density * (UnitMass / UnitLength**3)
    rho_physical = rho_cgs / time**3  # Comoving to physical
    
    Z_solar = 0.0134
    
    print()
    print(f"Snapshot: z={redshift:.2f}, a={time:.4f}")
    print(f"Total gas particles: {len(density):,}")
    print()
    
    # Define density bins (physical density in g/cm^3)
    rho_bins = [
        (0, 1e-28, "IGM (diffuse)"),
        (1e-28, 1e-26, "Filaments"),
        (1e-26, 1e-25, "Halos (outer)"),
        (1e-25, 1e-24, "Halos (inner)"),
        (1e-24, 1e-22, "Galaxies (ISM)"),
        (1e-22, np.inf, "Galaxies (dense)")
    ]
    
    print("METALLICITY BY DENSITY REGIME:")
    print("="*70)
    print(f"{'Regime':<20s} {'N_part':>10s} {'Mass %':>8s} {'<Z>':>12s} {'Z/Z☉':>8s}")
    print("-"*70)
    
    total_mass = np.sum(masses)
    
    for rho_min, rho_max, label in rho_bins:
        mask = (rho_physical >= rho_min) & (rho_physical < rho_max)
        n_part = mask.sum()
        
        if n_part > 0:
            mass_frac = np.sum(masses[mask]) / total_mass * 100
            Z_avg = np.average(Z[mask], weights=masses[mask])
            
            print(f"{label:<20s} {n_part:>10,d} {mass_frac:>7.1f}% {Z_avg:>12.6f} {Z_avg/Z_solar:>7.3f}")
    
    print("="*70)
    print()
    
    # Key statistics with observational comparisons
    igm_mask = rho_physical < 1e-26
    cgm_mask = (rho_physical >= 1e-26) & (rho_physical < 1e-24)
    galaxy_mask = rho_physical > 1e-24
    
    print("KEY STATISTICS & OBSERVATIONAL COMPARISON:")
    print("="*70)
    
    if igm_mask.sum() > 0:
        Z_igm = np.average(Z[igm_mask], weights=masses[igm_mask])
        mass_igm = np.sum(masses[igm_mask]) / total_mass * 100
        
        obs = OBS_DATA['igm_z']
        in_range = obs['range'][0] < Z_igm/Z_solar < obs['range'][1]
        status = "✓" if in_range else "⚠️"
        
        print(f"\nIGM (ρ < 10⁻²⁶ g/cm³):")
        print(f"  Simulation:  {mass_igm:5.1f}% of gas mass, <Z> = {Z_igm/Z_solar:.4f} Z☉")
        print(f"  Expected:    {obs['description']}")
        print(f"               Z ~ {obs['value']:.4f} Z☉ (range: {obs['range'][0]:.4f}-{obs['range'][1]:.4f})")
        print(f"               {obs['refs']}")
        print(f"  Status:      {status} {'In range' if in_range else 'Out of range'}")
    
    if cgm_mask.sum() > 0:
        Z_cgm = np.average(Z[cgm_mask], weights=masses[cgm_mask])
        mass_cgm = np.sum(masses[cgm_mask]) / total_mass * 100
        
        obs = OBS_DATA['cgm_z']
        in_range = obs['range'][0] < Z_cgm/Z_solar < obs['range'][1]
        status = "✓" if in_range else "⚠️"
        
        print(f"\nCGM/Halos (10⁻²⁶ < ρ < 10⁻²⁴):")
        print(f"  Simulation:  {mass_cgm:5.1f}% of gas mass, <Z> = {Z_cgm/Z_solar:.4f} Z☉")
        print(f"  Expected:    {obs['description']}")
        print(f"               Z ~ {obs['value']:.4f} Z☉ (range: {obs['range'][0]:.4f}-{obs['range'][1]:.4f})")
        print(f"               {obs['refs']}")
        print(f"  Status:      {status} {'In range' if in_range else 'Out of range'}")
    
    if galaxy_mask.sum() > 0:
        Z_gal = np.average(Z[galaxy_mask], weights=masses[galaxy_mask])
        mass_gal = np.sum(masses[galaxy_mask]) / total_mass * 100
        
        # Determine which galaxy type based on metallicity
        if Z_gal/Z_solar < 0.35:
            obs = OBS_DATA['ism_dwarf_z']
        elif Z_gal/Z_solar < 0.65:
            obs = OBS_DATA['ism_mw_z']
        else:
            obs = OBS_DATA['ism_massive_z']
        
        in_range = obs['range'][0] < Z_gal/Z_solar < obs['range'][1]
        status = "✓" if in_range else "⚠️"
        
        print(f"\nGalaxies/ISM (ρ > 10⁻²⁴):")
        print(f"  Simulation:  {mass_gal:5.1f}% of gas mass, <Z> = {Z_gal/Z_solar:.4f} Z☉")
        print(f"  Expected:    {obs['description']}")
        print(f"               Z ~ {obs['value']:.4f} Z☉ (range: {obs['range'][0]:.4f}-{obs['range'][1]:.4f})")
        print(f"               {obs['refs']}")
        print(f"  Status:      {status} {'In range' if in_range else 'Out of range'}")
        
        # Enrichment ratio
        if igm_mask.sum() > 0:
            enrichment_ratio = Z_gal / Z_igm
            
            obs_ratio = OBS_DATA['enrichment_ratio']
            in_range = obs_ratio['range'][0] < enrichment_ratio < obs_ratio['range'][1]
            status = "✓" if in_range else "⚠️"
            
            print(f"\nGalaxy/IGM Enrichment Ratio:")
            print(f"  Simulation:  {enrichment_ratio:.1f}x")
            print(f"  Expected:    {obs_ratio['value']:.0f}x (range: {obs_ratio['range'][0]:.0f}-{obs_ratio['range'][1]:.0f}x)")
            print(f"               {obs_ratio['refs']}")
            print(f"  Status:      {status} {'In range' if in_range else 'Out of range'}")
            
            print()
            if enrichment_ratio > obs_ratio['range'][1]:
                print("  ⚠️  VERY HIGH enrichment ratio!")
                print("      → Metals strongly concentrated in galaxies")
                print("      → May indicate over-enrichment if Z_galaxy > 1 Z☉")
            elif enrichment_ratio > obs_ratio['range'][0]:
                print("  ✓ Good concentration of metals in galaxies")
            else:
                print("  ⚠️  Low enrichment ratio!")
                print("      → Metals not well concentrated")
                print("      → May indicate diffuse metal injection or weak feedback")
    
    print("="*70)
    print()
    
    # Create plot
    print("Creating diagnostic plot...")
    
    # Get snapshot name for plot title
    if os.path.isdir(snapshot_path):
        snap_name = os.path.basename(snapshot_path.rstrip('/'))
    else:
        snap_name = os.path.basename(snapshot_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: 2D histogram
    ax = ax1
    h = ax.hist2d(np.log10(rho_physical), Z/Z_solar, bins=[50, 50],
                 cmap='viridis', norm=matplotlib.colors.LogNorm(),
                 cmin=1)
    plt.colorbar(h[3], ax=ax, label='Number of particles')
    ax.set_xlabel('log Density [g/cm³]', fontsize=12)
    ax.set_ylabel('Metallicity [Z☉]', fontsize=12)
    ax.set_title(f'Metallicity vs Density (z={redshift:.2f})', fontweight='bold')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Solar')
    ax.axvline(-24, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='SF threshold')
    
    # Add observational expectations
    ax.axhspan(OBS_DATA['igm_z']['range'][0], OBS_DATA['igm_z']['range'][1], 
              alpha=0.1, color='blue', label='Expected IGM')
    ax.axhspan(OBS_DATA['ism_mw_z']['range'][0], OBS_DATA['ism_mw_z']['range'][1], 
              alpha=0.1, color='green', label='Expected MW-like')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Mass-weighted average in bins
    ax = ax2
    rho_bin_edges = np.logspace(-30, -20, 30)
    rho_bin_centers = np.sqrt(rho_bin_edges[:-1] * rho_bin_edges[1:])
    Z_binned = []
    mass_binned = []
    
    for i in range(len(rho_bin_edges)-1):
        mask = (rho_physical >= rho_bin_edges[i]) & (rho_physical < rho_bin_edges[i+1])
        if mask.sum() > 0:
            Z_avg = np.average(Z[mask], weights=masses[mask])
            m_total = np.sum(masses[mask])
            Z_binned.append(Z_avg / Z_solar)
            mass_binned.append(m_total)
        else:
            Z_binned.append(np.nan)
            mass_binned.append(0)
    
    # Plot with point size proportional to mass
    mass_binned = np.array(mass_binned)
    mass_normalized = mass_binned / np.nanmax(mass_binned) * 200 + 10
    
    ax.scatter(np.log10(rho_bin_centers), Z_binned, s=mass_normalized,
              c='blue', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add observational expectations
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Solar')
    ax.axvline(-24, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='SF threshold')
    
    # Add expected ranges as horizontal bands
    ax.axhspan(OBS_DATA['igm_z']['range'][0], OBS_DATA['igm_z']['range'][1], 
              alpha=0.15, color='lightblue', label='Expected IGM')
    ax.axhspan(OBS_DATA['cgm_z']['range'][0], OBS_DATA['cgm_z']['range'][1], 
              alpha=0.15, color='cyan', label='Expected CGM')
    ax.axhspan(OBS_DATA['ism_mw_z']['range'][0], OBS_DATA['ism_mw_z']['range'][1], 
              alpha=0.15, color='lightgreen', label='Expected ISM (MW)')
    
    ax.set_xlabel('log Density [g/cm³]', fontsize=12)
    ax.set_ylabel('Mass-weighted <Z> [Z☉]', fontsize=12)
    ax.set_title(f'Mean Metallicity by Density ({snap_name})', fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_name = f'metallicity_vs_density_{snap_name}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_name}")
    print()
    
    print("="*70)
    print("INTERPRETATION:")
    print("="*70)
    print()
    print("For a uniform box simulation:")
    print("  • Most GAS MASS should be in low-density IGM (ρ < 10⁻²⁶)")
    print("  • Most METALS should be in high-density galaxies (ρ > 10⁻²⁴)")
    print("  • Galaxy metallicity should be 50-1000x higher than IGM")
    print()
    print("Plot interpretation:")
    print("  ✓ Strong upward trend → metals concentrated in galaxies (GOOD)")
    print("  ✗ Flat trend → metals evenly distributed (BAD - too diffuse)")
    print("  ✗ Very high IGM Z → metals escaping galaxies or over-injecting")
    print()
    print("Observational references:")
    print("  • IGM/warm-hot gas: Peeples+2014, Tumlinson+2011")
    print("  • CGM metallicity: Werk+2014, Prochaska+2017")
    print("  • Galaxy ISM: Zahid+2014, Tremonti+2004, Gallazzi+2005")
    print("="*70)
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python metallicity_vs_density.py snapshot_040.hdf5")
        print("  python metallicity_vs_density.py snapdir_040/")
        sys.exit(1)
    
    analyze_Z_vs_density(sys.argv[1])
