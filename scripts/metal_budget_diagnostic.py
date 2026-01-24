#!/usr/bin/env python3
"""
Metal Budget Diagnostic for Gadget-4 Simulations

Tracks metal production, distribution, and conservation to diagnose
why metallicity enrichment may be insufficient.

Usage:
    python metal_budget_diagnostic.py output.log ./output_dir
    python metal_budget_diagnostic.py  # Uses defaults
"""

import sys
import os
import glob
import re
import numpy as np
import h5py
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Physical constants
Z_SOLAR = 0.0134  # Solar metallicity
MSUN = 1.989e33  # Solar mass in grams

class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def find_snapshots(output_dir):
    """Find all snapshot files."""
    snapshots = []
    snapdirs = glob.glob(os.path.join(output_dir, "snapdir_*"))
    if snapdirs:
        for snapdir in sorted(snapdirs):
            files = glob.glob(os.path.join(snapdir, "*.hdf5"))
            if files:
                snapshots.append(sorted(files)[0])
    else:
        files = glob.glob(os.path.join(output_dir, "snapshot_*.hdf5"))
        snapshots = sorted(files)
    return snapshots

def analyze_metal_content(snapshot_file):
    """Extract comprehensive metal information from snapshot."""
    try:
        with h5py.File(snapshot_file, 'r') as f:
            header = f['Header'].attrs
            params = f['Parameters'].attrs
            result = {
                'file': os.path.basename(snapshot_file),
                'redshift': header['Redshift'],
                'time': header['Time'],
                'hubble': params['HubbleParam'],
                'metals': {}
            }
            
            # Check each particle type
            for ptype, pname in [(0, 'Gas'), (1, 'DM'), (2, 'Boundary'), 
                                  (3, 'DM2'), (4, 'Stars'), (5, 'BH')]:
                ptype_key = f'PartType{ptype}'
                if ptype_key not in f:
                    continue
                
                part = f[ptype_key]
                if 'Masses' not in part:
                    continue
                    
                masses = np.array(part['Masses'])
                n_particles = len(masses)
                
                result['metals'][pname] = {
                    'n_particles': n_particles,
                    'total_mass': np.sum(masses),
                    'metal_mass': 0.0,
                    'mean_Z': 0.0,
                    'median_Z': 0.0
                }
                
                # Get metallicity if available
                if 'Metallicity' in part:
                    Z = np.array(part['Metallicity'])
                    
                    # Handle multi-species vs scalar
                    if len(Z.shape) > 1:
                        Z_total = Z.sum(axis=1)
                    else:
                        Z_total = Z
                    
                    metal_mass = np.sum(Z_total * masses)
                    result['metals'][pname]['metal_mass'] = metal_mass
                    result['metals'][pname]['mean_Z'] = np.mean(Z_total)
                    result['metals'][pname]['median_Z'] = np.median(Z_total)
                    result['metals'][pname]['Z_array'] = Z_total
                    result['metals'][pname]['mass_array'] = masses
                    
                    # For gas, get additional properties
                    if pname == 'Gas' and 'Coordinates' in part:
                        result['metals'][pname]['coords'] = np.array(part['Coordinates'])
                        
                        # Temperature if available
                        if 'InternalEnergy' in part:
                            u = np.array(part['InternalEnergy'])
                            if 'ElectronAbundance' in part:
                                ne = np.array(part['ElectronAbundance'])
                            else:
                                ne = np.ones(len(masses))
                            
                            X_H = 0.76
                            mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*ne)
                            GAMMA = 5.0/3.0
                            BOLTZMANN = 1.38064852e-16
                            PROTONMASS = 1.6726219e-24
                            UnitVelocity = 1e5
                            u_cgs = u * UnitVelocity**2
                            temp = u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN
                            result['metals'][pname]['temperature'] = temp
                
                # For stars, get formation times if available
                if pname == 'Stars' and 'StellarFormationTime' in part:
                    result['metals'][pname]['formation_time'] = np.array(part['StellarFormationTime'])
            
            return result
            
    except Exception as e:
        print(f"Error reading {snapshot_file}: {e}")
        return None

def parse_feedback_events(log_file):
    """Parse feedback events from log to estimate expected metal production."""
    if not os.path.exists(log_file):
        return None
    
    feedback = {
        'sn_events': [],
        'agb_events': [],
        'metal_injection': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for supernova events
            if 'SN' in line and 'feedback' in line.lower():
                feedback['sn_events'].append(line.strip())
            
            # Look for AGB/stellar wind events
            if 'AGB' in line or 'stellar wind' in line.lower():
                feedback['agb_events'].append(line.strip())
            
            # Look for explicit metal injection messages
            if 'metal' in line.lower() and ('eject' in line.lower() or 'inject' in line.lower()):
                feedback['metal_injection'].append(line.strip())
    
    return feedback

def calculate_expected_yields(total_stellar_mass):
    """Calculate expected metal yields using MESA-calibrated values."""
    
    # SNe (unchanged)
    sn_fraction = 0.007
    metal_per_sn = 0.1  # Msun
    
    # AGB - UPDATED to match MESA table
    agb_fraction = 0.10  # Fraction of stars in AGB mass range
    metal_per_agb = 0.15  # ← UPDATED: Msun per AGB star (MESA-based)
    # This corresponds to ~0.015 per Msun formed (IMF-weighted)
    
    expected_sn_metals = total_stellar_mass * sn_fraction * metal_per_sn
    expected_agb_metals = total_stellar_mass * agb_fraction * metal_per_agb
    total_expected = expected_sn_metals + expected_agb_metals
    
    return {
        'total': total_expected,
        'sn_contribution': expected_sn_metals,
        'agb_contribution': expected_agb_metals
    }

def analyze_metal_spreading(metal_data):
    """Analyze spatial distribution of metals in gas."""
    if 'Gas' not in metal_data['metals']:
        return None
    
    gas = metal_data['metals']['Gas']
    if 'coords' not in gas or 'Z_array' not in gas:
        return None
    
    coords = gas['coords']
    Z = gas['Z_array']
    masses = gas['mass_array']
    
    # Calculate center of mass
    com = np.average(coords, weights=masses, axis=0)
    
    # Calculate distances from COM
    distances = np.sqrt(np.sum((coords - com)**2, axis=1))
    
    # Bin by distance
    r_bins = np.percentile(distances, [0, 25, 50, 75, 90, 100])
    
    spreading = {
        'com': com,
        'r_bins': r_bins,
        'Z_vs_r': []
    }
    
    for i in range(len(r_bins)-1):
        mask = (distances >= r_bins[i]) & (distances < r_bins[i+1])
        if mask.sum() > 0:
            Z_avg = np.average(Z[mask], weights=masses[mask])
            spreading['Z_vs_r'].append({
                'r_min': r_bins[i],
                'r_max': r_bins[i+1],
                'Z_avg': Z_avg,
                'n_particles': mask.sum()
            })
    
    return spreading

def print_header():
    """Print diagnostic header."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           METAL BUDGET DIAGNOSTIC - GADGET-4                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

def print_metal_inventory(snapshots):
    """Print comprehensive metal inventory."""
    print("┌─ METAL INVENTORY ACROSS SNAPSHOTS ────────────────────────────┐")
    print()
    
    evolution = []
    
    for snap in snapshots:
        data = analyze_metal_content(snap)
        if data:
            evolution.append(data)
    
    if not evolution:
        print("  No snapshot data available")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return None
    
    # Print summary table
    print("  z      Total Metals   Gas Metals    Star Metals   Gas/Star Ratio")
    print("  " + "─"*65)
    
    for data in evolution:
        total_metals = sum(m['metal_mass'] for m in data['metals'].values())
        gas_metals = data['metals'].get('Gas', {}).get('metal_mass', 0)
        star_metals = data['metals'].get('Stars', {}).get('metal_mass', 0)
        ratio = gas_metals / star_metals if star_metals > 0 else 0
        
        print(f"  {data['redshift']:5.2f}  {total_metals:12.3e}   {gas_metals:12.3e}   {star_metals:12.3e}   {ratio:6.3f}")
    
    print()
    print("└───────────────────────────────────────────────────────────────┘")
    print()
    
    return evolution

def print_metal_budget_analysis(evolution):
    """Analyze metal budget and conservation."""
    print("┌─ METAL BUDGET ANALYSIS ───────────────────────────────────────┐")
    print()
    
    if len(evolution) < 2:
        print("  Need at least 2 snapshots for budget analysis")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    # Compare first and last snapshots
    initial = evolution[0]
    final = evolution[-1]
    
    # Total metals
    initial_total = sum(m['metal_mass'] for m in initial['metals'].values())
    final_total = sum(m['metal_mass'] for m in final['metals'].values())
    metal_growth = final_total - initial_total
    
    # Stellar mass formed
    initial_stars = initial['metals'].get('Stars', {}).get('total_mass', 0)
    final_stars = final['metals'].get('Stars', {}).get('total_mass', 0)
    stars_formed = final_stars - initial_stars
    
    print(f"  OVERALL BUDGET (z={initial['redshift']:.2f} → z={final['redshift']:.2f}):")
    print(f"    Initial metal mass:     {initial_total:.3e} (code units)")
    print(f"    Final metal mass:       {final_total:.3e} (code units)")
    print(f"    Net metal production:   {metal_growth:.3e} (code units)")
    print()
    
    print(f"  STELLAR FORMATION:")
    print(f"    Initial stellar mass:   {initial_stars:.3e} (code units)")
    print(f"    Final stellar mass:     {final_stars:.3e} (code units)")
    print(f"    Stars formed:           {stars_formed:.3e} (code units)")
    print()
    
    # Calculate expected yields
    expected = calculate_expected_yields(stars_formed)
    
    print(f"  EXPECTED vs ACTUAL YIELDS:")
    print(f"    Expected total metals:  {expected['total']:.3e} (code units)")
    print(f"      → From SNe:           {expected['sn_contribution']:.3e}")
    print(f"      → From AGB:           {expected['agb_contribution']:.3e}")
    print(f"    Actual metal growth:    {metal_growth:.3e}")
    print()
    
    yield_efficiency = metal_growth / expected['total'] if expected['total'] > 0 else 0
    print(f"  YIELD EFFICIENCY: {yield_efficiency*100:.1f}%")
    
    if yield_efficiency < 0.1:
        print(f"    {Colors.FAIL}❌ CRITICAL: Only {yield_efficiency*100:.1f}% of expected metals!{Colors.ENDC}")
        print(f"    {Colors.FAIL}   This suggests feedback is NOT injecting metals properly{Colors.ENDC}")
    elif yield_efficiency < 0.5:
        print(f"    {Colors.WARNING}⚠️  Low efficiency - metals may be escaping or not injecting{Colors.ENDC}")
    elif yield_efficiency < 0.9:
        print(f"    {Colors.OKBLUE}• Moderate efficiency - some metals may be lost{Colors.ENDC}")
    else:
        print(f"    {Colors.OKGREEN}✓ Good metal conservation{Colors.ENDC}")
    
    print()
    
    # Metal distribution
    print(f"  FINAL METAL DISTRIBUTION:")
    for pname, pdata in final['metals'].items():
        metal_mass = pdata['metal_mass']
        fraction = metal_mass / final_total * 100 if final_total > 0 else 0
        print(f"    {pname:12s}: {metal_mass:12.3e} ({fraction:5.1f}%)")
    
    # Check for metal loss
    gas_final = final['metals'].get('Gas', {}).get('metal_mass', 0)
    star_final = final['metals'].get('Stars', {}).get('metal_mass', 0)
    accounted = gas_final + star_final
    
    print()
    print(f"  METAL ACCOUNTING:")
    print(f"    Gas + Stars:            {accounted:.3e}")
    print(f"    Total in snapshot:      {final_total:.3e}")
    if final_total > 0:
        unaccounted = final_total - accounted
        unaccounted_frac = unaccounted / final_total * 100
        print(f"    Unaccounted:            {unaccounted:.3e} ({unaccounted_frac:.1f}%)")
        
        if unaccounted_frac > 10:
            print(f"    {Colors.WARNING}⚠️  Significant metals in other particle types{Colors.ENDC}")
    
    print()
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_gas_phase_metals(latest_data):
    """Analyze metallicity in different gas phases."""
    print("┌─ GAS PHASE METALLICITY ───────────────────────────────────────┐")
    print()
    
    if 'Gas' not in latest_data['metals']:
        print("  No gas data available")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    gas = latest_data['metals']['Gas']
    
    if 'Z_array' not in gas or 'temperature' not in gas:
        print("  Insufficient data for phase analysis")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    Z = gas['Z_array']
    T = gas['temperature']
    masses = gas['mass_array']
    
    # Define phases
    phases = {
        'Cold (T<10⁴ K)': T < 1e4,
        'Cool (10⁴-10⁵ K)': (T >= 1e4) & (T < 1e5),
        'Warm (10⁵-10⁶ K)': (T >= 1e5) & (T < 1e6),
        'Hot (10⁶-10⁷ K)': (T >= 1e6) & (T < 1e7),
        'Very Hot (>10⁷ K)': T >= 1e7
    }
    
    print("  Phase              N_particles    <Z>         Z/Z☉      Metal Mass")
    print("  " + "─"*70)
    
    for phase_name, mask in phases.items():
        if mask.sum() > 0:
            Z_avg = np.average(Z[mask], weights=masses[mask])
            metal_mass = np.sum(Z[mask] * masses[mask])
            print(f"  {phase_name:18s} {mask.sum():8,d}    {Z_avg:8.6f}   {Z_avg/Z_SOLAR:5.2f}    {metal_mass:10.3e}")
    
    print()
    
    # Check for hot gas enrichment (signature of feedback)
    hot_mask = T >= 1e6
    cold_mask = T < 1e4
    
    if hot_mask.sum() > 0 and cold_mask.sum() > 0:
        Z_hot = np.average(Z[hot_mask], weights=masses[hot_mask])
        Z_cold = np.average(Z[cold_mask], weights=masses[cold_mask])
        enrichment_ratio = Z_hot / Z_cold if Z_cold > 0 else 0
        
        print(f"  HOT/COLD ENRICHMENT RATIO: {enrichment_ratio:.2f}")
        if enrichment_ratio > 2.0:
            print(f"    {Colors.OKGREEN}✓ Hot gas significantly enriched - feedback working!{Colors.ENDC}")
        elif enrichment_ratio > 1.2:
            print(f"    {Colors.OKBLUE}• Moderate hot gas enrichment{Colors.ENDC}")
        else:
            print(f"    {Colors.WARNING}⚠️  Hot gas not enriched - feedback may not be injecting metals{Colors.ENDC}")
    
    print()
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_spatial_analysis(latest_data):
    """Analyze spatial distribution of metals."""
    print("┌─ SPATIAL METAL DISTRIBUTION ──────────────────────────────────┐")
    print()
    
    spreading = analyze_metal_spreading(latest_data)
    
    if spreading is None:
        print("  Insufficient data for spatial analysis")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    print("  Metallicity vs Distance from Center:")
    print("  " + "─"*60)
    print("  R_min      R_max      <Z>         Z/Z☉      N_particles")
    print("  " + "─"*60)
    
    for zone in spreading['Z_vs_r']:
        print(f"  {zone['r_min']:8.2f}   {zone['r_max']:8.2f}   {zone['Z_avg']:8.6f}   {zone['Z_avg']/Z_SOLAR:5.2f}      {zone['n_particles']:8,d}")
    
    print()
    
    # Check if metals are concentrated or spread
    Z_values = [z['Z_avg'] for z in spreading['Z_vs_r']]
    if len(Z_values) > 1:
        Z_variation = np.std(Z_values) / np.mean(Z_values)
        print(f"  Metallicity variation (σ/μ): {Z_variation:.2f}")
        
        if Z_variation > 0.5:
            print(f"    {Colors.WARNING}⚠️  High variation - metals not well mixed{Colors.ENDC}")
            print(f"    → Metals may be concentrated near formation sites")
        else:
            print(f"    {Colors.OKGREEN}✓ Metals relatively well distributed{Colors.ENDC}")
    
    print()
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_diagnostic_summary(evolution, feedback_data):
    """Print summary of diagnostics and recommendations."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  DIAGNOSTIC SUMMARY & RECOMMENDATIONS                          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    if not evolution or len(evolution) < 2:
        print("  Insufficient data for diagnosis")
        return
    
    initial = evolution[0]
    final = evolution[-1]
    
    initial_total = sum(m['metal_mass'] for m in initial['metals'].values())
    final_total = sum(m['metal_mass'] for m in final['metals'].values())
    metal_growth = final_total - initial_total
    
    initial_stars = initial['metals'].get('Stars', {}).get('total_mass', 0)
    final_stars = final['metals'].get('Stars', {}).get('total_mass', 0)
    stars_formed = final_stars - initial_stars
    
    expected = calculate_expected_yields(stars_formed)
    yield_efficiency = metal_growth / expected['total'] if expected['total'] > 0 else 0
    
    issues = []
    recommendations = []
    
    # Check 1: Yield efficiency
    if yield_efficiency < 0.1:
        issues.append("❌ CRITICAL: Less than 10% of expected metal production")
        recommendations.append("Check Config.sh for yield/enrichment flags (GALSF_FB_FIRE_RT_YIELD, etc)")
        recommendations.append("Verify feedback module is computing metal yields")
        recommendations.append("Check if metals are being injected into neighboring gas particles")
    elif yield_efficiency < 0.5:
        issues.append("⚠️  Low metal yield efficiency (<50%)")
        recommendations.append("Metals may be escaping in outflows - check wind particles")
        recommendations.append("Verify SPH kernel for metal distribution")
    
    # Check 2: Gas/Star metal ratio
    gas_metals = final['metals'].get('Gas', {}).get('metal_mass', 0)
    star_metals = final['metals'].get('Stars', {}).get('metal_mass', 0)
    
    if star_metals > 0:
        gas_star_ratio = gas_metals / star_metals
        if gas_star_ratio < 0.5:
            issues.append("⚠️  Very little metal in gas compared to stars")
            recommendations.append("Metals not returning to ISM - check feedback mass/energy return")
            recommendations.append("May need to increase feedback coupling efficiency")
    
    # Check 3: Feedback events
    if feedback_data:
        n_sn = len(feedback_data['sn_events'])
        n_metal_inject = len(feedback_data['metal_injection'])
        
        if n_sn > 0 and n_metal_inject == 0:
            issues.append("⚠️  SN events found but no explicit metal injection messages")
            recommendations.append("Add diagnostic output to feedback module to confirm metal ejection")
    
    # Print findings
    if issues:
        print(f"  {Colors.BOLD}ISSUES DETECTED:{Colors.ENDC}")
        for issue in issues:
            print(f"    {issue}")
        print()
    
    if recommendations:
        print(f"  {Colors.BOLD}RECOMMENDATIONS:{Colors.ENDC}")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
        print()
    
    if not issues:
        print(f"  {Colors.OKGREEN}✓ Metal budget appears reasonable{Colors.ENDC}")
        print()
    
    # Specific checks
    print(f"  {Colors.BOLD}KEY METRICS:{Colors.ENDC}")
    print(f"    Yield efficiency:     {yield_efficiency*100:5.1f}%")
    print(f"    Gas/Star metal ratio: {gas_star_ratio:.2f}" if star_metals > 0 else "    Gas/Star metal ratio: N/A")
    print(f"    Metal growth:         {metal_growth:.3e} code units")
    print()

def create_plots(evolution, output_dir):
    """Create diagnostic plots."""
    if len(evolution) < 2:
        return
    
    print("┌─ CREATING DIAGNOSTIC PLOTS ───────────────────────────────────┐")
    
    # Extract data
    redshifts = [d['redshift'] for d in evolution]
    times = [d['time'] for d in evolution]
    
    gas_metals = [d['metals'].get('Gas', {}).get('metal_mass', 0) for d in evolution]
    star_metals = [d['metals'].get('Stars', {}).get('metal_mass', 0) for d in evolution]
    total_metals = [gas + star for gas, star in zip(gas_metals, star_metals)]
    
    gas_mass = [d['metals'].get('Gas', {}).get('total_mass', 0) for d in evolution]
    star_mass = [d['metals'].get('Stars', {}).get('total_mass', 0) for d in evolution]
    
    # Calculate mass-weighted metallicities
    gas_Z = [m / M if M > 0 else 0 for m, M in zip(gas_metals, gas_mass)]
    star_Z = [m / M if M > 0 else 0 for m, M in zip(star_metals, star_mass)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Metal mass evolution
    ax = axes[0, 0]
    ax.plot(redshifts, total_metals, 'k-', label='Total', linewidth=2)
    ax.plot(redshifts, gas_metals, 'b-', label='Gas', linewidth=1.5)
    ax.plot(redshifts, star_metals, 'r-', label='Stars', linewidth=1.5)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Metal Mass (code units)')
    ax.set_yscale('log')
    ax.set_xlim(max(redshifts), min(redshifts))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Metal Mass Evolution')
    
    # Plot 2: Metallicity evolution
    ax = axes[0, 1]
    ax.plot(redshifts, np.array(gas_Z)/Z_SOLAR, 'b-', label='Gas', linewidth=1.5)
    ax.plot(redshifts, np.array(star_Z)/Z_SOLAR, 'r-', label='Stars', linewidth=1.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Solar')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Metallicity (Z☉)')
    ax.set_xlim(max(redshifts), min(redshifts))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Metallicity Evolution')
    
    # Plot 3: Gas/Star metal ratio
    ax = axes[1, 0]
    ratios = [g/s if s > 0 else 0 for g, s in zip(gas_metals, star_metals)]
    ax.plot(redshifts, ratios, 'g-', linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Gas/Star Metal Ratio')
    ax.set_xlim(max(redshifts), min(redshifts))
    ax.grid(True, alpha=0.3)
    ax.set_title('Metal Distribution Ratio')
    
    # Plot 4: Stellar mass and metals
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(redshifts, star_mass, 'r-', label='Stellar Mass', linewidth=2)
    ax2.plot(redshifts, star_metals, 'orange', linestyle='--', label='Metals in Stars', linewidth=2)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Stellar Mass (code units)', color='r')
    ax2.set_ylabel('Metal Mass (code units)', color='orange')
    ax.set_xlim(max(redshifts), min(redshifts))
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stellar Mass & Metals')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'metal_budget_diagnostic.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'output.log'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    print_header()
    
    # Find snapshots
    snapshots = find_snapshots(output_dir)
    
    if not snapshots:
        print(f"{Colors.FAIL}No snapshots found in {output_dir}{Colors.ENDC}")
        return
    
    print(f"Found {len(snapshots)} snapshots")
    print()
    
    # Parse log file
    feedback_data = parse_feedback_events(log_file)
    
    # Analyze snapshots
    evolution = print_metal_inventory(snapshots)
    
    if evolution:
        print_metal_budget_analysis(evolution)
        print_gas_phase_metals(evolution[-1])
        print_spatial_analysis(evolution[-1])
        print_diagnostic_summary(evolution, feedback_data)
        
        # Create plots
        try:
            create_plots(evolution, output_dir)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    print(f"Rerun with: python {sys.argv[0]} {log_file} {output_dir}")
    print()

if __name__ == '__main__':
    main()
