#!/usr/bin/env python3
"""
Analyze metal and dust production from Gadget-4 simulation logs

This script parses DUST_TRACK, SNII_EVENT, and AGB_DEBUG outputs to compare
metal production from Type II SNe vs AGB stars and track dust evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

def parse_dust_track(logfile):
    """
    Parse DUST_TRACK lines from simulation log
    
    Format: DUST_TRACK: a=X.XXXX z=X.XX: M_dust=X.XXXe+XX Msun (N), M_metal=X.XXXe+XX Msun, D/Z=X.XXXX, D/G=X.XXXXe+XX
    
    Returns:
        dict with arrays: a, z, M_dust, N_dust, M_metal, D_Z, D_G
    """
    
    data = {
        'a': [],
        'z': [],
        'M_dust': [],
        'N_dust': [],
        'M_metal': [],
        'D_Z': [],
        'D_G': []
    }
    
    pattern = r'DUST_TRACK: a=([\d.]+) z=([\d.]+): M_dust=([\d.e+-]+) Msun \((\d+)\), M_metal=([\d.e+-]+) Msun, D/Z=([\d.e+-]+), D/G=([\d.e+-]+)'
    
    with open(logfile, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data['a'].append(float(match.group(1)))
                data['z'].append(float(match.group(2)))
                data['M_dust'].append(float(match.group(3)))
                data['N_dust'].append(int(match.group(4)))
                data['M_metal'].append(float(match.group(5)))
                data['D_Z'].append(float(match.group(6)))
                data['D_G'].append(float(match.group(7)))
    
    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"Parsed {len(data['z'])} DUST_TRACK entries")
    print(f"  Redshift range: z={data['z'].max():.2f} to z={data['z'].min():.2f}")
    print(f"  Final metal mass: {data['M_metal'][-1]:.2e} Msun")
    print(f"  Final dust mass: {data['M_dust'][-1]:.2e} Msun")
    print(f"  Final D/Z: {data['D_Z'][-1]:.4f}")
    
    return data


def parse_feedback_events(logfile):
    """
    Parse SNII_EVENT and AGB_DEBUG lines to track metal production sources
    
    Returns:
        dict with SN and AGB event lists
    """
    
    sn_events = []
    agb_events = []
    
    # [FEEDBACK|...] [SNII_EVENT] #10: Star 3164, age=19.4 Myr, M=1.276e+07 Msun
    sn_pattern = r'\[SNII_EVENT\] #(\d+): Star (\d+), age=([\d.]+) Myr, M=([\d.e+-]+) Msun'
    
    # [FEEDBACK|...] [AGB_DEBUG|T=0] Star 1234: age=150.2 Myr, M=1.28e+07 Msun, Z=0.0200 | MZ=1.27e+05 Msun (1.27e-04 code) | E=1.27e+52 erg
    agb_pattern = r'\[AGB_DEBUG.*\] Star (\d+): age=([\d.]+) Myr, M=([\d.e+-]+) Msun, Z=([\d.]+) \| MZ=([\d.e+-]+) Msun'
    
    with open(logfile, 'r') as f:
        for line in f:
            sn_match = re.search(sn_pattern, line)
            if sn_match:
                sn_events.append({
                    'event_num': int(sn_match.group(1)),
                    'star_id': int(sn_match.group(2)),
                    'age_Myr': float(sn_match.group(3)),
                    'M_star': float(sn_match.group(4))
                })
            
            agb_match = re.search(agb_pattern, line)
            if agb_match:
                agb_events.append({
                    'star_id': int(agb_match.group(1)),
                    'age_Myr': float(agb_match.group(2)),
                    'M_star': float(agb_match.group(3)),
                    'Z': float(agb_match.group(4)),
                    'MZ': float(agb_match.group(5))
                })
    
    print(f"\nParsed feedback events:")
    print(f"  Type II SNe: {len(sn_events)} events")
    if sn_events:
        total_sn_metals = sum([e['M_star'] * 0.017 for e in sn_events])  # 1.7% yield
        print(f"    Total SN metals: {total_sn_metals:.2e} Msun")
    
    print(f"  AGB stars: {len(agb_events)} events")
    if agb_events:
        total_agb_metals = sum([e['MZ'] for e in agb_events])
        print(f"    Total AGB metals: {total_agb_metals:.2e} Msun")
        
        if sn_events:
            print(f"    Ratio (AGB/SN): {total_agb_metals/total_sn_metals:.3f}")
    
    return {'sn': sn_events, 'agb': agb_events}


def plot_evolution(data, output_dir='.'):
    """Create comprehensive evolution plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metal and Dust Evolution', fontsize=16, fontweight='bold')
    
    # Convert redshift to lookback time (approximate)
    # t_lookback ≈ 13.8 Gyr × (1 - a) for low-z
    H0 = 67.4  # km/s/Mpc
    t_lookback = 13.8 * (1 - data['a'])  # Gyr (approximate)
    
    # Plot 1: Metal mass evolution
    ax = axes[0, 0]
    ax.semilogy(data['z'], data['M_metal'], 'b-', linewidth=2, label='Total metals')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Metal Mass (Msun)', fontsize=12)
    ax.set_title('Metal Production', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()
    
    # Expected metal yield markers
    # If M_star_total = 1e9 Msun by z=0:
    # SNe: 1.7% = 1.7e7 Msun
    # AGB: 1.0% = 1.0e7 Msun
    # Total: 2.7e7 Msun
    
    # Plot 2: Dust mass evolution
    ax = axes[0, 1]
    ax.semilogy(data['z'], data['M_dust'], 'r-', linewidth=2, label='Dust mass')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Dust Mass (Msun)', fontsize=12)
    ax.set_title('Dust Production', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Mark when dust first appears
    if np.any(data['M_dust'] > 0):
        first_dust_idx = np.where(data['M_dust'] > 0)[0][0]
        z_first = data['z'][first_dust_idx]
        ax.axvline(z_first, color='g', linestyle='--', alpha=0.5, 
                  label=f'First dust: z={z_first:.2f}')
        ax.legend()
    
    # Plot 3: Dust-to-metal ratio
    ax = axes[1, 0]
    ax.plot(data['z'], data['D_Z'], 'purple', linewidth=2, label='D/Z ratio')
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='MW disk (~0.3)')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Theoretical max (~0.5)')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Dust-to-Metal Ratio (D/Z)', fontsize=12)
    ax.set_title('Dust Depletion', fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    
    # Plot 4: Dust-to-gas ratio
    ax = axes[1, 1]
    ax.semilogy(data['z'], data['D_G'], 'teal', linewidth=2, label='D/G ratio')
    ax.axhline(0.01, color='orange', linestyle='--', alpha=0.7, label='MW disk (~1%)')
    ax.axhline(0.001, color='blue', linestyle='--', alpha=0.7, label='MW halo (~0.1%)')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Dust-to-Gas Ratio (D/G)', fontsize=12)
    ax.set_title('Dust Enrichment', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'metal_dust_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved evolution plot: {output_path}")
    plt.close()


def print_summary_statistics(data, events):
    """Print comprehensive summary statistics"""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Final state
    print("\n--- Final State (z={:.2f}) ---".format(data['z'][-1]))
    print(f"  Total metals:     {data['M_metal'][-1]:.2e} Msun")
    print(f"  Total dust:       {data['M_dust'][-1]:.2e} Msun")
    print(f"  Dust particles:   {data['N_dust'][-1]:,}")
    print(f"  D/Z ratio:        {data['D_Z'][-1]:.4f}")
    print(f"  D/G ratio:        {data['D_G'][-1]:.4e}")
    
    # Dust formation history
    if np.any(data['M_dust'] > 0):
        first_dust_idx = np.where(data['M_dust'] > 0)[0][0]
        z_first = data['z'][first_dust_idx]
        M_first = data['M_dust'][first_dust_idx]
        print(f"\n--- Dust Formation ---")
        print(f"  First dust at:    z = {z_first:.2f}")
        print(f"  Initial mass:     {M_first:.2e} Msun")
        print(f"  Growth factor:    {data['M_dust'][-1]/M_first:.1f}×")
    else:
        print("\n  WARNING: No dust formed!")
    
    # Feedback contributions
    if events['sn'] or events['agb']:
        print("\n--- Feedback Sources ---")
        
        if events['sn']:
            n_sn = len(events['sn'])
            avg_sn_metals = sum([e['M_star'] * 0.017 for e in events['sn']]) / n_sn
            print(f"  Type II SNe:      {n_sn} events")
            print(f"    Avg metals/SN:  {avg_sn_metals:.2e} Msun")
        
        if events['agb']:
            n_agb = len(events['agb'])
            avg_agb_metals = sum([e['MZ'] for e in events['agb']]) / n_agb
            print(f"  AGB stars:        {n_agb} events")
            print(f"    Avg metals/AGB: {avg_agb_metals:.2e} Msun")
        
        # Compare to final metal mass
        sn_metals_logged = sum([e['M_star'] * 0.017 for e in events['sn']])
        agb_metals_logged = sum([e['MZ'] for e in events['agb']])
        total_logged = sn_metals_logged + agb_metals_logged
        
        if total_logged > 0:
            print(f"\n  Fraction from SNe:  {sn_metals_logged/total_logged*100:.1f}%")
            print(f"  Fraction from AGB:  {agb_metals_logged/total_logged*100:.1f}%")
    
    # Observational comparison
    print("\n--- Observational Comparison ---")
    final_DZ = data['D_Z'][-1]
    final_DG = data['D_G'][-1]
    
    print(f"  D/Z = {final_DZ:.4f}")
    if final_DZ < 0.1:
        print("    → Low (expect 0.3-0.5 in MW disk)")
    elif final_DZ < 0.3:
        print("    → Moderate (dwarf galaxy range)")
    elif final_DZ < 0.5:
        print("    → Good (MW disk range)")
    else:
        print("    → High (above typical values)")
    
    print(f"  D/G = {final_DG:.4e}")
    if final_DG < 1e-4:
        print("    → Very low (CGM-like)")
    elif final_DG < 1e-3:
        print("    → Low (MW halo)")
    elif final_DG < 1e-2:
        print("    → Moderate (outskirts)")
    else:
        print("    → Good (MW disk-like)")
    
    print("="*70)


def compare_runs(old_log, new_log, output_dir='.'):
    """Compare two simulation runs (before/after AGB fix)"""
    
    print("Comparing simulation runs...")
    print(f"  OLD: {old_log}")
    print(f"  NEW: {new_log}")
    
    old_data = parse_dust_track(old_log)
    new_data = parse_dust_track(new_log)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Comparison: Before vs After AGB Fix', fontsize=16, fontweight='bold')
    
    # Metal mass
    ax = axes[0]
    ax.semilogy(old_data['z'], old_data['M_metal'], 'b--', linewidth=2, label='Before (old AGB)')
    ax.semilogy(new_data['z'], new_data['M_metal'], 'b-', linewidth=2, label='After (fixed AGB)')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Metal Mass (Msun)', fontsize=12)
    ax.set_title('Metal Production', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()
    
    # Dust mass
    ax = axes[1]
    ax.semilogy(old_data['z'], old_data['M_dust'], 'r--', linewidth=2, label='Before')
    ax.semilogy(new_data['z'], new_data['M_dust'], 'r-', linewidth=2, label='After')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Dust Mass (Msun)', fontsize=12)
    ax.set_title('Dust Production', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()
    
    # D/Z ratio
    ax = axes[2]
    ax.plot(old_data['z'], old_data['D_Z'], 'g--', linewidth=2, label='Before')
    ax.plot(new_data['z'], new_data['D_Z'], 'g-', linewidth=2, label='After')
    ax.axhline(0.3, color='orange', linestyle=':', alpha=0.7, label='MW disk')
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Dust-to-Metal Ratio (D/Z)', fontsize=12)
    ax.set_title('Dust Depletion', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'agb_fix_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    
    # Print improvement statistics
    print("\n" + "="*70)
    print("IMPROVEMENT METRICS")
    print("="*70)
    
    old_final = old_data['M_metal'][-1]
    new_final = new_data['M_metal'][-1]
    improvement = (new_final - old_final) / old_final * 100
    
    print(f"\nMetal mass (final):")
    print(f"  Before: {old_final:.2e} Msun")
    print(f"  After:  {new_final:.2e} Msun")
    print(f"  Change: {improvement:+.1f}%")
    
    old_dust = old_data['M_dust'][-1]
    new_dust = new_data['M_dust'][-1]
    dust_improvement = (new_dust - old_dust) / old_dust * 100 if old_dust > 0 else float('inf')
    
    print(f"\nDust mass (final):")
    print(f"  Before: {old_dust:.2e} Msun")
    print(f"  After:  {new_dust:.2e} Msun")
    if dust_improvement != float('inf'):
        print(f"  Change: {dust_improvement:+.1f}%")
    else:
        print(f"  Change: New dust production enabled!")
    
    print("="*70)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_metals.py <logfile> [old_logfile_for_comparison]")
        print("\nExamples:")
        print("  python analyze_metals.py output.log")
        print("  python analyze_metals.py new_output.log old_output.log")
        sys.exit(1)
    
    logfile = sys.argv[1]
    
    # Parse data
    data = parse_dust_track(logfile)
    events = parse_feedback_events(logfile)
    
    # Create plots
    plot_evolution(data)
    
    # Print summary
    print_summary_statistics(data, events)
    
    # Compare to old run if provided
    if len(sys.argv) > 2:
        old_logfile = sys.argv[2]
        compare_runs(old_logfile, logfile)


if __name__ == "__main__":
    main()
