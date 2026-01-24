#!/usr/bin/env python3
"""
Analyze dust destruction statistics from Gadget-4 simulation log

Extracts and reports on:
- Total thermal sputtering events
- Total shock destruction events
- Dust mass evolution
- Destruction rates vs time
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import sys

def parse_dust_destruction_events(logfile):
    """
    Parse all dust destruction events from log
    
    Returns:
        dict with thermal and shock destruction data
    """
    
    destruction_data = {
        'thermal': [],
        'shock': [],
        'thermal_count': 0,
        'shock_count': 0
    }
    
    # [SPUTTERING] Dust X.XXe-XX Msun → Gas, Z: X.XXXX → X.XXXX (T=XXXXXX K)
    thermal_pattern = r'\[SPUTTERING\] Dust ([\d.e+-]+) Msun.*T=([\d.e+-]+) K'
    
    # [SHOCK_DESTRUCTION] Dust X.XXe-XX Msun → Gas
    shock_pattern = r'\[SHOCK_DESTRUCTION\] Dust ([\d.e+-]+) Msun'
    
    with open(logfile, 'r') as f:
        for line in f:
            # Thermal sputtering
            if '[SPUTTERING]' in line:
                destruction_data['thermal_count'] += 1
                match = re.search(thermal_pattern, line)
                if match:
                    dust_mass = float(match.group(1))
                    temperature = float(match.group(2))
                    destruction_data['thermal'].append({
                        'mass': dust_mass,
                        'temperature': temperature
                    })
            
            # Shock destruction
            if '[SHOCK_DESTRUCTION]' in line:
                destruction_data['shock_count'] += 1
                match = re.search(shock_pattern, line)
                if match:
                    dust_mass = float(match.group(1))
                    destruction_data['shock'].append({
                        'mass': dust_mass
                    })
    
    return destruction_data


def parse_dust_statistics(logfile):
    """
    Parse DUST_TRACK lines to get overall dust evolution
    """
    
    data = {
        'z': [],
        'M_dust': [],
        'N_dust': [],
        'M_metal': []
    }
    
    pattern = r'DUST_TRACK: a=[\d.]+ z=([\d.]+): M_dust=([\d.e+-]+) Msun \((\d+)\), M_metal=([\d.e+-]+) Msun'
    
    with open(logfile, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data['z'].append(float(match.group(1)))
                data['M_dust'].append(float(match.group(2)))
                data['N_dust'].append(int(match.group(3)))
                data['M_metal'].append(float(match.group(4)))
    
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def parse_dust_module_statistics(logfile):
    """
    Look for module-level statistics at end of log
    
    These are the global counters: NDustCreated, NDustDestroyed, etc.
    """
    
    stats = {
        'dust_created': None,
        'dust_destroyed': None,
        'destroyed_by_thermal': None,
        'destroyed_by_shock': None
    }
    
    # Look in last 1000 lines for final statistics
    with open(logfile, 'r') as f:
        lines = f.readlines()
    
    # Search backwards through last lines
    for line in lines[-1000:]:
        if 'NDustCreated' in line or 'Dust created' in line.lower():
            match = re.search(r'(\d+)', line)
            if match and stats['dust_created'] is None:
                stats['dust_created'] = int(match.group(1))
        
        if 'NDustDestroyed' in line or 'Dust destroyed' in line.lower():
            match = re.search(r'(\d+)', line)
            if match and stats['dust_destroyed'] is None:
                stats['dust_destroyed'] = int(match.group(1))
        
        if 'NDustDestroyedByThermal' in line or 'thermal' in line.lower():
            match = re.search(r'(\d+)', line)
            if match and stats['destroyed_by_thermal'] is None:
                stats['destroyed_by_thermal'] = int(match.group(1))
        
        if 'NDustDestroyedByShock' in line or 'shock' in line.lower():
            match = re.search(r'(\d+)', line)
            if match and stats['destroyed_by_shock'] is None:
                stats['destroyed_by_shock'] = int(match.group(1))
    
    return stats


def analyze_destruction_statistics(logfile):
    """
    Main analysis function
    """
    
    print("="*70)
    print("DUST DESTRUCTION ANALYSIS")
    print("="*70)
    print(f"\nAnalyzing: {logfile}\n")
    
    # Parse destruction events
    print("Parsing destruction events...")
    destruction = parse_dust_destruction_events(logfile)
    
    # Parse dust evolution
    print("Parsing dust evolution...")
    evolution = parse_dust_statistics(logfile)
    
    # Parse module statistics
    print("Parsing module statistics...")
    module_stats = parse_dust_module_statistics(logfile)
    
    # ========== REPORT ==========
    print("\n" + "="*70)
    print("LOGGED DESTRUCTION EVENTS")
    print("="*70)
    
    print(f"\nThermal Sputtering:")
    print(f"  Logged events: {len(destruction['thermal'])}")
    print(f"  Total mentions: {destruction['thermal_count']}")
    
    if len(destruction['thermal']) > 0:
        thermal_masses = [e['mass'] for e in destruction['thermal']]
        thermal_temps = [e['temperature'] for e in destruction['thermal']]
        print(f"  Mass range: {min(thermal_masses):.2e} - {max(thermal_masses):.2e} Msun")
        print(f"  Avg mass: {np.mean(thermal_masses):.2e} Msun")
        print(f"  Temperature range: {min(thermal_temps):.2e} - {max(thermal_temps):.2e} K")
        print(f"  Avg temperature: {np.mean(thermal_temps)/1e6:.1f} million K")
        
        # Check if all masses are identical
        if len(set([f"{m:.3e}" for m in thermal_masses])) == 1:
            print(f"  ⚠️  WARNING: All logged masses are identical ({thermal_masses[0]:.2e} Msun)")
            print(f"      This might be MIN_DUST_PARTICLE_MASS threshold")
    
    print(f"\nShock Destruction:")
    print(f"  Logged events: {len(destruction['shock'])}")
    print(f"  Total mentions: {destruction['shock_count']}")
    
    if len(destruction['shock']) > 0:
        shock_masses = [e['mass'] for e in destruction['shock']]
        print(f"  Mass range: {min(shock_masses):.2e} - {max(shock_masses):.2e} Msun")
        print(f"  Avg mass: {np.mean(shock_masses):.2e} Msun")
    
    # ========== MODULE STATISTICS ==========
    print("\n" + "="*70)
    print("MODULE-LEVEL STATISTICS")
    print("="*70)
    
    if any(v is not None for v in module_stats.values()):
        for key, value in module_stats.items():
            if value is not None:
                print(f"  {key}: {value}")
    else:
        print("  No module statistics found in log")
        print("  (These are usually printed at end of simulation or in summary)")
    
    # ========== DUST EVOLUTION ==========
    print("\n" + "="*70)
    print("DUST EVOLUTION")
    print("="*70)
    
    if len(evolution['z']) > 0:
        print(f"\nRedshift range: z={evolution['z'].max():.2f} → z={evolution['z'].min():.2f}")
        print(f"\nInitial state (z={evolution['z'][0]:.2f}):")
        print(f"  Dust particles: {evolution['N_dust'][0]:,}")
        print(f"  Dust mass: {evolution['M_dust'][0]:.2e} Msun")
        
        print(f"\nFinal state (z={evolution['z'][-1]:.2f}):")
        print(f"  Dust particles: {evolution['N_dust'][-1]:,}")
        print(f"  Dust mass: {evolution['M_dust'][-1]:.2e} Msun")
        
        # Calculate changes
        dust_mass_change = evolution['M_dust'][-1] - evolution['M_dust'][0]
        particle_change = evolution['N_dust'][-1] - evolution['N_dust'][0]
        
        print(f"\nNet change:")
        print(f"  Dust mass: {dust_mass_change:+.2e} Msun ({dust_mass_change/evolution['M_dust'][0]*100:+.1f}%)")
        print(f"  Particles: {particle_change:+,}")
        
        # Find peak dust mass
        peak_idx = np.argmax(evolution['M_dust'])
        peak_z = evolution['z'][peak_idx]
        peak_mass = evolution['M_dust'][peak_idx]
        
        if peak_idx > 0 and peak_idx < len(evolution['z']) - 1:
            print(f"\nPeak dust mass:")
            print(f"  {peak_mass:.2e} Msun at z={peak_z:.2f}")
            print(f"  Declined by {(peak_mass - evolution['M_dust'][-1])/peak_mass*100:.1f}% since peak")
    else:
        print("  No DUST_TRACK data found")
    
    # ========== DESTRUCTION EFFICIENCY ==========
    print("\n" + "="*70)
    print("DESTRUCTION EFFICIENCY")
    print("="*70)
    
    if len(evolution['N_dust']) > 0 and destruction['thermal_count'] > 0:
        # Estimate destruction rate
        avg_dust_count = np.mean(evolution['N_dust'])
        thermal_fraction = destruction['thermal_count'] / avg_dust_count * 100
        
        print(f"\nThermal sputtering:")
        print(f"  {destruction['thermal_count']} events / {avg_dust_count:.0f} avg particles")
        print(f"  = {thermal_fraction:.2f}% of dust population destroyed")
        
        if thermal_fraction < 1:
            print(f"  ✓ Reasonable (most dust survives in cool ISM)")
        elif thermal_fraction < 10:
            print(f"  ✓ Moderate destruction (hot halo/outflows)")
        else:
            print(f"  ⚠️  High destruction rate (check parameters)")
    
    # ========== INTERPRETATION ==========
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    print("\n📊 What the numbers mean:")
    
    # Check if logging is limited
    if destruction['thermal_count'] == 20:
        print("\n⚠️  IMPORTANT: Exactly 20 thermal events logged!")
        print("   Your code has this limit:")
        print("   if(thermal_destruction_count <= 20 && All.ThisTask == 0)")
        print("\n   This means logging stopped after 20 events.")
        print("   Actual destruction events could be much higher!")
        print("\n   To see total: Look for module statistics or DUST_TRACK particle counts")
    
    # Mass analysis
    if len(destruction['thermal']) > 0:
        thermal_masses = [e['mass'] for e in destruction['thermal']]
        if all(abs(m - thermal_masses[0]) < 1e-10 for m in thermal_masses):
            print(f"\n🔍 All destroyed dust has identical mass: {thermal_masses[0]:.2e} Msun")
            print("   Possible explanations:")
            print("   1. MIN_DUST_PARTICLE_MASS threshold (1e-6 in your code)")
            print("   2. Particles at minimum mass get destroyed first")
            print("   3. This is actually expected behavior!")
    
    # Survival rate
    if len(evolution['N_dust']) > 0:
        survival_rate = evolution['N_dust'][-1] / max(evolution['N_dust']) * 100
        print(f"\n✓ Particle survival: {survival_rate:.1f}% of peak count")
        
        if survival_rate > 80:
            print("   → Most dust survives (disk-dominated, cool ISM)")
        elif survival_rate > 50:
            print("   → Moderate destruction (hot halo + cool disk)")
        else:
            print("   → Heavy destruction (hot dominated or many shocks)")
    
    print("\n" + "="*70)
    
    # ========== PLOT ==========
    if len(evolution['z']) > 0:
        create_destruction_plots(evolution, destruction, output_file='dust_destruction_analysis.png')


def create_destruction_plots(evolution, destruction, output_file='dust_destruction_analysis.png'):
    """
    Create diagnostic plots
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dust Destruction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Dust mass evolution
    ax = axes[0, 0]
    ax.semilogy(evolution['z'], evolution['M_dust'], 'b-', linewidth=2)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Dust Mass (Msun)')
    ax.set_title('Dust Mass Evolution')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Plot 2: Particle count evolution
    ax = axes[0, 1]
    ax.plot(evolution['z'], evolution['N_dust'], 'r-', linewidth=2)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number of Dust Particles')
    ax.set_title('Dust Particle Count')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Plot 3: Thermal destruction temperature distribution
    ax = axes[1, 0]
    if len(destruction['thermal']) > 0:
        temps = [e['temperature']/1e6 for e in destruction['thermal']]
        ax.hist(temps, bins=15, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(10, color='red', linestyle='--', linewidth=2, label='10 million K')
        ax.set_xlabel('Temperature (million K)')
        ax.set_ylabel('Number of Events')
        ax.set_title(f'Thermal Sputtering Temperatures ({len(temps)} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No thermal destruction data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Thermal Sputtering Temperatures')
    
    # Plot 4: Mass distribution of destroyed dust
    ax = axes[1, 1]
    if len(destruction['thermal']) > 0:
        masses = [e['mass']*1e6 for e in destruction['thermal']]  # Convert to 1e-6 Msun units
        ax.hist(masses, bins=15, color='brown', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Dust Mass (10⁻⁶ Msun)')
        ax.set_ylabel('Number of Events')
        ax.set_title(f'Mass of Destroyed Dust ({len(masses)} events)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No destruction mass data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Mass of Destroyed Dust')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostic plots: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dust_destruction.py <logfile>")
        print("\nExample:")
        print("  python analyze_dust_destruction.py ../output_zoom_512_20251216_141152.log")
        sys.exit(1)
    
    logfile = sys.argv[1]
    
    if not Path(logfile).exists():
        print(f"Error: Log file not found: {logfile}")
        sys.exit(1)
    
    analyze_destruction_statistics(logfile)


if __name__ == "__main__":
    main()
