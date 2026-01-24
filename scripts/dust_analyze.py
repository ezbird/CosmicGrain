#!/usr/bin/env python3
"""
Gadget-4 Cosmological Simulation Dust Diagnostics Plotter

This script parses dust output from Gadget-4 simulations and creates
diagnostic plots for dust creation, destruction, and evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

class DustParser:
    """Parse Gadget-4 dust output and extract relevant data."""
    
    def __init__(self):
        self.dust_creation_events = []
        self.dust_destruction_events = []
        self.dust_drag_events = []
        self.dust_stats = []
        self.dust_coupling = []
        self.dust_positions = []
        self.timestep_counter = 0
        self.current_timestep = None
        
    def parse_file(self, filename):
        """Parse the dust output file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if '[DUST_DEBUG]' in line:
                self._parse_dust_debug(line)
            elif '[DUST_POSITION]' in line:
                self._parse_dust_position(line)
            elif '[DUST_STATS]' in line:
                self._parse_dust_stats(line)
            elif '[DUST_COUPLING]' in line:
                self._parse_dust_coupling(line)
                
    def _parse_dust_debug(self, line):
        """Parse dust creation debug lines."""
        # Parse AGB wind dust creation
        if 'AGB wind from star' in line:
            match = re.search(r'star (\d+): Creating (\d+) dust particles, ([\d.e+-]+) mass each', line)
            if match:
                self.dust_creation_events.append({
                    'star_id': int(match.group(1)),
                    'n_particles': int(match.group(2)),
                    'mass_per_particle': float(match.group(3)),
                    'source': 'AGB',
                    'timestep': self.timestep_counter
                })
        
        # Parse Type II dust creation
        elif 'Type II SN from star' in line:
            match = re.search(r'star (\d+): Creating (\d+) dust particles, ([\d.e+-]+) mass each', line)
            if match:
                self.dust_creation_events.append({
                    'star_id': int(match.group(1)),
                    'n_particles': int(match.group(2)),
                    'mass_per_particle': float(match.group(3)),
                    'source': 'TypeII',
                    'timestep': self.timestep_counter
                })
        
        # Parse individual dust particle creation
        elif 'Created dust particle' in line:
            match = re.search(r'Created dust particle (\d+): mass=([\d.e+-]+), ID=(\d+)', line)
            if match:
                # Store additional info if needed
                pass
        
        # Parse thermal destruction events
        elif 'Destroying dust particle' in line and 'hot gas' in line:
            match = re.search(r'Destroying dust particle (\d+) in hot gas \(T=([\d.e+-]+) K\)', line)
            if match:
                self.dust_destruction_events.append({
                    'particle_idx': int(match.group(1)),
                    'temperature': float(match.group(2)),
                    'reason': 'thermal',
                    'timestep': self.timestep_counter
                })
        
        # Parse age-based destruction
        elif 'Destroying old dust particle' in line:
            match = re.search(r'Destroying old dust particle (\d+) \(age=([\d.]+) Myr\)', line)
            if match:
                self.dust_destruction_events.append({
                    'particle_idx': int(match.group(1)),
                    'age': float(match.group(2)),
                    'reason': 'age',
                    'timestep': self.timestep_counter
                })
        
        # Parse drag diagnostics
        elif '[DUST_DRAG]' in line:
            match = re.search(r'Particle (\d+): \|v_old\|=([\d.]+), \|v_new\|=([\d.]+), \|v_gas\|=([\d.]+), \|dv\|=([\d.]+), drag_factor=([\d.]+)', line)
            if match:
                self.dust_drag_events.append({
                    'particle_idx': int(match.group(1)),
                    'v_old': float(match.group(2)),
                    'v_new': float(match.group(3)),
                    'v_gas': float(match.group(4)),
                    'dv': float(match.group(5)),
                    'drag_factor': float(match.group(6)),
                    'timestep': self.timestep_counter
                })
                
    def _parse_dust_position(self, line):
        """Parse dust position information."""
        match = re.search(r'Star (\d+) at \(([\d.]+),([\d.]+),([\d.]+)\), Dust at \(([\d.]+),([\d.]+),([\d.]+)\), offset=([\d.]+)', line)
        if match:
            star_pos = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            dust_pos = [float(match.group(5)), float(match.group(6)), float(match.group(7))]
            offset = float(match.group(8))
            
            self.dust_positions.append({
                'star_id': int(match.group(1)),
                'star_pos': star_pos,
                'dust_pos': dust_pos,
                'offset': offset,
                'timestep': self.timestep_counter
            })
            
    def _parse_dust_stats(self, line):
        """Parse dust statistics."""
        # Parse per-step statistics
        if 'This step:' in line:
            match = re.search(r'created=(\d+), destroyed=(\d+), mass_change=([\d.e+-]+)', line)
            if match:
                self.current_timestep = {
                    'created': int(match.group(1)),
                    'destroyed': int(match.group(2)),
                    'mass_change': float(match.group(3)),
                    'timestep': self.timestep_counter
                }
        
        # Parse total statistics
        elif 'Total:' in line and self.current_timestep:
            match = re.search(r'created=(\d+), destroyed=(\d+), total_mass=([\d.e+-]+)', line)
            if match:
                self.current_timestep['total_created'] = int(match.group(1))
                self.current_timestep['total_destroyed'] = int(match.group(2))
                self.current_timestep['total_mass'] = float(match.group(3))
                
                self.dust_stats.append(self.current_timestep)
                self.current_timestep = None
                self.timestep_counter += 1
                
    def _parse_dust_coupling(self, line):
        """Parse dust-gas coupling information."""
        match = re.search(r'Step (\d+): (\d+) dust particles, avg \|v_dust-v_gas\|=([\d.]+), max=([\d.]+)', line)
        if match:
            self.dust_coupling.append({
                'step': int(match.group(1)),
                'n_particles': int(match.group(2)),
                'avg_velocity_diff': float(match.group(3)),
                'max_velocity_diff': float(match.group(4))
            })

def create_dust_diagnostic_plots(parser, output_prefix='dust_diagnostics'):
    """Create diagnostic plots from parsed dust data."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Dust mass evolution
    if parser.dust_stats:
        ax1 = fig.add_subplot(gs[0, :])
        timesteps = [s['timestep'] for s in parser.dust_stats]
        total_mass = [s['total_mass'] for s in parser.dust_stats]
        
        ax1.plot(timesteps, total_mass, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Output Number')
        ax1.set_ylabel('Total Dust Mass (code units)')
        ax1.set_title('Total Dust Mass Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Add text box with growth rate
        if len(total_mass) > 1:
            growth_rate = (total_mass[-1] - total_mass[0]) / len(total_mass)
            ax1.text(0.02, 0.95, f'Average growth rate: {growth_rate:.2e} per output',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Cumulative dust particles created/destroyed
    if parser.dust_stats:
        ax2 = fig.add_subplot(gs[1, 0])
        timesteps = [s['timestep'] for s in parser.dust_stats]
        total_created = [s['total_created'] for s in parser.dust_stats]
        total_destroyed = [s['total_destroyed'] for s in parser.dust_stats]
        
        ax2.plot(timesteps, total_created, 'g-', label='Created', linewidth=2, marker='o')
        ax2.plot(timesteps, total_destroyed, 'r-', label='Destroyed', linewidth=2, marker='x')
        ax2.set_xlabel('Output Number')
        ax2.set_ylabel('Cumulative Number of Particles')
        ax2.set_title('Cumulative Dust Particle Creation/Destruction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add net particles
        net_particles = np.array(total_created) - np.array(total_destroyed)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(timesteps, net_particles, 'k--', label='Net particles', alpha=0.7)
        ax2_twin.set_ylabel('Net Particles', color='k')
        ax2_twin.tick_params(axis='y', labelcolor='k')
    
    # 3. Dust creation rate per timestep
    if parser.dust_stats:
        ax3 = fig.add_subplot(gs[1, 1])
        created_per_step = [s['created'] for s in parser.dust_stats]
        destroyed_per_step = [s['destroyed'] for s in parser.dust_stats]
        
        width = 0.35
        x = np.arange(len(created_per_step))
        
        ax3.bar(x - width/2, created_per_step, width, label='Created', color='green', alpha=0.7)
        ax3.bar(x + width/2, destroyed_per_step, width, label='Destroyed', color='red', alpha=0.7)
        
        ax3.set_xlabel('Output Number')
        ax3.set_ylabel('Particles per Output')
        ax3.set_title('Dust Creation/Destruction Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Limit x-axis if too many points
        if len(x) > 50:
            ax3.set_xlim(len(x)-50, len(x))
            ax3.set_title('Dust Creation/Destruction Rate (Last 50 Outputs)')
    
    # 4. Mass per particle distribution
    if parser.dust_creation_events:
        ax4 = fig.add_subplot(gs[1, 2])
        
        agb_masses = [e['mass_per_particle'] for e in parser.dust_creation_events if e['source'] == 'AGB']
        typeii_masses = [e['mass_per_particle'] for e in parser.dust_creation_events if e['source'] == 'TypeII']
        
        if agb_masses:
            ax4.hist(np.log10(agb_masses), bins=20, alpha=0.5, color='blue', 
                    label=f'AGB (n={len(agb_masses)})', density=True)
        if typeii_masses:
            ax4.hist(np.log10(typeii_masses), bins=20, alpha=0.5, color='red', 
                    label=f'Type II (n={len(typeii_masses)})', density=True)
        
        ax4.set_xlabel('log₁₀(Mass per Particle)')
        ax4.set_ylabel('Normalized Count')
        ax4.set_title('Dust Particle Mass Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Dust-gas coupling velocity difference
    if parser.dust_coupling:
        ax5 = fig.add_subplot(gs[2, 0])
        steps = [c['step'] for c in parser.dust_coupling]
        avg_vdiff = [c['avg_velocity_diff'] for c in parser.dust_coupling]
        max_vdiff = [c['max_velocity_diff'] for c in parser.dust_coupling]
        
        ax5.plot(steps, avg_vdiff, 'b-', label='Average |v_dust - v_gas|', linewidth=2)
        ax5.plot(steps, max_vdiff, 'r--', label='Maximum |v_dust - v_gas|', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Simulation Step')
        ax5.set_ylabel('Velocity Difference (code units)')
        ax5.set_title('Dust-Gas Coupling Strength')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    
    # 6. Number of dust particles evolution
    if parser.dust_coupling:
        ax6 = fig.add_subplot(gs[2, 1])
        steps = [c['step'] for c in parser.dust_coupling]
        n_particles = [c['n_particles'] for c in parser.dust_coupling]
        
        ax6.plot(steps, n_particles, 'g-', linewidth=2, marker='o', markersize=4)
        ax6.set_xlabel('Simulation Step')
        ax6.set_ylabel('Number of Dust Particles')
        ax6.set_title('Active Dust Particle Count')
        ax6.grid(True, alpha=0.3)
        
        # Add growth rate
        if len(n_particles) > 1:
            growth_rate = (n_particles[-1] - n_particles[0]) / (steps[-1] - steps[0])
            ax6.text(0.02, 0.95, f'Growth rate: {growth_rate:.3f} particles/step',
                    transform=ax6.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 7. Dust source breakdown
    if parser.dust_creation_events:
        ax7 = fig.add_subplot(gs[2, 2])
        
        agb_count = sum(1 for e in parser.dust_creation_events if e['source'] == 'AGB')
        typeii_count = sum(1 for e in parser.dust_creation_events if e['source'] == 'TypeII')
        
        if agb_count > 0 or typeii_count > 0:
            labels = []
            sizes = []
            colors = []
            
            if agb_count > 0:
                labels.append(f'AGB\n({agb_count} events)')
                sizes.append(agb_count)
                colors.append('blue')
            
            if typeii_count > 0:
                labels.append(f'Type II\n({typeii_count} events)')
                sizes.append(typeii_count)
                colors.append('red')
            
            ax7.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax7.set_title('Dust Creation by Source Type')
    
    # 8. Dust offset distribution
    if parser.dust_positions:
        ax8 = fig.add_subplot(gs[3, 0])
        offsets = [p['offset'] for p in parser.dust_positions]
        
        ax8.hist(offsets, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax8.set_xlabel('Offset Distance (kpc)')
        ax8.set_ylabel('Count')
        ax8.set_title('Dust Creation Offset from Parent Star')
        ax8.grid(True, alpha=0.3)
        
        # Add statistics
        ax8.axvline(np.mean(offsets), color='red', linestyle='--', label=f'Mean: {np.mean(offsets):.2f} kpc')
        ax8.legend()
    
    # 9. Mass change per timestep
    if parser.dust_stats:
        ax9 = fig.add_subplot(gs[3, 1])
        timesteps = [s['timestep'] for s in parser.dust_stats]
        mass_changes = [s['mass_change'] for s in parser.dust_stats]
        
        positive_changes = [m if m > 0 else 0 for m in mass_changes]
        negative_changes = [m if m < 0 else 0 for m in mass_changes]
        
        ax9.bar(timesteps, positive_changes, color='green', alpha=0.7, label='Mass gained')
        ax9.bar(timesteps, negative_changes, color='red', alpha=0.7, label='Mass lost')
        
        ax9.set_xlabel('Output Number')
        ax9.set_ylabel('Mass Change (code units)')
        ax9.set_title('Dust Mass Change per Output')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        ax9.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax9.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Limit x-axis if too many points
        if len(timesteps) > 50:
            ax9.set_xlim(len(timesteps)-50, len(timesteps))
            ax9.set_title('Dust Mass Change per Output (Last 50)')
    
    # 10. Dust destruction mechanisms
    if parser.dust_destruction_events:
        ax10 = fig.add_subplot(gs[3, 2])
        
        thermal_count = sum(1 for e in parser.dust_destruction_events if e['reason'] == 'thermal')
        age_count = sum(1 for e in parser.dust_destruction_events if e['reason'] == 'age')
        
        if thermal_count > 0 or age_count > 0:
            labels = []
            sizes = []
            colors = []
            
            if thermal_count > 0:
                labels.append(f'Thermal\n({thermal_count} events)')
                sizes.append(thermal_count)
                colors.append('red')
            
            if age_count > 0:
                labels.append(f'Age\n({age_count} events)')
                sizes.append(age_count)
                colors.append('gray')
            
            ax10.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax10.set_title('Dust Destruction Mechanisms')
        else:
            ax10.text(0.5, 0.5, 'No dust destruction\nevents detected yet',
                     transform=ax10.transAxes, ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            ax10.set_title('Dust Destruction Mechanisms')
    
    # Additional plots on a new row
    # 11. Gas drag evolution
    if parser.dust_drag_events:
        ax11 = fig.add_subplot(gs[4, 0])
        
        v_old = [e['v_old'] for e in parser.dust_drag_events]
        v_new = [e['v_new'] for e in parser.dust_drag_events]
        drag_factors = [e['drag_factor'] for e in parser.dust_drag_events]
        
        # Plot velocity change distribution
        velocity_ratios = [v_n/v_o if v_o > 0 else 1.0 for v_n, v_o in zip(v_new, v_old)]
        
        ax11.hist(velocity_ratios, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax11.axvline(1.0, color='red', linestyle='--', label='No change')
        ax11.set_xlabel('Velocity Ratio (v_new/v_old)')
        ax11.set_ylabel('Count')
        ax11.set_title('Gas Drag Effect on Dust Velocities')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    
    # 12. Thermal destruction temperatures
    thermal_destructions = [e for e in parser.dust_destruction_events if e['reason'] == 'thermal']
    if thermal_destructions:
        ax12 = fig.add_subplot(gs[4, 1])
        
        temperatures = [e['temperature'] for e in thermal_destructions]
        
        ax12.hist(np.log10(temperatures), bins=20, alpha=0.7, color='red', edgecolor='black')
        ax12.axvline(np.log10(1e6), color='blue', linestyle='--', label='Destruction threshold (1e6 K)')
        ax12.set_xlabel('log₁₀(Temperature) [K]')
        ax12.set_ylabel('Count')
        ax12.set_title('Thermal Destruction Temperatures')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
    
    # 13. Age destruction distribution
    age_destructions = [e for e in parser.dust_destruction_events if e['reason'] == 'age']
    if age_destructions:
        ax13 = fig.add_subplot(gs[4, 2])
        
        ages = [e['age'] for e in age_destructions]
        
        ax13.hist(ages, bins=20, alpha=0.7, color='gray', edgecolor='black')
        ax13.axvline(1000.0, color='red', linestyle='--', label='Max lifetime (1000 Myr)')
        ax13.set_xlabel('Age at Destruction (Myr)')
        ax13.set_ylabel('Count')
        ax13.set_title('Age-based Dust Destruction')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
    
    # Save the figure
    plt.suptitle(f'Gadget-4 Dust Diagnostics (Total outputs: {len(parser.dust_stats)})', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    print(f"Saved plots to {output_prefix}.png and {output_prefix}.pdf")
    
    # Create summary statistics
    create_dust_summary_report(parser, output_prefix)

def create_dust_summary_report(parser, output_prefix):
    """Create a text summary report of the dust statistics."""
    
    with open(f'{output_prefix}_summary.txt', 'w') as f:
        f.write("Gadget-4 Dust Diagnostics Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Module parameters from code
        f.write("Dust Module Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write("DUST_YIELD_TYPEII: 0.01 (1% of metals)\n")
        f.write("DUST_YIELD_AGB: 0.01 (1% of metals)\n")
        f.write("MIN_DUST_PARTICLE_MASS: 1e-12\n")
        f.write("DUST_DESTRUCTION_TEMP: 1e6 K\n")
        f.write("DUST_MAX_LIFETIME: 1000 Myr\n")
        f.write("Type II: 3 particles per event, 100 km/s velocity\n")
        f.write("AGB: 2 particles per event, 10 km/s velocity\n")
        f.write("Offset distance: 0.1 * softening length\n\n")
        
        # Overall dust statistics
        if parser.dust_stats:
            f.write("Dust Evolution Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of outputs analyzed: {len(parser.dust_stats)}\n")
            
            final_stats = parser.dust_stats[-1]
            f.write(f"Final total dust particles created: {final_stats['total_created']}\n")
            f.write(f"Final total dust particles destroyed: {final_stats['total_destroyed']}\n")
            f.write(f"Final net dust particles: {final_stats['total_created'] - final_stats['total_destroyed']}\n")
            f.write(f"Final total dust mass: {final_stats['total_mass']:.6e} (code units)\n")
            
            # Convert to physical units
            if final_stats['total_mass'] > 0:
                # Assume code mass unit info available
                f.write(f"Final total dust mass: ~{final_stats['total_mass'] * 1e10:.3e} M☉ (assuming typical units)\n")
            
            # Calculate averages
            created_per_step = [s['created'] for s in parser.dust_stats]
            destroyed_per_step = [s['destroyed'] for s in parser.dust_stats]
            
            f.write(f"\nPer-output statistics:\n")
            f.write(f"Average created: {np.mean(created_per_step):.2f} particles\n")
            f.write(f"Average destroyed: {np.mean(destroyed_per_step):.2f} particles\n")
            f.write(f"Max created in one output: {max(created_per_step)}\n")
            f.write(f"Max destroyed in one output: {max(destroyed_per_step)}\n\n")
        
        # Dust creation events
        if parser.dust_creation_events:
            f.write("Dust Creation Events:\n")
            f.write("-" * 30 + "\n")
            
            agb_events = [e for e in parser.dust_creation_events if e['source'] == 'AGB']
            typeii_events = [e for e in parser.dust_creation_events if e['source'] == 'TypeII']
            
            f.write(f"Total AGB dust events: {len(agb_events)}\n")
            f.write(f"Total Type II dust events: {len(typeii_events)}\n")
            
            if agb_events:
                agb_masses = [e['mass_per_particle'] for e in agb_events]
                agb_particles = [e['n_particles'] for e in agb_events]
                f.write(f"\nAGB dust properties:\n")
                f.write(f"  Mass per particle: {np.mean(agb_masses):.3e} ± {np.std(agb_masses):.3e}\n")
                f.write(f"  Particles per event: {np.mean(agb_particles):.1f} ± {np.std(agb_particles):.1f}\n")
                f.write(f"  Total mass created: {sum(agb_masses) * sum(agb_particles):.3e}\n")
            
            if typeii_events:
                typeii_masses = [e['mass_per_particle'] for e in typeii_events]
                typeii_particles = [e['n_particles'] for e in typeii_events]
                f.write(f"\nType II dust properties:\n")
                f.write(f"  Mass per particle: {np.mean(typeii_masses):.3e} ± {np.std(typeii_masses):.3e}\n")
                f.write(f"  Particles per event: {np.mean(typeii_particles):.1f} ± {np.std(typeii_particles):.1f}\n")
                f.write(f"  Total mass created: {sum(typeii_masses) * sum(typeii_particles):.3e}\n")
        
        # Dust destruction analysis
        if parser.dust_destruction_events:
            f.write("\nDust Destruction Analysis:\n")
            f.write("-" * 30 + "\n")
            
            thermal_events = [e for e in parser.dust_destruction_events if e['reason'] == 'thermal']
            age_events = [e for e in parser.dust_destruction_events if e['reason'] == 'age']
            
            f.write(f"Total thermal destruction events: {len(thermal_events)}\n")
            f.write(f"Total age-based destruction events: {len(age_events)}\n")
            
            if thermal_events:
                temps = [e['temperature'] for e in thermal_events]
                f.write(f"  Temperature range: {min(temps):.2e} - {max(temps):.2e} K\n")
                f.write(f"  Mean destruction temperature: {np.mean(temps):.2e} K\n")
            
            if age_events:
                ages = [e['age'] for e in age_events]
                f.write(f"  Age range at destruction: {min(ages):.1f} - {max(ages):.1f} Myr\n")
                f.write(f"  Mean age at destruction: {np.mean(ages):.1f} Myr\n")
        
        # Dust-gas coupling
        if parser.dust_coupling:
            f.write("\nDust-Gas Coupling:\n")
            f.write("-" * 30 + "\n")
            
            avg_vdiffs = [c['avg_velocity_diff'] for c in parser.dust_coupling]
            max_vdiffs = [c['max_velocity_diff'] for c in parser.dust_coupling]
            n_particles = [c['n_particles'] for c in parser.dust_coupling]
            
            f.write(f"Final dust particle count: {n_particles[-1]}\n")
            f.write(f"Average velocity difference: {np.mean(avg_vdiffs):.3f} (code units)\n")
            f.write(f"Maximum velocity difference: {max(max_vdiffs):.3f} (code units)\n")
            
            # Check if coupling is improving
            if len(avg_vdiffs) > 10:
                early_avg = np.mean(avg_vdiffs[:10])
                late_avg = np.mean(avg_vdiffs[-10:])
                improvement = 100.0 * (early_avg - late_avg) / early_avg
                f.write(f"Coupling improvement: {improvement:.1f}% (early vs late)\n")
            
            f.write(f"Coupling trend: {'Improving' if avg_vdiffs[-1] < avg_vdiffs[0] else 'Worsening'}\n")
        
        # Dust drag analysis
        if parser.dust_drag_events:
            f.write("\nGas Drag Analysis:\n")
            f.write("-" * 30 + "\n")
            
            drag_factors = [e['drag_factor'] for e in parser.dust_drag_events]
            velocity_changes = [e['dv'] for e in parser.dust_drag_events]
            
            f.write(f"Number of drag events analyzed: {len(parser.dust_drag_events)}\n")
            f.write(f"Average drag factor: {np.mean(drag_factors):.4f}\n")
            f.write(f"Average velocity change: {np.mean(velocity_changes):.3f} (code units)\n")
            f.write(f"Max velocity change: {max(velocity_changes):.3f} (code units)\n")
        
        # Dust positions
        if parser.dust_positions:
            f.write("\nDust Creation Geometry:\n")
            f.write("-" * 30 + "\n")
            offsets = [p['offset'] for p in parser.dust_positions]
            f.write(f"Average offset from parent star: {np.mean(offsets):.3f} kpc\n")
            f.write(f"Offset range: {min(offsets):.3f} - {max(offsets):.3f} kpc\n")
            
            # Check if offset is constant
            if len(set(offsets)) == 1:
                f.write(f"Note: All dust created at fixed offset of {offsets[0]} kpc\n")
                f.write("This suggests using 0.1 * softening length consistently\n")
        
        # Recommendations
        f.write("\nRecommendations:\n")
        f.write("-" * 30 + "\n")
        
        if parser.dust_stats and final_stats['total_destroyed'] == 0:
            f.write("- No dust destruction detected yet - check if simulation is hot enough\n")
            f.write("  or if dust lifetime (1000 Myr) hasn't been reached\n")
        
        if parser.dust_coupling and np.mean(avg_vdiffs) > 50:
            f.write("- High dust-gas velocity differences suggest weak coupling\n")
            f.write("  Consider adjusting drag timescale (currently 10 Myr)\n")
        
        if parser.dust_positions and len(set(offsets)) == 1:
            f.write("- Fixed offset distance detected - consider varying offset\n")
            f.write("  based on local conditions or using random factors\n")
        
        if not parser.dust_drag_events:
            f.write("- No drag diagnostics found - ensure DUST_DRAG debug output is enabled\n")
    
    print(f"Saved summary to {output_prefix}_summary.txt")

def main():
    parser = argparse.ArgumentParser(description='Plot Gadget-4 dust diagnostics')
    parser.add_argument('input_file', help='Input file containing dust output')
    parser.add_argument('-o', '--output', default='dust_diagnostics', 
                       help='Output file prefix (default: dust_diagnostics)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Parse the dust data
    dust_parser = DustParser()
    print(f"Parsing {args.input_file}...")
    dust_parser.parse_file(args.input_file)
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    create_dust_diagnostic_plots(dust_parser, args.output)
    
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()