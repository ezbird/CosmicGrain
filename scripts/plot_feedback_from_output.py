#!/usr/bin/env python3
"""
Gadget-4 Cosmological Simulation Feedback Diagnostics Plotter

This script parses feedback output from Gadget-4 simulations and creates
diagnostic plots for Type II SNe and AGB wind feedback events.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
from collections import defaultdict
import matplotlib.gridspec as gridspec

class FeedbackParser:
    """Parse Gadget-4 feedback output and extract relevant data."""
    
    def __init__(self):
        self.star_summaries = []
        self.type_ii_events = []
        self.agb_events = []
        self.timestep_summaries = []
        self.star_ages = defaultdict(list)
        self.feedback_flags = defaultdict(list)
        self.timestep_counter = 0
        self.current_timestep_data = None
        
    def parse_file(self, filename):
        """Parse the feedback output file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if '[FEEDBACK] [FEEDBACK]' in line:
                self._parse_feedback_line(line)
            elif '[FEEDBACK] [AGB]' in line:
                self._parse_agb_line(line, lines, i)
            elif '[FEEDBACK] [TYPEII]' in line:
                self._parse_typeii_line(line, lines, i)
            elif '[FEEDBACK] [DEPOSIT]' in line:
                self._parse_deposit_line(line)
            elif '*** AGB WIND TRIGGERED ***' in line:
                self._parse_agb_trigger(line)
            elif '*** TYPE II SN TRIGGERED ***' in line:
                self._parse_typeii_trigger(line)
            elif 'This timestep:' in line:
                self._parse_timestep_summary(line)
                
    def _parse_feedback_line(self, line):
        """Parse general feedback lines."""
        # Parse star summaries
        if 'Total stars:' in line:
            match = re.search(r'Total stars: (\d+), In Type II range: (\d+), With feedback flag: (\d+)', line)
            if match:
                # Start a new timestep when we see star summary
                if self.current_timestep_data is not None:
                    self.star_summaries.append(self.current_timestep_data)
                
                self.current_timestep_data = {
                    'total': int(match.group(1)),
                    'type_ii_range': int(match.group(2)),
                    'with_feedback': int(match.group(3)),
                    'timestep': self.timestep_counter
                }
                self.timestep_counter += 1
        
        # Parse individual star data
        elif 'Star' in line and 'age=' in line:
            match = re.search(r'Star (\d+): age=([\d.]+) Myr, FeedbackFlag=(\d+), StellarAge=([\d.]+), CurrentTime=([\d.]+)', line)
            if match:
                star_id = int(match.group(1))
                age = float(match.group(2))
                feedback_flag = int(match.group(3))
                stellar_age = float(match.group(4))
                current_time = float(match.group(5))
                
                self.star_ages[star_id].append(age)
                self.feedback_flags[star_id].append(feedback_flag)
                
    def _parse_typeii_line(self, line, lines, line_idx):
        """Parse Type II supernova lines."""
        if 'stellar_mass=' in line:
            match = re.search(r'Star (\d+): stellar_mass=([\d.e+]+) Msun, n_sne=([\d.e+]+), energy_ergs=([\d.e+]+), energy_code=([\d.e+-]+)', line)
            if match:
                event = {
                    'star_id': int(match.group(1)),
                    'stellar_mass': float(match.group(2)),
                    'n_sne': float(match.group(3)),
                    'energy_ergs': float(match.group(4)),
                    'energy_code': float(match.group(5)),
                    'neighbors': 0,
                    'smoothing_length': 0,
                    'type': 'typeii'
                }
                
                # Look for additional info
                for j in range(line_idx, min(line_idx + 20, len(lines))):
                    if f'Star {event["star_id"]}:' in lines[j]:
                        if 'found' in lines[j] and 'neighbors' in lines[j]:
                            neighbor_match = re.search(r'found (\d+) neighbors, smoothing_length=([\d.]+)', lines[j])
                            if neighbor_match:
                                event['neighbors'] = int(neighbor_match.group(1))
                                event['smoothing_length'] = float(neighbor_match.group(2))
                
                self.type_ii_events.append(event)
                
    def _parse_agb_line(self, line, lines, line_idx):
        """Parse AGB feedback lines."""
        if 'stellar_mass=' in line:
            match = re.search(r'Star (\d+): stellar_mass=([\d.e+]+) Msun, mass_loss=([\d.]+), energy_ergs=([\d.e+]+), energy_code=([\d.e+-]+)', line)
            if match:
                event = {
                    'star_id': int(match.group(1)),
                    'stellar_mass': float(match.group(2)),
                    'mass_loss': float(match.group(3)),
                    'energy_ergs': float(match.group(4)),
                    'energy_code': float(match.group(5)),
                    'neighbors': 0,
                    'smoothing_length': 0,
                    'energy_deposited': 0,
                    'metals_deposited': 0,
                    'type': 'agb'
                }
                
                # Look for additional info in nearby lines
                for j in range(line_idx, min(line_idx + 20, len(lines))):
                    if f'Star {event["star_id"]}:' in lines[j]:
                        if 'found' in lines[j] and 'neighbors' in lines[j]:
                            neighbor_match = re.search(r'found (\d+) neighbors, smoothing_length=([\d.]+)', lines[j])
                            if neighbor_match:
                                event['neighbors'] = int(neighbor_match.group(1))
                                event['smoothing_length'] = float(neighbor_match.group(2))
                        elif 'Successfully deposited' in lines[j]:
                            deposit_match = re.search(r'Successfully deposited ([\d.e+]+) erg into (\d+) neighbors', lines[j])
                            if deposit_match:
                                event['energy_deposited'] = float(deposit_match.group(1))
                
                self.agb_events.append(event)
                
    def _parse_deposit_line(self, line):
        """Parse energy/metal deposition lines."""
        if 'SUMMARY:' in line:
            match = re.search(r'Total energy deposited=([\d.e+-]+) \(([\d.]+)%\), Total metals deposited=([\d.e+-]+) \(([\d.]+)%\)', line)
            if match and self.agb_events:
                # Update the most recent AGB event
                self.agb_events[-1]['energy_deposited_code'] = float(match.group(1))
                self.agb_events[-1]['energy_deposition_efficiency'] = float(match.group(2))
                self.agb_events[-1]['metals_deposited'] = float(match.group(3))
                self.agb_events[-1]['metal_deposition_efficiency'] = float(match.group(4))
                
    def _parse_agb_trigger(self, line):
        """Parse AGB wind trigger events."""
        match = re.search(r'Star (\d+) \(age=([\d.]+) Myr, prob=([\d.]+)\)', line)
        if match:
            star_id = int(match.group(1))
            age = float(match.group(2))
            prob = float(match.group(3))
            
            # Update the corresponding AGB event
            for event in reversed(self.agb_events):
                if event['star_id'] == star_id:
                    event['trigger_age'] = age
                    event['trigger_probability'] = prob
                    break
                    
    def _parse_typeii_trigger(self, line):
        """Parse Type II trigger events."""
        match = re.search(r'Star (\d+) \(age=([\d.]+) Myr, prob=([\d.]+)\)', line)
        if match:
            star_id = int(match.group(1))
            age = float(match.group(2))
            prob = float(match.group(3))
            
            # Update the corresponding Type II event
            for event in reversed(self.type_ii_events):
                if event['star_id'] == star_id:
                    event['trigger_age'] = age
                    event['trigger_probability'] = prob
                    break
                    
    def _parse_timestep_summary(self, line):
        """Parse timestep summary."""
        match = re.search(r'This timestep: (\d+) Type II SNe, (\d+) AGB winds', line)
        if match:
            self.timestep_summaries.append({
                'type_ii_count': int(match.group(1)),
                'agb_count': int(match.group(2)),
                'timestep': len(self.timestep_summaries)
            })

def create_diagnostic_plots(parser, output_prefix='feedback_diagnostics'):
    """Create diagnostic plots from parsed data."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Star population statistics over time
    if parser.star_summaries:
        ax1 = fig.add_subplot(gs[0, :])
        timesteps = [s['timestep'] for s in parser.star_summaries]
        total_stars = [s['total'] for s in parser.star_summaries]
        type_ii_stars = [s['type_ii_range'] for s in parser.star_summaries]
        feedback_stars = [s['with_feedback'] for s in parser.star_summaries]
        
        ax1.plot(timesteps, total_stars, 'b-', label='Total stars', linewidth=2, marker='o', markersize=4)
        ax1.plot(timesteps, feedback_stars, 'g-', label='With feedback flag', linewidth=2, marker='s', markersize=4)
        ax1.plot(timesteps, type_ii_stars, 'r-', label='In Type II range (3-40 Myr)', linewidth=2, marker='^', markersize=4)
        ax1.set_xlabel('Output Number')
        ax1.set_ylabel('Number of Stars')
        ax1.set_title('Star Population Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Age distribution histogram with feedback ranges
    if parser.star_ages:
        ax2 = fig.add_subplot(gs[1, 0])
        all_ages = []
        for ages in parser.star_ages.values():
            all_ages.extend(ages)
        
        bins = np.linspace(0, max(all_ages) + 10, 50)
        ax2.hist(all_ages, bins=bins, alpha=0.7, color='purple', edgecolor='black', label='All stars')
        
        # Highlight feedback ranges
        ax2.axvspan(3.0, 40.0, alpha=0.2, color='red', label='Type II range')
        ax2.axvspan(50.0, 150.0, alpha=0.2, color='blue', label='AGB range')
        
        ax2.set_xlabel('Age (Myr)')
        ax2.set_ylabel('Count')
        ax2.set_title('Star Age Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Feedback trigger probability distribution
    if parser.agb_events or parser.type_ii_events:
        ax3 = fig.add_subplot(gs[1, 1])
        
        agb_probs = [event.get('trigger_probability', 0) for event in parser.agb_events 
                     if 'trigger_probability' in event]
        typeii_probs = [event.get('trigger_probability', 0) for event in parser.type_ii_events 
                        if 'trigger_probability' in event]
        
        bins = np.linspace(0, 1, 20)
        if agb_probs:
            ax3.hist(agb_probs, bins=bins, alpha=0.5, color='blue', label=f'AGB (n={len(agb_probs)})', density=True)
        if typeii_probs:
            ax3.hist(typeii_probs, bins=bins, alpha=0.5, color='red', label=f'Type II (n={len(typeii_probs)})', density=True)
        
        ax3.set_xlabel('Trigger Probability')
        ax3.set_ylabel('Normalized Count')
        ax3.set_title('Feedback Trigger Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Energy vs stellar mass (separate for AGB and Type II)
    if parser.agb_events or parser.type_ii_events:
        ax4 = fig.add_subplot(gs[1, 2])
        
        if parser.agb_events:
            agb_masses = [event['stellar_mass'] for event in parser.agb_events]
            agb_energies = [event['energy_ergs'] for event in parser.agb_events]
            ax4.scatter(agb_masses, agb_energies, c='blue', alpha=0.6, s=100, label='AGB winds')
        
        if parser.type_ii_events:
            typeii_masses = [event['stellar_mass'] for event in parser.type_ii_events]
            typeii_energies = [event['energy_ergs'] for event in parser.type_ii_events]
            ax4.scatter(typeii_masses, typeii_energies, c='red', alpha=0.6, s=100, label='Type II SNe')
        
        ax4.set_xlabel('Stellar Mass (M☉)')
        ax4.set_ylabel('Energy (erg)')
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        ax4.set_title('Feedback Energy vs Stellar Mass')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Smoothing length distribution
    if parser.agb_events or parser.type_ii_events:
        ax5 = fig.add_subplot(gs[2, 0])
        
        all_smoothing = []
        labels = []
        
        if parser.agb_events:
            agb_smoothing = [event['smoothing_length'] for event in parser.agb_events if event['smoothing_length'] > 0]
            if agb_smoothing:
                all_smoothing.append(agb_smoothing)
                labels.append(f'AGB (n={len(agb_smoothing)})')
        
        if parser.type_ii_events:
            typeii_smoothing = [event['smoothing_length'] for event in parser.type_ii_events if event['smoothing_length'] > 0]
            if typeii_smoothing:
                all_smoothing.append(typeii_smoothing)
                labels.append(f'Type II (n={len(typeii_smoothing)})')
        
        if all_smoothing:
            ax5.hist(all_smoothing, bins=20, alpha=0.7, label=labels, color=['blue', 'red'][:len(all_smoothing)])
            ax5.set_xlabel('Smoothing Length (kpc)')
            ax5.set_ylabel('Count')
            ax5.set_title('SPH Smoothing Length Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # 6. Neighbor distribution
    if parser.agb_events or parser.type_ii_events:
        ax6 = fig.add_subplot(gs[2, 1])
        
        all_neighbors = []
        all_events = parser.agb_events + parser.type_ii_events
        if all_events:
            neighbors = [event['neighbors'] for event in all_events if event['neighbors'] > 0]
            if neighbors:
                max_neighbors = max(neighbors)
                bins = range(0, max_neighbors + 2)
                ax6.hist(neighbors, bins=bins, alpha=0.7, color='green', edgecolor='black')
                ax6.set_xlabel('Number of Neighbors')
                ax6.set_ylabel('Count')
                ax6.set_title('Feedback Neighbor Distribution')
                ax6.set_xlim(0, min(max_neighbors + 1, 20))
                ax6.grid(True, alpha=0.3)
                
                # Add statistics
                ax6.text(0.95, 0.95, f'Mean: {np.mean(neighbors):.1f}\nMedian: {np.median(neighbors):.0f}',
                        transform=ax6.transAxes, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Feedback events timeline
    if parser.timestep_summaries:
        ax7 = fig.add_subplot(gs[2, 2])
        timesteps = [s['timestep'] for s in parser.timestep_summaries]
        type_ii_counts = [s['type_ii_count'] for s in parser.timestep_summaries]
        agb_counts = [s['agb_count'] for s in parser.timestep_summaries]
        
        width = 0.35
        x = np.arange(len(timesteps))
        
        ax7.bar(x - width/2, type_ii_counts, width, label='Type II SNe', color='red', alpha=0.7)
        ax7.bar(x + width/2, agb_counts, width, label='AGB winds', color='blue', alpha=0.7)
        
        ax7.set_xlabel('Output Number')
        ax7.set_ylabel('Number of Events')
        ax7.set_title('Feedback Events per Output')
        ax7.set_xticks(x)
        ax7.set_xticklabels(timesteps)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Cumulative feedback events
    if parser.timestep_summaries:
        ax8 = fig.add_subplot(gs[3, 0])
        
        cumulative_typeii = np.cumsum([s['type_ii_count'] for s in parser.timestep_summaries])
        cumulative_agb = np.cumsum([s['agb_count'] for s in parser.timestep_summaries])
        timesteps = range(len(parser.timestep_summaries))
        
        ax8.plot(timesteps, cumulative_typeii, 'r-', label='Type II SNe', linewidth=2, marker='o')
        ax8.plot(timesteps, cumulative_agb, 'b-', label='AGB winds', linewidth=2, marker='s')
        ax8.plot(timesteps, cumulative_typeii + cumulative_agb, 'k--', label='Total', linewidth=2)
        
        ax8.set_xlabel('Output Number')
        ax8.set_ylabel('Cumulative Events')
        ax8.set_title('Cumulative Feedback Events')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Energy deposition efficiency histogram
    if parser.agb_events:
        ax9 = fig.add_subplot(gs[3, 1])
        efficiencies = [event.get('energy_deposition_efficiency', 0) for event in parser.agb_events 
                       if 'energy_deposition_efficiency' in event]
        
        if efficiencies:
            # Check if all efficiencies are 100%
            unique_eff = np.unique(efficiencies)
            if len(unique_eff) == 1 and unique_eff[0] == 100.0:
                ax9.text(0.5, 0.5, f'All {len(efficiencies)} events\nhad 100% efficiency\n(SPH kernel normalization)',
                        transform=ax9.transAxes, ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax9.set_title('Energy Deposition Efficiency')
                ax9.set_xlabel('Efficiency (%)')
                ax9.set_ylabel('Count')
            else:
                bins = np.linspace(min(efficiencies)*0.9, max(efficiencies)*1.1, 20)
                ax9.hist(efficiencies, bins=bins, alpha=0.7, color='red', edgecolor='black')
                ax9.set_xlabel('Deposition Efficiency (%)')
                ax9.set_ylabel('Count')
                ax9.set_title('Energy Deposition Efficiency Distribution')
            ax9.grid(True, alpha=0.3)
    
    # 10. Feedback age distribution
    if parser.agb_events or parser.type_ii_events:
        ax10 = fig.add_subplot(gs[3, 2])
        
        agb_ages = [event.get('trigger_age', 0) for event in parser.agb_events 
                    if 'trigger_age' in event]
        typeii_ages = [event.get('trigger_age', 0) for event in parser.type_ii_events 
                       if 'trigger_age' in event]
        
        if agb_ages or typeii_ages:
            bins = np.linspace(0, 200, 40)
            if agb_ages:
                ax10.hist(agb_ages, bins=bins, alpha=0.5, color='blue', label=f'AGB (n={len(agb_ages)})')
            if typeii_ages:
                ax10.hist(typeii_ages, bins=bins, alpha=0.5, color='red', label=f'Type II (n={len(typeii_ages)})')
            
            # Add theoretical ranges
            ax10.axvspan(3.0, 40.0, alpha=0.1, color='red', label='Type II range')
            ax10.axvspan(50.0, 150.0, alpha=0.1, color='blue', label='AGB range')
            
            ax10.set_xlabel('Stellar Age at Feedback (Myr)')
            ax10.set_ylabel('Count')
            ax10.set_title('Age Distribution of Feedback Events')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
    
    # Save the figure
    plt.suptitle(f'Gadget-4 Feedback Diagnostics (Total outputs: {len(parser.star_summaries)})', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    print(f"Saved plots to {output_prefix}.png and {output_prefix}.pdf")
    
    # Create summary statistics
    create_summary_report(parser, output_prefix)

def create_summary_report(parser, output_prefix):
    """Create a text summary report of the feedback statistics."""
    
    with open(f'{output_prefix}_summary.txt', 'w') as f:
        f.write("Gadget-4 Feedback Diagnostics Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Star population statistics
        if parser.star_summaries:
            f.write("Star Population Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of outputs analyzed: {len(parser.star_summaries)}\n")
            
            total_stars_avg = np.mean([s['total'] for s in parser.star_summaries])
            total_stars_min = np.min([s['total'] for s in parser.star_summaries])
            total_stars_max = np.max([s['total'] for s in parser.star_summaries])
            
            type_ii_avg = np.mean([s['type_ii_range'] for s in parser.star_summaries])
            feedback_avg = np.mean([s['with_feedback'] for s in parser.star_summaries])
            
            f.write(f"Total stars: min={total_stars_min}, max={total_stars_max}, avg={total_stars_avg:.1f}\n")
            f.write(f"Average in Type II range (3-40 Myr): {type_ii_avg:.1f}\n")
            f.write(f"Average with feedback flag: {feedback_avg:.1f}\n\n")
        
        # Type II supernova statistics
        if parser.type_ii_events:
            f.write("Type II Supernova Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Type II events: {len(parser.type_ii_events)}\n")
            
            energies = [event['energy_ergs'] for event in parser.type_ii_events]
            f.write(f"Energy range: {min(energies):.2e} - {max(energies):.2e} erg\n")
            f.write(f"Mean energy: {np.mean(energies):.2e} erg\n")
            
            if any('trigger_age' in e for e in parser.type_ii_events):
                ages = [event['trigger_age'] for event in parser.type_ii_events if 'trigger_age' in event]
                f.write(f"Age range: {min(ages):.1f} - {max(ages):.1f} Myr\n")
                f.write(f"Mean age: {np.mean(ages):.1f} Myr\n\n")
        
        # AGB wind statistics
        if parser.agb_events:
            f.write("AGB Wind Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total AGB wind events: {len(parser.agb_events)}\n")
            
            energies = [event['energy_ergs'] for event in parser.agb_events]
            if len(set(energies)) == 1:
                f.write(f"Energy (constant): {energies[0]:.2e} erg\n")
            else:
                f.write(f"Energy range: {min(energies):.2e} - {max(energies):.2e} erg\n")
                f.write(f"Mean energy: {np.mean(energies):.2e} erg\n")
            
            if any('neighbors' in e for e in parser.agb_events):
                neighbors = [event['neighbors'] for event in parser.agb_events]
                f.write(f"Neighbors: min={min(neighbors)}, max={max(neighbors)}, avg={np.mean(neighbors):.1f}\n")
            
            if any('trigger_age' in e for e in parser.agb_events):
                ages = [event['trigger_age'] for event in parser.agb_events if 'trigger_age' in event]
                f.write(f"Age range: {min(ages):.1f} - {max(ages):.1f} Myr\n")
                f.write(f"Mean age: {np.mean(ages):.1f} Myr\n\n")
        
        # Timestep summary
        if parser.timestep_summaries:
            f.write("Feedback Event Summary:\n")
            f.write("-" * 30 + "\n")
            total_type_ii = sum(s['type_ii_count'] for s in parser.timestep_summaries)
            total_agb = sum(s['agb_count'] for s in parser.timestep_summaries)
            f.write(f"Total Type II SNe: {total_type_ii}\n")
            f.write(f"Total AGB winds: {total_agb}\n")
            f.write(f"Outputs with events: {len(parser.timestep_summaries)}\n")
            
            # Events per timestep statistics
            typeii_per_step = [s['type_ii_count'] for s in parser.timestep_summaries]
            agb_per_step = [s['agb_count'] for s in parser.timestep_summaries]
            
            if typeii_per_step:
                f.write(f"Type II per output: min={min(typeii_per_step)}, max={max(typeii_per_step)}, avg={np.mean(typeii_per_step):.2f}\n")
            if agb_per_step:
                f.write(f"AGB per output: min={min(agb_per_step)}, max={max(agb_per_step)}, avg={np.mean(agb_per_step):.2f}\n")
            
            f.write(f"\nFeedback model parameters (from code):\n")
            f.write(f"Type II age range: 3.0 - 40.0 Myr\n")
            f.write(f"AGB age range: 50.0 - 150.0 Myr\n")
            f.write(f"Type II energy: 1e51 erg per SN\n")
            f.write(f"AGB wind energy: 5e17 erg/Msun\n")
            f.write(f"SPH kernel: Cubic spline\n")
            f.write(f"Smoothing length: Sampled from local gas particles\n")
    
    print(f"Saved summary to {output_prefix}_summary.txt")

def main():
    parser = argparse.ArgumentParser(description='Plot Gadget-4 feedback diagnostics')
    parser.add_argument('input_file', help='Input file containing feedback output')
    parser.add_argument('-o', '--output', default='feedback_diagnostics', 
                       help='Output file prefix (default: feedback_diagnostics)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Parse the feedback data
    feedback_parser = FeedbackParser()
    print(f"Parsing {args.input_file}...")
    feedback_parser.parse_file(args.input_file)
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    create_diagnostic_plots(feedback_parser, args.output)
    
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()