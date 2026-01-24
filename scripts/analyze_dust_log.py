#!/usr/bin/env python3
"""
Parse Gadget-4 dust diagnostics from log file
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class DustLogParser:
    def __init__(self, logfile):
        self.logfile = logfile
        self.timestamps = []
        self.redshifts = []
        self.avg_grain_sizes = []
        self.avg_temperatures = []
        self.dust_counts = []
        self.dust_masses = []
        
        # Size distribution history
        self.size_distributions = []
        
        # Growth/erosion events
        self.growth_events = []
        self.erosion_events = []
        
    def parse(self):
        """Parse the entire log file"""
        with open(self.logfile, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Parse dust statistics blocks
            if "=== DUST STATISTICS ===" in line:
                self._parse_dust_stats_block(lines, i)
            
            # Parse grain size distribution blocks
            elif "=== GRAIN SIZE DISTRIBUTION ===" in line:
                self._parse_size_distribution_block(lines, i)
            
            # Parse individual growth events
            elif "[GRAIN_GROWTH]" in line:
                self._parse_growth_event(line)
            
            # Parse individual erosion events
            elif "[EROSION]" in line:
                self._parse_erosion_event(line)
            
            i += 1
    
    def _parse_dust_stats_block(self, lines, start_idx):
        """Parse a dust statistics block"""
        # Extract timestamp and redshift from line
        line = lines[start_idx]
        match = re.search(r'a=([\d.e+-]+)\s+z=([\d.]+)', line)
        if match:
            a = float(match.group(1))
            z = float(match.group(2))
            self.timestamps.append(a)
            self.redshifts.append(z)
        
        # Parse next few lines for statistics
        for offset in range(1, 8):
            if start_idx + offset >= len(lines):
                break
            line = lines[start_idx + offset]
            
            if "Particles:" in line:
                # Extract: "  Particles: 8432  Mass: 1.23e+09 Msun"
                match = re.search(r'Particles:\s+(\d+)\s+Mass:\s+([\d.e+-]+)', line)
                if match:
                    self.dust_counts.append(int(match.group(1)))
                    self.dust_masses.append(float(match.group(2)))
            
            elif "Avg grain size:" in line:
                # Extract: "  Avg grain size: 0.0842 μm"
                match = re.search(r'Avg grain size:\s+([\d.]+)', line)
                if match:
                    self.avg_grain_sizes.append(float(match.group(1)))
            
            elif "Avg temperature:" in line:
                # Extract: "  Avg temperature: 1250.5 K"
                match = re.search(r'Avg temperature:\s+([\d.]+)', line)
                if match:
                    self.avg_temperatures.append(float(match.group(1)))
    
    def _parse_size_distribution_block(self, lines, start_idx):
        """Parse a grain size distribution block"""
        distribution = {
            'time': None,
            'z': None,
            'bins': []
        }
        
        # Extract timestamp
        line = lines[start_idx]
        match = re.search(r'a=([\d.e+-]+)\s+z=([\d.]+)', line)
        if match:
            distribution['time'] = float(match.group(1))
            distribution['z'] = float(match.group(2))
        
        # Parse size bins
        for offset in range(1, 15):
            if start_idx + offset >= len(lines):
                break
            line = lines[start_idx + offset]
            
            # Match lines like: "  [0.010-0.050 μm]: 4521 grains (53.6%), 4.8e+08 Msun (42.3%)"
            match = re.search(r'\[([\d.]+)-([\d.]+)\s+μm\]:\s+(\d+)\s+grains\s+\(([\d.]+)%\),\s+([\d.e+-]+)\s+Msun\s+\(([\d.]+)%\)', line)
            if match:
                distribution['bins'].append({
                    'size_min': float(match.group(1)),
                    'size_max': float(match.group(2)),
                    'count': int(match.group(3)),
                    'count_frac': float(match.group(4)),
                    'mass': float(match.group(5)),
                    'mass_frac': float(match.group(6))
                })
        
        if distribution['bins']:
            self.size_distributions.append(distribution)
    
    def _parse_growth_event(self, line):
        """Parse a grain growth event"""
        # "[GRAIN_GROWTH] Event #1000: a=0.010→0.015 μm, dm=1.2e-06 Msun, Z=0.0250→0.0248"
        match = re.search(r'a=([\d.]+)→([\d.]+)\s+μm,\s+dm=([\d.e+-]+)', line)
        if match:
            self.growth_events.append({
                'size_before': float(match.group(1)),
                'size_after': float(match.group(2)),
                'dm': float(match.group(3))
            })
    
    def _parse_erosion_event(self, line):
        """Parse an erosion event"""
        # "[EROSION] Grain shrunk: 0.0850 → 0.0720 μm (dm=3.4e-07, T=1500000 K)"
        match = re.search(r'Grain shrunk:\s+([\d.]+)\s+→\s+([\d.]+)\s+μm.*T=([\d.]+)', line)
        if match:
            self.erosion_events.append({
                'size_before': float(match.group(1)),
                'size_after': float(match.group(2)),
                'temperature': float(match.group(3))
            })
    
    def plot_grain_size_evolution(self, outfile='grain_size_evolution.png'):
        """Plot average grain size vs time/redshift"""
        if not self.avg_grain_sizes:
            print("No grain size data found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot vs scale factor
        ax1.plot(self.timestamps, self.avg_grain_sizes, 'b-', linewidth=2)
        ax1.set_xlabel('Scale factor (a)', fontsize=12)
        ax1.set_ylabel('Average grain size [μm]', fontsize=12)
        ax1.set_title('Dust Grain Size Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        
        # Plot vs redshift
        ax2.plot(self.redshifts, self.avg_grain_sizes, 'r-', linewidth=2)
        ax2.set_xlabel('Redshift (z)', fontsize=12)
        ax2.set_ylabel('Average grain size [μm]', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
        plt.close()
    
    def plot_size_distribution_evolution(self, outfile='size_dist_evolution.png'):
        """Plot how the size distribution evolves"""
        if not self.size_distributions:
            print("No size distribution data found!")
            return
        
        # Take snapshots at different times
        n_snapshots = min(6, len(self.size_distributions))
        indices = np.linspace(0, len(self.size_distributions)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, dist_idx in enumerate(indices):
            dist = self.size_distributions[dist_idx]
            ax = axes[idx]
            
            # Extract bin centers and mass fractions
            bin_centers = [(b['size_min'] + b['size_max'])/2 for b in dist['bins']]
            mass_fracs = [b['mass_frac'] for b in dist['bins']]
            
            ax.bar(bin_centers, mass_fracs, width=0.02, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Grain size [μm]', fontsize=10)
            ax.set_ylabel('Mass fraction [%]', fontsize=10)
            ax.set_title(f"z = {dist['z']:.2f}", fontsize=11, fontweight='bold')
            ax.set_xlim(0, 0.35)
            ax.set_ylim(0, max(mass_fracs)*1.2 if mass_fracs else 100)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Grain Size Distribution Evolution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
        plt.close()
    
    def plot_growth_vs_erosion(self, outfile='growth_vs_erosion.png'):
        """Plot growth vs erosion statistics"""
        if not self.growth_events and not self.erosion_events:
            print("No growth/erosion event data found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Growth events: size change distribution
        if self.growth_events:
            size_changes = [e['size_after'] - e['size_before'] for e in self.growth_events]
            ax1.hist(size_changes, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax1.set_xlabel('Grain size increase [μm]', fontsize=12)
            ax1.set_ylabel('Number of events', fontsize=12)
            ax1.set_title(f'Grain Growth Events (N={len(self.growth_events)})', 
                         fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Erosion events: size change distribution
        if self.erosion_events:
            size_changes = [e['size_before'] - e['size_after'] for e in self.erosion_events]
            ax2.hist(size_changes, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax2.set_xlabel('Grain size decrease [μm]', fontsize=12)
            ax2.set_ylabel('Number of events', fontsize=12)
            ax2.set_title(f'Grain Erosion Events (N={len(self.erosion_events)})', 
                         fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
        plt.close()
    
    def print_summary(self):
        """Print a summary of the parsed data"""
        print("\n" + "="*60)
        print("DUST GRAIN SIZE ANALYSIS SUMMARY")
        print("="*60)
        
        if self.avg_grain_sizes:
            print(f"\nAverage Grain Size Evolution:")
            print(f"  Initial: {self.avg_grain_sizes[0]:.4f} μm (z={self.redshifts[0]:.2f})")
            print(f"  Final:   {self.avg_grain_sizes[-1]:.4f} μm (z={self.redshifts[-1]:.2f})")
            print(f"  Min:     {min(self.avg_grain_sizes):.4f} μm")
            print(f"  Max:     {max(self.avg_grain_sizes):.4f} μm")
        
        if self.dust_masses:
            print(f"\nDust Mass Evolution:")
            print(f"  Initial: {self.dust_masses[0]:.2e} Msun")
            print(f"  Final:   {self.dust_masses[-1]:.2e} Msun")
            print(f"  Peak:    {max(self.dust_masses):.2e} Msun")
        
        if self.growth_events:
            total_growth = sum(e['size_after'] - e['size_before'] for e in self.growth_events)
            print(f"\nGrain Growth Events:")
            print(f"  Total events: {len(self.growth_events)}")
            print(f"  Total growth: {total_growth:.4f} μm")
            print(f"  Avg per event: {total_growth/len(self.growth_events):.6f} μm")
        
        if self.erosion_events:
            total_erosion = sum(e['size_before'] - e['size_after'] for e in self.erosion_events)
            print(f"\nGrain Erosion Events:")
            print(f"  Total events: {len(self.erosion_events)}")
            print(f"  Total erosion: {total_erosion:.4f} μm")
            print(f"  Avg per event: {total_erosion/len(self.erosion_events):.6f} μm")
        
        if self.size_distributions:
            print(f"\nSize Distribution Snapshots: {len(self.size_distributions)}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_dust_grains.py <logfile>")
        sys.exit(1)
    
    logfile = sys.argv[1]
    
    print(f"Parsing {logfile}...")
    parser = DustLogParser(logfile)
    parser.parse()
    
    parser.print_summary()
    parser.plot_grain_size_evolution()
    parser.plot_size_distribution_evolution()
    parser.plot_growth_vs_erosion()
    
    print("\nDone! Generated plots:")
    print("  - grain_size_evolution.png")
    print("  - size_dist_evolution.png")
    print("  - growth_vs_erosion.png")
