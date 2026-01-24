#!/usr/bin/env python3
"""
Comprehensive Gadget-4 Simulation Health Check
Combines log analysis with snapshot HDF5 data analysis

Usage:
    python health_check.py output.log ./output_dir
    python health_check.py  # Uses defaults: output.log and current directory
"""

import sys
import os
import glob
import re
import numpy as np
import h5py
from datetime import datetime, timedelta
from pathlib import Path

# Physical constants
BOLTZMANN = 1.38064852e-16  # erg/K
PROTONMASS = 1.6726219e-24  # g
GAMMA = 5.0/3.0

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def format_runtime(seconds):
    """Format runtime as human-readable string."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def text_histogram(values, bins, labels, title="Histogram", width=40):
    """Create a text-based histogram with bars."""
    hist, _ = np.histogram(values, bins=bins)
    total = len(values)
    percentages = (hist / total * 100) if total > 0 else np.zeros_like(hist)
    
    max_count = hist.max() if hist.max() > 0 else 1
    
    lines = [f"\n  {title}:"]
    lines.append("  " + "="*60)
    
    for count, pct, label in zip(hist, percentages, labels):
        bar_len = int(width * count / max_count)
        bar = '█' * bar_len
        lines.append(f"  {label:20s} │{bar:<{width}s}│ {count:8,d} ({pct:5.1f}%)")
    
    lines.append("  " + "="*60)
    return "\n".join(lines)

def find_snapshots(output_dir):
    """Find all snapshot files."""
    snapshots = []
    
    # Look for snapdir_XXX directories
    snapdirs = glob.glob(os.path.join(output_dir, "snapdir_*"))
    if snapdirs:
        for snapdir in sorted(snapdirs):
            files = glob.glob(os.path.join(snapdir, "*.hdf5"))
            if files:
                snapshots.append(sorted(files)[0])  # Take first file from each snapdir
    else:
        # Look for snapshot_XXX.hdf5 files
        files = glob.glob(os.path.join(output_dir, "snapshot_*.hdf5"))
        snapshots = sorted(files)
    
    return snapshots

def read_snapshot_basic_info(snapshot_file):
    """Read basic info from snapshot."""
    try:
        with h5py.File(snapshot_file, 'r') as f:
            header = f['Header'].attrs
            redshift = header['Redshift']
            time = header['Time']
            
            n_gas = header['NumPart_ThisFile'][0] if 'NumPart_ThisFile' in header else 0
            n_stars = header['NumPart_ThisFile'][4] if 'NumPart_ThisFile' in header and len(header['NumPart_ThisFile']) > 4 else 0
            
            return {
                'redshift': redshift,
                'time': time,
                'n_gas': n_gas,
                'n_stars': n_stars
            }
    except Exception as e:
        return None

def analyze_gas_from_snapshot(snapshot_file):
    """Analyze gas properties from latest snapshot."""
    try:
        with h5py.File(snapshot_file, 'r') as f:
            if 'PartType0' not in f:
                return None
            
            gas = f['PartType0']
            
            # Read properties
            density = np.array(gas['Density'])
            internal_energy = np.array(gas['InternalEnergy'])
            masses = np.array(gas['Masses'])
            
            # Electron abundance
            if 'ElectronAbundance' in gas:
                ne = np.array(gas['ElectronAbundance'])
            else:
                ne = np.ones(len(masses))
            
            # Calculate temperature
            X_H = 0.76
            mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*ne)
            
            UnitVelocity = 1e5  # km/s to cm/s
            u_cgs = internal_energy * UnitVelocity**2
            temperature = u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN
            
            # Get physical density
            header = f['Header'].attrs
            time = header['Time']
            UnitMass = 1.989e43  # 10^10 Msun in g
            UnitLength = 3.085678e24  # Mpc in cm
            
            rho_cgs = density * (UnitMass / UnitLength**3)
            rho_physical = rho_cgs / time**3  # Convert comoving to physical
            
            return {
                'temperature': temperature,
                'density': rho_physical,
                'n_particles': len(temperature)
            }
    except Exception as e:
        print(f"    Warning: Could not analyze gas in {snapshot_file}: {e}")
        return None

def parse_log_file(log_file):
    """Parse log file for key information."""
    if not os.path.exists(log_file):
        return None
    
    data = {
        'sync_points': [],
        'star_census': [],
        'star_formation_events': [],  # New format
        'feedback_events': [],
        'energy_caps': 0,
        'temp_caps': 0,
        'load_balance': [],
        'terminates': [],
        'convergence_fails': [],
        'warnings': [],
        'performance': []  # New: timing data
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # Sync points (progress)
            if 'Sync-Point' in line and 'Redshift' in line:
                # Example: "Sync-Point 1287, Time: 0.061665, Redshift: 15.2166, Systemstep: ..."
                match = re.search(r'Time:\s+([\d.]+).*?Redshift:\s+([\d.]+)', line)
                if match:
                    data['sync_points'].append({
                        'time': float(match.group(1)),
                        'redshift': float(match.group(2)),
                        'line': line.strip()
                    })
            
            # Star formation census (old format)
            if 'census: Nstar' in line:
                match = re.search(r'Nstar=(\d+)', line)
                if match:
                    data['star_census'].append({
                        'nstar': int(match.group(1)),
                        'line': line.strip()
                    })
            
            # Star formation events (new format)
            # [STARFORMATION|T=0|a=0.1401 z=6.138] SFR: spawned 0 stars, converted 1 gas particles into stars
            if 'STARFORMATION' in line and 'SFR: spawned' in line:
                match = re.search(r'a=([\d.]+)\s+z=([\d.]+).*?spawned (\d+) stars.*?converted (\d+) gas', line)
                if match:
                    data['star_formation_events'].append({
                        'a': float(match.group(1)),
                        'z': float(match.group(2)),
                        'spawned': int(match.group(3)),
                        'converted': int(match.group(4))
                    })
            
            # Star formation mass info (new format)
            if 'STARFORMATION' in line and 'M*=' in line:
                match = re.search(r'a=([\d.]+)\s+z=([\d.]+).*?M\*=([\d.e+]+).*?M_halo=([\d.e+]+).*?M\*/M_halo=([\d.e+-]+)', line)
                if match:
                    # Add to most recent star formation event
                    if data['star_formation_events']:
                        data['star_formation_events'][-1].update({
                            'mstar': float(match.group(3)),
                            'mhalo': float(match.group(4)),
                            'fstar': float(match.group(5))
                        })
            
            # Feedback events - parse full details
            if 'FEEDBACK' in line and 'events:' in line:
                match = re.search(r'a=([\d.]+)\s+z=([\d.]+).*?SNII=(\d+)\s+\(([\d.e+]+)\s+erg\).*?AGB=(\d+)\s+\(([\d.e+]+)\s+erg\)', line)
                if match:
                    data['feedback_events'].append({
                        'a': float(match.group(1)),
                        'z': float(match.group(2)),
                        'snii_count': int(match.group(3)),
                        'snii_energy': float(match.group(4)),
                        'agb_count': int(match.group(5)),
                        'agb_energy': float(match.group(6)),
                        'line': line.strip()
                    })
            
            # Energy caps
            if 'ENERGY_CAP' in line:
                data['energy_caps'] += 1
            if 'TEMP_CONV_CAP' in line:
                data['temp_caps'] += 1
            
            # Load balance
            if 'maximum imbalance' in line:
                match = re.search(r'maximum imbalance.*?(\d+\.\d+)', line)
                if match:
                    data['load_balance'].append(float(match.group(1)))
            
            # Performance timing - Gadget prints timing info
            # Look for lines like: "domain | gravity | hydro | ..."
            if 'domain' in line.lower() and 'sec' in line.lower():
                data['performance'].append(line.strip())
            if 'gravity' in line.lower() and 'sec' in line.lower() and 'domain' not in line.lower():
                data['performance'].append(line.strip())
            if 'hydro' in line.lower() and 'sec' in line.lower():
                data['performance'].append(line.strip())
            
            # Errors
            if 'TERMINATE' in line:
                data['terminates'].append(line.strip())
            if 'failed to converge' in line:
                data['convergence_fails'].append(line.strip())
            if 'WARNING' in line and 'TERMINATE' not in line:
                data['warnings'].append(line.strip())
    
    return data

def print_header():
    """Print main header."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          GADGET-4 SIMULATION HEALTH CHECK                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

def print_runtime_stats(output_dir):
    """Print runtime statistics from snapshots."""
    print("┌─ RUNTIME ─────────────────────────────────────────────────────┐")
    
    snapshots = find_snapshots(output_dir)
    
    if len(snapshots) >= 2:
        time_first = os.path.getmtime(snapshots[0])
        time_last = os.path.getmtime(snapshots[-1])
        
        dt_first = datetime.fromtimestamp(time_first)
        dt_last = datetime.fromtimestamp(time_last)
        
        runtime = time_last - time_first
        
        print(f"  First snapshot: {dt_first.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Last snapshot:  {dt_last.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total runtime:  {format_runtime(runtime)}")
        
        if len(snapshots) > 1:
            avg_time = runtime / (len(snapshots) - 1)
            print(f"  Snapshots: {len(snapshots)} (avg {format_runtime(avg_time)} per snapshot)")
    else:
        print(f"  Snapshots found: {len(snapshots)}")
        if len(snapshots) == 1:
            dt = datetime.fromtimestamp(os.path.getmtime(snapshots[0]))
            print(f"  Snapshot time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_progress(log_data):
    """Print simulation progress."""
    print("┌─ PROGRESS ────────────────────────────────────────────────────┐")
    
    if log_data and len(log_data['sync_points']) > 0:
        first = log_data['sync_points'][0]
        last = log_data['sync_points'][-1]
        
        print(f"  Start: z={first['redshift']:.2f}, a={first['time']:.6f}")
        print(f"  Now:   z={last['redshift']:.2f}, a={last['time']:.6f}")
        
        # Calculate progress using cosmological time
        # For matter-dominated universe: t ∝ a^(3/2)
        # This accounts for the fact that z=1→0 is HALF the simulation time!
        if first['redshift'] > last['redshift']:
            a_start = first['time']
            a_now = last['time']
            a_end = 1.0  # z=0
            
            # Cosmological time is proportional to a^(3/2)
            t_start = a_start**(3.0/2.0)
            t_now = a_now**(3.0/2.0)
            t_end = a_end**(3.0/2.0)
            
            progress = (t_now - t_start) / (t_end - t_start) * 100
            
            print(f"  Progress: {progress:.1f}% complete (z={first['redshift']:.1f} → z=0)")
            print(f"            [Uses cosmological time: t ∝ a^(3/2)]")
        
        print()
        print("  Recent steps:")
        for sp in log_data['sync_points'][-3:]:
            print(f"    z={sp['redshift']:6.2f}, a={sp['time']:.6f}")
    else:
        print("  No progress data found in log")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_star_formation(log_data):
    """Print star formation status."""
    print("┌─ STAR FORMATION ──────────────────────────────────────────────┐")
    
    # Try new format first
    if log_data and len(log_data['star_formation_events']) > 0:
        latest = log_data['star_formation_events'][-1]
        
        print(f"  Latest event (z={latest['z']:.2f}):")
        print(f"    Spawned: {latest['spawned']} new star particles")
        print(f"    Converted: {latest['converted']} gas particles")
        
        if 'mstar' in latest:
            print(f"    M* = {latest['mstar']:.3e} M☉")
            print(f"    M_halo = {latest['mhalo']:.3e} M☉")
            print(f"    M*/M_halo = {latest['fstar']:.4e} ({latest['fstar']*100:.4f}%)")
        
        # Calculate totals
        total_spawned = sum(e['spawned'] for e in log_data['star_formation_events'])
        total_converted = sum(e['converted'] for e in log_data['star_formation_events'])
        
        print()
        print(f"  Totals since start:")
        print(f"    Total spawned: {total_spawned:,} star particles")
        print(f"    Total converted: {total_converted:,} gas particles")
        
        # Show recent history
        print()
        print("  Recent M* growth:")
        for event in log_data['star_formation_events'][-5:]:
            if 'mstar' in event:
                print(f"    z={event['z']:5.2f}  M*={event['mstar']:.3e} M☉  (f*={event['fstar']:.4e})")
            else:
                print(f"    z={event['z']:5.2f}  spawned={event['spawned']} converted={event['converted']}")
        
        # Check for stalls in conversion
        recent_converted = [e['converted'] for e in log_data['star_formation_events'][-10:]]
        if len(recent_converted) > 5 and all(c == 0 for c in recent_converted):
            print()
            print(f"  {Colors.WARNING}⚠️  WARNING: No gas particles converted in last {len(recent_converted)} outputs!{Colors.ENDC}")
    
    # Fall back to old format
    elif log_data and len(log_data['star_census']) > 0:
        latest = log_data['star_census'][-1]
        print(f"  Current: {latest['nstar']:,} stars")
        
        # Check for stalls
        recent_counts = [c['nstar'] for c in log_data['star_census'][-10:]]
        if len(set(recent_counts)) == 1 and recent_counts[0] > 0:
            print(f"  {Colors.WARNING}⚠️  WARNING: Star count unchanged for last {len(recent_counts)} outputs!{Colors.ENDC}")
        
        print()
        print("  Recent history:")
        for census in log_data['star_census'][-5:]:
            print(f"    {census['nstar']:6,} stars")
    else:
        print("  No star formation data found")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_feedback(log_data):
    """Print feedback events with detailed SNII vs AGB comparison."""
    print("┌─ FEEDBACK ────────────────────────────────────────────────────┐")
    
    if log_data and len(log_data['feedback_events']) > 0:
        events = log_data['feedback_events']
        
        print(f"  Total feedback events: {len(events)}")
        
        # Calculate cumulative statistics
        total_snii = sum(e['snii_count'] for e in events)
        total_agb = sum(e['agb_count'] for e in events)
        total_snii_energy = sum(e['snii_energy'] for e in events)
        total_agb_energy = sum(e['agb_energy'] for e in events)
        
        print()
        print("  ┌─ Event Counts ──────────────────────────────────┐")
        print(f"  │  SNII: {total_snii:7,}    AGB: {total_agb:7,}            │")
        if total_snii + total_agb > 0:
            snii_pct = total_snii / (total_snii + total_agb) * 100
            agb_pct = total_agb / (total_snii + total_agb) * 100
            print(f"  │  SNII: {snii_pct:6.2f}%    AGB: {agb_pct:6.2f}%           │")
        print("  └─────────────────────────────────────────────────┘")
        
        print()
        print("  ┌─ Energy Injected ───────────────────────────────┐")
        print(f"  │  SNII: {total_snii_energy:.3e} erg           │")
        print(f"  │  AGB:  {total_agb_energy:.3e} erg           │")
        if total_snii_energy + total_agb_energy > 0:
            snii_e_pct = total_snii_energy / (total_snii_energy + total_agb_energy) * 100
            agb_e_pct = total_agb_energy / (total_snii_energy + total_agb_energy) * 100
            print(f"  │  SNII: {snii_e_pct:6.2f}%    AGB: {agb_e_pct:6.2f}%           │")
            print(f"  │  Total: {total_snii_energy + total_agb_energy:.3e} erg       │")
        print("  └─────────────────────────────────────────────────┘")
        
        # Average energy per event
        if total_snii > 0 or total_agb > 0:
            print()
            print("  ┌─ Average Energy per Event ──────────────────────┐")
            if total_snii > 0:
                avg_snii = total_snii_energy / total_snii
                print(f"  │  SNII: {avg_snii:.3e} erg/event          │")
            if total_agb > 0:
                avg_agb = total_agb_energy / total_agb
                print(f"  │  AGB:  {avg_agb:.3e} erg/event          │")
            if total_snii > 0 and total_agb > 0:
                ratio = avg_snii / avg_agb
                print(f"  │  SNII/AGB ratio: {ratio:.2f}x                   │")
            print("  └─────────────────────────────────────────────────┘")
        
        # Recent activity
        print()
        print("  Recent events (last 5):")
        print("  ┌────────┬──────────┬──────────┬────────────────────────┐")
        print("  │   z    │   SNII   │   AGB    │   Energy Ratio (%)     │")
        print("  ├────────┼──────────┼──────────┼────────────────────────┤")
        for event in events[-5:]:
            total_e = event['snii_energy'] + event['agb_energy']
            if total_e > 0:
                snii_pct = event['snii_energy'] / total_e * 100
                agb_pct = event['agb_energy'] / total_e * 100
                ratio_str = f"SNII:{snii_pct:4.1f}% AGB:{agb_pct:4.1f}%"
            else:
                ratio_str = "No energy"
            print(f"  │ {event['z']:6.2f} │ {event['snii_count']:8,} │ {event['agb_count']:8,} │ {ratio_str:22s} │")
        print("  └────────┴──────────┴──────────┴────────────────────────┘")
        
    else:
        print("  No feedback events yet")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_energy_caps(log_data):
    """Print energy cap statistics."""
    print("┌─ ENERGY CAPS (Should be LOW!) ───────────────────────────────┐")
    
    if log_data:
        energy_caps = log_data['energy_caps']
        temp_caps = log_data['temp_caps']
        total = energy_caps + temp_caps
        
        print(f"  Total energy caps: {energy_caps}")
        print(f"  Total temp caps:   {temp_caps}")
        print("  ────────────────────────────")
        print(f"  TOTAL:             {total}")
        print()
        
        if total > 1000:
            print(f"  {Colors.FAIL}⚠️  CRITICAL: >1000 caps - feedback energy too high!{Colors.ENDC}")
        elif total > 100:
            print(f"  {Colors.WARNING}⚠️  WARNING: >100 caps - check feedback parameters{Colors.ENDC}")
        elif total > 10:
            print(f"  {Colors.WARNING}⚠️  CAUTION: >10 caps - monitor closely{Colors.ENDC}")
        else:
            print(f"  {Colors.OKGREEN}✓ Good (< 10 caps){Colors.ENDC}")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_load_balance(log_data):
    """Print load balance statistics."""
    print("┌─ LOAD BALANCE ────────────────────────────────────────────────┐")
    
    if log_data and len(log_data['load_balance']) > 0:
        avg = np.mean(log_data['load_balance'])
        print(f"  Average imbalance: {avg:.3f}")
        
        if avg < 1.3:
            print(f"  {Colors.OKGREEN}✓ GOOD (< 1.3){Colors.ENDC}")
        elif avg < 1.5:
            print(f"  {Colors.WARNING}⚠️  ACCEPTABLE (< 1.5){Colors.ENDC}")
        else:
            print(f"  {Colors.WARNING}⚠️  POOR (> 1.5){Colors.ENDC}")
        
        print()
        print("  Recent values:")
        for val in log_data['load_balance'][-3:]:
            print(f"    {val:.3f}")
    else:
        print("  No load balance data found")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_errors(log_data):
    """Print error statistics."""
    print("┌─ ERRORS & WARNINGS ───────────────────────────────────────────┐")
    
    if log_data:
        n_term = len(log_data['terminates'])
        n_conv = len(log_data['convergence_fails'])
        n_warn = len(log_data['warnings'])
        
        print(f"  Terminations:      {n_term}")
        print(f"  Convergence fails: {n_conv}")
        print(f"  Warnings:          {n_warn}")
        
        if n_term > 0:
            print()
            print(f"  {Colors.FAIL}❌ TERMINATE messages:{Colors.ENDC}")
            for term in log_data['terminates'][-3:]:
                truncated = term[:60] + "..." if len(term) > 60 else term
                print(f"    {truncated}")
        
        if n_conv > 0:
            print()
            print(f"  {Colors.WARNING}⚠️  Convergence failures:{Colors.ENDC}")
            for conv in log_data['convergence_fails'][-3:]:
                truncated = conv[:60] + "..." if len(conv) > 60 else conv
                print(f"    {truncated}")
        
        if n_term == 0 and n_conv == 0:
            print()
            print(f"  {Colors.OKGREEN}✓ No critical errors{Colors.ENDC}")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_gas_diagnostics(output_dir):
    """Print gas temperature and density diagnostics from latest snapshot."""
    print("┌─ GAS DIAGNOSTICS (Latest Snapshot) ──────────────────────────┐")
    
    snapshots = find_snapshots(output_dir)
    
    if not snapshots:
        print("  No snapshots found")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    # Analyze latest snapshot
    latest = snapshots[-1]
    print(f"  Analyzing: {os.path.basename(os.path.dirname(latest)) if 'snapdir' in latest else os.path.basename(latest)}")
    
    gas_data = analyze_gas_from_snapshot(latest)
    
    if gas_data is None:
        print("  Could not read gas data")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    temp = gas_data['temperature']
    rho = gas_data['density']
    
    print(f"  Gas particles: {gas_data['n_particles']:,}")
    print()
    
    # Temperature histogram
    temp_bins = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    temp_labels = [
        "< 1e3 K (Cold)",
        "1e3-1e4 K (Cool)",
        "1e4-1e5 K (Warm)",
        "1e5-1e6 K (Hot)",
        "1e6-1e7 K (V.Hot)",
        "1e7-1e8 K (X.Hot)",
        "> 1e8 K (Ultra)"
    ]
    print(text_histogram(temp, temp_bins, temp_labels, "Temperature Distribution", width=30))
    
    print()
    
    # Density histogram
    rho_bins = [1e-30, 1e-28, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-20]
    rho_labels = [
        "< 1e-28",
        "1e-28–1e-26",
        "1e-26–1e-25",
        "1e-25–1e-24",
        "1e-24–1e-23 (SF)",
        "1e-23–1e-22",
        "> 1e-22"
    ]
    print(text_histogram(rho, rho_bins, rho_labels, "Density Distribution [g/cm³]", width=30))
    
    print()
    
    # Key statistics
    hot_frac = (temp > 1e7).sum() / len(temp) * 100
    dense_frac = (rho > 1e-24).sum() / len(rho) * 100
    sf_ready = ((rho > 1e-24) & (temp < 1e5)).sum()
    
    print(f"  Hot gas (T>10⁷ K):    {hot_frac:5.1f}%")
    print(f"  Dense gas (ρ>10⁻²⁴):  {dense_frac:5.1f}%")
    print(f"  SF-ready (dense+cool): {sf_ready:,} particles")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_performance(log_data):
    """Print performance timing information from the simulation."""
    print("┌─ PERFORMANCE ─────────────────────────────────────────────────┐")
    
    if not log_data or len(log_data['performance']) == 0:
        print("  No detailed performance timing found in log")
        print()
        print("  Note: Gadget-4 typically prints timing info like:")
        print("    'domain | gravity | hydro | ...'")
        print("  If you see these lines, they'll appear here.")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    print(f"  Found {len(log_data['performance'])} performance entries")
    print()
    
    # Try to parse timing data from the performance lines
    # Gadget output varies, so we'll show what we found
    print("  Recent timing information:")
    for line in log_data['performance'][-10:]:
        # Truncate long lines
        truncated = line if len(line) < 60 else line[:57] + "..."
        print(f"    {truncated}")
    
    print()
    print("  Common performance bottlenecks to watch for:")
    print("    • Domain decomposition taking excessive time")
    print("      → May need better load balancing parameters")
    print("    • Gravity tree taking too long")
    print("      → Check tree opening criterion, softening lengths")
    print("    • Hydro neighbor finding slow")
    print("      → Check HSML values, neighbor search efficiency")
    print("    • Cooling dominating timestep")
    print("      → May need to adjust cooling timestep criterion")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()

def print_summary(log_data):
    """Print overall summary."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  SUMMARY                                                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    issues = []
    
    if log_data:
        # Check for issues
        total_caps = log_data['energy_caps'] + log_data['temp_caps']
        if total_caps > 100:
            issues.append(f"{Colors.WARNING}⚠️  High energy caps ({total_caps}) - check feedback parameters{Colors.ENDC}")
        
        # Check star formation stalls (try new format first)
        if len(log_data['star_formation_events']) > 10:
            recent = [e['converted'] for e in log_data['star_formation_events'][-10:]]
            if len(recent) > 5 and all(c == 0 for c in recent):
                issues.append(f"{Colors.WARNING}⚠️  Star formation appears stalled (no conversions){Colors.ENDC}")
        elif len(log_data['star_census']) > 10:
            recent = [c['nstar'] for c in log_data['star_census'][-10:]]
            if len(set(recent)) == 1 and recent[0] > 0:
                issues.append(f"{Colors.WARNING}⚠️  Star formation appears stalled{Colors.ENDC}")
        
        if len(log_data['terminates']) > 0:
            issues.append(f"{Colors.FAIL}❌ Simulation has TERMINATE errors{Colors.ENDC}")
        
        if len(log_data['load_balance']) > 0:
            avg = np.mean(log_data['load_balance'])
            if avg > 1.5:
                issues.append(f"{Colors.WARNING}⚠️  Poor load balance ({avg:.2f}){Colors.ENDC}")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print(f"{Colors.OKGREEN}✓ Simulation appears healthy!{Colors.ENDC}")
    
    print()

def main():
    # Parse arguments
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'output.log'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    # Print header
    print_header()
    
    # Parse log file
    log_data = parse_log_file(log_file)
    
    # Print all sections
    print_runtime_stats(output_dir)
    print_progress(log_data)
    print_star_formation(log_data)
    print_feedback(log_data)
    print_energy_caps(log_data)
    print_load_balance(log_data)
    print_performance(log_data)  # New performance section
    print_errors(log_data)
    print_gas_diagnostics(output_dir)
    print_summary(log_data)
    
    print(f"Rerun with: python {sys.argv[0]} {log_file} {output_dir}")
    print()

if __name__ == '__main__':
    main()
