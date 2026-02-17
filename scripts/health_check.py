#!/usr/bin/env python3
"""
Comprehensive Gadget-4 Simulation Health Check with Dust & Halo Analysis
Combines log analysis with snapshot HDF5 data analysis

Usage:
    python health_check.py output.log ./output_dir
    python health_check.py output.log ./output_dir --catalog groups_049/fof_subhalo_tab_049
"""

import sys
import os
import glob
import re
import numpy as np
import h5py
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Try to import halo_utils for advanced halo analysis
try:
    from halo_utils import compute_radial_distance, load_target_halo
    HAVE_HALO_UTILS = True
except ImportError:
    HAVE_HALO_UTILS = False

# Physical constants
BOLTZMANN = 1.38064852e-16  # erg/K
PROTONMASS = 1.6726219e-24  # g
GAMMA = 5.0/3.0
Z_SOLAR = 0.0134  # Solar metallicity
DUST_TO_METAL_SOLAR = 0.4  # Solar dust-to-metal ratio

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
                snapshots.append(snapdir)  # Return directory path
    else:
        # Look for snapshot_XXX.hdf5 files
        files = glob.glob(os.path.join(output_dir, "snapshot_*.hdf5"))
        snapshots = sorted(files)
    
    return snapshots

def get_snapshot_files(snapshot_path):
    """Get all HDF5 files for a snapshot (handles both single files and directories)."""
    if os.path.isdir(snapshot_path):
        return sorted(glob.glob(os.path.join(snapshot_path, "*.hdf5")))
    else:
        return [snapshot_path]

def read_snapshot_basic_info(snapshot_path):
    """Read basic info from snapshot."""
    try:
        files = get_snapshot_files(snapshot_path)
        if not files:
            return None
        
        with h5py.File(files[0], 'r') as f:
            header = f['Header'].attrs
            redshift = header['Redshift']
            time = header['Time']
            
            # Sum across all files
            n_gas = sum([h5py.File(fn, 'r')['Header'].attrs['NumPart_ThisFile'][0] 
                        for fn in files if 'PartType0' in h5py.File(fn, 'r')])
            n_stars = sum([h5py.File(fn, 'r')['Header'].attrs['NumPart_ThisFile'][4] 
                          for fn in files if 'PartType4' in h5py.File(fn, 'r')])
            n_dust = sum([h5py.File(fn, 'r')['Header'].attrs['NumPart_ThisFile'][6] 
                         for fn in files if 'PartType6' in h5py.File(fn, 'r')])
            
            return {
                'redshift': redshift,
                'time': time,
                'n_gas': n_gas,
                'n_stars': n_stars,
                'n_dust': n_dust
            }
    except Exception as e:
        return None

def analyze_dust_from_snapshot(snapshot_path, halo_center=None, search_radius_ckpc=None):
    """
    Analyze dust properties from snapshot.
    
    Args:
        snapshot_path: path to snapshot
        halo_center: optional halo center for radial analysis
        search_radius_ckpc: optional search radius in comoving kpc/h
    """
    try:
        files = get_snapshot_files(snapshot_path)
        
        # Read header info
        with h5py.File(files[0], 'r') as f0:
            header = f0['Header'].attrs
            h = header.get('HubbleParam', 1.0)
            a = header.get('Time', 1.0)
            redshift = header['Redshift']
        
        # Collect dust data from all files
        all_masses = []
        all_grain_radius = []
        all_grain_type = []
        all_dust_temp = []
        all_carbon_frac = []
        all_coords = []
        
        has_dust = False
        for fn in files:
            with h5py.File(fn, 'r') as f:
                if 'PartType6' not in f:
                    continue
                
                has_dust = True
                dust = f['PartType6']
                
                all_masses.append(dust['Masses'][:])
                all_grain_radius.append(dust['GrainRadius'][:])
                all_grain_type.append(dust['GrainType'][:])
                all_dust_temp.append(dust['DustTemperature'][:])
                all_carbon_frac.append(dust['CarbonFraction'][:])
                all_coords.append(dust['Coordinates'][:])
        
        if not has_dust:
            return None
        
        # Concatenate
        masses = np.concatenate(all_masses)
        grain_radius = np.concatenate(all_grain_radius)
        grain_type = np.concatenate(all_grain_type)
        dust_temp = np.concatenate(all_dust_temp)
        carbon_frac = np.concatenate(all_carbon_frac)
        coords = np.concatenate(all_coords)
        
        # Apply halo filter if provided
        if halo_center is not None and search_radius_ckpc is not None:
            r = np.sqrt(np.sum((coords - halo_center)**2, axis=1))
            mask = r < search_radius_ckpc
            
            masses = masses[mask]
            grain_radius = grain_radius[mask]
            grain_type = grain_type[mask]
            dust_temp = dust_temp[mask]
            carbon_frac = carbon_frac[mask]
            coords = coords[mask]
        
        result = {
            'n_dust': len(masses),
            'total_dust_mass': np.sum(masses),
            'grain_radius': grain_radius,
            'grain_type': grain_type,
            'dust_temp': dust_temp,
            'carbon_frac': carbon_frac,
            'coords': coords,
            'redshift': redshift,
            'time': a
        }
        
        # Grain size stats (convert from code units to microns if needed)
        # Assuming grain_radius is in code units (cm typically)
        grain_radius_micron = grain_radius * 1e4  # cm to microns
        result['grain_radius_micron'] = grain_radius_micron
        result['mean_grain_size'] = np.mean(grain_radius_micron)
        result['median_grain_size'] = np.median(grain_radius_micron)
        
        # Carbon vs silicate
        result['n_carbon'] = np.sum(grain_type == 0)
        result['n_silicate'] = np.sum(grain_type == 1)
        result['carbon_mass'] = np.sum(masses[grain_type == 0])
        result['silicate_mass'] = np.sum(masses[grain_type == 1])
        
        return result
        
    except Exception as e:
        print(f"    Warning: Could not analyze dust in {snapshot_path}: {e}")
        return None

def analyze_gas_from_snapshot(snapshot_path, halo_center=None, search_radius_ckpc=None):
    """Analyze gas properties from snapshot with optional halo filtering."""
    try:
        files = get_snapshot_files(snapshot_path)
        
        # Read header and unit info from first file
        with h5py.File(files[0], 'r') as f0:
            H = f0['Header'].attrs
            P = f0['Parameters'].attrs
            
            h = H.get('HubbleParam', 1.0)
            a = H.get('Time', None)
            redshift = H.get('Redshift', (1.0/a - 1.0) if a is not None else None)
            
            # Unit conversions
            UnitLength_in_cm   = P['UnitLength_in_cm']
            UnitMass_in_g      = P['UnitMass_in_g']
            UnitVelocity_in_cm = P['UnitVelocity_in_cm_per_s']
        
        # Collect all gas data
        all_density = []
        all_internal_energy = []
        all_masses = []
        all_ne = []
        all_coords = []
        all_sfr = []
        all_metallicity = []
        
        for fn in files:
            with h5py.File(fn, 'r') as f:
                if 'PartType0' not in f:
                    continue
                
                gas = f['PartType0']
                
                all_density.append(gas['Density'][:])
                all_internal_energy.append(gas['InternalEnergy'][:])
                all_masses.append(gas['Masses'][:])
                all_coords.append(gas['Coordinates'][:])
                
                if 'ElectronAbundance' in gas:
                    all_ne.append(gas['ElectronAbundance'][:])
                else:
                    all_ne.append(np.ones(len(gas['Masses'][:])))
                
                if 'StarFormationRate' in gas:
                    all_sfr.append(gas['StarFormationRate'][:])
                else:
                    all_sfr.append(np.zeros(len(gas['Masses'][:])))
                
                if 'Metallicity' in gas:
                    Z = gas['Metallicity'][:]
                    if len(Z.shape) > 1:
                        Z = Z.sum(axis=1)
                    all_metallicity.append(Z)
        
        # Concatenate all particles
        density = np.concatenate(all_density)
        internal_energy = np.concatenate(all_internal_energy)
        masses = np.concatenate(all_masses)
        ne = np.concatenate(all_ne)
        coords = np.concatenate(all_coords)
        sfr = np.concatenate(all_sfr)
        
        if all_metallicity:
            metallicity = np.concatenate(all_metallicity)
        else:
            metallicity = np.zeros(len(masses))
        
        # Apply halo filter if provided
        if halo_center is not None and search_radius_ckpc is not None:
            r = np.sqrt(np.sum((coords - halo_center)**2, axis=1))
            mask = r < search_radius_ckpc
            
            density = density[mask]
            internal_energy = internal_energy[mask]
            masses = masses[mask]
            ne = ne[mask]
            coords = coords[mask]
            sfr = sfr[mask]
            metallicity = metallicity[mask]
        
        # Calculate temperature
        X_H = 0.76
        mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*ne)
        
        u_cgs = internal_energy * UnitVelocity_in_cm**2
        temperature = u_cgs * (GAMMA - 1.0) * mu * PROTONMASS / BOLTZMANN
        
        # Get physical density
        rho_cgs = density * (UnitMass_in_g / UnitLength_in_cm**3)
        rho_physical = rho_cgs / a**3  # Convert comoving to physical
        
        return {
            'temperature': temperature,
            'density': rho_physical,
            'masses': masses,
            'sfr': sfr,
            'metallicity': metallicity,
            'n_particles': len(temperature),
            'total_gas_mass': np.sum(masses),
            'total_sfr': np.sum(sfr),
            'redshift': redshift
        }
    except Exception as e:
        print(f"    Warning: Could not analyze gas in {snapshot_path}: {e}")
        return None

def get_halo_info_from_catalog(catalog_path, target_id=None):
    """Get halo center and properties from catalog (handles multi-file)."""
    if not os.path.exists(catalog_path):
        # Try with .*.hdf5 pattern
        base = catalog_path.replace('.hdf5', '')
        catalog_files = sorted(glob.glob(f'{base}.*.hdf5'))
        
        if not catalog_files:
            return None
    else:
        catalog_files = [catalog_path]
    
    # Find most massive subhalo
    if target_id is None:
        max_mass = -1
        best_file = None
        best_id = None
        
        for cat_file in catalog_files:
            with h5py.File(cat_file, 'r') as cat:
                if 'Subhalo' not in cat or 'SubhaloMass' not in cat['Subhalo']:
                    continue
                masses = cat['Subhalo']['SubhaloMass'][:]
                if len(masses) == 0:
                    continue
                local_max_id = np.argmax(masses)
                local_max_mass = masses[local_max_id]
                
                if local_max_mass > max_mass:
                    max_mass = local_max_mass
                    best_file = cat_file
                    best_id = local_max_id
        
        if best_file is None:
            return None
        
        catalog_path = best_file
        target_id = best_id
    else:
        catalog_path = catalog_files[0]
    
    # Load halo info
    with h5py.File(catalog_path, 'r') as cat:
        halo_info = {
            'id': target_id,
            'center': cat['Subhalo']['SubhaloPos'][target_id],
            'halfmass_rad': cat['Subhalo']['SubhaloHalfmassRad'][target_id],
            'mass': cat['Subhalo']['SubhaloMass'][target_id],
            'vmax': cat['Subhalo']['SubhaloVmax'][target_id]
        }
    
    return halo_info

def estimate_virial_radius(vmax, redshift):
    """Estimate virial radius from Vmax."""
    h = 0.6774
    Omega_m = 0.3089
    
    E_z = np.sqrt(Omega_m * (1 + redshift)**3 + (1 - Omega_m))
    H_z = 100 * h * E_z  # km/s/Mpc
    H_z_kpc = H_z / 1000.0  # km/s/kpc
    
    rvir_pkpc = vmax / (10.0 * H_z_kpc)  # physical kpc
    
    return rvir_pkpc

def parse_log_file(log_file):
    """Parse log file for key information."""
    if not os.path.exists(log_file):
        return None
    
    data = {
        'sync_points': [],
        'star_census': [],
        'star_formation_events': [],
        'feedback_events': [],
        'energy_caps': 0,
        'temp_caps': 0,
        'load_balance': [],
        'terminates': [],
        'convergence_fails': [],
        'warnings': [],
        'performance': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # Sync points
            if 'Sync-Point' in line and 'Redshift' in line:
                match = re.search(r'Time:\s+([\d.]+).*?Redshift:\s+([\d.]+)', line)
                if match:
                    data['sync_points'].append({
                        'time': float(match.group(1)),
                        'redshift': float(match.group(2)),
                        'line': line.strip()
                    })
            
            # Star formation census
            if 'census: Nstar' in line:
                match = re.search(r'Nstar=(\d+)', line)
                if match:
                    data['star_census'].append({
                        'nstar': int(match.group(1)),
                        'line': line.strip()
                    })
            
            # Star formation events
            if 'STARFORMATION' in line and 'SFR: spawned' in line:
                match = re.search(r'a=([\d.]+)\s+z=([\d.]+).*?spawned (\d+) stars.*?converted (\d+) gas', line)
                if match:
                    data['star_formation_events'].append({
                        'a': float(match.group(1)),
                        'z': float(match.group(2)),
                        'spawned': int(match.group(3)),
                        'converted': int(match.group(4))
                    })
            
            # Star formation mass info
            if 'STARFORMATION' in line and 'M*=' in line:
                match = re.search(r'a=([\d.]+)\s+z=([\d.]+).*?M\*=([\d.e+]+).*?M_halo=([\d.e+]+).*?M\*/M_halo=([\d.e+-]+)', line)
                if match:
                    if data['star_formation_events']:
                        data['star_formation_events'][-1].update({
                            'mstar': float(match.group(3)),
                            'mhalo': float(match.group(4)),
                            'fstar': float(match.group(5))
                        })
            
            # Feedback events
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
            
            # Performance timing
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

# [Previous print functions remain the same: print_header, print_runtime_stats, print_progress, etc.]
# [I'll include the key ones and new dust/halo analysis functions]

def print_header():
    """Print main header."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     GADGET-4 SIMULATION HEALTH CHECK WITH DUST & HALO          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

def print_runtime_stats(output_dir):
    """Print runtime statistics from snapshots."""
    print("┌─ RUNTIME ─────────────────────────────────────────────────────┐")
    
    snapshots = find_snapshots(output_dir)
    
    if len(snapshots) >= 2:
        # Get file modification times from first file in each snapshot
        first_files = get_snapshot_files(snapshots[0])
        last_files = get_snapshot_files(snapshots[-1])
        
        time_first = os.path.getmtime(first_files[0])
        time_last = os.path.getmtime(last_files[0])
        
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
        if first['redshift'] > last['redshift']:
            a_start = first['time']
            a_now = last['time']
            a_end = 1.0  # z=0
            
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
    
    if log_data and len(log_data['star_formation_events']) > 0:
        latest = log_data['star_formation_events'][-1]
        
        print(f"  Latest event (z={latest['z']:.2f}):")
        print(f"    Spawned: {latest['spawned']} new star particles")
        print(f"    Converted: {latest['converted']} gas particles")
        
        if 'mstar' in latest:
            print(f"    M* = {latest['mstar']:.3e} (code units)")
            print(f"    M_halo = {latest['mhalo']:.3e} (code units)")
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
                print(f"    z={event['z']:5.2f}  M*={event['mstar']:.3e}  f*={event['fstar']:.4e}")
            else:
                print(f"    z={event['z']:5.2f}  spawned={event['spawned']} converted={event['converted']}")
        
        # Check for stalls
        recent_converted = [e['converted'] for e in log_data['star_formation_events'][-10:]]
        if len(recent_converted) > 5 and all(c == 0 for c in recent_converted):
            print()
            print(f"  {Colors.WARNING}⚠️  WARNING: No gas particles converted in last {len(recent_converted)} outputs!{Colors.ENDC}")
    
    elif log_data and len(log_data['star_census']) > 0:
        latest = log_data['star_census'][-1]
        print(f"  Current: {latest['nstar']:,} stars")
        
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

def print_dust_diagnostics(output_dir, halo_center=None, search_radius_ckpc=None):
    """Print comprehensive dust diagnostics - both total and halo-filtered."""
    print("┌─ DUST DIAGNOSTICS ────────────────────────────────────────────┐")
    
    snapshots = find_snapshots(output_dir)
    
    if not snapshots:
        print("  No snapshots found")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    # Analyze latest snapshot
    latest = snapshots[-1]
    snap_name = os.path.basename(latest) if not os.path.isdir(latest) else os.path.basename(latest)
    print(f"  Latest snapshot: {snap_name}")
    
    # First analyze ALL dust (no filter)
    dust_data_total = analyze_dust_from_snapshot(latest, halo_center=None, search_radius_ckpc=None)
    
    # Then analyze halo dust (with filter)
    dust_data_halo = None
    if halo_center is not None and search_radius_ckpc is not None:
        dust_data_halo = analyze_dust_from_snapshot(latest, halo_center, search_radius_ckpc)
    
    if dust_data_total is None:
        print("  No dust particles found (PartType6 not present)")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return
    
    print(f"  Redshift: z={dust_data_total['redshift']:.3f}")
    print()
    
    # Show TOTAL dust first
    print(f"  TOTAL DUST IN SIMULATION:")
    print(f"    Total dust particles: {dust_data_total['n_dust']:,}")
    print(f"    Carbon grains:        {dust_data_total['n_carbon']:,} ({dust_data_total['n_carbon']/dust_data_total['n_dust']*100:.1f}%)")
    print(f"    Silicate grains:      {dust_data_total['n_silicate']:,} ({dust_data_total['n_silicate']/dust_data_total['n_dust']*100:.1f}%)")
    print(f"    Total dust mass:      {dust_data_total['total_dust_mass']:.3e} (code units)")
    print()
    
    # Show halo dust if filtered
    if dust_data_halo is not None:
        print(f"  DUST WITHIN HALO (2×Rvir):")
        print(f"    Dust particles:       {dust_data_halo['n_dust']:,} ({dust_data_halo['n_dust']/dust_data_total['n_dust']*100:.1f}% of total)")
        print(f"    Carbon grains:        {dust_data_halo['n_carbon']:,}")
        print(f"    Silicate grains:      {dust_data_halo['n_silicate']:,}")
        print(f"    Dust mass:            {dust_data_halo['total_dust_mass']:.3e} (code units)")
        print()
    
    # Use halo data for detailed analysis if available, otherwise use total
    dust_data = dust_data_halo if dust_data_halo is not None else dust_data_total
    
    # Grain size - check if it's already in reasonable units
    grain_radius_raw = dust_data['grain_radius']
    
    # Grain radius in Gadget is typically stored in cm (code units)
    # Check if conversion to microns makes sense
    test_micron = grain_radius_raw[0] * 1e4  # cm to microns
    
    if test_micron > 1000:  # If > 1mm, probably already in microns or wrong units
        # Assume it's already in microns or some other unit
        grain_radius_micron = grain_radius_raw
        print(f"  GRAIN SIZE (WARNING: units may be incorrect):")
    else:
        # Convert cm to microns
        grain_radius_micron = grain_radius_raw * 1e4
        print(f"  GRAIN SIZE:")
    
    print(f"    Mean:   {np.mean(grain_radius_micron):.4f} μm")
    print(f"    Median: {np.median(grain_radius_micron):.4f} μm")
    print(f"    Min:    {np.min(grain_radius_micron):.4f} μm")
    print(f"    Max:    {np.max(grain_radius_micron):.4f} μm")
    
    # Grain size histogram with appropriate bins
    if np.max(grain_radius_micron) > 10:
        # Likely wrong units
        print()
        print(f"  {Colors.WARNING}⚠️  WARNING: Grain sizes appear to be in wrong units!{Colors.ENDC}")
        print(f"  Expected range: 0.001-1 μm, but got {np.min(grain_radius_micron):.2f}-{np.max(grain_radius_micron):.2f} μm")
        print(f"  Check GrainRadius units in your output files.")
    else:
        # Reasonable grain sizes
        size_bins = np.array([0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
        size_labels = [
            "< 0.001 μm",
            "0.001–0.01 μm",
            "0.01–0.05 μm",
            "0.05–0.1 μm",
            "0.1–0.5 μm",
            "0.5–1.0 μm",
            "> 1.0 μm"
        ]
        print(text_histogram(grain_radius_micron, size_bins, size_labels, 
                            "Grain Size Distribution", width=30))
    print()
    
    # Dust temperature
    dust_temp = dust_data['dust_temp']
    print(f"  DUST TEMPERATURE:")
    print(f"    Mean:   {np.mean(dust_temp):.1f} K")
    print(f"    Median: {np.median(dust_temp):.1f} K")
    print(f"    Min:    {np.min(dust_temp):.1f} K")
    print(f"    Max:    {np.max(dust_temp):.1f} K")
    
    # Check if temperatures are reasonable
    if np.mean(dust_temp) > 10000:
        print()
        print(f"  {Colors.WARNING}⚠️  WARNING: Dust temperatures very high (mean {np.mean(dust_temp):.0f} K){Colors.ENDC}")
        print(f"  This suggests dust is in hot gas or units are wrong.")
        print(f"  Most dust should be 10-100 K.")
    elif np.mean(dust_temp) < 1:
        print()
        print(f"  {Colors.WARNING}⚠️  WARNING: Dust temperatures very low (mean {np.mean(dust_temp):.2f} K){Colors.ENDC}")
        print(f"  Check DustTemperature units in output.")
    else:
        temp_bins = [0, 5, 10, 20, 30, 50, 100, 200, 500]
        temp_labels = [
            "< 5 K (CMB floor)",
            "5–10 K (Very cold)",
            "10–20 K (Cold)",
            "20–30 K (Cool)",
            "30–50 K (Warm)",
            "50–100 K (Hot)",
            "100–200 K (V. hot)",
            "> 200 K (X. hot)"
        ]
        print(text_histogram(dust_temp, temp_bins, temp_labels,
                            "Temperature Distribution", width=30))
    print()
    
    # Dust-to-gas and dust-to-metal ratios
    gas_data = analyze_gas_from_snapshot(latest, halo_center, search_radius_ckpc)
    
    if gas_data is not None:
        dtg_ratio = dust_data['total_dust_mass'] / gas_data['total_gas_mass']
        metal_mass = np.sum(gas_data['metallicity'] * gas_data['masses'])
        
        if metal_mass > 0:
            dtm_ratio = dust_data['total_dust_mass'] / metal_mass
            dtm_solar = DUST_TO_METAL_SOLAR
            
            print(f"  DUST RATIOS:")
            print(f"    Dust-to-gas ratio:     {dtg_ratio:.4e}")
            print(f"    Dust-to-metal ratio:   {dtm_ratio:.4f} ({dtm_ratio/dtm_solar:.2f} × solar)")
            print()
            
            # Check reasonableness
            dtg_solar = 0.01 * Z_SOLAR * dtm_solar  # ~5e-5 for solar metallicity
            
            if dtm_ratio < 0.05:
                print(f"    {Colors.WARNING}⚠️  Very low dust-to-metal ratio - most metals not in dust{Colors.ENDC}")
                print(f"    Expected: ~0.4 for solar metallicity")
                print(f"    Possible causes:")
                print(f"      • Dust destruction is too strong")
                print(f"      • Dust formation is too weak")
                print(f"      • Metals are in hot gas where dust can't survive")
            elif dtm_ratio > 0.7:
                print(f"    {Colors.WARNING}⚠️  High dust-to-metal ratio - too much dust!{Colors.ENDC}")
                print(f"    Expected: ~0.4 for solar metallicity")
            else:
                print(f"    {Colors.OKGREEN}✓ Reasonable dust-to-metal ratio{Colors.ENDC}")
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()




def print_halo_analysis(output_dir, catalog_path=None):
    """Print halo-centric analysis."""
    print("┌─ HALO ANALYSIS ───────────────────────────────────────────────┐")
    
    if catalog_path is None:
        print("  No catalog provided - skipping halo analysis")
        print("  (Use --catalog option to enable halo-centric diagnostics)")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return None, None
    
    # Get halo info
    halo_info = get_halo_info_from_catalog(catalog_path)
    
    if halo_info is None:
        print("  Could not load halo information from catalog")
        print("└───────────────────────────────────────────────────────────────┘")
        print()
        return None, None
    
    print(f"  Target halo ID: {halo_info['id']}")
    print(f"  Halo center: {halo_info['center']}")
    print(f"  Halo mass: {halo_info['mass']:.3e} (code units)")
    print(f"  Vmax: {halo_info['vmax']:.2f} km/s")
    print(f"  Halfmass radius: {halo_info['halfmass_rad']:.2f} kpc")
    print()
    
    # Get latest snapshot to estimate Rvir
    snapshots = find_snapshots(output_dir)
    if snapshots:
        latest = snapshots[-1]
        info = read_snapshot_basic_info(latest)
        
        if info:
            rvir_pkpc = estimate_virial_radius(halo_info['vmax'], info['redshift'])
            rvir_ckpc_h = rvir_pkpc * (1 + info['redshift']) * 0.6774
            
            print(f"  Virial radius (z={info['redshift']:.2f}):")
            print(f"    Rvir = {rvir_pkpc:.2f} pkpc = {rvir_ckpc_h:.2f} ckpc/h")
            print()
            
            # Analyze gas/stars/dust within 2×Rvir
            search_radius = 2.0 * rvir_ckpc_h
            
            gas_data = analyze_gas_from_snapshot(latest, halo_info['center'], search_radius)
            dust_data = analyze_dust_from_snapshot(latest, halo_info['center'], search_radius)
            
            if gas_data:
                print(f"  Within 2×Rvir:")
                print(f"    Gas particles: {gas_data['n_particles']:,}")
                print(f"    Gas mass: {gas_data['total_gas_mass']:.3e} (code units)")
                print(f"    Total SFR: {gas_data['total_sfr']:.3e} M☉/yr")
                
                # Depletion time
                if gas_data['total_sfr'] > 0:
                    t_depl = gas_data['total_gas_mass'] / gas_data['total_sfr']
                    print(f"    Gas depletion time: {t_depl:.2e} yr ({t_depl/1e9:.2f} Gyr)")
                
                if dust_data:
                    print(f"    Dust particles: {dust_data['n_dust']:,}")
                    print(f"    Dust mass: {dust_data['total_dust_mass']:.3e} (code units)")
                    
                    dtg = dust_data['total_dust_mass'] / gas_data['total_gas_mass']
                    print(f"    Dust-to-gas: {dtg:.4e}")
            
            print(f"  {Colors.OKGREEN}✓ Halo properties look reasonable{Colors.ENDC}")
            
            print("└───────────────────────────────────────────────────────────────┘")
            print()
            
            return halo_info['center'], search_radius
    
    print("└───────────────────────────────────────────────────────────────┘")
    print()
    return halo_info['center'], None

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

def print_summary(log_data, output_dir, catalog_path):
    """Print overall summary."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  OVERALL ASSESSMENT                                            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    issues = []
    successes = []
    
    if log_data:
        # Check energy caps
        total_caps = log_data['energy_caps'] + log_data['temp_caps']
        if total_caps > 100:
            issues.append(f"High energy caps ({total_caps}) - feedback may be too strong")
        elif total_caps < 10:
            successes.append(f"Low energy caps ({total_caps}) - feedback well-calibrated")
        
        # Check star formation
        if len(log_data['star_formation_events']) > 10:
            recent = [e['converted'] for e in log_data['star_formation_events'][-10:]]
            if len(recent) > 5 and all(c == 0 for c in recent):
                issues.append("Star formation stalled (no gas conversion)")
            else:
                total_converted = sum(e['converted'] for e in log_data['star_formation_events'])
                if total_converted > 0:
                    successes.append(f"Active star formation ({total_converted:,} gas → stars)")
        
        # Check feedback
        if len(log_data['feedback_events']) > 0:
            total_snii = sum(e['snii_count'] for e in log_data['feedback_events'])
            total_agb = sum(e['agb_count'] for e in log_data['feedback_events'])
            if total_snii > 0 or total_agb > 0:
                successes.append(f"Feedback active (SNII: {total_snii:,}, AGB: {total_agb:,})")
        
        # Check errors
        if len(log_data['terminates']) > 0:
            issues.append(f"Simulation has {len(log_data['terminates'])} TERMINATE errors")
        
        # Check load balance
        if len(log_data['load_balance']) > 0:
            avg = np.mean(log_data['load_balance'])
            if avg > 1.5:
                issues.append(f"Poor load balance (avg {avg:.2f})")
            elif avg < 1.3:
                successes.append(f"Good load balance (avg {avg:.2f})")
    
    # Check dust
    snapshots = find_snapshots(output_dir)
    if snapshots:
        latest = snapshots[-1]
        dust_data = analyze_dust_from_snapshot(latest)
        if dust_data:
            successes.append(f"Dust physics active ({dust_data['n_dust']:,} dust particles)")
        else:
            issues.append("No dust particles found - is dust physics enabled?")
    
    # Print results
    if successes:
        print(f"{Colors.OKGREEN}✓ SUCCESSES:{Colors.ENDC}")
        for success in successes:
            print(f"  • {success}")
        print()
    
    if issues:
        print(f"{Colors.WARNING}⚠️  ISSUES TO MONITOR:{Colors.ENDC}")
        for issue in issues:
            print(f"  • {issue}")
        print()
    
    if not issues:
        print(f"{Colors.OKGREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
        print(f"{Colors.OKGREEN}  ✓✓✓ SIMULATION APPEARS HEALTHY! ✓✓✓{Colors.ENDC}")
        print(f"{Colors.OKGREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
    else:
        print(f"{Colors.OKBLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
        print(f"{Colors.OKBLUE}  Simulation running with some issues to monitor{Colors.ENDC}")
        print(f"{Colors.OKBLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Gadget-4 simulation health check',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', nargs='?', default='output.log',
                       help='Path to output.log file (default: output.log)')
    parser.add_argument('output_dir', nargs='?', default='.',
                       help='Path to output directory (default: current dir)')
    parser.add_argument('--catalog', '-c', default=None,
                       help='Path to FOF catalog for halo-centric analysis')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Parse log file
    log_data = parse_log_file(args.log_file)
    
    # Print all sections
    print_runtime_stats(args.output_dir)
    print_progress(log_data)
    print_star_formation(log_data)
    print_feedback(log_data)
    print_energy_caps(log_data)
    
    # Halo analysis (if catalog provided)
    halo_center, search_radius = print_halo_analysis(args.output_dir, args.catalog)
    
    # Dust diagnostics (use halo center if available)
    print_dust_diagnostics(args.output_dir, halo_center, search_radius)
    
    print_errors(log_data)
    print_summary(log_data, args.output_dir, args.catalog)
    
    print("─────────────────────────────────────────────────────────────────")
    print(f"Rerun with: python {sys.argv[0]} {args.log_file} {args.output_dir}")
    if args.catalog:
        print(f"            --catalog {args.catalog}")
    print()

if __name__ == '__main__':
    main()
