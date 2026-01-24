#!/usr/bin/env python3
"""
Analyze Gadget-4 HDF5 FOF/SUBFIND catalogs to find zoom targets
Usage: python analyze_halos_hdf5.py output_parent/
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import sys

def read_fof_hdf5(filename):
    """Read Gadget-4 HDF5 FOF catalog"""
    try:
        with h5py.File(filename, 'r') as f:
            # Print available groups to debug
            print(f"Groups in {filename}: {list(f.keys())}")
            
            # Try different possible group names
            group_names = ['Group', 'FOF', 'Halo']
            data = {}
            
            for group_name in group_names:
                if group_name in f:
                    grp = f[group_name]
                    print(f"Found group: {group_name}")
                    print(f"Datasets: {list(grp.keys())}")
                    
                    # Read common FOF properties
                    if 'GroupMass' in grp:
                        data['mass'] = grp['GroupMass'][:]
                    elif 'Mass' in grp:
                        data['mass'] = grp['Mass'][:]
                        
                    if 'GroupPos' in grp:
                        data['pos'] = grp['GroupPos'][:]
                    elif 'CenterOfMass' in grp:
                        data['pos'] = grp['CenterOfMass'][:]
                        
                    if 'GroupLen' in grp:
                        data['npart'] = grp['GroupLen'][:]
                    elif 'NumPart' in grp:
                        data['npart'] = grp['NumPart'][:]
                    
                    break
            
            # If no group found, try root level
            if not data:
                print("Checking root level datasets...")
                print(f"Root datasets: {list(f.keys())}")
                
                # Try root level datasets
                if 'GroupMass' in f:
                    data['mass'] = f['GroupMass'][:]
                if 'GroupPos' in f:
                    data['pos'] = f['GroupPos'][:]
                if 'GroupLen' in f:
                    data['npart'] = f['GroupLen'][:]
            
            return data if data else None
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def get_redshift_from_filename(filename):
    """Extract redshift from snapshot number"""
    # This is approximate - you'd need to check your actual output times
    snap_num = int(filename.split('_')[-1].split('.')[0])
    
    # Rough mapping based on typical cosmological runs
    # You may need to adjust this based on your actual TimeMax and outputs
    redshift_map = {
        33: 10, 34: 8, 35: 6, 36: 5, 37: 4, 38: 3.5, 39: 3, 
        40: 2.5, 41: 2, 42: 1.5, 43: 1.2, 44: 1.0, 45: 0.8, 
        46: 0.6, 47: 0.4, 48: 0.2
    }
    
    return redshift_map.get(snap_num, 0)

def analyze_halos_hdf5(output_dir):
    """Find suitable zoom targets in HDF5 halo catalogs"""
    
    # Find all FOF catalog files
    fof_files = glob.glob(f"{output_dir}/fof_subhalo_tab_*.hdf5")
    
    print(f"Found {len(fof_files)} FOF catalog files")
    
    best_halos = []
    
    for fof_file in sorted(fof_files):
        print(f"\n=== Analyzing {fof_file} ===")
        
        redshift = get_redshift_from_filename(fof_file)
        print(f"Estimated redshift: z ≈ {redshift}")
        
        data = read_fof_hdf5(fof_file)
        if data is None or 'mass' not in data:
            print("Could not read mass data")
            continue
            
        masses = data['mass']
        positions = data.get('pos', None)
        npart = data.get('npart', None)
        
        print(f"Found {len(masses)} halos")
        print(f"Mass range: {masses.min():.2e} - {masses.max():.2e} (code units)")
        
        # Convert mass to physical units
        # Assuming UnitMass = 1e10 Msun/h (adjust if different)
        UnitMass = 1e10  # Msun/h
        h = 0.6774
        mass_msun = masses * UnitMass / h  # Convert to Msun
        
        print(f"Mass range: {mass_msun.min():.2e} - {mass_msun.max():.2e} Msun")
        
        # Filter for galaxy-mass halos (10^11 to 3×10^12 Msun)
        mw_mask = (mass_msun >= 1e11) & (mass_msun <= 3e12)
        mw_halos = np.where(mw_mask)[0]
        
        print(f"Galaxy-mass halos (10^11 - 3×10^12 Msun): {len(mw_halos)}")
        
        if len(mw_halos) > 0 and positions is not None:
            print(f"Top 5 most massive galaxy-scale halos:")
            
            # Sort by mass (descending)
            sorted_indices = mw_halos[np.argsort(mass_msun[mw_halos])[::-1]]
            
            box_size = 50.0  # Mpc

            for i, idx in enumerate(sorted_indices[:5]):
                halo_mass = mass_msun[idx]
                pos = positions[idx] if positions is not None else [0, 0, 0]
                n_particles = npart[idx] if npart is not None else 0
                
                # Distance from edges (in Mpc)
                edge_dist = min(pos.min(), (box_size - pos).min()) if len(pos) == 3 else 0
                
                # Distance from box center (Mpc)
                center_dist = np.linalg.norm(np.array(pos) - box_size/2) if len(pos) == 3 else 0
                
                is_well_resolved = n_particles > 100
                is_away_from_edges = edge_dist > 5.0  # 5 Mpc
                
                print(f"  {i+1}. Mass: {halo_mass:.2e} Msun, Particles: {n_particles}")
                print(f"      Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) Mpc")
                print(f"      Edge distance: {edge_dist:.2f} Mpc")
                print(f"      Well resolved: {is_well_resolved}, Away from edges: {is_away_from_edges}")
                
                best_halos.append({
                    'file': fof_file,
                    'redshift': redshift,
                    'index': idx,
                    'mass': halo_mass,
                    'position': pos,
                    'npart': n_particles,
                    'edge_distance': edge_dist,
                    'well_resolved': is_well_resolved,
                    'away_from_edges': is_away_from_edges
                })
    
    # Find best candidates across all snapshots
    print(f"\n=== BEST ZOOM TARGETS (All Times) ===")
    
    # Filter for well-resolved, isolated halos
    good_halos = [h for h in best_halos if h['well_resolved'] and h['away_from_edges']]
    
    if not good_halos:
        print("No ideal candidates found. Relaxing criteria...")
        good_halos = [h for h in best_halos if h['npart'] > 50]
    
    # Sort by mass (descending)
    good_halos = sorted(good_halos, key=lambda x: x['mass'], reverse=True)
    
    for i, halo in enumerate(good_halos[:3]):  # Top 3 candidates
        print(f"\nCANDIDATE {i+1}:")
        print(f"  File: {os.path.basename(halo['file'])}")
        print(f"  Redshift: z ≈ {halo['redshift']}")
        print(f"  Mass: {halo['mass']:.2e} Msun")
        print(f"  Position: ({halo['position'][0]:.3f}, {halo['position'][1]:.3f}, {halo['position'][2]:.3f}) Mpc")
        print(f"  Particles: {halo['npart']}")
        print(f"  Edge distance: {halo['edge_distance']:.0f} Mpc")
        
        if i == 0:
            print(f"*** RECOMMENDED ZOOM TARGET ***")
            print(f"Center your zoom simulation at: ({halo['position'][0]:.3f}, {halo['position'][1]:.3f}, {halo['position'][2]:.3f}) Mpc")
            print(f"Use a zoom region of ~10-15 Mpc around this center")
    
    return good_halos

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_halos_hdf5.py output_parent/")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    best_halos = analyze_halos_hdf5(output_dir)
    
    if best_halos:
        print(f"\nAnalysis complete! Found {len(best_halos)} potential targets.")
    else:
        print("No suitable halos found! You may need to:")
        print("1. Check if FOF catalogs contain the expected data")
        print("2. Lower the mass criteria")
        print("3. Run the parent simulation longer")