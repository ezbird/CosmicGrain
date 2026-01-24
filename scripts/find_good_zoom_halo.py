#!/usr/bin/env python3
"""
Test multiple halos to find ones suitable for zoom simulations
(i.e., with compact Lagrangian regions)
"""
import numpy as np
import h5py
import argparse
import subprocess
import os
from pathlib import Path

def get_halo_properties(groupcat_file):
    """Get basic properties of all halos"""
    with h5py.File(groupcat_file, 'r') as f:
        if 'Group' in f:
            # FOF groups
            m200 = f['Group/Group_M_Crit200'][:]  # M200 in 10^10 M_sun/h
            r200 = f['Group/Group_R_Crit200'][:]  # R200 in kpc/h (physical)
            pos = f['Group/GroupPos'][:]  # Position in kpc/h
            npart = f['Group/GroupLen'][:]  # Number of particles
            
            # Convert units
            r200 = r200 / 1000.0  # kpc/h to Mpc/h
            pos = pos / 1000.0    # kpc/h to Mpc/h
            
            return m200, r200, pos, npart
        else:
            print("Error: Group catalog not found in file")
            return None, None, None, None

def test_halo(halo_id, snapshot, groupcat, parent_ic, kR=1.0, ptype=1):
    """Test a single halo for Lagrangian compactness"""
    
    # Create temporary file names
    ids_file = f"temp_ids_halo{halo_id}.txt"
    lag_file = f"temp_lag_halo{halo_id}.txt"
    
    try:
        # Step 1: Extract IDs
        cmd1 = [
            "python", "extract_halo_ids_for_zoom.py",
            "--snapshot", snapshot,
            "--groupcat", groupcat,
            "--group-id", str(halo_id),
            "--ptype", str(ptype),
            "--kR", str(kR),
            "--output", ids_file
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            return None, None, None
        
        # Parse number of particles selected
        npart = None
        for line in result1.stdout.split('\n'):
            if 'Selected' in line and 'DM IDs' in line:
                npart = int(line.split()[1])
        
        # Step 2: Trace to Lagrangian
        cmd2 = [
            "python", "trace_ids_to_lagrangian_fixed.py",
            "--evolved-ids", ids_file,
            "--parent-ic", parent_ic,
            "--evolved-snapshot", snapshot,
            "--output", lag_file,
            "--ptype", str(ptype)
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            return npart, None, None
        
        # Parse Lagrangian radius
        lag_radius = None
        lag_center = None
        for line in result2.stdout.split('\n'):
            if 'Max radius [Mpc/h]:' in line:
                lag_radius = float(line.split()[-1])
            if 'Lagrangian center [Mpc/h]:' in line:
                # Parse center from format: (x, y, z)
                center_str = line.split(':', 1)[1].strip()
                center_str = center_str.strip('()')
                lag_center = [float(x.strip()) for x in center_str.split(',')]
        
        return npart, lag_radius, lag_center
        
    finally:
        # Clean up temporary files
        for f in [ids_file, lag_file, lag_file.replace('.txt', '_analysis.txt')]:
            if os.path.exists(f):
                os.remove(f)

def main():
    parser = argparse.ArgumentParser(description='Find halos suitable for zoom simulations')
    parser.add_argument('--snapshot', required=True, help='Evolved snapshot')
    parser.add_argument('--groupcat', required=True, help='Group catalog')
    parser.add_argument('--parent-ic', required=True, help='Parent IC file')
    parser.add_argument('--min-mass', type=float, default=0.1, help='Minimum M200 in 10^10 M_sun/h')
    parser.add_argument('--max-mass', type=float, default=100.0, help='Maximum M200 in 10^10 M_sun/h')
    parser.add_argument('--max-lag-radius', type=float, default=3.0, help='Maximum acceptable Lagrangian radius (Mpc/h)')
    parser.add_argument('--kR', type=float, default=1.0, help='Selection radius multiplier')
    parser.add_argument('--test-n', type=int, default=50, help='Number of halos to test')
    parser.add_argument('--start-id', type=int, default=0, help='Starting halo ID')
    
    args = parser.parse_args()
    
    print(f"Finding halos suitable for zoom simulations...")
    print(f"Criteria: Lagrangian radius < {args.max_lag_radius} Mpc/h")
    print(f"Testing halos with M200 between {args.min_mass:.1f} and {args.max_mass:.1f} x 10^10 M_sun/h")
    print("="*80)
    
    # Get halo properties
    m200, r200, pos, npart = get_halo_properties(args.groupcat)
    if m200 is None:
        return
    
    # Filter by mass
    mass_mask = (m200 >= args.min_mass) & (m200 <= args.max_mass)
    valid_ids = np.where(mass_mask)[0]
    
    if len(valid_ids) == 0:
        print("No halos found in specified mass range!")
        return
    
    # Sort by mass (descending)
    valid_ids = valid_ids[np.argsort(m200[valid_ids])[::-1]]
    
    # Test halos
    good_halos = []
    tested = 0
    
    print(f"{'ID':>6} {'M200 [1e10 M☉/h]':>16} {'R200 [kpc/h]':>12} {'N_part':>8} {'Lag R [Mpc/h]':>13} {'Status':>10}")
    print("-"*80)
    
    for i, halo_id in enumerate(valid_ids[args.start_id:]):
        if tested >= args.test_n:
            break
            
        tested += 1
        
        # Test this halo
        npart_selected, lag_radius, lag_center = test_halo(
            halo_id, args.snapshot, args.groupcat, 
            args.parent_ic, args.kR
        )
        
        if lag_radius is not None:
            status = "✅ GOOD" if lag_radius < args.max_lag_radius else "❌ BAD"
            print(f"{halo_id:>6} {m200[halo_id]:>16.3f} {r200[halo_id]*1000:>12.1f} {npart_selected:>8} {lag_radius:>13.2f} {status:>10}")
            
            if lag_radius < args.max_lag_radius:
                good_halos.append({
                    'id': halo_id,
                    'mass': m200[halo_id],
                    'r200': r200[halo_id],
                    'pos': pos[halo_id],
                    'lag_radius': lag_radius,
                    'lag_center': lag_center,
                    'npart': npart_selected
                })
        else:
            print(f"{halo_id:>6} {m200[halo_id]:>16.3f} {r200[halo_id]*1000:>12.1f} {'--':>8} {'ERROR':>13} {'⚠️':>10}")
    
    print("="*80)
    
    if good_halos:
        print(f"\nFound {len(good_halos)} suitable halos for zoom simulations!\n")
        print("Top 5 candidates (sorted by mass):")
        print("-"*60)
        
        for i, halo in enumerate(good_halos[:5]):
            print(f"\nHalo {halo['id']}:")
            print(f"  M200: {halo['mass']:.3f} × 10^10 M☉/h")
            print(f"  R200: {halo['r200']:.3f} Mpc/h")
            print(f"  Position: ({halo['pos'][0]:.2f}, {halo['pos'][1]:.2f}, {halo['pos'][2]:.2f}) Mpc/h")
            print(f"  Lagrangian radius: {halo['lag_radius']:.2f} Mpc/h")
            print(f"  Selected particles: {halo['npart']}")
            
            if i == 0:
                print(f"\nRecommended zoom parameters for Halo {halo['id']}:")
                boxsize = 50.0  # Mpc/h
                center_frac = [c/boxsize for c in halo['lag_center']]
                extent_frac = (halo['lag_radius'] * 2.5) / boxsize  # 2.5x radius for safety
                print(f"  ref_center = {center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f}")
                print(f"  ref_extent = {extent_frac:.6f}, {extent_frac:.6f}, {extent_frac:.6f}")
    else:
        print(f"\n⚠️  No suitable halos found!")
        print(f"Consider:")
        print(f"  1. Increasing max_lag_radius (currently {args.max_lag_radius} Mpc/h)")
        print(f"  2. Testing different mass ranges")
        print(f"  3. Using a smaller kR value (currently {args.kR})")
        print(f"  4. Running a larger parent box")

if __name__ == "__main__":
    main()
