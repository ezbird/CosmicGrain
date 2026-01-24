#!/usr/bin/env python3
"""
Debug group catalog to understand its structure and units
"""
import numpy as np
import h5py
import sys

def debug_groupcat(filename):
    """Examine group catalog structure and data"""
    print(f"\n=== Debugging Group Catalog: {filename} ===\n")
    
    with h5py.File(filename, 'r') as f:
        # Print structure
        print("File structure:")
        for key in f.keys():
            print(f"  {key}")
            if isinstance(f[key], h5py.Group):
                for subkey in f[key].keys():
                    shape = f[key][subkey].shape if hasattr(f[key][subkey], 'shape') else 'scalar'
                    print(f"    {subkey}: {shape}")
        
        # Check header
        if 'Header' in f:
            print("\nHeader attributes:")
            for attr in f['Header'].attrs.keys():
                print(f"  {attr} = {f['Header'].attrs[attr]}")
        
        # Check groups
        if 'Group' in f:
            ngroups = len(f['Group']['GroupLen'][:])
            print(f"\nNumber of groups: {ngroups}")
            
            # Get data
            grouplen = f['Group']['GroupLen'][:]
            
            # Check for mass datasets
            mass_keys = [k for k in f['Group'].keys() if 'mass' in k.lower() or 'm_' in k.lower()]
            print(f"\nMass-related datasets: {mass_keys}")
            
            # Check for radius datasets  
            radius_keys = [k for k in f['Group'].keys() if 'radius' in k.lower() or 'r_' in k.lower() or 'r200' in k.lower()]
            print(f"Radius-related datasets: {radius_keys}")
            
            # Print first 10 groups with highest particle count
            sorted_idx = np.argsort(grouplen)[::-1][:10]
            
            print(f"\nTop 10 groups by particle count:")
            print(f"{'ID':>6} {'N_part':>10} {'FirstPart':>12}")
            print("-"*30)
            
            firstpart = f['Group']['GroupFirstSub'][:] if 'GroupFirstSub' in f['Group'] else np.zeros(ngroups)
            for i in sorted_idx:
                print(f"{i:>6} {grouplen[i]:>10} {firstpart[i]:>12.0f}")
            
            # Try to find and print mass/radius for these halos
            if 'Group_M_Crit200' in f['Group']:
                m200 = f['Group']['Group_M_Crit200'][:]
                print(f"\nMasses (Group_M_Crit200) for top halos:")
                print(f"  Min: {np.min(m200[m200>0]):.3e}, Max: {np.max(m200):.3e}")
                for i in sorted_idx[:5]:
                    print(f"  Halo {i}: {m200[i]:.3e} [units?]")
            
            if 'Group_R_Crit200' in f['Group']:
                r200 = f['Group']['Group_R_Crit200'][:]
                print(f"\nRadii (Group_R_Crit200) for top halos:")
                print(f"  Min: {np.min(r200[r200>0]):.6f}, Max: {np.max(r200):.6f}")
                for i in sorted_idx[:5]:
                    print(f"  Halo {i}: {r200[i]:.6f} [units?]")
            
            # Check positions
            if 'GroupPos' in f['Group']:
                pos = f['Group']['GroupPos'][:]
                print(f"\nPosition ranges:")
                for axis, name in enumerate(['X', 'Y', 'Z']):
                    print(f"  {name}: [{np.min(pos[:,axis]):.1f}, {np.max(pos[:,axis]):.1f}]")
            
            # Look for other mass definitions
            print(f"\nAll Group datasets:")
            for key in sorted(f['Group'].keys()):
                shape = f['Group'][key].shape
                dtype = f['Group'][key].dtype
                print(f"  {key}: shape={shape}, dtype={dtype}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <groupcat_file>")
        sys.exit(1)
    
    debug_groupcat(sys.argv[1])
