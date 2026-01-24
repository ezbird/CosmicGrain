#!/usr/bin/env python3
"""
Debug script to check particle positions in IC file
"""
import h5py
import numpy as np

def check_ic_file(filename):
    """Check positions and IDs in IC file"""
    print(f"\n=== Checking IC file: {filename} ===\n")
    
    with h5py.File(filename, 'r') as f:
        # Check structure
        print("File structure:")
        for key in f.keys():
            print(f"  {key}")
            if isinstance(f[key], h5py.Group):
                for subkey in f[key].keys():
                    print(f"    {subkey}")
        
        # Check header
        if 'Header' in f:
            header = f['Header']
            print("\nHeader attributes:")
            for attr in header.attrs.keys():
                val = header.attrs[attr]
                if attr == 'BoxSize':
                    print(f"  BoxSize = {val}")
                elif attr == 'NumPart_Total' or attr == 'NumPart_ThisFile':
                    print(f"  {attr} = {val}")
        
        # Check particle data for each type
        for ptype in range(6):
            group_name = f'PartType{ptype}'
            if group_name in f:
                print(f"\n{group_name}:")
                
                # Check coordinates
                if 'Coordinates' in f[group_name]:
                    pos = f[group_name]['Coordinates'][:]
                    print(f"  Positions shape: {pos.shape}")
                    print(f"  Position range: [{np.min(pos):.3f}, {np.max(pos):.3f}]")
                    print(f"  Position per axis:")
                    for i, axis in enumerate(['X', 'Y', 'Z']):
                        print(f"    {axis}: [{np.min(pos[:, i]):.3f}, {np.max(pos[:, i]):.3f}]")
                    
                    # Sample some positions
                    print(f"  First 5 positions:")
                    for i in range(min(5, len(pos))):
                        print(f"    {i}: ({pos[i,0]:.3f}, {pos[i,1]:.3f}, {pos[i,2]:.3f})")
                
                # Check IDs
                if 'ParticleIDs' in f[group_name]:
                    ids = f[group_name]['ParticleIDs'][:]
                    print(f"  IDs shape: {ids.shape}")
                    print(f"  ID range: [{np.min(ids)}, {np.max(ids)}]")
                    print(f"  First 5 IDs: {ids[:5]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ic_file.hdf5>")
        sys.exit(1)
    
    check_ic_file(sys.argv[1])
