#!/usr/bin/env python3
"""
Debug script to find the exact halo position in both coordinate systems
"""

import h5py
import numpy as np
import argparse

def check_halo_position(fof_file, snapshot_file):
    """Check halo position in both FoF catalog and snapshot"""
    
    print("=== CHECKING FoF CATALOG ===")
    with h5py.File(fof_file, 'r') as f:
        masses = f['Group/GroupMass'][:] * 1e10  # Convert to Msol/h
        positions = f['Group/GroupPos'][:]  # kpc/h
        boxsize = f['Header'].attrs['BoxSize']
        
        # Find most massive halo
        max_idx = np.argmax(masses)
        halo_pos_orig = positions[max_idx]
        halo_mass = masses[max_idx]
        
        print(f"Box size: {boxsize:.0f} kpc/h")
        print(f"Most massive halo:")
        print(f"  Mass: {halo_mass:.2e} Msol/h")
        print(f"  Position (original): [{halo_pos_orig[0]:.0f}, {halo_pos_orig[1]:.0f}, {halo_pos_orig[2]:.0f}] kpc/h")
        
        # Convert to centered coordinates
        halo_pos_centered = halo_pos_orig - boxsize/2
        print(f"  Position (centered): [{halo_pos_centered[0]:.0f}, {halo_pos_centered[1]:.0f}, {halo_pos_centered[2]:.0f}] kpc/h")
    
    print("\n=== CHECKING SNAPSHOT ===")
    with h5py.File(snapshot_file, 'r') as f:
        pos = f['PartType1/Coordinates'][:]
        boxsize_snap = f['Header'].attrs['BoxSize']
        
        print(f"Snapshot box size: {boxsize_snap:.0f} kpc/h")
        print(f"Particle position range:")
        print(f"  X: {pos[:, 0].min():.0f} to {pos[:, 0].max():.0f}")
        print(f"  Y: {pos[:, 1].min():.0f} to {pos[:, 1].max():.0f}")
        print(f"  Z: {pos[:, 2].min():.0f} to {pos[:, 2].max():.0f}")
        
        # Find particles near the halo
        search_radius = 1000  # kpc/h
        distances = np.linalg.norm(pos - halo_pos_orig, axis=1)
        near_halo = np.sum(distances < search_radius)
        
        print(f"\nParticles within {search_radius} kpc/h of halo center: {near_halo}")
        
        # Check density around halo vs center
        center_pos = np.array([boxsize_snap/2, boxsize_snap/2, boxsize_snap/2])
        distances_center = np.linalg.norm(pos - center_pos, axis=1)
        near_center = np.sum(distances_center < search_radius)
        
        print(f"Particles within {search_radius} kpc/h of box center: {near_center}")
        
        return halo_pos_orig, halo_pos_centered, boxsize_snap

def main():
    parser = argparse.ArgumentParser(description='Debug halo position')
    parser.add_argument('fof_file', help='FoF catalog file')
    parser.add_argument('snapshot_file', help='Snapshot file')
    
    args = parser.parse_args()
    
    halo_orig, halo_centered, boxsize = check_halo_position(args.fof_file, args.snapshot_file)
    
    print(f"\n=== SLICE COMMANDS ===")
    print(f"To center slice on halo using original coordinates:")
    print(f"  # Use position {halo_orig[2]:.0f} for z-slice")
    
    print(f"\nTo center slice on halo using centered coordinates:")
    print(f"  python plot_darkmatter_slice.py {args.snapshot_file} --position {halo_centered[2]:.0f}")
    
    print(f"\nTo see the halo region:")
    print(f"  python plot_darkmatter_slice.py {args.snapshot_file} --position {halo_centered[2]:.0f} --thickness 2000")

if __name__ == "__main__":
    main()