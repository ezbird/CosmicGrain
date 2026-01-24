#!/usr/bin/env python3
"""
Trace particle IDs from an evolved snapshot back to their Lagrangian (initial) 
positions in the parent IC file. 

FIXED VERSION: Handles MUSIC ICs with kpc/h units properly.
"""
import numpy as np
import h5py
import argparse
from pathlib import Path

def read_particle_ids(filename):
    """Read particle IDs from text file (one per line)"""
    ids = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))
    return np.array(ids, dtype=np.int64)

def read_snapshot(filename, ptype=1):
    """Read positions and IDs from a Gadget/MUSIC snapshot"""
    print(f"  Reading {filename}")
    with h5py.File(filename, 'r') as f:
        group = f[f'PartType{ptype}']
        pos = group['Coordinates'][:]
        ids = group['ParticleIDs'][:]
        
        # Get BoxSize from header
        boxsize = f['Header'].attrs['BoxSize']
        if hasattr(boxsize, '__len__'):
            boxsize = boxsize[0]
        
        # Check if this is likely in kpc/h (MUSIC IC files)
        if boxsize > 1000:
            print(f"  Detected kpc/h units (BoxSize={boxsize:.1f})")
            unit_conversion = 1000.0  # kpc/h to Mpc/h
        else:
            print(f"  Detected Mpc/h units (BoxSize={boxsize:.1f})")
            unit_conversion = 1.0
            
        return pos, ids, boxsize, unit_conversion

def trace_to_lagrangian(target_ids, ic_pos, ic_ids):
    """Map evolved IDs to their Lagrangian positions"""
    # Create ID->position mapping
    id_to_pos = {id: pos for id, pos in zip(ic_ids, ic_pos)}
    
    # Find Lagrangian positions
    lagrangian_pos = []
    found_ids = []
    missing_ids = []
    
    for tid in target_ids:
        if tid in id_to_pos:
            lagrangian_pos.append(id_to_pos[tid])
            found_ids.append(tid)
        else:
            missing_ids.append(tid)
    
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} IDs not found in IC")
        
    return np.array(lagrangian_pos), np.array(found_ids)

def analyze_lagrangian_region(positions, boxsize, unit_conversion=1.0):
    """Analyze the Lagrangian region extent"""
    # Convert to Mpc/h for analysis
    pos_mpc = positions / unit_conversion
    boxsize_mpc = boxsize / unit_conversion
    
    # Compute center (accounting for periodic boundaries)
    center = np.zeros(3)
    for axis in range(3):
        # Use trigonometric mean for periodic boundaries
        theta = 2 * np.pi * pos_mpc[:, axis] / boxsize_mpc
        mean_cos = np.mean(np.cos(theta))
        mean_sin = np.mean(np.sin(theta))
        mean_angle = np.arctan2(mean_sin, mean_cos)
        center[axis] = (mean_angle * boxsize_mpc / (2 * np.pi)) % boxsize_mpc
    
    # Compute distances accounting for periodicity
    distances = np.zeros((len(pos_mpc), 3))
    for axis in range(3):
        dx = pos_mpc[:, axis] - center[axis]
        # Wrap distances for periodic boundaries
        dx = np.where(dx > boxsize_mpc/2, dx - boxsize_mpc, dx)
        dx = np.where(dx < -boxsize_mpc/2, dx + boxsize_mpc, dx)
        distances[:, axis] = dx
    
    # Compute extent and radius
    extent = np.max(np.abs(distances), axis=0) * 2
    radius = np.max(np.sqrt(np.sum(distances**2, axis=1)))
    
    return center, extent, radius

def main():
    parser = argparse.ArgumentParser(description='Trace particle IDs to Lagrangian positions')
    parser.add_argument('--evolved-ids', required=True, help='Text file with particle IDs from evolved snapshot')
    parser.add_argument('--parent-ic', required=True, help='Parent IC file (HDF5)')
    parser.add_argument('--evolved-snapshot', required=True, help='Evolved snapshot for verification')
    parser.add_argument('--output', required=True, help='Output file for Lagrangian IDs')
    parser.add_argument('--ptype', type=int, default=1, help='Particle type (default: 1 for DM)')
    
    args = parser.parse_args()
    
    # Step 1: Read target IDs
    print(f"Reading target IDs from {args.evolved_ids}...")
    target_ids = read_particle_ids(args.evolved_ids)
    print(f"  Loaded {len(target_ids)} particle IDs")
    
    # Step 2: Verify in evolved snapshot (optional but good check)
    print(f"Verifying IDs in evolved snapshot {args.evolved_snapshot}...")
    evolved_pos, evolved_ids, _, _ = read_snapshot(args.evolved_snapshot, args.ptype)
    evolved_id_set = set(evolved_ids)
    found_in_evolved = sum(1 for tid in target_ids if tid in evolved_id_set)
    print(f"  Found {found_in_evolved}/{len(target_ids)} IDs in evolved snapshot")
    
    # Step 3: Trace in parent IC
    print(f"Tracing IDs in parent IC file {args.parent_ic}...")
    ic_pos, ic_ids, boxsize, unit_conversion = read_snapshot(args.parent_ic, args.ptype)
    lagrangian_pos, found_ids = trace_to_lagrangian(target_ids, ic_pos, ic_ids)
    
    # Step 4: Analyze Lagrangian region
    center, extent, radius = analyze_lagrangian_region(lagrangian_pos, boxsize, unit_conversion)
    
    print("=" * 70)
    print("LAGRANGIAN REGION ANALYSIS")
    print("=" * 70)
    print(f"Particles found in IC:     {len(found_ids)}/{len(target_ids)}")
    print(f"Lagrangian center [Mpc/h]: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"Region extent [Mpc/h]:     ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})")
    print(f"Max radius [Mpc/h]:        {radius:.4f}")
    print("=" * 70)
    
    # Convert to fractional coordinates for MUSIC
    boxsize_mpc = boxsize / unit_conversion
    center_frac = center / boxsize_mpc
    extent_frac = extent / boxsize_mpc
    
    print(f"\nFor MUSIC configuration file:")
    print(f"  ref_center = {center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f}")
    print(f"  ref_extent = {np.max(extent_frac)*1.2:.6f}, {np.max(extent_frac)*1.2:.6f}, {np.max(extent_frac)*1.2:.6f}")
    print(f"  (extent padded by 20% for safety)")
    
    # Warnings
    if radius > 5.0:
        print(f"\n⚠️  WARNING: Lagrangian region is large (radius={radius:.2f} Mpc/h)!")
        print(f"   This might cause issues with zoom simulations.")
        print(f"   Consider:")
        print(f"   1. Selecting a different halo")
        print(f"   2. Using a smaller kR multiplier")
        print(f"   3. Checking if this halo had a recent merger")
    
    # Step 5: Write output
    with open(args.output, 'w') as f:
        for id in found_ids:
            f.write(f"{id}\n")
    print(f"✓ Wrote {len(found_ids)} IDs to {args.output}")
    
    # Also save analysis results
    analysis_file = args.output.replace('.txt', '_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"# Lagrangian Region Analysis\n")
        f.write(f"# Original IDs: {len(target_ids)}\n")
        f.write(f"# Found IDs: {len(found_ids)}\n")
        f.write(f"# Center [Mpc/h]: {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")
        f.write(f"# Extent [Mpc/h]: {extent[0]:.6f} {extent[1]:.6f} {extent[2]:.6f}\n")
        f.write(f"# Radius [Mpc/h]: {radius:.6f}\n")
        f.write(f"# BoxSize [Mpc/h]: {boxsize_mpc:.1f}\n")
        f.write(f"# MUSIC ref_center: {center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f}\n")
        f.write(f"# MUSIC ref_extent: {np.max(extent_frac)*1.2:.6f}, {np.max(extent_frac)*1.2:.6f}, {np.max(extent_frac)*1.2:.6f}\n")
    print(f"✓ Wrote analysis to {analysis_file}")

if __name__ == "__main__":
    main()
