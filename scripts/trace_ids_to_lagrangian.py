#!/usr/bin/env python3
"""
Trace particle IDs from an evolved snapshot back to their Lagrangian (initial) 
positions in the parent IC file. This is crucial for zoom simulations.

Usage:
------
python trace_ids_to_lagrangian.py \
  --evolved-ids ids_2r200_group148.txt \
  --parent-ic ICs/uniform_IC_genetIC_50Mpc_128.hdf5 \
  --evolved-snapshot output/snapshot_061.hdf5 \
  --output ids_lagrangian.txt \
  --ptype 1

This will:
1. Read the IDs you selected from the evolved snapshot
2. Find those IDs in the evolved snapshot to verify selection
3. Trace them back to the parent IC file
4. Output the same IDs (which now reference Lagrangian positions)
"""
import argparse
import glob
import h5py
import numpy as np
import sys

def open_many(pattern_or_file):
    """Return a sorted list of files from a literal path or glob pattern."""
    import os
    if os.path.isfile(pattern_or_file):
        return [pattern_or_file]
    files = sorted(glob.glob(pattern_or_file))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern_or_file}")
    return files

def read_ids_from_snapshot(snap_files, ptype=1, chunk=2_000_000):
    """
    Read all particle IDs from snapshot file(s) for given particle type.
    Returns numpy array of IDs.
    """
    all_ids = []
    for fn in snap_files:
        with h5py.File(fn, 'r') as f:
            key = f'PartType{ptype}'
            if key not in f:
                continue
            pid = f[key]['ParticleIDs']
            N = pid.shape[0]
            
            # Read in chunks to handle large files
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                all_ids.append(pid[i0:i1])
    
    if all_ids:
        return np.concatenate(all_ids)
    return np.array([], dtype=np.int64)

def read_positions_for_ids(snap_files, target_ids, ptype=1, chunk=2_000_000):
    """
    Find positions of specific particle IDs in snapshot.
    Returns dict: {id: position} for found particles.
    """
    target_set = set(target_ids)
    id_to_pos = {}
    
    for fn in snap_files:
        with h5py.File(fn, 'r') as f:
            key = f'PartType{ptype}'
            if key not in f:
                continue
            
            coords = f[key]['Coordinates']
            pid = f[key]['ParticleIDs']
            N = pid.shape[0]
            
            for i0 in range(0, N, chunk):
                i1 = min(i0 + chunk, N)
                chunk_ids = pid[i0:i1]
                chunk_pos = coords[i0:i1]
                
                # Find which IDs in this chunk are in our target set
                for i, pid_val in enumerate(chunk_ids):
                    if pid_val in target_set:
                        id_to_pos[int(pid_val)] = chunk_pos[i]
    
    return id_to_pos

def compute_lagrangian_region_stats(ic_files, target_ids, ptype=1):
    """
    Find target IDs in IC file and compute their spatial extent.
    Returns dict with center, extent, and number of particles found.
    """
    id_to_pos = read_positions_for_ids(ic_files, target_ids, ptype)
    
    if not id_to_pos:
        return None
    
    positions = np.array(list(id_to_pos.values()))
    found_ids = np.array(list(id_to_pos.keys()))
    
    center = np.mean(positions, axis=0)
    extent = np.max(positions, axis=0) - np.min(positions, axis=0)
    max_radius = np.max(np.sqrt(np.sum((positions - center[None,:])**2, axis=1)))
    
    return {
        'found_ids': found_ids,
        'n_found': len(found_ids),
        'n_requested': len(target_ids),
        'center': center,
        'extent': extent,
        'max_radius': max_radius,
        'positions': positions
    }

def main():
    ap = argparse.ArgumentParser(
        description="Trace particle IDs to their Lagrangian positions in parent ICs"
    )
    ap.add_argument("--evolved-ids", required=True,
                    help="Text file with particle IDs from evolved snapshot (one per line)")
    ap.add_argument("--parent-ic", required=True,
                    help="Parent IC file (HDF5) or glob pattern")
    ap.add_argument("--evolved-snapshot", required=False,
                    help="Optional: evolved snapshot to verify IDs exist there")
    ap.add_argument("--output", required=True,
                    help="Output text file with traced IDs")
    ap.add_argument("--ptype", type=int, default=1,
                    help="Particle type (default: 1 = DM)")
    args = ap.parse_args()
    
    # Read the target IDs
    print(f"Reading target IDs from {args.evolved_ids}...")
    target_ids = np.loadtxt(args.evolved_ids, dtype=np.int64)
    print(f"  Loaded {len(target_ids)} particle IDs")
    
    # Optionally verify IDs exist in evolved snapshot
    if args.evolved_snapshot:
        print(f"\nVerifying IDs in evolved snapshot {args.evolved_snapshot}...")
        evolved_files = open_many(args.evolved_snapshot)
        evolved_positions = read_positions_for_ids(evolved_files, target_ids, args.ptype)
        print(f"  Found {len(evolved_positions)}/{len(target_ids)} IDs in evolved snapshot")
        
        if len(evolved_positions) < len(target_ids):
            missing = len(target_ids) - len(evolved_positions)
            print(f"  WARNING: {missing} IDs not found in evolved snapshot!")
    
    # Trace IDs back to parent ICs
    print(f"\nTracing IDs in parent IC file {args.parent_ic}...")
    ic_files = open_many(args.parent_ic)
    
    stats = compute_lagrangian_region_stats(ic_files, target_ids, args.ptype)
    
    if stats is None or stats['n_found'] == 0:
        sys.exit("ERROR: No target IDs found in parent IC file!")
    
    print(f"\n{'='*70}")
    print(f"LAGRANGIAN REGION ANALYSIS")
    print(f"{'='*70}")
    print(f"Particles found in IC:     {stats['n_found']}/{stats['n_requested']}")
    print(f"Lagrangian center [Mpc/h]: ({stats['center'][0]:.4f}, "
          f"{stats['center'][1]:.4f}, {stats['center'][2]:.4f})")
    print(f"Region extent [Mpc/h]:     ({stats['extent'][0]:.4f}, "
          f"{stats['extent'][1]:.4f}, {stats['extent'][2]:.4f})")
    print(f"Max radius [Mpc/h]:        {stats['max_radius']:.4f}")
    print(f"{'='*70}")
    
    # Check if region is reasonable
    if stats['max_radius'] > 10.0:
        print(f"\n⚠️  WARNING: Lagrangian region is very large (radius={stats['max_radius']:.2f} Mpc/h)!")
        print("   This might cause issues with zoom simulations.")
        print("   Consider:")
        print("   1. Selecting a different halo")
        print("   2. Using a smaller kR multiplier")
        print("   3. Checking if the parent simulation has the correct particle IDs")
    
    # Write output IDs
    found_ids = stats['found_ids']
    with open(args.output, 'w') as f:
        for pid in found_ids:
            f.write(f"{int(pid)}\n")
    
    print(f"\n✓ Wrote {len(found_ids)} IDs to {args.output}")
    print(f"\nUse this file with genetIC:")
    print(f"  id_file       {args.output}")
    print(f"  centre        {stats['center'][0]:.8f} {stats['center'][1]:.8f} {stats['center'][2]:.8f}")

if __name__ == "__main__":
    main()