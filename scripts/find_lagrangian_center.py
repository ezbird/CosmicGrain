#!/usr/bin/env python3
"""
Find the Lagrangian center of a halo for MUSIC zoom ICs (Gadget-4 compatible)
Usage: ./find_lagrangian_center.py <snapshot_dir> <ic_file> <halo_id> [snapshot_num]
"""

import numpy as np
import h5py
import sys
import glob
import os

def find_lagrangian_center(snapshot_dir, ic_file, halo_id, snapshot_num=None):
    """
    Trace halo particles back to their positions in the ICs
    """
    
    # Find catalog files
    catalog_files = sorted(glob.glob(f"{snapshot_dir}/fof_subhalo_tab_*.hdf5"))
    
    if not catalog_files:
        print(f"Error: No fof_subhalo_tab files found in {snapshot_dir}")
        sys.exit(1)
    
    # Use specified snapshot or the last one
    if snapshot_num is not None:
        catalog_pattern = f"{snapshot_dir}/fof_subhalo_tab_{snapshot_num:03d}.hdf5"
        snapshot_pattern = f"{snapshot_dir}/snapshot_{snapshot_num:03d}.hdf5"
        
        if not os.path.exists(catalog_pattern):
            print(f"Error: {catalog_pattern} not found!")
            sys.exit(1)
        if not os.path.exists(snapshot_pattern):
            print(f"Error: {snapshot_pattern} not found!")
            sys.exit(1)
            
        catalog_file = catalog_pattern
        snapshot_file = snapshot_pattern
    else:
        # Use the last snapshot (highest number)
        catalog_file = catalog_files[-1]
        snapshot_num = int(catalog_file.split('_')[-1].replace('.hdf5', ''))
        snapshot_file = f"{snapshot_dir}/snapshot_{snapshot_num:03d}.hdf5"
    
    print(f"Using catalog: {catalog_file}")
    print(f"Using snapshot: {snapshot_file}")
    
    # Read redshift and halo mass
    with h5py.File(snapshot_file, 'r') as f:
        redshift = f['Header'].attrs['Redshift']
    
    with h5py.File(catalog_file, 'r') as f:
        if 'Subhalo' in f:
            mass_field_names = ['SubhaloMass', 'Mass']
            for field in mass_field_names:
                if field in f['Subhalo']:
                    halo_mass = f['Subhalo'][field][halo_id]
                    print(f"Halo mass: {halo_mass:.2e} (code units)")
                    break
    
    print(f"Snapshot redshift: z = {redshift:.3f}")
    
    # Read catalog structure
    print(f"\nReading halo catalog...")
    with h5py.File(catalog_file, 'r') as f:
        # Check for different possible group names
        if 'Subhalo' in f:
            sub_group = f['Subhalo']
        elif 'Group' in f:
            sub_group = f['Group']
        else:
            print("Error: Cannot find Subhalo or Group data!")
            sys.exit(1)
        
        # Try different field names
        len_fields = ['SubhaloLenType', 'LenType', 'GroupLenType']
        offset_fields = ['SubhaloOffsetType', 'OffsetType', 'GroupOffsetType']
        
        sub_len = None
        sub_offset = None
        
        for field in len_fields:
            if field in sub_group:
                sub_len = sub_group[field][:]
                break
        
        for field in offset_fields:
            if field in sub_group:
                sub_offset = sub_group[field][:]
                break
        
        if sub_len is None or sub_offset is None:
            print("Error: Could not find length/offset fields")
            sys.exit(1)
        
        print(f"Total number of subhalos: {len(sub_len)}")
        
        if halo_id >= len(sub_len):
            print(f"Error: Halo ID {halo_id} out of range (0 to {len(sub_len)-1})")
            sys.exit(1)
        
        # Get DM particle info (PartType1)
        if len(sub_len.shape) == 1:
            print("Error: Catalog only has total particle counts")
            sys.exit(1)
        
        offset_dm = sub_offset[halo_id, 1]
        length_dm = sub_len[halo_id, 1]
        
        print(f"Halo {halo_id}: {length_dm} DM particles at z={redshift:.2f}")
        
        if length_dm == 0:
            print("Error: This halo has no DM particles!")
            sys.exit(1)
        
        if length_dm < 1000:
            print(f"⚠ WARNING: This is a VERY small halo ({length_dm} particles)")
            print(f"   Small halos have large Lagrangian regions and may not benefit from zoom ICs")
    
    # Read particle IDs from snapshot
    print(f"\nLoading particle IDs from snapshot...")
    with h5py.File(snapshot_file, 'r') as f:
        if 'PartType1' not in f:
            print("Error: No PartType1 (DM) in snapshot!")
            sys.exit(1)
        
        all_ids = f['PartType1/ParticleIDs'][:]
        halo_ids = all_ids[offset_dm:offset_dm+length_dm]
    
    # Load ICs and trace particles
    print(f"Loading IC file...")
    with h5py.File(ic_file, 'r') as f:
        ic_ids = f['PartType1/ParticleIDs'][:]
        ic_pos = f['PartType1/Coordinates'][:]
        boxsize = f['Header'].attrs['BoxSize']
        
        # Detect units
        if boxsize > 1000:
            print(f"Box size: {boxsize:.1f} kpc/h = {boxsize/1000:.1f} Mpc/h")
            boxsize_Mpc = boxsize / 1000.0
            pos_Mpc = ic_pos / 1000.0
        else:
            print(f"Box size: {boxsize:.1f} Mpc/h")
            boxsize_Mpc = boxsize
            pos_Mpc = ic_pos
        
        # Match particle IDs
        print("Matching particle IDs...")
        sort_idx = np.argsort(ic_ids)
        sorted_ids = ic_ids[sort_idx]
        idx_in_sorted = np.searchsorted(sorted_ids, halo_ids)
        
        # Validate matches
        valid_mask = (idx_in_sorted < len(sorted_ids)) & (sorted_ids[idx_in_sorted] == halo_ids)
        n_matched = valid_mask.sum()
        
        print(f"Matched {n_matched}/{len(halo_ids)} particles ({n_matched/len(halo_ids)*100:.1f}%)")
        
        if n_matched == 0:
            print("Error: No particles matched!")
            sys.exit(1)
        
        # Get Lagrangian positions in Mpc/h
        ic_indices = sort_idx[idx_in_sorted[valid_mask]]
        lagrangian_positions = pos_Mpc[ic_indices]
        
        # Calculate center (handle periodic boundaries)
        center = np.median(lagrangian_positions, axis=0)
        shifted_pos = lagrangian_positions - center
        shifted_pos[shifted_pos > boxsize_Mpc/2] -= boxsize_Mpc
        shifted_pos[shifted_pos < -boxsize_Mpc/2] += boxsize_Mpc
        refined_center = (center + np.mean(shifted_pos, axis=0)) % boxsize_Mpc
        
        # Calculate extent in Mpc/h
        distances = np.sqrt(np.sum(shifted_pos**2, axis=1))
        r_50 = np.percentile(distances, 50)
        r_90 = np.percentile(distances, 90)
        r_95 = np.percentile(distances, 95)
        r_99 = np.percentile(distances, 99)
        
        # More reasonable padding: extent should be slightly larger than r99
        # Not 4x r90 which is way too much!
        extent_tight = r_95 * 1.2   # 20% padding beyond 95th percentile
        extent_standard = r_99 * 1.1  # 10% padding beyond 99th percentile  
        extent_generous = r_99 * 1.3  # 30% padding beyond 99th percentile
        
        # Convert to FRACTIONAL coordinates for MUSIC2
        center_frac = refined_center / boxsize_Mpc
        extent_tight_frac = extent_tight / boxsize_Mpc
        extent_standard_frac = extent_standard / boxsize_Mpc
        extent_generous_frac = extent_generous / boxsize_Mpc
        
        # Print results
        print(f"\n{'='*60}")
        print(f"LAGRANGIAN REGION (in initial conditions)")
        print(f"{'='*60}")
        print(f"Center (physical):   ({refined_center[0]:.4f}, {refined_center[1]:.4f}, {refined_center[2]:.4f}) Mpc/h")
        print(f"Center (fractional): ({center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f})")
        print(f"\nParticle distribution (Lagrangian space):")
        print(f"  Median radius (r50):   {r_50:.4f} Mpc/h = {r_50/boxsize_Mpc:.4f} (frac)")
        print(f"  90th percentile (r90): {r_90:.4f} Mpc/h = {r_90/boxsize_Mpc:.4f} (frac)")
        print(f"  95th percentile (r95): {r_95:.4f} Mpc/h = {r_95/boxsize_Mpc:.4f} (frac)")
        print(f"  99th percentile (r99): {r_99:.4f} Mpc/h = {r_99/boxsize_Mpc:.4f} (frac)")
        
        print(f"\nSuggested ref_extent (half-width of refinement region):")
        print(f"  Tight (1.2×r95):    {extent_tight:.4f} Mpc/h = {extent_tight_frac:.6f} (frac)")
        print(f"  Standard (1.1×r99): {extent_standard:.4f} Mpc/h = {extent_standard_frac:.6f} (frac)  ← RECOMMENDED")
        print(f"  Generous (1.3×r99): {extent_generous:.4f} Mpc/h = {extent_generous_frac:.6f} (frac)")
        
        print(f"\n{'='*60}")
        print(f"COPY THIS TO YOUR MUSIC2 CONFIG:")
        print(f"{'='*60}")
        print(f"ref_center = {center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f}")
        print(f"ref_extent = {extent_standard_frac:.6f}, {extent_standard_frac:.6f}, {extent_standard_frac:.6f}")
        print(f"{'='*60}\n")
        
        # Sanity checks
        print("Sanity checks:")
        
        # Check if this is a good zoom candidate
        if extent_standard_frac > 0.4:
            print(f"  ✗ POOR ZOOM CANDIDATE: ref_extent = {extent_standard_frac:.3f} (> 40% of box)")
            print(f"     The Lagrangian region is too large - you'd refine most of the box!")
            print(f"     This halo is too small/dispersed for efficient zoom ICs")
            print(f"     Recommendation: Choose a more massive halo (>1000 particles at z=0)")
        elif extent_standard_frac > 0.25:
            print(f"  ⚠ MARGINAL: ref_extent = {extent_standard_frac:.3f} (25-40% of box)")
            print(f"     Zoom ICs will work but efficiency gains may be modest")
        elif extent_standard_frac > 0.1:
            print(f"  ✓ GOOD: ref_extent = {extent_standard_frac:.3f} (10-25% of box)")
            print(f"     Reasonable zoom candidate")
        else:
            print(f"  ✓ EXCELLENT: ref_extent = {extent_standard_frac:.3f} (< 10% of box)")
            print(f"     Great zoom candidate - tight Lagrangian region")
        
        # Check boundary proximity
        if any(center_frac < 0.15) or any(center_frac > 0.85):
            print(f"  ⚠ Note: Halo is near box edge (center at {center_frac})")
            print(f"     May have periodic boundary issues")
        
        # Particle count warning
        if length_dm < 500:
            print(f"  ⚠ Note: Very few particles ({length_dm}) - consider choosing a more massive halo")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ./find_lagrangian_center.py <snapshot_dir> <ic_file> <halo_id> [snapshot_num]")
        print("\nExample:")
        print("  ./find_lagrangian_center.py \\")
        print("    ../output_archive/output_parent_100Mpc_128_music \\")
        print("    ../ICs/IC_parent_100Mpc_128_music.hdf5 \\")
        print("    3087 13")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    ic_file = sys.argv[2]
    halo_id = int(sys.argv[3])
    snapshot_num = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    find_lagrangian_center(snapshot_dir, ic_file, halo_id, snapshot_num)
