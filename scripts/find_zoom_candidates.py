#!/usr/bin/env python3
"""
Find good zoom halo candidates
Usage: ./find_zoom_candidates.py <snapshot_dir> <ic_file> [min_particles]
"""

import numpy as np
import h5py
import sys
import glob

def find_zoom_candidates(snapshot_dir, ic_file, min_particles=500):
    """Find halos that are good zoom candidates"""
    
    # Find catalog files
    catalog_files = sorted(glob.glob(f"{snapshot_dir}/fof_subhalo_tab_*.hdf5"))
    if not catalog_files:
        print(f"Error: No catalog files found")
        sys.exit(1)
    
    catalog_file = catalog_files[-1]
    snapshot_num = int(catalog_file.split('_')[-1].replace('.hdf5', ''))
    snapshot_file = f"{snapshot_dir}/snapshot_{snapshot_num:03d}.hdf5"
    
    print(f"Analyzing: {catalog_file}")
    print(f"Minimum particles: {min_particles}")
    
    # Read catalog
    with h5py.File(catalog_file, 'r') as f:
        if 'Subhalo' not in f:
            print("Error: No Subhalo group found")
            sys.exit(1)
        
        sub_len = f['Subhalo/SubhaloLenType'][:]
        sub_mass = f['Subhalo/SubhaloMass'][:]
        sub_offset = f['Subhalo/SubhaloOffsetType'][:]
        
        n_dm = sub_len[:, 1]  # DM particles
        
        print(f"Total subhalos: {len(n_dm)}")
        print(f"Subhalos with >{min_particles} DM particles: {(n_dm > min_particles).sum()}")
    
    # Load IC info
    with h5py.File(ic_file, 'r') as f:
        ic_ids = f['PartType1/ParticleIDs'][:]
        ic_pos = f['PartType1/Coordinates'][:]
        boxsize = f['Header'].attrs['BoxSize']
        
        if boxsize > 1000:
            boxsize_Mpc = boxsize / 1000.0
            pos_Mpc = ic_pos / 1000.0
        else:
            boxsize_Mpc = boxsize
            pos_Mpc = ic_pos
    
    # Load snapshot particle IDs
    with h5py.File(snapshot_file, 'r') as f:
        snap_ids = f['PartType1/ParticleIDs'][:]
    
    print("\nAnalyzing halos (this may take a few minutes)...")
    print(f"{'ID':<8} {'N_DM':<8} {'Mass':<12} {'r99 [Mpc/h]':<12} {'ref_extent':<12} {'Quality':<10}")
    print("-" * 80)
    
    # Sort IDs once for fast lookup
    sort_idx = np.argsort(ic_ids)
    sorted_ids = ic_ids[sort_idx]
    
    candidates = []
    
    # Check top massive halos
    massive_indices = np.argsort(n_dm)[::-1][:50]  # Top 50 by particle count
    
    for halo_id in massive_indices:
        n_particles = n_dm[halo_id]
        
        if n_particles < min_particles:
            continue
        
        mass = sub_mass[halo_id]
        offset = sub_offset[halo_id, 1]
        
        # Get particle IDs for this halo
        halo_ids = snap_ids[offset:offset+n_particles]
        
        # Match to ICs
        idx_in_sorted = np.searchsorted(sorted_ids, halo_ids)
        valid_mask = (idx_in_sorted < len(sorted_ids)) & (sorted_ids[idx_in_sorted] == halo_ids)
        
        if valid_mask.sum() < n_particles * 0.9:
            continue  # Poor match, skip
        
        ic_indices = sort_idx[idx_in_sorted[valid_mask]]
        lagrangian_pos = pos_Mpc[ic_indices]
        
        # Calculate Lagrangian extent
        center = np.median(lagrangian_pos, axis=0)
        shifted = lagrangian_pos - center
        shifted[shifted > boxsize_Mpc/2] -= boxsize_Mpc
        shifted[shifted < -boxsize_Mpc/2] += boxsize_Mpc
        
        distances = np.sqrt(np.sum(shifted**2, axis=1))
        r_99 = np.percentile(distances, 99)
        ref_extent = r_99 * 1.1 / boxsize_Mpc
        
        # Quality assessment
        if ref_extent < 0.1:
            quality = "EXCELLENT"
        elif ref_extent < 0.2:
            quality = "GOOD"
        elif ref_extent < 0.3:
            quality = "OK"
        else:
            quality = "POOR"
        
        candidates.append({
            'id': halo_id,
            'n_dm': n_particles,
            'mass': mass,
            'r99': r_99,
            'ref_extent': ref_extent,
            'quality': quality
        })
        
        print(f"{halo_id:<8} {n_particles:<8} {mass:<12.2e} {r_99:<12.4f} {ref_extent:<12.6f} {quality:<10}")
    
    # Sort by quality (smallest ref_extent = best)
    candidates.sort(key=lambda x: x['ref_extent'])
    
    print("\n" + "="*80)
    print("TOP 10 ZOOM CANDIDATES (by compactness):")
    print("="*80)
    print(f"{'Rank':<6} {'ID':<8} {'N_DM':<8} {'Mass':<12} {'ref_extent':<12} {'Quality':<10}")
    print("-" * 80)
    
    for i, c in enumerate(candidates[:10], 1):
        print(f"{i:<6} {c['id']:<8} {c['n_dm']:<8} {c['mass']:<12.2e} {c['ref_extent']:<12.6f} {c['quality']:<10}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    excellent = [c for c in candidates if c['quality'] == 'EXCELLENT']
    good = [c for c in candidates if c['quality'] == 'GOOD']
    
    if excellent:
        top = excellent[0]
        print(f"\n✓ BEST CHOICE: Halo {top['id']}")
        print(f"  - {top['n_dm']} DM particles")
        print(f"  - Mass: {top['mass']:.2e} M☉")
        print(f"  - ref_extent: {top['ref_extent']:.4f} (only {top['ref_extent']*100:.1f}% of box)")
        print(f"  - Will refine ~{(2*top['ref_extent'])**3*100:.1f}% of box volume")
        print(f"\nTo get coordinates for this halo:")
        print(f"  python find_lagrangian_center.py {snapshot_dir} {ic_file} {top['id']} {snapshot_num}")
    elif good:
        top = good[0]
        print(f"\n✓ RECOMMENDED: Halo {top['id']}")
        print(f"  - {top['n_dm']} DM particles")
        print(f"  - Mass: {top['mass']:.2e} M☉")
        print(f"  - ref_extent: {top['ref_extent']:.4f}")
    else:
        print("\n⚠ No excellent candidates found with current criteria")
        print(f"  Try lowering min_particles or using a higher-resolution parent run")
    
    if len(candidates) == 0:
        print("\n⚠ No candidates found!")
        print("  This parent simulation may be too low resolution for efficient zoom ICs")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./find_zoom_candidates.py <snapshot_dir> <ic_file> [min_particles]")
        print("\nExample:")
        print("  ./find_zoom_candidates.py \\")
        print("    ../output_archive/output_parent_100Mpc_128_music \\")
        print("    ../ICs/IC_parent_100Mpc_128_music.hdf5 \\")
        print("    1000")
        print("\nThis will find halos with >1000 particles that make good zoom candidates")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    ic_file = sys.argv[2]
    min_particles = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    find_zoom_candidates(snapshot_dir, ic_file, min_particles)
