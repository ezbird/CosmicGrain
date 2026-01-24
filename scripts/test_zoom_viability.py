#!/usr/bin/env python3
"""
Complete test for zoom simulation viability
Tests multiple halos and kR values to find good candidates
"""
import numpy as np
import h5py
import sys
import os

def test_halo(snapshot_file, groupcat_file, ic_file, halo_id, kR=1.0):
    """Test if a halo is suitable for zoom"""
    
    # Step 1: Get halo properties from catalog
    with h5py.File(groupcat_file, 'r') as f:
        group_pos = f['Group']['GroupPos'][halo_id]
        group_r200 = f['Group']['Group_R_Crit200'][halo_id]
        group_m200 = f['Group']['Group_M_Crit200'][halo_id]
        group_len = f['Group']['GroupLen'][halo_id]
    
    # Skip tiny halos
    if group_len < 100:
        return None
    
    # Step 2: Extract particles from snapshot
    with h5py.File(snapshot_file, 'r') as f:
        boxsize = f['Header'].attrs['BoxSize']
        if hasattr(boxsize, '__len__'):
            boxsize = boxsize[0]
        
        positions = f['PartType1']['Coordinates'][:]
        ids = f['PartType1']['ParticleIDs'][:]
    
    # Select particles within kR * R200
    selection_radius = kR * group_r200
    dx = positions - group_pos
    dx = np.where(dx > boxsize/2, dx - boxsize, dx)
    dx = np.where(dx < -boxsize/2, dx + boxsize, dx)
    distances = np.sqrt(np.sum(dx**2, axis=1))
    mask = distances <= selection_radius
    selected_ids = ids[mask]
    
    if len(selected_ids) < 50:
        return None
    
    # Step 3: Trace to Lagrangian positions in IC
    with h5py.File(ic_file, 'r') as f:
        ic_boxsize = f['Header'].attrs['BoxSize']
        if hasattr(ic_boxsize, '__len__'):
            ic_boxsize = ic_boxsize[0]
        
        # Detect units
        if ic_boxsize > 1000:
            unit_conversion = 1000.0  # kpc/h to Mpc/h
            ic_boxsize = ic_boxsize / 1000.0
        else:
            unit_conversion = 1.0
        
        ic_positions = f['PartType1']['Coordinates'][:] / unit_conversion
        ic_ids = f['PartType1']['ParticleIDs'][:]
    
    # Map IDs to positions
    id_to_pos = {pid: pos for pid, pos in zip(ic_ids, ic_positions)}
    
    # Find Lagrangian positions
    lag_positions = []
    for sid in selected_ids:
        if sid in id_to_pos:
            lag_positions.append(id_to_pos[sid])
    
    if len(lag_positions) < len(selected_ids) * 0.9:
        return None
    
    lag_positions = np.array(lag_positions)
    
    # Step 4: Compute Lagrangian region size
    # Use trigonometric mean for periodic boundaries
    center = np.zeros(3)
    for axis in range(3):
        theta = 2 * np.pi * lag_positions[:, axis] / ic_boxsize
        mean_cos = np.mean(np.cos(theta))
        mean_sin = np.mean(np.sin(theta))
        mean_angle = np.arctan2(mean_sin, mean_cos)
        center[axis] = (mean_angle * ic_boxsize / (2 * np.pi)) % ic_boxsize
    
    # Compute radius
    dx = lag_positions - center
    dx = np.where(dx > ic_boxsize/2, dx - ic_boxsize, dx)
    dx = np.where(dx < -ic_boxsize/2, dx + ic_boxsize, dx)
    lag_radius = np.max(np.sqrt(np.sum(dx**2, axis=1)))
    
    return {
        'halo_id': halo_id,
        'mass': group_m200,
        'r200': group_r200,
        'npart_fof': group_len,
        'npart_selected': len(selected_ids),
        'lag_radius': lag_radius,
        'lag_center': center,
        'kR': kR
    }

def main():
    if len(sys.argv) != 4:
        print("Usage: python test_zoom_viability.py <snapshot> <groupcat> <parent_ic>")
        sys.exit(1)
    
    snapshot_file = sys.argv[1]
    groupcat_file = sys.argv[2]
    ic_file = sys.argv[3]
    
    print("\n" + "="*70)
    print("TESTING HALOS FOR ZOOM SIMULATION VIABILITY")
    print("="*70)
    
    # Test parameters
    test_halos = list(range(0, 200, 10))  # Test every 10th halo up to 200
    test_kRs = [0.3, 0.5, 0.75, 1.0]
    max_lag_radius = 5.0  # Mpc/h
    
    good_candidates = []
    
    print(f"\nTesting {len(test_halos)} halos with kR values: {test_kRs}")
    print(f"Maximum acceptable Lagrangian radius: {max_lag_radius} Mpc/h\n")
    
    for halo_id in test_halos:
        for kR in test_kRs:
            result = test_halo(snapshot_file, groupcat_file, ic_file, halo_id, kR)
            
            if result is None:
                continue
            
            status = "✅" if result['lag_radius'] < max_lag_radius else "❌"
            
            if result['lag_radius'] < max_lag_radius:
                good_candidates.append(result)
                print(f"Halo {halo_id:3d} | kR={kR:.2f} | "
                      f"M200={result['mass']:.0f}e10 | "
                      f"N={result['npart_selected']:4d} | "
                      f"Lag_R={result['lag_radius']:.2f} Mpc/h {status}")
    
    print("\n" + "="*70)
    
    if good_candidates:
        print(f"\nFOUND {len(good_candidates)} SUITABLE CONFIGURATIONS!\n")
        
        # Sort by Lagrangian radius
        good_candidates.sort(key=lambda x: x['lag_radius'])
        
        print("Best candidates (sorted by Lagrangian radius):")
        print("-"*60)
        
        for i, cand in enumerate(good_candidates[:5]):
            print(f"\n{i+1}. Halo {cand['halo_id']} with kR={cand['kR']}")
            print(f"   Mass: {cand['mass']:.1f} × 10^10 M☉/h")
            print(f"   Particles selected: {cand['npart_selected']}")
            print(f"   Lagrangian radius: {cand['lag_radius']:.3f} Mpc/h")
            
            if i == 0:
                # Print MUSIC config for best candidate
                boxsize = 50.0  # Mpc/h
                center_frac = cand['lag_center'] / boxsize
                extent_frac = (cand['lag_radius'] * 2.5) / boxsize
                
                print(f"\n   MUSIC configuration:")
                print(f"   ref_center = {center_frac[0]:.6f}, {center_frac[1]:.6f}, {center_frac[2]:.6f}")
                print(f"   ref_extent = {extent_frac:.6f}, {extent_frac:.6f}, {extent_frac:.6f}")
    else:
        print("\n⚠️ NO SUITABLE HALOS FOUND!")
        print("\nPossible issues:")
        print("1. Box too small (50 Mpc/h) - halos formed from material across entire volume")
        print("2. Snapshot too early (z=0.5) - structures not yet collapsed")
        print("3. Need even smaller kR values (try 0.1-0.2)")
        print("\nRecommendations:")
        print("- Run parent simulation to z=0")
        print("- Use larger parent box (100-200 Mpc/h)")
        print("- Consider Milky Way-mass halos (~10^12 Msun) instead of clusters")

if __name__ == "__main__":
    main()
