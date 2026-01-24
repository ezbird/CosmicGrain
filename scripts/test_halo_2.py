#!/usr/bin/env python3
"""
Track a halo across snapshots by POSITION (not ID!)
Verifies the halo is stable and suitable for zoom
"""

import h5py
import numpy as np
import sys

def find_halo_at_position(catalog_file, target_pos, search_radius=2.0):
    """
    Find halo closest to target position within search radius
    Returns: (index, distance, mass, position) or None if not found
    """
    with h5py.File(catalog_file, 'r') as f:
        pos = f['Group/GroupPos'][:] / 1000.0 / 0.6732  # Convert to Mpc/h
        mass = f['Group/GroupMass'][:] * 1e10 / 0.6732  # Convert to Msun
        
        # Calculate distances (handling periodic boundaries)
        dx = pos - target_pos
        # Periodic wrap
        dx[dx > 25.0] -= 50.0
        dx[dx < -25.0] += 50.0
        
        dist = np.sqrt(np.sum(dx**2, axis=1))
        
        # Find closest halo
        idx = np.argmin(dist)
        min_dist = dist[idx]
        
        if min_dist < search_radius:
            return idx, min_dist, mass[idx], pos[idx]
        else:
            return None


def track_halo(target_pos, target_halo_id, snapshot_files):
    """Track a halo across multiple snapshots by position"""
    
    print(f"\n{'='*80}")
    print(f"TRACKING HALO BY POSITION (not ID!)")
    print(f"{'='*80}")
    print(f"Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}) Mpc/h")
    print(f"Original halo ID at z~0.5: {target_halo_id}")
    print(f"\nSearching within 2 Mpc/h radius at each snapshot...")
    print(f"\n{'Snapshot':>10s} {'z':>8s} {'Catalog ID':>12s} {'Δpos':>10s} {'Mass':>13s} {'Δmass':>10s}")
    print("-" * 80)
    
    results = []
    
    for snap_file in snapshot_files:
        with h5py.File(snap_file, 'r') as f:
            z = f['Header'].attrs['Redshift']
        
        result = find_halo_at_position(snap_file, target_pos, search_radius=2.0)
        
        snap_num = snap_file.split('_')[-1].split('.')[0]
        
        if result is not None:
            idx, dist, mass, pos = result
            results.append({
                'snap': snap_num,
                'z': z,
                'idx': idx,
                'dist': dist,
                'mass': mass,
                'pos': pos
            })
            
            # Calculate mass change relative to z~0.5
            if len(results) > 0 and results[0]['snap'] == '011':
                ref_mass = results[0]['mass']
                dmass = (mass - ref_mass) / ref_mass * 100.0
                dmass_str = f"{dmass:+9.1f}%"
            else:
                dmass_str = "   -"
            
            print(f"{snap_num:>10s} {z:>8.3f} {idx:>12d} {dist:>9.3f} Mpc {mass:>12.2e} M☉ {dmass_str:>10s}")
        else:
            print(f"{snap_num:>10s} {z:>8.3f} {'NOT FOUND':>12s} {'> 2 Mpc':>10s} {'-':>13s} {'-':>10s}")
            results.append(None)
    
    return results


def analyze_stability(results):
    """Analyze if halo is stable enough for zoom"""
    print(f"\n{'='*80}")
    print(f"STABILITY ANALYSIS")
    print(f"{'='*80}")
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) < 2:
        print(f"❌ UNSTABLE: Halo not found in multiple snapshots!")
        print(f"   Only found in {len(valid_results)} snapshot(s)")
        return False
    
    # Check position stability
    max_drift = max([r['dist'] for r in valid_results])
    print(f"\n📍 Position Stability:")
    print(f"   Max drift from target: {max_drift:.3f} Mpc/h")
    
    if max_drift < 0.5:
        print(f"   ✅ EXCELLENT: Halo position very stable (<0.5 Mpc/h)")
        pos_stable = True
    elif max_drift < 1.0:
        print(f"   ✅ GOOD: Halo position stable (<1 Mpc/h)")
        pos_stable = True
    elif max_drift < 2.0:
        print(f"   ⚠️  MARGINAL: Halo moves up to {max_drift:.2f} Mpc/h")
        pos_stable = True
    else:
        print(f"   ❌ UNSTABLE: Halo position changes by {max_drift:.2f} Mpc/h")
        pos_stable = False
    
    # Check mass evolution
    masses = [r['mass'] for r in valid_results]
    mass_changes = []
    
    print(f"\n⚖️  Mass Evolution:")
    for i in range(len(valid_results) - 1):
        m1 = valid_results[i]['mass']
        m2 = valid_results[i+1]['mass']
        z1 = valid_results[i]['z']
        z2 = valid_results[i+1]['z']
        
        change = (m2 - m1) / m1 * 100.0
        mass_changes.append(abs(change))
        
        print(f"   z={z1:.2f} → z={z2:.2f}: {change:+.1f}% "
              f"({m1:.2e} → {m2:.2e} M☉)")
    
    if mass_changes:
        max_change = max(mass_changes)
        
        if max_change < 10:
            print(f"   ✅ EXCELLENT: Mass changes <10% between snapshots")
            mass_stable = True
        elif max_change < 20:
            print(f"   ✅ GOOD: Mass changes <20% between snapshots")
            mass_stable = True
        elif max_change < 30:
            print(f"   ⚠️  MARGINAL: Up to {max_change:.1f}% mass change")
            mass_stable = True
        else:
            print(f"   ❌ UNSTABLE: {max_change:.1f}% mass change (major merger?)")
            mass_stable = False
    else:
        mass_stable = True
    
    # Overall verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}")
    
    if pos_stable and mass_stable and len(valid_results) >= 3:
        print(f"✅ STABLE: Halo is suitable for zoom simulation!")
        print(f"   Found in {len(valid_results)}/{len(results)} snapshots")
        print(f"   Position drift: {max_drift:.3f} Mpc/h")
        print(f"   Max mass change: {max(mass_changes) if mass_changes else 0:.1f}%")
        return True
    elif pos_stable and mass_stable:
        print(f"⚠️  MARGINAL: Halo somewhat stable but limited data")
        print(f"   Only found in {len(valid_results)}/{len(results)} snapshots")
        return True
    else:
        print(f"❌ UNSTABLE: Not suitable for zoom!")
        if not pos_stable:
            print(f"   Position too variable ({max_drift:.2f} Mpc/h)")
        if not mass_stable:
            print(f"   Mass too variable ({max(mass_changes):.1f}%)")
        return False


def main():
    # Halo 378 from snapshot_011 (z=0.5)
    target_halo_id = 378
    target_pos = np.array([25.385, 15.568, 32.797])  # Mpc/h
    
    # Snapshots to check
    snapshot_files = [
        '../output_parent_50Mpc_128_music/fof_subhalo_tab_009.hdf5',  # z~1.0
        '../output_parent_50Mpc_128_music/fof_subhalo_tab_010.hdf5',  # z~0.75
        '../output_parent_50Mpc_128_music/fof_subhalo_tab_011.hdf5',  # z~0.5
        '../output_parent_50Mpc_128_music/fof_subhalo_tab_012.hdf5',  # z~0.25
        '../output_parent_50Mpc_128_music/fof_subhalo_tab_013.hdf5',  # z~0.0
    ]
    
    # Track halo
    results = track_halo(target_pos, target_halo_id, snapshot_files)
    
    # Analyze stability
    is_stable = analyze_stability(results)
    
    # Recommendation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    
    if is_stable:
        print(f"""
✅ USE THIS HALO for zoom simulation!

Position for MUSIC2 config:
  ref_center = {target_pos[0]/50:.5f}, {target_pos[1]/50:.5f}, {target_pos[2]/50:.5f}
  
  (Halo at: {target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f} Mpc/h)
  (In box units: ÷50 for MUSIC2 config)

Next steps:
  1. Update zoom_from_parent.conf with this ref_center
  2. Set ref_extent = 0.12, 0.12, 0.12  (6 Mpc/h cube)
  3. Run: ./MUSIC zoom_from_parent.conf
""")
    else:
        print(f"""
❌ DO NOT use this halo - too unstable!

Try one of these alternatives from your list:
  - Halo 492: (35.651, 19.708, 6.943), isolation 20.6 Mpc/h
  - Halo 495: (20.447, 24.691, 2.238), isolation 26.3 Mpc/h
  - Halo 504: (46.082, 32.384, 28.784), isolation 23.4 Mpc/h

Run this script again with a different halo position.
""")


if __name__ == '__main__':
    main()
