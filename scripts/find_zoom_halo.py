#!/usr/bin/env python3
"""
Find the main halo in a zoom simulation
Usually this is the most massive halo (what you zoomed in on)
"""
import numpy as np
import h5py
import glob
import sys
import os

def read_groupcat_multifile(groupcat_path):
    """Read group catalog from potentially multi-file format"""
    if os.path.isdir(groupcat_path):
        groupcat_files = sorted(glob.glob(os.path.join(groupcat_path, "fof_subhalo_tab_*.hdf5")))
        if not groupcat_files:
            raise FileNotFoundError(f"No group catalog files found in {groupcat_path}")
    else:
        base_path = groupcat_path.replace('.0.hdf5', '').replace('.hdf5', '')
        groupcat_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not groupcat_files:
            groupcat_files = [groupcat_path]
    
    print(f"Reading {len(groupcat_files)} group catalog file(s)...")
    
    group_pos_list = []
    group_r200_list = []
    group_m200_list = []
    group_len_list = []
    
    for fpath in groupcat_files:
        with h5py.File(fpath, 'r') as f:
            if 'Group' in f:
                group_pos_list.append(f['Group']['GroupPos'][:])
                group_r200_list.append(f['Group']['Group_R_Crit200'][:])
                group_m200_list.append(f['Group']['Group_M_Crit200'][:])
                group_len_list.append(f['Group']['GroupLen'][:])
    
    if group_pos_list:
        group_pos = np.concatenate(group_pos_list)
        group_r200 = np.concatenate(group_r200_list)
        group_m200 = np.concatenate(group_m200_list)
        group_len = np.concatenate(group_len_list)
    else:
        raise ValueError("No group data found")
    
    return group_pos, group_r200, group_m200, group_len

if len(sys.argv) < 2:
    print("Usage: python find_zoom_halo.py <groupcat_dir>")
    print("\nExample:")
    print("  python find_zoom_halo.py ../5_output_zoom_512_halo569_50Mpc_dust/groups_026/")
    sys.exit(1)

groupcat_path = sys.argv[1]

print(f"\nAnalyzing halos in: {groupcat_path}")
print("="*70)

group_pos, group_r200, group_m200, group_len = read_groupcat_multifile(groupcat_path)

print(f"\nTotal halos found: {len(group_pos)}")

# Sort by mass
sorted_indices = np.argsort(group_m200)[::-1]  # Descending order

print(f"\n{'='*70}")
print(f"TOP 10 MOST MASSIVE HALOS:")
print(f"{'='*70}")
print(f"{'Rank':<6} {'HaloID':<8} {'M200':<15} {'R200':<12} {'N_part':<10} {'Position':<30}")
print("-"*70)

for rank, idx in enumerate(sorted_indices[:10]):
    m200_msun = group_m200[idx] * 1e10  # Convert to Msun/h
    pos = group_pos[idx]
    print(f"{rank+1:<6} {idx:<8} {m200_msun:.3e} {group_r200[idx]:<12.3f} {group_len[idx]:<10} "
          f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

print(f"\n{'='*70}")
print(f"MOST LIKELY TARGET HALO:")
print(f"{'='*70}")

# The target is almost certainly the most massive
target_idx = sorted_indices[0]
target_pos = group_pos[target_idx]
target_r200 = group_r200[target_idx]
target_m200 = group_m200[target_idx] * 1e10

print(f"Halo ID: {target_idx}")
print(f"M200: {target_m200:.3e} Msun/h")
print(f"R200: {target_r200:.3f} (code units)")
print(f"Position: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
print(f"N_particles: {group_len[target_idx]}")

print(f"\n{'='*70}")
print(f"RECOMMENDED COMMAND:")
print(f"{'='*70}")
print(f"\npython extract_halo_particles.py \\")
print(f"    <snapshot_dir> \\")
print(f"    {groupcat_path} \\")
print(f"    {target_idx} \\")
print(f"    1.5 \\")
print(f"    -o halo{target_idx}_1.5R200.txt")

# Also show if center roughly matches what user expected
print(f"\n{'='*70}")
print(f"SANITY CHECK:")
print(f"{'='*70}")
print(f"If you previously found the halo center around [23947, 23949, 24405],")
print(f"the most massive halo should be near that location.")
print(f"Actual position: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")

# Check distance to expected position
expected = np.array([23947, 23949, 24405])
distance = np.sqrt(np.sum((target_pos - expected)**2))
print(f"Distance from expected: {distance:.2f} (code units)")
if distance < 500:
    print("✓ Position matches well!")
elif distance < 1000:
    print("⚠ Position is reasonably close")
else:
    print("✗ Position differs significantly - check if this is the right halo")
