#!/usr/bin/env python3
"""
Extract particle IDs from a halo using subfind R200
Outputs in TYPE PARTICLE_ID format
"""
import numpy as np
import h5py
import argparse
import os
import glob

def read_groupcat_multifile(groupcat_path):
    """Read group catalog from potentially multi-file catalog"""
    # Handle directory or file input
    if os.path.isdir(groupcat_path):
        groupcat_files = sorted(glob.glob(os.path.join(groupcat_path, "fof_subhalo_tab_*.hdf5")))
        if not groupcat_files:
            raise FileNotFoundError(f"No group catalog files found in {groupcat_path}")
    else:
        base_path = groupcat_path.replace('.0.hdf5', '').replace('.hdf5', '')
        groupcat_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not groupcat_files:
            groupcat_files = [groupcat_path]
    
    print(f"  Reading {len(groupcat_files)} group catalog file(s)")
    
    # Read group data from all files
    group_pos_list = []
    group_r200_list = []
    group_m200_list = []
    group_len_list = []
    
    for fpath in groupcat_files:
        with h5py.File(fpath, 'r') as f:
            if 'Group' in f and 'GroupPos' in f['Group']:
                group_pos_list.append(f['Group']['GroupPos'][:])
                group_r200_list.append(f['Group']['Group_R_Crit200'][:])
                group_m200_list.append(f['Group']['Group_M_Crit200'][:])
                group_len_list.append(f['Group']['GroupLen'][:])
    
    # Concatenate all groups
    if group_pos_list:
        group_pos = np.concatenate(group_pos_list)
        group_r200 = np.concatenate(group_r200_list)
        group_m200 = np.concatenate(group_m200_list)
        group_len = np.concatenate(group_len_list)
    else:
        raise ValueError("No groups found in catalog files!")
    
    print(f"  Total groups in catalog: {len(group_pos)}")
    
    return {
        'GroupPos': group_pos,
        'Group_R_Crit200': group_r200,
        'Group_M_Crit200': group_m200,
        'GroupLen': group_len
    }

def read_snapshot_multifile(snapshot_path, ptype):
    """Read particle data from potentially multi-file snapshot"""
    # Handle directory or file input
    if os.path.isdir(snapshot_path):
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not snapshot_files:
            snapshot_files = [snapshot_path]
    
    positions_list = []
    ids_list = []
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            ptype_key = f'PartType{ptype}'
            if ptype_key in f:
                positions_list.append(f[ptype_key]['Coordinates'][:])
                ids_list.append(f[ptype_key]['ParticleIDs'][:])
    
    if positions_list:
        positions = np.concatenate(positions_list)
        ids = np.concatenate(ids_list)
    else:
        positions = np.array([]).reshape(0, 3)
        ids = np.array([], dtype=np.int64)
    
    return positions, ids

def read_groupcat_multifile(groupcat_path):
    """Read group catalog from potentially multi-file format"""
    # Handle directory or file input
    if os.path.isdir(groupcat_path):
        groupcat_files = sorted(glob.glob(os.path.join(groupcat_path, "fof_subhalo_tab_*.hdf5")))
        if not groupcat_files:
            raise FileNotFoundError(f"No group catalog files found in {groupcat_path}")
    else:
        base_path = groupcat_path.replace('.0.hdf5', '').replace('.hdf5', '')
        groupcat_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not groupcat_files:
            groupcat_files = [groupcat_path]
    
    # Read and concatenate group data
    group_pos_list = []
    group_r200_list = []
    group_m200_list = []
    group_len_list = []
    
    print(f"  Reading {len(groupcat_files)} group catalog file(s)...")
    
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
        raise ValueError("No group data found in catalog files")
    
    print(f"  Total halos found: {len(group_pos)}")
    
    return group_pos, group_r200, group_m200, group_len

def extract_halo_particles(snapshot_path, groupcat_path, halo_id, kR=1.5, 
                           particle_types=[0, 1, 4, 6], output_file='halo_particles.txt',
                           manual_center=None, manual_radius=None):
    """
    Extract particles from a halo using subfind R200 or manual parameters
    
    Parameters:
    -----------
    snapshot_path : str
        Path to snapshot file or directory
    groupcat_path : str
        Path to subfind group catalog file or directory
    halo_id : int
        Halo ID in group catalog
    kR : float
        Multiplier for R200 (e.g., 1.5 means 1.5 × R200)
    particle_types : list
        Which particle types to extract (0=gas, 1=DM, 4=stars, 6=dust)
    output_file : str
        Output filename
    manual_center : array-like, optional
        Manual halo center [x, y, z] in code units (overrides subfind)
    manual_radius : float, optional
        Manual selection radius in code units (overrides R200)
    """
    
    print(f"\n{'='*60}")
    print(f"Extracting particles from halo {halo_id}")
    print(f"{'='*60}")
    
    # Read halo properties from subfind
    print(f"\nReading subfind catalog: {groupcat_path}")
    group_pos, group_r200, group_m200, group_len = read_groupcat_multifile(groupcat_path)
    
    if halo_id >= len(group_pos):
        raise ValueError(f"Halo ID {halo_id} out of range (only {len(group_pos)} halos found)")
    
    print(f"\nHalo {halo_id} properties:")
    print(f"  Subfind position: [{group_pos[halo_id][0]:.2f}, {group_pos[halo_id][1]:.2f}, {group_pos[halo_id][2]:.2f}] (code units)")
    print(f"  M200: {group_m200[halo_id]:.2e} (10^10 Msun/h)")
    print(f"  R200: {group_r200[halo_id]:.3f} (code units)")
    print(f"  N_particles (FOF): {group_len[halo_id]}")
    
    # Use manual parameters if provided
    if manual_center is not None:
        halo_center = np.array(manual_center)
        print(f"\n  Using MANUAL center: [{halo_center[0]:.2f}, {halo_center[1]:.2f}, {halo_center[2]:.2f}]")
    else:
        halo_center = group_pos[halo_id]
        print(f"\n  Using subfind center")
    
    if manual_radius is not None:
        selection_radius = manual_radius
        print(f"  Using MANUAL radius: {selection_radius:.2f} (code units)")
    else:
        selection_radius = kR * group_r200[halo_id]
        print(f"  Using kR × R200: {kR} × {group_r200[halo_id]:.3f} = {selection_radius:.3f} (code units)")
    
    # Read box size from first snapshot file
    if os.path.isdir(snapshot_path):
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        first_file = snapshot_files[0]
    else:
        first_file = snapshot_path
    
    with h5py.File(first_file, 'r') as f:
        boxsize = f['Header'].attrs['BoxSize']
        if hasattr(boxsize, '__len__'):
            boxsize = boxsize[0]
    
    print(f"  Box size: {boxsize:.2f}")
    
    # Extract particles for each type
    all_particles = []
    type_names = {0: 'Gas', 1: 'DM', 4: 'Stars', 6: 'Dust'}
    
    print(f"\nExtracting particle types: {particle_types}")
    print(f"Selection radius: {selection_radius:.2f} (code units)")
    print("-" * 60)
    
    for ptype in particle_types:
        print(f"\nType {ptype} ({type_names.get(ptype, 'Unknown')}):")
        
        # Read particle data
        positions, ids = read_snapshot_multifile(snapshot_path, ptype)
        
        if len(positions) == 0:
            print(f"  No particles of this type in snapshot")
            continue
        
        print(f"  Total particles in snapshot: {len(positions):,}")
        
        # Calculate distances with periodic boundaries
        dx = positions - halo_center
        dx = np.where(dx > boxsize/2, dx - boxsize, dx)
        dx = np.where(dx < -boxsize/2, dx + boxsize, dx)
        distances = np.sqrt(np.sum(dx**2, axis=1))
        
        # Select particles within radius
        mask = distances <= selection_radius
        selected_ids = ids[mask]
        
        print(f"  Selected particles: {len(selected_ids):,}")
        
        if len(selected_ids) > 0:
            # Store with type label
            for pid in selected_ids:
                all_particles.append((ptype, int(pid)))
    
    # Write output
    print(f"\n{'='*60}")
    print(f"Writing output to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write(f"# Particle IDs for halo {halo_id}\n")
        f.write("# Format: TYPE PARTICLE_ID\n")
        f.write("# TYPE: 0=Gas, 1=DM, 4=Stars, 6=Dust\n")
        
        if manual_center is not None or manual_radius is not None:
            f.write("# Manual parameters used:\n")
            if manual_center is not None:
                f.write(f"#   Center: [{manual_center[0]:.2f}, {manual_center[1]:.2f}, {manual_center[2]:.2f}]\n")
            if manual_radius is not None:
                f.write(f"#   Radius: {manual_radius:.2f}\n")
        else:
            f.write(f"# Subfind halo {halo_id}: kR={kR}, R200={group_r200[halo_id]:.3f}, radius={selection_radius:.3f}\n")
        
        for ptype, pid in all_particles:
            f.write(f"{ptype} {pid}\n")
    
    print(f"Total particles written: {len(all_particles):,}")
    
    # Summary
    print("\nSummary by type:")
    for ptype in particle_types:
        count = sum(1 for t, _ in all_particles if t == ptype)
        if count > 0:
            print(f"  Type {ptype} ({type_names.get(ptype, 'Unknown')}): {count:,} particles")
    
    print(f"{'='*60}\n")
    
    return len(all_particles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract particle IDs from a halo using subfind R200',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use subfind R200 with default kR=1.5 (directory paths)
  python extract_halo_particles.py snapdir_026/ groups_026/ 569 -o halo569.txt
  
  # Use subfind R200 with single file paths
  python extract_halo_particles.py snapshot_026.0.hdf5 fof_subhalo_tab_026.0.hdf5 569 -o halo569.txt
  
  # Use 2.0 × R200
  python extract_halo_particles.py snapdir_026/ groups_026/ 569 2.0 -o halo569.txt
  
  # Extract only gas, stars, dust (no DM)
  python extract_halo_particles.py snapdir_026/ groups_026/ 569 --types 0 4 6 -o halo569.txt
  
  # Use manual center and radius
  python extract_halo_particles.py --manual snapdir_026/ 23947 23949 24405 2000 569 -o halo569.txt
        """
    )
    
    parser.add_argument('--manual', action='store_true',
                       help='Use manual center and radius instead of subfind')
    parser.add_argument('snapshot', help='Snapshot file or directory (e.g., snapdir_026/ or snapshot_026.0.hdf5)')
    parser.add_argument('groupcat_or_x', help='Group catalog file/directory (or X coordinate if --manual)')
    parser.add_argument('halo_id_or_y', help='Halo ID (or Y coordinate if --manual)')
    parser.add_argument('kR_or_z', help='R200 multiplier (e.g., 1.5) or Z coordinate if --manual')
    parser.add_argument('radius_if_manual', nargs='?', help='Radius if --manual mode')
    parser.add_argument('halo_id_if_manual', nargs='?', help='Halo ID if --manual mode (optional, for labeling)')
    parser.add_argument('--types', type=int, nargs='+', default=[0, 1, 4, 6],
                       help='Particle types to extract (default: 0 1 4 6)')
    parser.add_argument('-o', '--output', default='halo_particles.txt',
                       help='Output filename (default: halo_particles.txt)')
    
    args = parser.parse_args()
    
    if args.manual:
        # Manual mode: snapshot X Y Z radius [halo_id]
        snapshot_path = args.snapshot
        manual_center = [float(args.groupcat_or_x), 
                        float(args.halo_id_or_y), 
                        float(args.kR_or_z)]
        manual_radius = float(args.radius_if_manual) if args.radius_if_manual else None
        
        # Halo ID is optional in manual mode (just for labeling)
        halo_id = int(args.halo_id_if_manual) if args.halo_id_if_manual else 0
        
        # Need a groupcat file - find groups directory
        if os.path.isdir(snapshot_path):
            snap_dir = snapshot_path.rstrip('/')
            output_dir = os.path.dirname(snap_dir)
            snap_num = snap_dir.split('_')[-1]
            groupcat_path = os.path.join(output_dir, f"groups_{snap_num}")
            if not os.path.exists(groupcat_path):
                # Try same directory as snapshot
                groupcat_path = snap_dir.replace('snapdir', 'groups')
        else:
            snap_dir = os.path.dirname(snapshot_path)
            snap_file = os.path.basename(snapshot_path)
            snap_num = snap_file.split('_')[1].split('.')[0]
            output_dir = os.path.dirname(snap_dir) if 'snapdir' in snap_dir else snap_dir
            groupcat_path = os.path.join(output_dir, f"groups_{snap_num}")
        
        if not os.path.exists(groupcat_path):
            print(f"Warning: Could not find group catalog at {groupcat_path}")
            print("Continuing with manual mode anyway...")
            groupcat_path = snapshot_path  # Dummy value
        
        if manual_radius is None:
            print("Error: --manual mode requires: snapshot X Y Z radius [halo_id]")
            parser.print_help()
            exit(1)
        
        extract_halo_particles(
            snapshot_path, groupcat_path, halo_id,
            particle_types=args.types,
            output_file=args.output,
            manual_center=manual_center,
            manual_radius=manual_radius
        )
    else:
        # Normal mode: snapshot groupcat halo_id [kR]
        snapshot_path = args.snapshot
        groupcat_path = args.groupcat_or_x
        halo_id = int(args.halo_id_or_y)
        kR = float(args.kR_or_z) if args.kR_or_z else 1.5
        
        extract_halo_particles(
            snapshot_path, groupcat_path, halo_id, 
            kR=kR,
            particle_types=args.types,
            output_file=args.output
        )
