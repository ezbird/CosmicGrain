import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import shutil
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def decompress_if_needed(filepath):
    """Decompress .gz file to temporary location if needed"""
    if filepath.endswith('.gz'):
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.hdf5')
        os.close(temp_fd)
        
        # Decompress
        with gzip.open(filepath, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return temp_path, True  # Return path and flag indicating it's temporary
    else:
        return filepath, False

def cleanup_temp_file(filepath, is_temp):
    """Clean up temporary file if needed"""
    if is_temp and os.path.exists(filepath):
        os.remove(filepath)

def read_multifile_snapshot(snapdir):
    """Read a multi-file Gadget-4 snapshot from a directory."""
    # Look for both compressed and uncompressed files
    snap_files_hdf5 = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    snap_files_gz = sorted(glob.glob(os.path.join(snapdir, "*.hdf5.gz")))
    
    # Prefer uncompressed files if available
    if snap_files_hdf5:
        snap_files = snap_files_hdf5
        print(f"Found {len(snap_files)} uncompressed files in {snapdir}")
    elif snap_files_gz:
        snap_files = snap_files_gz
        print(f"Found {len(snap_files)} compressed files in {snapdir}")
    else:
        print(f"Warning: No HDF5 files found in {snapdir}")
        return None, None, None
    
    combined_data = {}
    header = None
    temp_files = []
    
    try:
        for snap_file in snap_files:
            # Decompress if needed
            file_path, is_temp = decompress_if_needed(snap_file)
            if is_temp:
                temp_files.append(file_path)
            
            with h5py.File(file_path, 'r') as f:
                if header is None:
                    header = dict(f['Header'].attrs)
                
                for part_type in ['PartType0', 'PartType1', 'PartType4']:
                    if part_type in f:
                        coords = f[part_type]['Coordinates'][:]
                        
                        # Also get masses if available
                        masses = None
                        if 'Masses' in f[part_type]:
                            masses = f[part_type]['Masses'][:]
                        elif part_type == 'PartType0':  # Gas particles
                            masses = np.ones(len(coords)) * header['MassTable'][0]
                        elif part_type == 'PartType1':  # DM particles  
                            masses = np.ones(len(coords)) * header['MassTable'][1]
                        
                        if part_type not in combined_data:
                            combined_data[part_type] = {'coords': [], 'masses': []}
                        
                        combined_data[part_type]['coords'].append(coords)
                        if masses is not None:
                            combined_data[part_type]['masses'].append(masses)
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            cleanup_temp_file(temp_file, True)
    
    # Concatenate data
    final_data = {}
    for part_type, data in combined_data.items():
        if data['coords']:
            final_data[part_type] = {
                'coords': np.concatenate(data['coords'], axis=0),
                'masses': np.concatenate(data['masses'], axis=0) if data['masses'] else None
            }
    
    return final_data, header, snap_files[0]

def read_single_file_snapshot(filepath):
    """Read a single snapshot file (handles .gz automatically)"""
    print(f"Reading single file: {filepath}")
    
    file_path, is_temp = decompress_if_needed(filepath)
    
    try:
        with h5py.File(file_path, 'r') as f:
            header = dict(f['Header'].attrs)
            
            final_data = {}
            for part_type in ['PartType0', 'PartType1', 'PartType4']:
                if part_type in f:
                    coords = f[part_type]['Coordinates'][:]
                    
                    # Get masses
                    masses = None
                    if 'Masses' in f[part_type]:
                        masses = f[part_type]['Masses'][:]
                    elif part_type == 'PartType0':
                        masses = np.ones(len(coords)) * header['MassTable'][0]
                    elif part_type == 'PartType1':
                        masses = np.ones(len(coords)) * header['MassTable'][1]
                    
                    final_data[part_type] = {
                        'coords': coords,
                        'masses': masses
                    }
            
            return final_data, header, filepath
            
    finally:
        cleanup_temp_file(file_path, is_temp)

def create_density_map(coords, masses, box_size, resolution=512):
    """Create a 2D density map from particle coordinates"""
    # Project to 2D (X-Y plane)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=resolution, 
                                       range=[[0, box_size], [0, box_size]], 
                                       weights=masses)
    
    # Convert to proper density (mass per unit area)
    pixel_area = (box_size / resolution)**2
    density_map = H.T / pixel_area  # Transpose for correct orientation
    
    return density_map, xedges, yedges

def process_single_file_snapshot_density(args):
    """Process a single snapshot file with density maps"""
    frame, filename, output_folder = args
    
    print(f"Processing {filename}...")
    
    particle_data, header, sample_file = read_single_file_snapshot(filename)
    
    if particle_data is None:
        return f"Skipped {filename} - no data"
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create subplots for different components
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    box_size = header['BoxSize']
    time = header['Time']
    redshift = 1.0 / time - 1.0
    snap_num = int(os.path.basename(filename).split("_")[-1].split(".")[0])
    
    # Gas density map
    if 'PartType0' in particle_data:
        gas_coords = particle_data['PartType0']['coords']
        gas_masses = particle_data['PartType0']['masses']
        
        density_map, xedges, yedges = create_density_map(gas_coords, gas_masses, box_size)
        
        im1 = axes[0,0].imshow(np.log10(density_map + 1e-10), 
                              extent=[0, box_size, 0, box_size],
                              origin='lower', cmap='viridis', aspect='equal')
        axes[0,0].set_title(f'Gas Density Map\n{len(gas_coords):,} particles')
        axes[0,0].set_xlabel('X [kpc]')
        axes[0,0].set_ylabel('Y [kpc]')
        plt.colorbar(im1, ax=axes[0,0], label='log₁₀(Σ) [M☉/kpc²]')
    else:
        axes[0,0].text(0.5, 0.5, 'No Gas Particles', ha='center', va='center', 
                      transform=axes[0,0].transAxes, fontsize=20)
        axes[0,0].set_title('Gas Density Map')
    
    # Dark matter density map
    if 'PartType1' in particle_data:
        dm_coords = particle_data['PartType1']['coords']
        dm_masses = particle_data['PartType1']['masses']
        
        density_map, xedges, yedges = create_density_map(dm_coords, dm_masses, box_size)
        
        im2 = axes[0,1].imshow(np.log10(density_map + 1e-10), 
                              extent=[0, box_size, 0, box_size],
                              origin='lower', cmap='plasma', aspect='equal')
        axes[0,1].set_title(f'Dark Matter Density Map\n{len(dm_coords):,} particles')
        axes[0,1].set_xlabel('X [kpc]')
        axes[0,1].set_ylabel('Y [kpc]')
        plt.colorbar(im2, ax=axes[0,1], label='log₁₀(Σ) [M☉/kpc²]')
    else:
        axes[0,1].text(0.5, 0.5, 'No DM Particles', ha='center', va='center', 
                      transform=axes[0,1].transAxes, fontsize=20)
        axes[0,1].set_title('Dark Matter Density Map')
    
    # Star particles (if any) - scatter plot since usually fewer
    if 'PartType4' in particle_data:
        star_coords = particle_data['PartType4']['coords']
        
        # If too many stars, sample them
        if len(star_coords) > 10000:
            indices = np.random.choice(len(star_coords), 10000, replace=False)
            star_coords_plot = star_coords[indices]
        else:
            star_coords_plot = star_coords
            
        axes[1,0].scatter(star_coords_plot[:, 0], star_coords_plot[:, 1], 
                         s=1, c='red', alpha=0.7)
        axes[1,0].set_title(f'Star Particles\n{len(star_coords):,} total')
        axes[1,0].set_xlabel('X [kpc]')
        axes[1,0].set_ylabel('Y [kpc]')
        axes[1,0].set_xlim(0, box_size)
        axes[1,0].set_ylim(0, box_size)
    else:
        axes[1,0].text(0.5, 0.5, 'No Star Particles', ha='center', va='center', 
                      transform=axes[1,0].transAxes, fontsize=20)
        axes[1,0].set_title('Star Particles')
        axes[1,0].set_xlim(0, box_size)
        axes[1,0].set_ylim(0, box_size)
    
    # Combined density map (gas + dark matter)
    total_density = np.zeros((512, 512))
    if 'PartType0' in particle_data:
        gas_density, _, _ = create_density_map(particle_data['PartType0']['coords'], 
                                             particle_data['PartType0']['masses'], box_size)
        total_density += gas_density
    if 'PartType1' in particle_data:
        dm_density, _, _ = create_density_map(particle_data['PartType1']['coords'], 
                                            particle_data['PartType1']['masses'], box_size)
        total_density += dm_density
    
    im3 = axes[1,1].imshow(np.log10(total_density + 1e-10), 
                          extent=[0, box_size, 0, box_size],
                          origin='lower', cmap='magma', aspect='equal')
    axes[1,1].set_title('Total Matter Density')
    axes[1,1].set_xlabel('X [kpc]')
    axes[1,1].set_ylabel('Y [kpc]')
    plt.colorbar(im3, ax=axes[1,1], label='log₁₀(Σ) [M☉/kpc²]')
    
    # Add stars as overlay if they exist
    if 'PartType4' in particle_data:
        star_coords = particle_data['PartType4']['coords']
        if len(star_coords) > 5000:
            indices = np.random.choice(len(star_coords), 5000, replace=False)
            star_coords_plot = star_coords[indices]
        else:
            star_coords_plot = star_coords
        axes[1,1].scatter(star_coords_plot[:, 0], star_coords_plot[:, 1], 
                         s=0.5, c='yellow', alpha=0.8, marker='*')
    
    # Overall title
    num_stars = len(particle_data.get('PartType4', {}).get('coords', []))
    fig.suptitle(f'Snapshot {snap_num:03d} - z={redshift:.2f} - t={time:.3f} - Stars: {num_stars:,}', 
                fontsize=20)
    
    plt.tight_layout()
    
    # Save the figure
    frame_path = os.path.join(output_folder, f"density_frame_{frame:03d}.png")
    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return f"Saved density frame {frame:03d} for snapshot {snap_num}"
    """Process a single multi-file snapshot directory with density maps"""
    frame, snapdir, output_folder = args
    
    print(f"Processing {snapdir}...")
    
    particle_data, header, sample_file = read_multifile_snapshot(snapdir)
    
    if particle_data is None:
        return f"Skipped {snapdir} - no data"
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create subplots for different components
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    box_size = header['BoxSize']
    time = header['Time']
    redshift = 1.0 / time - 1.0
    snap_num = int(os.path.basename(snapdir).split("_")[-1])
    
    # Gas density map
    if 'PartType0' in particle_data:
        gas_coords = particle_data['PartType0']['coords']
        gas_masses = particle_data['PartType0']['masses']
        
        density_map, xedges, yedges = create_density_map(gas_coords, gas_masses, box_size)
        
        im1 = axes[0,0].imshow(np.log10(density_map + 1e-10), 
                              extent=[0, box_size, 0, box_size],
                              origin='lower', cmap='viridis', aspect='equal')
        axes[0,0].set_title(f'Gas Density Map\n{len(gas_coords):,} particles')
        axes[0,0].set_xlabel('X [kpc]')
        axes[0,0].set_ylabel('Y [kpc]')
        plt.colorbar(im1, ax=axes[0,0], label='log₁₀(Σ) [M☉/kpc²]')
    else:
        axes[0,0].text(0.5, 0.5, 'No Gas Particles', ha='center', va='center', 
                      transform=axes[0,0].transAxes, fontsize=20)
        axes[0,0].set_title('Gas Density Map')
    
    # Dark matter density map
    if 'PartType1' in particle_data:
        dm_coords = particle_data['PartType1']['coords']
        dm_masses = particle_data['PartType1']['masses']
        
        density_map, xedges, yedges = create_density_map(dm_coords, dm_masses, box_size)
        
        im2 = axes[0,1].imshow(np.log10(density_map + 1e-10), 
                              extent=[0, box_size, 0, box_size],
                              origin='lower', cmap='plasma', aspect='equal')
        axes[0,1].set_title(f'Dark Matter Density Map\n{len(dm_coords):,} particles')
        axes[0,1].set_xlabel('X [kpc]')
        axes[0,1].set_ylabel('Y [kpc]')
        plt.colorbar(im2, ax=axes[0,1], label='log₁₀(Σ) [M☉/kpc²]')
    else:
        axes[0,1].text(0.5, 0.5, 'No DM Particles', ha='center', va='center', 
                      transform=axes[0,1].transAxes, fontsize=20)
        axes[0,1].set_title('Dark Matter Density Map')
    
    # Star particles (if any) - scatter plot since usually fewer
    if 'PartType4' in particle_data:
        star_coords = particle_data['PartType4']['coords']
        
        # If too many stars, sample them
        if len(star_coords) > 10000:
            indices = np.random.choice(len(star_coords), 10000, replace=False)
            star_coords_plot = star_coords[indices]
        else:
            star_coords_plot = star_coords
            
        axes[1,0].scatter(star_coords_plot[:, 0], star_coords_plot[:, 1], 
                         s=1, c='red', alpha=0.7)
        axes[1,0].set_title(f'Star Particles\n{len(star_coords):,} total')
        axes[1,0].set_xlabel('X [kpc]')
        axes[1,0].set_ylabel('Y [kpc]')
        axes[1,0].set_xlim(0, box_size)
        axes[1,0].set_ylim(0, box_size)
    else:
        axes[1,0].text(0.5, 0.5, 'No Star Particles', ha='center', va='center', 
                      transform=axes[1,0].transAxes, fontsize=20)
        axes[1,0].set_title('Star Particles')
        axes[1,0].set_xlim(0, box_size)
        axes[1,0].set_ylim(0, box_size)
    
    # Combined density map (gas + dark matter)
    total_density = np.zeros((512, 512))
    if 'PartType0' in particle_data:
        gas_density, _, _ = create_density_map(particle_data['PartType0']['coords'], 
                                             particle_data['PartType0']['masses'], box_size)
        total_density += gas_density
    if 'PartType1' in particle_data:
        dm_density, _, _ = create_density_map(particle_data['PartType1']['coords'], 
                                            particle_data['PartType1']['masses'], box_size)
        total_density += dm_density
    
    im3 = axes[1,1].imshow(np.log10(total_density + 1e-10), 
                          extent=[0, box_size, 0, box_size],
                          origin='lower', cmap='magma', aspect='equal')
    axes[1,1].set_title('Total Matter Density')
    axes[1,1].set_xlabel('X [kpc]')
    axes[1,1].set_ylabel('Y [kpc]')
    plt.colorbar(im3, ax=axes[1,1], label='log₁₀(Σ) [M☉/kpc²]')
    
    # Add stars as overlay if they exist
    if 'PartType4' in particle_data:
        star_coords = particle_data['PartType4']['coords']
        if len(star_coords) > 5000:
            indices = np.random.choice(len(star_coords), 5000, replace=False)
            star_coords_plot = star_coords[indices]
        else:
            star_coords_plot = star_coords
        axes[1,1].scatter(star_coords_plot[:, 0], star_coords_plot[:, 1], 
                         s=0.5, c='yellow', alpha=0.8, marker='*')
    
    # Overall title
    num_stars = len(particle_data.get('PartType4', {}).get('coords', []))
    fig.suptitle(f'Snapshot {snap_num:03d} - z={redshift:.2f} - t={time:.3f} - Stars: {num_stars:,}', 
                fontsize=20)
    
    plt.tight_layout()
    
    # Save the figure
    frame_path = os.path.join(output_folder, f"density_frame_{frame:03d}.png")
    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return f"Saved density frame {frame:03d} for snapshot {snap_num}"

def main():
    # Check for snapshot directories first (multi-file format)
    snapshot_dirs = sorted(glob.glob("/home/cygnus/gadget4/output/snapdir_*"))
    
    # Check for single-file snapshots (both compressed and uncompressed)
    single_snapshots_hdf5 = sorted(glob.glob("/home/cygnus/gadget4/output/snapshot_*.hdf5"))
    single_snapshots_gz = sorted(glob.glob("/home/cygnus/gadget4/output/snapshot_*.hdf5.gz"))
    
    # Create output folder for frames
    output_folder = "/home/cygnus/gadget4/scripts/density_frames"
    os.makedirs(output_folder, exist_ok=True)
    
    # Use fewer cores for density maps (more memory intensive)
    num_cores = min(8, cpu_count())
    print(f"Using {num_cores} cores for parallel processing")
    
    if snapshot_dirs:
        print(f"Found {len(snapshot_dirs)} multi-file snapshot directories")
        use_multifile = True
        
        # Process only every few snapshots to save time (optional)
        step = max(1, len(snapshot_dirs) // 50)  # Max 50 frames
        selected_dirs = snapshot_dirs[::step]
        
        args_list = [(frame, snapdir, output_folder) 
                     for frame, snapdir in enumerate(selected_dirs)]
        
        with Pool(num_cores) as pool:
            results = pool.map(process_multifile_snapshot_density, args_list)
        
        print(f"Finished generating {len(selected_dirs)} density map frames!")
        
    elif single_snapshots_hdf5 or single_snapshots_gz:
        print("Using single-file snapshot format")
        
        # Prefer uncompressed files if available
        if single_snapshots_hdf5:
            single_snapshots = single_snapshots_hdf5
            print(f"Found {len(single_snapshots)} uncompressed single-file snapshots")
        else:
            single_snapshots = single_snapshots_gz
            print(f"Found {len(single_snapshots)} compressed single-file snapshots")
        
        use_multifile = False
        
        # Process only every few snapshots to save time (optional)
        step = max(1, len(single_snapshots) // 50)  # Max 50 frames
        selected_files = single_snapshots[::step]
        
        args_list = [(frame, filename, output_folder) 
                     for frame, filename in enumerate(selected_files)]
        
        with Pool(num_cores) as pool:
            results = pool.map(process_single_file_snapshot_density, args_list)
        
        print(f"Finished generating {len(selected_files)} density map frames!")
        
    else:
        print("ERROR: No snapshot files found!")
        print("Checked for:")
        print("  - Multi-file: /home/cygnus/gadget4/output/snapdir_*")
        print("  - Single HDF5: /home/cygnus/gadget4/output/snapshot_*.hdf5")
        print("  - Single GZ: /home/cygnus/gadget4/output/snapshot_*.hdf5.gz")
        return
    
    for result in results:
        print(result)
    
    print(f"Create animation with: ffmpeg -r 10 -i {output_folder}/density_frame_%03d.png -vcodec libx264 -pix_fmt yuv420p density_animation.mp4")

if __name__ == "__main__":
    main()