#!/usr/bin/env python3
"""
Visualize all snapshots from a simulation with dark matter, gas, and stars
Creates 2x3 panel plots showing surface density projections
Top row: full box view
Bottom row: zoomed to central 10,000 kpc
Handles both single-file and multi-file (snapdir) snapshots
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import h5py
import glob
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def find_snapshot_files(snapshot_path):
    """
    Find all files belonging to a snapshot.
    Handles both single files and multi-file snapshots in snapdir directories.
    
    Returns: list of HDF5 files to read
    """
    # Check if it's a directory (snapdir format)
    if os.path.isdir(snapshot_path):
        # Look for all .hdf5 files in the directory
        files = sorted(glob.glob(os.path.join(snapshot_path, '*.hdf5')))
        if len(files) == 0:
            files = sorted(glob.glob(os.path.join(snapshot_path, '*.HDF5')))
        return files
    
    # Check if it's a single file
    elif os.path.isfile(snapshot_path):
        return [snapshot_path]
    
    # Check if it's a base name for multi-file snapshot (e.g., snapshot_000.0.hdf5, .1.hdf5, etc.)
    else:
        # Try to find files matching pattern snapshot_XXX.N.hdf5
        base = snapshot_path.replace('.hdf5', '').replace('.HDF5', '')
        files = sorted(glob.glob(f"{base}.*.hdf5"))
        if len(files) == 0:
            files = sorted(glob.glob(f"{base}.*.HDF5"))
        if len(files) > 0:
            return files
        
        # If nothing found, return empty list
        return []

def read_snapshot_data(snapshot_path):
    """
    Read particle data from a snapshot (single or multi-file)
    """
    files = find_snapshot_files(snapshot_path)
    
    if len(files) == 0:
        raise ValueError(f"No snapshot files found at: {snapshot_path}")
    
    print(f"  Reading {len(files)} file(s)...")
    
    data = {}
    
    # Read header from first file
    with h5py.File(files[0], 'r') as f:
        header = f['Header'].attrs
        data['redshift'] = header['Redshift']
        data['time'] = header['Time']
        data['boxsize'] = header['BoxSize']
        if hasattr(data['boxsize'], '__len__'):
            data['boxsize'] = data['boxsize'][0]
        
        # Detect units
        if data['boxsize'] > 1000:
            unit_label = 'kpc/h'
        else:
            unit_label = 'Mpc/h'
        
        data['unit_label'] = unit_label
    
    # Initialize particle arrays
    gas_pos_list = []
    gas_mass_list = []
    dm_pos_list = []
    dm_mass_list = []
    star_pos_list = []
    star_mass_list = []
    star_age_list = []
    
    # Read data from all files
    for file_path in files:
        with h5py.File(file_path, 'r') as f:
            header = f['Header'].attrs
            mass_table = header.get('MassTable', [0,0,0,0,0,0])
            current_time = header.get('Time', 1.0)
            
            # Read gas (PartType0)
            if 'PartType0' in f:
                gas_group = f['PartType0']
                pos = gas_group['Coordinates'][:]
                
                if 'Masses' in gas_group:
                    mass = gas_group['Masses'][:]
                elif mass_table[0] > 0:
                    mass = np.ones(len(pos)) * mass_table[0]
                else:
                    mass = np.ones(len(pos)) * 1e-5
                
                gas_pos_list.append(pos)
                gas_mass_list.append(mass)
            
            # Read high-res DM (PartType1)
            if 'PartType1' in f:
                dm_group = f['PartType1']
                pos = dm_group['Coordinates'][:]
                
                if 'Masses' in dm_group:
                    mass = dm_group['Masses'][:]
                elif mass_table[1] > 0:
                    mass = np.ones(len(pos)) * mass_table[1]
                else:
                    mass = np.ones(len(pos)) * 1e-4
                
                dm_pos_list.append(pos)
                dm_mass_list.append(mass)
            
            # Read low-res DM (PartType2, 5) and add to DM
            for ptype in [2, 5]:
                if f'PartType{ptype}' in f:
                    extra_dm = f[f'PartType{ptype}']
                    pos = extra_dm['Coordinates'][:]
                    
                    if 'Masses' in extra_dm:
                        mass = extra_dm['Masses'][:]
                    elif mass_table[ptype] > 0:
                        mass = np.ones(len(pos)) * mass_table[ptype]
                    else:
                        mass = np.ones(len(pos)) * 1e-3
                    
                    dm_pos_list.append(pos)
                    dm_mass_list.append(mass)
            
            # Read stars (PartType4)
            if 'PartType4' in f:
                star_group = f['PartType4']
                pos = star_group['Coordinates'][:]
                
                if 'Masses' in star_group:
                    mass = star_group['Masses'][:]
                elif mass_table[4] > 0:
                    mass = np.ones(len(pos)) * mass_table[4]
                else:
                    mass = np.ones(len(pos)) * 1e-5
                
                # Read stellar ages if available
                if 'StellarFormationTime' in star_group:
                    birth_a = star_group['StellarFormationTime'][:]
                    # Convert to age in Myr (approximate)
                    age = (current_time - birth_a) * 13700.0  # Rough Hubble time
                else:
                    age = np.zeros(len(pos))
                
                star_pos_list.append(pos)
                star_mass_list.append(mass)
                star_age_list.append(age)
    
    # Combine all particles
    if len(gas_pos_list) > 0:
        data['gas_pos'] = np.vstack(gas_pos_list)
        data['gas_mass'] = np.hstack(gas_mass_list)
        print(f"  Found {len(data['gas_pos']):,} total gas particles")
    else:
        data['gas_pos'] = np.array([])
        data['gas_mass'] = np.array([])
        print("  No gas particles found")
    
    if len(dm_pos_list) > 0:
        data['dm_pos'] = np.vstack(dm_pos_list)
        data['dm_mass'] = np.hstack(dm_mass_list)
        print(f"  Found {len(data['dm_pos']):,} total DM particles")
    else:
        data['dm_pos'] = np.array([])
        data['dm_mass'] = np.array([])
        print("  No DM particles found")
    
    if len(star_pos_list) > 0:
        data['star_pos'] = np.vstack(star_pos_list)
        data['star_mass'] = np.hstack(star_mass_list)
        data['star_age'] = np.hstack(star_age_list)
        age_range = f"[{data['star_age'].min():.1f}, {data['star_age'].max():.1f}] Myr"
        print(f"  Found {len(data['star_pos']):,} total star particles (age {age_range})")
    else:
        data['star_pos'] = np.array([])
        data['star_mass'] = np.array([])
        data['star_age'] = np.array([])
        print("  No star particles found")
    
    return data

def create_surface_density_map(positions, masses, boxsize, nbins=256, depth_fraction=0.1, 
                               zoom_region=None):
    """
    Create a surface density map by projecting particles
    
    Parameters:
    -----------
    positions : array
        Particle positions
    masses : array
        Particle masses
    boxsize : float
        Size of the simulation box
    nbins : int
        Number of bins for the 2D histogram
    depth_fraction : float
        Fraction of box depth to project
    zoom_region : tuple or None
        If provided, (xmin, xmax, ymin, ymax) defines the zoomed region
    """
    
    if len(positions) == 0:
        return np.zeros((nbins, nbins))
    
    # Set up coordinate ranges
    if zoom_region is None:
        # Full box view
        xmin, xmax = 0, boxsize
        ymin, ymax = 0, boxsize
        center_z = boxsize / 2.0
        depth = boxsize * depth_fraction
    else:
        # Zoomed view
        xmin, xmax, ymin, ymax = zoom_region
        center_z = boxsize / 2.0
        # Keep same depth fraction relative to zoom size
        zoom_size = xmax - xmin
        depth = zoom_size * depth_fraction
    
    # Select particles in the depth slice
    mask = np.abs(positions[:, 2] - center_z) < depth/2
    pos_slice = positions[mask]
    mass_slice = masses[mask]
    
    if len(pos_slice) == 0:
        return np.zeros((nbins, nbins))
    
    # Select particles in the xy region
    mask_xy = ((pos_slice[:, 0] >= xmin) & (pos_slice[:, 0] <= xmax) &
               (pos_slice[:, 1] >= ymin) & (pos_slice[:, 1] <= ymax))
    pos_region = pos_slice[mask_xy]
    mass_region = mass_slice[mask_xy]
    
    if len(pos_region) == 0:
        return np.zeros((nbins, nbins))
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        pos_region[:, 0], pos_region[:, 1],
        bins=nbins,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=mass_region
    )
    
    # Convert to surface density (mass per area)
    pixel_area = ((xmax - xmin) / nbins) * ((ymax - ymin) / nbins)
    surface_density = H.T / pixel_area
    
    # Apply gaussian smoothing
    surface_density = gaussian_filter(surface_density, sigma=0.5)
    
    return surface_density

def plot_snapshot(data, output_file, nbins=256, zoom_size_kpc=10000):
    """
    Create a 2x3 panel plot showing gas, DM, and stars
    Top row: full box view
    Bottom row: zoomed to central region
    """
    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Determine zoom region (centered on box center)
    boxsize = data['boxsize']
    center = boxsize / 2.0
    half_zoom = zoom_size_kpc / 2.0
    zoom_region = (center - half_zoom, center + half_zoom, 
                   center - half_zoom, center + half_zoom)
    
    # Check if we have any stars
    has_stars = len(data['star_pos']) > 0
    
    # Create surface density maps
    # Full box views
    gas_map_full = create_surface_density_map(
        data['gas_pos'], data['gas_mass'], data['boxsize'], nbins=nbins
    )
    dm_map_full = create_surface_density_map(
        data['dm_pos'], data['dm_mass'], data['boxsize'], nbins=nbins
    )
    star_map_full = create_surface_density_map(
        data['star_pos'], data['star_mass'], data['boxsize'], nbins=nbins
    )
    
    # Zoomed views
    gas_map_zoom = create_surface_density_map(
        data['gas_pos'], data['gas_mass'], data['boxsize'], nbins=nbins,
        zoom_region=zoom_region
    )
    dm_map_zoom = create_surface_density_map(
        data['dm_pos'], data['dm_mass'], data['boxsize'], nbins=nbins,
        zoom_region=zoom_region
    )
    star_map_zoom = create_surface_density_map(
        data['star_pos'], data['star_mass'], data['boxsize'], nbins=nbins,
        zoom_region=zoom_region
    )
    
    # Color schemes
    cmap_gas = 'viridis'
    cmap_dm = 'inferno'
    cmap_star = 'magma'
    
    vmin_percentile = 1
    vmax_percentile = 99
    
    # ROW 1: FULL BOX VIEWS
    # Panel 1: Gas (Full)
    ax1 = fig.add_subplot(gs[0, 0])
    if np.sum(gas_map_full) > 0:
        nonzero_gas = gas_map_full[gas_map_full > 0]
        if len(nonzero_gas) > 0:
            vmin = np.percentile(nonzero_gas, vmin_percentile)
            vmax = np.percentile(nonzero_gas, vmax_percentile)
        else:
            vmin, vmax = 1e-10, 1e-5
        
        im1 = ax1.imshow(
            gas_map_full,
            origin='lower',
            extent=[0, data['boxsize'], 0, data['boxsize']],
            cmap=cmap_gas,
            norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
            interpolation='nearest'
        )
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, fraction=0.046)
    else:
        ax1.text(0.5, 0.5, 'No Gas', transform=ax1.transAxes,
                ha='center', va='center', fontsize=20, color='gray')
        ax1.set_facecolor('black')
    
    ax1.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax1.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax1.set_title('Gas Surface Density (Full Box)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    
    # Panel 2: Dark Matter (Full)
    ax2 = fig.add_subplot(gs[0, 1])
    if np.sum(dm_map_full) > 0:
        nonzero_dm = dm_map_full[dm_map_full > 0]
        if len(nonzero_dm) > 0:
            vmin = np.percentile(nonzero_dm, vmin_percentile)
            vmax = np.percentile(nonzero_dm, vmax_percentile)
        else:
            vmin, vmax = 1e-10, 1e-5
        
        im2 = ax2.imshow(
            dm_map_full,
            origin='lower',
            extent=[0, data['boxsize'], 0, data['boxsize']],
            cmap=cmap_dm,
            norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
            interpolation='nearest'
        )
        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, fraction=0.046)
    else:
        ax2.text(0.5, 0.5, 'No Dark Matter', transform=ax2.transAxes,
                ha='center', va='center', fontsize=20, color='gray')
        ax2.set_facecolor('black')
    
    ax2.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax2.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax2.set_title('Dark Matter Surface Density (Full Box)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    
    # Panel 3: Stars (Full)
    ax3 = fig.add_subplot(gs[0, 2])
    nonzero_star = star_map_full[star_map_full > 0]
    if len(nonzero_star) > 0:
        vmin = np.percentile(nonzero_star, vmin_percentile)
        vmax = np.percentile(nonzero_star, vmax_percentile)
    else:
        vmin, vmax = 1e-10, 1e-5
    
    im3 = ax3.imshow(
        star_map_full,
        origin='lower',
        extent=[0, data['boxsize'], 0, data['boxsize']],
        cmap=cmap_star,
        norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
        interpolation='nearest'
    )
    cbar3 = plt.colorbar(im3, ax=ax3, pad=0.02, fraction=0.046)
    
    if np.sum(star_map_full) > 0:
        n_stars = len(data['star_pos'])
        age_min, age_max = data['star_age'].min(), data['star_age'].max()
        info_text = f"N={n_stars:,}\nAge: [{age_min:.1f}, {age_max:.1f}] Myr"
        ax3.text(0.02, 0.98, info_text,
                transform=ax3.transAxes, ha='left', va='top',
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))
    
    ax3.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax3.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax3.set_title('Stellar Surface Density (Full Box)', fontsize=13, fontweight='bold')
    ax3.set_aspect('equal')
    
    # ROW 2: ZOOMED VIEWS
    xmin, xmax, ymin, ymax = zoom_region
    
    # Panel 4: Gas (Zoomed)
    ax4 = fig.add_subplot(gs[1, 0])
    if np.sum(gas_map_zoom) > 0:
        nonzero_gas = gas_map_zoom[gas_map_zoom > 0]
        if len(nonzero_gas) > 0:
            vmin = np.percentile(nonzero_gas, vmin_percentile)
            vmax = np.percentile(nonzero_gas, vmax_percentile)
        else:
            vmin, vmax = 1e-10, 1e-5
        
        im4 = ax4.imshow(
            gas_map_zoom,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap_gas,
            norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
            interpolation='nearest'
        )
        cbar4 = plt.colorbar(im4, ax=ax4, pad=0.02, fraction=0.046)
    else:
        ax4.text(0.5, 0.5, 'No Gas', transform=ax4.transAxes,
                ha='center', va='center', fontsize=20, color='gray')
        ax4.set_facecolor('black')
    
    ax4.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax4.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax4.set_title(f'Gas (Central {zoom_size_kpc:.0f} {data["unit_label"]})', 
                  fontsize=13, fontweight='bold')
    ax4.set_aspect('equal')
    
    # Panel 5: Dark Matter (Zoomed)
    ax5 = fig.add_subplot(gs[1, 1])
    if np.sum(dm_map_zoom) > 0:
        nonzero_dm = dm_map_zoom[dm_map_zoom > 0]
        if len(nonzero_dm) > 0:
            vmin = np.percentile(nonzero_dm, vmin_percentile)
            vmax = np.percentile(nonzero_dm, vmax_percentile)
        else:
            vmin, vmax = 1e-10, 1e-5
        
        im5 = ax5.imshow(
            dm_map_zoom,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap_dm,
            norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
            interpolation='nearest'
        )
        cbar5 = plt.colorbar(im5, ax=ax5, pad=0.02, fraction=0.046)
    else:
        ax5.text(0.5, 0.5, 'No Dark Matter', transform=ax5.transAxes,
                ha='center', va='center', fontsize=20, color='gray')
        ax5.set_facecolor('black')
    
    ax5.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax5.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax5.set_title(f'Dark Matter (Central {zoom_size_kpc:.0f} {data["unit_label"]})', 
                  fontsize=13, fontweight='bold')
    ax5.set_aspect('equal')
    
    # Panel 6: Stars (Zoomed)
    ax6 = fig.add_subplot(gs[1, 2])
    nonzero_star = star_map_zoom[star_map_zoom > 0]
    if len(nonzero_star) > 0:
        vmin = np.percentile(nonzero_star, vmin_percentile)
        vmax = np.percentile(nonzero_star, vmax_percentile)
    else:
        vmin, vmax = 1e-10, 1e-5
    
    im6 = ax6.imshow(
        star_map_zoom,
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap_star,
        norm=colors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
        interpolation='nearest'
    )
    cbar6 = plt.colorbar(im6, ax=ax6, pad=0.02, fraction=0.046)
    
    if np.sum(star_map_zoom) > 0 and has_stars:
        # Count stars in zoom region
        mask_zoom = ((data['star_pos'][:, 0] >= xmin) & 
                    (data['star_pos'][:, 0] <= xmax) &
                    (data['star_pos'][:, 1] >= ymin) & 
                    (data['star_pos'][:, 1] <= ymax))
        n_stars_zoom = np.sum(mask_zoom)
        info_text = f"N={n_stars_zoom:,} in zoom"
        ax6.text(0.02, 0.98, info_text,
                transform=ax6.transAxes, ha='left', va='top',
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))
    
    ax6.set_xlabel(f'x [{data["unit_label"]}]', fontsize=11)
    ax6.set_ylabel(f'y [{data["unit_label"]}]', fontsize=11)
    ax6.set_title(f'Stars (Central {zoom_size_kpc:.0f} {data["unit_label"]})', 
                  fontsize=13, fontweight='bold')
    ax6.set_aspect('equal')
    
    # Add main title with redshift info
    title_text = f'Snapshot at z = {data["redshift"]:.2f} (t = {data["time"]:.2f})'
    if has_stars:
        n_stars = len(data['star_pos'])
        title_text += f'   —   {n_stars:,} stars'
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_movie_command(image_dir, output_movie='simulation.mp4', fps=4):
    """Generate ffmpeg command to create a movie from images"""
    cmd = (
        f"ffmpeg -framerate {fps} -pattern_type glob -i '{image_dir}/*.png' "
        f"-c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
        f"{output_movie}"
    )
    print(f"\nTo create a movie, run:")
    print(f"  {cmd}")

def find_all_snapshots(snapshot_dir):
    """
    Find all snapshots in a directory.
    Handles both single files and snapdir directories.
    """
    snapshots = []
    
    # Look for snapdir directories
    snapdirs = sorted(glob.glob(os.path.join(snapshot_dir, 'snapdir_*')))
    for snapdir in snapdirs:
        snapshots.append(snapdir)
    
    # If no snapdirs found, look for individual snapshot files
    if len(snapshots) == 0:
        snap_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_*.hdf5')))
        if len(snap_files) == 0:
            snap_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snap_*.hdf5')))
        
        # Group multi-file snapshots
        snap_dict = {}
        for f in snap_files:
            # Extract snapshot number
            basename = os.path.basename(f)
            # Handle snapshot_000.0.hdf5 format
            parts = basename.replace('.hdf5', '').split('.')
            if len(parts) > 1:
                snap_num = parts[0]  # snapshot_000
            else:
                snap_num = basename.replace('.hdf5', '')
            
            if snap_num not in snap_dict:
                snap_dict[snap_num] = []
            snap_dict[snap_num].append(f)
        
        # Use first file from each snapshot group
        snapshots = [files[0] for files in sorted(snap_dict.values())]
    
    return snapshots

def main():
    parser = argparse.ArgumentParser(description='Visualize simulation snapshots (gas, DM, stars) with full box and zoomed views')
    parser.add_argument('snapshot_dir', help='Directory containing snapshot files or snapdir directories')
    parser.add_argument('--output-dir', default='surface_densities', 
                       help='Output directory for images')
    parser.add_argument('--nbins', type=int, default=256,
                       help='Number of bins for surface density map')
    parser.add_argument('--depth-fraction', type=float, default=0.1,
                       help='Fraction of box depth to project (0.1 = 10%%)')
    parser.add_argument('--zoom-size', type=float, default=10000,
                       help='Size of zoomed region in kpc (default: 10000)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth snapshot')
    
    args = parser.parse_args()
    
    # Find all snapshots
    snapshots = find_all_snapshots(args.snapshot_dir)
    
    if len(snapshots) == 0:
        print(f"No snapshots found in: {args.snapshot_dir}")
        return 1
    
    print(f"Found {len(snapshots)} snapshot(s)")
    
    # Create output directory
    Path(args.snapshot_dir+'/'+args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process snapshots
    if args.skip > 1:
        snapshots = snapshots[::args.skip]
        print(f"Processing every {args.skip} snapshots ({len(snapshots)} total)")
    
    for i, snap_path in enumerate(tqdm(snapshots, desc="Processing snapshots")):
        # Get snapshot number
        if os.path.isdir(snap_path):
            snap_num = os.path.basename(snap_path).replace('snapdir_', '')
        else:
            basename = os.path.basename(snap_path)
            snap_num = basename.replace('.hdf5', '').replace('.HDF5', '')
            snap_num = snap_num.split('.')[0]  # Handle snapshot_000.0.hdf5
            snap_num = snap_num.replace('snapshot_', '').replace('snap_', '')
        
        print(f"\n[{i+1}/{len(snapshots)}] Processing: {os.path.basename(snap_path)}")
        
        try:
            # Read data
            data = read_snapshot_data(snap_path)
            
            final_output_dir = args.snapshot_dir+'/'+args.output_dir
            # Create output filename
            output_file = os.path.join(final_output_dir, f"surface_density_{snap_num}.png")
            
            # Create plot
            plot_snapshot(data, output_file, nbins=args.nbins, zoom_size_kpc=args.zoom_size)
            
        except Exception as e:
            print(f"  ERROR processing {snap_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Visualization complete!")
    print(f"Images saved to: {final_output_dir}")
    
    # Suggest movie creation
    create_movie_command(args.output_dir)

if __name__ == "__main__":
    main()
