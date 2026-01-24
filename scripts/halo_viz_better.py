#!/usr/bin/env python3
"""
Better halo visualization showing actual halos as points, not particle soup
Usage: python halo_viz_better.py output_parent/
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import sys
import os

def read_fof_catalog_hdf5(filename):
    """Read FOF catalog to get halo information"""
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Available groups in {filename}: {list(f.keys())}")
            
            # Try different possible locations for halo data
            data = {}
            
            # Try Group first (most common)
            if 'Group' in f:
                grp = f['Group']
                print(f"Group datasets: {list(grp.keys())}")
                
                if 'GroupMass' in grp and len(grp['GroupMass']) > 0:
                    data['mass'] = grp['GroupMass'][:]
                    data['pos'] = grp['GroupPos'][:]
                    data['npart'] = grp['GroupLen'][:]
                    return data
            
            # Try other possible names
            for group_name in ['FOF', 'Halo']:
                if group_name in f:
                    grp = f[group_name]
                    if 'GroupMass' in grp or 'Mass' in grp:
                        data['mass'] = grp['GroupMass'][:] if 'GroupMass' in grp else grp['Mass'][:]
                        data['pos'] = grp['GroupPos'][:] if 'GroupPos' in grp else grp['CenterOfMass'][:]
                        data['npart'] = grp['GroupLen'][:] if 'GroupLen' in grp else grp['NumPart'][:]
                        return data
            
            # Try root level
            if 'GroupMass' in f:
                data['mass'] = f['GroupMass'][:]
                data['pos'] = f['GroupPos'][:]
                data['npart'] = f['GroupLen'][:]
                return data
            
            print("No halo data found in expected locations")
            return None
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def get_redshift_from_filename(filename):
    """Extract approximate redshift from filename"""
    snap_num = int(filename.split('_')[-1].split('.')[0])
    
    # Rough mapping - adjust based on your simulation
    redshift_map = {
        33: 15, 34: 12, 35: 10, 36: 8, 37: 6, 38: 5, 39: 4, 
        40: 3.5, 41: 3, 42: 2.5, 43: 2, 44: 1.5, 45: 1.2, 
        46: 1.0, 47: 0.7, 48: 0.5
    }
    
    return redshift_map.get(snap_num, 0)

def create_halo_overview_plot(output_dir, min_mass_fraction=0.001, box_size=50000):
    """
    Create overview plot showing all significant halos across time
    
    Parameters:
    - min_mass_fraction: show halos above this fraction of most massive halo
    - box_size: simulation box size in kpc
    """
    
    # Find all FOF files
    fof_files = glob.glob(f"{output_dir}/fof_subhalo_tab_*.hdf5")
    if not fof_files:
        print("No FOF files found!")
        return
    
    print(f"Found {len(fof_files)} FOF catalog files")
    
    # Read the latest (most evolved) catalog
    latest_file = sorted(fof_files)[-1]
    print(f"Using latest catalog: {latest_file}")
    
    data = read_fof_catalog_hdf5(latest_file)
    if data is None:
        print("Could not read halo data!")
        return
    
    masses = data['mass']
    positions = data['pos']
    npart = data['npart']
    redshift = get_redshift_from_filename(latest_file)
    
    print(f"Found {len(masses)} halos at z≈{redshift}")
    print(f"Mass range: {masses.min():.2e} - {masses.max():.2e} (code units)")
    print(f"Particle range: {npart.min()} - {npart.max()}")
    
    # Convert to physical units
    UnitMass = 1e10  # Msun/h
    h = 0.6774
    mass_msun = masses * UnitMass / h
    
    # Filter for significant halos
    max_mass = mass_msun.max()
    significant_mask = mass_msun > (min_mass_fraction * max_mass)
    significant_masses = mass_msun[significant_mask]
    significant_pos = positions[significant_mask]
    significant_npart = npart[significant_mask]
    
    print(f"Showing {len(significant_masses)} halos above {min_mass_fraction*100:.1f}% of max mass")
    print(f"Mass threshold: {min_mass_fraction * max_mass:.2e} Msun")
    
    # Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    projections = [
        ('XY', [0, 1], 'X', 'Y'),
        ('XZ', [0, 2], 'X', 'Z'), 
        ('YZ', [1, 2], 'Y', 'Z')
    ]
    
    for i, (name, dims, xlabel, ylabel) in enumerate(projections):
        ax = axes[i]
        
        # Sort halos by mass to plot smallest first, largest last (so big ones are on top)
        mass_order = np.argsort(significant_masses)
        x_data = significant_pos[mass_order, dims[0]]
        y_data = significant_pos[mass_order, dims[1]]
        ordered_masses = significant_masses[mass_order]
        
        # Size and color based on mass
        sizes = 20 + 200 * (np.log10(ordered_masses) - np.log10(ordered_masses.min())) / (np.log10(ordered_masses.max()) - np.log10(ordered_masses.min()))
        colors = np.log10(ordered_masses)
        
        scatter = ax.scatter(x_data, y_data, s=sizes, c=colors, 
                           cmap='plasma', alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Add colorbar for first panel
        if i == 2:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('log₁₀(Mass/M☉)')
        
        # Highlight most massive halos (top 5)
        most_massive_indices = np.argsort(significant_masses)[-5:]  # Top 5 indices in original array
        top_x = significant_pos[most_massive_indices, dims[0]]
        top_y = significant_pos[most_massive_indices, dims[1]]
        top_masses = significant_masses[most_massive_indices]
        
        # Sort top 5 by mass for consistent labeling
        top_5_order = np.argsort(top_masses)[::-1]  # Descending order
        top_x = top_x[top_5_order]
        top_y = top_y[top_5_order]
        top_masses = top_masses[top_5_order]
        
        # Mark top halos with red circles (bigger than the scatter points)
        ax.scatter(top_x, top_y, s=400, facecolors='none', 
                  edgecolors='red', linewidth=3, label='Top 5 Most Massive', zorder=10)
        
        # Label ALL top 5 halos
        colors_for_labels = ['yellow', 'orange', 'lightgreen', 'lightblue', 'pink']
        for j, (x, y, mass) in enumerate(zip(top_x, top_y, top_masses)):
            ax.annotate(f'#{j+1}: {mass:.1e} M☉', 
                       (x, y), 
                       xytext=(15, 15 + j*20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_for_labels[j], 
                                alpha=0.8, edgecolor='black'),
                       fontsize=9, ha='left', weight='bold',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1),
                       zorder=11)
        
        # Set labels and title
        ax.set_xlabel(f'{xlabel} (kpc)')
        ax.set_ylabel(f'{ylabel} (kpc)')
        ax.set_title(f'{name} Projection (z≈{redshift:.1f})')
        ax.set_aspect('equal')
        
        # Set plot limits
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for first panel
        if i == 2:
            ax.legend(loc='upper right')
    
    plt.suptitle(f'Halo Distribution at z≈{redshift:.1f} ({len(significant_masses)} halos shown)', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    output_file = f'halo_overview_z{redshift:.1f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overview plot as: {output_file}")
    
    plt.show()
    
    # Print summary of top candidates
    print(f"\n=== TOP 10 ZOOM CANDIDATES ===")
    sorted_indices = np.argsort(significant_masses)[::-1]  # Sort by mass, descending
    
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        mass = significant_masses[idx]
        pos = significant_pos[idx]
        particles = significant_npart[idx]
        
        # Check isolation (distance from box edges)
        edge_dist = min(pos.min(), (box_size - pos).min())
        
        print(f"{i+1:2d}. Mass: {mass:.2e} M☉  |  Pos: ({pos[0]:5.0f},{pos[1]:5.0f},{pos[2]:5.0f})  |  "
              f"Parts: {particles:5d}  |  Edge: {edge_dist:4.0f} kpc")
    
    return significant_masses, significant_pos, significant_npart

def create_zoom_region_plot(output_dir, target_pos, zoom_size=10000, plot_size=20000):
    """Create detailed plot around a specific target for zoom planning"""
    
    # Read latest FOF catalog
    fof_files = glob.glob(f"{output_dir}/fof_subhalo_tab_*.hdf5")
    latest_file = sorted(fof_files)[-1]
    
    data = read_fof_catalog_hdf5(latest_file)
    if data is None:
        return
    
    masses = data['mass'] * 1e10 / 0.6774  # Convert to Msun
    positions = data['pos']
    
    # Filter halos within plot region
    target_pos = np.array(target_pos)
    distances = np.linalg.norm(positions - target_pos, axis=1)
    nearby_mask = distances < plot_size
    
    nearby_masses = masses[nearby_mask]
    nearby_pos = positions[nearby_mask]
    
    print(f"Found {len(nearby_masses)} halos within {plot_size} kpc of target")
    
    # Create 3-panel zoom plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    projections = [
        ('XY', [0, 1], 'X', 'Y'),
        ('XZ', [0, 2], 'X', 'Z'), 
        ('YZ', [1, 2], 'Y', 'Z')
    ]
    
    for i, (name, dims, xlabel, ylabel) in enumerate(projections):
        ax = axes[i]
        
        if len(nearby_pos) > 0:
            x_data = nearby_pos[:, dims[0]]
            y_data = nearby_pos[:, dims[1]]
            
            # Size and color based on mass
            sizes = 30 + 300 * np.log10(nearby_masses / nearby_masses.min()) / np.log10(nearby_masses.max() / nearby_masses.min())
            colors = np.log10(nearby_masses)
            
            scatter = ax.scatter(x_data, y_data, s=sizes, c=colors, 
                               cmap='plasma', alpha=0.8, edgecolors='white', linewidth=1)
            
            if i == 2:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('log₁₀(Mass/M☉)')
        
        # Mark target center
        ax.plot(target_pos[dims[0]], target_pos[dims[1]], 'r*', markersize=20, 
               label='Zoom Target', markeredgecolor='white', markeredgewidth=1)
        
        # Draw zoom region
        zoom_box = plt.Rectangle((target_pos[dims[0]] - zoom_size/2, target_pos[dims[1]] - zoom_size/2),
                                zoom_size, zoom_size, fill=False, edgecolor='red', linewidth=3,
                                label=f'Zoom Region ({zoom_size/1000:.0f} Mpc)')
        ax.add_patch(zoom_box)
        
        # Set labels and limits
        ax.set_xlabel(f'{xlabel} (kpc)')
        ax.set_ylabel(f'{ylabel} (kpc)')
        ax.set_title(f'{name} Projection - Zoom Planning')
        ax.set_aspect('equal')
        
        ax.set_xlim(target_pos[dims[0]] - plot_size, target_pos[dims[0]] + plot_size)
        ax.set_ylim(target_pos[dims[1]] - plot_size, target_pos[dims[1]] + plot_size)
        
        ax.grid(True, alpha=0.3)
        
        if i == 2:
            ax.legend(loc='upper right')
    
    plt.suptitle(f'Zoom Region Planning at ({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f}) kpc', 
                 fontsize=16)
    plt.tight_layout()
    
    output_file = f'zoom_planning_{target_pos[0]:.0f}_{target_pos[1]:.0f}_{target_pos[2]:.0f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved zoom planning plot as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  General overview: python halo_viz_better.py output_parent/")
        print("  Zoom planning:    python halo_viz_better.py output_parent/ 25000 18500 32000")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if len(sys.argv) == 2:
        # General overview
        create_halo_overview_plot(output_dir)
    elif len(sys.argv) == 5:
        # Zoom planning for specific target
        target_x = float(sys.argv[2])
        target_y = float(sys.argv[3])
        target_z = float(sys.argv[4])
        
        create_halo_overview_plot(output_dir)  # Show overview first
        create_zoom_region_plot(output_dir, [target_x, target_y, target_z])  # Then zoom view
    else:
        print("Wrong number of arguments!")
        sys.exit(1)