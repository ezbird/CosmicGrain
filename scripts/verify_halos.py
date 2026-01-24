#!/usr/bin/env python3
"""
Match FOF halos to visual overdensities in cosmological simulations.
Handles unit conversions and creates verification plots.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import argparse

def load_snapshot(snap_file):
    """Load particle positions from snapshot."""
    with h5py.File(snap_file, 'r') as f:
        # Get header info
        header = dict(f['Header'].attrs)
        box_size = header['BoxSize']
        
        # Load DM positions (usually PartType1)
        if 'PartType1' in f:
            pos = f['PartType1']['Coordinates'][:]
        else:
            print("Warning: PartType1 not found, trying PartType0")
            pos = f['PartType0']['Coordinates'][:]
            
        print(f"Loaded {len(pos):,} particles")
        print(f"Position range: [{pos.min():.2f}, {pos.max():.2f}]")
        print(f"Box size from header: {box_size}")
        
    return pos, box_size

def load_fof_catalog(fof_file):
    """Load FOF halo catalog."""
    with h5py.File(fof_file, 'r') as f:
        header = dict(f['Header'].attrs)
        
        # Get unit conversions
        h = header.get('HubbleParam', 0.6774)
        box_size = header['BoxSize']
        
        # Load halo data
        group = f['Group']
        pos = group['GroupPos'][:]
        mass = group['GroupMass'][:]
        npart = group['GroupLen'][:]
        
        # Handle mass units (often in 1e10 Msun/h)
        if mass.max() < 1e8:
            mass *= 1e10  # Convert to Msun/h
        
        # Try to get R200
        r200 = None
        if 'Group_R_Crit200' in group:
            r200 = group['Group_R_Crit200'][:]
        
        print(f"Loaded {len(mass):,} halos")
        print(f"FOF position range: [{pos.min():.2f}, {pos.max():.2f}]")
        print(f"Mass range: {mass.min():.2e} - {mass.max():.2e} Msun/h")
        
    return pos, mass, npart, r200, box_size, h

def find_halos_visually(particles, halo_pos, halo_mass, halo_npart, box_size, 
                       mass_min=5e11, mass_max=2e12, top_n=10):
    """Create visualizations to match halos with particle overdensities."""
    
    # Select halos in mass range
    mask = (halo_mass >= mass_min) & (halo_mass <= mass_max)
    if not np.any(mask):
        print(f"No halos in mass range [{mass_min:.2e}, {mass_max:.2e}]")
        return
    
    selected_pos = halo_pos[mask]
    selected_mass = halo_mass[mask]
    selected_npart = halo_npart[mask]
    
    # Sort by mass and take top N
    sort_idx = np.argsort(selected_mass)[::-1][:top_n]
    
    print(f"\nVisualizing top {len(sort_idx)} halos in mass range:")
    
    # Create figure with subplots for each halo
    n_halos = len(sort_idx)
    n_cols = 3
    n_rows = (n_halos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_halos > 1 else [axes]
    
    for idx, ax_idx in enumerate(sort_idx):
        ax = axes[idx]
        
        # Get halo properties
        h_pos = selected_pos[ax_idx]
        h_mass = selected_mass[ax_idx]
        h_npart = selected_npart[ax_idx]
        
        # Define zoom region (2 Mpc around halo)
        zoom_size = 2.0
        
        # Apply periodic boundary conditions for particles near edges
        rel_pos = particles - h_pos
        rel_pos = rel_pos - np.round(rel_pos / box_size) * box_size
        
        # Select particles in zoom region
        mask = (np.abs(rel_pos[:, 0]) < zoom_size) & \
               (np.abs(rel_pos[:, 1]) < zoom_size)
        
        local_particles = rel_pos[mask]
        
        # Plot X-Y projection
        ax.scatter(local_particles[:, 0], local_particles[:, 1], 
                  s=0.1, alpha=0.3, c='black', rasterized=True)
        
        # Mark halo center
        ax.scatter(0, 0, s=200, c='red', marker='+', linewidths=2)
        
        # Add circle for approximate R200 (rough estimate)
        r200_estimate = 0.2 * (h_mass / 1e12) ** (1/3)  # Very rough!
        circle = Circle((0, 0), r200_estimate, fill=False, 
                       edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        # Labels
        ax.set_xlim(-zoom_size, zoom_size)
        ax.set_ylim(-zoom_size, zoom_size)
        ax.set_aspect('equal')
        ax.set_title(f'M={h_mass:.2e} Msun/h\n{h_npart} particles\n' + 
                    f'pos=({h_pos[0]:.1f}, {h_pos[1]:.1f}, {h_pos[2]:.1f})',
                    fontsize=10)
        ax.set_xlabel('X offset [code units]')
        ax.set_ylabel('Y offset [code units]')
        
        print(f"  Halo {idx+1}: M={h_mass:.2e}, N={h_npart}, "
              f"pos=({h_pos[0]:.2f}, {h_pos[1]:.2f}, {h_pos[2]:.2f})")
    
    # Hide unused subplots
    for idx in range(len(sort_idx), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Top {n_halos} halos in mass range [{mass_min:.1e}, {mass_max:.1e}] Msun/h', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('halo_verification.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_density_map(particles, halo_pos, halo_mass, box_size):
    """Create 2D density map with halos marked."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    
    for ax, (label1, label2, i, j) in zip(axes, projections):
        # Create 2D histogram (density map)
        H, xedges, yedges = np.histogram2d(particles[:, i], particles[:, j], 
                                          bins=256, range=[[0, box_size], [0, box_size]])
        
        # Log scale for better contrast
        H = np.log10(H + 1)
        
        # Plot density
        im = ax.imshow(H.T, origin='lower', extent=[0, box_size, 0, box_size],
                      cmap='hot', interpolation='nearest')
        
        # Mark massive halos
        massive_mask = halo_mass > 1e12
        for h_pos, h_mass in zip(halo_pos[massive_mask], halo_mass[massive_mask]):
            size = 100 * (h_mass / 1e12) ** 0.5
            ax.scatter(h_pos[i], h_pos[j], s=size, c='cyan', 
                      marker='o', alpha=0.7, edgecolors='white', linewidths=1)
        
        ax.set_xlabel(f'{label1} [code units]')
        ax.set_ylabel(f'{label2} [code units]')
        ax.set_title(f'{label1}-{label2} projection')
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log10(particle count)', rotation=270, labelpad=15)
    
    plt.suptitle(f'Density projections (box={box_size:.1f} code units)\n' +
                 'Cyan circles = halos > 10¹² Msun/h (size ∝ √mass)', fontsize=12)
    plt.tight_layout()
    plt.savefig('density_projections.png', dpi=150, bbox_inches='tight')
    plt.show()

def verify_specific_halo(particles, halo_pos, halo_mass, halo_npart, 
                        box_size, target_mass, tolerance=0.1):
    """Zoom in on a specific halo by mass."""
    
    # Find halo closest to target mass
    idx = np.argmin(np.abs(halo_mass - target_mass))
    h_pos = halo_pos[idx]
    h_mass = halo_mass[idx]
    h_npart = halo_npart[idx]
    
    print(f"\nVerifying halo:")
    print(f"  Target mass: {target_mass:.2e}")
    print(f"  Found mass: {h_mass:.2e}")
    print(f"  Particles: {h_npart}")
    print(f"  Position: ({h_pos[0]:.3f}, {h_pos[1]:.3f}, {h_pos[2]:.3f})")
    
    # Create detailed view
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Different zoom levels
    zoom_levels = [5.0, 2.0, 0.5]
    
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            zoom = zoom_levels[col]
            
            # Handle periodic boundaries
            rel_pos = particles - h_pos
            rel_pos = rel_pos - np.round(rel_pos / box_size) * box_size
            
            # Select particles
            if row == 0:
                # X-Y projection
                mask = (np.abs(rel_pos[:, 0]) < zoom) & \
                       (np.abs(rel_pos[:, 1]) < zoom)
                local = rel_pos[mask]
                ax.scatter(local[:, 0], local[:, 1], s=0.05, alpha=0.5, c='black')
                ax.set_xlabel('X offset')
                ax.set_ylabel('Y offset')
            else:
                # X-Z projection  
                mask = (np.abs(rel_pos[:, 0]) < zoom) & \
                       (np.abs(rel_pos[:, 2]) < zoom)
                local = rel_pos[mask]
                ax.scatter(local[:, 0], local[:, 2], s=0.05, alpha=0.5, c='black')
                ax.set_xlabel('X offset')
                ax.set_ylabel('Z offset')
            
            # Mark center
            ax.scatter(0, 0, s=200, c='red', marker='+', linewidths=2)
            
            # Set limits
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
            ax.set_aspect('equal')
            ax.set_title(f'Zoom: ±{zoom:.1f} code units\n{len(local)} particles shown')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Halo M={h_mass:.2e} Msun/h, {h_npart} particles\n' +
                f'Position: ({h_pos[0]:.2f}, {h_pos[1]:.2f}, {h_pos[2]:.2f})', 
                fontsize=14)
    plt.tight_layout()
    plt.savefig(f'halo_zoom_M{h_mass:.2e}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Return info for MUSIC2
    print(f"\nMUSIC2 parameters for this halo:")
    print(f"ref_center = {h_pos[0]/box_size:.6f}, {h_pos[1]/box_size:.6f}, {h_pos[2]/box_size:.6f}")
    zoom_extent = min(0.3, 5.0 * 0.2 * (h_mass/1e12)**(1/3) / box_size)
    print(f"ref_extent = {zoom_extent:.4f}, {zoom_extent:.4f}, {zoom_extent:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Match FOF halos to visual overdensities')
    parser.add_argument('snapshot', help='Snapshot file (HDF5)')
    parser.add_argument('fof_catalog', help='FOF catalog file (HDF5)')
    parser.add_argument('--mass-min', type=float, default=5e11, 
                       help='Minimum halo mass (Msun/h)')
    parser.add_argument('--mass-max', type=float, default=2e12,
                       help='Maximum halo mass (Msun/h)')
    parser.add_argument('--top', type=int, default=6,
                       help='Number of halos to visualize')
    parser.add_argument('--target-mass', type=float, default=None,
                       help='Specific halo mass to verify')
    args = parser.parse_args()
    
    # Load data
    print("Loading snapshot...")
    particles, box_size_snap = load_snapshot(args.snapshot)
    
    print("\nLoading FOF catalog...")
    halo_pos, halo_mass, halo_npart, r200, box_size_fof, h = load_fof_catalog(args.fof_catalog)
    
    # Check for consistency
    if abs(box_size_snap - box_size_fof) > 0.1:
        print(f"\nWARNING: Box size mismatch!")
        print(f"  Snapshot: {box_size_snap}")
        print(f"  FOF: {box_size_fof}")
    
    # Create density map overview
    print("\nCreating density projections...")
    create_density_map(particles, halo_pos, halo_mass, box_size_snap)
    
    # Visualize individual halos
    print("\nVisualizing individual halos...")
    find_halos_visually(particles, halo_pos, halo_mass, halo_npart, 
                       box_size_snap, args.mass_min, args.mass_max, args.top)
    
    # Verify specific halo if requested
    if args.target_mass:
        verify_specific_halo(particles, halo_pos, halo_mass, halo_npart,
                           box_size_snap, args.target_mass)

if __name__ == '__main__':
    main()