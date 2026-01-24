#!/usr/bin/env python3
"""
Visualize a specific halo with color-coded particle types
Gas (blue), Stars (yellow), Dust (brown)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_halo_particle_ids(halo_file):
    """Load particle IDs from text file"""
    data = np.loadtxt(halo_file, dtype={'names': ('type', 'id'), 
                                         'formats': ('i4', 'i8')})
    
    halo_pids = {
        'gas': data['id'][data['type'] == 0],
        'stars': data['id'][data['type'] == 4],
        'dust': data['id'][data['type'] == 6]
    }
    
    print(f"Loaded particle IDs:")
    for ptype, ids in halo_pids.items():
        print(f"  {ptype}: {len(ids)} particles")
    
    return halo_pids


def read_halo_particles(snapshot_path, halo_pids):
    """Read positions and properties for halo particles"""
    
    import glob
    import os
    
    # Check if input is a directory or a file
    if os.path.isdir(snapshot_path):
        # It's a directory - find snapshot files in it
        snapshot_files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        # It's a file - use the multi-file logic
        base_path = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        snapshot_files = sorted(glob.glob(f"{base_path}.*.hdf5"))
        if not snapshot_files:
            snapshot_files = [snapshot_path]
    
    print(f"\nReading {len(snapshot_files)} snapshot file(s)...")
    print(f"First file: {os.path.basename(snapshot_files[0])}")
    if len(snapshot_files) > 1:
        print(f"Last file:  {os.path.basename(snapshot_files[-1])}")
    
    data = {
        'gas_pos': [], 'gas_mass': [], 'gas_temp': [],
        'star_pos': [], 'star_mass': [], 'star_age': [],
        'dust_pos': [], 'dust_mass': []
    }
    
    # Store redshift from first file
    redshift = 0.0
    
    for fpath in snapshot_files:
        with h5py.File(fpath, 'r') as f:
            # Read redshift from header (only once)
            if redshift == 0.0:
                header = f['Header'].attrs
                redshift = header['Redshift']
            
            # Gas
            if 'PartType0' in f:
                gas_ids = f['PartType0/ParticleIDs'][:]
                mask = np.isin(gas_ids, halo_pids['gas'])
                if np.sum(mask) > 0:
                    data['gas_pos'].append(f['PartType0/Coordinates'][:][mask])
                    data['gas_mass'].append(f['PartType0/Masses'][:][mask])
                    if 'InternalEnergy' in f['PartType0']:
                        # Approximate temperature from internal energy
                        u = f['PartType0/InternalEnergy'][:][mask]
                        # T ~ u for ideal gas (rough approximation)
                        data['gas_temp'].append(u * 1e4)  # Convert to K (approximate)
            
            # Stars
            if 'PartType4' in f:
                star_ids = f['PartType4/ParticleIDs'][:]
                mask = np.isin(star_ids, halo_pids['stars'])
                if np.sum(mask) > 0:
                    data['star_pos'].append(f['PartType4/Coordinates'][:][mask])
                    data['star_mass'].append(f['PartType4/Masses'][:][mask])
                    if 'StellarFormationTime' in f['PartType4']:
                        data['star_age'].append(f['PartType4/StellarFormationTime'][:][mask])
            
            # Dust
            if 'PartType6' in f:
                dust_ids = f['PartType6/ParticleIDs'][:]
                mask = np.isin(dust_ids, halo_pids['dust'])
                if np.sum(mask) > 0:
                    data['dust_pos'].append(f['PartType6/Coordinates'][:][mask])
                    data['dust_mass'].append(f['PartType6/Masses'][:][mask])
    
    # Concatenate
    for key in data:
        if data[key]:
            data[key] = np.concatenate(data[key])
        else:
            data[key] = np.array([])
    
    # Add redshift to data dictionary
    data['redshift'] = redshift
    
    print(f"Filtered particles: Gas={len(data['gas_pos'])}, "
          f"Stars={len(data['star_pos'])}, Dust={len(data['dust_pos'])}")
    
    return data


def find_center(data):
    """Find halo center from stellar COM"""
    if len(data['star_pos']) > 0:
        center = np.average(data['star_pos'], weights=data['star_mass'], axis=0)
    elif len(data['gas_pos']) > 0:
        center = np.average(data['gas_pos'], weights=data['gas_mass'], axis=0)
    else:
        center = np.array([0., 0., 0.])
    
    print(f"\nHalo center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) kpc")
    return center


def plot_halo_projections(data, center, halo_id=None, output_file='halo_visualization.png', 
                          max_r=50, subsample=1):
    """
    Create multi-panel visualization with projections and 3D view
    
    Parameters:
    -----------
    max_r : float
        Maximum radius to plot (kpc)
    subsample : int
        Plot every Nth particle for speed (1 = all particles)
    """
    
    # Recenter coordinates
    gas_pos = data['gas_pos'] - center
    star_pos = data['star_pos'] - center
    dust_pos = data['dust_pos'] - center
    
    # Subsample for plotting if needed
    if subsample > 1:
        gas_idx = np.arange(0, len(gas_pos), subsample)
        star_idx = np.arange(0, len(star_pos), subsample)
        dust_idx = np.arange(0, len(dust_pos), subsample)
    else:
        gas_idx = np.arange(len(gas_pos))
        star_idx = np.arange(len(star_pos))
        dust_idx = np.arange(len(dust_pos))
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add master title
    if halo_id is not None:
        fig.suptitle(f'Visualize Halo {halo_id}', fontsize=20, fontweight='bold', y=0.98)
        # Add subtitle with particle counts and redshift (moved down more)
        n_gas = len(data['gas_pos'])
        n_stars = len(data['star_pos'])
        n_dust = len(data['dust_pos'])
        redshift = data.get('redshift', 0.0)
        fig.text(0.5, 0.945, 
                f'Gas: {n_gas:,}  |  Stars: {n_stars:,}  |  Dust: {n_dust:,}  |  z = {redshift:.2f}',
                ha='center', fontsize=12)
    
    # Colors and sizes
    gas_color = 'lightblue'
    star_color = 'gold'
    dust_color = 'saddlebrown'
    
    gas_size = 1
    star_size = 3
    dust_size = 5
    
    gas_alpha = 0.5  # Less transparent
    star_alpha = 0.8  # More opaque
    dust_alpha = 0.6  # Medium
    
    # XY Projection
    ax1 = fig.add_subplot(gs[0, 0])
    # Plot in order: gas, dust, stars (so stars end up on top)
    if len(gas_pos) > 0:
        ax1.scatter(gas_pos[gas_idx, 0], gas_pos[gas_idx, 1], 
                   s=gas_size, alpha=gas_alpha, color=gas_color, label='Gas')
    if len(dust_pos) > 0:
        ax1.scatter(dust_pos[dust_idx, 0], dust_pos[dust_idx, 1], 
                   s=dust_size, alpha=dust_alpha, color=dust_color, label='Dust')
    if len(star_pos) > 0:
        ax1.scatter(star_pos[star_idx, 0], star_pos[star_idx, 1], 
                   s=star_size, alpha=star_alpha, color=star_color, label='Stars')
    ax1.plot(0, 0, 'r*', markersize=10, label='Center')  # Smaller center marker
    ax1.set_xlim(-max_r, max_r)
    ax1.set_ylim(-max_r, max_r)
    ax1.set_xlabel('X (kpc)')
    ax1.set_ylabel('Y (kpc)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # XZ Projection
    ax2 = fig.add_subplot(gs[0, 1])
    if len(gas_pos) > 0:
        ax2.scatter(gas_pos[gas_idx, 0], gas_pos[gas_idx, 2], 
                   s=gas_size, alpha=gas_alpha, color=gas_color)
    if len(dust_pos) > 0:
        ax2.scatter(dust_pos[dust_idx, 0], dust_pos[dust_idx, 2], 
                   s=dust_size, alpha=dust_alpha, color=dust_color)
    if len(star_pos) > 0:
        ax2.scatter(star_pos[star_idx, 0], star_pos[star_idx, 2], 
                   s=star_size, alpha=star_alpha, color=star_color)
    ax2.plot(0, 0, 'r*', markersize=10)  # Smaller center marker
    ax2.set_xlim(-max_r, max_r)
    ax2.set_ylim(-max_r, max_r)
    ax2.set_xlabel('X (kpc)')
    ax2.set_ylabel('Z (kpc)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # YZ Projection
    ax3 = fig.add_subplot(gs[0, 2])
    if len(gas_pos) > 0:
        ax3.scatter(gas_pos[gas_idx, 1], gas_pos[gas_idx, 2], 
                   s=gas_size, alpha=gas_alpha, color=gas_color)
    if len(dust_pos) > 0:
        ax3.scatter(dust_pos[dust_idx, 1], dust_pos[dust_idx, 2], 
                   s=dust_size, alpha=dust_alpha, color=dust_color)
    if len(star_pos) > 0:
        ax3.scatter(star_pos[star_idx, 1], star_pos[star_idx, 2], 
                   s=star_size, alpha=star_alpha, color=star_color)
    ax3.plot(0, 0, 'r*', markersize=10)  # Smaller center marker
    ax3.set_xlim(-max_r, max_r)
    ax3.set_ylim(-max_r, max_r)
    ax3.set_xlabel('Y (kpc)')
    ax3.set_ylabel('Z (kpc)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 3D View
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    # Plot in order: gas, dust, stars (so stars are on top)
    if len(gas_pos) > 0:
        ax4.scatter(gas_pos[gas_idx, 0], gas_pos[gas_idx, 1], gas_pos[gas_idx, 2],
                   s=gas_size, alpha=gas_alpha*0.4, color=gas_color)  # Even less opaque in 3D
    if len(dust_pos) > 0:
        ax4.scatter(dust_pos[dust_idx, 0], dust_pos[dust_idx, 1], dust_pos[dust_idx, 2],
                   s=dust_size, alpha=dust_alpha, color=dust_color)
    if len(star_pos) > 0:
        ax4.scatter(star_pos[star_idx, 0], star_pos[star_idx, 1], star_pos[star_idx, 2],
                   s=star_size, alpha=star_alpha, color=star_color)
    ax4.scatter([0], [0], [0], s=80, color='red', marker='*')  # Smaller center marker
    ax4.set_xlim(-max_r, max_r)
    ax4.set_ylim(-max_r, max_r)
    ax4.set_zlim(-max_r, max_r)
    ax4.set_xlabel('X (kpc)')
    ax4.set_ylabel('Y (kpc)')
    ax4.set_zlabel('Z (kpc)')
    ax4.set_title('3D')
    
    # Radial density plot
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Compute radii
    gas_r = np.sqrt(np.sum(gas_pos**2, axis=1))
    star_r = np.sqrt(np.sum(star_pos**2, axis=1))
    dust_r = np.sqrt(np.sum(dust_pos**2, axis=1))
    
    # Histogram
    bins = np.linspace(0, max_r, 30)
    if len(gas_r) > 0:
        ax5.hist(gas_r, bins=bins, alpha=0.5, color=gas_color, label='Gas')
    if len(star_r) > 0:
        ax5.hist(star_r, bins=bins, alpha=0.5, color=star_color, label='Stars')
    if len(dust_r) > 0:
        ax5.hist(dust_r, bins=bins, alpha=0.4, color=dust_color, label='Dust')  # More transparent
    ax5.set_xlabel('Radius (kpc)')
    ax5.set_ylabel('Number of particles')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.set_title('Radial Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Cumulative mass profile
    ax6 = fig.add_subplot(gs[1, 2])
    
    if len(gas_r) > 0:
        sort_idx = np.argsort(gas_r)
        cumulative_gas = np.cumsum(data['gas_mass'][sort_idx])
        ax6.plot(gas_r[sort_idx], cumulative_gas, color=gas_color, lw=2, label='Gas')
    
    if len(star_r) > 0:
        sort_idx = np.argsort(star_r)
        cumulative_star = np.cumsum(data['star_mass'][sort_idx])
        ax6.plot(star_r[sort_idx], cumulative_star, color=star_color, lw=2, label='Stars')
    
    if len(dust_r) > 0:
        sort_idx = np.argsort(dust_r)
        cumulative_dust = np.cumsum(data['dust_mass'][sort_idx])
        ax6.plot(dust_r[sort_idx], cumulative_dust, color=dust_color, lw=2, label='Dust')
    
    ax6.set_xlabel('Radius (kpc)')
    ax6.set_ylabel('Cumulative Mass (code units)')
    ax6.set_yscale('log')
    ax6.set_xlim(0, max_r)
    ax6.legend()
    ax6.set_title('Cumulative Mass Profile')
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Visualize a specific halo with color-coded particle types")
        print("\nUsage:")
        print("  python visualize_halo.py <snapshot_or_dir> <particle_id_file> [max_radius] [subsample]")
        print("\nExample:")
        print("  python visualize_halo.py snapshot_026.0.hdf5 halo569_particles.txt 50 1")
        print("  python visualize_halo.py snapdir_026/ halo569_particles.txt 50 1")
        print("\nParameters:")
        print("  snapshot_or_dir: Snapshot file OR directory containing snapshot files")
        print("  max_radius: Maximum radius to plot in kpc (default: 50)")
        print("  subsample: Plot every Nth particle for speed (default: 1, i.e. all particles)")
        print("             Use higher values (5, 10) for faster plotting with many particles")
        sys.exit(1)
    
    snapshot_path = sys.argv[1]
    halo_file = sys.argv[2]
    max_r = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    subsample = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    print("=" * 70)
    print("HALO VISUALIZATION")
    print("=" * 70)
    
    # Load particle IDs
    print("\nStep 1: Loading particle IDs...")
    halo_pids = load_halo_particle_ids(halo_file)
    
    # Read snapshot
    print("\nStep 2: Reading snapshot...")
    data = read_halo_particles(snapshot_path, halo_pids)
    
    # Find center
    print("\nStep 3: Finding halo center...")
    center = find_center(data)
    
    # Create visualization
    print("\nStep 4: Creating visualization...")
    output_base = halo_file.replace('_particles.txt', '').replace('.txt', '')
    output_file = f'{output_base}_visualization.png'
    
    # Extract halo ID from filename (e.g., "halo569_particles.txt" -> 569)
    import re
    halo_id_match = re.search(r'halo(\d+)', halo_file)
    halo_id = int(halo_id_match.group(1)) if halo_id_match else None
    
    plot_halo_projections(data, center, halo_id=halo_id, output_file=output_file, 
                         max_r=max_r, subsample=subsample)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Halo center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) kpc")
    print(f"Particles plotted:")
    print(f"  Gas:   {len(data['gas_pos'])} particles")
    print(f"  Stars: {len(data['star_pos'])} particles")
    print(f"  Dust:  {len(data['dust_pos'])} particles")
    print(f"\nTotal mass:")
    if len(data['gas_mass']) > 0:
        print(f"  Gas:   {np.sum(data['gas_mass']):.3e} code units")
    if len(data['star_mass']) > 0:
        print(f"  Stars: {np.sum(data['star_mass']):.3e} code units")
    if len(data['dust_mass']) > 0:
        print(f"  Dust:  {np.sum(data['dust_mass']):.3e} code units")
    print("\nDone!")


if __name__ == '__main__':
    main()
