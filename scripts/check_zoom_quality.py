#!/usr/bin/env python3
"""
Check zoom simulation quality:
- Verify particle type distribution
- Check for contamination
- Visualize resolution structure
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os
import sys

def read_snapshot(snapshot_path):
    """Read snapshot (single or multi-file)"""
    if os.path.isdir(snapshot_path):
        files = sorted(glob.glob(os.path.join(snapshot_path, "*.hdf5")))
    else:
        files = [snapshot_path]
    
    data = {'pos': {}, 'mass': {}, 'ids': {}}
    header = None
    
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            if header is None:
                header = dict(f['Header'].attrs)
            
            for ptype in range(6):
                key = f'PartType{ptype}'
                if key in f:
                    pos = f[key]['Coordinates'][:] / 1000.0  # kpc to Mpc
                    
                    if 'Masses' in f[key]:
                        mass = f[key]['Masses'][:]
                    else:
                        mass_table = header.get('MassTable', [0]*6)
                        mass = np.full(len(pos), mass_table[ptype])
                    
                    ids = f[key]['ParticleIDs'][:]
                    
                    if ptype not in data['pos']:
                        data['pos'][ptype] = []
                        data['mass'][ptype] = []
                        data['ids'][ptype] = []
                    
                    data['pos'][ptype].append(pos)
                    data['mass'][ptype].append(mass)
                    data['ids'][ptype].append(ids)
    
    # Concatenate all files
    for ptype in list(data['pos'].keys()):
        data['pos'][ptype] = np.vstack(data['pos'][ptype])
        data['mass'][ptype] = np.hstack(data['mass'][ptype])
        data['ids'][ptype] = np.hstack(data['ids'][ptype])
    
    return data, header

def analyze_zoom_structure(data, header):
    """Analyze the zoom structure"""
    print("\n" + "="*70)
    print("ZOOM SIMULATION QUALITY CHECK")
    print("="*70)
    
    # Header info
    z = header.get('Redshift', 0.0)
    time = header.get('Time', 1.0)
    boxsize = header.get('BoxSize', 100000) / 1000.0  # Mpc
    print(f"\nSnapshot info:")
    print(f"  Redshift: {z:.3f}")
    print(f"  Time: {time:.3f}")
    print(f"  Box size: {boxsize:.1f} Mpc/h")
    
    # Particle counts and masses
    print(f"\nParticle types present:")
    type_names = {0: 'Gas', 1: 'DM (high-res)', 2: 'DM (boundary)', 
                  3: 'DM (boundary)', 4: 'Stars', 5: 'DM (low-res)', 6: 'Dust'}
    
    for ptype in sorted(data['pos'].keys()):
        n = len(data['pos'][ptype])
        if n > 0:
            mass_mean = data['mass'][ptype].mean()
            mass_min = data['mass'][ptype].min()
            mass_max = data['mass'][ptype].max()
            print(f"  Type {ptype} ({type_names.get(ptype, 'Unknown')}): "
                  f"{n:,} particles")
            print(f"    Mass: mean={mass_mean:.3e}, min={mass_min:.3e}, max={mass_max:.3e} (1e10 Msun)")
    
    # Calculate spatial extent of each type
    print(f"\nSpatial distribution:")
    for ptype in sorted(data['pos'].keys()):
        pos = data['pos'][ptype]
        if len(pos) > 0:
            center = pos.mean(axis=0)
            extent = pos.max(axis=0) - pos.min(axis=0)
            radius_90 = np.percentile(np.linalg.norm(pos - center, axis=1), 90)
            print(f"  Type {ptype}: center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) Mpc/h")
            print(f"    Extent: ({extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}) Mpc/h")
            print(f"    90th percentile radius: {radius_90:.2f} Mpc/h")
    
    # Check for contamination
    print(f"\n" + "="*70)
    print("CONTAMINATION CHECK")
    print("="*70)
    
    if 1 in data['pos'] and len(data['pos'][1]) > 0:
        # Find center of high-res region (use DM or gas)
        highres_center = data['pos'][1].mean(axis=0)
        print(f"\nHigh-res region center: ({highres_center[0]:.2f}, "
              f"{highres_center[1]:.2f}, {highres_center[2]:.2f}) Mpc/h")
        
        # Calculate distances from center for each type
        for ptype in sorted(data['pos'].keys()):
            pos = data['pos'][ptype]
            if len(pos) > 0:
                distances = np.linalg.norm(pos - highres_center, axis=1)
                r_median = np.median(distances)
                r_90 = np.percentile(distances, 90)
                r_min = distances.min()
                
                print(f"\nType {ptype} ({type_names.get(ptype, 'Unknown')}):")
                print(f"  Distance from center: min={r_min:.3f}, "
                      f"median={r_median:.2f}, 90th={r_90:.2f} Mpc/h")
                
                # Check contamination for boundary particles
                if ptype in [2, 3, 5]:
                    n_within_1mpc = np.sum(distances < 1.0)
                    n_within_500kpc = np.sum(distances < 0.5)
                    if n_within_1mpc > 0:
                        print(f"  ⚠️  WARNING: {n_within_1mpc:,} boundary particles "
                              f"within 1 Mpc of center!")
                    if n_within_500kpc > 0:
                        print(f"  ⚠️  CRITICAL: {n_within_500kpc:,} boundary particles "
                              f"within 500 kpc of center!")
                    else:
                        print(f"  ✓ No contamination within 500 kpc")
    
    # Mass ratio check
    print(f"\n" + "="*70)
    print("MASS RATIO CHECK")
    print("="*70)
    
    if 1 in data['mass']:
        m_highres = data['mass'][1].mean()
        print(f"\nHigh-res DM mass: {m_highres:.3e} (1e10 Msun)")
        
        for ptype in [2, 3, 5]:
            if ptype in data['mass']:
                m_boundary = data['mass'][ptype].mean()
                ratio = m_boundary / m_highres
                print(f"Type {ptype} / Type 1 mass ratio: {ratio:.1f}×")
                if ratio > 100:
                    print(f"  ⚠️  Very large mass ratio - potential issues")
                elif ratio > 10:
                    print(f"  ⚠️  Large mass ratio - watch for force errors")
                else:
                    print(f"  ✓ Good mass ratio")
    
    return data, header, highres_center if 1 in data['pos'] else None

def plot_particle_distribution(data, header, center=None):
    """Create diagnostic plots"""
    fig = plt.figure(figsize=(18, 6))
    
    # Color scheme
    colors = {0: 'blue', 1: 'black', 2: 'yellow', 3: 'orange', 
              4: 'red', 5: 'gray', 6: 'green'}
    labels = {0: 'Gas', 1: 'DM (HR)', 2: 'DM (boundary)', 3: 'DM (boundary)',
              4: 'Stars', 5: 'DM (LR)', 6: 'Dust'}
    sizes = {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.2, 4: 2.0, 5: 0.2, 6: 1.5}
    alphas = {0: 0.4, 1: 0.2, 2: 0.15, 3: 0.15, 4: 0.8, 5: 0.1, 6: 0.7}
    
    if center is None and 1 in data['pos']:
        center = data['pos'][1].mean(axis=0)
    
    # Three projection views
    views = [('X-Y', 0, 1), ('X-Z', 0, 2), ('Y-Z', 1, 2)]
    
    for i, (name, dim1, dim2) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1)
        
        # Plot in order: low-res first, high-res last
        for ptype in [5, 3, 2, 1, 0, 4, 6]:
            if ptype in data['pos']:
                pos = data['pos'][ptype]
                if len(pos) > 0:
                    # Sample for performance
                    n_sample = min(len(pos), 10000)
                    idx = np.random.choice(len(pos), n_sample, replace=False)
                    
                    ax.scatter(pos[idx, dim1], pos[idx, dim2],
                              s=sizes.get(ptype, 0.5), c=colors.get(ptype, 'gray'),
                              alpha=alphas.get(ptype, 0.3), 
                              label=f'{labels.get(ptype, "Unknown")} ({len(pos):,})',
                              rasterized=True)
        
        # Mark center if available
        if center is not None:
            ax.plot(center[dim1], center[dim2], 'r+', markersize=20, 
                   markeredgewidth=3, label='HR center')
            
            # Draw circles at 0.5, 1, 2, 5 Mpc
            for r in [0.5, 1.0, 2.0, 5.0]:
                circle = plt.Circle((center[dim1], center[dim2]), r,
                                   fill=False, color='red', linestyle='--',
                                   linewidth=1, alpha=0.5)
                ax.add_patch(circle)
        
        ax.set_xlabel(f'{views[i][0][0]} [Mpc/h]')
        ax.set_ylabel(f'{views[i][0][-1]} [Mpc/h]')
        ax.set_title(f'{name} Projection')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    z = header.get('Redshift', 0.0)
    fig.suptitle(f'Zoom Simulation - Particle Distribution (z={z:.3f})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_radial_profile(data, header, center):
    """Plot radial particle density profile"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {0: 'blue', 1: 'black', 2: 'yellow', 3: 'orange', 
              4: 'red', 5: 'gray', 6: 'green'}
    labels = {0: 'Gas', 1: 'DM (HR)', 2: 'DM (boundary)', 3: 'DM (boundary)',
              4: 'Stars', 5: 'DM (LR)', 6: 'Dust'}
    
    # Left panel: Number density
    r_bins = np.logspace(-2, 1.5, 50)  # 0.01 to ~30 Mpc
    r_centers = (r_bins[1:] + r_bins[:-1]) / 2
    
    for ptype in sorted(data['pos'].keys()):
        pos = data['pos'][ptype]
        if len(pos) > 0:
            distances = np.linalg.norm(pos - center, axis=1)
            hist, _ = np.histogram(distances, bins=r_bins)
            
            # Volume of shells
            volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
            density = hist / volumes
            
            mask = density > 0
            ax1.loglog(r_centers[mask], density[mask], 
                      color=colors.get(ptype, 'gray'), 
                      label=labels.get(ptype, f'Type {ptype}'),
                      linewidth=2)
    
    ax1.set_xlabel('Radius [Mpc/h]')
    ax1.set_ylabel('Number Density [particles/Mpc³/h³]')
    ax1.set_title('Radial Number Density Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='500 kpc')
    ax1.axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='1 Mpc')
    
    # Right panel: Cumulative particle count
    for ptype in sorted(data['pos'].keys()):
        pos = data['pos'][ptype]
        if len(pos) > 0:
            distances = np.linalg.norm(pos - center, axis=1)
            distances_sorted = np.sort(distances)
            cumulative = np.arange(1, len(distances_sorted) + 1)
            
            # Sample for plotting
            step = max(1, len(distances_sorted) // 1000)
            ax2.loglog(distances_sorted[::step], cumulative[::step],
                      color=colors.get(ptype, 'gray'),
                      label=labels.get(ptype, f'Type {ptype}'),
                      linewidth=2)
    
    ax2.set_xlabel('Radius [Mpc/h]')
    ax2.set_ylabel('Cumulative Particle Count')
    ax2.set_title('Cumulative Particle Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(1.0, color='orange', linestyle='--', alpha=0.5)
    
    z = header.get('Redshift', 0.0)
    fig.suptitle(f'Radial Profiles (z={z:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check zoom simulation quality')
    parser.add_argument('snapshot', help='Snapshot file or snapdir directory')
    parser.add_argument('--output', '-o', default='zoom_check.png',
                       help='Output figure filename')
    args = parser.parse_args()
    
    print(f"Loading snapshot: {args.snapshot}")
    data, header = read_snapshot(args.snapshot)
    
    # Analyze structure
    data, header, center = analyze_zoom_structure(data, header)
    
    # Create plots
    print(f"\nCreating diagnostic plots...")
    fig1 = plot_particle_distribution(data, header, center)
    
    if center is not None:
        fig2 = plot_radial_profile(data, header, center)
        
        # Save both figures
        fig1.savefig(args.output.replace('.png', '_distribution.png'), 
                    dpi=150, bbox_inches='tight')
        fig2.savefig(args.output.replace('.png', '_profiles.png'),
                    dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output.replace('.png', '_distribution.png')}")
        print(f"Saved: {args.output.replace('.png', '_profiles.png')}")
    else:
        fig1.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Check the plots for:")
    print("  - High-res particles (Type 0,1) concentrated in center")
    print("  - Boundary particles (Type 2) surrounding high-res region")
    print("  - No boundary particles within ~500 kpc of center")
    print("  - Smooth transition between particle types")
    print("  - Mass ratios < 10× between adjacent resolution levels")

if __name__ == '__main__':
    main()
