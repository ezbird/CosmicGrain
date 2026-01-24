"""
Visualize multi-resolution zoom simulation
Shows nested structure from 50 Mpc box down to central galaxy
Handles multi-file snapshots (snapdir_XXX/)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import os
import glob

def load_snapshot(snappath):
    """
    Load Gadget-4 snapshot (single file or multi-file snapdir)
    
    Parameters:
    -----------
    snappath : str
        Either:
        - Path to single snapshot file: 'snapshot_050.hdf5'
        - Path to snapdir: 'snapdir_050/'
        - Path to first file: 'snapdir_050/snapshot_050.0.hdf5'
    """
    
    # Determine if this is multi-file or single file FIRST
    if os.path.isdir(snappath):
        # It's a directory - find all files
        snapdir = snappath
        pattern = os.path.join(snapdir, '*.hdf5')
        snapfiles = sorted(glob.glob(pattern))
        if not snapfiles:
            # Try alternate pattern
            pattern = os.path.join(snapdir, 'snap_*.hdf5')
            snapfiles = sorted(glob.glob(pattern))
        if not snapfiles:
            raise FileNotFoundError(f"No HDF5 files found in {snapdir}")
        print(f"Found {len(snapfiles)} files in snapdir")
    elif os.path.isfile(snappath):
        # Check if it's part of a multi-file snapshot
        base = snappath.rsplit('.', 2)[0]  # Remove .X.hdf5
        pattern = f"{base}.*.hdf5"
        snapfiles = sorted(glob.glob(pattern))
        if len(snapfiles) <= 1:
            # Single file
            snapfiles = [snappath]
        else:
            print(f"Detected multi-file snapshot: {len(snapfiles)} files")
    else:
        raise FileNotFoundError(f"Path not found: {snappath}")
    
    # Load header from first file
    with h5py.File(snapfiles[0], 'r') as snap:
        header_attrs = snap['Header'].attrs
        
        # Get Hubble parameter (check both locations)
        if 'HubbleParam' in header_attrs:
            hubble = header_attrs['HubbleParam']
        elif 'Parameters' in snap and 'HubbleParam' in snap['Parameters'].attrs:
            hubble = snap['Parameters'].attrs['HubbleParam']
        else:
            print("Warning: HubbleParam not found, assuming h=0.7")
            hubble = 0.7
        
        header = {
            'time': header_attrs['Time'],
            'redshift': header_attrs['Redshift'],
            'boxsize': header_attrs['BoxSize'],
            'hubble': hubble,
            'num_files': header_attrs.get('NumFilesPerSnapshot', 1),
        }
    
    print(f"Snapshot: z={header['redshift']:.3f}, a={header['time']:.4f}, h={header['hubble']:.3f}")
    
    # Initialize data structure
    data = {'header': header}
    particle_types = ['PartType0', 'PartType1', 'PartType2', 'PartType4']
    
    # Track which types need mass from header
    needs_header_mass = {}
    
    # Load data from all files
    for ifile, snapfile in enumerate(snapfiles):
        print(f"Loading file {ifile+1}/{len(snapfiles)}: {os.path.basename(snapfile)}")
        
        with h5py.File(snapfile, 'r') as snap:
            for ptype in particle_types:
                if ptype not in snap:
                    continue
                
                # Initialize arrays on first file
                if ptype not in data:
                    data[ptype] = {}
                
                # Load particle data
                pos = snap[ptype]['Coordinates'][:]
                npart_this_file = len(pos)
                
                # Append or initialize
                if 'pos' not in data[ptype]:
                    data[ptype]['pos'] = pos
                else:
                    data[ptype]['pos'] = np.vstack([data[ptype]['pos'], pos])
                
                # Mass handling
                if 'Masses' in snap[ptype]:
                    # Individual particle masses
                    mass = snap[ptype]['Masses'][:]
                    if 'mass' not in data[ptype]:
                        data[ptype]['mass'] = mass
                    else:
                        data[ptype]['mass'] = np.concatenate([data[ptype]['mass'], mass])
                else:
                    # Use header mass (same for all particles of this type)
                    header_mass = snap['Header'].attrs['MassTable'][int(ptype[-1])]
                    if header_mass > 0:
                        # Create array of identical masses
                        mass_array = np.full(npart_this_file, header_mass)
                        if 'mass' not in data[ptype]:
                            data[ptype]['mass'] = mass_array
                        else:
                            data[ptype]['mass'] = np.concatenate([data[ptype]['mass'], mass_array])
                
                # Velocities (optional)
                if 'Velocities' in snap[ptype]:
                    vel = snap[ptype]['Velocities'][:]
                    if 'vel' not in data[ptype]:
                        data[ptype]['vel'] = vel
                    else:
                        data[ptype]['vel'] = np.vstack([data[ptype]['vel'], vel])
                
                # Gas-specific properties
                if ptype == 'PartType0':
                    for field in ['Density', 'InternalEnergy', 'SmoothingLength']:
                        if field in snap[ptype]:
                            field_data = snap[ptype][field][:]
                            field_key = field.lower().replace('internalenergy', 'u').replace('smoothinglength', 'hsml')
                            
                            if field_key not in data[ptype]:
                                data[ptype][field_key] = field_data
                            else:
                                data[ptype][field_key] = np.concatenate([data[ptype][field_key], field_data])
    
    # Print particle counts
    print("\nLoaded particles:")
    for ptype in particle_types:
        if ptype in data:
            npart = len(data[ptype]['pos'])
            print(f"  {ptype}: {npart:,} particles")
    
    return data

def calculate_temperature(u, gamma=5/3, mean_molecular_weight=0.59):
    """Convert internal energy to temperature (Kelvin)"""
    k_B = 1.38064852e-16  # erg/K
    m_p = 1.672621898e-24  # g
    
    # u is in code units (km/s)^2, convert to cgs
    u_cgs = u * 1e10  # (km/s)^2 = 1e10 (cm/s)^2
    
    temp = (gamma - 1) * mean_molecular_weight * m_p * u_cgs / k_B
    return temp

def find_main_halo(dm_pos, dm_mass, search_radius=500):
    """Find center of most massive halo using simple method"""
    # Start from box center
    center_guess = np.array([25000, 25000, 25000])
    
    # Handle scalar mass (all particles have same mass)
    if np.isscalar(dm_mass) or len(dm_mass) == 1:
        dm_mass_array = np.full(len(dm_pos), dm_mass if np.isscalar(dm_mass) else dm_mass[0])
    else:
        dm_mass_array = dm_mass
    
    # Iteratively refine
    for i in range(3):
        r = np.linalg.norm(dm_pos - center_guess, axis=1)
        mask = r < search_radius
        
        if np.sum(mask) < 100:
            print(f"Warning: Only {np.sum(mask)} particles found, using box center")
            return center_guess
            
        # Density-weighted center
        center_guess = np.average(dm_pos[mask], weights=dm_mass_array[mask]**2, axis=0)
        search_radius *= 0.7  # Shrink search radius
    
    return center_guess

def make_projection(pos, mass, center, width, npix=512, weights=None):
    """Make 2D projection of particles"""
    # Center and select particles in slab
    pos_centered = pos - center
    
    # Select particles within width
    mask = (np.abs(pos_centered[:, 0]) < width/2) & \
           (np.abs(pos_centered[:, 1]) < width/2) & \
           (np.abs(pos_centered[:, 2]) < width)
    
    if np.sum(mask) == 0:
        return np.zeros((npix, npix))
    
    pos_proj = pos_centered[mask]
    mass_proj = mass[mask] if mass is not None else np.ones(np.sum(mask))
    
    if weights is not None:
        mass_proj *= weights[mask]
    
    # Create 2D histogram
    img, _, _ = np.histogram2d(
        pos_proj[:, 0], pos_proj[:, 1],
        bins=npix,
        range=[[-width/2, width/2], [-width/2, width/2]],
        weights=mass_proj
    )
    
    return img.T

def visualize_zoom(snappath, output='zoom_visualization.png'):
    """Create comprehensive multi-panel visualization"""
    
    print(f"Loading {snappath}...")
    data = load_snapshot(snappath)
    
    z = data['header']['redshift']
    time = data['header']['time']
    
    print(f"\nCreating visualization for z={z:.3f}, a={time:.4f}")
    
    # Find central halo
    print("Finding main halo...")
    if 'PartType1' in data:
        dm_pos = data['PartType1']['pos']
        dm_mass = data['PartType1']['mass']
        center = find_main_halo(dm_pos, dm_mass)
        print(f"Halo center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] kpc")
    else:
        center = np.array([25000, 25000, 25000])
        print("No DM particles, using box center")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(20, 16))
    
    # Define scales for different panels
    scales = [
        ('50 Mpc - Full Box', 50000, 4),
        ('5 Mpc - Zoom Region', 5000, 3),
        ('500 kpc - Central Halo', 500, 2),
        ('50 kpc - Galaxy', 50, 1.5),
    ]
    
    # Color schemes
    gas_cmap = 'inferno'
    dm_cmap = 'viridis'
    star_cmap = 'Greys'
    
    for i, (title, width, smooth) in enumerate(scales):
        print(f"Creating panel {i+1}/16: {title}")
        
        # Gas density (top row)
        if 'PartType0' in data:
            ax = plt.subplot(4, 4, i+1)
            gas_pos = data['PartType0']['pos']
            gas_mass = data['PartType0']['mass']
            gas_rho = data['PartType0']['density']
            
            # Density-weighted projection
            img = make_projection(gas_pos, gas_mass, center, width, npix=512, weights=gas_rho)
            img = gaussian_filter(img, sigma=smooth)
            img = np.ma.masked_where(img <= 0, img)
            
            if img.max() > 0:
                vmin = img[img>0].min() if img[img>0].size > 0 else 1e-10
                vmax = img.max() if img.max() > 0 else 1
                im = ax.imshow(img, origin='lower', extent=[-width/2, width/2, -width/2, width/2],
                              cmap=gas_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.colorbar(im, ax=ax, label='Surface Density')
            
            ax.set_title(f'{title}\nGas Density', fontsize=10)
            ax.set_xlabel('x [kpc]')
            ax.set_ylabel('y [kpc]')
        
        # Gas temperature (second row)
        if 'PartType0' in data:
            ax = plt.subplot(4, 4, i+5)
            temp = calculate_temperature(data['PartType0']['u'])
            
            img = make_projection(gas_pos, gas_mass, center, width, npix=512, weights=temp)
            img = gaussian_filter(img, sigma=smooth)
            img = np.ma.masked_where(img <= 0, img)
            
            if img.max() > 0:
                im = ax.imshow(img, origin='lower', extent=[-width/2, width/2, -width/2, width/2],
                              cmap='RdYlBu_r', norm=LogNorm(vmin=max(1e3, img[img>0].min()), vmax=1e7))
                plt.colorbar(im, ax=ax, label='T [K]')
            
            ax.set_title(f'Gas Temperature', fontsize=10)
            ax.set_xlabel('x [kpc]')
            ax.set_ylabel('y [kpc]')
        
        # Stars (third row)
        if 'PartType4' in data:
            ax = plt.subplot(4, 4, i+9)
            star_pos = data['PartType4']['pos']
            star_mass = data['PartType4']['mass']
            
            img = make_projection(star_pos, star_mass, center, width, npix=512)
            img = gaussian_filter(img, sigma=smooth)
            img = np.ma.masked_where(img <= 0, img)
            
            # Check if we have any stars visible
            if img.max() > 0 and np.sum(img > 0) > 10:
                im = ax.imshow(img, origin='lower', extent=[-width/2, width/2, -width/2, width/2],
                              cmap=star_cmap, norm=LogNorm(vmin=img[img>0].min(), vmax=img.max()))
                plt.colorbar(im, ax=ax, label='Stellar Surface Density')
            else:
                # No stars visible at this scale
                ax.text(0.5, 0.5, 'No stars\nat this scale', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='gray')
                ax.set_xlim(-width/2, width/2)
                ax.set_ylim(-width/2, width/2)
            
            ax.set_title(f'Stars', fontsize=10)
            ax.set_xlabel('x [kpc]')
            ax.set_ylabel('y [kpc]')
        
        # DM with resolution levels (fourth row)
        ax = plt.subplot(4, 4, i+13)
        
        # Show HR and LR DM with different colors
        if 'PartType1' in data:  # HR DM
            hr_pos = data['PartType1']['pos']
            hr_mass = data['PartType1']['mass']
            img_hr = make_projection(hr_pos, hr_mass, center, width, npix=512)
            img_hr = gaussian_filter(img_hr, sigma=smooth)
        else:
            img_hr = np.zeros((512, 512))
        
        if 'PartType2' in data:  # LR DM
            lr_pos = data['PartType2']['pos']
            lr_mass = data['PartType2']['mass']
            img_lr = make_projection(lr_pos, lr_mass, center, width, npix=512)
            img_lr = gaussian_filter(img_lr, sigma=smooth)
        else:
            img_lr = np.zeros((512, 512))
        
        # Combine with different colors
        img_total = img_hr + img_lr
        img_total = np.ma.masked_where(img_total <= 0, img_total)
        
        if img_total.max() > 0:
            im = ax.imshow(img_total, origin='lower', extent=[-width/2, width/2, -width/2, width/2],
                          cmap=dm_cmap, norm=LogNorm(vmin=img_total[img_total>0].min(), vmax=img_total.max()))
            plt.colorbar(im, ax=ax, label='DM Surface Density')
        
        # Overlay HR region outline at larger scales
        if width > 1000:
            from matplotlib.patches import Rectangle
            hr_width = 5000  # 5 Mpc HR region
            rect = Rectangle((-hr_width/2, -hr_width/2), hr_width, hr_width,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(0.05, 0.95, 'HR Region', transform=ax.transAxes,
                   color='red', fontsize=8, va='top')
        
        ax.set_title(f'Dark Matter', fontsize=10)
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        plt.colorbar(im, ax=ax, label='DM Surface Density')
    
    # Add overall title
    fig.suptitle(f'Zoom Simulation: z={z:.3f}, t={time:.4f}\n'
                 f'Halo Center: [{center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f}] kpc',
                 fontsize=16, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    print(f"\nSaving to {output}...")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print("Done!")
    
    return fig

def make_phase_diagram(snappath, output='phase_diagram.png'):
    """Create density-temperature phase diagram"""
    
    print(f"\nLoading {snappath} for phase diagram...")
    data = load_snapshot(snappath)
    
    if 'PartType0' not in data:
        print("No gas particles found!")
        return
    
    rho = data['PartType0']['density']
    temp = calculate_temperature(data['PartType0']['u'])
    mass = data['PartType0']['mass']
    
    # Convert density to n_H (hydrogen number density)
    # rho is in code units (10^10 Msun/h) / (kpc/h)^3
    h = data['header']['hubble']
    rho_cgs = rho * 6.77e-22 * h**2  # g/cm^3
    n_H = rho_cgs / (1.4 * 1.67e-24)  # cm^-3, assume primordial composition
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 2D histogram
    h2d, xedges, yedges = np.histogram2d(
        np.log10(n_H), np.log10(temp),
        bins=256,
        weights=mass,
        range=[[-8, 2], [2, 8]]
    )
    
    im = ax.imshow(h2d.T, origin='lower',
                   extent=[-8, 2, 2, 8],
                   aspect='auto', cmap='inferno',
                   norm=LogNorm())
    
    # Add cooling floor line if present
    ax.axvline(np.log10(1e-4), color='red', linestyle='--', label='Your cooling floor (~10⁻²⁶ g/cm³)')
    
    # Add typical regimes
    ax.text(-6, 7, 'Hot CGM', fontsize=12, color='white')
    ax.text(-2, 3.5, 'Cold ISM', fontsize=12, color='white')
    ax.text(-4, 5, 'Warm Medium', fontsize=12, color='white')
    
    ax.set_xlabel('log₁₀ n_H [cm⁻³]', fontsize=14)
    ax.set_ylabel('log₁₀ T [K]', fontsize=14)
    ax.set_title(f'Gas Phase Diagram (z={data["header"]["redshift"]:.3f})', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=ax, label='Gas Mass')
    plt.tight_layout()
    
    print(f"Saving phase diagram to {output}...")
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print("Done!")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_zoom.py <snapshot>")
        print("\nExamples:")
        print("  python visualize_zoom.py snapshot_050.hdf5")
        print("  python visualize_zoom.py snapdir_050/")
        print("  python visualize_zoom.py snapdir_050/snapshot_050.0.hdf5")
        sys.exit(1)
    
    snappath = sys.argv[1]
    
    # Generate output names based on input
    if os.path.isdir(snappath):
        basename = os.path.basename(snappath.rstrip('/'))
    else:
        basename = os.path.basename(snappath).replace('.hdf5', '').split('.')[0]
    
    # Make main visualization
    visualize_zoom(snappath, output=f'{basename}_zoom_viz.png')
    
    # Make phase diagram
    make_phase_diagram(snappath, output=f'{basename}_phase.png')
