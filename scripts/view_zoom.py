#!/usr/bin/env python3
"""
Gadget-4 High-Resolution Halo Visualization Script
Visualizes gas, dark matter, and stellar components with adaptive centering
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import h5py
import argparse
from pathlib import Path
import re

class GadgetHaloViewer:
    def __init__(self, snapshot_path, output_dir="./frames", box_size=None, code_unit='kpc'):
        """
        Initialize the halo viewer
        
        Parameters:
        -----------
        snapshot_path : str
            Path to Gadget snapshot file or directory (e.g., 'snapdir_033' or 'snapshot_033.0.hdf5')
        output_dir : str
            Directory to save output frames
        box_size : float
            Simulation box size (if None, will try to read from header)
        code_unit : str
            What the code units represent: 'kpc', 'Mpc', 'kpc/h', 'Mpc/h' (default: 'kpc')
        """
        self.snapshot_path = Path(snapshot_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.box_size = box_size
        self.code_unit = code_unit
        
        # Find all snapshot files
        self.snapshot_files = self.find_snapshot_files()
        print(f"Found {len(self.snapshot_files)} snapshot files")
        
        # Load snapshot data
        self.load_snapshot()

    def find_snapshot_files(self):
        """
        Find all snapshot files for a given snapshot
        
        Returns:
        --------
        list : List of snapshot file paths
        """
        snapshot_files = []
        
        # Case 1: Single file (e.g., snapshot_033.hdf5)
        if self.snapshot_path.is_file():
            return [self.snapshot_path]
        
        # Case 2: Directory with multiple files (e.g., snapdir_033/)
        elif self.snapshot_path.is_dir():
            # Look for files like snapshot_033.0.hdf5, snapshot_033.1.hdf5, etc.
            pattern_base = self.snapshot_path.name.replace('snapdir_', 'snapshot_')
            
            # Try different file extensions and patterns
            patterns = [
                f"{pattern_base}.*.hdf5",
                f"{pattern_base}.*.h5",
                "snapshot_*.hdf5",
                "snapshot_*.h5",
                "snap_*.hdf5",
                "snap_*.h5"
            ]
            
            for pattern in patterns:
                files = list(self.snapshot_path.glob(pattern))
                if files:
                    snapshot_files = sorted(files, key=lambda x: self.extract_file_number(x.name))
                    break
            
            if not snapshot_files:
                raise FileNotFoundError(f"No snapshot files found in {self.snapshot_path}")
        
        # Case 3: Base name provided (e.g., snapshot_033)
        else:
            # Look for single file first
            for ext in ['.hdf5', '.h5']:
                single_file = self.snapshot_path.with_suffix(ext)
                if single_file.exists():
                    return [single_file]
            
            # Look for multi-file format
            base_name = self.snapshot_path.name
            parent_dir = self.snapshot_path.parent
            
            # Try different numbering schemes
            for i in range(100):  # Assume max 100 files
                for ext in ['.hdf5', '.h5']:
                    candidate = parent_dir / f"{base_name}.{i}{ext}"
                    if candidate.exists():
                        snapshot_files.append(candidate)
                    else:
                        break
                if len(snapshot_files) > 0 and not (parent_dir / f"{base_name}.{len(snapshot_files)}.hdf5").exists() and not (parent_dir / f"{base_name}.{len(snapshot_files)}.h5").exists():
                    break
        
        if not snapshot_files:
            raise FileNotFoundError(f"No snapshot files found for {self.snapshot_path}")
        
        return sorted(snapshot_files, key=lambda x: self.extract_file_number(x.name))
    
    def extract_file_number(self, filename):
        """Extract file number from filename for sorting"""
        # Look for pattern like .0.hdf5, .1.hdf5, etc.
        match = re.search(r'\.(\d+)\.(hdf5|h5)$', filename)
        if match:
            return int(match.group(1))
        return 0

    def load_snapshot(self):
        """Load particle data from Gadget snapshot(s)"""
        print(f"Loading {len(self.snapshot_files)} snapshot files...")
        
        # Initialize particle dictionaries with ALL required keys
        self.particles = {
            'dm': {'pos': [], 'mass': [], 'vel': []},
            'gas': {'pos': [], 'mass': [], 'vel': [], 'density': [], 'temp': [], 'hsml': []},
            'stars': {'pos': [], 'mass': [], 'vel': [], 'age': []}
        }
        
        header_read = False
        
        for i, snap_file in enumerate(self.snapshot_files):
            print(f"  Reading file {i+1}/{len(self.snapshot_files)}: {snap_file.name}")
            
            with h5py.File(snap_file, 'r') as f:
                # Read header info from first file only
                if not header_read:
                    header = f['Header']
                    self.redshift = header.attrs['Redshift']
                    self.time = header.attrs['Time']
                    
                    # Try to read Hubble parameter (may have different names or not exist)
                    if 'HubbleParam' in header.attrs:
                        self.hubble = header.attrs['HubbleParam']
                    elif 'HubblepParam' in header.attrs:  # Some codes use this typo
                        self.hubble = header.attrs['HubblepParam']
                    elif 'h' in header.attrs:
                        self.hubble = header.attrs['h']
                    else:
                        print("Warning: HubbleParam not found in header, assuming h=0.7")
                        self.hubble = 0.7
                    
                    if self.box_size is None:
                        self.box_size = header.attrs['BoxSize']
                    
                    # Try to read unit information
                    if 'UnitLength_in_cm' in header.attrs:
                        unit_length_cm = header.attrs['UnitLength_in_cm']
                        self.unit_length_mpc = unit_length_cm / 3.085678e24  # cm to Mpc
                        self.read_unit_from_header = True
                        print(f"UnitLength_in_cm: {unit_length_cm:.3e} -> {self.unit_length_mpc:.3e} Mpc")
                    else:
                        self.read_unit_from_header = False
                        print(f"Warning: UnitLength_in_cm not found, using code_unit='{self.code_unit}'")
                    
                    print(f"Snapshot info: z={self.redshift:.3f}, h={self.hubble:.3f}, BoxSize={self.box_size:.2f} code units")
                    print(f"Box size in Mpc: {self.code_to_mpc(self.box_size):.2f} Mpc")
                    header_read = True
                
                # Load different particle types
                # Dark matter (type 1)
                if 'PartType1' in f and len(f['PartType1/Coordinates']) > 0:
                    self.particles['dm']['pos'].append(f['PartType1/Coordinates'][:])
                    self.particles['dm']['vel'].append(f['PartType1/Velocities'][:])
                    if 'Masses' in f['PartType1']:
                        self.particles['dm']['mass'].append(f['PartType1/Masses'][:])
                
                # Gas (type 0)
                if 'PartType0' in f and len(f['PartType0/Coordinates']) > 0:
                    self.particles['gas']['pos'].append(f['PartType0/Coordinates'][:])
                    self.particles['gas']['vel'].append(f['PartType0/Velocities'][:])
                    if 'Masses' in f['PartType0']:
                        self.particles['gas']['mass'].append(f['PartType0/Masses'][:])
                    if 'Density' in f['PartType0']:
                        self.particles['gas']['density'].append(f['PartType0/Density'][:])
                    if 'InternalEnergy' in f['PartType0']:
                        self.particles['gas']['temp'].append(f['PartType0/InternalEnergy'][:])
                    if 'SmoothingLength' in f['PartType0']:
                        self.particles['gas']['hsml'].append(f['PartType0/SmoothingLength'][:])
                
                # Stars (type 4)
                if 'PartType4' in f and len(f['PartType4/Coordinates']) > 0:
                    self.particles['stars']['pos'].append(f['PartType4/Coordinates'][:])
                    self.particles['stars']['vel'].append(f['PartType4/Velocities'][:])
                    if 'Masses' in f['PartType4']:
                        self.particles['stars']['mass'].append(f['PartType4/Masses'][:])
                    if 'StellarFormationTime' in f['PartType4']:
                        self.particles['stars']['age'].append(f['PartType4/StellarFormationTime'][:])
        
        # Concatenate all arrays
        for ptype in self.particles:
            for prop in self.particles[ptype]:
                if self.particles[ptype][prop]:  # If list is not empty
                    self.particles[ptype][prop] = np.concatenate(self.particles[ptype][prop])
                else:
                    self.particles[ptype][prop] = np.array([])
        
        # Print summary
        for ptype, label in [('dm', 'dark matter'), ('gas', 'gas'), ('stars', 'stellar')]:
            if len(self.particles[ptype]['pos']) > 0:
                print(f"Total {label} particles: {len(self.particles[ptype]['pos'])}")
            else:
                print(f"No {label} particles found")

    def find_highres_region(self):
        """
        Find the high-resolution region by identifying the densest dark matter particles
        In zoom simulations, high-res particles have smaller masses
        
        Returns:
        --------
        center : array
            Center of high-res region
        radius : float
            Radius containing high-res particles
        """
        if len(self.particles['dm']['pos']) == 0:
            print("No dark matter particles found")
            return None, None
        
        # High-res particles have the smallest masses
        dm_masses = self.particles['dm']['mass']
        
        if len(dm_masses) == 0:
            print("Warning: No mass information for DM particles")
            # Fallback: use spatial density
            center = np.median(self.particles['dm']['pos'], axis=0)
            distances = np.linalg.norm(self.particles['dm']['pos'] - center, axis=1)
            radius = np.percentile(distances, 90)
            return center, radius
        
        # Find the minimum mass (high-res particles)
        min_mass = np.min(dm_masses)
        mass_threshold = min_mass * 1.5  # Allow some tolerance
        
        # Select high-res particles
        highres_mask = dm_masses < mass_threshold
        highres_pos = self.particles['dm']['pos'][highres_mask]
        
        if len(highres_pos) == 0:
            print("Warning: Could not identify high-res particles")
            return None, None
        
        print(f"Identified {len(highres_pos)} high-resolution DM particles (mass < {mass_threshold:.2e})")
        print(f"Total DM particles: {len(self.particles['dm']['pos'])}")
        
        # Find center of high-res region
        center = np.mean(highres_pos, axis=0)
        
        # Find radius that contains most high-res particles
        distances = np.linalg.norm(highres_pos - center, axis=1)
        radius = np.percentile(distances, 95)  # 95th percentile to avoid outliers
        
        print(f"High-res region center: {center}")
        print(f"High-res region radius (95th percentile): {radius:.2f}")
        
        return center, radius

    def find_halo_center(self, search_radius=None, use_highres=True):
        """
        Find halo center using iterative approach on dark matter
        
        Parameters:
        -----------
        search_radius : float
            Initial search radius in code units
        use_highres : bool
            If True, only use high-resolution particles for centering
        """
        if 'dm' not in self.particles or len(self.particles['dm']['pos']) == 0:
            print("No dark matter particles found, using gas center")
            if 'gas' in self.particles and len(self.particles['gas']['pos']) > 0:
                return np.mean(self.particles['gas']['pos'], axis=0)
            else:
                return np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        
        dm_pos = self.particles['dm']['pos']
        dm_masses = self.particles['dm']['mass']
        
        # If using high-res only, filter particles first
        if use_highres and len(dm_masses) > 0:
            min_mass = np.min(dm_masses)
            mass_threshold = min_mass * 1.5
            highres_mask = dm_masses < mass_threshold
            dm_pos = dm_pos[highres_mask]
            print(f"Using {len(dm_pos)} high-res particles for centering")
        
        # Start with center of mass
        center = np.mean(dm_pos, axis=0)
        
        # Iteratively refine center
        if search_radius is None:
            search_radius = self.box_size / 10
            
        for i in range(5):  # 5 iterations usually sufficient
            # Find particles within search radius
            distances = np.linalg.norm(dm_pos - center, axis=1)
            mask = distances < search_radius
            
            if np.sum(mask) < 10:
                break
                
            # Recalculate center
            center = np.mean(dm_pos[mask], axis=0)
            search_radius *= 0.8  # Shrink search radius
            
        return center

    def create_projection(self, center, view_radius, resolution=512, axis='z'):
        """
        Create density projections for all particle types
        
        Parameters:
        -----------
        center : array
            Center coordinates for projection
        view_radius : float
            Radius of region to visualize
        resolution : int
            Grid resolution for projection
        axis : str
            Projection axis ('x', 'y', or 'z')
        """
        axis_map = {'x': [1, 2], 'y': [0, 2], 'z': [0, 1]}
        proj_axes = axis_map[axis]
        
        # Create grid
        extent = [-view_radius, view_radius, -view_radius, view_radius]
        grid_coords = np.linspace(-view_radius, view_radius, resolution)
        
        projections = {}
        
        for ptype, data in self.particles.items():
            if len(data['pos']) == 0:
                continue
                
            # Shift coordinates relative to center
            pos_centered = data['pos'] - center
            
            # Select particles within view region
            distances = np.linalg.norm(pos_centered, axis=1)
            mask = distances < view_radius * 1.5  # Slightly larger to avoid edge effects
            
            if np.sum(mask) == 0:
                projections[ptype] = np.zeros((resolution, resolution))
                continue
            
            pos_proj = pos_centered[mask][:, proj_axes]
            masses = data['mass'][mask] if len(data['mass']) > 0 else np.ones(np.sum(mask))
            
            # Create 2D histogram (projection)
            proj, _, _ = np.histogram2d(
                pos_proj[:, 0], pos_proj[:, 1],
                bins=resolution,
                range=[[-view_radius, view_radius], [-view_radius, view_radius]],
                weights=masses
            )
            
            projections[ptype] = proj.T
            
        return projections, extent

    def code_to_mpc(self, value):
        """
        Convert code units to physical Mpc
        Uses UnitLength_in_cm from header if available, otherwise uses specified code_unit
        """
        # If we have explicit unit information from header, use it
        if hasattr(self, 'unit_length_mpc') and hasattr(self, 'read_unit_from_header'):
            if self.unit_length_mpc > 0.01:  # If already in Mpc scale
                return value * self.unit_length_mpc
            else:  # If in kpc or smaller, also account for h
                return value * self.unit_length_mpc / self.hubble
        
        # Otherwise use the specified code_unit
        if self.code_unit == 'Mpc':
            return value
        elif self.code_unit == 'Mpc/h':
            return value / self.hubble
        elif self.code_unit == 'kpc':
            return value / 1000.0
        elif self.code_unit == 'kpc/h':
            return value / 1000.0 / self.hubble
        else:
            print(f"Warning: Unknown code_unit '{self.code_unit}', assuming kpc/h")
            return value / 1000.0 / self.hubble
    
    def plot_composite_view(self, center, view_radius, filename=None, resolution=512, axis='z'):
        """
        Create a composite view with all particle types
        """
        projections, extent = self.create_projection(center, view_radius, resolution, axis)
        
        # Convert extent to Mpc for display
        extent_mpc = [self.code_to_mpc(e) for e in extent]
        center_mpc = [self.code_to_mpc(c) for c in center]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Halo at ({center_mpc[0]:.2f}, {center_mpc[1]:.2f}, {center_mpc[2]:.2f}) Mpc - z={self.redshift:.2f}', fontsize=14)
        
        # Individual component plots
        plot_configs = [
            ('dm', 'Dark Matter', 'viridis', (0, 0)),
            ('gas', 'Gas', 'plasma', (0, 1)),
            ('stars', 'Stars', 'hot', (1, 0))
        ]
        
        for ptype, title, cmap, (i, j) in plot_configs:
            ax = axes[i, j]
            if ptype in projections and np.max(projections[ptype]) > 0:
                im = ax.imshow(
                    projections[ptype], 
                    extent=extent_mpc, 
                    origin='lower', 
                    cmap=cmap,
                    norm=LogNorm(vmin=np.max(projections[ptype])*1e-5, 
                               vmax=np.max(projections[ptype]))
                )
                plt.colorbar(im, ax=ax, label='Surface Density')
            else:
                ax.text(0.5, 0.5, f'No {title} particles', 
                       ha='center', va='center', transform=ax.transAxes)
                
            ax.set_title(title)
            ax.set_xlabel('Distance [Mpc]')
            ax.set_ylabel('Distance [Mpc]')
        
        # Composite plot
        ax = axes[1, 1]
        ax.set_title('Composite RGB')
        
        # Create RGB composite with better normalization
        rgb_image = np.zeros((resolution, resolution, 3))
        
        # Normalize each channel independently with log scaling
        if 'dm' in projections and np.max(projections['dm']) > 0:
            # Dark matter -> Blue channel
            dm_proj = projections['dm']
            dm_proj = np.where(dm_proj > 0, dm_proj, np.min(dm_proj[dm_proj > 0]) if np.any(dm_proj > 0) else 1)
            dm_log = np.log10(dm_proj)
            dm_log = (dm_log - np.min(dm_log)) / (np.max(dm_log) - np.min(dm_log))
            rgb_image[:, :, 2] = dm_log
            
        if 'gas' in projections and np.max(projections['gas']) > 0:
            # Gas -> Green channel
            gas_proj = projections['gas']
            gas_proj = np.where(gas_proj > 0, gas_proj, np.min(gas_proj[gas_proj > 0]) if np.any(gas_proj > 0) else 1)
            gas_log = np.log10(gas_proj)
            gas_log = (gas_log - np.min(gas_log)) / (np.max(gas_log) - np.min(gas_log))
            rgb_image[:, :, 1] = gas_log
            
        if 'stars' in projections and np.max(projections['stars']) > 0:
            # Stars -> Red channel
            stars_proj = projections['stars']
            stars_proj = np.where(stars_proj > 0, stars_proj, np.min(stars_proj[stars_proj > 0]) if np.any(stars_proj > 0) else 1)
            stars_log = np.log10(stars_proj)
            stars_log = (stars_log - np.min(stars_log)) / (np.max(stars_log) - np.min(stars_log))
            rgb_image[:, :, 0] = stars_log
        
        # Apply gamma correction for better visibility
        rgb_image = np.power(rgb_image, 0.5)  # Gamma = 0.5 brightens mid-tones
        
        # Clip to valid range
        rgb_image = np.clip(rgb_image, 0, 1)
        
        ax.imshow(rgb_image, extent=extent_mpc, origin='lower')
        ax.set_xlabel('Distance [Mpc]')
        ax.set_ylabel('Distance [Mpc]')
        ax.text(0.05, 0.95, 'R=Stars, G=Gas, B=DM', 
                transform=ax.transAxes, color='white', 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()
            
        plt.close()

    def create_movie_frame(self, frame_number, view_radius_kpc=100):
        """Create a single movie frame"""
        center = self.find_halo_center()
        filename = self.output_dir / f"frame_{frame_number:04d}.png"
        
        self.plot_composite_view(
            center=center, 
            view_radius=view_radius_kpc,
            filename=filename
        )

def main():
    parser = argparse.ArgumentParser(description='Visualize Gadget-4 halo simulation')
    parser.add_argument('snapshot', help='Path to snapshot file or directory (e.g., snapdir_033 or snapshot_033.0.hdf5)')
    parser.add_argument('--radius', type=float, default=None, 
                       help='View radius in code units (default: auto-detect from high-res region)')
    parser.add_argument('--resolution', type=int, default=512,
                       help='Image resolution (default: 512)')
    parser.add_argument('--output', default='halo_view.png',
                       help='Output filename')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='z',
                       help='Projection axis (default: z)')
    parser.add_argument('--zoom-factor', type=float, default=1.2,
                       help='Zoom factor relative to high-res region (default: 1.2)')
    parser.add_argument('--code-unit', choices=['kpc', 'Mpc', 'kpc/h', 'Mpc/h'], default='Mpc/h',
                       help='What code units represent (default: Mpc/h for 50 Mpc/h box)')
    parser.add_argument('--hubble', type=float, default=None,
                       help='Hubble parameter h (default: read from header or 0.7)')
    
    args = parser.parse_args()
    
    # Create viewer and generate plot
    viewer = GadgetHaloViewer(args.snapshot, code_unit=args.code_unit)
    
    # Override hubble if specified
    if args.hubble is not None:
        viewer.hubble = args.hubble
        print(f"Using user-specified h={args.hubble}")
    
    # Find high-resolution region
    center, auto_radius = viewer.find_highres_region()
    
    if center is None:
        print("Could not identify high-res region, using manual center")
        center = viewer.find_halo_center()
        view_radius = args.radius if args.radius else 100
    else:
        # Use high-res region radius with zoom factor
        view_radius = args.radius if args.radius else auto_radius * args.zoom_factor
    
    print(f"Halo center: {center}")
    print(f"View radius: {view_radius:.2f} code units = {viewer.code_to_mpc(view_radius):.3f} Mpc")
    
    viewer.plot_composite_view(
        center=center,
        view_radius=view_radius,
        filename=args.output,
        resolution=args.resolution,
        axis=args.axis
    )

if __name__ == "__main__":
    main()
