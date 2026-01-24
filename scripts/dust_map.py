#!/usr/bin/env python3
"""
Dust Map Visualization Script for Gadget-4 Zoom Simulations
Visualizes PartType6 (dust particles) with mass-scaled markers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import sys
from pathlib import Path
import glob

class DustMapper:
    """Visualize dust particle distribution in Gadget-4 zoom simulations"""
    
    def __init__(self, snapshot_path, center=None, box_size=None):
        """
        Initialize dust mapper
        
        Parameters:
        -----------
        snapshot_path : str
            Path to Gadget-4 snapshot - can be:
            - Single file: snapshot_100.hdf5
            - Directory: snapdir_100 (will load all chunks)
        center : array-like, optional
            Center coordinates [x, y, z] in code units. If None, uses center of mass
        box_size : float, optional
            Size of region to plot in kpc. If None, uses full extent
        """
        self.snapshot_path = snapshot_path
        self.center = center
        self.box_size = box_size
        self.load_snapshot()
        
    def find_snapshot_files(self):
        """
        Find all snapshot files (handles both single file and multi-file)
        
        Returns:
        --------
        files : list
            List of snapshot file paths to read
        """
        snap_path = Path(self.snapshot_path)
        
        # Case 1: Direct file path
        if snap_path.is_file():
            return [str(snap_path)]
        
        # Case 2: Directory containing chunks
        elif snap_path.is_dir():
            # Look for snapshot files in directory
            pattern1 = snap_path / "snapshot_*.hdf5"
            pattern2 = snap_path / "snap_*.hdf5"
            
            files = sorted(glob.glob(str(pattern1))) + sorted(glob.glob(str(pattern2)))
            
            if len(files) == 0:
                raise ValueError(f"No snapshot files found in directory: {snap_path}")
            
            return files
        
        else:
            raise ValueError(f"Snapshot path not found: {snap_path}")
    
    def load_snapshot(self):
        """Load dust particle data from snapshot (handles multi-file)"""
        print(f"Loading snapshot: {self.snapshot_path}")
        
        # Find all files to read
        snap_files = self.find_snapshot_files()
        
        if len(snap_files) > 1:
            print(f"  Multi-file snapshot with {len(snap_files)} chunks")
        
        # Read header from first file
        with h5py.File(snap_files[0], 'r') as f:
            self.time = f['Header'].attrs['Time']
            self.redshift = f['Header'].attrs['Redshift']
            self.boxsize = f['Header'].attrs['BoxSize']
            
            # HubbleParam location can vary
            if 'HubbleParam' in f['Header'].attrs:
                self.hubble = f['Header'].attrs['HubbleParam']
            elif 'Parameters' in f and 'HubbleParam' in f['Parameters'].attrs:
                self.hubble = f['Parameters'].attrs['HubbleParam']
            else:
                print("  Warning: HubbleParam not found, assuming h=0.7")
                self.hubble = 0.7
        
        # Load particle data from all files
        dust_pos_list = []
        dust_mass_list = []
        dust_vel_list = []
        dust_ids_list = []
        dust_species_list = []
        dust_temp_list = []
        
        has_ids = False
        has_species = False
        has_temp = False
        
        for i, snap_file in enumerate(snap_files):
            with h5py.File(snap_file, 'r') as f:
                # Check if PartType6 exists in this file
                if 'PartType6' not in f:
                    continue
                
                pt6 = f['PartType6']
                
                # Load required fields
                dust_pos_list.append(pt6['Coordinates'][:])
                dust_mass_list.append(pt6['Masses'][:])
                dust_vel_list.append(pt6['Velocities'][:])
                
                # Load optional fields
                if 'ParticleIDs' in pt6:
                    dust_ids_list.append(pt6['ParticleIDs'][:])
                    has_ids = True
                
                if 'DustSpecies' in pt6:
                    dust_species_list.append(pt6['DustSpecies'][:])
                    has_species = True
                
                if 'DustTemperature' in pt6:
                    dust_temp_list.append(pt6['DustTemperature'][:])
                    has_temp = True
        
        # Check if we found any dust particles
        if len(dust_pos_list) == 0:
            raise ValueError("No PartType6 (dust) found in snapshot!")
        
        # Concatenate all chunks
        self.dust_pos = np.vstack(dust_pos_list) / self.hubble  # Convert to physical kpc
        self.dust_mass = np.concatenate(dust_mass_list) * 1e10 / self.hubble  # Convert to Msun
        self.dust_vel = np.vstack(dust_vel_list)
        
        # Handle optional fields
        self.dust_ids = np.concatenate(dust_ids_list) if has_ids else None
        self.dust_species = np.concatenate(dust_species_list) if has_species else None
        self.dust_temp = np.concatenate(dust_temp_list) if has_temp else None
                
        print(f"  Time: {self.time:.4f}, Redshift: {self.redshift:.4f}")
        print(f"  Found {len(self.dust_mass)} dust particles")
        print(f"  Total dust mass: {np.sum(self.dust_mass):.2e} Msun")
        print(f"  Dust mass range: {np.min(self.dust_mass):.2e} - {np.max(self.dust_mass):.2e} Msun")
        
    def set_center(self, center=None):
        """Set or compute center of visualization region"""
        if center is not None:
            self.center = np.array(center)
        else:
            # Use center of mass of dust
            self.center = np.average(self.dust_pos, weights=self.dust_mass, axis=0)
        print(f"  Center: [{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}] kpc")
        
    def make_projection_map(self, axis='z', size=None, n_pixels=512, 
                           mass_weighted=True, cmap='viridis', 
                           vmin=None, vmax=None, show_particles=False,
                           particle_size_scale=1.0, save_path=None):
        """
        Create 2D projection map of dust distribution
        
        Parameters:
        -----------
        axis : str
            Projection axis ('x', 'y', or 'z')
        size : float, optional
            Size of region in kpc. If None, shows all particles
        n_pixels : int
            Resolution of projection grid
        mass_weighted : bool
            If True, shows surface density. If False, shows particle count
        cmap : str
            Matplotlib colormap name
        vmin, vmax : float, optional
            Color scale limits
        show_particles : bool
            If True, overlay individual particles as scatter points
        particle_size_scale : float
            Scale factor for particle marker sizes
        save_path : str, optional
            If provided, saves figure to this path
        """
        # Set center if not already set
        if self.center is None:
            self.set_center()
        
        # Recentering
        pos = self.dust_pos - self.center
        
        # Determine extent
        if size is None:
            size = 2 * np.max(np.abs(pos))
        extent = [-size/2, size/2, -size/2, size/2]
        
        # Select particles within region
        if axis == 'z':
            x, y = pos[:, 0], pos[:, 1]
            axis_idx = 2
        elif axis == 'y':
            x, y = pos[:, 0], pos[:, 2]
            axis_idx = 1
        elif axis == 'x':
            x, y = pos[:, 1], pos[:, 2]
            axis_idx = 0
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        mask = (np.abs(x) < size/2) & (np.abs(y) < size/2)
        x_sel, y_sel = x[mask], y[mask]
        mass_sel = self.dust_mass[mask]
        
        print(f"\n{axis.upper()}-projection:")
        print(f"  {np.sum(mask)} particles within {size:.1f} kpc box")
        print(f"  Dust mass in box: {np.sum(mass_sel):.2e} Msun")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Create 2D histogram
        if mass_weighted:
            weights = mass_sel
            H, xedges, yedges = np.histogram2d(x_sel, y_sel, bins=n_pixels, 
                                              range=[extent[:2], extent[2:]], 
                                              weights=weights)
            # Convert to surface density (Msun/kpc^2)
            pixel_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
            H = H.T / pixel_area
            label = r'Dust Surface Density [M$_\odot$ kpc$^{-2}$]'
        else:
            H, xedges, yedges = np.histogram2d(x_sel, y_sel, bins=n_pixels, 
                                              range=[extent[:2], extent[2:]])
            H = H.T
            label = 'Dust Particle Count'
        
        # Plot
        H_plot = np.ma.masked_where(H == 0, H)
        if vmin is None and mass_weighted:
            vmin = np.percentile(H_plot.compressed(), 5) if len(H_plot.compressed()) > 0 else 1e-6
        if vmax is None and mass_weighted:
            vmax = np.percentile(H_plot.compressed(), 99.5) if len(H_plot.compressed()) > 0 else 1e6
        
        norm = LogNorm(vmin=vmin, vmax=vmax) if mass_weighted else None
        im = ax.imshow(H_plot, extent=extent, origin='lower', 
                      cmap=cmap, norm=norm, interpolation='nearest')
        
        # Optionally overlay particles
        if show_particles:
            # Scale marker sizes by mass
            sizes = particle_size_scale * (mass_sel / np.median(mass_sel))**0.5
            ax.scatter(x_sel, y_sel, s=sizes, c='white', alpha=0.3, 
                      edgecolors='black', linewidths=0.1)
        
        # Formatting
        ax.set_xlabel('X [kpc]' if axis != 'x' else 'Y [kpc]', fontsize=12)
        ax.set_ylabel('Y [kpc]' if axis == 'z' else 'Z [kpc]', fontsize=12)
        ax.set_aspect('equal')
        
        title = f'Dust Distribution ({axis.upper()}-projection)\n'
        title += f'Time = {self.time:.3f}, z = {self.redshift:.3f}'
        ax.set_title(title, fontsize=13)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(label, fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        return fig, ax
    
    def make_particle_scatter(self, axis='z', size=None, 
                             color_by='mass', size_scale=10.0,
                             cmap='plasma', vmin=None, vmax=None,
                             alpha=0.6, save_path=None):
        """
        Create scatter plot with mass-scaled particle sizes
        
        Parameters:
        -----------
        axis : str
            Projection axis ('x', 'y', or 'z')
        size : float, optional
            Size of region in kpc
        color_by : str
            What to color by: 'mass', 'velocity', 'species', 'temperature'
        size_scale : float
            Scale factor for particle sizes
        cmap : str
            Matplotlib colormap
        vmin, vmax : float, optional
            Color scale limits
        alpha : float
            Transparency of particles
        save_path : str, optional
            If provided, saves figure to this path
        """
        # Set center if not already set
        if self.center is None:
            self.set_center()
        
        # Recentering
        pos = self.dust_pos - self.center
        
        # Determine extent
        if size is None:
            size = 2 * np.max(np.abs(pos))
        
        # Select coordinates based on axis
        if axis == 'z':
            x, y = pos[:, 0], pos[:, 1]
            xlabel, ylabel = 'X [kpc]', 'Y [kpc]'
        elif axis == 'y':
            x, y = pos[:, 0], pos[:, 2]
            xlabel, ylabel = 'X [kpc]', 'Z [kpc]'
        elif axis == 'x':
            x, y = pos[:, 1], pos[:, 2]
            xlabel, ylabel = 'Y [kpc]', 'Z [kpc]'
        
        mask = (np.abs(x) < size/2) & (np.abs(y) < size/2)
        x_sel, y_sel = x[mask], y[mask]
        mass_sel = self.dust_mass[mask]
        
        # Determine color values
        if color_by == 'mass':
            c = mass_sel
            clabel = r'Dust Mass [M$_\odot$]'
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif color_by == 'velocity':
            vel_mag = np.linalg.norm(self.dust_vel[mask], axis=1)
            c = vel_mag
            clabel = 'Velocity [km/s]'
            norm = None
        elif color_by == 'species' and self.dust_species is not None:
            c = self.dust_species[mask]
            clabel = 'Dust Species'
            norm = None
        elif color_by == 'temperature' and self.dust_temp is not None:
            c = self.dust_temp[mask]
            clabel = 'Dust Temperature [K]'
            norm = None
        else:
            c = mass_sel
            clabel = r'Dust Mass [M$_\odot$]'
            norm = LogNorm(vmin=vmin, vmax=vmax)
        
        # Scale particle sizes by mass
        sizes = size_scale * (mass_sel / np.median(mass_sel))**0.5
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 9))
        
        sc = ax.scatter(x_sel, y_sel, s=sizes, c=c, cmap=cmap, 
                       alpha=alpha, norm=norm, edgecolors='none')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-size/2, size/2)
        ax.set_ylim(-size/2, size/2)
        ax.set_aspect('equal')
        
        title = f'Dust Particles ({axis.upper()}-projection)\n'
        title += f'Time = {self.time:.3f}, z = {self.redshift:.3f}\n'
        title += f'{np.sum(mask)} particles shown'
        ax.set_title(title, fontsize=13)
        
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(clabel, fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        return fig, ax
    
    def make_multiview(self, size=None, save_path=None):
        """Create 3-panel plot showing all three projections"""
        if self.center is None:
            self.set_center()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (ax_name, axis) in enumerate(zip(['xy', 'xz', 'yz'], ['z', 'y', 'x'])):
            pos = self.dust_pos - self.center
            
            if axis == 'z':
                x, y = pos[:, 0], pos[:, 1]
                xlabel, ylabel = 'X [kpc]', 'Y [kpc]'
            elif axis == 'y':
                x, y = pos[:, 0], pos[:, 2]
                xlabel, ylabel = 'X [kpc]', 'Z [kpc]'
            else:
                x, y = pos[:, 1], pos[:, 2]
                xlabel, ylabel = 'Y [kpc]', 'Z [kpc]'
            
            if size is not None:
                mask = (np.abs(x) < size/2) & (np.abs(y) < size/2)
            else:
                mask = np.ones(len(x), dtype=bool)
                size = 2 * np.max(np.abs(pos))
            
            x_sel, y_sel = x[mask], y[mask]
            mass_sel = self.dust_mass[mask]
            
            # Create histogram
            H, xedges, yedges = np.histogram2d(x_sel, y_sel, bins=256,
                                              range=[[-size/2, size/2], [-size/2, size/2]],
                                              weights=mass_sel)
            pixel_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
            H = H.T / pixel_area
            H = np.ma.masked_where(H == 0, H)
            
            im = axes[i].imshow(H, extent=[-size/2, size/2, -size/2, size/2],
                              origin='lower', cmap='viridis', norm=LogNorm(),
                              interpolation='nearest')
            
            axes[i].set_xlabel(xlabel, fontsize=11)
            axes[i].set_ylabel(ylabel, fontsize=11)
            axes[i].set_aspect('equal')
            axes[i].set_title(f'{axis.upper()}-projection', fontsize=12)
            
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        fig.suptitle(f'Dust Distribution - Time = {self.time:.3f}, z = {self.redshift:.3f}',
                    fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to {save_path}")
        
        return fig, axes


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python dust_map.py <snapshot_path> [options]")
        print("\nOptions:")
        print("  --center X Y Z      Center coordinates in kpc")
        print("  --size SIZE         Box size in kpc")
        print("  --axis AXIS         Projection axis (x/y/z, default: z)")
        print("  --particles         Show individual particles as scatter")
        print("  --multiview         Create 3-panel multiview plot")
        print("  --output PATH       Save figure to this path")
        return
    
    snapshot_path = sys.argv[1]
    
    # Parse options
    center = None
    size = None
    axis = 'z'
    show_particles = False
    multiview = False
    output = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--center':
            center = [float(sys.argv[i+1]), float(sys.argv[i+2]), float(sys.argv[i+3])]
            i += 4
        elif sys.argv[i] == '--size':
            size = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--axis':
            axis = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--particles':
            show_particles = True
            i += 1
        elif sys.argv[i] == '--multiview':
            multiview = True
            i += 1
        elif sys.argv[i] == '--output':
            output = sys.argv[i+1]
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            i += 1
    
    # Create dust mapper
    mapper = DustMapper(snapshot_path, center=center, box_size=size)
    
    # Generate plots
    if multiview:
        mapper.make_multiview(size=size, save_path=output)
    elif show_particles:
        mapper.make_particle_scatter(axis=axis, size=size, save_path=output)
    else:
        mapper.make_projection_map(axis=axis, size=size, save_path=output)
    
    if output is None:
        plt.show()


if __name__ == '__main__':
    main()
