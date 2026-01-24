#!/usr/bin/env python3
"""
Batch process dust snapshots to create time series and movies
Useful for tracking dust evolution in Gadget-4 zoom simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from dust_map import DustMapper
from pathlib import Path
import glob
import re

class DustEvolution:
    """Analyze dust evolution across multiple snapshots"""
    
    def __init__(self, snapshot_dir, pattern='snapshot_*'):
        """
        Initialize with directory containing snapshots
        
        Parameters:
        -----------
        snapshot_dir : str
            Directory containing snapshot files or snapdir subdirectories
        pattern : str
            Pattern for snapshot files/directories (e.g., 'snapshot_*', 'snapdir_*', 'snap_*')
        """
        self.snapshot_dir = Path(snapshot_dir)
        
        # Try to find snapshots - could be files or directories
        self.snapshots = []
        
        # Look for snapdir_* directories (multi-file snapshots)
        snapdirs = sorted(self.snapshot_dir.glob('snapdir_*'))
        if len(snapdirs) > 0:
            self.snapshots = snapdirs
            print(f"Found {len(self.snapshots)} snapdir directories")
        else:
            # Look for single snapshot files
            self.snapshots = sorted(self.snapshot_dir.glob(pattern + '.hdf5'))
            if len(self.snapshots) == 0:
                self.snapshots = sorted(self.snapshot_dir.glob(pattern + '.*.hdf5'))
            
            if len(self.snapshots) == 0:
                raise ValueError(f"No snapshots found matching {pattern} in {snapshot_dir}")
            
            print(f"Found {len(self.snapshots)} snapshot files")
        
        # Extract snapshot numbers
        self.snap_nums = []
        for snap in self.snapshots:
            # Try different naming patterns
            match = re.search(r'(?:snapshot|snapdir|snap)_(\d+)', snap.name)
            if match:
                self.snap_nums.append(int(match.group(1)))
        
    def extract_dust_stats(self, center=None, radius=None):
        """
        Extract dust statistics from all snapshots
        
        Parameters:
        -----------
        center : array-like, optional
            Center for measuring enclosed dust mass
        radius : float, optional
            Radius for measuring enclosed dust mass (kpc)
        
        Returns:
        --------
        stats : dict
            Dictionary containing time series of dust properties
        """
        times = []
        redshifts = []
        total_masses = []
        n_particles = []
        enclosed_masses = []
        mean_particle_masses = []
        
        print("\nExtracting dust statistics...")
        for i, snap in enumerate(self.snapshots):
            try:
                mapper = DustMapper(str(snap))
                
                times.append(mapper.time)
                redshifts.append(mapper.redshift)
                total_masses.append(np.sum(mapper.dust_mass))
                n_particles.append(len(mapper.dust_mass))
                mean_particle_masses.append(np.mean(mapper.dust_mass))
                
                # Calculate enclosed mass if center and radius provided
                if center is not None and radius is not None:
                    pos = mapper.dust_pos - np.array(center)
                    r = np.linalg.norm(pos, axis=1)
                    enclosed_masses.append(np.sum(mapper.dust_mass[r < radius]))
                
                print(f"  {i+1}/{len(self.snapshots)}: snap {self.snap_nums[i]}, "
                      f"t={mapper.time:.3f}, N={len(mapper.dust_mass)}, "
                      f"M_dust={np.sum(mapper.dust_mass):.2e} Msun")
                
            except Exception as e:
                print(f"  Error processing {snap.name}: {e}")
                continue
        
        stats = {
            'time': np.array(times),
            'redshift': np.array(redshifts),
            'total_mass': np.array(total_masses),
            'n_particles': np.array(n_particles),
            'mean_particle_mass': np.array(mean_particle_masses),
            'snap_nums': np.array(self.snap_nums[:len(times)])
        }
        
        if len(enclosed_masses) > 0:
            stats['enclosed_mass'] = np.array(enclosed_masses)
        
        return stats
    
    def plot_evolution(self, stats, save_path='dust_evolution.png'):
        """
        Create multi-panel plot showing dust evolution
        
        Parameters:
        -----------
        stats : dict
            Statistics dictionary from extract_dust_stats()
        save_path : str
            Output path for figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Total dust mass vs time
        ax = axes[0, 0]
        ax.plot(stats['time'], stats['total_mass'], 'o-', lw=2, ms=4)
        ax.set_xlabel('Time [code units]', fontsize=11)
        ax.set_ylabel(r'Total Dust Mass [M$_\odot$]', fontsize=11)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.set_title('Dust Mass Evolution', fontsize=12)
        
        # Panel 2: Number of dust particles
        ax = axes[0, 1]
        ax.plot(stats['time'], stats['n_particles'], 'o-', lw=2, ms=4, color='C1')
        ax.set_xlabel('Time [code units]', fontsize=11)
        ax.set_ylabel('Number of Dust Particles', fontsize=11)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.set_title('Dust Particle Count', fontsize=12)
        
        # Panel 3: Mean particle mass
        ax = axes[1, 0]
        ax.plot(stats['time'], stats['mean_particle_mass'], 'o-', lw=2, ms=4, color='C2')
        ax.set_xlabel('Time [code units]', fontsize=11)
        ax.set_ylabel(r'Mean Particle Mass [M$_\odot$]', fontsize=11)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.set_title('Mean Dust Particle Mass', fontsize=12)
        
        # Panel 4: Enclosed mass (if available) or mass growth rate
        ax = axes[1, 1]
        if 'enclosed_mass' in stats:
            ax.plot(stats['time'], stats['enclosed_mass'], 'o-', lw=2, ms=4, color='C3')
            ax.set_ylabel(r'Enclosed Dust Mass [M$_\odot$]', fontsize=11)
            ax.set_title('Dust Mass in Target Region', fontsize=12)
            ax.set_yscale('log')
        else:
            # Plot mass growth rate
            dt = np.diff(stats['time'])
            dM = np.diff(stats['total_mass'])
            growth_rate = dM / dt
            t_mid = (stats['time'][1:] + stats['time'][:-1]) / 2
            ax.plot(t_mid, growth_rate, 'o-', lw=2, ms=4, color='C4')
            ax.axhline(0, color='k', ls='--', alpha=0.3)
            ax.set_ylabel(r'Dust Growth Rate [M$_\odot$/time]', fontsize=11)
            ax.set_title('Dust Production/Destruction Rate', fontsize=12)
        
        ax.set_xlabel('Time [code units]', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved evolution plot to {save_path}")
        
        return fig
    
    def create_movie_frames(self, output_dir='movie_frames', 
                           axis='z', size=100, center=None,
                           n_pixels=512, every_n=1):
        """
        Create individual frames for making a movie
        
        Parameters:
        -----------
        output_dir : str
            Directory to save frames
        axis : str
            Projection axis
        size : float
            Box size in kpc
        center : array-like, optional
            Fixed center for all frames
        n_pixels : int
            Resolution
        every_n : int
            Process every n-th snapshot
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nCreating movie frames in {output_dir}...")
        print(f"Processing every {every_n} snapshot(s)")
        
        # Determine global color scale
        vmin, vmax = None, None
        if center is None:
            print("Computing global color scale...")
            all_vals = []
            for snap in self.snapshots[::every_n]:
                try:
                    mapper = DustMapper(str(snap))
                    mapper.set_center(center)
                    pos = mapper.dust_pos - mapper.center
                    if axis == 'z':
                        x, y = pos[:, 0], pos[:, 1]
                    elif axis == 'y':
                        x, y = pos[:, 0], pos[:, 2]
                    else:
                        x, y = pos[:, 1], pos[:, 2]
                    
                    mask = (np.abs(x) < size/2) & (np.abs(y) < size/2)
                    if np.sum(mask) > 0:
                        all_vals.append(np.sum(mapper.dust_mass[mask]))
                except:
                    continue
            
            if len(all_vals) > 0:
                all_vals = np.array(all_vals)
                vmin = np.percentile(all_vals, 5) / (size * size)
                vmax = np.percentile(all_vals, 95) / (size * size)
        
        # Create frames
        for i, snap in enumerate(self.snapshots[::every_n]):
            try:
                frame_num = self.snap_nums[i * every_n]
                mapper = DustMapper(str(snap), center=center)
                
                filename = output_path / f'frame_{frame_num:04d}.png'
                mapper.make_projection_map(
                    axis=axis,
                    size=size,
                    n_pixels=n_pixels,
                    vmin=vmin,
                    vmax=vmax,
                    save_path=str(filename)
                )
                plt.close()
                
                print(f"  Created frame {i+1}/{len(self.snapshots[::every_n])}: {filename.name}")
                
            except Exception as e:
                print(f"  Error creating frame for {snap.name}: {e}")
                continue
        
        print(f"\nFrames saved to {output_dir}/")
        print("To create movie with ffmpeg:")
        print(f"  cd {output_dir}")
        print(f"  ffmpeg -framerate 10 -pattern_type glob -i 'frame_*.png' "
              f"-c:v libx264 -pix_fmt yuv420p dust_evolution.mp4")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dust_evolution.py <snapshot_directory> [options]")
        print("\nOptions:")
        print("  --stats             Extract and plot statistics")
        print("  --movie             Create movie frames")
        print("  --center X Y Z      Fixed center for analysis")
        print("  --radius R          Radius for enclosed mass (kpc)")
        print("  --size SIZE         Box size for movie (kpc)")
        print("  --axis AXIS         Projection axis for movie")
        return
    
    snap_dir = sys.argv[1]
    
    # Parse options
    do_stats = '--stats' in sys.argv
    do_movie = '--movie' in sys.argv
    
    center = None
    radius = None
    size = 100
    axis = 'z'
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--center':
            center = [float(sys.argv[i+1]), float(sys.argv[i+2]), float(sys.argv[i+3])]
            i += 4
        elif sys.argv[i] == '--radius':
            radius = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--size':
            size = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--axis':
            axis = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    # Create evolution analyzer
    evol = DustEvolution(snap_dir)
    
    # Extract and plot statistics
    if do_stats or (not do_stats and not do_movie):
        stats = evol.extract_dust_stats(center=center, radius=radius)
        evol.plot_evolution(stats)
        plt.show()
    
    # Create movie frames
    if do_movie:
        evol.create_movie_frames(axis=axis, size=size, center=center)


if __name__ == '__main__':
    main()
