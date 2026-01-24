#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import struct
import argparse
import sys

class GadgetICVisualizer:
    def __init__(self, filename):
        """
        Initialize the visualizer with a GADGET IC file.
        
        Parameters:
        filename (str): Path to the GADGET initial conditions file
        """
        self.filename = filename
        self.header = {}
        self.positions = {}
        self.masses = {}
        self.particle_types = {}
        
    def read_gadget_header(self, f):
        """Read GADGET format header"""
        header_data = struct.unpack('6I6d4d2I6I2I4d2I', f.read(256))
        
        self.header = {
            'npart': header_data[0:6],
            'mass': header_data[6:12],
            'time': header_data[12],
            'redshift': header_data[13],
            'flag_sfr': header_data[14],
            'flag_feedback': header_data[15],
            'npartTotal': header_data[16:22],
            'flag_cooling': header_data[22],
            'num_files': header_data[23],
            'BoxSize': header_data[24],
            'Omega0': header_data[25],
            'OmegaLambda': header_data[26],
            'HubbleParam': header_data[27]
        }
        
    def read_hdf5_format(self):
        """Read HDF5 format GADGET file"""
        try:
            with h5py.File(self.filename, 'r') as f:
                header_group = f['Header']
                for key in header_group.attrs.keys():
                    self.header[key] = header_group.attrs[key]
                
                for ptype in range(6):
                    group_name = f'PartType{ptype}'
                    if group_name in f:
                        group = f[group_name]
                        if 'Coordinates' in group:
                            self.positions[ptype] = group['Coordinates'][:] / 1000.0
                        if 'Masses' in group:
                            self.masses[ptype] = group['Masses'][:]
                        elif 'MassTable' in self.header and self.header['MassTable'][ptype] > 0:
                            n_particles = len(self.positions[ptype]) if ptype in self.positions else 0
                            self.masses[ptype] = np.full(n_particles, self.header['MassTable'][ptype])
                            
        except Exception as e:
            print(f"Error reading HDF5 format: {e}")
            return False
        return True
    
    def read_binary_format(self):
        """Read binary format GADGET file"""
        try:
            with open(self.filename, 'rb') as f:
                f.read(4)
                self.read_gadget_header(f)
                f.read(8)
                
                total_particles = sum(self.header['npart'])
                if total_particles > 0:
                    positions_data = struct.unpack(f'{total_particles*3}f', 
                                                   f.read(total_particles * 3 * 4))
                    positions_array = np.array(positions_data).reshape(-1, 3) / 1000.0
                    
                    start_idx = 0
                    for ptype in range(6):
                        n = self.header['npart'][ptype]
                        if n > 0:
                            end_idx = start_idx + n
                            self.positions[ptype] = positions_array[start_idx:end_idx]
                            start_idx = end_idx
                
        except Exception as e:
            print(f"Error reading binary format: {e}")
            return False
        return True
    
    def load_data(self):
        """Load data from GADGET IC file"""
        if self.read_hdf5_format():
            print("Successfully read HDF5 format file")
            return True
        
        if self.read_binary_format():
            print("Successfully read binary format file")
            return True
        
        print("Failed to read file in either format")
        return False
    
    def print_summary(self):
        print("\n" + "="*50)
        print("GADGET Initial Conditions Summary")
        print("="*50)
        
        if 'BoxSize' in self.header:
            box_size_mpc = self.header['BoxSize'] / 1000.0 if self.header['BoxSize'] > 1000 else self.header['BoxSize']
            print(f"Box Size: {box_size_mpc:.2f} Mpc/h")
        if 'Omega0' in self.header:
            print(f"Omega_m: {self.header['Omega0']:.3f}")
        if 'OmegaLambda' in self.header:
            print(f"Omega_Lambda: {self.header['OmegaLambda']:.3f}")
        if 'HubbleParam' in self.header:
            print(f"Hubble parameter: {self.header['HubbleParam']:.3f}")
        if 'redshift' in self.header:
            print(f"Redshift: {self.header['redshift']:.3f}")
        
        print("\nParticle counts by type:")
        particle_names = ['Gas', 'Dark Matter (high-res)', 'Dark Matter (low-res)', 'Bulge', 'Stars', 'Boundary']
        
        if 'npart' in self.header:
            npart = self.header['npart']
        else:
            npart = [len(self.positions.get(i, [])) for i in range(6)]
            
        for i, (name, count) in enumerate(zip(particle_names, npart)):
            if count > 0:
                print(f"  Type {i} ({name}): {count:,} particles")
        
        print(f"\nTotal particles: {sum(npart):,}")
        
        print("\nSpatial extent by particle type:")
        for ptype, pos in self.positions.items():
            if len(pos) > 0:
                min_pos = np.min(pos, axis=0)
                max_pos = np.max(pos, axis=0)
                extent = max_pos - min_pos
                print(f"  Type {ptype}: [{min_pos[0]:.2f}, {max_pos[0]:.2f}] × "
                      f"[{min_pos[1]:.2f}, {max_pos[1]:.2f}] × "
                      f"[{min_pos[2]:.2f}, {max_pos[2]:.2f}] Mpc/h")
                print(f"           Extent: {extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} Mpc/h")
    
    def plot_2d_projections(self, sample_fraction=0.1):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        axis_info = {
            'x': {'dims': (1, 2), 'labels': ('Y [Mpc/h]', 'Z [Mpc/h]'), 'title': 'Y-Z plane'},
            'y': {'dims': (0, 2), 'labels': ('X [Mpc/h]', 'Z [Mpc/h]'), 'title': 'X-Z plane'},
            'z': {'dims': (0, 1), 'labels': ('X [Mpc/h]', 'Y [Mpc/h]'), 'title': 'X-Y plane'}
        }
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        labels = ['Gas', 'DM (high-res)', 'DM (low-res)', 'Bulge', 'Stars', 'Boundary']
        
        for i, (axis, info) in enumerate(axis_info.items()):
            ax = axes[i]
            dim1, dim2 = info['dims']
            xlabel, ylabel = info['labels']
            
            for ptype, pos in self.positions.items():
                if len(pos) > 0:
                    n_sample = int(len(pos) * sample_fraction)
                    n_sample = min(n_sample, 10000)
                    
                    if n_sample > 0:
                        indices = np.random.choice(len(pos), n_sample, replace=False)
                        pos_sample = pos[indices]
                        
                        ax.scatter(pos_sample[:, dim1], pos_sample[:, dim2], 
                                  c=colors[ptype], s=0.1, alpha=0.6, 
                                  label=f'{labels[ptype]} (N={len(pos):,})' if i == 0 else "")
            
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{info["title"]} - Full Box', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if handles:
            from matplotlib.lines import Line2D
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                                   markersize=10, alpha=0.8) for i in range(len(handles))]
            plt.legend(legend_handles, legend_labels, loc='upper center', ncol=len(handles))
        
        plt.tight_layout()
        plt.show()
    
    def plot_3d_sample(self, max_particles=5000):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        labels = ['Gas', 'DM (high-res)', 'DM (low-res)', 'Bulge', 'Stars', 'Boundary']
        
        for ptype, pos in self.positions.items():
            if len(pos) > 0:
                n_sample = min(len(pos), max_particles)
                indices = np.random.choice(len(pos), n_sample, replace=False)
                pos_sample = pos[indices]
                
                ax.scatter(pos_sample[:, 0], pos_sample[:, 1], pos_sample[:, 2], 
                          c=colors[ptype], s=1, alpha=0.6, 
                          label=f'{labels[ptype]} (N={len(pos):,})')
        
        ax.set_xlabel('X [Mpc/h]')
        ax.set_ylabel('Y [Mpc/h]')
        ax.set_zlabel('Z [Mpc/h]')
        ax.set_title('3D Particle Distribution (Sample)')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_radial_profile(self, center=None, max_radius=None):
        if center is None:
            if 'BoxSize' in self.header:
                box_size_mpc = self.header['BoxSize'] / 1000.0 if self.header['BoxSize'] > 1000 else self.header['BoxSize']
                center = np.array([box_size_mpc/2] * 3)
            else:
                all_pos = np.vstack([pos for pos in self.positions.values() if len(pos) > 0])
                center = np.mean(all_pos, axis=0)
        
        if max_radius is None:
            if 'BoxSize' in self.header:
                box_size_mpc = self.header['BoxSize'] / 1000.0 if self.header['BoxSize'] > 1000 else self.header['BoxSize']
                max_radius = box_size_mpc / 2
            else:
                max_radius = 25
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        labels = ['Gas', 'DM (high-res)', 'DM (low-res)', 'Bulge', 'Stars', 'Boundary']
        
        for ptype, pos in self.positions.items():
            if len(pos) > 0:
                distances = np.sqrt(np.sum((pos - center)**2, axis=1))
                
                r_bins = np.logspace(-1, np.log10(max_radius), 50)
                r_centers = (r_bins[1:] + r_bins[:-1]) / 2
                
                hist, _ = np.histogram(distances, bins=r_bins)
                volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
                density = hist / volumes
                
                mask = density > 0
                ax.loglog(r_centers[mask], density[mask], 
                         color=colors[ptype], label=labels[ptype], linewidth=2)
        
        ax.set_xlabel('Radius [Mpc/h]')
        ax.set_ylabel('Number Density [particles/Mpc³/h³]')
        ax.set_title('Radial Number Density Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# -------------------------------
#  Command-line Interface Section
# -------------------------------
def visualize_gadget_ic(filename):
    viz = GadgetICVisualizer(filename)
    
    if not viz.load_data():
        print("Failed to load initial conditions file")
        return
    
    viz.print_summary()
    print("\nCreating visualizations...")
    
    viz.plot_2d_projections(sample_fraction=0.05)
    viz.plot_3d_sample(max_particles=3000)
    viz.plot_radial_profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize GADGET-4 initial conditions (HDF5 or binary)."
    )
    parser.add_argument(
        "ic_file",
        type=str,
        help="Path to IC file (e.g., IC_zoom_4Mpc_1028_halo3698_100Mpc_music.hdf5)"
    )

    args = parser.parse_args()

    print("GADGET-4 Initial Conditions Visualizer")
    print("=" * 40)
    print(f"Loading file: {args.ic_file}")

    visualize_gadget_ic(args.ic_file)
