#!/usr/bin/env python3
"""
Analyze Radial Velocities and Outflows in Gadget-4 Snapshots

Usage:
    python analyze_outflows.py snapshot_XXX.hdf5
    python analyze_outflows.py snapdir_XXX/  (for multi-file snapshots)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import sys
import os
from pathlib import Path

# Physical constants
PROTONMASS = 1.6726e-24  # g
BOLTZMANN = 1.3806e-16   # erg/K
GAMMA = 5.0/3.0
SOLAR_MASS = 1.989e33    # g
KPC_IN_CM = 3.086e21     # cm

def load_snapshot(snappath):
    """Load snapshot from single file or directory"""
    snappath = Path(snappath)
    
    if snappath.is_dir():
        # Multi-file snapshot
        files = sorted(snappath.glob("*.hdf5"))
        if not files:
            files = sorted(snappath.glob("*.*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {snappath}")
        
        print(f"Loading multi-file snapshot: {len(files)} files")
        
        # Read header from first file
        with h5py.File(files[0], 'r') as f:
            header = dict(f['Header'].attrs)
        
        # Read gas data from all files
        data = {}
        for field in ['Coordinates', 'Velocities', 'Masses', 'InternalEnergy', 'Density']:
            data[field] = []
        
        for fname in files:
            with h5py.File(fname, 'r') as f:
                if 'PartType0' not in f:
                    continue
                for field in data.keys():
                    if field in f['PartType0']:
                        data[field].append(f['PartType0'][field][:])
        
        # Concatenate
        for field in data:
            if data[field]:
                data[field] = np.concatenate(data[field])
            else:
                data[field] = np.array([])
    
    else:
        # Single file
        print(f"Loading single-file snapshot: {snappath}")
        with h5py.File(snappath, 'r') as f:
            header = dict(f['Header'].attrs)
            
            data = {}
            if 'PartType0' in f:
                for field in ['Coordinates', 'Velocities', 'Masses', 'InternalEnergy', 'Density']:
                    if field in f['PartType0']:
                        data[field] = f['PartType0'][field][:]
    
    return data, header

def find_halo_center(data, header):
    """Find halo center using shrinking sphere method"""
    
    pos = data['Coordinates']
    mass = data['Masses']
    
    # Handle periodic boundaries
    BoxSize = header.get('BoxSize', 50000.0)  # kpc
    
    # Initial guess: center of box or mass-weighted center
    if len(pos) == 0:
        return np.array([BoxSize/2, BoxSize/2, BoxSize/2]), np.array([0, 0, 0])
    
    center = np.average(pos, weights=mass, axis=0)
    
    # Shrinking sphere
    radius = BoxSize / 4.0
    for iteration in range(10):
        # Particles within radius
        dx = pos - center
        dx = dx - BoxSize * np.round(dx / BoxSize)  # Periodic wrap
        r = np.sqrt(np.sum(dx**2, axis=1))
        
        mask = r < radius
        if np.sum(mask) < 100:
            break
        
        # New center
        center = np.average(pos[mask], weights=mass[mask], axis=0)
        radius *= 0.8
    
    # Halo velocity (mass-weighted within 50 kpc)
    dx = pos - center
    dx = dx - BoxSize * np.round(dx / BoxSize)
    r = np.sqrt(np.sum(dx**2, axis=1))
    mask = r < 50.0  # kpc
    
    if np.sum(mask) > 0:
        vel_halo = np.average(data['Velocities'][mask], weights=mass[mask], axis=0)
    else:
        vel_halo = np.zeros(3)
    
    print(f"Halo center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) kpc")
    print(f"Halo velocity: ({vel_halo[0]:.1f}, {vel_halo[1]:.1f}, {vel_halo[2]:.1f}) km/s")
    
    return center, vel_halo

def calculate_temperature(u, header):
    """Calculate temperature from internal energy"""
    # Get units
    UnitVelocity = header.get('UnitVelocity_in_cm_per_s', 1e5)
    
    # Mean molecular weight (assume fully ionized)
    mu = 0.6
    
    # u is in code units (velocity^2)
    # T = (gamma-1) * u * mu * m_p / k_B
    T = (GAMMA - 1.0) * u * (UnitVelocity**2) * mu * PROTONMASS / BOLTZMANN
    
    return T

def analyze_outflows(snappath, output_dir=None):
    """Main analysis function"""
    
    # Setup output
    if output_dir is None:
        snapname = Path(snappath).stem
        output_dir = f"outflow_analysis_{snapname}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"OUTFLOW ANALYSIS")
    print(f"{'='*60}\n")
    
    # Load data
    data, header = load_snapshot(snappath)
    
    if len(data['Coordinates']) == 0:
        print("ERROR: No gas particles found!")
        return
    
    # Get snapshot info
    redshift = header.get('Redshift', 0.0)
    time = header.get('Time', 1.0)
    boxsize = header.get('BoxSize', 50000.0)
    h = header.get('HubbleParam', 0.6732)
    
    print(f"Redshift: z = {redshift:.2f}")
    print(f"Scale factor: a = {time:.4f}")
    print(f"Box size: {boxsize:.1f} kpc/h")
    print(f"Gas particles: {len(data['Coordinates']):,}")
    
    # Get units
    UnitLength = header.get('UnitLength_in_cm', 3.086e21)  # kpc
    UnitVelocity = header.get('UnitVelocity_in_cm_per_s', 1e5)  # km/s
    UnitMass = header.get('UnitMass_in_g', 1.989e43)  # 1e10 Msun
    
    # Find halo center
    center, vel_halo = find_halo_center(data, header)
    
    # Calculate radial quantities
    pos = data['Coordinates']
    vel = data['Velocities']
    mass = data['Masses']
    u = data['InternalEnergy']
    rho = data['Density']
    
    # Position relative to halo center (handle periodic boundaries)
    dx = pos - center
    dx = dx - boxsize * np.round(dx / boxsize)
    r = np.sqrt(np.sum(dx**2, axis=1))
    
    # Velocity relative to halo
    dv = vel - vel_halo
    
    # Radial velocity
    r_hat = dx / r[:, np.newaxis]  # Unit radial vector
    v_r = np.sum(dv * r_hat, axis=1)  # km/s
    
    # Temperature
    T = calculate_temperature(u, header)
    
    # Physical density (g/cm^3)
    rho_phys = rho * (UnitMass / UnitLength**3) * time**(-3)  # Account for comoving
    
    print(f"\n{'='*60}")
    print(f"OUTFLOW STATISTICS")
    print(f"{'='*60}\n")
    
    # Define outflow: v_r > 50 km/s and T > 1e5 K
    outflow_mask = (v_r > 50.0) & (T > 1e5)
    
    n_outflow = np.sum(outflow_mask)
    M_outflow = np.sum(mass[outflow_mask]) * UnitMass / SOLAR_MASS / h  # Msun
    
    print(f"Outflow criterion: v_r > 50 km/s and T > 10^5 K")
    print(f"Outflowing particles: {n_outflow:,} ({100*n_outflow/len(r):.1f}%)")
    print(f"Outflowing mass: {M_outflow:.2e} Msun")
    
    # Statistics at different radii
    radii = [10, 20, 50, 100]
    print(f"\nRadial velocity statistics:")
    print(f"{'Radius':>8} {'N_out':>10} {'M_out':>12} {'<v_r>':>10} {'v_r,max':>10}")
    print(f"{'(kpc)':>8} {'':>10} {'(Msun)':>12} {'(km/s)':>10} {'(km/s)':>10}")
    print("-" * 60)
    
    for R in radii:
        mask = (r < R) & outflow_mask
        if np.sum(mask) > 0:
            n = np.sum(mask)
            m = np.sum(mass[mask]) * UnitMass / SOLAR_MASS / h
            vr_mean = np.mean(v_r[mask])
            vr_max = np.max(v_r[mask])
            print(f"{R:8.0f} {n:10,} {m:12.2e} {vr_mean:10.1f} {vr_max:10.1f}")
        else:
            print(f"{R:8.0f} {'0':>10} {'0.00e+00':>12} {'---':>10} {'---':>10}")
    
    # Hot gas statistics
    hot_mask = T > 1e6
    print(f"\nHot gas (T > 10^6 K):")
    print(f"  Particles: {np.sum(hot_mask):,}")
    print(f"  Mass: {np.sum(mass[hot_mask]) * UnitMass / SOLAR_MASS / h:.2e} Msun")
    print(f"  Fraction outflowing: {100*np.sum(hot_mask & outflow_mask)/np.sum(hot_mask):.1f}%")
    
    # Save summary
    with open(f"{output_dir}/outflow_summary.txt", 'w') as f:
        f.write(f"Outflow Analysis Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Snapshot: {snappath}\n")
        f.write(f"Redshift: {redshift:.3f}\n")
        f.write(f"Time: {time:.4f}\n\n")
        f.write(f"Outflow criterion: v_r > 50 km/s, T > 1e5 K\n")
        f.write(f"Total outflowing mass: {M_outflow:.2e} Msun\n")
        f.write(f"Outflowing particles: {n_outflow:,} ({100*n_outflow/len(r):.1f}%)\n")
    
    # Make plots
    print(f"\n{'='*60}")
    print(f"CREATING PLOTS")
    print(f"{'='*60}\n")
    
    plot_radial_velocity_profile(r, v_r, T, mass, UnitMass, SOLAR_MASS, h, output_dir)
    plot_phase_diagram(rho_phys, T, v_r, mass, UnitMass, SOLAR_MASS, h, output_dir)
    plot_outflow_map(dx, v_r, T, output_dir)
    plot_velocity_distribution(v_r, T, output_dir)
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"{'='*60}\n")

def plot_radial_velocity_profile(r, v_r, T, mass, UnitMass, SOLAR_MASS, h, output_dir):
    """Plot radial velocity vs radius"""
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Separate by temperature
    cold = T < 1e5
    warm = (T >= 1e5) & (T < 1e6)
    hot = T >= 1e6
    
    # Top: v_r vs r scatter
    ax = axes[0]
    
    # Sample for plotting (too many points otherwise)
    n_sample = min(50000, len(r))
    idx = np.random.choice(len(r), n_sample, replace=False)
    
    ax.scatter(r[idx][cold[idx]], v_r[idx][cold[idx]], s=0.5, alpha=0.3, 
               c='blue', label=r'Cold ($T < 10^5$ K)', rasterized=True)
    ax.scatter(r[idx][warm[idx]], v_r[idx][warm[idx]], s=0.5, alpha=0.5, 
               c='orange', label=r'Warm ($10^5 < T < 10^6$ K)', rasterized=True)
    ax.scatter(r[idx][hot[idx]], v_r[idx][hot[idx]], s=1.0, alpha=0.7, 
               c='red', label=r'Hot ($T > 10^6$ K)', rasterized=True)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(50, color='green', linestyle=':', alpha=0.5, label='Outflow threshold')
    
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_xlim(0, 200)
    ax.set_ylim(-500, 500)
    ax.legend(loc='upper right', fontsize=10, markerscale=5)
    ax.grid(alpha=0.3)
    
    # Bottom: median profiles
    ax = axes[1]
    
    r_bins = np.logspace(0, 2.5, 30)  # 1 to 300 kpc
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    for mask, color, label in [(cold, 'blue', 'Cold'), 
                                (warm, 'orange', 'Warm'),
                                (hot, 'red', 'Hot')]:
        if np.sum(mask) > 100:
            median = []
            p16 = []
            p84 = []
            
            for i in range(len(r_bins) - 1):
                bin_mask = mask & (r >= r_bins[i]) & (r < r_bins[i+1])
                if np.sum(bin_mask) > 10:
                    median.append(np.median(v_r[bin_mask]))
                    p16.append(np.percentile(v_r[bin_mask], 16))
                    p84.append(np.percentile(v_r[bin_mask], 84))
                else:
                    median.append(np.nan)
                    p16.append(np.nan)
                    p84.append(np.nan)
            
            median = np.array(median)
            p16 = np.array(p16)
            p84 = np.array(p84)
            
            ax.plot(r_centers, median, color=color, linewidth=2, label=label)
            ax.fill_between(r_centers, p16, p84, color=color, alpha=0.2)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(50, color='green', linestyle=':', alpha=0.5, label='Outflow threshold')
    
    ax.set_xlabel('Radius (kpc)', fontsize=12)
    ax.set_ylabel('Median Radial Velocity (km/s)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(1, 300)
    ax.set_ylim(-200, 200)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radial_velocity_profile.png", dpi=150, bbox_inches='tight')
    print("  ✓ radial_velocity_profile.png")
    plt.close()

def plot_phase_diagram(rho, T, v_r, mass, UnitMass, SOLAR_MASS, h, output_dir):
    """Phase diagram colored by radial velocity"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sample for plotting
    n_sample = min(100000, len(rho))
    idx = np.random.choice(len(rho), n_sample, replace=False)
    
    # Left: colored by v_r
    ax = axes[0]
    
    scatter = ax.scatter(rho[idx], T[idx], c=v_r[idx], s=1, alpha=0.5,
                        cmap='RdBu_r', vmin=-200, vmax=200, rasterized=True)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Radial Velocity (km/s)', fontsize=12)
    
    ax.set_xlabel(r'Density (g cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-28, 1e-22)
    ax.set_ylim(1e3, 1e8)
    ax.grid(alpha=0.3)
    ax.set_title('Phase Diagram (colored by v_r)', fontsize=14)
    
    # Right: outflowing vs inflowing
    ax = axes[1]
    
    outflow = v_r > 50
    inflow = v_r < -50
    ambient = ~(outflow | inflow)
    
    for mask, color, label, alpha in [(ambient[idx], 'gray', 'Ambient', 0.2),
                                       (inflow[idx], 'blue', 'Inflow (v_r < -50)', 0.4),
                                       (outflow[idx], 'red', 'Outflow (v_r > 50)', 0.6)]:
        if np.sum(mask) > 0:
            ax.scatter(rho[idx][mask], T[idx][mask], s=1, c=color, 
                      alpha=alpha, label=label, rasterized=True)
    
    ax.set_xlabel(r'Density (g cm$^{-3}$)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-28, 1e-22)
    ax.set_ylim(1e3, 1e8)
    ax.legend(loc='upper right', fontsize=10, markerscale=5)
    ax.grid(alpha=0.3)
    ax.set_title('Outflow vs Inflow', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase_diagram.png", dpi=150, bbox_inches='tight')
    print("  ✓ phase_diagram.png")
    plt.close()

def plot_outflow_map(dx, v_r, T, output_dir):
    """2D map showing outflow regions"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sample hot gas
    hot = T > 1e5
    n_sample = min(50000, np.sum(hot))
    idx = np.random.choice(np.where(hot)[0], n_sample, replace=False)
    
    # Left: Face-on (x-y)
    ax = axes[0]
    scatter = ax.scatter(dx[idx, 0], dx[idx, 1], c=v_r[idx], s=2, alpha=0.6,
                        cmap='RdBu_r', vmin=-200, vmax=200, rasterized=True)
    ax.set_xlabel('X (kpc)', fontsize=12)
    ax.set_ylabel('Y (kpc)', fontsize=12)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_aspect('equal')
    ax.set_title(r'Face-on View (Hot Gas, $T > 10^5$ K)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Radial Velocity (km/s)', fontsize=12)
    
    # Right: Edge-on (x-z)
    ax = axes[1]
    scatter = ax.scatter(dx[idx, 0], dx[idx, 2], c=v_r[idx], s=2, alpha=0.6,
                        cmap='RdBu_r', vmin=-200, vmax=200, rasterized=True)
    ax.set_xlabel('X (kpc)', fontsize=12)
    ax.set_ylabel('Z (kpc)', fontsize=12)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_aspect('equal')
    ax.set_title(r'Edge-on View (Hot Gas, $T > 10^5$ K)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Radial Velocity (km/s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outflow_map.png", dpi=150, bbox_inches='tight')
    print("  ✓ outflow_map.png")
    plt.close()

def plot_velocity_distribution(v_r, T, output_dir):
    """Distribution of radial velocities"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature cuts
    cold = T < 1e5
    warm = (T >= 1e5) & (T < 1e6)
    hot = T >= 1e6
    
    # Left: Histogram
    ax = axes[0]
    
    bins = np.linspace(-500, 500, 100)
    
    ax.hist(v_r[cold], bins=bins, alpha=0.5, color='blue', 
           label=r'Cold ($T < 10^5$ K)', density=True)
    ax.hist(v_r[warm], bins=bins, alpha=0.5, color='orange', 
           label=r'Warm ($10^5 < T < 10^6$ K)', density=True)
    ax.hist(v_r[hot], bins=bins, alpha=0.7, color='red', 
           label=r'Hot ($T > 10^6$ K)', density=True)
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(50, color='green', linestyle=':', label='Outflow threshold')
    ax.axvline(-50, color='purple', linestyle=':', label='Inflow threshold')
    
    ax.set_xlabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Normalized Counts', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-500, 500)
    ax.grid(alpha=0.3)
    
    # Right: Cumulative distribution
    ax = axes[1]
    
    for mask, color, label in [(cold, 'blue', 'Cold'),
                                (warm, 'orange', 'Warm'),
                                (hot, 'red', 'Hot')]:
        if np.sum(mask) > 0:
            sorted_vr = np.sort(v_r[mask])
            cdf = np.arange(1, len(sorted_vr) + 1) / len(sorted_vr)
            ax.plot(sorted_vr, cdf, color=color, linewidth=2, label=label)
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(50, color='green', linestyle=':', alpha=0.5)
    ax.axvline(-50, color='purple', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Radial Velocity (km/s)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-500, 500)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/velocity_distribution.png", dpi=150, bbox_inches='tight')
    print("  ✓ velocity_distribution.png")
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_outflows.py <snapshot_file_or_dir>")
        print("\nExamples:")
        print("  python analyze_outflows.py snapshot_010.hdf5")
        print("  python analyze_outflows.py snapdir_010/")
        sys.exit(1)
    
    snappath = sys.argv[1]
    
    if not os.path.exists(snappath):
        print(f"ERROR: {snappath} does not exist!")
        sys.exit(1)
    
    analyze_outflows(snappath)
