#!/usr/bin/env python3
"""
Plot the Lilly–Madau star formation history from Gadget-4 zoom simulations.
Only includes HIGH-RESOLUTION region, not the entire box.

Usage:
    python plot_lilly_madau_highres.py /path/to/output/
    python plot_lilly_madau_highres.py sim1/ sim2/ -o lilly_madau.png
"""
import os
import sys
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Observational data (Madau & Dickinson 2014)
OBS_Z = np.array([0.01, 0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.75,
                  2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5])
OBS_SFR = np.array([0.015, 0.017, 0.03, 0.055, 0.08, 0.10, 0.11,
                    0.12, 0.10, 0.09, 0.07, 0.045, 0.03, 0.01,
                    0.003, 0.001])
OBS_ERR = OBS_SFR * 0.2  # Assume 20% errors


def load_halo_particle_ids(halo_file):
    """
    Load gas particle IDs for a specific halo from text file.
    
    Format expected:
        TYPE PARTICLE_ID
        0 12345
        0 12346
        ...
    
    Args:
        halo_file: path to halo particle ID file
    
    Returns:
        array of gas particle IDs
    """
    data = np.loadtxt(halo_file, dtype={'names': ('type', 'id'), 
                                        'formats': ('i4', 'i8')})
    
    # Extract only gas particles (type 0)
    gas_ids = data['id'][data['type'] == 0]
    
    print(f"  Loaded {len(gas_ids)} gas particle IDs from halo file")
    
    return gas_ids


def identify_highres_gas(all_masses, mass_threshold=None):
    """
    Identify high-resolution gas particles in a zoom simulation.
    
    Args:
        all_masses: array of gas particle masses
        mass_threshold: optional manual threshold (in code units)
    
    Returns:
        boolean mask identifying high-res particles
    """
    if mass_threshold is not None:
        return all_masses < mass_threshold
    
    # Automatic detection: find dominant low-mass population
    hist, bin_edges = np.histogram(np.log10(all_masses), bins=50)
    peak_bin = np.argmax(hist)
    peak_mass = 10**bin_edges[peak_bin]
    
    # High-res particles are within 2× of peak mass
    highres_mask = (all_masses > peak_mass * 0.5) & (all_masses < peak_mass * 2.0)
    
    # Sanity check: if >80% are "high-res", probably not a zoom
    if np.sum(highres_mask) / len(all_masses) > 0.8:
        highres_mask = np.ones_like(all_masses, dtype=bool)
    
    return highres_mask


def estimate_highres_volume(positions_highres, h_param):
    """
    Estimate the volume of the high-resolution region.
    Uses a bounding box approach.
    
    Args:
        positions_highres: Nx3 array of high-res particle positions (comoving kpc/h)
        h_param: Hubble parameter h
    
    Returns:
        volume in comoving Mpc^3 (no h factors)
    """
    # Find bounding box
    xmin, ymin, zmin = positions_highres.min(axis=0)
    xmax, ymax, zmax = positions_highres.max(axis=0)
    
    # Dimensions in kpc/h
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    
    # Convert to Mpc (no h): kpc/h → Mpc = (kpc/h) / 1000 / h
    Lx_mpc = Lx / 1000.0 / h_param
    Ly_mpc = Ly / 1000.0 / h_param
    Lz_mpc = Lz / 1000.0 / h_param
    
    volume_mpc3 = Lx_mpc * Ly_mpc * Lz_mpc
    
    return volume_mpc3


def compute_sfr_density_highres(snapshot_path, halo_ids=None, mass_threshold=None, verbose=False):
    """
    Compute SFR density using ONLY high-resolution region.
    
    Args:
        snapshot_path: path to snapshot directory or file
        halo_ids: optional array of gas particle IDs to use (halo filtering)
        mass_threshold: optional mass threshold for high-res identification
        verbose: print diagnostic info
    
    Returns:
        (redshift, rho_SFR, n_highres, volume_highres)
        where rho_SFR is in Msun/yr/Mpc^3
    """
    # Gather part-files
    if os.path.isdir(snapshot_path):
        files = sorted(glob.glob(os.path.join(snapshot_path, '*.hdf5')))
    else:
        files = [snapshot_path]
    
    if not files:
        raise FileNotFoundError(f"No HDF5 files for snapshot: {snapshot_path}")
    
    # Read header and unit info from first file
    with h5py.File(files[0], 'r') as f0:
        H = f0['Header'].attrs
        P = f0['Parameters'].attrs
        
        h = H.get('HubbleParam', 1.0)
        a = H.get('Time', None)
        redshift = H.get('Redshift', (1.0/a - 1.0) if a is not None else None)
        
        # Unit conversions
        UnitLength_in_cm   = P['UnitLength_in_cm']
        UnitMass_in_g      = P['UnitMass_in_g']
        UnitVelocity_in_cm = P['UnitVelocity_in_cm_per_s']
        UnitTime_in_s      = UnitLength_in_cm / UnitVelocity_in_cm
        
        SEC_PER_YEAR = 3.15576e7
        SOLAR_MASS   = 1.98847e33
        sfr_conv = (UnitMass_in_g / SOLAR_MASS) / (UnitTime_in_s / SEC_PER_YEAR)
    
    # Collect all gas data
    all_masses = []
    all_sfr = []
    all_positions = []
    all_ids = []
    
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if 'PartType0' not in f:
                continue
            
            masses = f['PartType0']['Masses'][:]
            all_masses.append(masses)
            
            if 'StarFormationRate' in f['PartType0']:
                sfr = f['PartType0']['StarFormationRate'][:]
            else:
                sfr = np.zeros_like(masses)
            all_sfr.append(sfr)
            
            if 'Coordinates' in f['PartType0']:
                pos = f['PartType0']['Coordinates'][:]
                all_positions.append(pos)
            
            if 'ParticleIDs' in f['PartType0']:
                ids = f['PartType0']['ParticleIDs'][:]
                all_ids.append(ids)
    
    # Concatenate all particles
    all_masses = np.concatenate(all_masses)
    all_sfr = np.concatenate(all_sfr)
    all_positions = np.concatenate(all_positions)
    all_ids = np.concatenate(all_ids) if all_ids else np.arange(len(all_masses))
    
    # Identify high-res particles
    if halo_ids is not None:
        # Use halo filtering (particle IDs)
        highres_mask = np.isin(all_ids, halo_ids)
        if verbose:
            print(f"  Using halo filtering: {np.sum(highres_mask)} / {len(all_ids)} gas particles in halo")
    else:
        # Use mass-based filtering
        highres_mask = identify_highres_gas(all_masses, mass_threshold)
        if verbose:
            print(f"  Using mass filtering: {np.sum(highres_mask)} / {len(all_ids)} gas particles in high-res region")
    
    # Filter to high-res only
    sfr_highres = all_sfr[highres_mask]
    positions_highres = all_positions[highres_mask]
    n_highres = np.sum(highres_mask)
    
    # Total SFR from high-res particles (Msun/yr)
    total_sfr_msun_per_yr = np.sum(sfr_highres) * sfr_conv
    
    # Estimate high-res volume (Mpc^3, no h)
    volume_highres = estimate_highres_volume(positions_highres, h)
    
    # SFR density (Msun/yr/Mpc^3)
    rho_sfr = total_sfr_msun_per_yr / volume_highres
    
    if verbose:
        print(f"  z={redshift:.2f}: {n_highres} high-res gas, "
              f"V={volume_highres:.2f} Mpc^3, "
              f"SFR={total_sfr_msun_per_yr:.2e} Msun/yr, "
              f"rho_SFR={rho_sfr:.2e} Msun/yr/Mpc^3")
    
    return redshift, rho_sfr, n_highres, volume_highres


def list_snapshots(root):
    """
    Return a sorted list of snapshot identifiers under 'root'.
    """
    if os.path.isdir(root):
        # Check for snapdir_* subdirectories first
        dirs = sorted(glob.glob(os.path.join(root, 'snapdir_*')))
        if dirs:
            return dirs
        
        # Otherwise look for HDF5 files
        files = sorted(glob.glob(os.path.join(root, '*.hdf5')))
        if files:
            return files
        
        raise FileNotFoundError(f"No snapshots found under {root}")
    
    elif os.path.isfile(root) and root.endswith('.hdf5'):
        return [root]
    else:
        raise FileNotFoundError(f"Invalid snapshot path: {root}")


def plot_lilly_madau(paths, labels=None, output=None, halo_file=None, mass_threshold=None):
    """
    Plot Lilly-Madau diagram for one or more simulations.
    
    Args:
        paths: list of paths to simulation outputs
        labels: list of labels for each simulation
        output: output filename
        halo_file: path to halo particle ID file for filtering
        mass_threshold: manual mass threshold for high-res identification
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'D', 'v']
    
    if labels is None:
        labels = [f'Simulation {i+1}' for i in range(len(paths))]
    
    # Load halo particle IDs if provided
    halo_ids = None
    if halo_file is not None:
        print(f"Loading halo particle IDs from {halo_file}...")
        halo_ids = load_halo_particle_ids(halo_file)
    
    for i, (path, label) in enumerate(zip(paths, labels)):
        if path is None:
            continue
        
        print(f"\nProcessing {label}...")
        snaps = list_snapshots(path)
        print(f"  Found {len(snaps)} snapshots")
        
        zs, sfrs, n_highres_list, volumes = [], [], [], []
        
        for snap in snaps:
            try:
                z, sfr, n_hr, vol = compute_sfr_density_highres(
                    snap, halo_ids, mass_threshold, verbose=False
                )
                zs.append(z)
                sfrs.append(sfr)
                n_highres_list.append(n_hr)
                volumes.append(vol)
            except Exception as e:
                print(f"  Warning: Could not read {snap}: {e}")
                continue
        
        zs = np.array(zs)
        sfrs = np.array(sfrs)
        
        # Sort by redshift (high to low)
        idx = np.argsort(zs)[::-1]
        zs = zs[idx]
        sfrs = sfrs[idx]
        
        # Print summary
        print(f"  Redshift range: z={zs.max():.1f} to {zs.min():.2f}")
        print(f"  SFR density range: {sfrs.min():.2e} to {sfrs.max():.2e} Msun/yr/Mpc^3")
        print(f"  High-res volume: ~{np.mean(volumes):.2f} Mpc^3")
        
        # Plot
        ax.plot(zs, sfrs, marker=markers[i % len(markers)], 
               color=colors[i % len(colors)],
               label=label, linewidth=2, markersize=6, alpha=0.8)
    
    # Observational points
    ax.errorbar(OBS_Z, OBS_SFR, yerr=OBS_ERR, fmt='^', color='black',
                ecolor='gray', elinewidth=1.5, capsize=3, markersize=8,
                label='Madau & Dickinson (2014)', zorder=10)
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'SFR Density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]', fontsize=14)
    ax.set_ylim([1e-4, 1e0])
    ax.set_yscale('log')
    ax.set_xlim([min(max(OBS_Z)+1, 10), -0.1])
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    if halo_file is not None:
        title = 'Cosmic Star Formation History (Single Halo)'
    else:
        title = 'Cosmic Star Formation History (High-Res Region Only)'
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure to {output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Lilly–Madau SFR density from Gadget-4 zoom simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation with automatic high-res detection
  python plot_lilly_madau_zoom.py output_zoom_512/
  
  # Single simulation with halo filtering (RECOMMENDED)
  python plot_lilly_madau_zoom.py output_zoom_512/ --halo-file halo569_2Rvir_particles.txt
  
  # Compare two simulations with halo filtering
  python plot_lilly_madau_zoom.py sim1/ sim2/ --halo-file halo569_2Rvir_particles.txt \\
                                  --labels "No Dust" "With Dust" -o lilly_madau.png
  
  # Custom mass threshold instead of halo filtering
  python plot_lilly_madau_zoom.py output/ --mass-threshold 1e-5 -o lilly_madau.png

Note: Halo filtering is recommended for zoom simulations to ensure only the
      target halo is included in SFR calculations.
        """
    )
    
    parser.add_argument('paths', nargs='*', help='Path(s) to simulation output(s)')
    parser.add_argument('--halo-file', '-f', default=None,
                       help='Text file with halo particle IDs (format: TYPE ID)')
    parser.add_argument('--labels', '-l', nargs='+', default=None,
                       help='Labels for each simulation')
    parser.add_argument('--output', '-o', default=None,
                       help='Output image file')
    parser.add_argument('--mass-threshold', '-m', type=float, default=None,
                       help='Mass threshold for high-res particles (code units)')
    
    args = parser.parse_args()
    
    # Print usage if no paths provided
    if not args.paths:
        parser.print_help()
        print("\n" + "="*70)
        print("QUICK START:")
        print("="*70)
        print("\nMost common usage with halo filtering:")
        print("  python plot_lilly_madau_zoom.py output_zoom_512/ \\")
        print("         --halo-file halo569_2Rvir_particles.txt \\")
        print("         -o lilly_madau.png")
        print("\nWithout halo filtering (uses automatic high-res detection):")
        print("  python plot_lilly_madau_zoom.py output_zoom_512/ -o lilly_madau.png")
        print("\n" + "="*70)
        sys.exit(0)
    
    # Filter out None paths
    paths = [p for p in args.paths if p is not None]
    
    if args.labels and len(args.labels) != len(paths):
        print(f"Warning: Number of labels ({len(args.labels)}) doesn't match "
              f"number of paths ({len(paths)}). Using default labels.")
        args.labels = None
    
    plot_lilly_madau(paths, args.labels, args.output, args.halo_file, args.mass_threshold)
