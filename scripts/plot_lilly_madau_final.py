#!/usr/bin/env python3
"""
Plot the Lilly–Madau star formation history from Gadget-4 zoom simulations.
Uses halo_utils for proper halo identification and volume calculation.

Usage:
    python plot_lilly_madau_zoom.py /path/to/output/ --catalog groups_049/fof_subhalo_tab_049
    python plot_lilly_madau_zoom.py sim1/ sim2/ --catalog groups_049/fof_subhalo_tab_049 -o lilly_madau.png
"""
import os
import sys
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import halo utilities
try:
    from halo_utils import compute_radial_distance
except ImportError:
    print("Error: Cannot import halo_utils. Make sure halo_utils.py is in your Python path.")
    sys.exit(1)

# Observational data (Madau & Dickinson 2014)
OBS_Z = np.array([0.01, 0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.75,
                  2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5])
OBS_SFR = np.array([0.015, 0.017, 0.03, 0.055, 0.08, 0.10, 0.11,
                    0.12, 0.10, 0.09, 0.07, 0.045, 0.03, 0.01,
                    0.003, 0.001])
OBS_ERR = OBS_SFR * 0.2  # Assume 20% errors


def get_halo_info_from_catalog(catalog_path, target_id=None):
    """
    Extract halo center and properties from FOF catalog.
    Handles both single-file and multi-file catalogs.
    
    Args:
        catalog_path: path to fof_subhalo_tab file (with or without .0.hdf5 or .hdf5)
        target_id: specific subhalo ID, or None for most massive
    
    Returns:
        dict with 'center', 'halfmass_rad', 'mass', 'id'
    """
    # Handle multi-file catalogs
    if not os.path.exists(catalog_path):
        # Try with .*.hdf5 pattern
        base = catalog_path.replace('.hdf5', '')
        catalog_files = sorted(glob.glob(f'{base}.*.hdf5'))
        
        if not catalog_files:
            raise FileNotFoundError(f"Cannot find catalog: {catalog_path}")
        
        print(f"  Found {len(catalog_files)} catalog files")
    else:
        # Single file
        catalog_files = [catalog_path]
    
    # If we want most massive, we need to scan all files
    if target_id is None:
        print("  Finding most massive subhalo across all catalog files...")
        max_mass = -1
        best_file = None
        best_id = None
        
        for cat_file in catalog_files:
            with h5py.File(cat_file, 'r') as cat:
                if 'Subhalo' not in cat or 'SubhaloMass' not in cat['Subhalo']:
                    continue
                masses = cat['Subhalo']['SubhaloMass'][:]
                if len(masses) == 0:
                    continue
                local_max_id = np.argmax(masses)
                local_max_mass = masses[local_max_id]
                
                if local_max_mass > max_mass:
                    max_mass = local_max_mass
                    best_file = cat_file
                    best_id = local_max_id
        
        if best_file is None:
            raise ValueError("No subhalos found in catalog!")
        
        catalog_path = best_file
        target_id = best_id
        print(f"  Most massive subhalo: ID {target_id} in {os.path.basename(best_file)}")
    else:
        # For specific target_id, try first file (often works for target halo)
        catalog_path = catalog_files[0]
        print(f"  Using target subhalo ID {target_id} from {os.path.basename(catalog_path)}")
    
    # Now load the halo info
    with h5py.File(catalog_path, 'r') as cat:
        halo_info = {
            'id': target_id,
            'center': cat['Subhalo']['SubhaloPos'][target_id],
            'halfmass_rad': cat['Subhalo']['SubhaloHalfmassRad'][target_id],
            'mass': cat['Subhalo']['SubhaloMass'][target_id],
            'vmax': cat['Subhalo']['SubhaloVmax'][target_id]
        }
    
    return halo_info


def estimate_virial_radius(halo_info, redshift):
    """
    Estimate virial radius from Vmax using Bryan & Norman (1998).
    
    Rvir ≈ Vmax / (10 * H(z))
    
    Args:
        halo_info: dict with 'vmax' in km/s
        redshift: redshift of snapshot
    
    Returns:
        Rvir in kpc (physical)
    """
    # Cosmological parameters (should match your simulation)
    h = 0.6774
    Omega_m = 0.3089
    
    # Hubble parameter at redshift z
    E_z = np.sqrt(Omega_m * (1 + redshift)**3 + (1 - Omega_m))
    H_z = 100 * h * E_z  # km/s/Mpc
    
    # Convert to km/s/kpc
    H_z_kpc = H_z / 1000.0
    
    # Rvir from Vmax (Klypin et al. 2011 fitting formula)
    vmax = halo_info['vmax']  # km/s
    rvir_pkpc = vmax / (10.0 * H_z_kpc)  # physical kpc
    
    return rvir_pkpc


def compute_sfr_density_halo(snapshot_path, halo_center, search_radius_pkpc, verbose=False):
    """
    Compute SFR density within a spherical region around halo center.
    
    Args:
        snapshot_path: path to snapshot directory or file
        halo_center: halo center position [x, y, z] in comoving kpc/h
        search_radius_pkpc: search radius in physical kpc
        verbose: print diagnostic info
    
    Returns:
        (redshift, rho_SFR, n_gas, volume_mpc3, scale_factor)
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
    
    # Convert search radius: physical kpc → comoving kpc/h
    search_radius_ckpc_h = search_radius_pkpc * (1 + redshift) * h
    
    # Collect all gas data
    all_sfr = []
    all_positions = []
    
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if 'PartType0' not in f:
                continue
            
            pos = f['PartType0']['Coordinates'][:]
            
            if 'StarFormationRate' in f['PartType0']:
                sfr = f['PartType0']['StarFormationRate'][:]
            else:
                sfr = np.zeros(len(pos))
            
            all_positions.append(pos)
            all_sfr.append(sfr)
    
    # Concatenate all particles
    all_positions = np.concatenate(all_positions)
    all_sfr = np.concatenate(all_sfr)
    
    # Find particles within search radius
    r = compute_radial_distance(all_positions, halo_center)
    halo_mask = r < search_radius_ckpc_h
    
    # Filter to halo particles only
    sfr_halo = all_sfr[halo_mask]
    n_gas = np.sum(halo_mask)
    
    # Total SFR from halo particles (Msun/yr)
    total_sfr_msun_per_yr = np.sum(sfr_halo) * sfr_conv
    
    # Volume: spherical within search_radius
    # Convert radius: physical kpc → Mpc
    radius_mpc = search_radius_pkpc / 1000.0
    volume_mpc3 = (4.0/3.0) * np.pi * radius_mpc**3
    
    # SFR density (Msun/yr/Mpc^3)
    rho_sfr = total_sfr_msun_per_yr / volume_mpc3
    
    if verbose:
        print(f"  z={redshift:.2f}: {n_gas} gas particles in halo")
        print(f"  Search radius: {search_radius_pkpc:.2f} pkpc = {search_radius_ckpc_h:.2f} ckpc/h")
        print(f"  Volume: {volume_mpc3:.4f} Mpc^3")
        print(f"  SFR: {total_sfr_msun_per_yr:.2e} Msun/yr")
        print(f"  rho_SFR: {rho_sfr:.2e} Msun/yr/Mpc^3")
    
    return redshift, rho_sfr, n_gas, volume_mpc3, a


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


def plot_lilly_madau(paths, labels=None, output=None, catalog_ref=None, 
                     radius_factor=2.0, target_id=None):
    """
    Plot Lilly-Madau diagram for one or more simulations.
    
    Args:
        paths: list of paths to simulation outputs
        labels: list of labels for each simulation
        output: output filename
        catalog_ref: reference catalog to get halo center/size (e.g., z=0 catalog)
        radius_factor: multiplier for virial radius
        target_id: specific subhalo ID (None = most massive)
    """
    if catalog_ref is None:
        raise ValueError("Must specify --catalog for halo identification")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    markers = ['o', 's', '^', 'D', 'v']
    
    if labels is None:
        labels = [f'Simulation {i+1}' for i in range(len(paths))]
    
    # Get halo info from reference catalog
    print(f"Loading halo info from {catalog_ref}...")
    halo_info = get_halo_info_from_catalog(catalog_ref, target_id)
    print(f"  Target halo {halo_info['id']}")
    print(f"  Center: {halo_info['center']}")
    print(f"  Halfmass radius: {halo_info['halfmass_rad']:.2f} kpc")
    print(f"  Vmax: {halo_info['vmax']:.2f} km/s")
    print()
    
    for i, (path, label) in enumerate(zip(paths, labels)):
        if path is None:
            continue
        
        print(f"Processing {label}...")
        snaps = list_snapshots(path)
        print(f"  Found {len(snaps)} snapshots")
        
        zs, sfrs, n_gas_list, volumes = [], [], [], []
        
        for snap in snaps:
            try:
                # Extract snapshot number from path
                # e.g., snapdir_025 → 025, snapshot_025.hdf5 → 025
                if 'snapdir_' in snap:
                    snap_num = snap.split('snapdir_')[-1].split('/')[0]
                elif 'snapshot_' in snap:
                    snap_num = snap.split('snapshot_')[-1].split('.')[0].split('_')[0]
                else:
                    print(f"  Warning: Cannot parse snapshot number from {snap}")
                    continue
                
                # Find corresponding catalog
                catalog_dir = os.path.dirname(catalog_ref)
                catalog_base = os.path.basename(catalog_ref).split('_')[0]  # 'groups' or 'fof'
                catalog_pattern = f"{catalog_base}_{snap_num}/fof_subhalo_tab_{snap_num}"
                catalog_this_snap = os.path.join(catalog_dir.replace('groups_049', f'groups_{snap_num}'), 
                                                f'fof_subhalo_tab_{snap_num}')
                
                # Check multiple possible catalog locations
                possible_catalogs = [
                    catalog_this_snap,
                    os.path.join(os.path.dirname(path), f'groups_{snap_num}', f'fof_subhalo_tab_{snap_num}'),
                    os.path.join(path, f'groups_{snap_num}', f'fof_subhalo_tab_{snap_num}')
                ]
                
                catalog_found = None
                for cat in possible_catalogs:
                    if os.path.exists(cat + '.hdf5') or os.path.exists(cat + '.0.hdf5'):
                        catalog_found = cat
                        break
                
                if catalog_found is None:
                    print(f"  Warning: No catalog for snapshot {snap_num}, skipping")
                    continue
                
                # ✅ CORRECT: Load halo info at THIS redshift
                halo_info_snap = get_halo_info_from_catalog(catalog_found, target_id)
                
                # Read redshift
                if os.path.isdir(snap):
                    test_file = sorted(glob.glob(os.path.join(snap, '*.hdf5')))[0]
                else:
                    test_file = snap
                
                with h5py.File(test_file, 'r') as f:
                    z_snap = f['Header'].attrs['Redshift']
                
                # Use Rvir from this snapshot's catalog
                rvir_pkpc = estimate_virial_radius(halo_info_snap, z_snap)
                search_radius_pkpc = radius_factor * rvir_pkpc
                
                z, sfr, n_gas, vol, a = compute_sfr_density_halo(
                    snap, halo_info_snap['center'], search_radius_pkpc, verbose=False  # ← Now uses correct center!
                )
                zs.append(z)
                sfrs.append(sfr)
                n_gas_list.append(n_gas)
                volumes.append(vol)
                
            except Exception as e:
                print(f"  Warning: Could not read {snap}: {e}")
                continue
        
        zs = np.array(zs)
        sfrs = np.array(sfrs)
        volumes = np.array(volumes)
        
        # Sort by redshift (high to low)
        idx = np.argsort(zs)[::-1]
        zs = zs[idx]
        sfrs = sfrs[idx]
        volumes = volumes[idx]
        
        # Print summary
        print(f"  Redshift range: z={zs.max():.1f} to {zs.min():.2f}")
        print(f"  SFR density range: {sfrs.min():.2e} to {sfrs.max():.2e} Msun/yr/Mpc^3")
        print(f"  Halo volume: ~{np.mean(volumes):.4f} Mpc^3 (mean, {radius_factor}×Rvir)")
        print()
        
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
    
    ax.set_title(f'Cosmic Star Formation History (Halo {halo_info["id"]}, {radius_factor}×Rvir)', 
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Lilly–Madau SFR density from Gadget-4 zoom simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation (catalog without .hdf5 or with it both work)
  python plot_lilly_madau_zoom.py output_zoom/ --catalog groups_049/fof_subhalo_tab_049
  
  # Or with full path including .hdf5
  python plot_lilly_madau_zoom.py output_zoom/ --catalog groups_049/fof_subhalo_tab_049.hdf5
  
  # Compare two simulations
  python plot_lilly_madau_zoom.py sim1/ sim2/ \\
         --catalog groups_049/fof_subhalo_tab_049 \\
         --labels "No Dust" "With Dust" -o lilly_madau.png
  
  # Use 3×Rvir instead of 2×Rvir
  python plot_lilly_madau_zoom.py output/ \\
         --catalog groups_049/fof_subhalo_tab_049 --radius-factor 3.0
        """
    )
    
    parser.add_argument('paths', nargs='*', help='Path(s) to simulation output(s)')
    parser.add_argument('--catalog', '-c', required=True,
                       help='FOF catalog base path (e.g., groups_049/fof_subhalo_tab_049)')
    parser.add_argument('--target-id', '-i', type=int, default=None,
                       help='Specific subhalo ID (default: most massive)')
    parser.add_argument('--radius-factor', '-r', type=float, default=2.0,
                       help='Multiplier for virial radius (default: 2.0)')
    parser.add_argument('--labels', '-l', nargs='+', default=None,
                       help='Labels for each simulation')
    parser.add_argument('--output', '-o', default=None,
                       help='Output image file')
    
    args = parser.parse_args()
    
    # Print usage if no paths provided
    if not args.paths:
        parser.print_help()
        print("\n" + "="*70)
        print("QUICK START:")
        print("="*70)
        print("\nBasic usage:")
        print("  python plot_lilly_madau_zoom.py output_zoom_512/ \\")
        print("         --catalog groups_049/fof_subhalo_tab_049")
        print("\nWith output file:")
        print("  python plot_lilly_madau_zoom.py output_zoom_512/ \\")
        print("         --catalog groups_049/fof_subhalo_tab_049 \\")
        print("         -o lilly_madau.png")
        print("\n" + "="*70)
        sys.exit(0)
    
    # Filter out None paths
    paths = [p for p in args.paths if p is not None]
    
    if args.labels and len(args.labels) != len(paths):
        print(f"Warning: Number of labels ({len(args.labels)}) doesn't match "
              f"number of paths ({len(paths)}). Using default labels.")
        args.labels = None
    
    plot_lilly_madau(paths, args.labels, args.output, args.catalog, 
                     args.radius_factor, args.target_id)
