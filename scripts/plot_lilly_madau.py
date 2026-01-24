#!/usr/bin/env python3
"""
Plot the Lilly–Madau star formation history from Gadget-4 snapshots,
supporting both single-file and multi-file (snapdir_*/) snapshot formats.

# Single‐file snapshots in a folder:
python plot_lilly_madau.py /path/to/snap_*.hdf5

# Multi‐file snapshots in snapdir_*/:
python plot_lilly_madau.py /path/to/output/

# Two sets side‐by‐side:
python plot_lilly_madau.py sim1_output/ sim2_output/ -o lilly_madau.png

"""
import os
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


def compute_sfr_density(snapshot_path):
    """
    Returns (redshift, rho_SFR) with rho_SFR in Msun/yr/Mpc^3 (no h factors),
    suitable for direct comparison to Madau & Dickinson (2014).
    """
    import os, glob, h5py, numpy as np

    # Gather part-files
    if os.path.isdir(snapshot_path):
        files = sorted(glob.glob(os.path.join(snapshot_path, '*.hdf5')))
    else:
        files = [snapshot_path]
    if not files:
        raise FileNotFoundError(f"No HDF5 files for snapshot: {snapshot_path}")

    # Header & unit info from first part
    with h5py.File(files[0], 'r') as f0:
        H = f0['Header'].attrs
        P = f0['Parameters'].attrs
        L_box_mpc_h = P['BoxSize']                  
        h = H.get('HubbleParam', 1.0)
        a = H.get('Time', None)
        redshift = H.get('Redshift', (1.0/a - 1.0) if a is not None else None)

        # Code unit definitions
        UnitLength_in_cm   = P['UnitLength_in_cm']
        UnitMass_in_g      = P['UnitMass_in_g']
        UnitVelocity_in_cm = P['UnitVelocity_in_cm_per_s']
        UnitTime_in_s      = UnitLength_in_cm / UnitVelocity_in_cm

    # Convert the *instantaneous* SFR field from code units → Msun/yr
    # SFR_code has units: (code mass / code time)
    # Msun/yr conversion:
    SEC_PER_YEAR = 3.15576e7
    SOLAR_MASS   = 1.98847e33
    sfr_conv = (UnitMass_in_g / SOLAR_MASS) / (UnitTime_in_s / SEC_PER_YEAR)

    total_sfr_msun_per_yr = 0.0
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if 'PartType0' in f and 'StarFormationRate' in f['PartType0']:
                sfr_code = f['PartType0']['StarFormationRate'][:]
                total_sfr_msun_per_yr += np.sum(sfr_code) * sfr_conv

    # Comoving volume in **Mpc^3** (no h): BoxSize is kpc/h → Mpc = (kpc/h)/1000 / h
    #box_mpc = (box_kpc_h / 1.0e3) / h
    #volume_mpc3 = box_mpc**3

    SFR_phys = sfr_code * 0.010223   # Msun / yr
    V_mpc3 = (L_box_mpc_h / h)**3
    rho_sfr = SFR_phys.sum() / V_mpc3   # Msun / yr / Mpc^3


    #rho_sfr = total_sfr_msun_per_yr / volume_mpc3  # Msun/yr/Mpc^3
    return redshift, rho_sfr


def list_snapshots(root):
    """
    Return a sorted list of snapshot identifiers under 'root':
      - If root has subdirs 'snapdir_*', return those dirs;
      - Else if root has *.hdf5 files, return those;
      - Else if root is a file, return [root].
    """
    if os.path.isdir(root):
        dirs = sorted(glob.glob(os.path.join(root, 'snapdir_*')))
        if dirs:
            return dirs
        files = sorted(glob.glob(os.path.join(root, '*.hdf5')))
        if files:
            return files
        raise FileNotFoundError(f"No snapshots found under {root}")
    elif os.path.isfile(root) and root.endswith('.hdf5'):
        return [root]
    else:
        raise FileNotFoundError(f"Invalid snapshot path: {root}")


def plot_lilly_madau(path1, path2=None, path3=None, output=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    for path, label, style in [(path1, 'Simulation 1', '-o'),(path2, 'Simulation 2', '-o'),(path3, 'Simulation 3', '-o')]:
        if not path:
            continue
        snaps = list_snapshots(path)
        zs, sfrs = [], []
        for snap in snaps:
            z, sfr = compute_sfr_density(snap)
            zs.append(z)
            sfrs.append(sfr)
        zs, sfrs = np.array(zs), np.array(sfrs)
        idx = np.argsort(zs)[::-1]
        ax.plot(zs[idx], sfrs[idx], style, label=label, linewidth=1.5)
        print(sfrs[idx])

    # Observational points
    ax.errorbar(OBS_Z, OBS_SFR, yerr=OBS_ERR, fmt='^', color='black',
                ecolor='gray', elinewidth=1.5, capsize=3,
                label='Madau & Dickinson (2014)')

    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel(r'SFR Density (M$_\odot$ yr$^{-1}$ Mpc$^{-3}$)', fontsize=14)
    ax.set_ylim([1e-4, 1e0])  # typical MD14 range
    ax.set_yscale('log')
    ax.set_xlim([max(OBS_Z)+1, 0])
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.title('Lilly–Madau Diagram', fontsize=16)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Lilly–Madau SFR density from Gadget-4 output')
    parser.add_argument('path1', help='Path to simulation 1 snapshots')
    parser.add_argument('path2', nargs='?', default=None,
                        help='Path to simulation 2 snapshots (optional)')
    parser.add_argument('path3', nargs='?', default=None,
                        help='Path to simulation 3 snapshots (optional)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output image file (optional)')
    args = parser.parse_args()

    plot_lilly_madau(args.path1, args.path2, args.path3, args.output)
