#!/usr/bin/env python3
"""
Plot histograms of SPH particle densities and temperatures from a Gadget-4 snapshot,
with three panels: mass density (g/cm^3), number density (cm^-3), and temperature (K),
supporting both single-file and multi-file (snapdir_*/) formats.
"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Physical constants
MPC_IN_CM = 3.085678e24      # 1 Mpc in cm
M_SUN_IN_G = 1.989e33        # Solar mass in g
M_PROTON = 1.6726219e-24     # Proton mass in g
MU = 1.2                     # Mean molecular weight (approximate)
N_THRESH = 0.1               # cm^-3, default SH03 number-density threshold
GAMMA = 5.0/3.0              # Adiabatic index
K_BOLTZ = 1.380649e-16       # Boltzmann constant in erg/K

# Unit conversion defaults
def compute_units(header):
    UnitMass = header.get('UnitMass_in_g', M_SUN_IN_G * 1e10)
    UnitLength = header.get('UnitLength_in_cm', 1e21)
    UnitVel = header.get('UnitVelocity_in_cm_per_s', 1e5)  # default 1 km/s
    return UnitMass, UnitLength, UnitVel


def get_snapshot_paths(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.hdf5')))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {path}")
    return files


def extract_sph_data(files):
    densities = []
    internal_u = []
    header = None
    for fn in files:
        with h5py.File(fn, 'r') as f:
            if header is None:
                header = dict(f['Header'].attrs)
            if 'PartType0' in f:
                part0 = f['PartType0']
                if 'Density' in part0:
                    densities.append(part0['Density'][:])
                if 'InternalEnergy' in part0:
                    internal_u.append(part0['InternalEnergy'][:])
    if not densities:
        raise RuntimeError("No SPH densities found in snapshot")
    dens = np.concatenate(densities)
    u_code = np.concatenate(internal_u) if internal_u else None
    return dens, u_code, header


def main():
    parser = argparse.ArgumentParser(
        description="Three-panel histograms: mass density, number density, temperature.")
    parser.add_argument('snapshot', help='Path to snapshot file or snapdir_*/ folder')
    parser.add_argument('--output', help='Output image file (optional)', default=None)
    parser.add_argument('--bins', type=int, help='Number of histogram bins', default=50)
    args = parser.parse_args()

    files = get_snapshot_paths(args.snapshot)
    dens_code, u_code, header = extract_sph_data(files)

    # Read scale factor and compute redshift
    a = header.get('Time', 1.0)
    z = 1.0/a - 1.0

    # Convert to physical mass density [g/cm^3], including cosmic factor
    UnitMass, UnitLength, UnitVel = compute_units(header)
    dens_mass = dens_code * UnitMass / UnitLength**3 / a**3

    # Convert to number density [cm^-3]
    dens_number = dens_mass / (MU * M_PROTON)

    # Convert internal energy to temperature if available
    if u_code is not None:
        # u_code has units of (velocity)^2
        u_phys = u_code * UnitVel**2        # erg/g
        # Temperature: u = k_B T / ((gamma-1) mu m_p)  => T = u * (gamma-1) mu m_p / k_B
        temps = u_phys * (GAMMA - 1.0) * MU * M_PROTON / K_BOLTZ
    else:
        temps = None

    # Create three-panel figure
    ncols = 3 if temps is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    # Panel 1: Mass density
    ax = axes[0]
    ax.hist(dens_mass, bins=args.bins, log=True)
    ax.set_xscale('log')
    ax.set_xlabel('Mass density [g/cm$^3$]')
    ax.set_ylabel('Number of particles')
    ax.set_title('Mass Density')

    # Panel 2: Number density
    ax = axes[1]
    ax.hist(dens_number, bins=args.bins, log=True)
    ax.set_xscale('log')
    ax.axvline(N_THRESH, color='red', linestyle='--',
               label=f'SH03 n$_{{th}}$={N_THRESH} cm$^{{-3}}$')
    ax.legend()
    ax.set_xlabel('Number density [cm$^{-3}$]')
    ax.set_title('Number Density')

    # Panel 3: Temperature, if computed
    if temps is not None:
        ax = axes[2]
        ax.hist(temps, bins=args.bins, log=True)
        ax.set_xscale('log')
        ax.set_xlabel('Temperature [K]')
        ax.set_title('Temperature')

    # Supertitle with redshift
    fig.suptitle(f"z = {z:.2f} — SPH Distributions ({os.path.basename(args.snapshot)})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
