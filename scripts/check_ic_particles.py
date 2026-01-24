#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Check particle types and positions in a Gadget/MUSIC HDF5 IC file."
    )
    parser.add_argument(
        "ic_file", 
        type=str,
        help="Path to the IC HDF5 file (e.g., IC_uniform_50Mpc_256_music.hdf5)"
    )
    args = parser.parse_args()

    # Open file
    try:
        f = h5py.File(args.ic_file, 'r')
    except OSError:
        print(f"Error: could not open file '{args.ic_file}'. Check the path.")
        sys.exit(1)

    with f:
        print(f"\nOpened IC file: {args.ic_file}")
        print("Particle types present:\n")

        # Loop through PartType0–5
        for i in range(7):
            key = f'PartType{i}'
            if key in f:
                group = f[key]

                # Count particles
                npart = len(group['ParticleIDs'])

                # Mass: prefer per-particle table, else use MassTable[i]
                if 'Masses' in group:
                    mass = group['Masses'][0]
                else:
                    mass = f['Header'].attrs['MassTable'][i]

                print(f"  Type {i}: {npart:,} particles, mass = {mass:.3e}")

                # Position distribution
                pos = group['Coordinates'][:]
                print("    Position range:")
                print(f"      x = [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
                print(f"      y = [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
                print(f"      z = [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]\n")

        print("Done.\n")


if __name__ == "__main__":
    main()
