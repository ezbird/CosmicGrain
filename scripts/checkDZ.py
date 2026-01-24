#!/usr/bin/env python3
"""
Quick check of global dust-to-metal ratio
"""
import numpy as np
import h5py
import sys
import glob
import os

def check_global_DZ(snapshot_path):
    """Calculate global dust-to-metal ratio"""
    
    # Find snapshot files
    if os.path.isdir(snapshot_path):
        files = sorted(glob.glob(os.path.join(snapshot_path, "snapshot_*.hdf5")))
    else:
        base = snapshot_path.replace('.0.hdf5', '').replace('.hdf5', '')
        files = sorted(glob.glob(f"{base}.*.hdf5"))
        if not files:
            files = [snapshot_path]
    
    print(f"Reading {len(files)} file(s)...")
    
    total_dust_mass = 0.0
    total_gas_metals = 0.0
    
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            # Dust mass
            if 'PartType6' in f:
                dust_masses = f['PartType6/Masses'][:]
                total_dust_mass += np.sum(dust_masses)
            
            # Gas metal mass
            if 'PartType0' in f:
                gas_masses = f['PartType0/Masses'][:]
                gas_metallicity = f['PartType0/Metallicity'][:]
                total_gas_metals += np.sum(gas_masses * gas_metallicity)
    
    # Calculate D/Z
    if total_gas_metals > 0:
        DZ = total_dust_mass / total_gas_metals
    else:
        DZ = 0.0
    
    # Print results
    print("\n" + "="*50)
    print("GLOBAL DUST-TO-METAL RATIO")
    print("="*50)
    print(f"Total dust mass:       {total_dust_mass:.3e} code units")
    print(f"Total gas metal mass:  {total_gas_metals:.3e} code units")
    print(f"\nD/Z = {DZ:.3f}")
    print(f"\nExpected range: 0.30-0.50 (30-50%)")
    
    if DZ < 0.2:
        print("Status: LOW - Dust underproduced or over-destroyed")
    elif DZ < 0.3:
        print("Status: Slightly low but reasonable")
    elif DZ <= 0.5:
        print("Status: EXCELLENT - Within observed range!")
    else:
        print("Status: HIGH - Dust overproduced or metals depleted")
    print("="*50)
    
    return DZ

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_DZ.py <snapshot_or_dir>")
        print("\nExample:")
        print("  python check_DZ.py snapshot_026.0.hdf5")
        print("  python check_DZ.py snapdir_026/")
        sys.exit(1)
    
    snapshot_path = sys.argv[1]
    check_global_DZ(snapshot_path)
