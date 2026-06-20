#!/usr/bin/env python3
"""
Add empty PartType6 (dust) to MUSIC2/Gadget4 initial conditions
Handles various IC formats with or without optional attributes
Usage: python add_dust_type.py IC_zoom_2Mpc_music.hdf5
"""

import h5py
import numpy as np
import sys
import shutil

def add_dust_type_to_ics(input_file, output_file=None):
    """
    Add empty PartType6 group to initial conditions for dust support
    
    Parameters:
    - input_file: Original IC file
    - output_file: Output filename (default: adds '_with_dust' to input name)
    """
    
    if output_file is None:
        output_file = input_file.replace('.hdf5', '_with_dust.hdf5')
    
    print(f"Adding PartType6 (dust) support to {input_file}")
    print(f"Output: {output_file}")
    print("="*60)
    
    # Copy original file
    print("Creating backup copy...")
    shutil.copy2(input_file, output_file)
    
    # Open the copied file for modification
    with h5py.File(output_file, 'r+') as f:
        print(f"Original particle types: {[key for key in f.keys() if key.startswith('PartType')]}")
        
        # Read header to get current particle counts
        header = f['Header']
        
        # Get current attributes (handle different formats)
        print("\nReading header attributes...")
        
        # NumPart_ThisFile (required)
        if 'NumPart_ThisFile' in header.attrs:
            npart = header.attrs['NumPart_ThisFile'][:]
        else:
            print("ERROR: NumPart_ThisFile not found in header!")
            sys.exit(1)
        
        # NumPart_Total (required)
        if 'NumPart_Total' in header.attrs:
            npart_total = header.attrs['NumPart_Total'][:]
        else:
            print("WARNING: NumPart_Total not found, using NumPart_ThisFile")
            npart_total = npart.copy()
        
        # NumPart_Total_HighWord (optional - for very large simulations)
        if 'NumPart_Total_HighWord' in header.attrs:
            npart_total_highword = header.attrs['NumPart_Total_HighWord'][:]
            has_highword = True
        else:
            print("Note: NumPart_Total_HighWord not present (not needed for most simulations)")
            npart_total_highword = np.zeros_like(npart_total)
            has_highword = False
        
        # MassTable (required)
        if 'MassTable' in header.attrs:
            mass_table = header.attrs['MassTable'][:]
        else:
            print("WARNING: MassTable not found, creating zero array")
            mass_table = np.zeros(len(npart), dtype=np.float64)
        
        print(f"Current NumPart_ThisFile: {npart}")
        print(f"Current NumPart_Total: {npart_total}")
        print(f"Current MassTable: {mass_table}")
        
        # Check current array length
        current_ntypes = len(npart)
        print(f"\nCurrent number of particle types: {current_ntypes}")
        
        # Only extend if we have exactly 6 types (standard Gadget)
        if current_ntypes == 6:
            print("Extending particle arrays from 6 to 7 types for dust support...")
            
            # Extend arrays to include Type 6 (dust) with 0 particles
            npart_new = np.zeros(7, dtype=npart.dtype)
            npart_new[:6] = npart
            npart_new[6] = 0  # No dust particles initially
            
            npart_total_new = np.zeros(7, dtype=npart_total.dtype)
            npart_total_new[:6] = npart_total
            npart_total_new[6] = 0
            
            npart_total_highword_new = np.zeros(7, dtype=npart_total_highword.dtype)
            npart_total_highword_new[:6] = npart_total_highword
            npart_total_highword_new[6] = 0
            
            mass_table_new = np.zeros(7, dtype=mass_table.dtype)
            mass_table_new[:6] = mass_table
            mass_table_new[6] = 0.0  # Dust mass will be individual particle masses
            
            # Update header attributes
            del header.attrs['NumPart_ThisFile']
            del header.attrs['NumPart_Total']
            del header.attrs['MassTable']
            
            header.attrs.create('NumPart_ThisFile', npart_new)
            header.attrs.create('NumPart_Total', npart_total_new)
            header.attrs.create('MassTable', mass_table_new)
            
            # Only update HighWord if it existed
            if has_highword:
                del header.attrs['NumPart_Total_HighWord']
                header.attrs.create('NumPart_Total_HighWord', npart_total_highword_new)
            
            print(f"Updated NumPart_ThisFile: {npart_new}")
            print(f"Updated NumPart_Total: {npart_total_new}")
            print(f"Updated MassTable: {mass_table_new}")
            
        elif current_ntypes == 7:
            print("Arrays already support 7 particle types")
        else:
            print(f"WARNING: Unusual number of types ({current_ntypes}), not modifying arrays")
        
        # Create empty PartType6 group if it doesn't exist
        if 'PartType6' not in f:
            print("\nCreating empty PartType6 group for dust particles...")
            parttype6 = f.create_group('PartType6')
            
            # Create empty datasets with proper data types
            # These will be populated when dust particles are created during the simulation
            empty_coords = np.empty((0, 3), dtype=np.float64)
            empty_vels = np.empty((0, 3), dtype=np.float64) 
            empty_ids = np.empty((0,), dtype=np.uint64)
            empty_masses = np.empty((0,), dtype=np.float64)
            
            parttype6.create_dataset('Coordinates', data=empty_coords, maxshape=(None, 3))
            parttype6.create_dataset('Velocities', data=empty_vels, maxshape=(None, 3))
            parttype6.create_dataset('ParticleIDs', data=empty_ids, maxshape=(None,))
            parttype6.create_dataset('Masses', data=empty_masses, maxshape=(None,))
            
            print("Created empty PartType6 datasets:")
            print("  - Coordinates (0, 3)")
            print("  - Velocities (0, 3)")
            print("  - ParticleIDs (0,)")
            print("  - Masses (0,)")
        else:
            print("\nPartType6 already exists")
        
        # Verify final state
        print("\n" + "="*60)
        print("VERIFICATION:")
        print(f"Final particle types: {[key for key in f.keys() if key.startswith('PartType')]}")
        
        # Count particles
        total_particles = 0
        for i in range(7):
            ptype_key = f'PartType{i}'
            if ptype_key in f:
                if 'Coordinates' in f[ptype_key]:
                    n = len(f[ptype_key]['Coordinates'])
                    if n > 0:
                        print(f"  {ptype_key}: {n:,} particles")
                        total_particles += n
        
        print(f"Total particles: {total_particles:,}")
    
    print("\n" + "="*60)
    print(f"SUCCESS! Modified IC file ready: {output_file}")
    print("\nIMPORTANT: Make sure your Gadget4 compilation uses:")
    print("  NTYPES=7")
    print("  Config.sh should have: NTYPES=7")
    print("\nUpdate your parameter file to use:")
    print(f"  InitCondFile = {output_file}")
    
    return output_file

def verify_ic_file(filename):
    """Verify the IC file has proper dust support"""
    print(f"\nVerifying {filename}...")
    
    with h5py.File(filename, 'r') as f:
        # Check header
        if 'Header' in f:
            header = f['Header']
            if 'NumPart_ThisFile' in header.attrs:
                npart = header.attrs['NumPart_ThisFile']
                print(f"  NumPart array length: {len(npart)}")
                if len(npart) >= 7:
                    print("  ✓ Supports 7 particle types")
                else:
                    print("  ✗ Only supports {} particle types".format(len(npart)))
        
        # Check for PartType6
        if 'PartType6' in f:
            print("  ✓ PartType6 group exists")
            pt6 = f['PartType6']
            for dset in ['Coordinates', 'Velocities', 'ParticleIDs', 'Masses']:
                if dset in pt6:
                    print(f"    ✓ {dset} dataset present")
                else:
                    print(f"    ✗ {dset} dataset missing")
        else:
            print("  ✗ PartType6 group missing")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_dust_type.py IC_file.hdf5 [output_file.hdf5]")
        print("\nThis script adds PartType6 support for dust particles to Gadget4 ICs")
        print("If no output file is specified, '_with_dust' is appended to the input name")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_file = add_dust_type_to_ics(input_file, output_file)
        verify_ic_file(output_file)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)