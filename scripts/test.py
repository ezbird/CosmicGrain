import h5py
import numpy as np
import glob

# Get all snapshot files
files = sorted(glob.glob('../6_output_zoom_512_halo569_50Mpc_dust_metal_return/snapdir_049/snapshot_049.*.hdf5'))

# Collect all type 1 positions
all_type1_pos = []

for f in files:
    with h5py.File(f, 'r') as hf:
        if 'PartType1' in hf and 'Coordinates' in hf['PartType1']:
            pos = hf['PartType1']['Coordinates'][:]
            all_type1_pos.append(pos)

if all_type1_pos:
    all_pos = np.vstack(all_type1_pos)
    print(f"Total Type 1 particles: {len(all_pos)}")
    for i, dim in enumerate(['x', 'y', 'z']):
        print(f"  {dim} = [{all_pos[:,i].min():.3f}, {all_pos[:,i].max():.3f}] -> span = {all_pos[:,i].max() - all_pos[:,i].min():.3f} kpc")
