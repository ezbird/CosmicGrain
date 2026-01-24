import h5py
import glob

files = sorted(glob.glob('../ICs/IC_zoom_1024_halo569_50Mpc_music_ellipsoid.hdf5'))
all_pos = []

for f in files:
    with h5py.File(f, 'r') as hf:
        if 'PartType1' in hf:
            pos = hf['PartType1']['Coordinates'][:]
            all_pos.append(pos)

if all_pos:
    import numpy as np
    all_pos = np.vstack(all_pos)
    for i, dim in enumerate(['x','y','z']):
        span = all_pos[:,i].max() - all_pos[:,i].min()
        print(f"{dim}: {span:.1f} kpc")
