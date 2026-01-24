import h5py
import numpy as np

subfind_file = '../output_parent_50Mpc_128_music_DM_only/fof_subhalo_tab_049.hdf5'
boxsize = 50000  # kpc/h

with h5py.File(subfind_file, 'r') as f:
    masses = f['Subhalo/SubhaloMass'][:]
    npart = f['Subhalo/SubhaloLenType'][:, 1]
    pos = f['Subhalo/SubhaloPos'][:]
    
    # Find MW-mass halos (2-5 × 10^12 Msun/h with >500 particles)
    mw_mask = (masses > 200) & (masses < 500) & (npart > 500)
    mw_halos = np.where(mw_mask)[0]
    
    print("MW-mass halos (2-5 × 10^12 Msun/h):\n")
    
    for idx in mw_halos:
        m = masses[idx]
        n = npart[idx]
        x, y, z = pos[idx]
        
        # Check distance from edges
        edge_dist = min(x, y, z, boxsize-x, boxsize-y, boxsize-z) / 1000  # Mpc/h
        
        # Find nearest neighbor
        distances = np.sqrt(np.sum((pos - pos[idx])**2, axis=1)) / 1000  # Mpc/h
        distances[idx] = np.inf  # Exclude self
        nearest_dist = distances.min()
        
        print(f"Halo {idx}: {m:.2e} × 10^10 Msun/h, {n} particles")
        print(f"  Position: ({x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f}) Mpc/h")
        print(f"  Edge distance: {edge_dist:.1f} Mpc/h")
        print(f"  Nearest neighbor: {nearest_dist:.1f} Mpc/h")
        print()
