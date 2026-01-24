import h5py
import numpy as np

# ===== CONFIGURATION =====
halo_id = 569
particle_id_file = 'halo_569_particles.txt'
parent_ic_file = '../ICs/IC_parent_50Mpc_128_music.hdf5'

# ===== READ PARTICLE IDs =====
print(f"Reading particle IDs from {particle_id_file}")
particle_ids = np.loadtxt(particle_id_file, dtype=np.int64)

# ===== READ PARENT ICs =====
print(f"Reading parent ICs: {parent_ic_file}")

with h5py.File(parent_ic_file, 'r') as f:
    all_ids = f['PartType1/ParticleIDs'][:]
    all_pos = f['PartType1/Coordinates'][:]
    boxsize = f['Header'].attrs['BoxSize']
    
    print(f"Box size: {boxsize} Mpc/h")
    
    # Find our halo particles
    indices = np.isin(all_ids, particle_ids)
    halo_positions = all_pos[indices]
    
    # NORMALIZE to [0, 1]
    halo_positions_norm = halo_positions / boxsize
    
    print(f"Found {len(halo_positions)} particles")
    print(f"\nNormalized position ranges:")
    print(f"  X: [{halo_positions_norm[:,0].min():.6f}, {halo_positions_norm[:,0].max():.6f}]")
    print(f"  Y: [{halo_positions_norm[:,1].min():.6f}, {halo_positions_norm[:,1].max():.6f}]")
    print(f"  Z: [{halo_positions_norm[:,2].min():.6f}, {halo_positions_norm[:,2].max():.6f}]")

# Save normalized positions
output_file = f'halo_{halo_id}_IC_positions_normalized.txt'
np.savetxt(output_file, halo_positions_norm, fmt='%.8f')

print(f"\nSaved normalized positions to: {output_file}")
