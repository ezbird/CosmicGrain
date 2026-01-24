import h5py
import numpy as np

# ===== CONFIGURATION =====
halo_id = 569  # Change this to your chosen halo

subfind_file = '../output_parent_50Mpc_128_music_DM_only/fof_subhalo_tab_049.hdf5'
snapshot_file = '../output_parent_50Mpc_128_music_DM_only/snapshot_049.hdf5'

# ===== READ SUBFIND DATA =====
print(f"Extracting particle IDs for Halo {halo_id}")
print("="*60)

with h5py.File(subfind_file, 'r') as f:
    # Get halo information
    subhalo_len = f['Subhalo/SubhaloLenType'][halo_id, 1]  # Type 1 = DM
    subhalo_offset = f['Subhalo/SubhaloOffsetType'][halo_id, 1]
    mass = f['Subhalo/SubhaloMass'][halo_id]
    
    print(f"Halo {halo_id}:")
    print(f"  Mass: {mass:.3e} × 10^10 Msun/h = {mass*1e10:.3e} Msun/h")
    print(f"  Contains {subhalo_len} DM particles")
    print(f"  Global offset: {subhalo_offset}")

# ===== EXTRACT PARTICLE IDs =====
with h5py.File(snapshot_file, 'r') as f:
    # Read all DM particle IDs
    all_dm_ids = f['PartType1/ParticleIDs'][:]
    
    print(f"\nTotal DM particles in snapshot: {len(all_dm_ids)}")
    
    # Extract IDs for this halo
    halo_particle_ids = all_dm_ids[subhalo_offset:subhalo_offset + subhalo_len]

# ===== VERIFY AND SAVE =====
if len(halo_particle_ids) == subhalo_len:
    print(f"✓ Successfully extracted all {len(halo_particle_ids)} particle IDs")
else:
    print(f"✗ WARNING: Expected {subhalo_len} but got {len(halo_particle_ids)}")
    print("Check that snapshot and Subfind catalog match!")

# Save to file
output_file = f'halo_{halo_id}_particles.txt'
np.savetxt(output_file, halo_particle_ids, fmt='%d')

print(f"\nSaved to: {output_file}")
print("="*60)
print(f"\nNext steps:")
print(f"1. Use this file in MUSIC2 config:")
print(f"   region = particle_file")
print(f"   region_particle_file = {output_file}")
