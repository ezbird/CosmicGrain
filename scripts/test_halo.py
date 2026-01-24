import h5py
import numpy as np

target_pos = np.array([25.385, 15.568, 32.797])  # Halo 378

files = [
    '../output_parent_50Mpc_128_music/fof_subhalo_tab_010.hdf5',  # z=0.75
    '../output_parent_50Mpc_128_music/fof_subhalo_tab_011.hdf5',  # z=0.50
    '../output_parent_50Mpc_128_music/fof_subhalo_tab_012.hdf5',  # z=0.25
]

print("Tracking Halo 378 across snapshots:")
print(f"{'Snapshot':>10s} {'z':>6s} {'Halo ID':>8s} {'Distance':>10s} {'Mass':>12s}")
print("-" * 48)

for fname in files:
    with h5py.File(fname, 'r') as f:
        pos = f['Group/GroupPos'][:] / 1000.0 / 0.6732  # Convert to Mpc/h
        mass = f['Group/GroupMass'][:] * 1e10 / 0.6732  # Convert to Msun
        z = f['Header'].attrs['Redshift']
        
        # Find closest halo
        dist = np.sqrt(np.sum((pos - target_pos)**2, axis=1))
        idx = np.argmin(dist)
        
        snap_num = fname.split('_')[-1].split('.')[0]
        print(f"{snap_num:>10s} {z:>6.2f} {idx:>8d} {dist[idx]:>9.3f} Mpc {mass[idx]:>11.2e} M☉")

print("\n✅ If distance <1 Mpc and mass stable → Halo is stable!")
print("❌ If distance >1 Mpc or mass changes >20% → Pick different halo")
