# check_merger_history.py
import h5py
import numpy as np
import pandas as pd

def check_merger_history(snapshot_dir, csv_file, rank):
    """Check if halo had recent major mergers"""
    df = pd.read_csv(csv_file)
    halo_id = int(df.iloc[rank-1]['halo_id'])
    
    # Load multiple snapshots to track mass history
    snapshots = [f'fof_subhalo_tab_{i:03d}.hdf5' for i in range(6, 10)]  # last few snapshots
    
    print(f"Tracking halo {halo_id} across snapshots...")
    
    # Track the most massive progenitor
    # (simplified - real merger trees are more complex)
    
    for snap in snapshots:
        try:
            with h5py.File(f'{snapshot_dir}/{snap}', 'r') as f:
                mass = f['Group']['GroupMass'][halo_id] * 1e10
                z = f['Header'].attrs['Redshift']
                print(f"  z={z:.3f}: M={mass:.2e} Msun")
        except:
            print(f"  Could not read {snap}")
    
if __name__ == '__main__':
    import sys
    check_merger_history(sys.argv[1], sys.argv[2], int(sys.argv[3]))
