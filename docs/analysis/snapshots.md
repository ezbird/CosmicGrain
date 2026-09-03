# Reading Snapshots

CosmicGrain snapshots are HDF5 and may be split across files:

```text
snapdir_020/snapshot_020.0.hdf5
snapdir_020/snapshot_020.1.hdf5
...
```

Analysis must concatenate every file part and must not assume that one rank's
file contains the complete halo.

## Required header checks

Before measuring a galaxy, record:

- `Time`, `Redshift`, `BoxSize`, and the total particle counts;
- the seven-element particle-type layout;
- cosmology and `HubbleParam`;
- whether the snapshot is comoving; and
- the complete list of fields present for PartTypes 0, 4, and 6.

Positions are comoving \(\mathrm{kpc}/h\), masses use
\(10^{10}\,M_\odot/h\), and physical positions are obtained with \(a/h\).
Use periodic coordinate differences. `GrainRadius` is already written in nm.

## Integrity checks

Use `dust_snapshot_summary.py` for a compact particle/dust inventory,
`cosmicgrain_global_conservation_audit.py` for whole-box closure, and
`check_mass_conservation.py` when comparing the initial and final state.
Any non-finite mass, negative mass, invalid carbon fraction, or mismatch
between dust carbon+silicate mass and total dust mass is a blocking error.
