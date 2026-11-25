# Dust Particle Fields

CosmicGrain stores dust as explicit `PartType6` superparticles. The current
build (`src/io/snap_io.cc`) writes:

| Field | Units / meaning |
| --- | --- |
| `Coordinates`, `Velocities`, `ParticleIDs`, `Masses` | Standard GADGET-4 conventions |
| `GrainRadius` | Grain radius, nm |
| `CarbonMassFraction` | Carbonaceous mass fraction, 0–1; the rest is silicate |
| `DustTemperature` | K |
| `DustSource` | Birth channel: 0=SNII, 1=AGB, 2=LRN |
| `DustFormationTime` | Scale factor at creation |
| `BirthPos` | Parent star position at creation, comoving kpc/h |

Gas (`PartType0`) carries `Metallicity` and the gas-phase element fractions
`GasCarbonMassFraction`, `GasNitrogenMassFraction`, `GasOxygenMassFraction`,
`GasNeonMassFraction`, `GasMagnesiumMassFraction`, `GasSiliconMassFraction`,
`GasIronMassFraction`. Older runs may lack some of these, so check the fields
present before analysis.

Destruction events are not in the snapshots; each task writes them to
`<OutputDir>/dust_logs/dust_log_task<N>.txt` (ID, birth and event scale
factor and position, mass, radius, carbon fraction, gas density, event type).
The column list is in the header of `src/dust/dust_particle_log.cc`.

## Units and metadata

Snapshot metadata should be read from the appropriate HDF5 groups. In the current CosmicGrain/GADGET-4 output convention, **`HubbleParam` is stored in `Parameters`, not `Header`**. Record the scale factor/redshift and unit convention before converting positions, masses, or velocities.

`GrainRadius` is written in nm with no unit conversion. Confirm units when analyzing older or externally produced snapshots.

## Surviving particles

Some analyses need to exclude destroyed or effectively zero-mass numerical remnants. Apply one documented criterion consistently across all compared samples. Never allow a tiny residual-mass population to dominate a number-weighted histogram by accident.

## Reproducibility

For derived catalogs, archive the snapshot, halo/group identifier, centering method, aperture, weighting, field cuts, and code version or analysis-script revision.
