# Internal Dust Interfaces

CosmicGrain does not currently expose a stable external library API. This page
is a map of the principal **internal** interfaces useful when modifying the
code. Function names are implementation details and may change.

| Area | Current interface / location | Purpose |
| --- | --- | --- |
| Dust state | `src/data/dust_particle_data.h` | Per-particle grain radius, composition, temperature, birth position, and source tag |
| Dust physics | `src/dust/dust.cc`, `dust.h` | Drag/transport, processing, destruction, creation helpers, diagnostics and synchronization |
| Stellar sources | `src/cooling_sfr/feedback.cc` | SNII/HN, AGB and LRN enrichment and dust-source coupling |
| Star formation / astration | `src/cooling_sfr/starformation.cc` | Dust removal/coupling during gas-to-star conversion |
| Main integration | `src/main/run.cc` | Dust cadence accumulation and calls to `update_dust_dynamics()` |
| Startup safeguards | `src/main/init.cc` | Dust ID initialization, snapshot-restart repair, integrity checks |
| Snapshot I/O | `src/io/snap_io.cc` | Registration of dust HDF5 fields |
| Binary restart I/O | `src/io/restart.cc` | GADGET restart payload plus CosmicGrain cadence state |
| Runtime parameters | `src/data/allvars.cc` | Parameter registration, including dust controls |

## `dust_data`

Each live PartType6 particle stores, in addition to the normal collisionless
particle state:

```cpp
float GrainRadius;          // nm
float CarbonMassFraction;   // 0--1
float DustTemperature;      // K
float BirthPos[3];          // comoving kpc/h
int   DustSource;           // 0=SNII, 1=AGB, 2=LRN
```

The source structure is authoritative for the in-memory representation. HDF5
field names and file precision are defined separately in `snap_io.cc`.

## Creation

`create_dust_particles_from_feedback()` creates PartType6 representatives from
dust mass already partitioned by the stellar-feedback/enrichment machinery.
The creation helper does **not** independently subtract that dust mass from a
gas particle. Keep mass/element partitioning in the feedback layer and particle
spawning in the dust layer distinct when adding a new stellar source.

## Integration

`update_dust_dynamics()` receives the accumulated physical dust interval from
the cadence layer. `dust_global_synchronization()` then reconciles creation,
destruction, mass-change, and global particle bookkeeping.

Before adding a new interface, see [Adding New Physics](adding-new-physics.md)
and [Time Integration](time-integration.md).
