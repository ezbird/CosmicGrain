# Source Tree

CosmicGrain is the full GADGET-4 tree with dust and feedback code added. The
files below are the ones a dust or feedback change usually touches; everything
else is standard GADGET-4. Files with suffixes such as `_old`, `_original`,
`.pre_*` are kept for reference and are not compiled (see `OBJS` in the
`Makefile`).

| Path | Contents |
| --- | --- |
| `src/dust/dust.cc`, `dust.h` | All dust processes: creation helper, drag, growth, coagulation, shattering, sputtering, SN shocks, astration, radiation pressure, temperature, D/G grid, dust–gas association, global synchronization |
| `src/dust/dust_particle_log.cc` | Per-task destruction-event log in `<OutputDir>/dust_logs/` and the coagulation density histogram |
| `src/data/dust_particle_data.h` | `dust_data`: per-grain `GrainRadius`, `CarbonMassFraction`, `DustTemperature`, `BirthPos`, `DustSource` |
| `src/data/allvars.cc`, `allvars.h` | Runtime parameter registration, including all `Dust*` parameters |
| `src/cooling_sfr/feedback.cc` | SNII, AGB and LRN feedback, enrichment, dust-mass partitioning and dust spawning calls |
| `src/cooling_sfr/snii_yields.cc`, `agb_yields.cc` | Yield tables |
| `src/cooling_sfr/starformation.cc` | Star formation and dust astration calls |
| `src/cooling_sfr/cg_distributed.h` | Collective search pattern (allgather events, local tree search, exchange) used by feedback and dust so neighbour sets do not depend on the domain decomposition |
| `src/cooling_sfr/rc1_gas_geometry.h`, `rc1_gas_index.h`, `agb_distributed_recipients.h`, `agb_feedback_math.h` | Helpers for the distributed AGB recipient selection |
| `src/cooling_sfr/spatial_hash_zoom.h` | Zoom-aware sparse spatial hash for gas, star and dust neighbour lookups |
| `src/main/run.cc` | Main loop, startup checks, dust cadence (`advance_dust_physics()`) |
| `src/main/init.cc` | Dust ID repair and integrity checks on snapshot restart |
| `src/main/energy_ledger.h` | Gas thermal/kinetic energy ledger per main-loop phase |
| `src/time_integration/timestep_limiter.h`, `kicks.cc` | SPH timestep limiter with wake-up |
| `src/sph/hydro_watch.h`, `density.cc` | Runaway-force watchdog and density-jump flags |
| `src/io/snap_io.cc` | HDF5 snapshot fields, including PartType6 |
| `src/io/restart.cc` | Restart files plus the dust cadence state |

When moving existing routines, keep code motion and physics changes in
separate commits, and check that a pure move reproduces the same diagnostics.
