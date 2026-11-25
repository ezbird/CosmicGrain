# Code Architecture

CosmicGrain adds an explicit dust particle population (`PartType6`) to the GADGET-4 simulation framework. Dust physics necessarily touches several host-code systems: particle storage, gravity, neighbor searches, time integration, stellar feedback/enrichment, domain decomposition, snapshots, and restarts.

That coupling is intentional: the grains evolve **during** the simulation rather than in post-processing. The design goal is not to eliminate those interfaces, but to keep the dust-specific implementation behind a small and legible set of boundaries.

## Conceptual layers

1. **GADGET-4 infrastructure** — gravity, hydrodynamics, domain decomposition, time integration, MPI, HDF5 I/O.
2. **Dust particle infrastructure** — `PartType6` state, creation/removal, bookkeeping, neighbor access, restart/output integration.
3. **Physical processes** — stellar creation, drag, growth, collisions, sputtering, shocks, astration, cooling, and optional radiation pressure.
4. **Diagnostics and analysis interface** — startup summaries, conservation checks, particle fields, and analysis scripts.

## Current implementation boundaries

Most dust-process implementations are grouped in `src/dust/dust.cc`, with
declarations in `src/dust/dust.h` and per-grain state in
`src/data/dust_particle_data.h`. Feedback and stellar ejecta are handled in
`src/cooling_sfr/feedback.cc`, which partitions dust mass and calls the dust
spawning helper. `src/main/run.cc` accumulates physical time and calls
`update_dust_dynamics()` at the configured cadence (see
[Time Integration](time-integration.md)). File-by-file detail is in
[Source Tree](source-tree.md).

## Decomposition-independent searches

Feedback and dust act on everything within a radius of an event (a supernova,
an AGB star, a forming star, a grain), but each MPI task holds only its own
domain. Searches therefore follow one collective pattern
(`src/cooling_sfr/cg_distributed.h`): every task sees every event, searches
its own particles, and results are exchanged with the event and particle
owners. Ties are broken by particle ID, so the neighbour set and the result do
not depend on the number of ranks.

- **Dust–gas association.** Each grain uses its nearest gas particle within
  10 code length units. If that gas lives on another task, the grain is
  processed there for the update and sent back.
- **D/G lookups** use a fine mass grid; queries are routed to the tasks whose
  cells the search sphere overlaps (occupancy routing, the default from 64
  ranks; `CG_DTG_ROUTING=bbox|occupancy` overrides).
- An all-task self-check of the dust–gas association runs only below 200000
  grains.

## Run-time safeguards

The host code also carries checks added for high-resolution runs: a gas energy
ledger per main-loop phase (`src/main/energy_ledger.h`), an SPH timestep
limiter with wake-up (`src/time_integration/timestep_limiter.h`), a
runaway-force watchdog and density-jump flags in the SPH code, and guards
against dust and star velocity runaways.
