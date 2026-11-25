# Performance and Scaling

CosmicGrain cost is governed by both the initial zoom geometry and the dust
population spawned during evolution. Nominal resolution alone is therefore
not a sufficient resource estimate.

## Initial-condition size

The accepted suite spans from a few million particles in the smallest \(512^3\)-equivalent zooms to hundreds of millions in the largest high-resolution ICs. Measure each file directly before choosing MPI tasks, memory limits, or storage allocations.

## Dust time bins

At high resolution, newly spawned dust can become a substantial fraction of
the gravity tree. `spawn_dust_particle()` assigns a synchronized birth
timebin. The ongoing gravity timestep criterion then allows dust to migrate
toward appropriately long synchronized bins after its first force evaluation.
This prevents a newly created particle from entering an unsynchronized bin and
causing a collective gravity hang.

## Dust-physics cadence

The expensive dust update runs every `DustUpdateCadence` global sync points
(7 in v1.2.0) rather than on every step. The physical time is accumulated and
applied in one update, with exact exponential forms such as
\(1-\exp(-\Delta t/\tau)\) where applicable, and pending time is flushed
before snapshots and at z = 0.

## Distributed feedback and dust

Feedback and dust neighbour searches (SNII, AGB, shock erosion, astration,
dust–gas association, radiation pressure) are exchanged across ranks in
batches, so results do not depend on the domain decomposition or rank count.
Local D/G queries are routed through a distributed cell directory: by
bounding box below 64 ranks, by cell occupancy from 64 ranks up (override
with `CG_DTG_ROUTING`). The `feedback`/`fbhash` and `dust`/`dustdtg` timers in
`cpu.txt` show their cost; at \(2048^3\) dust can dominate the run time.

These exchanges are all-rank collectives, so multi-node runs are sensitive to
the MPI transport: on Colibri, UCX over InfiniBand is about 8× faster than
TCP for 2-node \(2048^3\) runs (see [HPC Systems](../getting-started/hpc.md#colibri)).

## Typical cost

| Run | Resources | Wall time to z = 0 |
|---|---|---|
| \(512^3\) halo 441 | 24 ranks, one workstation | ~1.5–2 h |
| \(1024^3\) halo 441 | 2 Colibri nodes (380 ranks), UCX | ~3 h |

## FOF and SUBFIND

FOF/SUBFIND can dominate runtime when executed frequently. Snapshot cadence
and group-finding cadence are separate scientific choices: retain enough halo
catalogs to follow the target, but avoid invoking SUBFIND at every fine output
unless the analysis requires it.

## Before a production launch

1. run through the first domain decomposition and force calculation;
2. record peak resident memory per rank;
3. check particle imbalance and top-level tree balance;
4. estimate output size using the actual HDF5 IC;
5. confirm restart writing; and
6. repeat the test when changing resolution, halo, MPI layout, or major
   particle-creation parameters.
