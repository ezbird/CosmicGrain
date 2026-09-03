# Performance and Scaling

CosmicGrain cost is governed by both the initial zoom geometry and the dust
population spawned during evolution. Nominal resolution alone is therefore
not a sufficient resource estimate.

## Initial-condition size

The accepted suite ranges from 2.12 million particles for halo 3879 at
\(512^3\) to 517.54 million for halo 1534 at \(4096^3\). Measure each file
directly before choosing MPI tasks, memory limits, or storage allocations.

## Dust time bins

At high resolution, newly spawned dust can become a substantial fraction of
the gravity tree. `spawn_dust_particle()` assigns a synchronized birth
timebin. The ongoing gravity timestep criterion then allows dust to migrate
toward appropriately long synchronized bins after its first force evaluation.
This prevents a newly created particle from entering an unsynchronized bin and
causing a collective gravity hang.

## Dust-physics cadence

The expensive dust update is cadence controlled rather than called on every
gravity step. Processes using the scaled interval employ exact exponential
updates where applicable, such as \(1-\exp(-\Delta t/\tau)\), rather than
unstable linear approximations.

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
