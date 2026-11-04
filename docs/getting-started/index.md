# Getting Started

CosmicGrain is distributed as a complete **GADGET-4-based simulation code** with explicit, on-the-fly dust evolution. It is not intended to be copied into an arbitrary GADGET-4 or AREPO checkout as a drop-in plug-in.

A reliable first run has four prerequisites:

1. obtain the CosmicGrain source and build it with `NTYPES=7`, `IDS_64BIT`, and the required dust, hydrodynamic, star-formation, feedback, and metals options;
2. use an HDF5 initial condition whose particle-count arrays support seven particle types;
3. provide every required runtime dust and feedback parameter; and
4. verify the parsed `[DUST_FLAGS]` summary before evolving the simulation.

## Choose your path

**Run the existing model:** start with [Installation](installation.md), [Compilation](compilation.md), and [Quick Start](quick-start.md).

**Change a dust scenario:** after a successful reference run, use [Parameter Files](../running/parameter-files.md), the [Dust Physics](../physics/index.md) pages, and the [Parameters](../reference/parameters.md) reference.

**Add new dust physics:** first reproduce a reference run, then read the [Developer Guide](../developer/index.md) and [Adding New Physics](../developer/adding-new-physics.md).

The current research suite uses MUSIC2 zoom initial conditions and seven particle types. Empty `PartType6` at the initial redshift is expected: dust particles are created later by the stellar source channels.
