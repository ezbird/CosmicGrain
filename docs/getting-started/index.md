# Getting Started

CosmicGrain is a seven-particle-type extension of GADGET-4 for explicit,
on-the-fly dust evolution. A reliable first run has four prerequisites:

1. build GADGET-4 with `NTYPES=7`, `IDS_64BIT`, and the required dust,
   hydrodynamic, star-formation, feedback, and metals flags;
2. use an HDF5 initial condition whose header arrays have seven entries and
   whose empty `PartType6` group is valid;
3. provide every required runtime dust and feedback parameter; and
4. verify the parsed `[DUST_FLAGS]` summary before evolving the simulation.

Use these pages in order:

- [Installation](installation.md) — libraries and source tree
- [Compilation](compilation.md) — compile flags and build verification
- [Quick Start](quick-start.md) — short startup smoke test
- [First Simulation](first-simulation.md) — staged validation run
- [Zoom Setup](zoom-setup.md) — current 12-halo MUSIC2 suite
- [HPC Systems](hpc.md) — machine-specific considerations

The accepted production IC suite contains 12 halos at four resolution levels.
All 48 files passed the independent [IC-suite validation
checklist](../validation/ic-suite.md).
