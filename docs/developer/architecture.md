# Code Architecture

CosmicGrain adds an explicit dust particle population (`PartType6`) to the GADGET-4 simulation framework. Dust physics necessarily touches several host-code systems: particle storage, gravity, neighbor searches, time integration, stellar feedback/enrichment, domain decomposition, snapshots, and restarts.

That coupling is intentional: the grains evolve **during** the simulation rather than in post-processing. The design goal is not to eliminate those interfaces, but to keep the dust-specific implementation behind a small and legible set of boundaries.

## Conceptual layers

1. **GADGET-4 infrastructure** — gravity, hydrodynamics, domain decomposition, time integration, MPI, HDF5 I/O.
2. **Dust particle infrastructure** — `PartType6` state, creation/removal, bookkeeping, neighbor access, restart/output integration.
3. **Physical processes** — stellar creation, drag, growth, collisions, sputtering, shocks, astration, cooling, and optional radiation pressure.
4. **Diagnostics and analysis interface** — startup summaries, conservation checks, particle fields, and analysis scripts.

## Planned modular direction

The current source is not yet organized exactly this way, but future cleanup should move toward a structure such as:

```text
src/
  dust/
    dust.h
    dust.cpp

    creation/
      snii.cpp
      agb.cpp
      lrne.cpp

    dynamics/
      drag.cpp
      radiation_pressure.cpp

    processing/
      growth.cpp
      coagulation.cpp
      shattering.cpp

    destruction/
      sputtering.cpp
      sn_shocks.cpp
      astration.cpp

    cooling/
      dust_cooling.cpp

    io/
      dust_io.cpp

    diagnostics/
      dust_diagnostics.cpp
```

This is a **development target**, not a statement about the current on-disk source tree. Refactoring should preserve scientific behavior and be validated incrementally rather than performed as one large physics-changing rewrite.
