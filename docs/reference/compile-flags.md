# Compile-Time Configuration

CosmicGrain is compiled as a GADGET-4-based code. The authoritative build for a particular simulation is its archived `Config.sh`; runtime dust processes are then selected with the `DustEnable*` parameters.

## Required CosmicGrain structure

| Flag | Purpose |
| --- | --- |
| `NTYPES=7` | Provides particle type 6 for dust superparticles. |
| `DUST` | Compiles the CosmicGrain dust data structures, snapshot fields, creation/evolution routines, and integration hooks. |

For the current galaxy-formation implementation, the supplied configuration also enables `COOLING`, `STARFORMATION`, `FEEDBACK`, and `METALS`, which provide the gas cooling, star formation, delayed stellar feedback/enrichment, and elemental reservoirs used by the dust model.

## Configuration in the audited source checkout

The `Config.sh` supplied for this audit currently enables, among other options:

```text
PERIODIC
NTYPES=7
SELFGRAVITY
EVALPOTENTIAL
PMGRID=512
FFT_COLUMN_BASED
PM_ZOOM_OPTIMIZED
TREEPM_NOTIMESPLIT
HIERARCHICAL_GRAVITY
INDIVIDUAL_GRAVITY_SOFTENING=4
ADAPTIVE_HYDRO_SOFTENING
CUBIC_SPLINE_KERNEL
COOLING
STARFORMATION
FEEDBACK
METALS
DUST
POSITIONS_IN_64BIT
DOUBLEPRECISION=2
IDS_64BIT
```

This is a description of the **uploaded checkout**, not a claim that every CosmicGrain run must use those numerical choices. In particular, PM resolution, kernel choice, precision mode, softening configuration, and other GADGET-4 options may legitimately differ between experiments.

## Important distinction

Do not copy compile flags from a paper, an old parameter generator, or this documentation and assume they describe a particular simulation. Archive the exact `Config.sh` used to build that executable. The current source checkout, for example, uses the cubic-spline kernel and `DOUBLEPRECISION=2`; older CosmicGrain configurations have used different choices.
