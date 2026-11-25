# Compile-Time Configuration

CosmicGrain is built like GADGET-4: `make CONFIG=<file> SYSTYPE=<system>`.
Dust processes are then switched at run time with the `DustEnable*`
parameters. Archive the exact `Config.sh` with every run.

## CosmicGrain flags

| Flag | Purpose |
| --- | --- |
| `NTYPES=7` | Adds particle type 6 for dust superparticles |
| `DUST` | Builds `src/dust/` (dust particles, snapshot fields, creation, evolution) and registers the `Dust*` parameters |
| `FEEDBACK` | Delayed SNII/HN, AGB and LRN feedback and enrichment, the yield tables and the `AGB*`/`SNEfficiency`/`HypernovaFraction` parameters. Requires a periodic 3D cubic box (`PERIODIC`, no `LONG_*_BITS`, `GRAVITY_TALLBOX`, `ONEDIMS`, `TWODIMS` or `POSITIONS_IN_128BIT`) |
| `COOLING` | Cooling; also compiles `src/cooling_sfr/feedback.cc` and the yield readers, so `FEEDBACK` needs it |
| `STARFORMATION` | Springel & Hernquist (2003) star formation |
| `METALS` | Element and metallicity tracking used by enrichment, cooling and dust growth |

The SPH timestep limiter, runaway watchdog and gas energy ledger are always
compiled in; the limiter and watchdog are controlled by environment variables
(see [Runtime Parameters](parameters.md#environment-variables)).

## Production configuration

`Config.sh` and `configs/Config_zoom.sh` are identical and are used for all
zoom runs:

```text
PERIODIC  NTYPES=7  RANDOMIZE_DOMAINCENTER  SELFGRAVITY  EVALPOTENTIAL
PMGRID=512  ASMTH=1.25  RCUT=6.0  FFT_COLUMN_BASED  PM_ZOOM_OPTIMIZED
TREE_NUM_BEFORE_NODESPLIT=10  TREEPM_NOTIMESPLIT  HIERARCHICAL_GRAVITY
INDIVIDUAL_GRAVITY_SOFTENING=4  ADAPTIVE_HYDRO_SOFTENING  CUBIC_SPLINE_KERNEL
COOLING  STARFORMATION  FEEDBACK  METALS  DUST
POSITIONS_IN_64BIT  DOUBLEPRECISION=2  IDS_64BIT
FOF  FOF_PRIMARY_LINK_TYPES=2  FOF_SECONDARY_LINK_TYPES=1+16+32  FOF_GROUP_MIN_LEN=32
SUBFIND  SUBFIND_STORE_LOCAL_DENSITY  EXPLICIT_VECTORIZATION
```

Multi-node builds (e.g. Colibri) also need
`NUMBER_OF_MPI_LISTENERS_PER_NODE=3`: GADGET-4 splits each node into
shared-memory islands of at most 64 ranks, and with the default single
listener a node with more than 64 ranks stops at start-up. Single-node runs do
not need it.

Other configurations in `configs/`:

| File | Use |
| --- | --- |
| `Config_parent.sh` | Dark-matter-only parent box (`NTYPES=6`, `PMGRID=1024`, FOF/SUBFIND, no gas physics) |
| `Config_uniform.sh` | Uniform-box gas run (`NTYPES=6`, `PMGRID=256`, cooling, star formation, feedback, metals; no `DUST`) |

## Compiler options

`buildsystem/Makefile.comp.gcc` and `Makefile.comp.cygnus` build with
`-ffast-math -fno-finite-math-only`. `-ffast-math` alone lets GCC remove the
`std::isfinite()` guards the dust and feedback code relies on; keep
`-fno-finite-math-only` on any new system type.
