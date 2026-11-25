# Compilation

<div class="cg-card summary" markdown="1">

This page covers configuring and building CosmicGrain itself — the steps you repeat every time you change which physics modules are active, adjust resolution, or switch between production and debug builds. It assumes [Installation](installation.md) is already done.

</div>

## Choosing a configuration

<div class="cg-card implementation" markdown="1">

Compile-time options live in a `Config.sh`-style file. The top-level
`Makefile` selects one of the maintained files in `configs/`:

| File | Use |
|---|---|
| `configs/Config_zoom.sh` | Dust-enabled zoom runs (the Makefile default) |
| `configs/Config_parent.sh` | Dark-matter-only parent volume |
| `configs/Config_uniform.sh` | Uniform-resolution boxes |

Edit the `CONFIG =` line in the `Makefile`, or override it on the command
line (`make CONFIG=configs/Config_parent.sh EXEC=CosmicGrain_parent`).
`make DIR=<dir>` instead builds `<dir>/CosmicGrain` from `<dir>/Config.sh`.

The flags required for a dust-enabled run are:

| Flag | Purpose |
|---|---|
| `DUST` | CosmicGrain dust module (creation, transport, growth, collisions, destruction, radiation pressure). |
| `FEEDBACK` | Stellar feedback: SNII, AGB, and LRN enrichment and dust creation. |
| `STARFORMATION` | Springel & Hernquist (2003) star formation. |
| `COOLING` | Radiative H/He cooling. |
| `METALS` | Element-resolved metals and metal cooling. |
| `IDS_64BIT` | **Required.** Dust spawning overflows 32-bit IDs without any error. |
| `NTYPES=7` | **Required.** Reserves PartType6 for live dust. |

`configs/Config_zoom.sh` also sets `PERIODIC`, `DOUBLEPRECISION=2`,
`POSITIONS_IN_64BIT`, `ADAPTIVE_HYDRO_SOFTENING`, `CUBIC_SPLINE_KERNEL`,
`PMGRID=512`, `PM_ZOOM_OPTIMIZED`, `TREEPM_NOTIMESPLIT`,
`HIERARCHICAL_GRAVITY`, `INDIVIDUAL_GRAVITY_SOFTENING=4`, `FOF`, `SUBFIND`,
and `EXPLICIT_VECTORIZATION`.

**Multi-node builds.** On more than one node GADGET-4 splits each node into
shared-memory islands of at most 64 ranks and reserves one listener rank per
island. With many ranks per node, set `NUMBER_OF_MPI_LISTENERS_PER_NODE=3`
(or more) in the config, otherwise startup stops with *"We have shared memory
islands with N MPI ranks, which is larger than
MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY=64"*. Single-node builds leave it
unset. See [HPC Systems](hpc.md#colibri).

</div>

## Building

<div class="cg-card implementation" markdown="1">

`Makefile.systype` names the system (`SYSTYPE="Cygnus"` on the development
workstation); its compiler settings come from `buildsystem/Makefile.comp.*`.
Then:

```bash
make -j8
```

This produces `./CosmicGrain`. The Cygnus and `Generic-gcc` settings compile
with `mpicxx -std=c++11 -O3 -march=native -ffast-math -fno-finite-math-only
... -flto`. Keep `-fno-finite-math-only`: plain `-ffast-math` lets GCC remove
the `std::isfinite()` guards the code relies on. Pass `OPT=` to drop `-flto`
where the link-time optimizer runs out of memory.

Record `sha256sum CosmicGrain`, the git commit, and the config file with
every run. For long job chains, copy the binary to a per-job name so a
rebuild cannot change a queued run (the Colibri scripts do this).

</div>

## Verifying the build

<div class="cg-card implementation" markdown="1">

On the first dust update, task 0 prints the parsed dust switches once:

```text
[DUST_FLAGS|Step=0] Creation=1 Drag=1 Growth=1 Coagulation=1 Sputtering=1 ShockDestruction=1 Astration=1 RadPressure=1 Clumping=2 Cooling=1
```

`Clumping` is the `DustEnableClumping` mode (0 off, 1 legacy factor,
2 Blitz & Rosolowsky dense phase, the v1.2.0 default). `DustEnableShattering`
is not in this line; check it in `parameters-usedvalues`. A value that differs
from the parameter file usually means a misspelled parameter name.

</div>
