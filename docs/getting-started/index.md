This section is still under construction! We'll have this pages filled out as soon as we can.

Last Updated: July 9, 2026.


# Installation

<div class="cg-card summary" markdown="1">

This page covers system-level prerequisites — the things you set up once per machine, before CosmicGrain-specific configuration begins. If you're reconfiguring an existing environment for a new physics module or resolution, you probably want [Compilation](compilation.md) instead.

</div>

## Prerequisites

<div class="cg-card implementation" markdown="1">

CosmicGrain requires an MPI-aware C++11 compiler and three external libraries:

| Requirement | Notes |
|---|---|
| MPI C++ compiler (`mpicxx`) | <!-- TODO: confirm minimum supported MPI implementation/version, e.g. OpenMPI 4.x, MPICH --> |
| HDF5 | <!-- TODO: confirm tested version; note whether serial or parallel HDF5 is required --> |
| GSL | <!-- TODO: confirm tested version --> |
| FFTW3 | <!-- TODO: confirm tested version --> |
| Python 3.x | Required for the analysis and plotting scripts (`halo_utils.py` and related tools), not for the simulation code itself. |

If your library paths aren't in the default system locations, note them now — you'll need them in `Makefile.systype` or your local Makefile configuration during [Compilation](compilation.md).

</div>

## Getting the source

<div class="cg-card implementation" markdown="1">

```bash
git clone <!-- TODO: repository URL --> cosmicgrain
cd cosmicgrain
```

<!-- TODO: recommended branch/tag for new users — main? a tagged release? -->

</div>

## Directory layout

<div class="cg-card implementation" markdown="1">

A quick orientation before diving into `Config.sh`:

- `src/dust/` — CosmicGrain's dust physics module (`dust.cc`, `dust.h`)
- `src/cooling_sfr/` — stellar feedback and spatial hashing (`feedback.cc`, `spatial_hash_zoom.h`)
- <!-- TODO: fill in other top-level directories worth knowing about early -->

</div>

Once libraries are in place, continue to [Compilation](compilation.md).
