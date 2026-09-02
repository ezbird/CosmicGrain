This section is still under construction! We'll have these pages filled out as soon as we can.

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
| MPI C++ compiler (`mpicxx`) | Tested with **OpenMPI 4.1.6** (desktop, Ubuntu 24.04, g++ 13.3.0) and **Intel MPI 2021.x** (TACC Stampede). Not tied to a specific MPI vendor -- both build and run successfully. |
| HDF5 | Tested with **1.10.10** (desktop, `libhdf5-dev` / serial variant). CosmicGrain links against **serial HDF5**, not parallel/collective HDF5 -- on Debian/Ubuntu systems, make sure you're linking `libhdf5-serial`, not `libhdf5-openmpi`, even if both are installed. |
| GSL | Tested with **2.7.1** (desktop). <!-- TODO: confirm GSL version on Stampede via `module list` --> |
| FFTW3 | Tested with **3.3.10** on both desktop (Ubuntu `libfftw3-dev`) and Stampede (`/opt/apps/intel24/impi21/fftw3/3.3.10/`) -- confirmed consistent across both build environments. |
| Python 3.x | Required for the analysis and plotting scripts (`halo_utils.py` and related tools), not for the simulation code itself. |

</div>

If your library paths aren't in the default system locations, note them now — you'll need them in `Makefile.systype` or your local Makefile configuration during [Compilation](compilation.md).


## Getting the source

<div class="cg-card implementation" markdown="1">

```bash
git clone https://github.com/ezbird/CosmicGrain.git```

<!-- TODO: recommended branch/tag for new users — main? a tagged release? -->

</div>

## Directory layout

<div class="cg-card implementation" markdown="1">

A quick orientation before diving into `Config.sh`:

- `src/dust/` — CosmicGrain's dust physics module (`dust.cc`, `dust.h`)
- `src/cooling_sfr/` — stellar feedback and spatial hashing (`feedback.cc`, `spatial_hash_zoom.h`)
</div>

Once libraries are in place, continue to [Compilation](compilation.md).
