# Installation

<div class="cg-card summary" markdown="1">

This page covers system-level prerequisites — the things you set up once per machine, before CosmicGrain-specific configuration begins. If you're reconfiguring an existing environment for a new physics module or resolution, you probably want [Compilation](compilation.md) instead.

</div>

## Prerequisites

<div class="cg-card implementation" markdown="1">

CosmicGrain requires an MPI-aware C++11 compiler and three external libraries: HDF5, GSL, and FFTW3.

| Requirement | Notes |
|---|---|
| MPI C++ compiler (`mpicxx`) | Must support C++11 (`-std=c++11`). |
| HDF5 | A system-installed serial HDF5 package is sufficient (e.g. `libhdf5-serial-dev` on Debian/Ubuntu). |
| GSL | System package (`libgsl-dev`) or a source build. |
| FFTW3 | System package (`libfftw3-dev`) or a source build. |
| Python 3.x | Required for the analysis and plotting scripts (`halo_utils.py` and related tools), not for the simulation code itself. |

### System packages

```bash
# Debian/Ubuntu
sudo apt install libhdf5-serial-dev libgsl-dev libfftw3-dev

# On HPC systems, check for a module instead:
module avail hdf5
```

### Building GSL and FFTW3 from source

Without root access or a suitable module, build both from source:

```bash
# GSL
wget https://ftp.gnu.org/gnu/gsl/gsl-<version>.tar.gz
tar xzf gsl-<version>.tar.gz
cd gsl-<version>
./configure --prefix=$LIB_DIR/gsl/build
make -j4 && make install

# FFTW3
wget https://www.fftw.org/fftw-<version>.tar.gz
tar xzf fftw-<version>.tar.gz
cd fftw-<version>
./configure --prefix=$LIB_DIR/fftw3/build
make -j4 && make install
```

`buildsystem/Makefile.gen.libs` looks for `$(LIB_DIR)/gsl/build` and
`$(LIB_DIR)/fftw3/build`; export `LIB_DIR` or override `GSL_INCL`/`GSL_LIBS`,
`FFTW_INCL`/`FFTW_LIBS`, and `HDF5_INCL`/`HDF5_LIBS` on the `make` line.

</div>

## Getting the source

<div class="cg-card implementation" markdown="1">

```bash
git clone https://github.com/ezbird/CosmicGrain.git
```

</div>

## Directory layout

<div class="cg-card implementation" markdown="1">

- `src/dust/` — dust module (`dust.cc`, `dust.h`, `dust_particle_log.cc`)
- `src/cooling_sfr/` — cooling, star formation, feedback, yields, and the
  zoom spatial hash (`feedback.cc`, `snii_yields.cc`, `agb_yields.cc`,
  `cg_distributed.h`, `spatial_hash_zoom.h`)
- `src/main/energy_ledger.h`, `src/sph/hydro_watch.h`,
  `src/time_integration/timestep_limiter.h` — run-time guards (see
  [Troubleshooting](../running/troubleshooting.md))
- `configs/` — compile-time configurations
- `params/` — `param_base.txt` (v1.2.0 defaults) and per-run parameter files
- `data/` — cooling tables and SNII/AGB/hypernova yield tables
- `ICs/` — MUSIC2 configurations and initial conditions
- `scripts/` — zoom preparation, parameter generation, validation, analysis,
  SKIRT utilities, and `scripts/colibri/` batch scripts
- `docs/` — this MkDocs documentation

</div>

Once libraries are in place, continue to [Compilation](compilation.md).
