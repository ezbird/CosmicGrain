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
| GSL | Typically built from source rather than relying on a system package, to control version and optimization flags — see below. |
| FFTW3 | Same as GSL — commonly built from source. |
| Python 3.x | Required for the analysis and plotting scripts (`halo_utils.py` and related tools), not for the simulation code itself. |

### Installing HDF5 (system package)

```bash
# Debian/Ubuntu
sudo apt install libhdf5-serial-dev

# On HPC systems, check for a module instead:
module avail hdf5
```

### Building GSL and FFTW3 from source

If your system's package manager doesn't provide a suitable version, or you're on an HPC system without root access, build both from source into a local directory:

```bash
# GSL
wget https://ftp.gnu.org/gnu/gsl/gsl-<version>.tar.gz
tar xzf gsl-<version>.tar.gz
cd gsl-<version>
./configure --prefix=/path/to/gsl/build
make -j4 && make install

# FFTW3
wget https://www.fftw.org/fftw-<version>.tar.gz
tar xzf fftw-<version>.tar.gz
cd fftw-<version>
./configure --prefix=/path/to/fftw3/build
make -j4 && make install
```

Take note of these install paths — you'll pass them to the compiler via `-I` and `-L` flags in [Compilation](compilation.md).

</div>

## Getting the source

<div class="cg-card implementation" markdown="1">

```bash
git clone https://github.com/ezbird/CosmicGrain.git
```

</div>

## Directory layout

<div class="cg-card implementation" markdown="1">

A quick orientation before diving into `Config.sh`:

- `src/dust/` — CosmicGrain's dust physics module (`dust.cc`, `dust.h`)
- `src/cooling_sfr/` — stellar feedback and spatial hashing (`feedback.cc`, `spatial_hash_zoom.h`)
- <!-- TODO: fill in other top-level directories worth knowing about early -->

</div>

## A known-working compiler invocation

<div class="cg-card implementation" markdown="1">

For reference, here's a tested build configuration showing how the library paths above get wired together (full details on flags and `Config.sh` options are in [Compilation](compilation.md)):

```bash
mpicxx -std=c++11 -O3 -march=native -mtune=native \
    -ffast-math -fno-math-errno -fno-trapping-math \
    -funroll-loops -flto \
    -I/usr/include/hdf5/serial \
    -I/path/to/gsl/build/include \
    -I/path/to/fftw3/build/include \
    -Ibuild -Isrc -c src/<file>.cc -o build/<file>.o
```

The `-I/usr/include/hdf5/serial` path corresponds to a system-installed HDF5 package; the GSL and FFTW3 paths point to custom builds under dedicated directories, following the pattern above.

</div>

Once libraries are in place, continue to [Compilation](compilation.md).
