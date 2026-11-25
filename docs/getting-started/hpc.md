# HPC Systems

<div class="cg-card summary" markdown="1">

Platform-specific setup and known issues for the clusters CosmicGrain has been run on. If something here doesn't apply to your system, it probably belongs on [Installation](installation.md) or [Compilation](compilation.md) instead — this page is only for things that are true on one specific machine.

</div>

## Colibri

<div class="cg-card implementation" markdown="1">

Slurm partition `general-cpu`, 190 MPI ranks per node with
`--mem-per-cpu=3500M`. Batch scripts are in `scripts/colibri/`:

| Script | Use |
|---|---|
| `runBR06.slurm` | Build-and-run: copies `./CosmicGrain` to `~/jobexe/CosmicGrain.$SLURM_JOB_ID`, then runs it |
| `runRel.slurm` | 1024³ run from a fixed binary on 2 nodes; deletes `restartfiles/` after a complete run |
| `runCSgrid.slurm`, `submit_cs_grid.sh` | Coagulation–shattering grid as an `afterany` job chain from one snapshotted binary |
| `runDTcs.slurm`, `runDTcs1n.slurm` | As above with `CG_FEEDBACK_DELTA_T=3.16e7` (10^7.5 K SN heating), 2 or 1 nodes |

Each script takes the parameter file as `$1` and starts with
`srun --export=ALL --mpi=pmix_v5 --unbuffered <exe> "$1" 0`. Put the restart
mode on that line when resuming. Set your own `--mail-user` on the cluster
copy.

**Environment.**

```bash
module load sstack/latest
module load spack/cosmicgrain
spack load openmpi@5.0.10 fftw@3.3.10 hdf5@1.14.6 gsl@2.8
export OMPI_MCA_pml=ucx UCX_NET_DEVICES=mlx5_0:1
unset OMPI_MCA_btl_tcp_if_include
```

The scripts also prepend each package's `lib/` to `LD_LIBRARY_PATH`. Use UCX
(native InfiniBand): the module default, Open MPI TCP over `ib0`, made 2-node
2048³ runs about 8× slower because the dust and feedback exchanges are
communication-bound.

**Building.** Build with `SYSTYPE="Generic-gcc"`, the Spack prefixes in
`GSL_INCL`/`GSL_LIBS`, `FFTW_INCL`/`FFTW_LIBS`, `HDF5_INCL`/`HDF5_LIBS`, and
`OPT=` (no `-flto`: the GCC 14 LTO link exhausts node memory). Do not run
`make` with many jobs on the login node, which has a per-user process limit;
use `make -j 4` there or build inside a Slurm job. For multi-node runs set
`NUMBER_OF_MPI_LISTENERS_PER_NODE=3` in `configs/Config_zoom.sh` (3 islands of
64/63/63 ranks per node).

**Timing.** A 1024³ zoom to z = 0 takes about 3 h on 2 nodes with UCX.
Partial runs are ended with the `stop` file (see
[Restart Files](../running/restart-files.md)).

</div>

## TACC Stampede

<div class="cg-card implementation" markdown="1">

Record the compiler, MPI, HDF5, FFTW, and GSL modules in every batch script.
Task count and node layout must be chosen from measured memory and particle
balance for the specific halo/resolution pair; the \(4096^3\) ICs differ by
more than an order of magnitude in particle count.

Restart files must be read with the same MPI task count that wrote them;
use a snapshot restart to change the task count.

</div>

## NMSU Discovery Cluster

<div class="cg-card implementation" markdown="1">

Before production, rerun the startup and collective-communication smoke test
with the currently loaded compiler/MPI stack. Preserve `module list`, the
batch script, and the first domain-decomposition log with the run.

</div>

## Desktop / workstation builds

<div class="cg-card implementation" markdown="1">

On the 24-core workstation, begin with 24 or fewer MPI ranks and measure memory
per rank rather than maximizing rank count automatically. `MaxMemSize` is a
per-rank allowance and must leave room for MPI, HDF5 buffers, FFT meshes, and
the operating system. One CosmicGrain run fills all 24 cores: never start a
second MPI job beside it (a concurrent job once stalled a production run for
hours). Very large \(4096^3\) ICs belong on a cluster.

</div>
