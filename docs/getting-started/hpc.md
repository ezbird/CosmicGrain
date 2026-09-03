# HPC Systems

<div class="cg-card summary" markdown="1">

Platform-specific setup and known issues for the clusters CosmicGrain has been run on. If something here doesn't apply to your system, it probably belongs on [Installation](installation.md) or [Compilation](compilation.md) instead — this page is only for things that are true on one specific machine.

</div>

## TACC Stampede

<div class="cg-card implementation" markdown="1">

Record the compiler, MPI, HDF5, FFTW, and GSL modules in every batch script.
Task count and node layout must be chosen from measured memory and particle
balance for the specific halo/resolution pair; the \(4096^3\) ICs differ by
more than an order of magnitude in particle count.

**Known issue — restart-file portability.** Restart files are not portable across different MPI task counts on Stampede; restarting with a different task count than the original run hangs. Use snapshot-based restarts instead when changing task count.

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
the operating system. Very large \(4096^3\) ICs should be moved to an
appropriately sized cluster rather than forced into the workstation layout.

</div>
