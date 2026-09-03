# Running CosmicGrain

A production run combines three versioned inputs:

- the compiled executable and its `Config.sh`;
- a seven-type HDF5 initial condition; and
- the runtime parameter file.

Before launch, retain copies of all three and run a short startup test using
the same MPI layout intended for the first production segment. Confirm the
seven particle types, cosmology, softenings, memory allowance, output list,
runtime dust switches, and initial domain decomposition.

The pages in this section cover [parameter files](parameter-files.md),
[compile-time options](compile-options.md), [initial
conditions](initial-conditions.md), [outputs](output-files.md),
[restarts](restart-files.md), [performance](performance.md), and
[troubleshooting](troubleshooting.md).
