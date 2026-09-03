# Quick Start

<div class="cg-card summary" markdown="1">

The fastest path from a built executable to confirming it runs. Minimal explanation by design — if you want to understand *why* each choice is made, see [First Simulation](first-simulation.md) instead.

</div>

## Run a minimal test

<div class="cg-card implementation" markdown="1">

Use an already validated \(512^3\)-equivalent zoom IC and the intended
parameter file, then perform a short MPI startup test:

```bash
mpirun -np 4 ./CosmicGrain parameterfile.txt
```

The complete IC suite should already have passed
`validate_music2_ic_suite.py`; do not rescan all 48 large files merely as part
of each startup smoke test.

</div>

## Confirming success

<div class="cg-card implementation" markdown="1">

Within the first few timesteps you should see:

- successful HDF5 loading with seven particle types;
- the expected cosmology and initial redshift;
- a `[DUST_FLAGS]` line matching the parameter file;
- successful domain decomposition and gravity-tree construction; and
- no missing-parameter, non-finite-value, or particle-ID errors.

Stop the smoke test after the first few synchronized steps if it was launched
only to verify startup. Do not treat a successful startup as an end-to-end
physics validation.
