# Quick Start

<div class="cg-card summary" markdown="1">

The fastest path from a built executable to confirming it runs. Minimal explanation by design — if you want to understand *why* each choice is made, see [First Simulation](first-simulation.md) instead.

</div>

## Run a minimal test

<div class="cg-card implementation" markdown="1">

Use an already validated \(512^3\)-equivalent zoom IC and a parameter file
generated from `params/param_base.txt` (see [Parameter
Files](../running/parameter-files.md)), then start a short MPI test:

```bash
mpirun -np 4 ./CosmicGrain params/param_halo441_512.txt
```

The third argument is the GADGET-4 start mode (omitted = 0, start from the
IC). The complete IC suite should already have passed
`validate_music2_ic_suite.py`; do not rescan all 48 files for each smoke
test.

The code stops at startup unless `ComovingIntegrationOn 1` and
`AGBFeedbackMode 2`. `TimeBegin` other than 0.01 or `TimeMax` other than 1
only print a `WARNING`, since the calibrated runs go from z = 99 to z = 0.

</div>

## Confirming success

<div class="cg-card implementation" markdown="1">

Within the first few timesteps you should see:

- successful HDF5 loading with seven particle types;
- the expected cosmology and initial redshift;
- `[DUST_LOG] Opened dust_logs/dust_log_task*.txt`;
- a `[DUST_FLAGS]` line matching the parameter file;
- successful domain decomposition and gravity-tree construction; and
- no missing-parameter, non-finite-value, or particle-ID errors.

To end the test cleanly, create a file named `stop` in the `OutputDir`; the
code writes restart files and exits. A successful startup is not an
end-to-end physics validation.

</div>
