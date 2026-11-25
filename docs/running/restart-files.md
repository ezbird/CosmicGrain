# Restart Files

Restart continuity is a required CosmicGrain regression test because live
`PartType6` particles carry additional state and the dust update can span more
than one synchronization point.

## Binary restarts

Resume from `OutputDir/restartfiles/` with start mode 1:

```bash
mpirun -np <N> ./CosmicGrain param.txt 1
```

Use the same MPI task count, and in practice the same executable, as the run
that wrote the restart set. CosmicGrain prefixes the restart payload with a
check of `AGBFeedbackMode` and of the sizes of the global, particle, and SPH
data structures; a binary whose layout differs stops with *"Incompatible AGB
restart memory layout"*. To change the task count, binary, or machine, use a
snapshot restart.

Restart files are written every `CpuTimeBetRestartFile` seconds and when the
run stops. Files in `OutputDir` control a running job:

- `stop` — write restart files and exit;
- `restart` — write restart files and continue;
- at 85 % of `TimeLimitCPU` the code writes restart files, exits, and leaves a
  `cont` file.

## CosmicGrain restart state

After the normal GADGET-4 payload, restart files carry two appended,
versioned CosmicGrain blocks:

- **Dust cadence:** accumulated sync points and physical time since the last
  dust update, created/destroyed counters, accumulated dust mass change, and
  the pending-update flag. `DustUpdateCadence` itself always comes from the
  parameter file.
- **D/G cache:** each grain's cached local dust-to-gas ratio and its age.

A restart without the cadence block starts a new cadence window; one without
the cache rebuilds it on the first dust update.

Per-run text logs behave differently on a restart: `dust_logs/` files are
appended, `sfr.txt`'s cumulative stellar-mass column restarts from 0, and
`agb_coupling_diagnostics.csv` is rewritten.

## Snapshot restarts

```bash
mpirun -np <N> ./CosmicGrain param.txt 2 <snapnum>
```

Start mode 2 needs 0 < `TimeBegin` < `TimeMax` and, unlike a fresh start,
does not warn about `TimeBegin`. CosmicGrain then:

1. recomputes the global maximum particle ID;
2. reassigns live dust IDs to a unique range above the maximum non-dust ID;
3. applies a tiny integer-position jitter to dust particles to avoid tree
   failures from coincident positions;
4. builds the cosmic-time table from min(`TimeBegin`, 0.01) so stars born
   before the snapshot keep their true ages, and clamps each star's AGB
   ledger age to its recomputed age (stopping if any shift exceeds 1 Myr); and
5. checks dust state integrity before and after the first domain
   decomposition.

A snapshot restart is supported but is not numerically identical to a binary
restart.

## Regression checklist

After changes to feedback, particle exchange, I/O, or dust integration:

1. stop a short dust-bearing run at a known synchronization point;
2. resume from its binary restart files;
3. compare gas/star/dust counts and masses across the boundary;
4. verify `GrainRadius`, `CarbonMassFraction`, `DustTemperature`, `DustSource`,
   and `BirthPos` remain valid;
5. verify the dust-cadence state resumes rather than silently resetting; and
6. continue long enough to cross at least one subsequent dust update.
