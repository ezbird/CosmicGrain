# Restart Files

Restart continuity is a required CosmicGrain regression test because live
`PartType6` particles carry additional state and the dust update can span more
than one synchronization point.

## Standard GADGET-4 restart modes

For an ordinary binary restart, resume with:

```bash
mpirun -np <N> ./Gadget4 param.txt 1
```

Use the same executable and MPI task count as the run that wrote the restart
set. GADGET-4 binary restart files encode internal structures and should not be
assumed portable across MPI layouts, compilers, or machines. When the MPI task
count or platform must change, use a supported snapshot continuation and
validate the resumed run. This follows the underlying GADGET-4 restart model.

A clean manual stop can be requested by placing a file named `stop` in the
simulation output directory. The code finishes the current step, writes restart
files, and exits.

## CosmicGrain restart state

Current CosmicGrain restart files append versioned dust state after the normal
GADGET-4 restart payload. The dust-cadence block preserves:

- accumulated dust synchronization points;
- accumulated physical time since the last dust update;
- dust-created and dust-destroyed counters for the current interval;
- accumulated dust mass change; and
- the pending dust-synchronization flag.

An additional DTG-cache block stores each dust particle's cached local dust-to-
gas ratio and refresh age. The particle records themselves retain their legacy
layout; the cache is appended separately. Older restarts without either
CosmicGrain extension remain readable: the cadence window starts fresh, and
local DTG estimates are rebuilt on the first dust update. A restart containing
the cadence block but not the cache also remains readable, with the cache
rebuilt.

`DustUpdateCadence` itself is intentionally **not** taken from that block. The
active value comes from the parameter file when the executable starts.

This mechanism fixes an important historical failure mode in which a restarted
run could lose the pending dust-update state and subsequently evolve dust
incorrectly.

## Snapshot continuation

When starting from a snapshot (`RestartFlag=2` in the GADGET-4 interface),
CosmicGrain performs additional dust safeguards. The current source:

1. recomputes the global maximum particle ID;
2. reassigns live dust IDs to a unique range above the maximum non-dust ID;
3. applies a very small integer-position jitter to dust particles to avoid tree
   failures from coincident positions; and
4. checks dust state integrity before and after the initial domain decomposition.

Snapshot continuation is therefore supported, but it is not numerically
identical to resuming from a complete binary restart set.

## Regression checklist

After changes to feedback, particle exchange, I/O, or dust integration:

1. stop a short dust-bearing run at a known synchronization point;
2. resume from its binary restart files;
3. compare gas/star/dust counts and masses across the boundary;
4. verify `GrainRadius`, `CarbonMassFraction`, `DustTemperature`, `DustSource`,
   and `BirthPos` remain valid;
5. verify the dust-cadence state resumes rather than silently resetting; and
6. continue long enough to cross at least one subsequent dust update.
