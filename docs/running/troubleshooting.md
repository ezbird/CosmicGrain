# Troubleshooting

## Startup stops

| Message | Fix |
|---|---|
| `CosmicGrain requires cosmological integration and AGB mode 2` | Set `ComovingIntegrationOn 1` and `AGBFeedbackMode 2`. |
| `Diagnostic test switches are forbidden in the production build` | Unset the `CG_AGB_*` / `CG_RC1_*` test variables. |
| `Snapshot restart requires 0 < TimeBegin < TimeMax` | Check the snapshot time against `TimeMax`. |
| `Incompatible AGB restart ...` | The restart set was written by a different binary or `AGBFeedbackMode`; use that binary or a snapshot restart. |
| `... larger than MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY=64` | Multi-node run: rebuild with `NUMBER_OF_MPI_LISTENERS_PER_NODE=3`. |
| `WARNING: TimeBegin=...` / `WARNING: TimeMax=...` | Informational: the run is not the calibrated z = 99 → 0 setup. |

## IC rejected at startup

Confirm that all header arrays have length seven and that the empty PartType6
group contains zero-length Coordinates, Velocities, ParticleIDs, and Masses.
Run `validate_music2_ic_suite.py` before changing the simulation code.

## Particle-ID errors

Use `IDS_64BIT` and rerun the exact global uniqueness check. Never renumber an
accepted IC casually: IDs are also used to trace Lagrangian regions and match
particles across time.

## Runaway gas, stars, or grains

Several guards act at run time and report instead of stopping:

- **SN reheating:** stochastic SN heating skips gas already above 10^8 K
  (counted as `hot_skips` in `[ENERGY_LEDGER]`).
- **AGB fast donor:** ejecta from a star moving > 3000 km/s relative to the
  receiving gas are mixed at the gas velocity (`[AGB_FAST_DONOR]`).
- **Radiation pressure:** the flux is evaluated no closer than the stellar
  softening and the velocity change is capped at 100 km/s per update.
- **SPH timestep limiter:** neighbours of short-step and feedback-heated gas
  are moved to, or woken onto, shorter steps (`[TIMESTEP_LIMITER]`; on by
  default, `CG_TIMESTEP_LIMITER=0` disables it).

Warnings to read when a run goes wrong:

- `[HYDRO_WATCH]` (+ `[HYDRO_WATCH_NGB]`): a hydro half-kick changes a
  particle's velocity by > 3000 km/s or its entropy by > 10×. Not checked in
  the first 100 steps after a start; capped per rank.
- `[DENSITY_JUMP]`: converged density > 10× the drift prediction, usually an
  inactive particle overrun by a shock. Capped at 200 per rank.
- `[ENERGY_JUMP]` / `[ENERGY_HOTTEST]`: the phase that injected the energy.
- `[FAST_PARTICLES]`, `[FAST_STAR]`, `[FAST_GAS]`: objects above 3000 km/s.

If a runaway is not caught, the run usually ends with `time-step of integer
size 1 not allowed` or `integer timestep reached zero`. Resume from the last
restart set and look for these tags just before the failure.

## Collective gravity or domain-decomposition hang

Check the last successful sync point, newly spawned dust counts, timebin
assignment, rank imbalance, memory pressure, and whether all ranks reached the
same collective. `CheckpointDebugLevel 1` prints named checkpoints in the main loop with a
barrier after each. Under `srun`, use `--unbuffered`, or the
log may look stuck in initialisation. Preserve the full log rather than only
the final line.

## Out of particle slots

`dust: no room for N visiting grains` means a rank's particle arrays are full
during the distributed dust update. Raise `MaxMemSize` or use more ranks.

## Unexpected dust mass jump

Break the result down by SNII, AGB, and LRN source; inspect feedback tranche
counts; verify that `DustParticlesPerSNII` is distributed across the four SNII
tranches of an event rather than applied on each; then run the whole-box mass
and element audit.

## D/G or D/Z changes unexpectedly

Verify center, aperture, phase selection, and denominator before changing
physics. D/Z using gas-phase metals differs from D/Z using gas+dust metals.

## Slow execution

Compare the `feedback`, `fbhash`, `dust`, and `dustdtg` timers in `cpu.txt`
with the gravity and hydro timers. Count domain decompositions, FOF/SUBFIND
calls, and snapshot writes, and compare the actual particle population and
timebin distribution rather than inferring cost from nominal resolution. On
Colibri, check that the job uses UCX (`OMPI_MCA_pml=ucx`); see
[Performance](performance.md).
