# Parameter Files

Treat the parameter file as part of the scientific result. Copy it into the
output provenance record before launch; the code also writes the parsed
values to `parameters-usedvalues` in the `OutputDir`.

## Generating parameter files

`params/param_base.txt` holds the shared v1.2.0 defaults.
`scripts/generate_params.py` merges it with resolution-dependent and per-run
overrides and writes the files in `params/`:

```bash
python3 scripts/generate_params.py                # regenerate all
python3 scripts/generate_params.py halo295_512    # one halo/resolution
python3 scripts/generate_params.py --dry-run      # show diffs only
```

Make shared changes in `param_base.txt` or the script, not in the generated
files. Parameter files from v1.1.0 must be regenerated: the dust calibration
values were rescaled by 1/h and several defaults changed.

## Preflight checklist

- IC path and HDF5 format
- output directory and output list
- `TimeBegin`, `TimeMax`, and cosmology
- comoving integration and periodic boundaries
- softening classes for gas, dark matter, stars, and dust
- neighbor count and hydrodynamic kernel settings
- star-formation and feedback parameters
- all dust enable flags and calibration values
- memory, restart interval, CPU limit, and group-finding cadence

The code stops at startup unless `ComovingIntegrationOn 1` and
`AGBFeedbackMode 2`. It prints a `WARNING` if `TimeBegin` ≠ 0.01 or
`TimeMax` ≠ 1, because the calibrated runs cover z = 99 to 0; partial runs may
use other values. A snapshot restart requires 0 < `TimeBegin` < `TimeMax`.

The `[DUST_FLAGS]` line printed at the first dust update must match the
intended physics. Missing or misspelled parameter names should be corrected
before production.

## Controlled experiments

Change one parameter at a time when the goal is causal interpretation. For a
controlled star-formation-threshold comparison, change only
`CritPhysDensity` relative to the matched baseline. Keep
`MaxSfrTimescale` and `DustCollisionDensityThresh` unchanged unless the
experiment explicitly targets those parameters too.

## Environment overrides

These variables are read once at startup. Pass them to every rank
(`srun --export=ALL`, `mpirun -x`) and record them with the run.

| Variable | Default | Effect |
|---|---|---|
| `CG_FEEDBACK_DELTA_T` | `1e7` | SNII heating temperature jump in K (e.g. `3.16e7` for 10^7.5 K). There is no parameter-file equivalent. |
| `CG_TIMESTEP_LIMITER` | on | `0` disables the SPH timestep limiter and wake-up. |
| `CG_LIMITER_DELTA` | `2` | Largest allowed bin difference between SPH neighbours (dt ratio 2^delta). |
| `CG_LIMITER_VSIG` | on | `0` disables the pair signal-velocity rule of the limiter. |
| `CG_HYDRO_WATCH_KMS` | `3000` | Velocity change per half-kick (km/s) that triggers `[HYDRO_WATCH]`. |
| `CG_HYDRO_WATCH_ENTR` | `10` | Entropy change factor per half-kick that triggers `[HYDRO_WATCH]`. |
| `CG_DTG_ROUTING` | by rank count | `bbox` or `occupancy` routing of D/G queries; default is occupancy from 64 ranks up. Both give identical results. |
| `CG_DTG_ROUTE_RATIO` | `5` | Fine D/G cells per routing cell. |

The internal test switches (`CG_AGB_*_SELFTEST`, `CG_AGB_EVOLUTION_TEST`,
`CG_AGB_SYNC_TEST_TIMESTEPS`, `CG_RC1_*`) make the code stop at startup.

See [Runtime Parameters](../reference/parameters.md) for CosmicGrain-specific
conventions.
