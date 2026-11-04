# Resolution and Gas Smoothing Length

CosmicGrain's dust predictions are coupled to the resolved gas density field.
A numerical gas smoothing-length floor can therefore propagate into star
formation, enrichment, and dust production even though `MinGasHsml` is not a
dust-physics parameter.

## Current implementation

The SPH density solver applies `MinGasHsml` only when its value is greater than
zero. A value of zero therefore disables the explicit floor; the normal
neighbor solution and other hydrodynamic/gravitational limits still apply.

The current source also contains an optional diagnostic for particles that hit
the floor, which is useful when establishing whether the parameter is actually
active in a run.

## CosmicGrain resolution tests

Dedicated CG10 tests showed that this parameter can be scientifically
important:

- at 512³-equivalent resolution, `MinGasHsml=1.0` produced floor hits and a
  substantially lower present-day SFR than 0.5, 0.25, or 0.0; the latter three
  runs were effectively identical in the tested diagnostics;
- at 1024³-equivalent resolution, lowering the floor from 1.0 to 0.25 removed
  the floor-limited gas population and increased the high-density tail, SFR,
  and surviving dust mass;
- a 1024³ run with `MinGasHsml=0.0` is therefore used as the primary CG10
  analysis realization while higher-resolution behavior continues to be
  checked explicitly.

These results should **not** be interpreted as a universal physical calibration
of `MinGasHsml`. They demonstrate that an imposed smoothing-length floor can
become an active resolution limit and must be audited rather than inherited
unchanged between resolutions.

## Recommended validation

For each production resolution:

1. record the minimum and percentile distribution of gas smoothing lengths;
2. report the fraction of relevant gas at the explicit floor;
3. compare the high-density tail (`nH`, SFR-producing gas) between candidate
   floor values;
4. compare stellar mass, SFR, and dust mass at matched outputs; and
5. only then adopt a production value.

A floor that is never reached should not change the solution. A floor that is
frequently reached is part of the effective numerical resolution and should be
reported with the simulation setup.
