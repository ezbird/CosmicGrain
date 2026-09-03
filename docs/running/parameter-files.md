# Parameter Files

Treat the parameter file as part of the scientific result. Copy it into the
output provenance record before launch.

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

The startup `[DUST_FLAGS]` diagnostic must match the intended physics. Missing
or misspelled parameter names should be corrected before production.

## Controlled experiments

Change one parameter at a time when the goal is causal interpretation. The
current star-formation-threshold experiment changes only `CritPhysDensity`
from 0.7 to \(0.1\,\mathrm{cm^{-3}}\). `MaxSfrTimescale` and
`DustCollisionDensityThresh` remain unchanged in that test.

See [Runtime Parameters](../reference/parameters.md) for CosmicGrain-specific
conventions.
