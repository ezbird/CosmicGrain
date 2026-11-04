# Runtime Parameters

CosmicGrain registers its dust runtime controls in `src/data/allvars.cc`. The names below were checked against the current source tree supplied for this documentation audit. The parameter file used for a simulation remains the authority for the **values** used by that run.

## Update cadence and diagnostics

| Parameter | Type | Meaning |
| --- | --- | --- |
| `DustUpdateCadence` | integer | Number of synchronized outer calls accumulated before the regular dust-physics update. The source requires a value from 1 to 1,000,000. |
| `DustDebugLevel` | integer | Dust diagnostic verbosity. |

`DustUpdateCadence` is initialized to 10 in the source before the parameter file is read; production runs should record the explicit value in their parameter file rather than relying on that implementation initialization.

## Physics switches

Each switch is an integer runtime control (`0` off, nonzero on).

| Parameter | Process |
| --- | --- |
| `DustEnableCreation` | Stellar dust creation |
| `DustEnableDrag` | Gas–dust drag |
| `DustEnableGrowth` | Gas-phase metal accretion onto grains |
| `DustEnableClumping` | Subgrid density boost used by dust processing |
| `DustEnableCoagulation` | Coagulation |
| `DustEnableShattering` | Shattering |
| `DustEnableSputtering` | Thermal sputtering / sublimation path |
| `DustEnableShockDestruction` | SN-shock destruction |
| `DustEnableAstration` | Dust removal during star formation |
| `DustEnableRadiationPressure` | Radiation pressure from young stars |
| `DustEnableCooling` | Dust thermal cooling treatment |

The startup summary in `begrun.cc` prints the state of these physics switches and is a useful run-time sanity check.

## Stellar dust creation

| Parameter | Units / role |
| --- | --- |
| `DustParticlesPerSNII` | Requested total PartType6 sampling for the SNII/HN episode of an SSP |
| `DustParticlesPerAGB` | PartType6 sampling for AGB dust creation |
| `DustParticlesPerLRN` | PartType6 sampling for the LRN channel |
| `DustYieldSNII` | SNII condensation/yield factor used by the feedback dust calculation |
| `DustYieldAGB` | AGB condensation/yield factor used by the feedback dust calculation |
| `DustLRNRatePerCCSN` | LRN rate relative to the CCSN rate |
| `DustLRNDustMassMsun` | Dust mass per LRN event, in solar masses |
| `DustVelocitySNII` | SNII birth kick, km s⁻¹ |
| `DustVelocityAGB` | AGB birth kick, km s⁻¹ |
| `DustVelocityLRN` | LRN birth kick, km s⁻¹ |
| `DustOffsetMinSNII`, `DustOffsetMaxSNII` | SNII birth-offset range, kpc |
| `DustOffsetMinAGB`, `DustOffsetMaxAGB` | AGB birth-offset range, kpc |
| `DustOffsetMinLRN`, `DustOffsetMaxLRN` | LRN birth-offset range, kpc |

The source assigns representative birth radii of 10 nm (SNII), 100 nm (AGB), and 10 nm (LRN). Those radii are currently hardcoded rather than runtime parameters. Particle sampling controls numerical representation; it should not be interpreted as a physical yield multiplier.

## Grain processing and destruction

| Parameter | Meaning |
| --- | --- |
| `DustGrowthCalibration` | Multiplicative calibration of the grain-growth timescales |
| `DustCoagulationCalibration` | Multiplicative calibration of the coagulation timescale |
| `DustShatteringCalibration` | Multiplicative calibration of the shattering timescale |
| `DustShatteringFragmentSlope` | Power-law slope \(q\) of the shattering fragment number distribution, \(dN/da \propto a^{-q}\); must be finite and positive |
| `DustShatteringMaxFragmentFraction` | Maximum fragment radius as a fraction of the parent radius; must be finite and strictly between 0 and 1 |
| `DustAstrationCalibration` | Multiplicative astration calibration |
| `DustCollisionDensityThresh` | Shared effective-density boundary separating coagulation and shattering regimes |
| `DustThermalSputteringTemp` | Gas-temperature threshold for the thermal sputtering path, K |
| `DustShockAmbientDensity` | Ambient-density cap used in the shock-radius calculation; stored as a number density in cm⁻³ and converted internally to mass density |
| `DustMinGrainSize` | Minimum allowed representative grain radius, nm |
| `DustMaxGrainSize` | Maximum allowed representative grain radius, nm |

`DustMinGrainSize` and `DustMaxGrainSize` are shared limits used by multiple processing paths. There is **no** separate `DustCoagulationMaxSize` parameter in the audited source.

## Thermal and radiation coupling

| Parameter | Meaning |
| --- | --- |
| `DustAccommodationCoeff` | Thermal accommodation coefficient used in gas–grain energy exchange |
| `DustRadiationPressureEfficiency` | Multiplicative efficiency applied to the radiation-pressure acceleration |

## Related non-dust parameters

Several GADGET-4 / galaxy-physics parameters materially affect the environment in which dust evolves. In particular, `CritPhysDensity` is the star-formation density threshold and is distinct from `DustCollisionDensityThresh`. Gas smoothing lengths, gravitational softenings, timestep controls, PM settings, and feedback parameters should be archived with any reproducible run.

`MinGasHsml` deserves particular attention in resolution studies: a smoothing-length floor can alter the resolved high-density gas distribution and thereby propagate into star formation and dust evolution. It is a numerical-resolution choice, not a dust-physics calibration parameter.

## Reproducibility

For a published or shared run, archive the complete parameter file and compile configuration. This page documents parameter meaning; it intentionally does not define a universal set of production values.
