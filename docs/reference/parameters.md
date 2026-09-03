# Runtime Parameters

CosmicGrain combines standard GADGET parameters with runtime switches and
calibration values for individual dust processes. This page highlights the
parameters most likely to affect reproducibility; it is not a substitute for
the parameter-registration table in the source.

## Dust process switches

Archive the enabled/disabled state of creation, drag, grain growth,
coagulation, shattering, sputtering, shock destruction, astration, radiation
pressure, clumping, and dust cooling. Confirm the parsed state in the startup
`[DUST_FLAGS]` diagnostic.

## Stellar dust sampling

| Parameter | Current controlled value | Meaning |
| --- | ---: | --- |
| `DustParticlesPerSNII` | 4 | Superparticle samples across a complete SNII event |
| `DustParticlesPerAGB` | 6 | Superparticle samples per AGB event |
| `DustParticlesPerLRN` | 1 | Superparticle samples per LRN event |
| `DustVelocitySNII` | run parameter | SNII dust birth kick |
| `DustVelocityAGB` | run parameter | AGB dust birth kick |

Particle counts control numerical sampling. They do not change the physical
dust yield when tranche bookkeeping is operating correctly.

## Star formation and collisions

`CritPhysDensity` sets the physical star-formation threshold and also provides
the density scale used by the subgrid-clumping prescription. The controlled
feedback test changes only `CritPhysDensity` from 0.7 to
\(0.1\,\mathrm{cm^{-3}}\); other parameters remain fixed so the effect can be
isolated.

`DustCollisionDensityThresh` is a separate gate used by coagulation and
shattering. Do not change it implicitly when changing `CritPhysDensity`.

## Resolution-sensitive parameters

For each resolution, record gravitational softenings, target gas mass,
neighbor count, PM mesh settings, memory allowance, output cadence, and any
dust-collision thresholds. Parameters that vary with resolution must be
explicitly tabulated rather than inferred from a directory name.
