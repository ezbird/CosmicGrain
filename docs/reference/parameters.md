# Runtime Parameters

All runtime parameters are registered in `register_parameters()` in
`src/data/allvars.cc` and parsed by `src/io/parameters.cc`. Every registered
tag must appear exactly once in the parameter file: a missing tag, a repeated
tag, or an unknown tag stops the run. There are therefore no code defaults; the
**Default** column below gives the v1.2.0 values in `params/param_base.txt`.
`scripts/generate_params.py` overrides the resolution-dependent entries
(softenings, `MaxSizeTimestep`, `MinimumComovingHydroSoftening`,
`CritPhysDensity`, the dust calibrations, particle counts, collision and shock
densities). Each run writes the values it used to `<parameterfile>-usedvalues`.

## Start-up checks

`src/main/run.cc` enforces the validated configuration:

- `ComovingIntegrationOn` must be 1 and `AGBFeedbackMode` must be 2, otherwise
  the run stops.
- `TimeBegin` other than 0.01 (z = 99) or `TimeMax` other than 1 (z = 0) only
  prints a warning. A snapshot restart requires `0 < TimeBegin < TimeMax`.
- `DustUpdateCadence` must be in [1, 1 000 000];
  `DustShatteringFragmentSlope` > 0; `DustShatteringMaxFragmentFraction` in
  (0, 1); injection radii > 0 and dispersions ≥ 0.

## Dust (`DUST`)

### Cadence, switches, diagnostics

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DustUpdateCadence` | 7 | Run dust microphysics every N global sync points; the elapsed physical time is accumulated and flushed before snapshots |
| `DustEnableCreation` | 1 | Stellar dust creation (SNII, AGB, LRN) |
| `DustEnableDrag` | 1 | Gas–dust drag |
| `DustEnableGrowth` | 1 | Accretion of gas-phase metals |
| `DustEnableClumping` | 2 | Sub-grid density model for coagulation, shattering and growth: 0 off; 1 legacy clumping tiers with the `DustCollisionDensityThresh` switch; 2 Blitz & Rosolowsky (2006) dense phase at n_dense = 100 cm⁻³; 3 as 2 but n_dense = (P/k)/100 K |
| `DustEnableCoagulation` | 1 | Coagulation |
| `DustEnableShattering` | 1 | Shattering |
| `DustEnableSputtering` | 1 | Thermal sputtering |
| `DustEnableShockDestruction` | 1 | SN-shock destruction |
| `DustEnableAstration` | 1 | Dust removal by star formation |
| `DustEnableRadiationPressure` | 1 | Radiation pressure from young stars |
| `DustEnableCooling` | 1 | Gas–grain thermal coupling |
| `DustDebugLevel` | 0 | Dust diagnostic verbosity |

`begrun.cc` prints the state of every switch at start-up.

### Stellar creation

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DustParticlesPerSNII` | 25 | PartType6 particles for the whole SNII/HN episode of a star particle, split over its 4 time bins (at least 1 per bin) |
| `DustParticlesPerAGB` | 25 | Maximum PartType6 particles per AGB release |
| `DustParticlesPerLRN` | 1 | PartType6 particles per LRN release |
| `DustYieldSNII` | 0.2 | Fraction of SNII metal ejecta condensed into dust |
| `DustYieldAGB` | 0.4 | Fraction of AGB metal ejecta condensed into dust |
| `DustLRNRatePerCCSN` | 0.8 | LRN rate relative to the CCSN rate |
| `DustLRNDustMassMsun` | 1e-4 | Dust mass per LRN event, M☉ |
| `DustSNIIInitialRadius` | 10 | Median SNII injection radius, nm |
| `DustSNIISizeSigmaDex` | 0.2 | Log-normal dispersion of SNII radii, dex (0 = fixed radius) |
| `DustAGBInitialRadius` | 100 | Median AGB injection radius, nm |
| `DustAGBSizeSigmaDex` | 0.2 | Log-normal dispersion of AGB radii, dex (0 = fixed radius) |
| `DustVelocitySNII` / `AGB` / `LRN` | 30 / 10 / 20 | Radial birth kick, km s⁻¹ |
| `DustOffsetMinSNII`, `DustOffsetMaxSNII` | 0.02, 0.10 | SNII spawn-offset range, kpc |
| `DustOffsetMinAGB`, `DustOffsetMaxAGB` | 0.01, 0.05 | AGB spawn-offset range, kpc |
| `DustOffsetMinLRN`, `DustOffsetMaxLRN` | 0.01, 0.05 | LRN spawn-offset range, kpc |

Injection radii are clipped to [`DustMinGrainSize`, `DustMaxGrainSize`]; LRN
grains are injected at a fixed 10 nm. The particle counts set the numerical
sampling only; they do not change the dust mass.

### Processing and destruction

The three calibrations are **timescale multipliers** (1 = literature value;
larger = slower).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DustCoagulationCalibration` | 22.3 | Multiplier on the Trayford et al. Eq. 10 coagulation timescale; τ_co ≈ 1.36 Myr × value for a = 0.1 µm, D/G = 0.01, n = 100 cm⁻³ (≈ 30 Myr at the default) |
| `DustShatteringCalibration` | 1.485 | Multiplier on the Trayford et al. Eq. 9 shattering timescale; τ_sh ≈ 54 Myr × value for a = 0.1 µm, D/G = 0.01, n_H = 1 cm⁻³ (≈ 80 Myr) |
| `DustGrowthCalibration` | 1.12 | Multiplier on both HK11 accretion timescales (carbonaceous, silicate) |
| `DustAstrationCalibration` | 1.0 | Astration removes m★ × local D/G × value, with D/G from dust and gas in the same sphere; 1 is physical |
| `DustShatteringFragmentSlope` | 3.3 | Fragment slope q, dN/da ∝ a^−q |
| `DustShatteringMaxFragmentFraction` | 0.80 | Largest fragment radius as a fraction of the parent radius |
| `DustCollisionDensityThresh` | 15 | cm⁻³. Legacy mode (`DustEnableClumping` 0 or 1) only: coagulation at clumped n_eff ≥ value, shattering at n_eff ≤ value. Not used in modes 2 and 3 |
| `DustThermalSputteringTemp` | 5e5 | K; sputtering above, no accretion above |
| `DustShockAmbientDensity` | 0.5 | cm⁻³; cap on the ambient density in the Sedov shock radius (higher = smaller radius, less destruction) |
| `DustMinGrainSize`, `DustMaxGrainSize` | 1, 1000 | Grain-radius limits, nm, shared by all processes |

### Thermal and radiation coupling

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DustAccommodationCoeff` | 0.3 | Gas–grain thermal accommodation coefficient |
| `DustRadiationPressureEfficiency` | 1.0 | Scaling of the radiation-pressure acceleration |

## Star formation and feedback

| Parameter | Default | Meaning |
| --- | --- | --- |
| `CritPhysDensity` | 0.3 | Star-formation threshold, cm⁻³ (also scales the legacy clumping tiers). Distinct from `DustCollisionDensityThresh` |
| `CritOverDensity` | 57.7 | Minimum overdensity for star formation |
| `MaxSfrTimescale` | 1.5 | Springel & Hernquist (2003) gas consumption timescale |
| `FactorSN`, `FactorEVP` | 0.1, 1000 | SH03 effective-EOS supernova fraction and cloud evaporation |
| `TempSupernova`, `TempClouds` | 1e8, 1000 | SH03 effective-EOS hot and cold phase temperatures, K. The SN heating temperature of the feedback is separate (10⁷ K, see `CG_FEEDBACK_DELTA_T`) |
| `SNEfficiency` | 0.25 | Fraction of the SNII/HN energy injected |
| `HypernovaFraction` | 0.5 | Hypernova fraction of massive stars |
| `SNIIYieldFile`, `HypernovaYieldFile` | `data/*_kobayashi2006.txt` | Massive-star yield tables |
| `AGBYieldFile` | `data/agb_yields_huscher2025.txt` | AGB yield table |
| `AGBFeedbackMode` | 2 | Must be 2: age-dependent return of total AGB ejecta (metals, gas, dust) |
| `AGBFeedbackIntervalMyr` | 5 | Minimum stellar-age step between AGB releases, Myr |
| `StarformationDebugLevel`, `FeedbackDebugLevel` | 0 | Diagnostic verbosity |
| `TreecoolFile`, `MetalcoolFile` | `data/TREECOOL_fg_dec11`, `data/METALCOOL` | UV background and metal-cooling tables (`COOLING`) |

## Additions to GADGET-4 numerics

| Parameter | Default | Meaning |
| --- | --- | --- |
| `CourantFac_z20`, `_z10`, `_z2`, `_z0` | 0.5, 0.4, 0.3, 0.2 | Courant factor for z > 20, 20–10, 10–2 and < 2; replaces `CourantFac` during the run |
| `MinGasHsml` | 0 | Smoothing-length floor; a resolution choice, not a dust calibration |
| `MaxGasHsml` | 1000 | Smoothing-length ceiling |
| `CheckpointDebugLevel` | 0 | 1 prints and barriers at main-loop checkpoints to locate hangs |
| `SpatialHashDebugLevel` | 0 | Registered but currently unused |

## Standard GADGET-4 parameters

These keep their GADGET-4 meaning (see the GADGET-4 user guide). Defaults are
from `param_base.txt`.

| Group | Parameters (default) |
| --- | --- |
| Files | `InitCondFile`, `OutputDir`, `SnapshotFileBase` (snapshot), `OutputListFilename`, `OutputListOn` (1) |
| Run | `TimeBegin` (0.01), `TimeMax` (1.0), `ComovingIntegrationOn` (1), `ICFormat` (3), `SnapFormat` (3), `TimeLimitCPU` (1728000), `CpuTimeBetRestartFile` (7200), `MaxMemSize` (5200) |
| Cosmology | `Omega0` (0.3158), `OmegaLambda` (0.6842), `OmegaBaryon` (0.04936), `HubbleParam` (0.6732), `Hubble` (0.1), `BoxSize` (50000) |
| Output | `TimeBetSnapshot` (1.067), `TimeOfFirstSnapshot` (0.02), `TimeBetStatistics` (0.005), `NumFilesPerSnapshot` (4), `MaxFilesWithConcurrentIO` (4) |
| Time integration | `MaxSizeTimestep`, `MinSizeTimestep` (1e-9), `ErrTolIntAccuracy` (0.05), `CourantFac` (placeholder) |
| Tree | `TypeOfOpeningCriterion` (1), `ErrTolTheta` (0.7), `ErrTolThetaMax` (1.0), `ErrTolForceAcc` (0.025), `TopNodeFactor` (4.0), `ActivePartFracForNewDomainDecomp` (0.01), `ActivePartFracForPMinsteadOfEwald` (0.05; `TREEPM_NOTIMESPLIT`) |
| Units | `UnitLength_in_cm` (3.085678e21), `UnitMass_in_g` (1.989e43), `UnitVelocity_in_cm_per_s` (1e5), `GravityConstantInternal` (0) |
| SPH | `DesNumNgb` (32), `MaxNumNgbDeviation` (8), `ArtBulkViscConst` (1.0), `InitGasTemp` (169), `MinEgySpec` (0) |
| Softening | `SofteningComovingClass0–6`, `SofteningMaxPhysClass0–6`, `SofteningClassOfPartType0–6`; `GasSoftFactor` (1.5), `AdaptiveHydroSofteningSpacing` (1.05), `MinimumComovingHydroSoftening` (`ADAPTIVE_HYDRO_SOFTENING`) |
| SUBFIND | `DesLinkNgb` (32; `SUBFIND`) |

Flag-dependent GADGET-4 parameters not used by the CosmicGrain configurations
(`ViscosityAlphaMin`, light-cone, `FlushCpuTimeDiff`, NGenIC, `GridSize`,
static-halo) are registered only when their compile flag is set.

## Environment variables

These are read once at start-up; unset means the default.

| Variable | Default | Effect |
| --- | --- | --- |
| `CG_FEEDBACK_DELTA_T` | 1e7 | SN heating temperature jump, K (e.g. 3.16e7 for 10^7.5 K) |
| `CG_TIMESTEP_LIMITER` | on | `0` disables the SPH timestep limiter with wake-up |
| `CG_LIMITER_DELTA` | 2 | Maximum bin difference between SPH neighbours (dt_j ≤ 2^δ dt_i) |
| `CG_LIMITER_VSIG` | on | `0` disables the pair signal-velocity rule of the limiter |
| `CG_HYDRO_WATCH_KMS` | 3000 | Runaway-force watchdog: report a hydro half-kick larger than this, km s⁻¹ |
| `CG_HYDRO_WATCH_ENTR` | 10 | Watchdog: report an entropy change larger than this factor |
| `CG_DTG_ROUTING` | by rank count | `occupancy` or `bbox` forces the D/G query routing (occupancy from 64 ranks up); results are identical |
| `CG_DTG_ROUTE_RATIO` | 5 | D/G routing-cell size in fine-grid cells |
| `CG_TEST_LOCAL_ASTRATION`, `CG_TEST_LOCAL_SHOCK` | off | `1` restricts astration / shock destruction to dust on the same task (attribution tests only) |
| `OUTPUT_DIR` | — | Substituted when a string parameter is literally `OUTPUT_DIR` |

`CG_AGB_MPI_SELFTEST`, `CG_AGB_SSP_SELFTEST`, `CG_AGB_EVOLUTION_TEST`,
`CG_AGB_DONOR_SELFTEST`, `CG_AGB_SYNC_TEST_TIMESTEPS`, `CG_RC1_NEIGHBOR_MODE`
and `CG_RC1_PERF_CALL_LIMIT` are diagnostic switches; if any is set the run
stops at start-up.
