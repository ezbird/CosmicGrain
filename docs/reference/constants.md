# Source-Level Constants

Most user-facing dust controls are runtime parameters, but several numerical or physical choices remain hardcoded in the current implementation. This page records the most important ones; it is not a replacement for reading the implementation when changing the model.

## Dust bookkeeping

| Source constant / value | Current value | Role |
| --- | ---: | --- |
| `MIN_DUST_PARTICLE_MASS` | `1e-10` | General minimum-mass scale used by dust bookkeeping paths |
| `DUST_MASS_TO_DESTROY` | `1e-30` | Sentinel mass used when marking a dust particle for removal |
| `DUST_SFR_EPS` | `1e-14` | Numerical SFR threshold used to identify star-forming gas in dust processing |
| Silicate O/Mg/Si/Fe mass fractions | 0.69876708 / 0.07680773 / 0.08097301 / 0.14345218 | Adopted silicate stoichiometric partition |

The stellar-creation routine also uses a local `1e-15` minimum representative particle-mass scale when deciding how many requested superparticles can represent a newly created dust mass.

## Grain processing

The current source includes several hardcoded model choices, including:

- thermal-sputtering characteristic temperature `2e6 K` and exponent `2.5`;
- shock Sedov coefficient `ξ = 1.033`;
- radiation-pressure transition grain size `a0 = 35 nm`; the stellar flux is evaluated no closer than the star's softening length and the velocity change is limited to `100 km s⁻¹` per update (`RADP_MAX_DV_KMS`);
- grain material density `2.4 g cm⁻³` in drag/radiation calculations;
- grain-growth effective temperature `20 K`, sticking coefficient `0.3`, and reference metallicity mass fraction `0.02`;
- shattering Mach number `1`, velocity normalization `1.1 km s⁻¹`, material densities `3.5` / `2.24 g cm⁻³`, and composition-dependent `1–2 km s⁻¹` threshold;
- shattering reference timescale `54.1 Myr`, density break `1 cm⁻³`, and timescale clamps `[1 Myr, 10 Gyr]`;
- coagulation reference timescale `27.1 Myr` with `v_co = 0.2 km s⁻¹`; local D/G clamped to `[1e-6, 0.1]`;
- dense-phase model (`DustEnableClumping` ≥ 2): Blitz & Rosolowsky pressure law with `P0/k = 4.3e4 K cm⁻³` and `α = 0.92`; `n_dense = 100 cm⁻³` (mode 2) or `(P/k)/100 K` (mode 3); non-star-forming gas above `3e4 K` has no dense phase; dense or diffuse fractions below `1e-4` are skipped;
- legacy clumping factors (`DustEnableClumping = 1`): 30 for star-forming gas, otherwise 10 / 3 / 1.5 above 0.5 / 0.2 / 0.05 × `CritPhysDensity`;
- dust-cooling emissivity index `β = 2`, reference temperature `100 K`, reference radius `0.1 µm`, and reference absorption efficiency `1.3e-4`.

These values are implementation choices rather than runtime parameters. If one becomes a quantity that users are expected to vary systematically, it should preferably be promoted to a named runtime parameter and documented there.

The shattering fragment slope and maximum fragment fraction are runtime
parameters, not hardcoded constants. Their defaults are \(q=3.3\) and 0.80;
see [Runtime Parameters](parameters.md).

## Stellar feedback

| Constant | Value | Role |
| --- | ---: | --- |
| `SNII_TIME_BINS` | 4 | Delayed SNII/HN release bins per star particle |
| `CCSN_NUMBER_PER_MSUN` | 0.011 | CCSN per M☉ formed (LRN rate) |
| SN heating temperature | `1e7 K` | ΔT of stochastic SN heating; `CG_FEEDBACK_DELTA_T` overrides it |
| `FB_SKIP_HEATING_ABOVE_K` | `1e8 K` | Gas already this hot is not heated again (SN reheating guard) |
| `AGB_MAX_RELATIVE_KMS` | 3000 km s⁻¹ | AGB donors faster than this relative to the gas are treated as corrupt; ejecta take the gas velocity |
| `MIN_FEEDBACK_NEIGHBORS` | 16 | Feedback kernel is grown (×1.25, up to 4×) until it holds this many gas neighbours |
| Velocity watchdog | 3000 km s⁻¹ | Stars, grains and gas faster than this are reported or counted (`[FAST_STAR]`, `[FAST_PARTICLES]`) |

## Parallel bookkeeping

| Constant | Value | Role |
| --- | ---: | --- |
| `DTG_ROUTE_RATIO` | 5 | D/G routing-cell size in fine-grid cells (`CG_DTG_ROUTE_RATIO`) |
| `DTG_OCCUPANCY_MIN_TASKS` | 64 | Occupancy routing of D/G queries from this rank count up (`CG_DTG_ROUTING`) |
| `DUST_SELFCHECK_MAX_GRAINS` | 200000 | The all-task dust association self-check is skipped above this many grains |

## Documentation rule

Hardcoded values are particularly easy for documentation to become stale against. When modifying one of these constants, update this page and the corresponding Physics page in the same change.
