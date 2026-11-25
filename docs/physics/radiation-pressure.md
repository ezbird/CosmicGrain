# Radiation Pressure

## Overview

<div class="cg-card summary" markdown="1">

Radiation pressure applies a velocity kick to dust grains from the momentum flux of nearby young, UV-bright stars, pushing grains outward from active star-forming regions. It is the only dust physics process in CosmicGrain driven by stellar radiation rather than gas-phase interactions, and is expected to be most significant for small grains near recent star formation, where the flux is highest and the momentum-to-mass ratio of small grains is largest.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

The radiative force on a single grain follows the standard optically-thin form,

<div class="cg-equation" markdown="1">

\[
F_{\rm rad} = \frac{L}{4\pi r^2 c}\, Q_{\rm pr}\, \pi a^2
\]

</div>

where \(L\) is the source star's luminosity, \(r\) the star–grain distance, \(a\) the grain radius, and \(Q_{\rm pr}\) the radiation pressure efficiency. CosmicGrain uses a Lorentzian-ramp parametrization for \(Q_{\rm pr}\) (Draine & Lee 1984), calibrated to a luminosity-weighted OB population,

<div class="cg-equation" markdown="1">

\[
Q_{\rm pr} = \frac{a}{a + a_0}\times\left(0.8 + 0.4\,{\rm CF}\right), \qquad a_0 = 35\,{\rm nm}
\]

</div>

so that silicate grains (CF = 0) have a lower effective absorption cross-section than carbonaceous grains (CF = 1). Contributions from every young star within the search radius are summed as vector accelerations before the velocity kick is applied, correctly superposing radiation pressure from multiple nearby sources.

**Stellar luminosity** is estimated from a simplified piecewise light-to-mass ratio as a function of stellar age, explicitly documented in the source as a placeholder pending a proper population synthesis table (Starburst99, BPASS, or FSPS) interpolated by age and metallicity.

**Terminal velocity correction.** Because dust grains are drag-coupled to the surrounding gas rather than free-flying, a grain in the terminal-velocity regime (short stopping time relative to the timestep) does not accelerate freely under radiation pressure — it drifts at whatever velocity balances the radiative force against drag. CosmicGrain applies a coupling factor,

<div class="cg-equation" markdown="1">

\[
f_{\rm coupling} = \left(1 - e^{-\Delta t/t_{\rm stop}}\right)\frac{t_{\rm stop}}{\Delta t}
\]

</div>

which approaches 1 when the timestep is much shorter than the stopping time (the grain responds freely) and suppresses the kick toward \(t_{\rm stop}/\Delta t\) when the timestep is much longer (the terminal-velocity limit). This uses the grain's true, unclamped stopping time rather than the capped value used for the drag velocity update, since the 50 Myr cap there is a numerical safeguard that should not suppress a genuine physical limit here.

### Assumptions

- Only stellar particles younger than 20 Myr are treated as radiation sources, representing the OB-dominated UV population; older stars are excluded entirely from the neighbor sum.
- \(Q_{\rm pr}\) depends only on grain radius and composition, not on the spectral type or age of the illuminating star — the same optical efficiency is applied regardless of which young star is providing the flux.
- Grain material density is fixed at the silicate value (2.4 g/cm³) for both the single-grain mass calculation and the terminal-velocity stopping time, regardless of the grain's actual carbon fraction — the same simplification used in drag coupling.
- The terminal-velocity stopping time uses fixed mean molecular weight (μ = 0.6) and adiabatic index (γ = 5/3), consistent with the same fixed values used in `calculate_drag_timescale()`.
- The optically thin limit is assumed between star and grain — no attenuation of the stellar flux by intervening gas or dust is modeled.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

At each dust update, every task's stars younger than 20 Myr are gathered
into one list with their luminosities, so grains near a domain boundary see
stars on other tasks. For each associated grain:

1. **Find young stars** within
   \(\min[\max(2H_{\rm sml},\,2\times\text{star hash cell}),\,10]\) code length
   units, where \(H_{\rm sml}\) is the associated gas particle's smoothing
   length.
2. **Sum the vector accelerations** from all of them. The star–grain distance
   is floored at the stellar softening length, since a star particle is a
   cluster; unsoftened flux gave kicks of thousands of km/s to new grains.
3. **Compute the unclamped subsonic stopping time** against the associated gas
   and the coupling factor.
4. **Apply the kick** \(\Delta v = a_{\rm rad}\,\Delta t\,f_{\rm coupling}\),
   limited to 100 km/s (physical) per update, converted to the comoving
   momentum variable.

### Known simplifications

- `stellar_luminosity()` is a placeholder piecewise light-to-mass model; the
  source recommends a population-synthesis table (Starburst99, BPASS, FSPS).
  Its curve runs to 100 Myr, but only stars younger than 20 Myr are used.
- The terminal-velocity stopping time uses the subsonic Epstein form only,
  without the supersonic correction applied in the drag update.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableRadiationPressure` | Enables or disables radiation pressure. |
| `DustRadiationPressureEfficiency` | Multiplier on the radiative acceleration; default 1. |

The 20 Myr stellar age cutoff, the 100 km/s per-update kick limit, the Lorentzian \(Q_{\rm pr}\) parametrization and its 35 nm transition scale, the silicate grain density (2.4 g/cm³), the fixed mean molecular weight (0.6) and adiabatic index (5/3) used in the terminal-velocity stopping time, and the piecewise stellar light-to-mass model are all hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Radiation pressure routine**
  - `dust_radiation_pressure()`
    - Sums the acceleration from nearby young stars, applies the terminal-velocity coupling and the kick limit, and updates the grain velocity.
  - `refresh_young_star_list()`
    - Gathers all tasks' stars younger than 20 Myr once per dust update.
  - `radiation_pressure_efficiency()`
    - Computes \(Q_{\rm pr}\) from grain radius and carbon fraction (Draine & Lee 1984 parametrization).
  - `stellar_luminosity()`
    - Placeholder age-based light-to-mass ratio model; documented in-source as needing replacement by a population synthesis table.

- **Caller**
  - `update_dust_dynamics()` calls `dust_radiation_pressure()` for each associated grain, after drag.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to grain radiation pressure efficiency and optical properties.*

### Draine & Lee (1984)

**Draine, B. T., & Lee, H. M. (1984). *Optical Properties of Interstellar Graphite and Silicate Grains.* The Astrophysical Journal, 285, 89–108.**

<p class="reference-note">
Source of the graphite/silicate optical properties underlying the Q_pr parametrization CosmicGrain uses for radiation pressure efficiency.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1984ApJ...285...89D" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/162480" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
