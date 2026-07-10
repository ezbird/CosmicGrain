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

so that silicate grains (CF = 0) have a lower effective absorption cross-section than carbonaceous grains (CF = 1). Contributions from every young star found within the search radius are summed as vector accelerations before the velocity kick is applied, correctly superposing radiation pressure from multiple nearby sources.

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

Radiation pressure is evaluated per dust particle by searching for nearby stars via the star spatial hash, summing the radiative acceleration from every sufficiently young star found, applying the terminal-velocity coupling correction, and updating the grain's velocity.

### An age-cut inconsistency worth checking

`dust_radiation_pressure()` excludes any star older than 20 Myr *before* `stellar_luminosity()` is ever called on it. But `stellar_luminosity()`'s own piecewise model extends smoothly out to 100 Myr (declining from L/M = 20 at 40 Myr to zero by 100 Myr), and its docstring states that "particles older than 100 Myr (enforced upstream) contribute negligible UV" — implying an intended 100 Myr enforcement boundary elsewhere in the codebase. Within this function specifically, the entire 20–100 Myr portion of the luminosity curve is unreachable: those stars are filtered out by the tighter 20 Myr cut before their luminosity is ever computed. This may be intentional (a deliberately tighter OB-only cut specific to radiation pressure, separate from wherever the 100 Myr enforcement the docstring refers to actually happens), or it may be a stale boundary left over from an earlier version of one function or the other. Worth confirming whether `stellar_luminosity()` is called from any other site with a wider age range where the rest of its curve is actually used, or whether the two boundaries should be reconciled.

### The stellar luminosity placeholder

The current age-based light-to-mass model is a simplified stand-in, not intended for production-quality results — the source comments explicitly recommend replacing it with a population synthesis table. This is worth flagging as a natural pairing with any future work using BPASS or similar tables elsewhere in the pipeline (e.g. for synthetic spectra), since a shared luminosity/SED source could serve both purposes rather than maintaining two separate stellar-population treatments.

### A missing supersonic correction in the terminal-velocity stopping time

`calculate_drag_timescale()` includes a supersonic Mach-number correction on top of the subsonic Epstein form. The unclamped stopping time recomputed inline here for the terminal-velocity coupling factor only ever evaluates the subsonic form — it never applies the equivalent supersonic correction. For a grain moving supersonically relative to the local gas (plausible for freshly-kicked dust from a nearby feedback event, or dust already being accelerated by radiation pressure itself), this means the coupling factor here and the drag velocity kick applied elsewhere are computed from two different notions of the "true" stopping time. This is a second-order effect — relevant specifically in the supersonic regime — but worth reconciling if it hasn't been checked already.

### Numerical floors

- Search radius is floored at twice the star hash cell size (ensuring the search always finds candidate neighbors) and capped at 10 kpc, beyond which the inverse-square flux is considered negligible.
- Stars with non-positive computed luminosity, or grains with a non-positive computed star–grain distance, are skipped for that pair rather than contributing a degenerate acceleration.

### Algorithm

1. **Search for nearby stars** within a radius set by the local gas smoothing length and the star hash cell size, capped at 10 kpc.
2. **For each stellar neighbor found**, apply the type and 20 Myr age cut; skip stars that fail either.
3. **Compute the star's luminosity** via the piecewise age-based light-to-mass placeholder.
4. **Compute the radiative acceleration** from that star's flux, the grain's \(Q_{\rm pr}\), and its cross-sectional area, and accumulate it as a vector contribution.
5. **Compute the grain's true, unclamped stopping time** against its nearest gas neighbor (independent of the capped value used for drag).
6. **Compute the terminal-velocity coupling factor** from that stopping time and the timestep.
7. **Apply the velocity kick**, scaled by the coupling factor, from the summed acceleration of all contributing stars.
8. **Update diagnostics** tracking the fraction of calls that found any stars, found young stars specifically, and the average coupling factor and acceleration magnitude.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableRadiationPressure` | Enables or disables radiation pressure. |
| `DustRadiationPressureEfficiency` | Overall multiplicative efficiency/calibration factor applied to the computed radiative acceleration. |

The 20 Myr stellar age cutoff, the Lorentzian \(Q_{\rm pr}\) parametrization and its 35 nm transition scale, the silicate grain density (2.4 g/cm³), the fixed mean molecular weight (0.6) and adiabatic index (5/3) used in the terminal-velocity stopping time, and the piecewise stellar light-to-mass model are all hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Radiation pressure routine**
  - `dust_radiation_pressure()`
    - Searches for young stellar neighbors, sums their radiative acceleration contributions, applies the terminal-velocity coupling correction, and updates the grain velocity.
  - `radiation_pressure_efficiency()`
    - Computes \(Q_{\rm pr}\) from grain radius and carbon fraction (Draine & Lee 1984 parametrization).
  - `stellar_luminosity()`
    - Placeholder age-based light-to-mass ratio model; documented in-source as needing replacement by a population synthesis table.

### `src/dust/update_dust_dynamics()`

- **Caller**
  - Invokes `dust_radiation_pressure()` for each live dust particle when `DustEnableRadiationPressure` is set, on the same cadence as drag and other dust physics.

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
