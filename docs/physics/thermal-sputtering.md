# Thermal Sputtering

## Overview

<div class="cg-card summary" markdown="1">

Thermal sputtering erodes dust grains through collisions with energetic gas particles in hot gas, progressively shrinking each grain and returning the eroded mass to the gas phase as metals. It is one of the principal destruction channels in the CosmicGrain dust lifecycle, and — alongside SN shock destruction — regulates how much dust survives in hot halo gas, superbubbles, and the CGM. The same routine also enforces grain sublimation, an independent temperature-driven destruction mechanism triggered when a grain's own equilibrium temperature exceeds its material sublimation point.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain implements the sputtering timescale of Tsai & Mathews (1995), which approximates the results of Draine & Salpeter (1979) and Tielens et al. (1994) for grains embedded in hot gas:

<div class="cg-equation" markdown="1">

\[
\tau_{\rm sp} = 0.17\,{\rm Gyr} \left(\frac{a_{-1}}{\rho_{-27}}\right)\left[\left(\frac{T_0}{T}\right)^{\omega} + 1\right]
\]

</div>

where

- \(a_{-1} = a / 0.1\,\mu{\rm m}\) is the grain radius in units of 0.1 μm,
- \(\rho_{-27} = \rho_{\rm gas} / 10^{-27}\,{\rm g\,cm^{-3}}\) is the local gas density,
- \(T\) is the local gas temperature and \(T_0 = 2\times10^6\) K is the characteristic sputtering temperature,
- \(\omega = 2.5\) is the power-law index of Tielens et al. (1994).

The current implementation evolves the carbonaceous and silicate mass components separately. Each component decays exponentially on its own sputtering timescale,

<div class="cg-equation" markdown="1">

\[
M_{i,\rm new}=M_{i,\rm old}\exp(-\Delta t/\tau_i),
\]

</div>

with carbon assigned the shorter timescale because of its lower adopted binding energy (4 eV versus 6 eV for silicate). The surviving component masses are recombined, `CarbonMassFraction` is updated, and the representative radius follows the surviving total mass through \(a_{\rm new}=a(M_{\rm new}/M_{\rm old})^{1/3}\). Thus repeated sputtering can make a surviving particle progressively more silicate rich.

**Sublimation.** Before sputtering, a grain is destroyed instantaneously if its own dust temperature exceeds its material sublimation point, \(T_{\rm sublimate} = 1500 + 500 \times {\rm CarbonMassFraction}\) K — silicates sublimate near 1500 K, pure carbon grains near 2000 K. In practice this never fires: `update_dust_temperature()` caps `DustTemperature` at the same sublimation temperature, and the check is a strict "greater than".

### Assumptions

- The Tsai & Mathews (1995) approximation is valid across the full range of gas temperatures and densities a grain may encounter; no separate regime is used for very hot or very dense gas beyond the built-in clamps.
- Carbonaceous and silicate mass components are sputtered separately using fixed 4 eV and 6 eV binding-energy factors; the surviving composition is recomputed after erosion.
- Sputtering and sublimation are only evaluated once a grain's nearest gas neighbor is hotter than `DustThermalSputteringTemp`.
- Sublimation and gas-driven sputtering are distinct physical triggers but are treated by the same routine and are not distinguished in destruction bookkeeping — both are recorded under the same "thermal" destruction counters.
- Eroded and fully sputtered mass is returned to the nearest gas particle as metals, consistent with sputtering being a mass-conserving erosion process rather than a sink.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Thermal sputtering is evaluated for every dust particle whose nearest gas neighbor is hotter than `DustThermalSputteringTemp`, right after the drag update in `dust_gas_interaction()`. That call sits inside the drag routine, so **sputtering and sublimation need both `DustEnableSputtering` and `DustEnableDrag`**. Sublimation is checked first, but only for grains that reach this call, i.e. in gas above the sputtering threshold.

For grains that survive sublimation and clear the gas-temperature gate, the exact exponential erosion form is used rather than a linear approximation — important because dust physics may run on a coarser cadence controlled by `DustUpdateCadence`; the linear form could over-erode grains when the accumulated update interval approaches the sputtering timescale.

Carbonaceous and silicate material are eroded separately. Carbon erodes
faster in the adopted model, so surviving mass is used to recompute both grain
radius and `CarbonMassFraction`; prolonged hot-gas exposure can drive CF below
its birth value. The lost component masses are returned to their corresponding
gas-phase element reservoirs.

### Numerical floors and destruction thresholds

- The sputtering timescale is clamped to [1 Myr, 1 Gyr], preventing runaway erosion in extremely hot, dense gas and vanishing erosion in marginal conditions.
- If unphysical input (zero or negative gas density) is encountered, a safe fallback density is substituted rather than letting the timescale calculation diverge.
- A grain is fully destroyed, and its whole remaining mass returned to the gas, if the new radius would be at or below `DustMinGrainSize`, or if the new radius or mass is non-finite or non-positive.
- Otherwise radius, mass, and carbon fraction are updated together.

### Algorithm

1. **After the drag update, gas hotter than `DustThermalSputteringTemp` triggers the call.**
2. **Sublimation is checked first**: if the grain's dust temperature exceeds its composition-dependent sublimation point, it is destroyed and its mass returned to the nearest gas particle.
3. **The sputtering timescales are computed** per species and clamped to [1 Myr, 1 Gyr].
4. **Each species decays exponentially**; the new radius follows the surviving mass.
5. **If the grain has eroded to or below the minimum size**, it is fully destroyed and its remaining mass returned to gas.
7. **Otherwise, carbon and silicate survival are applied separately**; grain
   radius, total mass, and carbon fraction are updated, and the eroded
   components are returned to gas.
8. **Dust diagnostics and bookkeeping are updated.**

</div>

## Parameters Involved

| Parameter | Description |
|-----------|-------------|
| `DustEnableSputtering` | Enables or disables thermal sputtering and sublimation (also needs `DustEnableDrag`). |
| `DustThermalSputteringTemp` | Gas temperature above which sputtering is evaluated; default \(5\times10^5\) K. Also switches off grain growth. |
| `DustMinGrainSize` | Minimum grain radius; erosion below this fully destroys the grain (shared across all destruction/erosion modules). |

The 4 eV / 6 eV binding energies used for the composition correction, and the sublimation temperatures (1500 K silicate, 2000 K carbon), are hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Sputtering and sublimation routine**
  - `erode_dust_grain_thermal()`
    - Checks sublimation, computes the Tsai & Mathews (1995) sputtering timescale with the composition correction, applies the exact exponential erosion, and either partially erodes or fully destroys the grain.
  - Shared destruction path
    - `destroy_dust_particle_to_gas()` — invoked by `erode_dust_grain_thermal()` for both sublimation and full sputtering destruction; returns the grain's remaining mass to the nearest gas particle as metals.
  - Bookkeeping
    - `NDustDestroyedByThermal`
    - `TotalMassDestroyedByThermal`
    - `TotalMassErodedByThermal`

- **Caller**
  - `dust_gas_interaction()` invokes `erode_dust_grain_thermal()` once the local gas temperature exceeds `DustThermalSputteringTemp`, immediately after the drag velocity update for the same dust–gas pair.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to thermal sputtering and grain destruction in hot gas.*

### Tsai & Mathews (1995)

**Tsai, J. C., & Mathews, W. G. (1995). *Interstellar Grains in Elliptical Galaxies: Grain Evolution.* The Astrophysical Journal, 448, 84.**

<p class="reference-note">
Source of the sputtering timescale formula CosmicGrain implements directly, approximating the destruction results of Draine & Salpeter (1979) and Tielens et al. (1994).
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1995ApJ...448...84T" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/175942" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Draine & Salpeter (1979)

**Draine, B. T., & Salpeter, E. E. (1979). *Destruction Mechanisms for Interstellar Dust.* The Astrophysical Journal, 231, 438–455.**

<p class="reference-note">
Foundational theoretical treatment of grain sputtering yields and destruction mechanisms, underlying the Tsai & Mathews (1995) approximation.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1979ApJ...231..438D" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/157206" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### McKinnon, Torrey, Vogelsberger, Hayward & Marinacci (2017)

**McKinnon, R., Torrey, P., Vogelsberger, M., Hayward, C. C., & Marinacci, F. (2017). *Simulating the Dust Content of Galaxies: Successes and Failures.* Monthly Notices of the Royal Astronomical Society, 468, 1505–1521.**

<p class="reference-note">
Cosmological implementation of thermal sputtering built on the Tsai & Mathews (1995) timescale, and an important conceptual predecessor to CosmicGrain's treatment of hot-gas dust destruction.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.1505M" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/1606.02714" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/stx467" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
