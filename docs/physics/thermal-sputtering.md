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

The grain radius then decays according to the exact exponential solution,

<div class="cg-equation" markdown="1">

\[
a_{\rm new} = a \times \exp\left(-\Delta t / \tau_{\rm sp}\right)
\]

</div>

**Composition dependence.** Carbonaceous grains have a lower binding energy (4 eV) than silicates (6 eV) and sputter proportionally faster. This is implemented as a linear blend in binding energy weighted by carbon fraction, giving a composition factor that ranges from 1 (pure silicate) to roughly 1.5 (pure carbon).

**Sublimation.** Independently of gas-driven sputtering, a grain is destroyed instantaneously if its own equilibrium dust temperature exceeds its material sublimation point, \(T_{\rm sublimate} = 1500 + 500 \times {\rm CarbonFraction}\) K — silicates sublimate near 1500 K, pure carbon grains near 2000 K.

### Assumptions

- The Tsai & Mathews (1995) approximation is valid across the full range of gas temperatures and densities a grain may encounter; no separate regime is used for very hot or very dense gas beyond the built-in clamps.
- Composition affects sputtering rate only through a binding-energy-weighted linear blend between fixed silicate and carbon endpoints.
- Sputtering is only evaluated once a grain's nearest gas neighbor exceeds a minimum temperature threshold; below that threshold the routine takes no action.
- Sublimation and gas-driven sputtering are distinct physical triggers but are treated by the same routine and are not distinguished in destruction bookkeeping — both are recorded under the same "thermal" destruction counters.
- Eroded and fully sputtered mass is returned to the nearest gas particle as metals, consistent with sputtering being a mass-conserving erosion process rather than a sink.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Thermal sputtering is evaluated for every dust particle whose nearest gas neighbor exceeds `DustThermalSputteringTemp`, following the drag update in the same particle pass. The sublimation check runs first and unconditionally, ahead of the gas-temperature gate, since it depends only on the grain's own temperature.

For grains that survive sublimation and clear the gas-temperature gate, the exact exponential erosion form is used rather than a linear approximation — required because dust physics runs on a cadence of every 10 gravity steps with a correspondingly scaled Δt, and the linear form would over-erode grains when Δt approaches τ_sp.

### Numerical floors and destruction thresholds

- The sputtering timescale is clamped to [1 Myr, 1 Gyr], preventing runaway erosion in extremely hot, dense gas and vanishing erosion in marginal conditions.
- If unphysical input (zero or negative gas density) is encountered, a safe fallback density is substituted rather than letting the timescale calculation diverge.
- A grain is fully destroyed — rather than partially eroded — if the new radius would fall at or below the minimum grain size, or if the erosion fraction reaches or exceeds unity in a single call.
- A grain is also fully destroyed if the erosion calculation produces a non-finite or non-positive new radius, logged as a numerical bug rather than a physical destruction event.

### Known implementation detail

In the partial-erosion path, the grain radius is updated before the corresponding mass-ratio sanity check is performed. If that check fails (a non-finite, negative, or implausibly large mass ratio), the function returns without updating the particle's mass — leaving the grain radius and mass briefly inconsistent until the next cleanup pass. This is a defensive guard against a rare numerical edge case rather than expected behavior, and does not occur in a well-behaved run.

### Algorithm

1. **The nearest gas particle's temperature is used to check for sputtering eligibility.**
2. **Sublimation is checked first**, independent of gas temperature: if the grain's own dust temperature exceeds its composition-dependent sublimation point, it is destroyed immediately and its mass returned to the nearest gas particle.
3. **If the gas temperature is below the sputtering threshold**, no further action is taken.
4. **The Tsai & Mathews (1995) sputtering timescale is computed**, including the composition correction, and clamped to [1 Myr, 1 Gyr].
5. **The new grain radius is computed** via the exact exponential erosion form.
6. **If the grain has eroded below the minimum size** (or the erosion fraction reaches unity), it is fully destroyed and its remaining mass returned to gas.
7. **Otherwise, the grain radius and mass are updated** according to \(m \propto a^3\), and the eroded mass is returned to the nearest gas particle as metals, updating its metallicity.
8. **Dust diagnostics and bookkeeping are updated.**

</div>

## Parameters Involved

| Parameter | Description |
|-----------|-------------|
| `DustEnableSputtering` | Enables or disables thermal sputtering. |
| `DustThermalSputteringTemp` | Minimum gas temperature above which sputtering is evaluated. |
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

### `src/dust/dust_gas_interaction()`

- **Caller**
  - Invokes `erode_dust_grain_thermal()` once the local gas temperature exceeds `DustThermalSputteringTemp`, immediately after the drag velocity update for the same dust–gas pair.

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
