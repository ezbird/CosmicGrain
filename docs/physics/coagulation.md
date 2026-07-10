# Coagulation

## Overview

<div class="cg-card summary" markdown="1">

Coagulation increases the representative size of a dust superparticle by modeling the statistical effect of frequent grain-grain collisions in dense, cold gas — without changing its mass. This is the key distinction from grain growth: growth adds new mass by accreting gas-phase metals, while coagulation redistributes existing dust mass into fewer, larger representative grains, consistent with real grains sticking together in molecular clouds rather than gaining material from the gas phase. Coagulation and shattering are complementary size-transfer mechanisms gated by the same density threshold, active in opposite density regimes.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain adopts a coagulation timescale in the spirit of Hirashita & Kuo (2011, HK11),

<div class="cg-equation" markdown="1">

\[
\tau_{\rm coag} \approx 10\,{\rm Myr} \times \left(\frac{100\,{\rm cm^{-3}}}{n_{\rm eff}}\right) \times \left(\frac{0.1\,\mu{\rm m}}{a}\right)
\]

</div>

where \(n_{\rm eff}\) is the clumping-boosted effective density and \(a\) is the grain radius — denser gas and smaller grains coagulate faster. Grain radius grows continuously toward larger sizes,

<div class="cg-equation" markdown="1">

\[
a_{\rm new} = a \times \min\!\left[1.2,\ \left(1 + \left(1 - e^{-\Delta t/\tau_{\rm coag}}\right)\right)^{1/3}\right]
\]

</div>

The cube-root form follows from volume conservation of the swept mass — treating the fractional volume increase as \(1-e^{-\Delta t/\tau_{\rm coag}}\), the corresponding radius increase scales as the cube root of that volume factor. **Total superparticle mass is unchanged**; only the representative radius increases, consistent with treating the superparticle as a population of grains whose mean size is growing through mutual collisions rather than through mass addition.

### Assumptions

- The 3000 K temperature ceiling used to gate coagulation is deliberately far more permissive than the true physical regime for grain-grain sticking (~10–50 K in real molecular clouds). This is an intentional compensation for SPH kernel averaging smoothing out sharp temperature contrasts at finite resolution, not a claim that coagulation is physically active up to 3000 K.
- Coagulation is modeled as a continuous, deterministic process (rather than the stochastic, catastrophic treatment used for shattering) because many grain-grain collisions are assumed to occur per timestep in sufficiently dense gas.
- No partner grain is identified, tracked, or removed — coagulation acts on a single superparticle's own radius in response to ambient density and grain size, approximating the aggregate effect of many collisions within the population the superparticle represents.
- No mass or metals are exchanged with the surrounding gas; coagulation is a closed process entirely internal to the existing dust population.
- The per-call radius growth is capped at 1.2× to prevent unphysical single-step jumps when the timestep greatly exceeds the coagulation timescale.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Coagulation is gated by density first, then temperature, then grain size, before the timescale and radius update are computed. It shares the same clumping-boosted effective density (`dust_clumping_factor()`) used by grain growth and shattering.

### Shared density threshold with shattering

Coagulation requires `n_eff` to be *above* `DustCollisionDensityThresh`; shattering requires it to be *below* the same threshold. Because both modules gate on the identical parameter, they are mutually exclusive by construction for any given gas cell at a given timestep — a grain cannot simultaneously qualify for both, which is the intended physical picture (dense gas favors sticking, diffuse gas favors fragmentation) but is worth knowing since a single parameter change affects both processes' active density regimes simultaneously.

### A cross-module size-ceiling risk worth checking

The file-level constant `DUST_MAX_GRAIN_SIZE` (200 nm, a compile-time `#define`) is used elsewhere in `dust.cc` for grain-size validity checks in other modules, while coagulation itself is bounded by the separate runtime parameter `All.DustCoagulationMaxSize`. The code's own comment states these "should match," but nothing enforces that at compile time or runtime — changing `DustCoagulationMaxSize` in the parameter file without also updating and recompiling the `DUST_MAX_GRAIN_SIZE` macro would silently desynchronize the two ceilings, potentially allowing coagulation to grow grains to a size that other modules don't expect. Worth a quick grep across parameter files and the macro definition before changing either value.

### Numerical floors

- Coagulation timescale is clamped to [10⁶, 10⁹] yr.
- Grains already at or above `DustCoagulationMaxSize` skip coagulation entirely rather than being clamped after the fact.
- A grain whose computed `a_new` doesn't exceed its current radius (e.g. from a degenerate timescale) is left unchanged rather than forced to grow.

### Algorithm

1. **Compute the clumping-boosted effective density** `n_eff` for the paired gas cell, via the shared `dust_clumping_factor()`.
2. **Gate on density** — `n_eff` must exceed `DustCollisionDensityThresh`.
3. **Gate on temperature** — gas above 3000 K is excluded.
4. **Gate on grain validity and current size** — invalid grains are skipped; grains already at the maximum coagulation size are excluded.
5. **Compute the coagulation timescale** from `n_eff` and grain radius, calibrated and clamped.
6. **Compute the new radius** via the cube-root, mass-conserving size update, capped at 1.2× per call and at `DustCoagulationMaxSize`.
7. **Update the grain radius only** — mass is left untouched.
8. **Record the event** for diagnostics and update bookkeeping counters.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableCoagulation` | Enables or disables coagulation. |
| `DustCollisionDensityThresh` | Effective density threshold above which coagulation is active (and below which shattering is active — shared between the two modules). |
| `DustMaxGrainSize` | Maximum grain radius reachable via coagulation, growth, or any other size-increasing process (shared parameter, no longer coagulation-specific). |
| `DustCoagulationCalibration` | Multiplicative calibration factor applied to the coagulation timescale. |

The 10 Myr / 100 cm⁻³ / 0.1 µm reference scaling in the timescale formula, the 3000 K temperature ceiling, and the 1.2× per-call growth cap are hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Coagulation routine**
  - `dust_grain_coagulation()`
    - Applies the density, temperature, and size gates; computes the coagulation timescale and updates grain radius via the mass-conserving cube-root relation.
  - `dust_clumping_factor()`
    - Supplies the shared clumping-boosted effective density (documented separately).
  - `record_coagulation_event()`
    - Accumulates per-event density samples into a histogram consumed by `print_coag_histogram()` for periodic diagnostics; internals not covered here.
  - Bookkeeping: `NCoagulationEvents`.

### `src/dust/update_dust_dynamics()`

- **Caller**
  - Invokes `dust_grain_coagulation()` for each live dust particle within 2 kpc of its nearest gas neighbor, alongside growth and shattering, on the same physics cadence.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to grain-grain coagulation via dense-gas collisions.*

### Hirashita & Kuo (2011)

**Hirashita, H., & Kuo, T.-M. (2011). *Effects of Grain Size Distribution on the Interstellar Dust Mass Growth.* Monthly Notices of the Royal Astronomical Society, 416, 1340–1353.**

<p class="reference-note">
Source of the coagulation timescale scaling CosmicGrain implements directly (HK11), alongside the same paper's accretion timescale used for grain growth.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2011MNRAS.416.1340H" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1111/j.1365-2966.2011.19131.x" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Asano, Takeuchi, Hirashita & Nozawa (2013)

**Asano, R. S., Takeuchi, T. T., Hirashita, H., & Nozawa, T. (2013). *What Determines the Grain Size Distribution in Galaxies?* Monthly Notices of the Royal Astronomical Society, 432, 637–652.**

<p class="reference-note">
Combines the HK11 accretion and coagulation framework with shattering and destruction to model the evolving grain size distribution — the broader framework CosmicGrain's growth and coagulation modules jointly draw on.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2013MNRAS.432..637A" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1093/mnras/stt506" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
