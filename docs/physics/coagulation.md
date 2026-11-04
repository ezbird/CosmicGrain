# Coagulation

## Overview

<div class="cg-card summary" markdown="1">

Coagulation increases the representative grain radius of dust superparticles by modeling the statistical effect of frequent grain-grain collisions in dense, cold gas, without changing superparticle mass or computational particle count. This is the key distinction from grain growth: growth adds new mass by accreting gas-phase metals, while coagulation changes the representative size at fixed dust mass, as a continuous closure for microscopic grains sticking together in molecular clouds. Coagulation and shattering are complementary size-transfer mechanisms gated by the same density threshold, active in opposite density regimes.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain uses the Trayford et al. (2025) coagulation-timescale scaling,
adapted to its continuous representative-grain radius,

<div class="cg-equation" markdown="1">

\[
\tau_{\rm co} = 27.1\,{\rm Myr}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{v_{\rm co}}{0.1\,{\rm km\,s^{-1}}}\right)^{-1}
\left(\frac{n_{\rm H,eff}}{10\,{\rm cm^{-3}}}\right)^{-1}
\mathcal{C}_{\rm co}
\]

</div>

Here \(v_{\rm co}=0.2\,{\rm km\,s^{-1}}\), \(n_{\rm H,eff}\) is the associated
gas particle's hydrogen density multiplied by the shared clumping factor, and
\(\mathcal{C}_{\rm co}\) is `DustCoagulationCalibration`. D/G is the dust-mass
sum divided by the gas-mass sum in the dust particle's resolved association
sphere. Contributions are queried across MPI ranks using the gas and dust
spatial hashes. The ratio is clamped to \([10^{-6},0.1]\). This is a local
particle-sampled mass ratio, not a kernel-weighted density ratio.

Trayford's equation uses a small-grain radius in its two-size-bin model.
CosmicGrain intentionally retains its continuous grain radius and substitutes
that representative radius into the timescale. With the closure
\(da/dt=a/(3\tau_{\rm co})\), and \(\tau_{\rm co}\propto a\) at fixed
environment, the continuous-radius update is

<div class="cg-equation" markdown="1">

\[
a_{\rm new} = \min\!\left[a_{\max},
a\left(1+\frac{\Delta t}{3\tau_{\rm co}(a)}\right)\right].
\]

</div>

The update is subcycled so no numerical substep grows the radius by more than
20%; the full interval is integrated unless the shared `DustMaxGrainSize`
ceiling is reached. The subcycling preserves the fixed-environment continuous
solution and does not change the superparticle's mass or composition.

### Assumptions

- Trayford et al. do not specify a separate temperature cutoff in the coagulation prescription. CosmicGrain treats star-forming effective-EOS gas as the unresolved cold-phase proxy and applies the 3000 K ceiling only to non-star-forming gas.
- Coagulation is modeled as a continuous, deterministic process (rather than the stochastic, catastrophic treatment used for shattering) because many grain-grain collisions are assumed to occur per timestep in sufficiently dense gas.
- CosmicGrain does not explicitly consume a partner superparticle or transfer
  mass between discrete size bins. It changes representative radius at fixed
  superparticle mass as a continuous-size closure.
- No mass or metals are exchanged with the surrounding gas; coagulation is a closed process entirely internal to the existing dust population.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Coagulation is gated by effective density, temperature, grain validity, and
local D/G availability before the timescale and radius update are computed.
The timescale density uses the shared `dust_clumping_factor()` used by grain
growth and shattering. D/G is determined from the resolved association sphere
\(\max(2\,{\rm kpc},\min(H_{\rm sml},10\,{\rm kpc}))\) centered on the dust
particle. The sphere's gas and dust mass sums are combined across MPI ranks;
hash neighbor accumulation does not use a fixed-size neighbor buffer.

The spatial hash is refreshed by the feedback/hash subsystem on its normal
schedule (first build and then after the scale factor advances by
\(\Delta\log a \ge 0.002\)). The estimator therefore inherits the same
spatial-index staleness between rebuilds as other dust-neighbor operations.
D/G also remains resolution- and sampling-dependent. Each dust particle caches
its estimate and refreshes it every 10 dust updates; the cache is carried with
the particle during domain exchange and invalidated when the particle has no
valid gas association. Consequently, the local ratio can also lag changes in
the nearby dust/gas population between refreshes.

### Shared density threshold with shattering

Coagulation requires `n_eff` to be *above* `DustCollisionDensityThresh`; shattering requires it to be *below* the same threshold. Because both modules gate on the identical parameter, they are mutually exclusive by construction for any given gas cell at a given timestep — a grain cannot simultaneously qualify for both, which is the intended physical picture (dense gas favors sticking, diffuse gas favors fragmentation) but is worth knowing since a single parameter change affects both processes' active density regimes simultaneously.

### Shared grain-size ceiling

Coagulation uses the runtime `DustMaxGrainSize` ceiling shared with other grain-size-changing processes. The audited source does not contain a separate `DustCoagulationMaxSize` parameter.

### Numerical floors

- The D/G estimate is clamped to \([10^{-6},0.1]\); neighborhoods with no
  measurable gas or dust mass are rejected and counted as D/G failures.
- Grains already at or above `DustMaxGrainSize` skip coagulation; a computed update is also capped at `DustMaxGrainSize`.
- A grain whose computed `a_new` doesn't exceed its current radius (e.g. from a degenerate timescale) is left unchanged rather than forced to grow.

### Algorithm

1. **Associate the dust particle with gas** and determine the local association sphere from the nearest gas smoothing length.
2. **Estimate local D/G** from dust and gas mass sums inside that sphere, combining contributions across MPI ranks.
3. **Compute the clumping-boosted effective density** `n_eff` for the associated gas particle.
4. **Gate on density and temperature** — `n_eff` must exceed `DustCollisionDensityThresh`; non-star-forming gas above 3000 K is excluded.
5. **Gate on D/G, grain validity, and size** — reject unavailable local ratios and invalid grains; skip grains at the maximum size.
6. **Compute the coagulation timescale** using the Trayford Eq. 10 scaling and the calibrated continuous grain radius.
7. **Subcycle the continuous-radius update** with at most 20% radius growth per substep, conserving superparticle mass and composition and respecting `DustMaxGrainSize`.
8. **Record the event** for diagnostics and update bookkeeping counters.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableCoagulation` | Enables or disables coagulation. |
| `DustCollisionDensityThresh` | Effective density threshold above which coagulation is active (and below which shattering is active — shared between the two modules). |
| `DustMaxGrainSize` | Maximum grain radius reachable via coagulation, growth, or any other size-increasing process (shared parameter, no longer coagulation-specific). |
| `DustCoagulationCalibration` | Multiplicative calibration factor applied to the coagulation timescale. |

The Trayford timescale normalization, collision velocity, D/G clamp, 3000 K non-star-forming temperature ceiling, 20% substep limit, and 10-update D/G-cache refresh interval are hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Coagulation routine**
  - `dust_grain_coagulation()`
    - Applies the physical gates and updates continuous grain radius using
      the local D/G estimate and Trayford-inspired timescale.
  - `estimate_local_dust_to_gas_ratio()`
    - Routes neighborhood queries to overlapping MPI ranks and combines
      gas/dust mass sums from their spatial hashes.
  - `dust_clumping_factor()`
    - Supplies the shared clumping-boosted effective density (documented separately).
  - `record_coagulation_event()`
    - Accumulates per-event density samples into a histogram consumed by `print_coag_histogram()` for periodic diagnostics; internals not covered here.
  - Bookkeeping: `NCoagulationEvents`.

### `src/dust/update_dust_dynamics()`

- **Caller**
  - Invokes `dust_grain_coagulation()` using the Hsml-aware association radius
    \(\max(2\,{\rm kpc},\min(H_{\rm sml},10\,{\rm kpc}))\), alongside growth
    and shattering.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*References relevant to grain-grain coagulation via dense-gas collisions.*

### Trayford et al. (2025)

**Trayford, J., et al. (2025). *Modelling the evolution and influence of dust in cosmological simulations that include the cold phase of the interstellar medium.* Monthly Notices of the Royal Astronomical Society, 545, staf2040.**

<p class="reference-note">
Source of the coagulation-timescale scaling (Eq. 10) adapted here to CosmicGrain's continuous representative-grain radius.
</p>

<div class="reference-links">

<a href="https://arxiv.org/abs/2505.13056" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/staf2040" target="_blank" rel="noopener">🔗 DOI</a>

</div>

### Asano, Takeuchi, Hirashita & Nozawa (2013)

**Asano, R. S., Takeuchi, T. T., Hirashita, H., & Nozawa, T. (2013). *What Determines the Grain Size Distribution in Galaxies?* Monthly Notices of the Royal Astronomical Society, 432, 637–652.**

<p class="reference-note">
Combines grain growth, coagulation, shattering, and destruction to model the evolving grain size distribution.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2013MNRAS.432..637A" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1093/mnras/stt506" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
