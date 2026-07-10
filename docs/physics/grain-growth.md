# Grain Growth

## Overview

<div class="cg-card summary" markdown="1">

Grain growth increases dust mass by accreting gas-phase metals directly onto existing grains, following Hirashita & Kuo (2011, HK11). Alongside coagulation, it is one of the two mechanisms by which CosmicGrain builds up dust mass beyond what is injected by stellar sources, and is typically the dominant channel for reconciling short stellar dust-production timescales with the larger dust masses observed in evolved, metal-enriched galaxies. Growth is most efficient in dense, cold, metal-rich gas, and is capped by a redshift-dependent maximum dust-to-metal ratio to prevent runaway growth in metal-poor early-universe conditions.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain uses the HK11 accretion timescale,

<div class="cg-equation" markdown="1">

\[
\tau_{\rm acc} = \tau_0 \left(\frac{a}{0.1\,\mu{\rm m}}\right)\left(\frac{Z_\odot}{Z}\right)\left(\frac{1000\,{\rm cm^{-3}}}{n_H}\right)\left(\frac{50\,{\rm K}}{T}\right)^{1/2}\left(\frac{0.3}{S}\right)
\]

</div>

where

- \(a\) is the grain radius, \(Z\) the gas metallicity, \(n_H\) the (clumping-boosted) hydrogen number density, \(T\) an assumed effective grain surface temperature, and \(S\) the sticking coefficient,
- \(\tau_0\) is a species-dependent prefactor: \(6.30\times10^7\) yr for silicates, \(5.59\times10^7\) yr for carbonaceous grains,
- species is assigned by a hard threshold on carbon fraction (CF ≥ 0.5 → carbon, otherwise silicate), rather than a continuous compositional blend.

Grain radius grows continuously,

<div class="cg-equation" markdown="1">

\[
a_{\rm new} = a \times \exp\!\left(\frac{f_{\rm mol}\,\Delta t}{\tau_{\rm acc}}\right)
\]

</div>

where \(f_{\rm mol}\) is an effective molecular-gas fraction that boosts accretion in denser, colder, more metal-rich environments where the gas is more molecular. The corresponding mass change is computed from the exact cubic relation \(dm = M_{\rm dust}\times[(a_{\rm new}/a)^3 - 1]\), not the linear approximation \(dm \approx 3M_{\rm dust}(da/a)\) — the linear form was found to overestimate growth when \(da/a\) is not small, previously producing a spurious dust mass leak during peak star formation.

Growth is bounded by a redshift-dependent maximum dust-to-metal ratio,

<div class="cg-equation" markdown="1">

\[
\left(\frac{M_{\rm dust}}{M_{\rm metals}}\right)_{\rm max} = \max\!\left(0.05,\ \frac{0.5}{1+0.15z}\right)
\]

</div>

which interpolates from ~0.05 at high redshift (z ≈ 6) to 0.5 at z = 0, preventing unphysical runaway growth in metal-poor early-universe gas.

### Assumptions

- Growth is evaluated per dust particle against its already-paired nearest gas neighbor (established by the caller); no independent grain–grain or grain–gas search occurs inside this routine.
- The effective grain surface temperature used in the accretion timescale is a fixed 20 K, independent of the particle's actual tracked dust temperature (`DustTemperature`, computed elsewhere via radiative equilibrium).
- Sticking coefficient is a fixed 0.3, independent of grain size or composition.
- Composition is treated as a binary switch (silicate vs. carbon) at CF = 0.5 for the purpose of selecting the HK11 prefactor, unlike sputtering and shattering, which blend composition continuously.
- The molecular fraction \(f_{\rm mol}\) is an observationally-motivated proxy binned by density/star-formation state, not a computed H₂ fraction.
- Accreted metal mass is removed from the gas particle's metallicity, conserving total metal mass between gas and dust phases.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Growth is gated by four sequential checks — hot gas, insufficient metals, low density, and the D/Z cap — before the accretion timescale and size update are computed. Up to three separate caps (D/Z ratio, metal availability, and a per-step growth limit) can each reduce the mass a grain is allowed to accrete; the final grain radius is derived exactly from whichever mass survives all three, applied once at the end, so radius and mass always remain a self-consistent \(m \propto a^3\) pair regardless of which cap ends up binding.

### Early D/Z pre-filter (intentional, not a bug)

Before computing the redshift-dependent cap, a cheap early-exit check compares dust and gas particle masses directly using a hardcoded 0.5 rather than the true cap value. Since the true cap `max_dust_to_metal` is always ≤ 0.5 for any redshift ≥ 0 (0.5 being its value exactly at z = 0), this pre-filter can never incorrectly reject a grain that would go on to pass the real, more expensive check — it only ever saves work in cases guaranteed to fail regardless of redshift.

### An unreachable failure path

The low-\(f_{\rm mol}\) failure branch (`f_mol < 0.01`) can never currently fire: \(f_{\rm mol}\) is initialized at a 0.05 floor and is only ever multiplied upward by the metal-rich boost, never downward. The associated diagnostic counter will always read zero in practice.

### A known inconsistency in the D/Z-capped growth path

The general growth update uses the exact cubic mass relation described above, fixing a previously identified mass-leak bug. However, the fine-grained clipping applied when a grain's growth would overshoot the D/Z cap mid-timestep still recomputes the radius using the older linear approximation (\(da/a \approx dm/(3M_{\rm dust})\)) rather than the exact cubic inverse. This means grains that hit the D/Z cap exactly mid-step are still subject to the same small inaccuracy the cubic-form fix was meant to eliminate elsewhere — worth revisiting alongside any future growth-path changes.

### Numerical floors

- Accretion timescale is clamped to [10⁶, 5×10⁹] yr.
- The clumping-boosted density used in the timescale calculation is floored at 10⁻³ cm⁻³ as a purely numerical safeguard.
- Accreted mass can never exceed 99% of the gas particle's available metal mass.
- Mass gain per call is capped at 20% of the dust particle's current mass.
- Multiple corruption guards detect and remove grains with unphysical state (zeroed fields from domain exchange, or NaN/negative values from upstream numerical issues) before they can propagate into the growth calculation.


### Algorithm

1. **Gate on gas temperature** — hot gas above the sputtering threshold inhibits accretion entirely.
2. **Gate on gas metallicity** — gas with negligible metal content cannot supply growth.
3. **Gate on local density** — diffuse gas below a minimum clumping-boosted density is skipped early.
4. **Apply the cheap D/Z pre-filter**, then the exact redshift-dependent D/Z cap.
5. **Determine the molecular fraction** \(f_{\rm mol}\) from local density, star-forming state, and metallicity.
6. **Validate the dust particle's state**, removing corrupted particles if found.
7. **Compute the HK11 accretion timescale** from grain size, density, metallicity, and composition-dependent prefactor.
8. **Compute a candidate grain radius and mass change** via the exponential growth form and the exact cubic mass relation.
9. **Apply the D/Z cap, metals-availability cap, and 20%-per-step cap to the mass change, in sequence**, then derive the final grain radius exactly from whichever mass survives all three caps.
10. **Apply the mass and metallicity updates** to the dust and gas particles, and update diagnostics.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableGrowth` | Enables or disables grain growth via accretion. |
| `DustGrowthCalibration` | Multiplicative calibration factor applied to the HK11 accretion timescale. |
| `DustMinGrainSize` | Minimum grain radius (shared across all dust physics modules). |
| `DustMaxGrainSize` | Maximum grain radius reachable via growth (shared across all dust physics modules). |

The HK11 prefactors (6.30×10⁷ / 5.59×10⁷ yr), the effective grain surface temperature (20 K), the sticking coefficient (0.3), the CF = 0.5 composition threshold, the molecular-fraction density bins, and the D/Z cap's functional form and floor (0.05) are hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Growth routine**
  - `dust_grain_growth_subgrid()`
    - Applies all gating checks, computes the HK11 accretion timescale, updates grain radius and mass via the exact cubic relation, and enforces the D/Z cap.
  - `tau_acc_yr_HK11()`
    - Evaluates the HK11 accretion timescale formula for a given density, temperature, metallicity, grain size, sticking coefficient, and species.
  - `dust_clumping_factor()`
    - Supplies the subgrid density enhancement used to compute the clumping-boosted \(n_{\rm eff}\) that gates growth and sets \(f_{\rm mol}\).
  - Bookkeeping: `NGrainGrowthEvents`, `TotalMassGrown`.

### `src/dust/update_dust_dynamics()`

- **Caller**
  - Invokes `dust_grain_growth_subgrid()` for each live dust particle within 2 kpc of its nearest gas neighbor, on the same cadence as drag, sputtering, and other dust physics.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to interstellar grain growth via accretion.*

### Hirashita & Kuo (2011)

**Hirashita, H., & Kuo, T.-M. (2011). *Effects of Grain Size Distribution on the Interstellar Dust Mass Growth.* Monthly Notices of the Royal Astronomical Society, 416, 1340–1353.**

<p class="reference-note">
Source of the accretion timescale formula CosmicGrain implements directly (HK11).
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2011MNRAS.416.1340H" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1111/j.1365-2966.2011.19131.x" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Asano, Takeuchi, Hirashita & Nozawa (2013)

**Asano, R. S., Takeuchi, T. T., Hirashita, H., & Nozawa, T. (2013). *What Determines the Grain Size Distribution in Galaxies?* Monthly Notices of the Royal Astronomical Society, 432, 637–652.**

<p class="reference-note">
Extends the HK11 accretion framework alongside coagulation, shattering, and destruction to model the evolving grain size distribution — the broader framework CosmicGrain's combined growth and coagulation modules draw on.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2013MNRAS.432..637A" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1093/mnras/stt506" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
