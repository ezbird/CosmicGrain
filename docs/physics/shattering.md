# Shattering

## Overview

<div class="cg-card summary" markdown="1">

Shattering fragments a dust superparticle into a population of smaller grains when turbulence-driven relative velocities between grains exceed a material-dependent threshold, following Jones et al. (1994, 1996). It is the mirror process to coagulation: both are gated by the same clumping-boosted effective density, but shattering dominates in warm, diffuse gas while coagulation dominates in dense, cold gas. Unlike coagulation's continuous, deterministic treatment, shattering is modeled as a discrete, stochastic event — a grain either survives a timestep intact or is caught in a catastrophic collision and fragmented.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

A grain shatters when the local turbulent velocity exceeds a composition-dependent threshold (Jones et al. 1994),

<div class="cg-equation" markdown="1">

\[
v_{\rm shatter} = 1 + {\rm CF} \quad {\rm km\,s^{-1}}
\]
</div>

blending linearly from 1 km/s for pure silicate to 2 km/s for pure carbon. CosmicGrain uses the local gas sound speed as a proxy for this turbulent velocity, since turbulent velocity itself is not tracked per gas particle — a conservative approximation appropriate for the warm, diffuse ISM where shattering is expected to dominate.

Given the local dust-to-gas ratio, grain size, and how far the turbulent velocity exceeds the threshold, the shattering timescale is

<div class="cg-equation" markdown="1">

\[
\tau_{\rm shat} = 10^8\,{\rm yr} \times \frac{1}{({\rm D/G})_{\rm local}} \times \left(\frac{a}{0.1\,\mu{\rm m}}\right) \times \frac{1}{1 + (v_{\rm turb} - v_{\rm shatter})/v_{\rm shatter}}
\]
</div>

Denser local dust populations shatter faster (more collision partners), larger grains take longer to catastrophically fragment (more cross-section relative to volume), and velocities well above threshold shorten the timescale further. Whether a shattering event actually occurs in a given timestep follows a Poisson process,

<div class="cg-equation" markdown="1">

\[
P_{\rm shatter} = 1 - e^{-\Delta t/\tau_{\rm shat}}
\]
</div>

When an event fires, the grain radius is reduced to \(a_{\rm new} = 0.33\,a\), consistent with the mean fragment radius of an MRN \(a^{-3.5}\) fragment distribution (Jones et al. 1996). Total superparticle mass is conserved — the superparticle continues to represent the same total dust mass, now as a population of smaller fragments.

### Assumptions

- Sound speed is used as a proxy for turbulent velocity, since per-particle turbulent velocity is not directly tracked.
- The velocity threshold blends linearly and continuously between silicate and carbon endpoints by carbon fraction, unlike grain growth's binary species classification.
- Shattering is treated as a genuinely catastrophic, discrete event (Poisson-gated), not a continuous erosion process — physically appropriate since real shattering is triggered by individual high-velocity collisions rather than gradual wear.
- If the fragmented radius would fall below the minimum grain size, the fragment population is deemed too small to survive as a distinct entity and the entire superparticle dissolves into the gas phase as metals, rather than being clamped at the floor.
- Grain-grain shattering physics in the diffuse ISM (Hirashita & Yan 2009) and shock-driven shattering physics (Jones et al. 1994, 1996) are both drawn on here — the threshold velocities and fragment-size relation come from the shock literature, while the overall timescale scaling follows the turbulence-driven treatment.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Shattering is gated by turbulent velocity, then the shared clumping-boosted effective density, then grain validity, before the timescale and stochastic draw are evaluated.

### Shared density threshold with coagulation

Shattering requires `n_eff` to be *below* `DustCollisionDensityThresh`; coagulation requires it to be *above* the same threshold (documented in the coagulation page). The two are mutually exclusive by construction for a given gas cell at a given timestep, matching the intended physical picture — turbulence-driven fragmentation dominates diffuse gas, sticking dominates dense gas.

### A bookkeeping quirk worth knowing about

When a fragmenting grain would drop below the minimum grain size and dissolves entirely, that destruction is booked under the **shock destruction** counters (`NDustDestroyedByShock`, `TotalMassDestroyedByShock`), not a shattering-specific counter. `NShatteringEvents` and `TotalSizeReductionShattering` only track successful fragmentation events that *don't* trigger full dissolution. This means the "Shock destruction" totals reported in `print_dust_statistics()` are actually shock-destroyed **plus** shatter-dissolved mass combined — worth keeping in mind when interpreting per-mechanism destruction statistics or writing up the methods, since a reader could reasonably assume "shock destruction" events came only from SN shocks.

### A possible citation error worth verifying

The source code attributes the shattering timescale formula to "Hirashita & Kuo (2011), eq. 9" — but Hirashita & Kuo (2011) is specifically about grain growth by accretion (it's the same HK11 already cited correctly for growth and coagulation) and does not contain shattering physics. The formula's actual scaling — inversely proportional to the local dust-to-gas ratio — matches the defining result of Hirashita & Yan (2009) instead. This looks like a mislabeled citation in the code comment rather than a physics error; worth confirming against the actual HK11 and HY09 papers directly before the methods paper cites one or the other.

### Numerical floors

- Shattering timescale is clamped to [1 Myr, 10 Gyr].
- Grains at or below the minimum grain size skip shattering entirely (Gate 3), rather than being evaluated and immediately failing.

### Algorithm

1. **Gate on turbulent velocity** — the local sound-speed proxy must exceed the composition-blended shattering threshold.
2. **Gate on the shared clumping-boosted effective density** — must be below `DustCollisionDensityThresh` (the diffuse regime).
3. **Gate on grain validity** — invalid or already-minimal grains are excluded.
4. **Compute the shattering timescale** from local dust-to-gas ratio, grain size, and velocity excess above threshold.
5. **Draw a Poisson event** for this timestep; if no event occurs, the grain is left unchanged.
6. **If an event occurs, compute the fragment radius** as 0.33× the current radius.
7. **If the fragment radius would fall below the minimum grain size**, dissolve the entire superparticle, returning its mass to the nearest gas particle as metals (booked as shock destruction).
8. **Otherwise, update the grain radius only** — mass is conserved — and record the event.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableShattering` | Enables or disables shattering. |
| `DustCollisionDensityThresh` | Effective density threshold below which shattering is active (and above which coagulation is active — shared between the two modules). |
| `DustShatteringCalibration` | Multiplicative calibration factor applied to the shattering timescale. |
| `DustMinGrainSize` | Minimum grain radius; fragments that would fall below this dissolve the superparticle entirely (shared across all dust physics modules). |

The 10⁸ yr / 0.1 µm reference scaling in the timescale formula, the 1–2 km/s velocity threshold endpoints, and the 0.33 fragment-radius factor are hardcoded and not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Shattering routine**
  - `dust_grain_shattering()`
    - Applies the velocity, density, and validity gates; computes the shattering timescale, draws the stochastic event, and either fragments the grain or dissolves it entirely.
  - `dust_clumping_factor()`
    - Supplies the shared clumping-boosted effective density (documented separately).
  - Shared destruction path
    - `destroy_dust_particle_to_gas()` — invoked when a fragment would fall below the minimum grain size; returns mass to gas as metals and is booked under shock destruction counters.
  - Bookkeeping: `NShatteringEvents`, `TotalSizeReductionShattering`.

### `src/dust/update_dust_dynamics()`

- **Caller**
  - Invokes `dust_grain_shattering()` for each live dust particle within 2 kpc of its nearest gas neighbor, alongside growth and coagulation, on the same physics cadence.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to turbulence- and shock-driven grain shattering.*

### Jones, Tielens, Hollenbach & McKee (1994)

**Jones, A. P., Tielens, A. G. G. M., Hollenbach, D. J., & McKee, C. F. (1994). *Grain Destruction in Shocks in the Interstellar Medium.* The Astrophysical Journal, 433, 797.**

<p class="reference-note">
Source of the composition-dependent velocity thresholds (1–2 km/s) used to gate shattering events.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1994ApJ...433..797J" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/174689" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Jones, Tielens & Hollenbach (1996)

**Jones, A. P., Tielens, A. G. G. M., & Hollenbach, D. J. (1996). *Grain Shattering in Shocks: The Interstellar Grain Size Distribution.* The Astrophysical Journal, 469, 740–764.**

<p class="reference-note">
Source of the MRN fragment-distribution result motivating the 0.33× fragment-radius factor applied on a successful shattering event.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1996ApJ...469..740J" target="_blank" rel="noopener">📚 ADS</a>

</div>

---

### Hirashita & Yan (2009)

**Hirashita, H., & Yan, H. (2009). *Shattering and Coagulation of Dust Grains in Interstellar Turbulence.* Monthly Notices of the Royal Astronomical Society, 394, 1061–1074.**

<p class="reference-note">
Likely the correct source for the shattering timescale's inverse dependence on the local dust-to-gas ratio — the code currently attributes this formula to Hirashita & Kuo (2011), which is a different paper about grain growth by accretion. Recommend verifying against both papers directly.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2009MNRAS.394.1061H" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1111/j.1365-2966.2009.14405.x" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
