# Shock Destruction

## Overview

<div class="cg-card summary" markdown="1">

Shock destruction erodes and destroys dust grains caught in the blast wave of a supernova, returning eroded mass to the gas phase as metals. Alongside thermal sputtering, it is one of the two dominant destruction channels in the CosmicGrain dust lifecycle, and is typically the primary regulator of dust survival in the immediate vicinity of star-forming regions, where SN shocks sweep through recently enriched gas before it has time to disperse.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain models the SN blast wave using the Sedov-Taylor self-similar solution (Sedov 1959), using the true event energy passed from the triggering feedback event (McKee & Ostriker 1977 motivates the canonical 10⁵¹ erg single-SN scale, but a stellar particle typically represents many SNe fired as one event, so the energy used here scales accordingly). The physical shock radius at a characteristic post-explosion time is

<div class="cg-equation" markdown="1">

\[
R(t) = \xi \left(\frac{E t^2}{\rho}\right)^{1/5}, \qquad \xi = 1.033
\]

</div>

and the corresponding shock velocity at that radius is

<div class="cg-equation" markdown="1">

\[
v(R) = \frac{2}{5}\,\xi^{5/2}\sqrt{\frac{E}{\rho}}\,R^{-3/2}
\]

</div>

**\(R(t)\) and \(v(R)\) always use the same explosion energy \(E\)** — both are derived from the true per-event energy, converted once from code units to erg. This matters because the two quantities only form a valid Sedov-Taylor pair when computed from a shared \(E\); using a fixed canonical value for one and the true event energy for the other would silently decouple the radius and velocity from each other.

Grain destruction efficiency as a function of shock velocity and composition follows a piecewise-linear interpolation of Bocchio et al. (2014, Table 6), blended between silicate and carbonaceous curves by carbon fraction. Carbonaceous grains are far more vulnerable to weak shocks (77% destruction efficiency at 50 km/s) than silicates (2% at the same velocity); both converge toward complete destruction above roughly 175–200 km/s.

**Grain-size vulnerability.** Within the mass earmarked for destruction, smaller grains are weighted as more vulnerable than larger ones (more surface area per unit mass): a factor of 1.5× for grains below 20 nm, 1.2× for 20–50 nm, and 0.7× for grains above 100 nm, relative to a baseline of 1.0 in between. Total destroyed mass is unaffected by this weighting — it only changes *which* grains within the search volume lose that mass, redistributing loss toward smaller, more vulnerable grains and away from larger, more resilient ones.

**Subgrid volume correction.** At the resolutions CosmicGrain typically runs at, the physical Sedov radius (tens of parsecs) is much smaller than the spatial hash cell used to search for nearby dust (order kpc). A direct grain-by-grain search at the physical radius would usually find zero grains per SN event. CosmicGrain instead computes the total dust mass expected to be destroyed within the true physical shock volume, then distributes that expected mass across all dust found in a larger search volume — weighted by each grain's size-adjusted share of the local dust mass, per the vulnerability weighting above. This keeps the total destroyed mass converged across resolutions: at coarse resolution few grains are found but the volume correction factor is near unity; at fine resolution many grains are found but the correction factor is correspondingly small.

### Assumptions

- The shock is evaluated at a single characteristic post-explosion time (0.3 Myr) rather than tracked continuously through its evolution.
- A gas-density floor is applied when computing the Sedov radius, preventing the radius from collapsing to zero in very dense star-forming gas — the floor is a numerical safeguard for resolution independence, not a claim about the physical density at the explosion site.
- Shock velocity used to set the destruction efficiency is always derived from the true physical shock radius, never from the (typically larger) search radius used to find dust — using the inflated search radius would understate the velocity and artificially suppress destruction.
- All grains within the search volume see the same shock velocity; there is currently no attenuation of velocity by an individual grain's distance from the SN within that volume.
- Destroyed and eroded dust mass is returned to the nearest gas particle as metals, consistent with shock destruction being an erosive process rather than a sink.
- Grain mass and radius are updated self-consistently via \(m \propto a^3\) for grains that are eroded rather than fully destroyed.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Shock destruction is triggered once per SN feedback event via `destroy_dust_from_sn_shocks()`, which implements the subgrid, volume-corrected, size-weighted treatment described above. It computes the physical shock radius and velocity from the local gas density and the true event energy, sets an effective search radius (at least twice the dust hash cell size, capped at 3 kpc), and finds all dust particles within that radius. The expected destroyed mass is distributed across those particles in proportion to their size-adjusted share of the local dust mass; grains that would be eroded below the minimum grain size are fully destroyed, and survivors have their radius reduced according to mass-conserving scaling.

Carbon and silicate destruction budgets are evaluated separately and
distributed using the corresponding component mass and grain-size weighting.
Survivors receive updated mass, radius, and carbon fraction. The obsolete
standalone `erode_dust_grain_shock()` path has been replaced by the unified
volume-corrected implementation and shared
`destroy_dust_particle_to_gas()` cleanup path.

### Numerical floors

- The physical shock radius is floored at 0.001 kpc (1 pc) to avoid degenerate values.
- The gas density used in the Sedov radius calculation is capped via `DustShockAmbientDensity`, preventing the radius from vanishing in very dense gas.
- The effective search radius is floored at twice the dust hash cell size and capped at 3 kpc.
- The event energy is validated before use; an invalid or non-positive value falls back to the canonical 10⁵¹ erg with a logged warning — this should not trigger in a well-formed call from `feedback.cc`.
- If no dust particles are found within the search radius, the event is a silent no-op — expected behavior in dust-sparse regions, not an error.

### Algorithm

1. **A stellar feedback event with associated SN energy occurs.**
2. **The local gas density is sampled** near the SN, with a density cap applied for numerical stability.
3. **The event energy is converted from code units to erg**, with a validated fallback.
4. **The physical Sedov-Taylor shock radius and velocity are computed** at the characteristic post-explosion time, from the same event energy.
5. **An effective search radius is set**, at least twice the dust hash cell size and capped at 3 kpc.
6. **Dust particles within the search radius are located** via the spatial hash.
7. **The volume correction factor is computed** from the ratio of the physical shock volume to the search volume.
8. **The expected destroyed mass is calculated** from the local dust mass, the volume correction, and the Bocchio et al. (2014) destruction efficiency at the physical shock velocity.
9. **Carbon and silicate destruction are distributed separately**, weighted
   by species mass and grain-size vulnerability; particles reduced below the
   minimum size are destroyed, while survivors receive updated mass, radius,
   and CF.
10. **Dust diagnostics and bookkeeping are updated.**

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableShockDestruction` | Enables or disables SN shock destruction. |
| `DustShockAmbientDensity` | Density cap used when computing the Sedov shock radius, preventing radius collapse in dense star-forming gas. |
| `DustMinGrainSize` | Minimum grain radius; grains eroded below this in a shock are fully destroyed (shared across all dust physics modules). |

The characteristic evaluation time (0.3 Myr), the Sedov constant (ξ = 1.033), the Bocchio et al. (2014) efficiency curve coefficients, and the grain-size vulnerability factors (1.5× / 1.2× / 1.0 / 0.7×) are hardcoded and not currently exposed as parameters. Event energy is no longer hardcoded — it is passed through from the triggering feedback event.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Primary shock destruction routine**
  - `destroy_dust_from_sn_shocks()`
    - Computes the Sedov-Taylor shock radius and velocity from the true event energy, applies the subgrid volume correction, and distributes destruction/erosion across dust in the search volume, weighted by grain-size vulnerability.
  - `calculate_sn_shock_radius()`
    - Computes the physical Sedov-Taylor shock radius at a given post-explosion time and energy.
  - `calculate_sedov_velocity_from_radius()`
    - Computes the Sedov-Taylor blast-wave velocity at a given physical radius and energy — energy is passed explicitly so it always matches the radius calculation's energy.
  - `get_shock_destruction_efficiency()`
    - Piecewise-linear interpolation of Bocchio et al. (2014, Table 6) destruction efficiency, blended by carbon fraction.
  - `shock_size_factor()`
    - Shared grain-size vulnerability weight (1.5× / 1.2× / 1.0 / 0.7× by grain radius), used to bias per-grain mass-loss distribution toward smaller, more vulnerable grains.

- **Shared destruction path**
  - `destroy_dust_particle_to_gas()` — invoked for grains eroded below the minimum grain size; returns remaining mass to the nearest gas particle as metals.
  - Bookkeeping: `NDustDestroyedByShock`, `TotalMassDestroyedByShock`, `TotalMassErodedByShock`.

### `src/cooling_sfr/feedback.cc`

- **Stellar feedback interface**
  - Invokes `destroy_dust_from_sn_shocks(Sp, p, E_code, comm)` following SNII feedback events, passing the true per-event energy in code units.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to SN blast-wave physics and grain destruction in shocks.*

### Sedov (1959)

**Sedov, L. I. (1959). *Similarity and Dimensional Methods in Mechanics.* New York: Academic Press.**

<p class="reference-note">
Source of the self-similar blast-wave solution CosmicGrain uses to compute the shock radius and velocity.
</p>

---

### McKee & Ostriker (1977)

**McKee, C. F., & Ostriker, J. P. (1977). *A Theory of the Interstellar Medium: Three Components Regulated by Supernova Explosions in an Inhomogeneous Substrate.* The Astrophysical Journal, 218, 148–169.**

<p class="reference-note">
Establishes the canonical 10⁵¹ erg single-supernova energy scale used as a fallback when no valid per-event energy is passed.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1977ApJ...218..148M" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/155667" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Jones, Tielens, Hollenbach & McKee (1994)

**Jones, A. P., Tielens, A. G. G. M., Hollenbach, D. J., & McKee, C. F. (1994). *Grain Destruction in Shocks in the Interstellar Medium.* The Astrophysical Journal, 433, 797.**

<p class="reference-note">
Foundational motivation for the grain-size and composition dependence of
shock processing; the active CosmicGrain destruction efficiencies are taken
from the Bocchio et al. prescription described above.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1994ApJ...433..797J" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/174689" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Jones, Tielens & Hollenbach (1996)

**Jones, A. P., Tielens, A. G. G. M., & Hollenbach, D. J. (1996). *Grain Shattering in Shocks: The Interstellar Grain Size Distribution.* The Astrophysical Journal, 469, 740–764.**

<p class="reference-note">
Companion study on shock-driven grain fragmentation and the resulting interstellar grain size distribution.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1996ApJ...469..740J" target="_blank" rel="noopener">📚 ADS</a>

</div>

---

### Bocchio, Jones & Slavin (2014)

**Bocchio, M., Jones, A. P., & Slavin, J. D. (2014). *A Re-evaluation of Dust Processing in Supernova Shock Waves.* Astronomy & Astrophysics, 570, A32.**

<p class="reference-note">
Source of the Table 6 destruction efficiency curves CosmicGrain interpolates directly.
</p>

<div class="reference-links">

<a href="https://www.aanda.org/articles/aa/full_html/2014/10/aa24368-14/aa24368-14.html" target="_blank" rel="noopener">📚 Article</a>
<a href="https://doi.org/10.1051/0004-6361/201424368" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
