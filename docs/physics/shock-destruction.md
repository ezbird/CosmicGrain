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

Grain destruction efficiency as a function of shock velocity and composition follows a piecewise-linear interpolation of Bocchio et al. (2014, Table 6), with separate silicate and carbonaceous curves applied to each grain's two components. Carbonaceous grains are far more vulnerable to weak shocks (77% destruction efficiency at 50 km/s) than silicates (2% at the same velocity); carbonaceous grains reach complete destruction at 175 km/s, silicates level off at 67%.

**Grain-size vulnerability.** Within the mass earmarked for destruction, smaller grains are weighted as more vulnerable than larger ones (more surface area per unit mass): a factor of 1.5× for grains below 20 nm, 1.2× for 20–50 nm, and 0.7× for grains above 100 nm, relative to a baseline of 1.0 in between. Total destroyed mass is unaffected by this weighting — it only changes *which* grains within the search volume lose that mass, redistributing loss toward smaller, more vulnerable grains and away from larger, more resilient ones.

**Subgrid volume correction.** At the resolutions CosmicGrain typically runs at, the physical Sedov radius (tens of parsecs) is much smaller than the spatial hash cell used to search for nearby dust (order kpc). A direct grain-by-grain search at the physical radius would usually find zero grains per SN event. CosmicGrain instead computes the total dust mass expected to be destroyed within the true physical shock volume, then distributes that expected mass across all dust found in a larger search volume — weighted by each grain's size-adjusted share of the local dust mass, per the vulnerability weighting above. This keeps the total destroyed mass converged across resolutions: at coarse resolution few grains are found but the volume correction factor is near unity; at fine resolution many grains are found but the correction factor is correspondingly small.

Because the two components are destroyed separately, shocks change
`CarbonMassFraction`.

### Assumptions

- The shock is evaluated at a single characteristic post-explosion time (0.3 Myr) rather than tracked continuously through its evolution.
- The ambient density in the Sedov solution is capped at `DustShockAmbientDensity`, so the radius does not collapse in dense star-forming gas; this is a numerical choice, not a claim about the density at the explosion site.
- Shock velocity used to set the destruction efficiency is always derived from the true physical shock radius, never from the (typically larger) search radius used to find dust — using the inflated search radius would understate the velocity and artificially suppress destruction.
- All grains within the search volume see the same shock velocity; there is currently no attenuation of velocity by an individual grain's distance from the SN within that volume.
- Destroyed and eroded dust mass is returned to the nearest gas particle as metals, consistent with shock destruction being an erosive process rather than a sink.
- Grain mass and radius are updated self-consistently via \(m \propto a^3\) for grains that are eroded rather than fully destroyed.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Each SNII/hypernova tranche that is deposited queues one shock event at its
star, carrying the tranche energy and the nearest gas particle within 50 code
length units (which sets the ambient density and receives returned material).
After the feedback call, `erode_dust_from_sn_shocks_batch()` gathers all events
from all tasks, sums the dust in each search sphere over all tasks, and
applies the events in star-ID order, so results do not depend on the domain
decomposition. Dust created in the same feedback call is excluded.

For each event:

- \(\rho_{\rm Sedov}=\min(\rho_{\rm gas},\;{\tt DustShockAmbientDensity}\times m_p)\);
  \(\rho_{\rm gas}\) falls back to \(1\,m_p\,{\rm cm^{-3}}\) if no gas was found
  or it is below \(0.01\,m_p\,{\rm cm^{-3}}\).
- \(R_{\rm phys}=R(0.3\,{\rm Myr})\), floored at 1 pc; \(v=v(R_{\rm phys})\).
- \(R_{\rm search}=\min[\max(R_{\rm phys},\,2\times\text{dust hash cell}),\,3]\)
  code length units; \(f_{\rm vol}=(R_{\rm phys}/R_{\rm search})^3\).
- Destroyed masses \(M_{\rm C}f_{\rm vol}\epsilon_{\rm C}(v)\) and
  \(M_{\rm sil}f_{\rm vol}\epsilon_{\rm sil}(v)\), from the carbon and silicate
  dust mass in the sphere, are split over the grains by component mass times
  the size factor.
- A grain left below `DustMinGrainSize` is destroyed and its whole mass
  returned to the gas (if the event found gas; otherwise it is left
  untouched). Survivors get new mass, radius (\(a\propto m^{1/3}\)), and carbon
  fraction, and the eroded C and silicate elements go to the gas particle.
- An invalid event energy falls back to \(10^{51}\) erg with a warning.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableShockDestruction` | Enables or disables SN shock destruction. |
| `DustShockAmbientDensity` | Cap on the Sedov ambient density, in \(m_p\,{\rm cm^{-3}}\); default 0.5. Higher gives a smaller shock radius and less destruction. |
| `DustMinGrainSize` | Grains eroded below this radius are destroyed. |

The evaluation time (0.3 Myr), \(\xi=1.033\), the Bocchio et al. (2014)
efficiencies, the size factors, and the search-radius limits are hardcoded.
The event energy is the tranche's SN energy.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- `erode_dust_from_sn_shocks_batch()` applies all queued shock events collectively.
- `calculate_sn_shock_radius()` and `calculate_sedov_velocity_from_radius()` give the Sedov-Taylor radius and velocity from the same energy.
- `get_shock_destruction_efficiency()` interpolates Bocchio et al. (2014, Table 6) for carbon and silicate.
- `shock_size_factor()` gives the grain-size weight.
- Bookkeeping: `NDustDestroyedByShock`, `TotalMassDestroyedByShock`, `TotalMassErodedByShock`.

### `src/cooling_sfr/feedback.cc`

- Queues one `DustShockEvent` per deposited SNII/HN tranche and calls `erode_dust_from_sn_shocks_batch()` once per feedback call.

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
