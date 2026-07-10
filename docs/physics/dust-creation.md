# Dust Creation

## Overview

<div class="cg-card summary" markdown="1">

Dust creation initializes the CosmicGrain dust lifecycle by injecting explicit dust superparticles into cosmological simulations from stellar sources. Rather than treating dust as a passive scalar, CosmicGrain creates explicit **PartType6** particles with individual masses, grain sizes, compositions, positions, and velocities. These particles subsequently evolve through transport, grain growth, coagulation, shattering, sputtering, shock destruction, radiation pressure, and eventual astration, allowing the complete lifecycle of cosmic dust to be followed self-consistently alongside gas, stars, and dark matter.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

For each stellar feedback event, a fraction of the newly produced metals is assumed to condense into dust,

<div class="cg-equation" markdown="1">

\[
M_{\rm dust}
=
f_{\rm cond}
M_{\rm metals}
\]

</div>

where

- \(M_{\rm metals}\) is the metal mass returned by the stellar particle,
- \(f_{\rm cond}\) is the condensation efficiency,
- \(M_{\rm dust}\) is the newly created dust mass.

Separate condensation efficiencies are adopted for SNII and AGB stars.

### Assumptions

- Dust forms immediately following stellar feedback.
- Condensation efficiencies are fixed for each stellar source.
- Dust is represented by superparticles rather than individual grains.
- Each superparticle represents an ensemble of physical grains.
- Metals condensed into dust are removed from the gas phase.
- The condensed dust mass is drawn from a single nearest gas cell rather than distributed across a neighborhood.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

When stellar feedback occurs, CosmicGrain computes the metal ejecta from the stellar particle and converts a prescribed fraction into dust. The total dust mass is divided equally among a fixed number of superparticles per event.

Rather than storing dust as an abundance field on nearby gas cells, the code creates entirely new **PartType6** particles. The single nearest gas cell (within a 2.0 kpc search radius) is updated simultaneously so that metals incorporated into dust are removed from the gas phase, preserving mass conservation. A safety floor prevents the gas particle's mass from being driven to zero or negative if the condensed dust mass would otherwise exceed it.

### Numerical floors

Two floors, hardcoded in `dust.cc` rather than exposed as parameters, prevent creation of negligible or numerically fragile particles:

- If the total dust mass produced by an event falls below `MIN_DUST_PARTICLE_MASS` (10⁻¹⁰ M☉), no dust particles are created at all.
- If dividing the total dust mass among `n_dust_particles` would leave less than 10⁻¹⁵ M☉ per particle, creation is skipped for that event.

Both are silent no-ops — low-yield feedback events simply produce no dust, which is expected behavior rather than a bug.

### Algorithm

1. **A stellar feedback event occurs.**
2. **The stellar source is identified** as either SNII or AGB.
3. **The dust yield is computed** using the corresponding condensation efficiency, and checked against the total-mass floor.
4. **The number of dust superparticles is determined** and the per-particle mass is checked against the per-particle floor.
5. **The nearest gas cell is located** (within a 2.0 kpc search radius) and condensed metals are removed from its mass and metallicity, subject to a safety floor.
6. **Each dust particle is initialized** with placeholder properties, then immediately assigned its feedback-type-specific grain radius, carbon fraction, and grain type.
7. **Particle bookkeeping is updated** (IDs, particle type, timestep bin, softening, diagnostics).
8. **The new PartType6 particles enter the simulation** and evolve through subsequent dust physics.

### Initial Particle Properties

Grain radius, carbon fraction, and grain type are hardcoded per feedback channel and are not adjustable via the parameter file. Initial velocity is parameter-driven and is applied as an isotropic random **kick added to the parent star's velocity**, not an absolute velocity.

| Quantity | SNII | AGB | Configurable? |
|-----------|------:|------:|:------:|
| Grain radius | 10 nm | 100 nm | No (hardcoded) |
| Carbon fraction | 0.10 | 0.60 | No (hardcoded) |
| Grain type | 0 | 1 | No (hardcoded) |
| Initial velocity kick | 100 km s⁻¹ | 10 km s⁻¹ | Yes (`DustVelocitySNII` / `DustVelocityAGB`) |

Newly spawned particles also receive an initial dust temperature at the CMB floor, \(T_{\rm CMB} = 2.7 / a\).

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustYieldSNII` | SNII condensation efficiency |
| `DustYieldAGB` | AGB condensation efficiency |
| `DustVelocitySNII` | Initial SNII velocity kick |
| `DustVelocityAGB` | Initial AGB velocity kick |
| `DustParticlesPerSNII` | Number of particles created per SNII event |
| `DustParticlesPerAGB` | Number of particles created per AGB event |
| `DustOffsetMinSNII` | Minimum SNII birth offset |
| `DustOffsetMaxSNII` | Maximum SNII birth offset |
| `DustOffsetMinAGB` | Minimum AGB birth offset |
| `DustOffsetMaxAGB` | Maximum AGB birth offset |

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Dust creation routines**
  - `create_dust_particles_from_feedback()`
    - Computes dust production from stellar ejecta, reduces the nearest gas cell's mass and metallicity, spawns the new particles, and overwrites their grain radius, carbon fraction, and grain type with feedback-type-specific values.
  - `spawn_dust_particle()`
    - Allocates a new PartType6 particle with placeholder grain properties, assigns its position (star position plus a random offset), initial velocity, unique ID, softening class, and CMB-floor dust temperature. Its placeholder grain properties are overwritten by the calling routine immediately after.

- **Supporting routines**
  - Neighbor search
    - `find_nearest_gas_particle()` identifies the single nearest gas cell receiving stellar ejecta.
  - Particle bookkeeping
    - Updates particle arrays, timestep bins, softening classes, and diagnostic counters.

### `src/starformation/feedback.cc`

- **Stellar feedback interface**
  - Computes stellar ejecta.
  - Invokes the dust creation routines following SNII or AGB feedback events.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to stellar dust production and live dust evolution.*

### McKinnon, Torrey & Vogelsberger (2016)

**McKinnon, R., Torrey, P., & Vogelsberger, M. (2016). *Simulating the Dust Content of Galaxies: Implementing Dust Physics in AREPO.* Monthly Notices of the Royal Astronomical Society, 457, 3775–3800.**

<p class="reference-note">
One of the first cosmological implementations of explicit live dust evolution and an important conceptual predecessor to CosmicGrain.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3775M" target="_blank" rel="noopener">📚 ADS</a>

<a href="https://arxiv.org/abs/1601.02689" target="_blank" rel="noopener">📄 arXiv</a>

<a href="https://doi.org/10.1093/mnras/stw260" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Nozawa et al. (2003)

**Nozawa, T., Kozasa, T., Umeda, H., Maeda, K., & Nomoto, K. (2003). *Dust in the Early Universe: Dust Formation in the Ejecta of Population III Supernovae.* The Astrophysical Journal, 598, 785–803.**

<p class="reference-note">
Classic theoretical study of dust formation in supernova ejecta, motivating modern SNII dust production prescriptions.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2003ApJ...598..785N" target="_blank" rel="noopener">📚 ADS</a>

<a href="https://arxiv.org/abs/astro-ph/0307108" target="_blank" rel="noopener">📄 arXiv</a>

<a href="https://doi.org/10.1086/378792" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Ferrarotti & Gail (2006)

**Ferrarotti, A. S., & Gail, H.-P. (2006). *Composition and Quantity of Dust Produced by AGB Stars and Returned to the Interstellar Medium.* Astronomy & Astrophysics, 447, 553–576.**

<p class="reference-note">
Seminal calculations of dust production by AGB stars that underpin many stellar condensation prescriptions.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2006A&A...447..553F" target="_blank" rel="noopener">📚 ADS</a>

<a href="https://doi.org/10.1051/0004-6361:20041198" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Dwek (1998)

**Dwek, E. (1998). *The Evolution of the Elemental Abundances in the Gas and Dust Phases of the Galaxy.* The Astrophysical Journal, 501, 643–665.**

<p class="reference-note">
Classic review of the complete interstellar dust lifecycle, including stellar dust production, astration, destruction, and ISM processing.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1998ApJ...501..643D" target="_blank" rel="noopener">📚 ADS</a>

<a href="https://doi.org/10.1086/305829" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
