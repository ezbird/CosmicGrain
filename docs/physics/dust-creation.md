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

Separate source prescriptions are used for SNII, AGB stars, and luminous red
novae (LRNe). In the audited source, SNII/hypernova feedback is delayed over four tranches rather
than injected in a single instantaneous event. LRN dust is created once with
the first SNII tranche; SNII dust sampling is divided across the tranche
sequence so the requested total particle sampling is not multiplied by four. The
absolute LRN normalization remains under active calibration.

### Assumptions

- Dust forms when the relevant delayed stellar-feedback channel releases its
  ejecta.
- Condensation efficiencies are fixed for each stellar source.
- Dust is represented by superparticles rather than individual grains.
- Each superparticle represents an ensemble of physical grains.
- Metals condensed into dust are removed from the gas phase.
- Gas-phase ejecta are deposited by the stellar-feedback machinery; the dust-spawning routine receives the already partitioned carbonaceous and silicate dust masses.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

When stellar feedback occurs, CosmicGrain computes element-resolved ejecta
from the stellar particle and converts an allowed fraction into dust. The
condensed mass is limited by the available carbon and silicate-forming
material. The requested number of dust superparticles is distributed across
the full source event.

Rather than storing dust as an abundance field, the code creates entirely new **PartType6** particles. Elemental partitioning between gas-phase ejecta and condensed dust is performed in `feedback.cc` **before** the PartType6 spawning routine is called. `create_dust_particles_from_feedback()` then only creates the dust representatives; it does not independently remove material from a nearest gas particle. This separation is important when tracing mass and element conservation.

### Numerical floors

Two floors, hardcoded in `dust.cc` rather than exposed as parameters, prevent creation of negligible or numerically fragile particles:

- If the total dust mass produced by an event falls below `MIN_DUST_PARTICLE_MASS` (10⁻¹⁰ M☉), no dust particles are created at all.
- If the requested sampling would produce representatives below a local `1e-15` code-mass floor, the routine **reduces the number of superparticles** rather than discarding the dust mass. At least one representative is retained for an event that already passed the total-mass floor.

Thus the total-mass floor can suppress a negligible creation event, while the per-representative floor changes numerical sampling rather than the physical dust mass.

### Algorithm

1. **A stellar feedback event occurs.**
2. **The stellar source channel is identified** (SNII, AGB, or LRN where applicable).
3. **The dust yield is computed** using the corresponding condensation efficiency, and checked against the total-mass floor.
4. **The number of dust superparticles is determined** and the per-particle mass is checked against the per-particle floor.
5. **Gas-phase and condensed ejecta are partitioned by the feedback module**, which deposits the gas-phase share through the feedback machinery.
6. **Each dust particle is initialized** with the source-specific representative radius and source tag; its carbon fraction is computed from the carbonaceous and silicate dust masses passed by the feedback calculation.
7. **Particle bookkeeping is updated** (IDs, particle type, timestep bin, softening, diagnostics).
8. **The new PartType6 particles enter the simulation** and evolve through subsequent dust physics.

### Initial Particle Properties

Representative birth radius and source tag are hardcoded per feedback channel. The **target** birth composition is source-specific, but the actual `CarbonMassFraction` written to each new particle is computed from the element-limited carbonaceous and silicate dust masses delivered by the feedback calculation. Initial velocity is parameter-driven and is applied as an isotropic random **kick added to the parent star's velocity**, not an absolute velocity.

| Quantity | SNII | AGB | LRN | Configurable? |
|-----------|------:|------:|------:|:------:|
| Grain radius | 10 nm | 100 nm | 10 nm | No (hardcoded) |
| Carbon fraction | 0.10 target, element-limited | 0.60 target, element-limited | 0.10 | Target hardcoded |
| `DustSource` | 0 | 1 | 2 | No |
| Initial velocity kick | `DustVelocitySNII` | `DustVelocityAGB` | `DustVelocityLRN` | Yes |

The current AGB target is 60% carbon and 40% silicate, subject to the
available carbon and total-metal budget. LRN dust is 10% carbon and 90%
silicate. Subsequent processing can evolve carbon fraction away from its birth
value.

Newly spawned particles also receive an initial dust temperature at the CMB floor, \(T_{\rm CMB} = 2.7 / a\).

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustYieldSNII` | SNII condensation efficiency |
| `DustYieldAGB` | AGB condensation efficiency |
| `DustVelocitySNII` | Initial SNII velocity kick |
| `DustVelocityAGB` | Initial AGB velocity kick |
| `DustVelocityLRN` | Initial LRN velocity kick |
| `DustParticlesPerSNII` | Number of particles created per SNII event |
| `DustParticlesPerAGB` | Number of particles created per AGB event |
| `DustParticlesPerLRN` | Number of particles created per LRN event |
| `DustLRNRatePerCCSN` | LRN event-rate normalization relative to CCSNe |
| `DustLRNDustMassMsun` | Adopted dust mass per LRN event |
| `DustOffsetMinSNII` | Minimum SNII birth offset |
| `DustOffsetMaxSNII` | Maximum SNII birth offset |
| `DustOffsetMinAGB` | Minimum AGB birth offset |
| `DustOffsetMaxAGB` | Maximum AGB birth offset |
| `DustOffsetMinLRN` | Minimum LRN birth offset |
| `DustOffsetMaxLRN` | Maximum LRN birth offset |

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Dust creation routines**
  - `create_dust_particles_from_feedback()`
    - Receives already partitioned carbonaceous and silicate dust masses, chooses the numerical sampling, and spawns PartType6 representatives with source-specific radii/tags and composition derived from those masses.
  - `spawn_dust_particle()`
    - Allocates a new PartType6 particle with placeholder grain properties, assigns its position (star position plus a random offset), initial velocity, unique ID, softening class, and CMB-floor dust temperature. Its placeholder grain properties are overwritten by the calling routine immediately after.

- **Supporting routines**
  - `spawn_dust_particle()` handles allocation, birth offset, velocity, ID, softening, timestep-bin insertion, and initial temperature.
  - Particle bookkeeping updates the particle arrays and creation diagnostics.

### `src/cooling_sfr/feedback.cc`

- **Stellar feedback interface**
  - Computes stellar ejecta.
  - Invokes the dust creation routines following SNII or AGB feedback events.

</div>

## Current Calibration Status

`DustParticlesPerSNII`, `DustParticlesPerAGB`, and `DustParticlesPerLRN` are numerical sampling controls, not physical event rates or dust yields. Their values belong to the run parameter file rather than to the code definition. The LRN source channel is implemented and produces explicitly tagged particles; its quantitative normalization is controlled by `DustLRNRatePerCCSN` and `DustLRNDustMassMsun` and should be reported with the run.

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
