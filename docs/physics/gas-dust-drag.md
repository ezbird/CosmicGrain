# Drag

## Overview

<div class="cg-card summary" markdown="1">

Drag couples dust superparticles to the surrounding gas flow, transferring momentum via direct collisions between grains and gas molecules. As grains are born with a velocity kick from stellar feedback, drag is the primary mechanism that subsequently relaxes dust kinematics toward the local gas velocity, setting the characteristic aerodynamic decoupling that governs where dust can drift relative to the ISM, escape into the CGM, or lag behind gas inflows. Drag is applied every time dust physics runs and underlies the velocity update used by every other dust process in CosmicGrain.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain implements Epstein drag (McKinnon et al. 2018, eqs. 8–9), valid in the regime where the grain radius is much smaller than the gas mean free path — appropriate for ISM-sized grains. The stopping timescale in the subsonic limit is

<div class="cg-equation" markdown="1">

\[
t_{\rm stop} = \sqrt{\frac{\pi \gamma}{8}}\, \frac{a\, \rho_{\rm grain}}{\rho_{\rm gas}\, c_s}
\]

</div>

where

- \(a\) is the grain radius,
- \(\rho_{\rm grain}\) is the grain material density,
- \(\rho_{\rm gas}\) is the local gas density,
- \(c_s\) is the local gas sound speed,
- \(\gamma\) is the adiabatic index.

For supersonic relative motion (grain–gas Mach number \(\mathcal{M} > 0.1\)), a correction factor is applied,

<div class="cg-equation" markdown="1">

\[
t_{\rm stop} \rightarrow t_{\rm stop} \times \left(1 + \frac{9\pi}{128}\mathcal{M}^2\right)^{-1/2}
\]

</div>

Given \(t_{\rm stop}\), the dust velocity relaxes toward the gas velocity via the exact exponential solution,

<div class="cg-equation" markdown="1">

\[
\mathbf{v}_{\rm dust}^{\rm new} = \mathbf{v}_{\rm dust} + \left(1 - e^{-\Delta t / t_{\rm stop}}\right)\left(\mathbf{v}_{\rm gas} - \mathbf{v}_{\rm dust}\right)
\]

</div>

### Assumptions

- Drag operates in the Epstein regime; grains are assumed small relative to the gas mean free path.
- Coupling is one-way: dust velocity relaxes toward the gas velocity, but gas velocity is unaffected by the drag force. This is justified by the large mass ratio between a gas resolution element and an individual dust superparticle (McKinnon et al. 2018).
- Grain material density is fixed at the silicate value (2.4 g cm⁻³) regardless of a particle's actual carbon fraction.
- Mean molecular weight (μ = 0.6) and adiabatic index (γ = 5/3) are fixed constants in the drag timescale calculation, rather than being drawn from the gas particle's actual ionization state.
- The exact exponential form is used rather than a linear approximation, so the result remains valid for any ratio of timestep to stopping time.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Drag is applied to every live dust particle each time dust physics runs, using the properties of its nearest gas neighbor. The relative velocity, gas density, and gas temperature are used to compute a stopping timescale, and the dust velocity is updated using the exact exponential solution above — valid regardless of the timestep-to-stopping-time ratio, which matters because dust physics runs on a cadence of every 10 gravity steps with a correspondingly scaled \(\Delta t\).

### Guards and numerical floors

- If the gas neighbor returned by the spatial hash has since changed particle type (a stale hash entry), the drag update is skipped for that particle and a bad-index counter is incremented.
- In near-vacuum conditions (sound speed below 10³ cm s⁻¹, or gas density below 10⁻³⁰ g cm⁻³), the stopping timescale is set to a 50 Myr ceiling — effectively decoupling the grain from the gas rather than attempting an ill-conditioned calculation.
- The stopping timescale is clamped to the range [0.001, 50] Myr. This is a numerical safeguard: an unclamped stopping time in the diffuse CGM could otherwise place a grain on an unrealistically short gravity timebin via the drag velocity kick. The radiation pressure module computes its own unclamped stopping time separately for the terminal-velocity correction, since that physical limit should not be artificially suppressed.

### Relationship to sputtering

The same routine that applies drag also checks whether the local gas temperature exceeds the sputtering threshold and, if so, hands off to the thermal sputtering routine. Drag and sputtering share a neighbor lookup for efficiency, but sputtering's physics — grain erosion and mass return to gas — is documented separately.

### Algorithm

1. **The nearest gas particle is located** for each live dust particle.
2. **The gas neighbor's type is validated**; a stale hash result is skipped.
3. **Local gas properties are read**: velocity, density, and temperature (via entropy).
4. **The relative velocity** between dust and gas is computed.
5. **The Epstein stopping timescale is calculated**, including the supersonic correction if applicable, and clamped to [0.001, 50] Myr.
6. **The dust velocity is updated** toward the gas velocity using the exact exponential solution.
7. **If the gas is hot enough**, thermal sputtering is triggered on the same particle pair.
8. **A small diagnostic sample** (roughly 1% of particles, capped at 500 samples) is logged for verification.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableDrag` | Enables or disables drag coupling. |

Grain material density (2.4 g cm⁻³), mean molecular weight (0.6), and adiabatic index (5/3) are hardcoded in the drag timescale calculation and are not currently exposed as parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Drag routines**
  - `dust_gas_interaction()`
    - Applies the Epstein drag velocity update to a dust–gas pair and, when the gas is hot enough, delegates to thermal sputtering.
  - `calculate_drag_timescale()`
    - Computes the Epstein stopping timescale, including the supersonic correction and the [0.001, 50] Myr clamp.

- **Supporting routines**
  - `find_nearest_gas_particle()`
    - Identifies the gas neighbor used for the drag calculation via the spatial hash.
  - `get_temperature_from_entropy()`
    - Converts stored gas entropy to temperature, accounting for the local electron fraction.

### `src/dust/update_dust_dynamics()`

- **Per-step entry point**
  - Calls `dust_gas_interaction()` for every live dust particle on each dust-physics cadence step.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to grain–gas drag coupling.*

### McKinnon, Vogelsberger, Torrey, Marinacci & Kannan (2018)

**McKinnon, R., Vogelsberger, M., Torrey, P., Marinacci, F., & Kannan, R. (2018). *Simulating Galactic Dust Grain Evolution on a Moving Mesh.* Monthly Notices of the Royal Astronomical Society, 478, 2851–2886.**

<p class="reference-note">
Presents the Epstein drag and grain dynamical framework that CosmicGrain's drag coupling directly follows.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.2851M" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/1805.04521" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/sty1248" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Epstein (1924)

**Epstein, P. S. (1924). *On the Resistance Experienced by Spheres in their Motion through Gases.* Physical Review, 23, 710–733.**

<p class="reference-note">
Foundational derivation of drag on small spheres moving through a rarefied gas, the physical basis for the Epstein drag regime.
</p>

<div class="reference-links">

<a href="https://doi.org/10.1103/PhysRev.23.710" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### McKinnon, Torrey & Vogelsberger (2016)

**McKinnon, R., Torrey, P., & Vogelsberger, M. (2016). *Simulating the Dust Content of Galaxies: Implementing Dust Physics in AREPO.* Monthly Notices of the Royal Astronomical Society, 457, 3775–3800.**

<p class="reference-note">
Predecessor framework establishing the dust-particle approach that CosmicGrain's dynamical treatment builds on.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3775M" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/1601.02689" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/stw260" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
