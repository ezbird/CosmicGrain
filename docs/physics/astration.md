# Astration

## Overview

<div class="cg-card summary" markdown="1">

Astration removes dust from the resolved interstellar medium when gas is converted into stars, representing the permanent incorporation of dust into stellar populations. As one of the principal sink terms in the CosmicGrain dust lifecycle, astration regulates the available dust reservoir for transport, grain growth, coagulation, shattering, sputtering, and radiation pressure. Without astration, simulations would systematically overestimate dust masses in star-forming galaxies and fail to self-consistently exchange material between gas, dust, and stars.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

Astration assumes that dust is well mixed with star-forming gas on unresolved scales. When gas is converted into stars, dust is removed in proportion to the **local dust-to-gas ratio** measured in a sphere around the star-forming gas particle.

The mass removed is

<div class="cg-equation" markdown="1">

\[
\Delta M_{\rm dust}
=
\min\!\left[
M_{\star,\rm formed}
\left(\frac{M_{\rm dust}}{M_{\rm gas}+M_{\star,\rm formed}}\right)_{\rm sphere}
C_{\rm astr},\;
M_{\rm dust,sphere}\right]
\]

</div>

where

- \(M_{\star,\rm formed}\) is the stellar mass formed,
- \(M_{\rm dust}\) and \(M_{\rm gas}\) are the dust and gas masses inside the same sphere, summed over all MPI ranks; the gas just converted into the star counts as gas,
- \(C_{\rm astr}\) is `DustAstrationCalibration` (1 is physical; the v1.2.0 default).

Before v1.2.0 the ratio divided all dust in the sphere by the single star-forming gas particle's mass, which overestimated D/G by roughly the neighbour number; the old default calibration of 0.15 compensated for that.

### Assumptions

- Dust is well mixed with star-forming gas on unresolved scales.
- Dust locked into stars no longer participates in ISM dust evolution.
- Astration is a sink, not a grain-destruction mechanism: no metals return to the gas.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Star formation only queues an astration event (`consume_dust_by_astration()`).
After the star-formation loop, `apply_queued_dust_astration()` gathers the
events from all tasks, finds dust and gas in each sphere on every task, sums
the masses over tasks, and applies the events in star-ID order, so the result
does not depend on the domain decomposition.

- **Sphere radius:** the larger of the gas particle's SPH smoothing length and
  the radius of a sphere holding its mass at its density.
- **Distribution:** the consumed mass is split over all grains in the sphere,
  weighted by inverse distance to the gas particle.
- **Deletion:** a grain left below `DUST_MASS_TO_DESTROY` (\(10^{-30}\) code
  units, hardcoded) is removed entirely.
- **Transfer:** the consumed dust mass is added to the new star's mass and
  metals.
- If no grain is in the sphere, nothing is removed.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableAstration` | Enables or disables astration. |
| `DustAstrationCalibration` | Multiplier on the astrated mass; default 1 (physical). |

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- `consume_dust_by_astration()` queues an event per star-forming gas particle.
- `apply_queued_dust_astration()` measures the local D/G across tasks, removes the dust, and credits the new stars.
- Bookkeeping: `NDustDestroyedByAstration`, `TotalDustMassAstrated`.

### `src/cooling_sfr/starformation.cc`

- Calls `consume_dust_by_astration()` for each conversion and `apply_queued_dust_astration()` once per star-formation step.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to astration and the stellar dust lifecycle.*

### Springel et al. (2021)

**Springel, V., Pakmor, R., Zier, O., & Reinecke, M. (2021). *GADGET-4: A Novel Computational Framework for Cosmological Simulations.* Monthly Notices of the Royal Astronomical Society, 506, 2871–2949.**

<p class="reference-note">
Describes the simulation framework that CosmicGrain extends with explicit dust physics.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2871S" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/2010.03567" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/stab1855" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### McKinnon, Torrey & Vogelsberger (2016)

**McKinnon, R., Torrey, P., & Vogelsberger, M. (2016). *Simulating the Dust Content of Galaxies: Implementing Dust Physics in AREPO.* Monthly Notices of the Royal Astronomical Society, 457, 3775–3800.**

<p class="reference-note">
One of the first cosmological implementations of live dust evolution and an important predecessor to CosmicGrain.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3775M" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/1601.02689" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/stw260" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Dwek (1998)

**Dwek, E. (1998). *The Evolution of the Elemental Abundances in the Gas and Dust Phases of the Galaxy.* The Astrophysical Journal, 501, 643–665.**

<p class="reference-note">
Classic review of the complete interstellar dust lifecycle, including stellar production, astration, and dust destruction.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1998ApJ...501..643D" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/305829" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
