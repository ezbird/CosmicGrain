# Astration

## Overview

<div class="cg-card summary" markdown="1">

Astration removes dust from the resolved interstellar medium when gas is converted into stars, representing the permanent incorporation of dust into stellar populations. As one of the principal sink terms in the CosmicGrain dust lifecycle, astration regulates the available dust reservoir for transport, grain growth, coagulation, shattering, sputtering, and radiation pressure. Without astration, simulations would systematically overestimate dust masses in star-forming galaxies and fail to self-consistently exchange material between gas, dust, and stars.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

Astration assumes that dust is well mixed with star-forming gas on unresolved scales. When gas is converted into stars, dust is removed from the local dust population in proportion to the **local dust-to-gas ratio**, sampled from dust particles found within a search radius around the star-forming gas cell.

The mass removed is

<div class="cg-equation" markdown="1">

\[
\Delta M_{\rm dust}
=
M_{\star,\rm formed}
\times
\left(\frac{M_{\rm dust}}{M_{\rm gas}}\right)_{\rm local}
\]

</div>

where

- \(M_{\star,\rm formed}\) is the stellar mass formed in the star formation event,
- \((M_{\rm dust}/M_{\rm gas})_{\rm local}\) is the dust-to-gas ratio measured from dust particles within the local search volume, not a global or galaxy-averaged value,
- \(\Delta M_{\rm dust}\) is the total dust mass permanently removed from the resolved ISM.

This is mathematically equivalent to removing a fraction \(f_{\rm sf} \approx M_{\star,\rm formed}/M_{\rm gas}\) of the locally sampled dust mass, but the local D/G is recomputed at each star formation event from the dust actually found nearby, rather than assumed from a single associated reservoir.

### Assumptions

- Dust is well mixed with star-forming gas on unresolved (subgrid) scales.
- The locally sampled D/G ratio is representative of the dust available for astration in that star formation event.
- Dust locked into stars no longer participates in ISM dust evolution.
- Astration is a sink process rather than a grain destruction mechanism.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Whenever a star formation event occurs, CosmicGrain identifies nearby dust particles via a spatial hash search, computes the local dust-to-gas ratio from them, and removes dust mass proportionally.

Unlike sputtering or shock destruction, astration does **not** return metals to the gas phase. Instead, consumed dust mass is transferred directly to the newly formed star particle's mass and metallicity, and permanently removed from the resolved ISM.

### Neighbor search and distributed consumption

The search radius is set from the gas cell's physical size (derived from its density and mass), capped at the SPH smoothing length. All dust particles found within this radius are treated as locally available for consumption:

- The **local D/G ratio** is computed from the total dust mass found in the search volume divided by the gas particle mass.
- The total dust mass to consume is `stellar_mass_formed × local D/G`, capped at the total dust mass found.
- Consumption is **distributed across all dust particles in the search volume, weighted by inverse distance** to the star-forming gas cell — closer particles lose proportionally more mass.
- Dust particles whose masses fall below the minimum surviving dust threshold are deleted outright; the remainder simply lose mass.

If no dust particles are found within the search radius, the function exits with **no astration applied** for that event — this is a silent no-op rather than an error, and is expected in dust-sparse regions.

### Algorithm

1. **A star formation event occurs**, converting a gas particle's mass into a new star particle.
2. **The search radius is set** from the gas cell's physical radius, capped at its SPH smoothing length.
3. **Nearby dust particles are located** via a spatial hash neighbor search.
4. **The local dust-to-gas ratio is computed** from the dust mass found in the search volume.
5. **Dust mass to consume is calculated** as stellar mass formed × local D/G, capped at the dust mass found.
6. **Consumption is distributed** across nearby dust particles, weighted by inverse distance.
7. **Dust particles below the destruction threshold are deleted**; survivors have their mass reduced.
8. **Consumed mass and metallicity are transferred to the new star particle.**
9. **Dust diagnostics and bookkeeping are updated.**
10. **The simulation continues.**

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableAstration` | Enables or disables astration. |
| `DUST_MASS_TO_DESTROY` | Minimum surviving dust particle mass before deletion. Shared threshold used across all dust destruction channels, not astration-specific. |

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Astration routine**
  - `consume_dust_by_astration()`
    - Performs the spatial hash neighbor search, computes the local D/G ratio, distributes proportional mass removal across nearby dust particles, deletes particles below the survival threshold, and transfers consumed mass and metallicity to the new star particle.
  - Astration bookkeeping
    - `NDustDestroyedByAstration`
    - `TotalDustMassAstrated`

### `src/starformation/feedback.cc`

- **Star formation interface**
  - Invokes `consume_dust_by_astration()` when gas is converted into stars.
  - Determines the stellar mass formed during each star formation event.

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
