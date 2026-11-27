# Coagulation

## Overview

<div class="cg-card summary" markdown="1">

Coagulation increases the representative grain radius of a dust superparticle
at fixed mass and composition, as a continuous closure for grains sticking
together in dense, cold gas. Unlike grain growth it adds no mass: fewer,
larger grains carry the same dust mass. Coagulation and shattering are the two
size-transfer processes; in the default dense-phase model coagulation acts on
the dense fraction of each gas element and shattering on the diffuse rest.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain uses the Trayford et al. (2026) Eq. 10 coagulation timescale,
with the representative radius in place of their small-grain bin:

<div class="cg-equation" markdown="1">

\[
\tau_{\rm co} = 27.1\,{\rm Myr}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{v_{\rm co}}{0.1\,{\rm km\,s^{-1}}}\right)^{-1}
\left(\frac{n_{\rm H}}{10\,{\rm cm^{-3}}}\right)^{-1}
\frac{\mathcal{C}_{\rm co}}{f_{\rm dense}}
\]

</div>

with \(v_{\rm co}=0.2\,{\rm km\,s^{-1}}\) and \(\mathcal{C}_{\rm co}\) =
`DustCoagulationCalibration`, a timescale multiplier (larger = slower). The
density and phase fraction depend on `DustEnableClumping`
([Subgrid Clumping](subgrid-clumping.md)):

- **Modes 2/3 (default 2):** \(n_{\rm H}=n_{\rm dense}\) (100 cm\(^{-3}\) in
  mode 2) and \(f_{\rm dense}\) from Blitz & Rosolowsky (2006). Coagulation is
  skipped only where \(f_{\rm dense}<10^{-4}\).
- **Modes 0/1:** \(n_{\rm H}=n_{\rm eff}=C\,n_{\rm H}\), \(f_{\rm dense}=1\),
  and coagulation runs only where \(n_{\rm eff}\ge\)
  `DustCollisionDensityThresh` and, for non-star-forming gas, \(T\le3000\) K.

With the v1.2.0 default \(\mathcal{C}_{\rm co}=22.3\),
\(\tau_{\rm co}\approx30\) Myr for \(a=0.1\,\mu\)m, D/G = 0.01,
\(n_{\rm dense}=100\,{\rm cm^{-3}}\) and \(f_{\rm dense}=1\).

D/G is the dust mass over the gas mass inside the grain's association sphere
(see below), summed over all MPI ranks and clamped to \([10^{-6},0.1]\). It is
a particle-sampled mass ratio, not a kernel-weighted one.

With the closure \(da/dt=a/(3\tau_{\rm co})\) and \(\tau_{\rm co}\propto a\),
the radius update over \(\Delta t\) is

<div class="cg-equation" markdown="1">

\[
a_{\rm new} = \min\!\left[a_{\max},\,
a\left(1+\frac{\Delta t}{3\tau_{\rm co}(a)}\right)\right],
\]

</div>

subcycled so no substep grows the radius by more than 20%. The superparticle's
mass and carbon fraction are unchanged, and no mass is exchanged with gas.

### Assumptions

- Coagulation is deterministic and continuous: many collisions per step.
- No partner superparticle is consumed; only the representative radius changes.
- The legacy 3000 K ceiling applies only to non-star-forming gas in modes 0/1;
  effective-EOS gas stands in for unresolved cold clouds.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

### Association sphere and local D/G

Each grain is associated with its nearest eligible gas particle (zoom-region
gas within 10 code length units), searched across all ranks. Growth,
coagulation, and shattering run only if the grain lies within
\(r_{\rm assoc}=\max(2,\min(H_{\rm sml},10))\) code length units (comoving
\(h^{-1}\) kpc) of that gas particle. A grain whose nearest gas is on another
rank is sent to that rank for its update and returned.

D/G is summed over the sphere of radius \(r_{\rm assoc}\) around the grain.
The sums use a 2-code-unit grid rebuilt at every dust call from the particle
lists of the shared feedback hash, at current positions. Queries go to the
ranks whose particles can overlap the sphere: by bounding box below 64 ranks,
and through a distributed directory of occupied routing cells from 64 ranks
up. Both routings give bit-identical results (`CG_DTG_ROUTING=bbox|occupancy`
forces one). The feedback hash is rebuilt after each domain decomposition and
whenever the scale factor has advanced by 0.002.

Each grain caches its D/G (`LocalDTG`, `DTGAge`) and refreshes it every 10
dust calls, staggered over particles; the cache moves with the particle and is
cleared when the grain has no valid association.

### Numerical limits

- Grains at or above `DustMaxGrainSize` are skipped; updates are capped there.
- A missing or non-positive D/G skips the grain.
- An update that does not increase the radius is discarded.

### Algorithm

1. Get \(n_{\rm H}\) and \(f_{\rm dense}\) (modes 2/3) or \(n_{\rm eff}\) and
   the density/temperature gates (modes 0/1).
2. Reject invalid grains, grains at the maximum size, and missing D/G.
3. Compute \(\tau_{\rm co}\).
4. Subcycle the radius update (at most 20% per substep, capped at
   `DustMaxGrainSize`).
5. Record the event for diagnostics.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableCoagulation` | Enables or disables coagulation. |
| `DustCoagulationCalibration` | Timescale multiplier, larger = slower; v1.2.0 default 22.3. |
| `DustEnableClumping` | Selects the dense-phase (2/3) or legacy (0/1) density model. |
| `DustCollisionDensityThresh` | Minimum \(n_{\rm eff}\) for coagulation in modes 0/1 only (default 15 cm\(^{-3}\)). |
| `DustMaxGrainSize` | Shared maximum grain radius (nm). |

The timescale normalization, \(v_{\rm co}\), D/G clamp, 3000 K legacy
ceiling, 20% substep limit, association radii, and 10-call D/G refresh are
hardcoded.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- `dust_grain_coagulation()` applies the gates and updates the radius.
- `estimate_local_dust_to_gas_ratio()` computes D/G across ranks (shared with shattering).
- `dust_dense_fraction()` / `dust_clumping_factor()` supply the density model.
- `update_dust_dynamics()` associates grains with gas, refreshes D/G, and calls growth, coagulation, and shattering.
- Bookkeeping: `NCoagulationEvents`; `record_coagulation_event()` fills the periodic density histogram.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*References relevant to grain-grain coagulation via dense-gas collisions.*

### Trayford et al. (2026)

**Trayford, J. W., et al. (2026). *Modelling the Evolution and Influence of Dust in Cosmological Simulations That Include the Cold Phase of the Interstellar Medium.* Monthly Notices of the Royal Astronomical Society, 545, staf2040.**

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
