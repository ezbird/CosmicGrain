# Shattering

## Overview

<div class="cg-card summary" markdown="1">

Shattering changes a dust superparticle's representative radius while
conserving its mass and composition. It is a stochastic, continuous-size
adaptation of the Trayford et al. (2026) shattering timescale, with an
additional grain-size- and environment-dependent turbulent-velocity gate.
In the default dense-phase model (`DustEnableClumping 2`) shattering acts on
the diffuse fraction \(1-f_{\rm dense}\) of every gas element and
coagulation on the dense fraction; in the legacy modes a shared
effective-density threshold switches between them.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

The model estimates a turbulent grain velocity using the scaling attributed
in the source to Hirashita & Aoyama (2019), based on Yan et al. (2004) and
Ormel & Cuzzi (2007):

<div class="cg-equation" markdown="1">

\[
v_{\rm gr} = 1.1\,{\rm km\,s^{-1}}\,
\mathcal{M}^{3/2}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)^{1/2}
\left(\frac{T}{10^4\,{\rm K}}\right)^{1/4}
\left(\frac{n_{\rm H}}{1\,{\rm cm}^{-3}}\right)^{-1/4}
\left(\frac{\rho_{\rm gr}}{3.5\,{\rm g\,cm}^{-3}}\right)^{1/2}.
\]

</div>

Here \(a\) is the representative radius, \(T\) is the associated gas
temperature, and \(n_{\rm H}\) is the unclumped hydrogen density, floored at
\(10^{-4}\,{\rm cm}^{-3}\). The turbulent Mach number is fixed at
\(\mathcal{M}=1\). Material density is interpolated by carbon mass fraction,
from \(3.5\,{\rm g\,cm}^{-3}\) for silicate to \(2.24\,{\rm g\,cm}^{-3}\)
for graphite.

Shattering is eligible when this velocity reaches the composition-dependent
threshold

<div class="cg-equation" markdown="1">

\[
v_{\rm shatter} = (1 + {\rm CF})\,{\rm km\,s^{-1}},
\]

</div>

where \({\rm CF}\) is the particle's carbon mass fraction, giving thresholds
from 1 km/s for pure silicate to 2 km/s for pure carbon. These endpoints
follow the threshold treatment cited to Jones et al. (1994).

For eligible grains, the Trayford et al. (2026) Eq. 9 timescale is adapted to
the continuous representative radius:

<div class="cg-equation" markdown="1">

\[
\tau_{\rm sh} = 54.1\,{\rm Myr}
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
f(\hat n_{\rm H})\,\frac{\mathcal{C}_{\rm sh}}{f_{\rm diff}},
\qquad
f(\hat n_{\rm H}) =
\begin{cases}
\hat n_{\rm H}^{-1}, & \hat n_{\rm H}<1,\\
\hat n_{\rm H}^{-1/3}, & \hat n_{\rm H}\ge 1.
\end{cases}
\]

</div>

Here \(\hat n_{\rm H}=n_{\rm H}/(1\,{\rm cm}^{-3})\). D/G is the
MPI-combined local dust-to-gas mass
estimate shared with coagulation, while the timescale's density factor uses
the unclumped \(n_{\rm H}\). The D/G estimate is clamped to
\([10^{-6},0.1]\). \(\mathcal{C}_{\rm sh}\) is
`DustShatteringCalibration`, a timescale multiplier (larger = slower; v1.2.0
default 1.485, giving \(\tau_{\rm sh}\approx80\) Myr at \(a=0.1\,\mu\)m,
D/G = 0.01, \(n_{\rm H}=1\,{\rm cm^{-3}}\)). \(f_{\rm diff}=1-f_{\rm dense}\)
in modes 2/3 and 1 in modes 0/1. The timescale is clamped to
[1 Myr, 10 Gyr]. Whether an event occurs in a timestep follows a Poisson
process,

<div class="cg-equation" markdown="1">

\[
P_{\rm shatter} = 1 - e^{-\Delta t/\tau_{\rm sh}}.
\]

</div>

When an event fires, a representative fragment radius is sampled from a
power-law fragment mass distribution. The assumed fragment number
distribution is \(dN/da \propto a^{-q}\); multiplying by the individual
grain mass (\(\propto a^3\)) gives \(dM/da \propto a^{3-q}\). The sampled
radius lies between `DustMinGrainSize` and
`DustShatteringMaxFragmentFraction` times the parent radius. Only the
representative radius changes: the superparticle retains its mass and carbon
fraction.

### Assumptions

- The turbulent Mach number is fixed at 1; the model uses an analytic
  grain-velocity scaling rather than a resolved turbulent velocity field.
- The velocity threshold blends linearly between silicate and carbon
  endpoints using the particle's current carbon fraction.
- Shattering is applied as a Poisson event, not continuous erosion.
- If the maximum allowed fragment radius is at or below the minimum grain
  size, no shattering event is applied because no resolved fragment interval
  is available.
- The fragment power law and maximum fragment fraction are model controls;
  they are not a full fragmentation-kernel calculation.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Shattering checks grain validity and whether a fragment-size interval exists,
then applies the phase/density gate, the velocity threshold, the timescale,
and the stochastic draw.

### Phase and density gate

- **Modes 2/3 (default):** no density threshold. The rate is scaled by
  \(f_{\rm diff}=1-f_{\rm dense}\) and shattering is skipped only where
  \(f_{\rm diff}<10^{-4}\). See [Subgrid Clumping](subgrid-clumping.md).
- **Modes 0/1 (legacy):** shattering is eligible for
  `n_eff <= DustCollisionDensityThresh`, coagulation for
  `n_eff >= DustCollisionDensityThresh`; both pass at exact equality.

### Numerical limits

- The local D/G estimate is shared with coagulation and must be finite and
  positive; see [Coagulation](coagulation.md) for the association sphere and
  cache details.
- The unclumped density used in the velocity and timescale is floored at
  \(10^{-4}\,{\rm cm}^{-3}\).
- The shattering timescale is clamped to [1 Myr, 10 Gyr].
- Grains at or below `DustMinGrainSize` skip shattering.
- If
  `DustShatteringMaxFragmentFraction * GrainRadius <= DustMinGrainSize`,
  the call is skipped rather than destroying the superparticle.

### Algorithm

1. **Validate the grain and fragment interval** — reject invalid radii or
   masses, radii at or below the minimum, or an empty allowed fragment range.
2. **Gate on phase or density** — \(f_{\rm diff}\ge10^{-4}\) (modes 2/3) or
   `n_eff <= DustCollisionDensityThresh` (modes 0/1).
3. **Validate local D/G** — use the shared MPI-combined association-sphere
   estimate.
4. **Gate on turbulent grain velocity** — the size- and environment-dependent
   velocity must meet the composition-dependent threshold.
5. **Compute the timescale** from local D/G, representative radius, and
   unclumped hydrogen density.
6. **Draw a Poisson event**; if it occurs, sample a fragment radius from
   \(dM/da \propto a^{3-q}\).
7. **Update the representative radius only** — dust mass and carbon fraction
   are conserved.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableShattering` | Enables or disables shattering. |
| `DustEnableClumping` | Selects the dense-phase (2/3, default 2) or legacy (0/1) gate. |
| `DustCollisionDensityThresh` | Legacy modes 0/1 only: shattering is eligible at/below it and coagulation at/above it. |
| `DustShatteringCalibration` | Timescale multiplier, larger = slower; default 1.485. |
| `DustShatteringFragmentSlope` | Slope \(q\) of the assumed fragment number distribution \(dN/da \propto a^{-q}\); default 3.3, finite and positive. |
| `DustShatteringMaxFragmentFraction` | Upper fragment-radius limit as a fraction of the parent radius; default 0.80, strictly between 0 and 1. |
| `DustMinGrainSize` | Minimum allowed fragment radius; if the available fragment interval is empty, the shattering event is skipped. |

The timescale normalization, density-factor break and exponents, Mach number,
grain-velocity normalization, material densities, velocity-threshold
endpoints, and timescale clamps are hardcoded and are not runtime parameters.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Shattering routine**
  - `dust_grain_shattering()`
    - Applies the validity, density, and velocity gates; computes the
      timescale, draws the stochastic event, and samples a new representative
      radius on success.
  - `dust_dense_fraction()` / `dust_clumping_factor()`
    - Supply \(f_{\rm dense}\) (modes 2/3) or the legacy clumping factor.
  - `sample_shattering_fragment_radius()`
    - Samples the configurable fragment mass spectrum.
  - Bookkeeping: `NShatteringEvents`, `TotalSizeReductionShattering`.

- **Caller**
  - `update_dust_dynamics()` calls `dust_grain_shattering()` for grains within
    the association radius \(\max(2,\min(H_{\rm sml},10))\) code length units
    of their gas, after growth and coagulation (see
    [Coagulation](coagulation.md)).

</div>

## Primary References

<div class="cg-card references" markdown="1">

*References relevant to the adopted timescale, turbulent grain velocities,
and shattering threshold.*

### Trayford et al. (2026)

**Trayford, J. W., Schaye, J., Correa, C., Ploeckinger, S., Richings, A. J., Chaikin, E., Schaller, M., Benítez-Llambay, A., Frenk, C., & Huško, F. (2026). *Modelling the Evolution and Influence of Dust in Cosmological Simulations That Include the Cold Phase of the Interstellar Medium.* Monthly Notices of the Royal Astronomical Society, 545, staf2040.**

<p class="reference-note">
Source of the shattering-timescale scaling (Eq. 9), adapted here to CosmicGrain's continuous representative radius and combined with an explicit turbulent-velocity gate.
</p>

<div class="reference-links">

<a href="https://arxiv.org/abs/2505.13056" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/staf2040" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Jones, Tielens, Hollenbach & McKee (1994)

**Jones, A. P., Tielens, A. G. G. M., Hollenbach, D. J., & McKee, C. F. (1994). *Grain Destruction in Shocks in the Interstellar Medium.* The Astrophysical Journal, 433, 797.**

<p class="reference-note">
Source cited for the composition-dependent velocity thresholds (1–2 km/s) used to gate shattering events.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1994ApJ...433..797J" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/174689" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Jones, Tielens & Hollenbach (1996)

**Jones, A. P., Tielens, A. G. G. M., & Hollenbach, D. J. (1996). *Grain Shattering in Shocks: The Interstellar Grain Size Distribution.* The Astrophysical Journal, 469, 740–764.**

<p class="reference-note">
Relevant shock-shattering and fragment-size-distribution reference; CosmicGrain samples a configurable power-law fragment mass spectrum rather than applying a fixed fragment radius.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/1996ApJ...469..740J" target="_blank" rel="noopener">📚 ADS</a>

</div>

---

### Hirashita & Yan (2009)

**Hirashita, H., & Yan, H. (2009). *Shattering and Coagulation of Dust Grains in Interstellar Turbulence.* Monthly Notices of the Royal Astronomical Society, 394, 1061–1074.**

<p class="reference-note">
Background on grain shattering and coagulation in interstellar turbulence; the current shattering timescale attribution in CosmicGrain is Trayford et al. (2026), Eq. 9.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2009MNRAS.394.1061H" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1111/j.1365-2966.2009.14405.x" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
