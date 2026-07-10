# Subgrid Clumping

## Overview

<div class="cg-card summary" markdown="1">

Subgrid clumping is not a dust creation or destruction mechanism in its own right — it is a shared density-enhancement model that feeds into grain growth, coagulation, and shattering, accounting for unresolved sub-grid ISM structure that the simulation's finite resolution cannot directly capture. Dense molecular gas is assumed to be far more centrally concentrated than its resolved (smoothed) density implies, and this module supplies an effective, boosted density — used in place of the raw resolved density — for the rate calculations in those three downstream processes.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain assigns a subgrid clumping factor \(C\) based on a gas particle's star-forming status and resolved hydrogen number density \(n_H\), expressed as a fraction of the star formation density threshold `CritPhysDensity` (\(n_{\rm sf}\)):

<div class="cg-equation" markdown="1">

\[
C =
\begin{cases}
30 & \text{gas is actively star-forming} \\
10 & n_H > 0.5\,n_{\rm sf} \\
3 & n_H > 0.2\,n_{\rm sf} \\
1.5 & n_H > 0.05\,n_{\rm sf} \\
1 & \text{otherwise}
\end{cases}
\]

</div>

The effective density used by growth, coagulation, and shattering is then

<div class="cg-equation" markdown="1">

\[
n_{\rm eff} = n_H \times C
\]

</div>

Star-forming gas is assigned the maximum clumping factor (30) unconditionally, regardless of its exact resolved density — the presence of active star formation is treated as a stronger indicator of unresolved dense sub-structure than the density field itself.

### Assumptions

- Subgrid clumping is understood as a numerical correction for finite resolution, not a physical process in its own right — CosmicGrain anticipates that the need for this correction diminishes and eventually disappears at sufficiently high resolution (e.g. ≳4096³), once molecular cloud densities are directly resolved rather than inferred.
- The four clumping values (1.5, 3, 10, 30) are fixed constants and do not vary with simulation resolution.
- The *thresholds* that select between these values scale with `CritPhysDensity` rather than being fixed physical densities, so the same relative behavior applies whether `CritPhysDensity` is tuned differently across runs.
- Active star formation is treated as an independent, overriding signal of clumping, evaluated before (and regardless of) the density-based ladder.
- The clumping factor only rescales the *effective* density used in downstream rate calculations — it does not modify the gas particle's actual resolved density (`SphP.Density`) or any other physical quantity.
- The same clumping factor is applied identically across growth, coagulation, and shattering; there is no process-specific tuning of the clumping model.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

`dust_clumping_factor()` is a small, stateless lookup function called identically from three separate physics routines, each of which multiplies its own resolved `n_H` by the returned factor to obtain `n_eff` before applying its own density gate or timescale calculation:

- **Grain growth** (`dust_grain_growth_subgrid()`) — `n_eff` gates whether growth is attempted at all (density floor) and determines which molecular-fraction bin \(f_{\rm mol}\) applies.
- **Coagulation** (`dust_grain_coagulation()`) — `n_eff` gates whether coagulation is active and sets the coagulation timescale.
- **Shattering** (`dust_grain_shattering()`) — `n_eff` gates whether the gas is diffuse enough for shattering to dominate over coagulation.

### A cross-cutting dependency worth knowing about

`dust_clumping_factor()` returns `1.0` unconditionally whenever `DustEnableClumping` is disabled. Because growth, coagulation, and shattering all compute their density gates from `n_eff = n_H × C` rather than from raw `n_H` directly, **disabling clumping doesn't just remove a standalone effect — it silently changes the effective density seen by all three of those other modules simultaneously**, since they fall back to the unboosted resolved density. This is worth keeping in mind both when interpreting resolution-ladder comparisons (e.g. S5 growth vs. S6 clumping vs. S8 coagulation) and when writing up the methods, since toggling this one flag has consequences for three physics rungs at once, not just its own.

### Algorithm

1. **Check whether clumping is enabled** — if not, return a no-op factor of 1.0.
2. **Check whether the gas particle is actively star-forming** — if so, return the maximum clumping factor (30) immediately.
3. **Otherwise, compare the resolved density against `CritPhysDensity`-scaled thresholds**, in descending order, returning the corresponding clumping factor.
4. **The caller multiplies this factor by the resolved density** to obtain `n_eff`, which is then used in that module's own gating and rate calculations.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableClumping` | Enables or disables subgrid clumping. When disabled, `n_eff` collapses to the raw resolved density for growth, coagulation, and shattering alike. |

The clumping factor values (1.5, 3, 10, 30) and the threshold fractions of `CritPhysDensity` (0.05, 0.2, 0.5) are hardcoded and not currently exposed as parameters. `CritPhysDensity` itself is a base GADGET star-formation parameter, not a dust-specific one — it sets the scale against which these thresholds are expressed. `DustCollisionDensityThresh`, used downstream by coagulation and shattering to gate on `n_eff`, is a separate parameter not controlled by this module directly.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- **Subgrid clumping routine**
  - `dust_clumping_factor()`
    - Returns the clumping factor for a given resolved density and star-forming state; the sole source of `n_eff` used across growth, coagulation, and shattering.

- **Consumers**
  - `dust_grain_growth_subgrid()` — uses `n_eff` for the density gate and to select the molecular-fraction bin.
  - `dust_grain_coagulation()` — uses `n_eff` for the density gate and the coagulation timescale.
  - `dust_grain_shattering()` — uses `n_eff` for the density gate that determines whether shattering (vs. coagulation) is the active regime.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to subgrid density clumping for unresolved ISM structure.*

### Trayford et al. (2026)

**Trayford, J. W., Schaye, J., Correa, C., Ploeckinger, S., Richings, A. J., Chaikin, E., Schaller, M., Benítez-Llambay, A., Frenk, C., & Huško, F. (2026). *Modelling the Evolution and Influence of Dust in Cosmological Simulations That Include the Cold Phase of the Interstellar Medium.* Monthly Notices of the Royal Astronomical Society, 545, staf2040.**

<p class="reference-note">
Introduces a subgrid clumping factor boosting unresolved molecular-cloud density for dust accretion, with fiducial and comparison runs (NoC, ConstC30, MaxC10) spanning a similar maximum clumping range to CosmicGrain's.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f2040T" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://arxiv.org/abs/2505.13056" target="_blank" rel="noopener">📄 arXiv</a>
<a href="https://doi.org/10.1093/mnras/staf2040" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Li, Narayanan, Torrey, Davé & Vogelsberger (2021)

**Li, Q., Narayanan, D., Torrey, P., Davé, R., & Vogelsberger, M. (2021). *The Origin of the Dust Extinction Curve in Milky Way-like Galaxies.* Monthly Notices of the Royal Astronomical Society, 507, 548–559.**

<p class="reference-note">
Cosmological dust evolution model incorporating a subgrid clumping correction for grain growth and coagulation, motivating the general approach CosmicGrain adopts for unresolved dense-gas dust activity.
</p>

<div class="reference-links">

<a href="https://academic.oup.com/mnras/article/507/1/548/6321031" target="_blank" rel="noopener">📚 Article</a>
<a href="https://doi.org/10.1093/mnras/stab2196" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Mao et al. (2019)

**Mao, Y., Koda, J., Shapiro, P. R., Iliev, I. T., Mellema, G., Park, H., Ahn, K., & Bianco, M. (2019). *The Impact of Inhomogeneous Subgrid Clumping on Cosmic Reionization.* Monthly Notices of the Royal Astronomical Society, 491, 1600–1621.**

<p class="reference-note">
Drawn from a different physical context (subgrid recombination clumping during reionization, not ISM dust evolution), cited here as broader precedent for representative clumping-factor values used to correct for unresolved density structure in cosmological simulations more generally.
</p>

<div class="reference-links">

<a href="https://academic.oup.com/mnras/article/491/2/1600/5645344" target="_blank" rel="noopener">📚 Article</a>
<a href="https://doi.org/10.1093/mnras/stz2986" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
