# Subgrid Clumping

## Overview

<div class="cg-card summary" markdown="1">

Subgrid clumping is not a dust process in its own right. It decides how much
of a gas element is unresolved dense (cold, molecular) gas, and at what
density, for grain growth, coagulation, and shattering. `DustEnableClumping`
selects the model:

| Mode | Model |
|------|-------|
| 0 | Off: resolved density, legacy density switch for coagulation/shattering. |
| 1 | Legacy clumping-factor tiers and the `DustCollisionDensityThresh` switch. |
| **2** | **Blitz & Rosolowsky (2006) dense phase at \(n_{\rm H}=100\,{\rm cm^{-3}}\) (v1.2.0 default).** |
| 3 | As 2, with the dense-phase density from pressure balance (experimental). |

</div>

## Physical Model

<div class="cg-card model" markdown="1">

### Dense-phase model (modes 2 and 3)

Each gas element has a dense-phase mass fraction from the Blitz & Rosolowsky
(2006) pressure law,

<div class="cg-equation" markdown="1">

\[
f_{\rm dense} = \frac{R}{1+R},\qquad
R = \left(\frac{P/k}{4.3\times10^{4}\,{\rm K\,cm^{-3}}}\right)^{0.92},
\]

</div>

where \(P\) is the particle pressure from its entropy (for star-forming gas,
the Springel & Hernquist effective-EOS pressure). Non-star-forming gas hotter
than \(3\times10^4\) K gets \(f_{\rm dense}=0\).

The dense phase has hydrogen density
\(n_{\rm dense}=\max(n_{\rm H},100\,{\rm cm^{-3}})\) in mode 2 and
\(n_{\rm dense}=\max[n_{\rm H},(P/k)/100\,{\rm K}]\) in mode 3.

!!! note "Why mode 2 is the default"

    Mode 3 is uncalibrated and has not been used in a production run. Two
    reasons favour the fixed cloud density of mode 2. First, for star-forming
    gas \(P\) is the Springel & Hernquist effective pressure, which rises with
    the resolved density; mode 3 therefore makes \(n_{\rm dense}\), and with it
    coagulation and growth, faster at higher resolution, the dependence the
    dense-phase model was built to remove. Second, that effective pressure
    stands for unresolved turbulence and SN heating, not the thermal pressure
    confining cold clouds. In typical star-forming gas mode 3 gives
    \(n_{\rm dense}\approx400\)–\(2000\ {\rm cm^{-3}}\), 4–20× mode 2, so it
    would need its own calibration. It is kept as a robustness test of the
    single-cloud-density assumption.

- **Coagulation** acts on the dense fraction: its rate is multiplied by
  \(f_{\rm dense}\) and uses \(n_{\rm dense}\).
- **Grain growth** uses \(f_{\rm dense}\) as its molecular fraction and
  \(n_{\rm dense}\) in the HK11 timescale.
- **Shattering** acts on the diffuse fraction: its rate is multiplied by
  \(1-f_{\rm dense}\) at the resolved \(n_{\rm H}\).

No process switches on or off at a density threshold, so the rates do not
grow with the resolved density; `DustCollisionDensityThresh` is not used.
Coagulation and shattering are skipped only where their phase fraction is
below \(10^{-4}\).

### Legacy clumping factor (mode 1)

A clumping factor \(C\) is assigned from the star-forming state and the
resolved density relative to `CritPhysDensity` (\(n_{\rm sf}\)):

<div class="cg-equation" markdown="1">

\[
C =
\begin{cases}
30 & \text{star-forming gas} \\
10 & n_{\rm H} > 0.5\,n_{\rm sf} \\
3 & n_{\rm H} > 0.2\,n_{\rm sf} \\
1.5 & n_{\rm H} > 0.05\,n_{\rm sf} \\
1 & \text{otherwise}
\end{cases}
\qquad n_{\rm eff}=C\,n_{\rm H}.
\]

</div>

Coagulation runs where \(n_{\rm eff}\ge\) `DustCollisionDensityThresh` and
shattering where \(n_{\rm eff}\le\) it; growth uses \(n_{\rm eff}\) for its
density gate, molecular-fraction tiers, and timescale. Mode 0 sets \(C=1\)
with the same switch.

### Assumptions

- Clumping corrects for unresolved structure; it never changes the gas
  particle's density or any other stored quantity.
- The Blitz & Rosolowsky law is applied per gas element, and the dense phase
  is assumed to be at a single density.
- The constants (\(P_0/k\), exponent 0.92, 100 cm\(^{-3}\), 100 K,
  \(3\times10^4\) K; legacy factors and `CritPhysDensity` fractions) are
  hardcoded.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

Changing `DustEnableClumping` changes growth, coagulation, and shattering
together. Keep this in mind when comparing physics-ladder rungs.

Because coagulation and growth calibrations were tuned for one mode, changing
the mode requires recalibrating `DustCoagulationCalibration` and
`DustGrowthCalibration`. Moving from mode 1 to mode 2, both were multiplied
by 2.5 (8.91 to 22.3 and 0.446 to 1.12) to keep the disk-integrated rates of
the legacy model.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableClumping` | 0 off, 1 legacy tiers, 2 Blitz & Rosolowsky dense phase (default), 3 pressure-balance dense phase. |
| `DustCollisionDensityThresh` | Coagulation/shattering switch on \(n_{\rm eff}\) in modes 0 and 1 only. |
| `CritPhysDensity` | Star-formation threshold; sets the legacy tier boundaries. |

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- `dust_dense_fraction()` returns \(f_{\rm dense}\) and \(n_{\rm dense}\) (modes 2 and 3).
- `dust_clumping_factor()` returns the legacy factor \(C\) (modes 0 and 1).
- Consumers: `dust_grain_growth_subgrid()`, `dust_grain_coagulation()`, `dust_grain_shattering()`.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to subgrid density clumping for unresolved ISM structure.*

### Blitz & Rosolowsky (2006)

**Blitz, L., & Rosolowsky, E. (2006). *The Role of Pressure in GMC Formation II: The H2-Pressure Relation.* The Astrophysical Journal, 650, 933–944.**

<p class="reference-note">
Source of the pressure law \(R_{\rm mol}=(P/P_0)^{0.92}\), \(P_0/k=4.3\times10^4\) K cm\(^{-3}\), used for the dense-phase fraction in modes 2 and 3.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2006ApJ...650..933B" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1086/505417" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Trayford et al. (2026)

**Trayford, J. W., Schaye, J., Correa, C., Ploeckinger, S., Richings, A. J., Chaikin, E., Schaller, M., Benítez-Llambay, A., Frenk, C., & Huško, F. (2026). *Modelling the Evolution and Influence of Dust in Cosmological Simulations That Include the Cold Phase of the Interstellar Medium.* Monthly Notices of the Royal Astronomical Society, 545, staf2040.**

<p class="reference-note">
Introduces a subgrid clumping factor for dust accretion (runs NoC, ConstC30, MaxC10) spanning a similar range to CosmicGrain's legacy mode 1.
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
