# Grain Growth

## Overview

<div class="cg-card summary" markdown="1">

Grain growth adds dust mass by accreting gas-phase metals onto existing
grains, following Hirashita & Kuo (2011, HK11). It is the main way dust mass
grows beyond stellar injection (coagulation only changes grain size). Growth
is fastest in dense, cold, metal-rich gas and is capped by a redshift-dependent
maximum dust-to-metal ratio.

</div>

## Physical Model

<div class="cg-card model" markdown="1">

CosmicGrain uses the HK11 accretion timescale,

<div class="cg-equation" markdown="1">

\[
\tau_{\rm acc} = \tau_0 \left(\frac{a}{0.1\,\mu{\rm m}}\right)\left(\frac{Z_\odot}{Z}\right)\left(\frac{1000\,{\rm cm^{-3}}}{n_{\rm H}}\right)\left(\frac{50\,{\rm K}}{T}\right)^{1/2}\left(\frac{0.3}{S}\right)\mathcal{C}_{\rm gr}
\]

</div>

where \(\tau_0=6.30\times10^7\) yr for silicate and \(5.59\times10^7\) yr for
carbonaceous material, \(Z_\odot=0.02\), \(T=20\) K, \(S=0.3\), and
\(\mathcal{C}_{\rm gr}\) is `DustGrowthCalibration`, a timescale multiplier
(larger = slower; v1.2.0 default 1.12). \(\tau_{\rm acc}\) is clamped to
\([10^6, 5\times10^9]\) yr.

The density \(n_{\rm H}\) and the molecular fraction \(f_{\rm mol}\) depend
on `DustEnableClumping` ([Subgrid Clumping](subgrid-clumping.md)):

- **Modes 2/3 (default 2):** \(n_{\rm H}=n_{\rm dense}\) (100 cm\(^{-3}\) in
  mode 2) and \(f_{\rm mol}=f_{\rm dense}\) from Blitz & Rosolowsky (2006).
- **Modes 0/1:** \(n_{\rm H}=n_{\rm eff}=C\,n_{\rm H}\); growth needs
  \(n_{\rm eff}\ge0.1\,{\rm cm^{-3}}\), and \(f_{\rm mol}\) is 0.8 in
  star-forming gas, 0.5 for \(n_{\rm eff}>100\), 0.2 for \(n_{\rm eff}>10\),
  else 0.05, times 1.5 (capped at 1) for \(Z>0.01\).

Carbonaceous and silicate components grow separately:

\[
\Delta M_i = M_i
\left[\exp\!\left(\frac{3 f_{\rm mol}\Delta t}{\tau_i}\right)-1\right],
\]

so growth evolves `CarbonMassFraction`. For a pure component this is
\(a_{\rm new}=a\exp(f_{\rm mol}\Delta t/\tau_{\rm acc})\). The new radius is
computed from the final total mass, \(a_{\rm new}=a\,(M_{\rm new}/M)^{1/3}\).

Growth is bounded by

<div class="cg-equation" markdown="1">

\[
\left(\frac{M_{\rm dust}}{M_{\rm metals}}\right)_{\rm max} = \max\!\left(0.05,\ \frac{0.5}{1+0.15z}\right),
\]

</div>

which is 0.5 at \(z=0\), 0.29 at \(z=5\), and 0.20 at \(z=10\) (the 0.05 floor
binds only above \(z=60\)). \(M_{\rm dust}\) is the grain's mass and
\(M_{\rm metals}\) the associated gas particle's metal mass.

### Assumptions

- Each grain grows from its associated nearest gas particle only.
- The grain surface temperature in \(\tau_{\rm acc}\) is a fixed 20 K, not
  the tracked `DustTemperature`.
- The sticking coefficient is a fixed 0.3.
- Accreted C and silicate-forming elements (O, Mg, Si, Fe in fixed ratios)
  are removed from the gas particle, conserving metal mass.

</div>

## Implementation

<div class="cg-card implementation" markdown="1">

### Gates, in order

1. Gas temperature above `DustThermalSputteringTemp`: no growth.
2. Gas metallicity below \(10^{-4}\): no growth.
3. Modes 0/1 only: \(n_{\rm eff}<0.1\,{\rm cm^{-3}}\): no growth.
4. Grain already at the D/Z cap: no growth.
5. \(f_{\rm mol}<0.01\): no growth (reachable only in modes 2/3).
6. Grain farther than the association radius
   \(\max(2,\min(H_{\rm sml},10))\) code length units from its gas: no growth.
7. Corrupted grains (zero, NaN or negative radius or mass) are removed.

### Caps on the accreted mass

The requested \(\Delta M=\Delta M_{\rm C}+\Delta M_{\rm sil}\) is reduced, in
order, to the D/Z cap, to 99% of the gas particle's metal mass, and to 20% of
the grain mass per call. The capped mass keeps the requested C/silicate split.
The element-resolved gas removal can reduce either component further if an
element runs out; the grain's mass, radius, and carbon fraction are then set
from the mass actually removed. Growth that would leave
[`DustMinGrainSize`, `DustMaxGrainSize`] is skipped.

</div>

## Parameters

| Parameter | Description |
|-----------|-------------|
| `DustEnableGrowth` | Enables or disables grain growth. |
| `DustGrowthCalibration` | Timescale multiplier, larger = slower; default 1.12. |
| `DustEnableClumping` | Selects the dense-phase (2/3) or legacy (0/1) density and \(f_{\rm mol}\). |
| `DustThermalSputteringTemp` | Gas temperature above which growth is off (default \(5\times10^5\) K). |
| `DustMinGrainSize`, `DustMaxGrainSize` | Shared radius limits (nm). |

The HK11 prefactors, \(T=20\) K, \(S=0.3\), \(Z_\odot=0.02\), the legacy
\(f_{\rm mol}\) tiers, the D/Z cap, and the timescale clamp are hardcoded.

## Where in the Code?

<div class="cg-card implementation" markdown="1">

### `src/dust/dust.cc`

- `dust_grain_growth_subgrid()` applies the gates and caps and updates mass, radius, and composition.
- `tau_acc_yr_HK11()` evaluates the HK11 timescale per species.
- `remove_dust_material_from_gas()` takes the accreted elements from the gas particle.
- `dust_dense_fraction()` / `dust_clumping_factor()` supply the density model.
- Called from `update_dust_dynamics()` before coagulation and shattering.
- Bookkeeping: `NGrainGrowthEvents`, `TotalMassGrown`.

</div>

## Primary References

<div class="cg-card references" markdown="1">

*Foundational papers relevant to interstellar grain growth via accretion.*

### Hirashita & Kuo (2011)

**Hirashita, H., & Kuo, T.-M. (2011). *Effects of Grain Size Distribution on the Interstellar Dust Mass Growth.* Monthly Notices of the Royal Astronomical Society, 416, 1340–1353.**

<p class="reference-note">
Source of the accretion timescale formula CosmicGrain implements directly (HK11).
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2011MNRAS.416.1340H" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1111/j.1365-2966.2011.19131.x" target="_blank" rel="noopener">🔗 DOI</a>

</div>

---

### Asano, Takeuchi, Hirashita & Nozawa (2013)

**Asano, R. S., Takeuchi, T. T., Hirashita, H., & Nozawa, T. (2013). *What Determines the Grain Size Distribution in Galaxies?* Monthly Notices of the Royal Astronomical Society, 432, 637–652.**

<p class="reference-note">
Extends the HK11 accretion framework alongside coagulation, shattering, and destruction to model the evolving grain size distribution — the broader framework CosmicGrain's combined growth and coagulation modules draw on.
</p>

<div class="reference-links">

<a href="https://ui.adsabs.harvard.edu/abs/2013MNRAS.432..637A" target="_blank" rel="noopener">📚 ADS</a>
<a href="https://doi.org/10.1093/mnras/stt506" target="_blank" rel="noopener">🔗 DOI</a>

</div>

</div>
