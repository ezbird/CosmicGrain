# Equation Map

This page is a compact map from the principal dust processes to the equations implemented in the current source. The individual Physics pages provide the assumptions and numerical gates around each expression.

## Dense phase (`DustEnableClumping` = 2, default)

Coagulation, accretion and shattering use the Blitz & Rosolowsky (2006)
dense-phase mass fraction of each gas particle,

\[
R = \left(\frac{P/k}{4.3\times10^{4}\,{\rm K\,cm^{-3}}}\right)^{0.92},
\qquad f_{\rm dense} = \frac{R}{1+R},
\]

with \(P\) the particle pressure (the SH03 effective pressure in star-forming
gas) and \(f_{\rm dense}=0\) in non-star-forming gas above \(3\times10^4\) K.
Coagulation and accretion act on the dense phase at
\(n_{\rm dense}=\max(n_{\rm H},100\,{\rm cm^{-3}})\) with rates
\(\times f_{\rm dense}\); shattering acts on the diffuse phase at the resolved
\(n_{\rm H}\) with rate \(\times(1-f_{\rm dense})\). There is no density switch
and `DustCollisionDensityThresh` is unused. Mode 3 takes
\(n_{\rm dense}=\max(n_{\rm H},(P/k)/100\,{\rm K})\). Mode 1 keeps the legacy
clumping factors and the `DustCollisionDensityThresh` switch.

## Coagulation

The implementation follows Trayford et al. (2026), Eq. 10, with a continuous
representative-grain radius and the local particle-sampled D/G estimate:

\[
\tau_{\rm co} = 27.1\,{\rm Myr}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{v_{\rm co}}{0.1\,{\rm km\,s^{-1}}}\right)^{-1}
\left(\frac{n_{\rm H,eff}}{10\,{\rm cm}^{-3}}\right)^{-1}
\mathcal{C}_{\rm co},
\]

divided by \(f_{\rm dense}\) in the dense-phase model, where
\(v_{\rm co}=0.2\,{\rm km\,s^{-1}}\), \(n_{\rm H,eff}\) is \(n_{\rm dense}\)
(legacy mode: the clumped density), and \(\mathcal{C}_{\rm co}\) is
`DustCoagulationCalibration`. D/G is the dust-to-gas mass ratio in the
association sphere and is clamped to \([10^{-6},0.1]\). The continuous-size
closure is \(da/dt=a/(3\tau_{\rm co})\); coagulation changes representative
radius while conserving superparticle mass and composition, with large
updates integrated through substeps limited to 20% radius growth each.

## Grain growth

Growth uses separate Hirashita & Kuo-style accretion timescales for carbonaceous and silicate material. For each material component, the uncapped mass-growth law follows the radius-growth prescription through

\[
\frac{M_{\rm new}}{M_{\rm old}} =
\exp\!\left(\frac{3 f_{\rm mol}\,\Delta t}{\tau_{\rm acc}}\right).
\]

Here \(\tau_{\rm acc}\) is evaluated at \(n_{\rm dense}\), multiplied by
`DustGrowthCalibration`, and \(f_{\rm mol}=f_{\rm dense}\) in the dense-phase
model (legacy mode: fixed tiers by density and star formation). The source then limits the requested growth by available gas-phase metals, a redshift-dependent maximum dust-to-metal ratio, and a per-step mass-growth cap.

## Thermal sputtering

The sputtering path uses a Tsai & Mathews / Tielens-style temperature dependence with a characteristic temperature of `2e6 K` and exponent `2.5`, together with composition-dependent corrections. Material removed from the dust particle is returned to the gas-phase elemental reservoirs.

## Drag

Gas–dust coupling uses an Epstein-like stopping-time calculation based on grain radius, grain material density, local gas density, sound speed, and relative velocity. The drag update is applied analytically over the dust timestep rather than as an unrestricted explicit kick.

## Radiation pressure

The acceleration is constructed from neighboring young-star luminosity and geometric dilution,

\[
a_{\rm rad} \propto
\frac{L\,Q_{\rm pr}}{4\pi r^2 c}\,
\frac{1}{a\rho_{\rm grain}},
\]

with an additional runtime efficiency factor and a drag/terminal-velocity limiter. See the Radiation Pressure page for the current stellar-age and `Q_pr` prescriptions.

## Shattering

The explicit turbulent-velocity gate uses

\[
v_{\rm gr} = 1.1\,{\rm km\,s^{-1}}\,
\mathcal{M}^{3/2}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)^{1/2}
\left(\frac{T}{10^4\,{\rm K}}\right)^{1/4}
\left(\frac{n_{\rm H}}{1\,{\rm cm}^{-3}}\right)^{-1/4}
\left(\frac{\rho_{\rm gr}}{3.5\,{\rm g\,cm}^{-3}}\right)^{1/2},
\]

with fixed \(\mathcal{M}=1\) and a composition-dependent threshold
\(v_{\rm shatter}=(1+\mathrm{CF})\,{\rm km\,s^{-1}}\). Eligible grains are
then assigned the Trayford et al. (2026), Eq. 9 timescale

\[
\tau_{\rm sh}=54.1\,{\rm Myr}
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
f(\hat n_{\rm H})\,\frac{\mathcal{C}_{\rm sh}}{1-f_{\rm dense}},
\quad
f(\hat n_{\rm H}) =
\begin{cases}
\hat n_{\rm H}^{-1}, & \hat n_{\rm H}<1,\\
\hat n_{\rm H}^{-1/3}, & \hat n_{\rm H}\ge1.
\end{cases}
\]

Here \(\hat n_{\rm H}=n_{\rm H}/(1\,{\rm cm}^{-3})\) is the unclumped density
(floored at \(10^{-4}\)), \(\mathcal{C}_{\rm sh}\) is
`DustShatteringCalibration`, \(\tau_{\rm sh}\) is clamped to [1 Myr, 10 Gyr],
and \(1-f_{\rm dense}\) is replaced by 1 in legacy mode, where shattering is
instead restricted to clumped \(n_{\rm eff}\le\) `DustCollisionDensityThresh`.
The shattering event is Poisson sampled with
\(P=1-\exp(-\Delta t/\tau_{\rm sh})\).

## Stellar injection radii

SNII and AGB representative radii are sampled from log-normal distributions:

\[
\log_{10}a \sim \mathcal{N}\!\left(\log_{10}a_0,\sigma_{\rm dex}^2\right),
\]

where \(a_0\) is the runtime median and \(\sigma_{\rm dex}\) is the runtime
standard deviation in dex. The samples are clipped to the configured minimum
and maximum grain radii. LRN injection remains fixed at 10 nm.

## Shock destruction

The SN-shock path estimates a characteristic shock radius from a Sedov-like relation and applies a velocity- and composition-dependent destruction efficiency to dust in the affected region. The triggering feedback event supplies the event energy.

## Authority

These expressions summarize the implementation but deliberately omit many gates, caps, composition updates, and conservation operations. For exact behavior, use the current source and the detailed Physics pages together.
