# Equation Map

This page is a compact map from the principal dust processes to the equations implemented in the current source. The individual Physics pages provide the assumptions and numerical gates around each expression.

## Coagulation

The implementation follows Trayford et al. (2025), Eq. 10, with a continuous
representative-grain radius and the local particle-sampled D/G estimate:

\[
\tau_{\rm co} = 27.1\,{\rm Myr}
\left(\frac{a}{0.1\,\mu{\rm m}}\right)
\left(\frac{\mathrm{D/G}}{0.01}\right)^{-1}
\left(\frac{v_{\rm co}}{0.1\,{\rm km\,s^{-1}}}\right)^{-1}
\left(\frac{n_{\rm H,eff}}{10\,{\rm cm}^{-3}}\right)^{-1}
\mathcal{C}_{\rm co},
\]

where \(v_{\rm co}=0.2\,{\rm km\,s^{-1}}\), \(n_{\rm H,eff}\) includes the
clumping factor, and \(\mathcal{C}_{\rm co}\) is
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

The source then limits the requested growth by available gas-phase metals, a redshift-dependent maximum dust-to-metal ratio, and a per-step mass-growth cap.

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

## Shock destruction

The SN-shock path estimates a characteristic shock radius from a Sedov-like relation and applies a velocity- and composition-dependent destruction efficiency to dust in the affected region. The triggering feedback event supplies the event energy.

## Authority

These expressions summarize the implementation but deliberately omit many gates, caps, composition updates, and conservation operations. For exact behavior, use the current source and the detailed Physics pages together.
