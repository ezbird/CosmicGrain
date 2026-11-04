# Optical Properties, Extinction, and R(V)

CosmicGrain can connect the evolved grain population to extinction diagnostics because each dust particle carries grain size and composition information.

For a grain of radius \(a\), the adopted optical tables provide wavelength-dependent extinction efficiencies. The extinction cross section is

\[
C_\mathrm{ext}(a,\lambda)=\pi a^2 Q_\mathrm{ext}(a,\lambda).
\]

Particle contributions can then be accumulated along a line of sight to estimate extinction in chosen bands. For Johnson B and V,

\[
R(V) = \frac{A_V}{A_B-A_V}.
\]

## Current optical tables

Current CosmicGrain analysis uses separate carbonaceous/graphitic and astrosilicate optical-property tables and interpolates them over grain radius and wavelength. The exact table files and interpolation choices should be archived with a result; they are analysis assumptions, not immutable properties of the simulation.

## Interpretation

A spatial variation in \(R(V)\) can reflect changes in the evolved grain-size/composition distribution, while attenuation measured from a full radiative-transfer calculation can additionally depend on star-dust geometry and scattering. Keep **extinction from the grain population** conceptually separate from **attenuation through a galaxy**.

## Reproducible maps

Record pixel size, projection depth, orientation, minimum dust signal/count cuts, optical tables, and any smoothing. When comparing with resolved Milky Way extinction maps, match the physical spatial scale before comparing distributions or power spectra.
