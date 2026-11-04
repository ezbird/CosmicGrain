# Synthetic Observations and SKIRT

Radiative-transfer calculations provide a bridge between CosmicGrain's evolved particle population and integrated or resolved observables.

Two useful experiment classes are:

- **evolved grains** — preserve the simulated grain-size/composition information; and
- **fixed-distribution controls** — hold the grain prescription fixed to isolate the effect of grain evolution.

This controlled comparison is often more informative than interpreting a single synthetic observation in isolation.

## Archive with every run

Record the source snapshot and halo, stellar and dust apertures, orientation/inclination, wavelength grid, grain-size/composition binning, optical tables, stochastic-heating settings, photon statistics, and random seed.

Changes in SED shape, colors, or attenuation can arise from grain physics, total dust abundance, and star-dust geometry. Design comparisons so that the intended variable is actually isolated.
