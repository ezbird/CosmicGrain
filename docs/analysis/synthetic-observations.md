# Synthetic Observations and SKIRT

Radiative-transfer calculations provide a bridge between CosmicGrain's evolved particle population and integrated or resolved observables.

Two useful experiment classes are:

- **evolved grains** — preserve the simulated grain-size/composition information; and
- **fixed-distribution controls** — hold the grain prescription fixed to isolate the effect of grain evolution.

This controlled comparison is often more informative than interpreting a single synthetic observation in isolation.

## Archive with every run

Record the source snapshot and halo, stellar and dust apertures, orientation/inclination, wavelength grid, grain-size/composition binning, optical tables, stochastic-heating settings, photon statistics, and random seed.

Changes in SED shape, colors, or attenuation can arise from grain physics, total dust abundance, and star-dust geometry. Design comparisons so that the intended variable is actually isolated.

## Reproducible figures and extinction

Record the halo, snapshot/redshift, resolution, centering method, aperture,
projection depth, weighting, and enabled physics with each figure. Extinction
calculations use grain radius and carbon fraction with the adopted graphite /
carbonaceous and astrosilicate optical tables. Compare spatial maps and radial
profiles with gas and recent-star-formation maps before attributing a trend to
grain processing or transport.

## SKIRT exports

The maintained export scripts provide a fixed-MRN control and an export that
preserves simulated size and composition bins. Generate SKIRT configurations
with `generate_ski.py` and inspect results with `plot_skirt_sed.py` and
`plot_images.py`. Archive
the halo center, stellar and dust apertures, camera orientation, wavelength
grid, optical tables, stochastic-heating settings, random seed, photon-package
count, and whether the export used simulated grains or the MRN control.
