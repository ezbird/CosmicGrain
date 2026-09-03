# Figures, Extinction, and SKIRT

The figure pipeline should make the simulation selection reproducible: every
caption or metadata record should include halo, snapshot/redshift, resolution,
centering method, aperture, projection depth, weighting, and enabled physics.

## Extinction and \(R_V\)

The extinction analysis uses each dust particle's radius and carbon fraction
with the adopted graphite/carbonaceous and astrosilicate optical tables.
Spatial maps and radial profiles should be compared with gas and recent-star
formation maps to distinguish processing, transport, and geometry.

## SKIRT exports

Two complementary exports are maintained:

- `export_skirt_inputs_MRN.py` provides a fixed-distribution control;
- `export_skirt_inputs_sizebins.py` preserves the simulated grain-size and
  composition information.

Generate SKIRT configurations with `generate_ski.py` and inspect outputs with
`plot_skirt_sed.py` and `plot_images.py`.

For every SKIRT run, archive:

- source snapshot and halo center;
- stellar and dust apertures;
- camera orientation and distance;
- wavelength grid;
- grain-size/composition binning;
- optical tables and stochastic-heating settings;
- random seed and photon-package count; and
- whether the run used simulated grains or the MRN control.

Changes in SED shape can arise from grain physics, total dust abundance, or
star–dust geometry. Compare controlled exports before attributing a difference
to composition alone.
