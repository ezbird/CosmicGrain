# Grain-Size Distributions

The grain-size distribution (GSD) is one of CosmicGrain's primary outputs because grain radius is carried by the dust particles and evolves with processing.

Use logarithmic radius bins and state the plotted quantity explicitly. Common choices include

- \(dN/d\log a\) — number-weighted distribution;
- \(dM_\mathrm{dust}/d\log a\) — dust-mass-weighted distribution; and
- a distribution normalized by gas or hydrogen mass for comparison with external dust models.

Number and mass weighting answer different physical questions and should not be mixed without labeling.

## Comparing epochs or halos

When comparing redshift, halo mass, or environment, keep the aperture, radius range, binning, and normalization fixed. If the goal is a cosmically or suite-averaged GSD, document the weighting used to combine halos or volumes.

## Useful subdivisions

The same GSD can be conditioned on

- stellar birth source (SNII, AGB, LRN),
- current carbon/silicate composition,
- galactocentric radius,
- gas environment, or
- redshift/time.

A convenient small-grain diagnostic used in current analysis is the fraction below a stated threshold (for example 50 nm), but the threshold must always be reported rather than treated as a built-in CosmicGrain definition.
