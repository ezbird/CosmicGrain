# Dust Distributions

Dust population analysis should retain mass, number, grain radius,
carbon fraction, stellar source tag, age, and environment. Number-weighted
and mass-weighted histograms answer different questions and should always be
labeled.

## Grain size

Use logarithmic radius bins and report whether the distribution is
\(dN/d\log a\), \(dM/d\log a\), or normalized per hydrogen mass. Comparisons
with MRN, Weingartner–Draine, or THEMIS models require matching the plotted
normalization and size range.

## Composition

CosmicGrain tracks an evolving carbon fraction. Carbon and silicate masses
should close to total dust mass. Creation channels begin near their adopted
source compositions, but sputtering, shocks, growth, coagulation, and
shattering can modify the later distribution.

## Stellar origin

The SNII, AGB, and LRN source tag records birth origin, not immutable chemical
identity. Analyze both:

- current dust mass grouped by birth source; and
- current size/composition conditioned on source.

This separation tests how much stellar-origin information survives later ISM
processing.

Destroyed or effectively zero-mass grains must be filtered consistently.
Record the threshold used rather than allowing empty numerical remnants to
dominate number histograms.
