# Analysis

CosmicGrain's explicit dust particles make it possible to connect galaxy-scale dust budgets to the underlying grain population. Analysis should preserve the distinction between **where a grain is**, **what it is now**, and **where it came from**.

## Core analysis rules

1. **Halo versus galaxy apertures** — `R200c` is not interchangeable with a fixed physical ISM aperture.
2. **Gas-phase versus total metals** — D/Z depends on which metal reservoir is used.
3. **Number versus mass weighting** — grain-size and provenance distributions can look very different under the two weightings.
4. **Birth source versus present composition** — the stellar source tag records provenance; later processing can change size and composition.
5. **Numerical validation versus scientific inference** — a conserved whole-box budget does not imply a converged galaxy.

The shared analysis utilities should centralize periodic coordinates, halo centers, spherical-overdensity quantities, unit conversion, and target matching rather than duplicating these operations across plotting scripts.

## Recommended workflow

1. audit the complete snapshot;
2. identify and center the target halo;
3. measure or load `M200c` and `R200c`;
4. define and record the galaxy/ISM aperture;
5. compute baryonic, metal, and dust budgets;
6. examine grain size, composition, and stellar provenance;
7. construct spatial and radial diagnostics;
8. derive extinction diagnostics where appropriate; and
9. export controlled inputs for radiative-transfer calculations when needed.

## Analysis topics

- [Reading snapshots](snapshots.md)
- [Dust particle fields](dust-particle-fields.md)
- [Grain-size distributions](grain-size-distributions.md)
- [Composition & provenance](composition-provenance.md)
- [Dust-to-gas and dust-to-metal ratios](dust-ratios.md)
- [Spatial maps](spatial-maps.md)
- [Optical properties, extinction & R(V)](extinction-rv.md)
- [Synthetic observations & SKIRT](synthetic-observations.md)
- [Halo catalogs](halo-catalogs.md)
- [Radial profiles](radial-profiles.md)
