# Analysis

CosmicGrain analysis must preserve three distinctions:

1. **halo versus galaxy apertures** — `R200c` is not interchangeable with a
   fixed physical ISM aperture;
2. **gas-phase versus total metals** — D/Z depends on which metal reservoir is
   used; and
3. **numerical validation versus scientific inference** — a conserved
   whole-box budget does not imply a converged galaxy.

The shared analysis module is `halo_utils.py`.
They centralize periodic coordinates, halo centers, spherical-overdensity
quantities, unit conversion, and target matching. New scripts should use
these utilities rather than reimplementing centering or units.

Recommended analysis order:

1. audit the complete snapshot;
2. identify and center the target halo;
3. measure `M200c` and `R200c`;
4. define and record the galaxy/ISM aperture;
5. compute baryonic and dust budgets;
6. examine grain size, composition, and source tags;
7. construct spatial/radial diagnostics; and
8. export selected snapshots to SKIRT.
