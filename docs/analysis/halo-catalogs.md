# Halo Catalogs and Centers

FOF/SUBFIND catalogs identify candidate structures, but the analysis center
and spherical-overdensity quantities should be recomputed consistently for
each snapshot.

`halo_utils.py` provides the shared FOF association, periodic centering,
shrinking-sphere refinement, and `M200c`/`R200c` calculation. The 12-halo
suite additionally uses `zoom_halo_utils.py` and the target metadata produced
during candidate selection.

## Required catalog quantities

For every halo and redshift, retain:

- target identifier and matching method;
- center and bulk velocity;
- `M200c` and `R200c`;
- distance to any low-resolution contaminant;
- particle counts by type inside `R200c`; and
- the aperture used for galaxy-integrated quantities.

`Mstar`, `Mgas`, SFR, metallicity, `Mdust`, D/G, and D/Z must not be labeled
simply "halo properties" without an aperture. Use `R200c` for halo-wide
budgets and an explicitly declared physical or scaled radius for the galaxy.

## Tracking through time

Do not assume a fixed coordinate center. Match the target between catalogs
and refine the center at each output. Likewise, raw dust displacement from
`BirthPos` includes halo bulk motion; subtract the evolving halo trajectory
before interpreting it as transport or outflow.
