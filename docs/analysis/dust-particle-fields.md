# Dust Particle Fields

CosmicGrain stores dust as explicit `PartType6` superparticles. Before building a scientific diagnostic, inspect the fields actually present in the snapshot rather than assuming that every historical run used the same output schema.

At minimum, dust analysis commonly needs particle coordinates and velocities, mass, grain radius, composition information, and provenance/source information. Additional fields may encode birth state, local environment, or other diagnostics depending on the build and output configuration.

## Units and metadata

Snapshot metadata should be read from the appropriate HDF5 groups. In the current CosmicGrain/GADGET-4 output convention, **`HubbleParam` is stored in `Parameters`, not `Header`**. Record the scale factor/redshift and unit convention before converting positions, masses, or velocities.

`GrainRadius` is written in nm in the current analysis convention. Confirm field units when analyzing older or externally produced snapshots.

## Surviving particles

Some analyses need to exclude destroyed or effectively zero-mass numerical remnants. Apply one documented criterion consistently across all compared samples. Never allow a tiny residual-mass population to dominate a number-weighted histogram by accident.

## Reproducibility

For derived catalogs, archive the snapshot, halo/group identifier, centering method, aperture, weighting, field cuts, and code version or analysis-script revision.
