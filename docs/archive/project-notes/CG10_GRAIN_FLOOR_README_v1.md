# CG10 grain-size comparison before the late dust decline

Extract from gadget4 root, then run:

```bash
tar -xzf cg10_grain_floor_compare_20261013_v1.tar.gz

python3 scripts/cosmicgrain_cg10_grain_floor_compare_v1.py \
  halo441_output_1024_mingashsml0.0 \
  halo441_output_2048_mingashsml0.0 \
  --history-1024 analysis_products/cg10_tracking_fixed_ids/cg10_1024_target_history.csv \
  --history-2048 analysis_products/cg10_tracking_fixed_ids/cg10_2048_target_history.csv \
  --snapshots 21 22 \
  --outdir analysis_products/cg10_grain_floor_v1
```

The script belongs beside the corrected halo_utils.py. Dependencies: numpy,
h5py, matplotlib. It uses cosmicgrain.mplstyle when available. It opens only
dust datasets, reads bounded chunks, uses audited history centres, and caches
completed dust selections. Repeating the command reuses unchanged caches.
There is no new halo tracking, gas read, dark matter read, or optics calculation.

Send grain_floor_summary.csv, grain_field_metadata.json and
grain_floor_comparison.png. The PDF is included for export.

The central comparison uses the same r<30 pkpc sphere in both runs. A second
row compares the 30 pkpc to R200 region. Dust mass and count inside 30 pkpc must
match the audited histories before the script proceeds. Each run's actual
redshift is printed: snapshot 021 has slightly different output times.

## Measurements

- Grain-size histograms weighted by represented dust mass, normalized by
  region dust mass and log10 bin width (mass fraction per dex).
- Mass-weighted radius percentiles (10, 50, 90).
- Exact mass and count fractions at/below 1.05, 1.1, 2, 3, 5, 10 and 50 nm.
- Invalid-radius mass/count, fraction below 1 nm, and fraction outside plotted
  histogram limits. No smoothing or interpolation is applied to distributions.

Canonical snapshot field: PartType6/GrainRadius, interpreted in nm as in the
provided dust source and halo_utils field inventory. DustGrainRadius is also
recognized. Other schemas fail with a dataset listing rather than guessing.
Use --radius-field NAME for an explicit field, and --radius-unit micron only
if that dataset is written in microns. Dataset names and attributes are recorded
in grain_field_metadata.json so the interpretation can be reviewed. Snapshot
mass/coordinate conversion follows the same 1e10 Msun and ckpc/h convention
as the completed audited histories.

The 1 nm marker and 1–1.1 nm shaded band visualize the current recorded minimum
grain size. Histograms omit invalid sizes, zero-mass bins and values outside
0.5–300 nm; summary statistics retain all positive finite sizes. Size data
below the plotting range may indicate a unit/schema issue and must be reviewed.

## What this can and cannot establish

A much larger near-floor mass fraction in 2048 before the loss would support
an unusually small-grain reservoir as part of the mechanism. Similar
distributions would make differences in gas exposure, association and timing
more important next hypotheses. Neither outcome alone identifies shattering,
growth, coagulation or injection as the cause: all affect the size distribution.
Two outputs cannot establish the path of every particle between snapshots.
The 1024 and 2048 runs have different recorded physics parameters; this is
not a controlled resolution convergence test.

Confirmed event analysis already shows thermal removal accounting for ~95–98%
of missing original central mass across 022–024. Near-floor radii immediately
before deletion are not by themselves proof of a spurious cutoff: genuinely
eroding grains also approach that boundary. This comparison tests the reservoir
before the event, rather than only the selected particles at death. We have
not measured how much additional mass the cutoff returns beyond the physical
erosion update or whether the model is converged with respect to minimum size.

Validation: synthetic HDF5 tests passed dust-only periodic aperture selection,
nm/micron conversion, mass-weighted statistics, cache reuse, aperture mass
mismatch rejection and figure rendering. The real snapshots remain on cygnus
and have not been read here.
